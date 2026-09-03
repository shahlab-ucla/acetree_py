"""Validated, undoable application of image-analysis proposals to AceTree.

Tracking plugins never mutate a dataset directly.  A result is first shown as
a proposal and, if accepted, this command converts it to legacy nucleus
records in one atomic history operation.  Existing nuclei are retained by
object identity; only the reciprocal successor fields of accepted seed nuclei
may gain new children.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypeAlias

from ..core.nucleus import NILLI, Nucleus
from ..editing.commands import EditCommand, NucleiRecord
from .api import Calibration, Detection, TrackingResult

NucleusLocation: TypeAlias = tuple[int, int]
_NodeKey: TypeAlias = tuple[str, ...]


class TrackingProposalConflict(ValueError):
    """Raised when a proposal cannot be added without changing curated data."""


@dataclass(frozen=True)
class _PlannedNode:
    key: _NodeKey
    frame: int
    index: int
    x: int
    y: int
    z: float
    size: int
    weight: int = 0
    rweight: int = 0
    rsum: int = 0
    rcount: int = 0
    rwraw: int = 0
    rwcorr1: int = 0
    rwcorr2: int = 0
    rwcorr3: int = 0
    rwcorr4: int = 0
    existing: bool = False
    detection_id: str | None = None

    @property
    def location(self) -> NucleusLocation:
        return self.frame, self.index


@dataclass(frozen=True)
class _ApplicationPlan:
    nodes: dict[_NodeKey, _PlannedNode]
    append_order: tuple[_NodeKey, ...]
    arcs: tuple[tuple[_NodeKey, _NodeKey], ...]
    detection_mapping: dict[str, NucleusLocation]
    changed: bool


@dataclass
class ApplyTrackingProposal(EditCommand):
    """Append an accepted tracking proposal as one exact undoable edit.

    Args:
        result: Immutable detector/tracker proposal in physical microns.
        calibration: Dataset voxel calibration used to create pixel/plane
            :class:`~acetree_py.core.nucleus.Nucleus` records.

    The command rejects backward links, merges, more than two daughters,
    duplicate anchors, and links that would replace existing curated lineage
    relationships.  A ``gap`` edge is expanded into one linearly interpolated
    nucleus per missing frame because AceTree stores only adjacent-frame links.
    """

    result: TrackingResult
    calibration: Calibration

    _original_record_len: int = field(default=0, init=False)
    _original_frame_lengths: list[int] = field(default_factory=list, init=False)
    _existing_snapshots: list[tuple[Nucleus, Nucleus]] = field(
        default_factory=list, init=False
    )
    _detection_mapping: dict[str, NucleusLocation] = field(
        default_factory=dict, init=False
    )
    _applied: bool = field(default=False, init=False)
    _noop: bool = field(default=False, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        self._capture_snapshot(nuclei_record)
        plan = _build_application_plan(self.result, self.calibration, nuclei_record)
        self._detection_mapping = dict(plan.detection_mapping)
        self._noop = not plan.changed
        if self._noop:
            self._applied = True
            return

        self._mark_rollback_ready()
        try:
            _apply_plan(plan, nuclei_record)
        except BaseException:
            self._restore_snapshot(nuclei_record)
            self._detection_mapping = {}
            self._applied = False
            raise
        self._applied = True

    def undo(self, nuclei_record: NucleiRecord) -> None:
        if not self._existing_snapshots and not self._original_frame_lengths:
            return
        self._restore_snapshot(nuclei_record)
        self._detection_mapping = {}
        self._applied = False

    @property
    def description(self) -> str:
        return (
            "Apply tracking proposal "
            f"({len(self.result.detections)} detections, "
            f"{len(self.result.edges)} links)"
        )

    @property
    def is_noop(self) -> bool:
        return self._noop

    @property
    def detection_mapping(self) -> dict[str, NucleusLocation]:
        """Accepted detection ID to ``(1-based time, 1-based index)``."""
        if not self._applied:
            raise RuntimeError("The tracking proposal is not currently applied")
        return dict(self._detection_mapping)

    def _capture_snapshot(self, nuclei_record: NucleiRecord) -> None:
        self._original_record_len = len(nuclei_record)
        self._original_frame_lengths = [len(frame) for frame in nuclei_record]
        self._existing_snapshots = [
            (nucleus, nucleus.copy())
            for frame in nuclei_record
            for nucleus in frame
        ]
        self._detection_mapping = {}
        self._applied = False
        self._noop = False

    def _restore_snapshot(self, nuclei_record: NucleiRecord) -> None:
        # Restore fields in place so external GUI/tree references to curated
        # nuclei remain valid after undo.
        for current, saved in self._existing_snapshots:
            current.__dict__.clear()
            current.__dict__.update(saved.__dict__)

        for frame_index, old_length in enumerate(self._original_frame_lengths):
            if frame_index < len(nuclei_record):
                del nuclei_record[frame_index][old_length:]
        del nuclei_record[self._original_record_len:]


def proposal_to_nucleus_mapping(
    command: ApplyTrackingProposal,
) -> dict[str, NucleusLocation]:
    """Return the committed detection-to-nucleus mapping for *command*.

    Keeping this conversion explicit prevents a preview's temporary detection
    IDs from leaking into AceTree's index-based lineage model.
    """
    return command.detection_mapping


# Alias used by acceptance/export code that speaks in terms of accepted runs.
accepted_proposal_mapping = proposal_to_nucleus_mapping


def _build_application_plan(
    result: TrackingResult,
    calibration: Calibration,
    nuclei_record: NucleiRecord,
) -> _ApplicationPlan:
    detections = tuple(result.detections)
    by_id: dict[str, Detection] = {}
    for detection in detections:
        if detection.detection_id in by_id:
            raise TrackingProposalConflict(
                f"Duplicate detection ID: {detection.detection_id!r}"
            )
        by_id[detection.detection_id] = detection

    nodes: dict[_NodeKey, _PlannedNode] = {}
    detection_keys: dict[str, _NodeKey] = {}
    detection_mapping: dict[str, NucleusLocation] = {}
    used_anchor_locations: set[NucleusLocation] = set()

    for detection_id, anchor in result.existing_anchors.items():
        detection = by_id.get(detection_id)
        if detection is None:
            raise TrackingProposalConflict(
                f"Existing anchor references unknown detection {detection_id!r}"
            )
        time, index = int(anchor[0]), int(anchor[1])
        location = (time, index)
        if location in used_anchor_locations:
            raise TrackingProposalConflict(
                f"More than one detection anchors existing nucleus t={time} idx={index}"
            )
        used_anchor_locations.add(location)
        if detection.frame != time:
            raise TrackingProposalConflict(
                f"Anchor for {detection_id!r} is at t={time}, "
                f"but its detection is at t={detection.frame}"
            )
        nucleus = _existing_nucleus(nuclei_record, time, index)
        if not nucleus.is_alive:
            raise TrackingProposalConflict(
                f"Anchor t={time} idx={index} is not an alive nucleus"
            )
        key = ("detection", detection_id)
        node = _PlannedNode(
            key=key,
            frame=time,
            index=index,
            x=nucleus.x,
            y=nucleus.y,
            z=nucleus.z,
            size=nucleus.size,
            weight=nucleus.weight,
            rweight=nucleus.rweight,
            rsum=nucleus.rsum,
            rcount=nucleus.rcount,
            rwraw=nucleus.rwraw,
            rwcorr1=nucleus.rwcorr1,
            rwcorr2=nucleus.rwcorr2,
            rwcorr3=nucleus.rwcorr3,
            rwcorr4=nucleus.rwcorr4,
            existing=True,
            detection_id=detection_id,
        )
        nodes[key] = node
        detection_keys[detection_id] = key
        detection_mapping[detection_id] = location

    next_indices = {
        frame: len(nuclei_record[frame - 1])
        for frame in range(1, len(nuclei_record) + 1)
    }

    def allocate(frame: int) -> int:
        next_indices[frame] = next_indices.get(frame, 0) + 1
        return next_indices[frame]

    append_order: list[_NodeKey] = []
    for detection in detections:
        if detection.detection_id in detection_keys:
            continue
        key = ("detection", detection.detection_id)
        x, y, z, size = _detection_geometry(detection, calibration)
        measurements = _detection_measurements(detection)
        node = _PlannedNode(
            key=key,
            frame=detection.frame,
            index=allocate(detection.frame),
            x=x,
            y=y,
            z=z,
            size=size,
            **measurements,
            detection_id=detection.detection_id,
        )
        nodes[key] = node
        detection_keys[detection.detection_id] = key
        detection_mapping[detection.detection_id] = node.location
        append_order.append(key)

    incoming: dict[str, str] = {}
    outgoing: dict[str, set[str]] = {}
    outgoing_kinds: dict[str, list[str]] = {}
    edge_pairs: set[tuple[str, str]] = set()
    for edge in result.edges:
        source = by_id.get(edge.source_id)
        target = by_id.get(edge.target_id)
        if source is None or target is None:
            raise TrackingProposalConflict("Every link must reference a result detection")
        pair = (edge.source_id, edge.target_id)
        if pair in edge_pairs:
            raise TrackingProposalConflict(
                f"Duplicate tracking link {edge.source_id!r} -> {edge.target_id!r}"
            )
        edge_pairs.add(pair)
        if target.frame <= source.frame:
            raise TrackingProposalConflict(
                f"Tracking link {edge.source_id!r} -> {edge.target_id!r} "
                "must point strictly forward in time"
            )
        previous_source = incoming.get(edge.target_id)
        if previous_source is not None and previous_source != edge.source_id:
            raise TrackingProposalConflict(
                f"Detection {edge.target_id!r} has two parents (merges are forbidden)"
            )
        incoming[edge.target_id] = edge.source_id
        children = outgoing.setdefault(edge.source_id, set())
        children.add(edge.target_id)
        outgoing_kinds.setdefault(edge.source_id, []).append(edge.kind)
        if len(children) > 2:
            raise TrackingProposalConflict(
                f"Detection {edge.source_id!r} has more than two children"
            )
        if target.frame - source.frame > 1 and edge.kind != "gap":
            raise TrackingProposalConflict(
                f"Non-adjacent link {edge.source_id!r} -> {edge.target_id!r} "
                "must be marked as a gap"
            )

    for source_id, children in outgoing.items():
        kinds = outgoing_kinds[source_id]
        has_split = "split" in kinds
        if has_split and (len(children) != 2 or any(kind != "split" for kind in kinds)):
            raise TrackingProposalConflict(
                f"Division source {source_id!r} must have exactly two split edges"
            )
        if len(children) == 2 and not has_split:
            raise TrackingProposalConflict(
                f"Source {source_id!r} has two children without an explicit split event"
            )

    arcs: list[tuple[_NodeKey, _NodeKey]] = []
    for edge_number, edge in enumerate(result.edges):
        source_detection = by_id[edge.source_id]
        target_detection = by_id[edge.target_id]
        source_key = detection_keys[edge.source_id]
        target_key = detection_keys[edge.target_id]
        previous_key = source_key
        delta = target_detection.frame - source_detection.frame
        for step in range(1, delta):
            fraction = step / delta
            frame = source_detection.frame + step
            gap_key = ("gap", str(edge_number), str(step))
            gap_detection = Detection(
                detection_id=f"__gap__:{edge_number}:{step}",
                frame=frame,
                x_um=_lerp(source_detection.x_um, target_detection.x_um, fraction),
                y_um=_lerp(source_detection.y_um, target_detection.y_um, fraction),
                z_um=_lerp(source_detection.z_um, target_detection.z_um, fraction),
                radius_um=_lerp(
                    source_detection.radius_um,
                    target_detection.radius_um,
                    fraction,
                ),
                quality=_lerp(
                    source_detection.quality,
                    target_detection.quality,
                    fraction,
                ),
                # Legacy StarryNite export carries the parent's measurement
                # values through an interpolated false-negative point.
                features=dict(source_detection.features),
            )
            x, y, z, size = _detection_geometry(gap_detection, calibration)
            measurements = _detection_measurements(gap_detection)
            gap_node = _PlannedNode(
                key=gap_key,
                frame=frame,
                index=allocate(frame),
                x=x,
                y=y,
                z=z,
                size=size,
                **measurements,
            )
            nodes[gap_key] = gap_node
            append_order.append(gap_key)
            arcs.append((previous_key, gap_key))
            previous_key = gap_key
        arcs.append((previous_key, target_key))

    changed = bool(append_order)
    simulated_predecessors: dict[NucleusLocation, NucleusLocation | None] = {}
    simulated_successors: dict[NucleusLocation, set[NucleusLocation]] = {}
    for node in nodes.values():
        if node.existing:
            nucleus = _existing_nucleus(nuclei_record, node.frame, node.index)
            predecessor = (
                (node.frame - 1, nucleus.predecessor)
                if node.frame > 1 and nucleus.predecessor != NILLI
                else None
            )
            successors = {
                (node.frame + 1, successor)
                for successor in (nucleus.successor1, nucleus.successor2)
                if successor != NILLI
            }
        else:
            predecessor = None
            successors = set()
        simulated_predecessors[node.location] = predecessor
        simulated_successors[node.location] = successors

    for source_key, target_key in arcs:
        source_node = nodes[source_key]
        target_node = nodes[target_key]
        if target_node.frame != source_node.frame + 1:
            raise TrackingProposalConflict("AceTree links must join adjacent frames")
        source_location = source_node.location
        target_location = target_node.location
        old_predecessor = simulated_predecessors[target_location]
        if target_node.existing and old_predecessor != source_location:
            raise TrackingProposalConflict(
                f"Proposal would replace the curated predecessor of existing "
                f"nucleus t={target_node.frame} idx={target_node.index}"
            )
        if old_predecessor is not None and old_predecessor != source_location:
            raise TrackingProposalConflict(
                f"Nucleus t={target_node.frame} idx={target_node.index} "
                "would have more than one parent"
            )
        simulated_predecessors[target_location] = source_location

        successors = simulated_successors[source_location]
        if target_location not in successors:
            if len(successors) >= 2:
                raise TrackingProposalConflict(
                    f"Nucleus t={source_node.frame} idx={source_node.index} "
                    "already has two successors"
                )
            successors.add(target_location)
            changed = True

    return _ApplicationPlan(
        nodes=nodes,
        append_order=tuple(append_order),
        arcs=tuple(arcs),
        detection_mapping=detection_mapping,
        changed=changed,
    )


def _apply_plan(plan: _ApplicationPlan, nuclei_record: NucleiRecord) -> None:
    for key in plan.append_order:
        node = plan.nodes[key]
        while len(nuclei_record) < node.frame:
            nuclei_record.append([])
        frame = nuclei_record[node.frame - 1]
        if len(frame) + 1 != node.index:
            raise RuntimeError(
                f"Tracking proposal index drift at t={node.frame}: "
                f"expected {node.index}, got {len(frame) + 1}"
            )
        frame.append(
            Nucleus(
                index=node.index,
                x=node.x,
                y=node.y,
                z=node.z,
                size=node.size,
                identity="",
                assigned_id="",
                status=1,
                predecessor=NILLI,
                successor1=NILLI,
                successor2=NILLI,
                weight=node.weight,
                rweight=node.rweight,
                rsum=node.rsum,
                rcount=node.rcount,
                rwraw=node.rwraw,
                rwcorr1=node.rwcorr1,
                rwcorr2=node.rwcorr2,
                rwcorr3=node.rwcorr3,
                rwcorr4=node.rwcorr4,
            )
        )

    for source_key, target_key in plan.arcs:
        source_node = plan.nodes[source_key]
        target_node = plan.nodes[target_key]
        source = _existing_nucleus(
            nuclei_record, source_node.frame, source_node.index
        )
        target = _existing_nucleus(
            nuclei_record, target_node.frame, target_node.index
        )
        if target.predecessor == NILLI:
            target.predecessor = source_node.index
        elif target.predecessor != source_node.index:
            raise RuntimeError("Tracking proposal predecessor changed after validation")

        if target_node.index not in (source.successor1, source.successor2):
            if source.successor1 == NILLI:
                source.successor1 = target_node.index
            elif source.successor2 == NILLI:
                source.successor2 = target_node.index
            else:
                raise RuntimeError("Tracking proposal successor changed after validation")


def _existing_nucleus(
    nuclei_record: NucleiRecord,
    time: int,
    index: int,
) -> Nucleus:
    if time < 1 or time > len(nuclei_record):
        raise TrackingProposalConflict(f"Timepoint {time} is outside the nuclei record")
    frame = nuclei_record[time - 1]
    if index < 1 or index > len(frame):
        raise TrackingProposalConflict(
            f"Nucleus index {index} does not exist at t={time}"
        )
    return frame[index - 1]


def _detection_geometry(
    detection: Detection,
    calibration: Calibration,
) -> tuple[int, int, float, int]:
    if not all(
        math.isfinite(value)
        for value in (
            detection.x_um,
            detection.y_um,
            detection.z_um,
            detection.radius_um,
        )
    ):
        raise TrackingProposalConflict(
            f"Detection {detection.detection_id!r} has non-finite geometry"
        )
    if detection.radius_um <= 0:
        raise TrackingProposalConflict(
            f"Detection {detection.detection_id!r} has a non-positive radius"
        )
    x_px, y_px, z_plane = detection.to_pixel(calibration)
    size = max(
        1,
        _round_half_away_from_zero(2.0 * detection.radius_um / calibration.xy_um),
    )
    return (
        _round_half_away_from_zero(x_px),
        _round_half_away_from_zero(y_px),
        float(z_plane),
        size,
    )


_MEASUREMENT_FEATURES = {
    "weight": "ACETREE_WEIGHT",
    "rweight": "ACETREE_RWEIGHT",
    "rsum": "ACETREE_RSUM",
    "rcount": "ACETREE_RCOUNT",
    "rwraw": "ACETREE_RWRAW",
    "rwcorr1": "ACETREE_RWCORR1",
    "rwcorr2": "ACETREE_RWCORR2",
    "rwcorr3": "ACETREE_RWCORR3",
    "rwcorr4": "ACETREE_RWCORR4",
}


def _detection_measurements(detection: Detection) -> dict[str, int]:
    """Map explicit interchange features into the fixed legacy nuclei columns."""

    measurements: dict[str, int] = {}
    for field_name, feature_name in _MEASUREMENT_FEATURES.items():
        value = detection.features.get(feature_name, 0)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TrackingProposalConflict(
                f"Detection {detection.detection_id!r} has non-numeric "
                f"{feature_name}"
            ) from exc
        if not math.isfinite(number):
            raise TrackingProposalConflict(
                f"Detection {detection.detection_id!r} has non-finite "
                f"{feature_name}"
            )
        measurements[field_name] = _round_half_away_from_zero(number)
    return measurements


def _round_half_away_from_zero(value: float) -> int:
    """Match MATLAB/AceTree midpoint rounding instead of Python bankers' rounding."""

    value = float(value)
    if not math.isfinite(value):
        raise TrackingProposalConflict("Cannot round a non-finite legacy value")
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _lerp(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction
