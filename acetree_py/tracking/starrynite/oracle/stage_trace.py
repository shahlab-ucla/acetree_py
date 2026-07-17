"""Stage-by-stage parity state for StarryNite's geometry linker.

The classifier event trace starts at ``greedydeleteFPbranches``.  This module
covers the earlier whole-movie boundary: detector rows, mutual easy links,
candidate gathering, every nondivision/division threshold pass, optional
hysteresis, and the final raw pointer state handed to the classifier driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from .event_trace import EventTraceFormatError, LegacyNodePointerState

if TYPE_CHECKING:
    from ..legacy_early import LegacyEarlyStageSnapshot, LegacyEarlyTrackingResult
    from ..legacy_state import LegacyTrackingContext
    from .matlab_backend import MatlabOracleRun


GEOMETRY_STAGE_TRACE_SCHEMA = "acetree.starrynite.geometry-stage-trace/v1"


@dataclass(frozen=True, slots=True)
class GeometryCandidateRelation:
    """One ordered entry in MATLAB's forward or backward candidate cells."""

    direction: str
    source_id: str
    target_id: str

    def __post_init__(self) -> None:
        if self.direction not in {"forward", "backward"}:
            raise EventTraceFormatError(
                "geometry candidate direction must be forward or backward"
            )
        for name in ("source_id", "target_id"):
            value = str(getattr(self, name))
            if not value:
                raise EventTraceFormatError(
                    f"geometry candidate {name} must be non-empty"
                )
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, str]:
        return {
            "direction": self.direction,
            "source": self.source_id,
            "target": self.target_id,
        }


@dataclass(frozen=True, slots=True)
class GeometryStageSnapshot:
    """Raw lineage and candidate state after one geometry-linking stage."""

    index: int
    label: str
    threshold: float | None
    nodes: tuple[LegacyNodePointerState, ...]
    candidate_relations: tuple[GeometryCandidateRelation, ...] = ()

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise EventTraceFormatError("geometry stage index must be non-negative")
        if type(self.label) is not str or not self.label:
            raise EventTraceFormatError("geometry stage label must be non-empty text")
        if self.threshold is not None:
            value = float(self.threshold)
            if math.isnan(value):
                raise EventTraceFormatError("geometry stage threshold cannot be NaN")
            object.__setattr__(self, "threshold", value)
        nodes = tuple(self.nodes)
        if any(not isinstance(item, LegacyNodePointerState) for item in nodes):
            raise EventTraceFormatError(
                "geometry stages must contain LegacyNodePointerState values"
            )
        identities = [item.node_id for item in nodes]
        positions = [(item.frame_0based, item.row_0based) for item in nodes]
        if len(identities) != len(set(identities)):
            raise EventTraceFormatError("geometry stage node IDs must be unique")
        if len(positions) != len(set(positions)):
            raise EventTraceFormatError(
                "geometry stage frame/row references must be unique"
            )
        known = set(identities)
        for node in nodes:
            references = (node.predecessor_id, *node.successor_slots)
            if any(item is not None and item not in known for item in references):
                raise EventTraceFormatError(
                    "geometry stage pointers must reference known detector rows"
                )
        relations = tuple(self.candidate_relations)
        if any(not isinstance(item, GeometryCandidateRelation) for item in relations):
            raise EventTraceFormatError(
                "geometry stages must contain GeometryCandidateRelation values"
            )
        relation_keys = tuple(
            (item.direction, item.source_id, item.target_id) for item in relations
        )
        if len(relation_keys) != len(set(relation_keys)):
            raise EventTraceFormatError("geometry candidate relations must be unique")
        if any(
            item.source_id not in known or item.target_id not in known
            for item in relations
        ):
            raise EventTraceFormatError(
                "geometry candidates must reference known detector rows"
            )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "candidate_relations", relations)

    @property
    def by_id(self) -> Mapping[str, LegacyNodePointerState]:
        return MappingProxyType({item.node_id: item for item in self.nodes})

    @property
    def deleted_ids(self) -> frozenset[str]:
        return frozenset(item.node_id for item in self.nodes if item.deleted)

    @property
    def edge_count(self) -> int:
        return sum(
            successor is not None
            for item in self.nodes
            for successor in item.successor_slots
        )

    @property
    def forward_candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (item.source_id, item.target_id)
            for item in self.candidate_relations
            if item.direction == "forward"
        )

    @property
    def backward_candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (item.source_id, item.target_id)
            for item in self.candidate_relations
            if item.direction == "backward"
        )

    @property
    def candidate_pairs(self) -> tuple[tuple[str, str], ...]:
        """Compatibility alias for forward candidate pairs."""

        return self.forward_candidate_pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "threshold": self.threshold,
            "nodes": [item.to_dict() for item in self.nodes],
            "candidate_relations": [
                item.to_dict() for item in self.candidate_relations
            ],
        }


@dataclass(frozen=True, slots=True)
class GeometryStageTrace:
    """Ordered geometry checkpoints preceding classifier/repair validation."""

    stages: tuple[GeometryStageSnapshot, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if len(stages) < 6:
            raise EventTraceFormatError(
                "geometry stage trace is missing required whole-movie boundaries"
            )
        if any(not isinstance(item, GeometryStageSnapshot) for item in stages):
            raise EventTraceFormatError(
                "geometry trace stages must be GeometryStageSnapshot values"
            )
        if tuple(item.index for item in stages) != tuple(range(len(stages))):
            raise EventTraceFormatError("geometry stage indices must be contiguous")
        required_prefix = (
            "detected",
            "initialized",
            "easy_links",
            "post_polar_filter",
            "candidates",
        )
        if tuple(item.label for item in stages[:5]) != required_prefix:
            raise EventTraceFormatError(
                "geometry stage trace has an invalid initialization sequence"
            )
        if stages[-1].label != "geometry_final":
            raise EventTraceFormatError(
                "geometry stage trace must end at geometry_final"
            )
        allowed_dynamic = {
            "nondivision",
            "hysteresis_cleanup",
            "division",
        }
        if any(item.label not in allowed_dynamic for item in stages[5:-1]):
            raise EventTraceFormatError("geometry stage trace has an unknown stage")
        if any(
            (item.label in {"nondivision", "division"})
            != (item.threshold is not None)
            for item in stages
        ):
            raise EventTraceFormatError(
                "only greedy nondivision/division stages may carry thresholds"
            )
        first_identity = {
            item.node_id: (item.frame_0based, item.row_0based)
            for item in stages[0].nodes
        }
        for stage in stages[1:]:
            identity = {
                item.node_id: (item.frame_0based, item.row_0based)
                for item in stage.nodes
            }
            if identity != first_identity:
                raise EventTraceFormatError(
                    "geometry stages changed immutable detector row identity"
                )
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def final(self) -> GeometryStageSnapshot:
        return self.stages[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GEOMETRY_STAGE_TRACE_SCHEMA,
            "stages": [item.to_dict() for item in self.stages],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class GeometryStageComparison:
    """Exact stage comparison with concise, actionable mismatch paths."""

    matched: bool
    compared_stage_count: int
    mismatches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.matched) is not bool:
            raise TypeError("matched must be a boolean")
        if type(self.compared_stage_count) is not int or self.compared_stage_count < 0:
            raise TypeError("compared_stage_count must be a non-negative integer")
        mismatches = tuple(str(item) for item in self.mismatches)
        if self.matched != (not mismatches):
            raise ValueError("matched must agree with whether mismatches are empty")
        object.__setattr__(self, "mismatches", mismatches)


def geometry_stage_trace_from_legacy_early(
    initial_context: LegacyTrackingContext,
    result: LegacyEarlyTrackingResult,
) -> GeometryStageTrace:
    """Normalize Python's lightweight early-stage snapshots to MATLAB IDs."""

    from ..legacy_early import LegacyEarlyTrackingResult
    from ..legacy_state import LegacyTrackingContext

    if not isinstance(initial_context, LegacyTrackingContext):
        raise TypeError("initial_context must be a LegacyTrackingContext")
    if not isinstance(result, LegacyEarlyTrackingResult):
        raise TypeError("result must be a LegacyEarlyTrackingResult")
    if not result.stages:
        raise EventTraceFormatError(
            "Python early tracking must capture snapshots for stage parity"
        )
    id_map = {
        nucleus.nucleus_id: f"matlab:{nucleus.frame - 1}:{nucleus.matlab_row}"
        for nucleus in sorted(
            initial_context.nuclei,
            key=lambda item: (item.frame, item.matlab_row, item.nucleus_id),
        )
    }
    detected = GeometryStageSnapshot(
        0,
        "detected",
        None,
        _context_nodes(initial_context, id_map),
        (),
    )
    normalized: list[GeometryStageSnapshot] = [detected]
    source_stages = list(result.stages)
    if source_stages[0].label != "initialized":
        raise EventTraceFormatError(
            "Python early trace must begin with the initialized stage"
        )

    def append(source: LegacyEarlyStageSnapshot, label: str) -> None:
        normalized.append(
            GeometryStageSnapshot(
                len(normalized),
                label,
                source.threshold,
                _early_snapshot_nodes(initial_context, source, id_map),
                _early_candidate_relations(source, id_map),
            )
        )

    append(source_stages.pop(0), "initialized")
    if not source_stages or source_stages[0].label != "easy_links":
        raise EventTraceFormatError("Python early trace omitted easy_links")
    easy = source_stages.pop(0)
    append(easy, "easy_links")
    if not source_stages or source_stages[0].label != "post_polar_filter":
        raise EventTraceFormatError(
            "Python early trace omitted post_polar_filter"
        )
    append(source_stages.pop(0), "post_polar_filter")
    if not source_stages or source_stages[0].label != "candidates":
        raise EventTraceFormatError("Python early trace omitted candidates")
    append(source_stages.pop(0), "candidates")
    for source in source_stages:
        if source.label not in {
            "nondivision",
            "hysteresis_cleanup",
            "division",
            "geometry_final",
        }:
            raise EventTraceFormatError(
                f"Unknown Python early stage label: {source.label!r}"
            )
        append(source, source.label)
    if normalized[-1].label != "geometry_final":
        raise EventTraceFormatError(
            "Python early trace must end at geometry_final"
        )
    # Cross-check that the lightweight final snapshot and executable context
    # agree, so a trace cannot claim parity while returning a different graph.
    if normalized[-1].nodes != _context_nodes(result.context, id_map):
        raise EventTraceFormatError(
            "Python geometry_final snapshot differs from result.context"
        )
    final_relations = _candidate_state_relations(result, id_map)
    if normalized[-1].candidate_relations != final_relations:
        raise EventTraceFormatError(
            "Python geometry_final candidates differ from result.candidates"
        )
    return GeometryStageTrace(
        tuple(normalized),
        {"source": "python", "implementation": "legacy_early"},
    )


def compare_geometry_stage_traces(
    reference: GeometryStageTrace,
    candidate: GeometryStageTrace,
) -> GeometryStageComparison:
    """Compare stage order, thresholds, raw pointers, and ordered candidates."""

    if not isinstance(reference, GeometryStageTrace):
        raise TypeError("reference must be a GeometryStageTrace")
    if not isinstance(candidate, GeometryStageTrace):
        raise TypeError("candidate must be a GeometryStageTrace")
    mismatches: list[str] = []
    if len(reference.stages) != len(candidate.stages):
        mismatches.append(
            "stage_count: "
            f"MATLAB={len(reference.stages)} Python={len(candidate.stages)}"
        )
    for index, (expected, actual) in enumerate(
        zip(reference.stages, candidate.stages)
    ):
        prefix = f"stage[{index}]"
        if expected.label != actual.label:
            mismatches.append(
                f"{prefix}.label: MATLAB={expected.label!r} Python={actual.label!r}"
            )
        if not _equal_threshold(expected.threshold, actual.threshold):
            mismatches.append(
                f"{prefix}.threshold: MATLAB={expected.threshold!r} "
                f"Python={actual.threshold!r}"
            )
        if expected.nodes != actual.nodes:
            mismatches.append(f"{prefix}.pointers_or_deletions")
        if expected.candidate_relations != actual.candidate_relations:
            mismatches.append(f"{prefix}.ordered_candidates")
    return GeometryStageComparison(
        not mismatches,
        min(len(reference.stages), len(candidate.stages)),
        tuple(mismatches),
    )


def assert_geometry_stage_parity(
    reference: GeometryStageTrace,
    candidate: GeometryStageTrace,
) -> None:
    """Raise an assertion with stage-specific diagnostics on any mismatch."""

    comparison = compare_geometry_stage_traces(reference, candidate)
    if not comparison.matched:
        raise AssertionError(
            "StarryNite geometry-stage parity failed:\n- "
            + "\n- ".join(comparison.mismatches)
        )


def _context_nodes(
    context: LegacyTrackingContext,
    id_map: Mapping[str, str],
) -> tuple[LegacyNodePointerState, ...]:
    result = []
    for nucleus in sorted(
        context.nuclei,
        key=lambda item: (item.frame, item.matlab_row, item.nucleus_id),
    ):
        native_id = nucleus.nucleus_id
        if native_id not in id_map:
            raise EventTraceFormatError(
                f"Python geometry context invented detector row {native_id!r}"
            )
        predecessor = context.predecessor(native_id)
        slots = context.successor_slots(native_id)
        result.append(
            LegacyNodePointerState(
                node_id=id_map[native_id],
                frame_0based=nucleus.frame - 1,
                row_0based=nucleus.matlab_row,
                deleted=native_id in context.deleted_ids,
                predecessor_id=None if predecessor is None else id_map[predecessor],
                successor_slots=tuple(
                    None if item is None else id_map[item] for item in slots
                ),
            )
        )
    return tuple(result)


def _early_snapshot_nodes(
    context: LegacyTrackingContext,
    snapshot: LegacyEarlyStageSnapshot,
    id_map: Mapping[str, str],
) -> tuple[LegacyNodePointerState, ...]:
    slots: dict[str, list[str | None]] = {
        nucleus_id: [None, None] for nucleus_id in id_map
    }
    predecessor: dict[str, str | None] = {nucleus_id: None for nucleus_id in id_map}
    for source_id, slot, target_id in snapshot.pointers:
        if source_id not in slots or target_id not in slots:
            raise EventTraceFormatError(
                "Python geometry snapshot pointer references an unknown row"
            )
        if slot not in {0, 1} or slots[source_id][slot] is not None:
            raise EventTraceFormatError(
                "Python geometry snapshot contains an invalid successor slot"
            )
        if predecessor[target_id] is not None:
            raise EventTraceFormatError(
                "Python geometry snapshot contains a lineage merge"
            )
        slots[source_id][slot] = target_id
        predecessor[target_id] = source_id
    deleted = set(snapshot.deleted_ids)
    if not deleted <= set(id_map):
        raise EventTraceFormatError(
            "Python geometry snapshot deleted an unknown detector row"
        )
    nodes = []
    for nucleus in sorted(
        context.nuclei,
        key=lambda item: (item.frame, item.matlab_row, item.nucleus_id),
    ):
        native_id = nucleus.nucleus_id
        nodes.append(
            LegacyNodePointerState(
                node_id=id_map[native_id],
                frame_0based=nucleus.frame - 1,
                row_0based=nucleus.matlab_row,
                deleted=native_id in deleted,
                predecessor_id=(
                    None
                    if predecessor[native_id] is None
                    else id_map[predecessor[native_id]]
                ),
                successor_slots=tuple(
                    None if item is None else id_map[item]
                    for item in slots[native_id]
                ),
            )
        )
    return tuple(nodes)


def _early_candidate_relations(
    snapshot: LegacyEarlyStageSnapshot,
    id_map: Mapping[str, str],
) -> tuple[GeometryCandidateRelation, ...]:
    relations: list[GeometryCandidateRelation] = []
    for source_id, target_id in snapshot.forward_candidates:
        relations.append(
            GeometryCandidateRelation("forward", id_map[source_id], id_map[target_id])
        )
    for source_id, target_id in snapshot.backward_candidates:
        relations.append(
            GeometryCandidateRelation("backward", id_map[source_id], id_map[target_id])
        )
    return tuple(relations)


def _candidate_state_relations(
    result: LegacyEarlyTrackingResult,
    id_map: Mapping[str, str],
) -> tuple[GeometryCandidateRelation, ...]:
    relations: list[GeometryCandidateRelation] = []
    for source_id in id_map:
        for target_id in result.candidates.forward(source_id):
            relations.append(
                GeometryCandidateRelation(
                    "forward", id_map[source_id], id_map[target_id]
                )
            )
    for target_id in id_map:
        for source_id in result.candidates.backward(target_id):
            relations.append(
                GeometryCandidateRelation(
                    "backward", id_map[source_id], id_map[target_id]
                )
            )
    return tuple(relations)


def _equal_threshold(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is second
    if math.isinf(first) or math.isinf(second):
        return first == second
    return math.isclose(first, second, rel_tol=1e-13, abs_tol=1e-15)


def matlab_geometry_stage_trace(run: MatlabOracleRun) -> GeometryStageTrace:
    """Decode and validate the neutral stage tables emitted by MATLAB."""

    value = run.result.get("tracking_stage_trace")
    if not isinstance(value, Mapping):
        raise EventTraceFormatError(
            "MATLAB tracking result lacks a tracking_stage_trace structure"
        )
    version = _float_integer(value.get("schema_version"), "stage schema_version")
    if version != 1:
        raise EventTraceFormatError(
            f"Unsupported MATLAB geometry stage schema version: {version}"
        )
    if not _logical_scalar(value.get("finished"), "stage finished"):
        raise EventTraceFormatError("MATLAB geometry stage trace is unfinished")
    count = _float_integer(value.get("stage_count"), "stage_count")
    if count < 6:
        raise EventTraceFormatError("MATLAB geometry stage trace is incomplete")
    labels = _text_items(value.get("stage_labels"), count, "stage_labels")
    thresholds = _numeric_vector(
        value.get("stage_thresholds"), count, "stage_thresholds"
    )
    snapshots = _table_cells(
        value.get("stage_snapshots"), count, 9, "stage_snapshots"
    )
    candidate_tables = _table_cells(
        value.get("stage_candidate_tables"),
        count,
        5,
        "stage_candidate_tables",
    )
    expected_snapshot_columns = (
        "frame_0based",
        "node_0based",
        "deleted",
        "predecessor_frame_0based",
        "predecessor_node_0based",
        "successor1_frame_0based",
        "successor1_node_0based",
        "successor2_frame_0based",
        "successor2_node_0based",
    )
    expected_candidate_columns = (
        "direction_code",
        "source_frame_0based",
        "source_node_0based",
        "target_frame_0based",
        "target_node_0based",
    )
    if _all_text(value.get("snapshot_columns")) != expected_snapshot_columns:
        raise EventTraceFormatError("MATLAB geometry snapshot columns changed")
    if _all_text(value.get("candidate_columns")) != expected_candidate_columns:
        raise EventTraceFormatError("MATLAB geometry candidate columns changed")

    node_counts = _numeric_integer_vector(value, "stage_node_counts", count)
    active_counts = _numeric_integer_vector(value, "stage_active_counts", count)
    deleted_counts = _numeric_integer_vector(value, "stage_deleted_counts", count)
    edge_counts = _numeric_integer_vector(value, "stage_edge_counts", count)
    candidate_counts = _numeric_integer_vector(
        value, "stage_candidate_counts", count
    )

    stages: list[GeometryStageSnapshot] = []
    for index, (label, threshold, snapshot, candidate_table) in enumerate(
        zip(labels, thresholds, snapshots, candidate_tables, strict=True)
    ):
        nodes = tuple(_matlab_snapshot_node(row) for row in snapshot)
        candidates = tuple(
            _matlab_candidate_relation(row) for row in candidate_table
        )
        stage = GeometryStageSnapshot(
            index=index,
            label=label,
            threshold=None if math.isnan(float(threshold)) else float(threshold),
            nodes=nodes,
            candidate_relations=candidates,
        )
        if len(nodes) != node_counts[index]:
            raise EventTraceFormatError("MATLAB geometry node count is inconsistent")
        if len(stage.deleted_ids) != deleted_counts[index]:
            raise EventTraceFormatError(
                "MATLAB geometry deleted-node count is inconsistent"
            )
        if len(nodes) - len(stage.deleted_ids) != active_counts[index]:
            raise EventTraceFormatError(
                "MATLAB geometry active-node count is inconsistent"
            )
        if stage.edge_count != edge_counts[index]:
            raise EventTraceFormatError("MATLAB geometry edge count is inconsistent")
        if len(candidates) != candidate_counts[index]:
            raise EventTraceFormatError(
                "MATLAB geometry candidate count is inconsistent"
            )
        stages.append(stage)

    return GeometryStageTrace(
        tuple(stages),
        {
            "source": "matlab",
            "matlab_version": run.matlab_version,
            "starrynite_root": str(run.result.get("starrynite_root", "")),
            "upstream_function": str(run.result.get("upstream_function", "")),
        },
    )


def _matlab_snapshot_node(row: Sequence[float]) -> LegacyNodePointerState:
    frame = _float_integer(row[0], "snapshot frame")
    node = _float_integer(row[1], "snapshot node")
    deleted = _float_integer(row[2], "snapshot deleted")
    if deleted not in {0, 1}:
        raise EventTraceFormatError("snapshot deleted flag must be 0 or 1")
    return LegacyNodePointerState(
        node_id=f"matlab:{frame}:{node}",
        frame_0based=frame,
        row_0based=node,
        deleted=bool(deleted),
        predecessor_id=_optional_reference(row[3], row[4], "snapshot predecessor"),
        successor_slots=(
            _optional_reference(row[5], row[6], "snapshot successor 1"),
            _optional_reference(row[7], row[8], "snapshot successor 2"),
        ),
    )


def _matlab_candidate_relation(row: Sequence[float]) -> GeometryCandidateRelation:
    direction_code = _float_integer(row[0], "candidate direction")
    if direction_code not in {0, 1}:
        raise EventTraceFormatError("candidate direction code must be 0 or 1")
    source_frame = _float_integer(row[1], "candidate source frame")
    source_node = _float_integer(row[2], "candidate source node")
    target_frame = _float_integer(row[3], "candidate target frame")
    target_node = _float_integer(row[4], "candidate target node")
    if min(source_frame, source_node, target_frame, target_node) < 0:
        raise EventTraceFormatError("candidate references must be non-negative")
    if target_frame <= source_frame:
        raise EventTraceFormatError("geometry candidates must point forward in time")
    return GeometryCandidateRelation(
        "forward" if direction_code == 0 else "backward",
        f"matlab:{source_frame}:{source_node}",
        f"matlab:{target_frame}:{target_node}",
    )


def _optional_reference(frame: Any, node: Any, label: str) -> str | None:
    frame_value = _float_integer(frame, f"{label} frame")
    node_value = _float_integer(node, f"{label} node")
    if frame_value == -1 and node_value == -1:
        return None
    if frame_value < 0 or node_value < 0:
        raise EventTraceFormatError(
            f"{label} must use non-negative indices or the -1/-1 sentinel"
        )
    return f"matlab:{frame_value}:{node_value}"


def _table_cells(
    value: Any,
    count: int,
    width: int,
    label: str,
) -> tuple[np.ndarray, ...]:
    items = _cell_items(value)
    if len(items) != count:
        raise EventTraceFormatError(
            f"MATLAB emitted {len(items)} {label}; expected {count}"
        )
    return tuple(
        _numeric_table(item, width, f"{label}[{index}]")
        for index, item in enumerate(items)
    )


def _numeric_table(value: Any, width: int, label: str) -> np.ndarray:
    try:
        table = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EventTraceFormatError(f"{label} must be numeric") from exc
    if table.size == 0:
        return np.empty((0, width), dtype=float)
    table = np.atleast_2d(table)
    if table.ndim != 2 or table.shape[1] != width:
        raise EventTraceFormatError(f"{label} must be an N-by-{width} table")
    if not np.all(np.isfinite(table)):
        raise EventTraceFormatError(f"{label} must be finite")
    return table


def _cell_items(value: Any) -> tuple[Any, ...]:
    if isinstance(value, np.ndarray) and value.dtype == object:
        return tuple(value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _all_text(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _cell_items(value))


def _text_items(value: Any, count: int, label: str) -> tuple[str, ...]:
    items = _all_text(value)
    if len(items) != count or any(not item for item in items):
        raise EventTraceFormatError(f"{label} must contain {count} text values")
    return items


def _numeric_vector(value: Any, count: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise EventTraceFormatError(f"{label} must be numeric") from exc
    if result.size != count:
        raise EventTraceFormatError(f"{label} must contain {count} values")
    return result


def _numeric_integer_vector(
    value: Mapping[str, Any], name: str, count: int
) -> tuple[int, ...]:
    raw = _numeric_vector(value.get(name), count, name)
    return tuple(_float_integer(item, name) for item in raw)


def _float_integer(value: Any, label: str) -> int:
    try:
        number = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise EventTraceFormatError(f"{label} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise EventTraceFormatError(f"{label} must be an integer")
    return int(number)


def _logical_scalar(value: Any, label: str) -> bool:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise EventTraceFormatError(f"{label} must be scalar")
    scalar = array[0]
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    try:
        number = float(scalar)
    except (TypeError, ValueError) as exc:
        raise EventTraceFormatError(f"{label} must be logical") from exc
    if number not in {0.0, 1.0}:
        raise EventTraceFormatError(f"{label} must be logical")
    return bool(number)


__all__ = [
    "GEOMETRY_STAGE_TRACE_SCHEMA",
    "GeometryCandidateRelation",
    "GeometryStageComparison",
    "GeometryStageSnapshot",
    "GeometryStageTrace",
    "assert_geometry_stage_parity",
    "compare_geometry_stage_traces",
    "geometry_stage_trace_from_legacy_early",
    "matlab_geometry_stage_trace",
]
