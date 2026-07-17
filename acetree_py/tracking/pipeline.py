"""Orchestration for global and selected-cell image-analysis runs.

The pipeline is intentionally side-effect free: it reads image stacks and an
optional nuclei record, then returns a :class:`TrackingResult` proposal.  The
editing layer owns preview/acceptance and is the only layer allowed to mutate
AceTree nuclei.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.nucleus import Nucleus
from .api import (
    Calibration,
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackerGraphResult,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    WholeMoviePreflightContext,
)
from .registry import TrackingRegistry, get_default_registry

if TYPE_CHECKING:
    from ..io.image_provider import ImageProvider


class TrackingCancelled(RuntimeError):
    """Raised when a caller cancels an analysis run."""


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class _ForwardBranchState:
    """One live branch in selected-cell sparse forward tracking."""

    previous: Detection | None
    last: Detection
    missing: int = 0


class TrackingPipeline:
    """Run registered detector and tracker components against an image source."""

    def __init__(self, registry: TrackingRegistry | None = None) -> None:
        self.registry = registry or get_default_registry()

    def run(
        self,
        image_provider: ImageProvider,
        calibration: Calibration,
        request: TrackingRequest,
        *,
        nuclei_record: list[list[Nucleus]] | None = None,
        cancelled: CancelCallback | None = None,
        progress: ProgressCallback | None = None,
    ) -> TrackingResult:
        """Execute *request* and return an immutable, uncommitted proposal."""
        _validate_source_bounds(image_provider, request)
        kind = request.scope.kind
        if kind == "global":
            return self._run_global(
                image_provider, calibration, request,
                cancelled=cancelled, progress=progress,
            )
        if kind == "selected_forward":
            if nuclei_record is None:
                raise ValueError("selected_forward tracking requires a nuclei record")
            return self._run_selected_forward(
                image_provider, calibration, request, nuclei_record,
                cancelled=cancelled, progress=progress,
            )
        raise ValueError(f"Unsupported tracking scope: {kind!r}")

    def detect_frame(
        self,
        image_provider: ImageProvider,
        calibration: Calibration,
        detector_spec: ComponentSpec,
        *,
        frame: int,
        cancelled: CancelCallback | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Detection, ...]:
        """Run only one detector on one complete 3D frame.

        This is the fast parameter-tuning path used by the whole-dataset
        workbench.  It never constructs a tracker, creates links, or returns
        an accept-capable tracking proposal.
        """

        frame = int(frame)
        if frame < 1 or frame > image_provider.num_timepoints:
            raise ValueError(
                f"Detector preview frame {frame} is unavailable; the image source "
                f"has {image_provider.num_timepoints} timepoint(s)"
            )
        channel = _target_channel_from_settings(detector_spec.settings)
        if channel >= image_provider.num_channels:
            raise ValueError(
                f"TARGET_CHANNEL {channel + 1} is unavailable; the image source has "
                f"{image_provider.num_channels} channel(s)"
            )

        _check_cancelled(cancelled)
        detector = self.registry.create_detector(detector_spec.plugin_id)
        _check_cancelled(cancelled)
        if progress is not None:
            progress(0, 1, f"Loading detector image at time {frame}")
        stack = image_provider.get_stack(frame, channel)
        _check_cancelled(cancelled)
        detections = tuple(
            detector.detect(
                stack,
                frame,
                calibration,
                detector_spec.settings,
            )
        )
        _check_cancelled(cancelled)

        if any(detection.frame != frame for detection in detections):
            raise ValueError(
                "A detector returned a position for a different frame during preview"
            )
        ids = [detection.detection_id for detection in detections]
        if len(ids) != len(set(ids)):
            raise ValueError("Detector preview IDs must be unique")
        ordered = tuple(
            sorted(
                detections,
                key=lambda detection: (
                    detection.z_um,
                    detection.y_um,
                    detection.x_um,
                    detection.detection_id,
                ),
            )
        )
        if progress is not None:
            progress(
                1,
                1,
                f"Found {len(ordered)} detector candidate(s) at time {frame}",
            )
        return ordered

    def _run_global(
        self,
        image_provider: ImageProvider,
        calibration: Calibration,
        request: TrackingRequest,
        *,
        cancelled: CancelCallback | None,
        progress: ProgressCallback | None,
    ) -> TrackingResult:
        detector = self.registry.create_detector(request.detector.plugin_id)
        tracker = self.registry.create_tracker(request.tracker.plugin_id)
        channel = _target_channel(request)
        preflight_movie = getattr(tracker, "preflight_movie", None)
        if callable(preflight_movie):
            _check_cancelled(cancelled)
            preflight_result = preflight_movie(
                request.tracker.settings,
                context=WholeMoviePreflightContext(
                    detector_spec=request.detector,
                    calibration=calibration,
                    scope=request.scope,
                    source_num_timepoints=image_provider.num_timepoints,
                    source_num_channels=image_provider.num_channels,
                    target_channel=channel,
                ),
            )
            if preflight_result is not None:
                raise TypeError("A tracker's preflight_movie method must return None")
            _check_cancelled(cancelled)
        frames = range(request.scope.start_frame, request.scope.end_frame + 1)
        total = max(0, request.scope.end_frame - request.scope.start_frame + 1)
        detections: list[Detection] = []

        for done, frame in enumerate(frames, start=1):
            _check_cancelled(cancelled)
            stack = image_provider.get_stack(frame, channel)
            detections.extend(
                detector.detect(
                    stack,
                    frame,
                    calibration,
                    request.detector.settings,
                )
            )
            if progress is not None:
                progress(done, total, f"Detecting nuclei at time {frame}")

        _check_cancelled(cancelled)
        raw_detections = tuple(detections)
        refine_movie = getattr(tracker, "refine_movie", None)
        refine_graph = getattr(tracker, "refine_graph", None)
        graph_provenance: dict[str, Any] = {}
        warnings: tuple[str, ...] = ()
        used_refinement = False
        if callable(refine_movie):
            # Whole-movie compatibility backends need the immutable run scope
            # and voxel calibration to validate legacy anisotropy and temporal
            # history.  Keep this separate from the lightweight refine_graph
            # hook so existing graph refiners retain their two-argument API.
            graph = refine_movie(
                raw_detections,
                request.tracker.settings,
                detector_spec=request.detector,
                calibration=calibration,
                start_frame=request.scope.start_frame,
                end_frame=request.scope.end_frame,
                cancelled=cancelled,
                progress=progress,
            )
            used_refinement = True
        elif callable(refine_graph):
            graph = refine_graph(raw_detections, request.tracker.settings)
            used_refinement = True
        else:
            graph = None
        if used_refinement:
            if not isinstance(graph, TrackerGraphResult):
                raise TypeError(
                    "A tracker's whole-graph refinement method must return "
                    "TrackerGraphResult"
                )
            raw_ids = {item.detection_id for item in raw_detections}
            retained_ids = {item.detection_id for item in graph.detections}
            if not retained_ids <= raw_ids:
                raise ValueError("A graph refiner cannot invent detector positions")
            expected_rejected = raw_ids - retained_ids
            if set(graph.rejected_detection_ids) != expected_rejected:
                raise ValueError(
                    "Graph refinement must identify every omitted detector position"
                )
            result_detections = graph.detections
            edges = graph.edges
            warnings = graph.warnings
            graph_provenance = {
                "rejected_detection_count": len(graph.rejected_detection_ids),
                **dict(graph.provenance),
            }
        else:
            result_detections = raw_detections
            edges = tracker.track(raw_detections, request.tracker.settings)
        if progress is not None:
            progress(total, total, "Linking detections")
        provenance = _provenance(self.registry, request, mode="global")
        if graph_provenance:
            provenance["graph_refinement"] = graph_provenance
        return TrackingResult(
            request=request,
            detections=tuple(result_detections),
            edges=tuple(edges),
            existing_anchors={},
            warnings=warnings,
            provenance=provenance,
        )

    def _run_selected_forward(
        self,
        image_provider: ImageProvider,
        calibration: Calibration,
        request: TrackingRequest,
        nuclei_record: list[list[Nucleus]],
        *,
        cancelled: CancelCallback | None,
        progress: ProgressCallback | None,
    ) -> TrackingResult:
        anchors = request.scope.seed_anchors
        if len(anchors) != 1:
            raise ValueError("The prototype supports exactly one selected-forward seed")
        seed_time, seed_index = anchors[0]
        seed = _get_alive_nucleus(nuclei_record, seed_time, seed_index)
        if request.scope.start_frame != seed_time:
            raise ValueError("selected_forward scope must start at the seed time")
        if request.scope.end_frame <= seed_time:
            raise ValueError("The ending time must be after the selected nucleus")

        detector = self.registry.create_detector(request.detector.plugin_id)
        tracker = self.registry.create_tracker(request.tracker.plugin_id)
        channel = _target_channel(request)
        x_um, y_um, z_um = calibration.pixel_to_physical(seed.x, seed.y, seed.z)
        radius_um = max(calibration.xy_um, seed.size * calibration.xy_um / 2.0)
        seed_id = f"existing:{seed_time}:{seed_index}"
        seed_detection = Detection(
            detection_id=seed_id,
            frame=seed_time,
            x_um=x_um,
            y_um=y_um,
            z_um=z_um,
            radius_um=radius_um,
            quality=1.0,
            features={"MANUAL_SEED": True, "acetree_existing": 1.0},
        )

        branch_policy = getattr(request.scope, "branch_policy", "stop")
        if branch_policy == "follow_both":
            return self._run_selected_forward_branches(
                image_provider,
                calibration,
                request,
                nuclei_record,
                seed_detection,
                detector,
                tracker,
                channel,
                cancelled=cancelled,
                progress=progress,
            )

        accepted: list[Detection] = [seed_detection]
        edges: list[TrackEdge] = []
        warnings: list[str] = []
        last = seed_detection
        previous: Detection | None = None
        missing = 0
        max_missing = max(
            0,
            int(request.tracker.settings.get("MAX_FRAME_GAP", 1)) - 1,
        )
        allow_gap = bool(request.tracker.settings.get("ALLOW_GAP_CLOSING", True))
        ambiguity_ratio = max(1.0, float(request.scope.ambiguity_ratio))
        search_radius_um = (
            request.scope.roi_radius_um
            if request.scope.roi_radius_um is not None
            else 12.0
        )
        total = request.scope.end_frame - seed_time
        outcome: TrackingOutcome | None = None
        last_attempt_frame: int | None = None
        last_prediction: tuple[float, float, float] | None = None
        last_review_candidates: tuple[Detection, ...] = ()

        for done, frame in enumerate(range(seed_time + 1, request.scope.end_frame + 1), start=1):
            _check_cancelled(cancelled)
            stack = np.asarray(image_provider.get_stack(frame, channel))
            predicted = _predict_position(previous, last, frame)
            last_attempt_frame = frame
            last_prediction = predicted
            crop, offset_zyx = _crop_around(
                stack,
                predicted,
                calibration,
                search_radius_um,
            )
            detected_candidates = list(
                detector.detect(
                    crop,
                    frame,
                    calibration,
                    request.detector.settings,
                    offset_zyx=offset_zyx,
                )
            )
            last_review_candidates = _ordered_review_candidates(detected_candidates)
            ranked_before_conflicts = _rank_candidates(
                last, detected_candidates, request
            )
            candidates, collided_ids = _exclude_existing_detections(
                detected_candidates,
                nuclei_record,
                frame,
                calibration,
            )
            if (
                ranked_before_conflicts
                and ranked_before_conflicts[0][1] in collided_ids
            ):
                warnings.append(
                    f"Stopped at t={frame}: a candidate overlaps an existing "
                    "curated nucleus"
                )
                outcome = _stopped_outcome(
                    "conflict",
                    frame,
                    last,
                    predicted,
                    search_radius_um,
                    detected_candidates,
                )
                break

            ranked = _rank_candidates(last, candidates, request)
            last_review_candidates = _ordered_review_candidates(candidates)
            # A close, balanced pair around the predicted position is more
            # biologically actionable than the generic equal-cost condition.
            # Diagnose the probable division first so review can show both
            # daughter candidates with the appropriate explanation.
            probable_division = _looks_like_division(candidates, ranked, predicted)
            if probable_division and branch_policy == "stop":
                warnings.append(
                    f"Stopped at t={frame}: two candidates form a probable division; "
                    "division behavior is set to stop and review"
                )
                outcome = _stopped_outcome(
                    "division",
                    frame,
                    last,
                    predicted,
                    search_radius_um,
                    _ranked_review_candidates(candidates, ranked),
                )
                break
            if (
                len(ranked) > 1
                and _costs_are_ambiguous(ranked, ambiguity_ratio)
                and not (probable_division and branch_policy == "follow_best")
            ):
                warnings.append(
                    f"Stopped at t={frame}: two candidates had similar assignment costs"
                )
                outcome = _stopped_outcome(
                    "ambiguity",
                    frame,
                    last,
                    predicted,
                    search_radius_um,
                    _ranked_review_candidates(candidates, ranked),
                )
                break

            # With one active source, the first LAP stage reduces to choosing
            # the lowest admissible assignment.  Asking the registered tracker
            # to produce the edge keeps distance units and cost semantics
            # identical to global tracking.
            candidate_edges = tuple(
                tracker.track((last, *candidates), request.tracker.settings)
            )
            outgoing = sorted(
                (edge for edge in candidate_edges if edge.source_id == last.detection_id),
                key=lambda edge: (edge.cost, edge.target_id),
            )

            split_edges = tuple(edge for edge in outgoing if edge.kind == "split")
            if split_edges and len(split_edges) != 2:
                raise RuntimeError("Tracker returned an incomplete division event")
            if split_edges and branch_policy == "stop":
                by_id = {candidate.detection_id: candidate for candidate in candidates}
                daughters = tuple(
                    by_id[edge.target_id]
                    for edge in split_edges
                    if edge.target_id in by_id
                )
                warnings.append(
                    f"Stopped at t={frame}: the tracker proposed a two-daughter division"
                )
                outcome = _stopped_outcome(
                    "division",
                    frame,
                    last,
                    predicted,
                    search_radius_um,
                    daughters,
                )
                break

            if not outgoing:
                missing += 1
                if not allow_gap or missing > max_missing:
                    warnings.append(
                        f"Stopped at t={frame}: no unique candidate passed the distance gate"
                    )
                    outcome = _stopped_outcome(
                        "lost",
                        frame,
                        last,
                        predicted,
                        search_radius_um,
                        candidates,
                    )
                    break
                if progress is not None:
                    progress(done, total, f"No candidate at time {frame}; trying gap closure")
                continue

            chosen_edge = outgoing[0]
            by_id = {candidate.detection_id: candidate for candidate in candidates}
            chosen = by_id.get(chosen_edge.target_id)
            if chosen is None:
                raise RuntimeError("Tracker returned an edge to an unknown detection")
            frame_delta = chosen.frame - last.frame
            kind = "gap" if frame_delta > 1 else "link"
            edges.append(replace(chosen_edge, kind=kind))
            accepted.append(chosen)
            previous, last = last, chosen
            missing = 0
            if progress is not None:
                progress(done, total, f"Tracking selected cell at time {frame}")

        if outcome is None:
            if last.frame == request.scope.end_frame:
                outcome = TrackingOutcome(
                    code="completed",
                    stop_frame=None,
                    last_accepted_frame=last.frame,
                    predicted_position_um=None,
                    search_radius_um=search_radius_um,
                )
            else:
                # A missing observation at the end of the requested range can
                # be within the configured gap allowance, but there is no later
                # frame available to close it.  Report it as lost instead of
                # silently calling the shorter track complete.
                if last_attempt_frame is None or last_prediction is None:
                    raise RuntimeError("Selected-forward tracking made no attempt")
                warnings.append(
                    f"Stopped at t={last_attempt_frame}: no unique candidate "
                    "passed the distance gate"
                )
                outcome = TrackingOutcome(
                    code="lost",
                    stop_frame=last_attempt_frame,
                    last_accepted_frame=last.frame,
                    predicted_position_um=last_prediction,
                    search_radius_um=search_radius_um,
                    review_candidates=last_review_candidates,
                )

        return TrackingResult(
            request=request,
            detections=tuple(accepted),
            edges=tuple(edges),
            existing_anchors={seed_id: (seed_time, seed_index)},
            warnings=tuple(warnings),
            provenance=_provenance(self.registry, request, mode="selected_forward"),
            outcome=outcome,
        )

    def _run_selected_forward_branches(
        self,
        image_provider: ImageProvider,
        calibration: Calibration,
        request: TrackingRequest,
        nuclei_record: list[list[Nucleus]],
        seed_detection: Detection,
        detector,
        tracker,
        channel: int,
        *,
        cancelled: CancelCallback | None,
        progress: ProgressCallback | None,
    ) -> TrackingResult:
        """Track a small selected lineage frontier and retain two-daughter splits.

        This path intentionally stays local: every active branch owns one moving
        ROI and asks the registered tracker to choose among only those nearby
        observations.  Events are then reconciled deterministically so two
        branches cannot claim the same detection.  It is suitable for selected
        cells, not embryo-wide exhaustive detection.
        """

        descriptor = self.registry.get_descriptor(request.tracker.plugin_id)
        if "splitting" not in descriptor.capabilities:
            raise ValueError(
                f"Tracker {request.tracker.plugin_id!r} cannot follow both daughters"
            )
        if not bool(request.tracker.settings.get("ALLOW_TRACK_SPLITTING", False)):
            raise ValueError("Follow-both tracking requires ALLOW_TRACK_SPLITTING")

        scope = request.scope
        search_radius_um = scope.roi_radius_um or 12.0
        ambiguity_ratio = max(1.0, float(scope.ambiguity_ratio))
        max_missing = max(
            0,
            int(request.tracker.settings.get("MAX_FRAME_GAP", 1)) - 1,
        )
        allow_gap = bool(request.tracker.settings.get("ALLOW_GAP_CLOSING", True))
        if (
            allow_gap
            and max_missing > 0
            and "frontier_tracking" not in descriptor.capabilities
        ):
            raise ValueError(
                f"Tracker {request.tracker.plugin_id!r} cannot combine follow-both "
                "division tracking with gap closure"
            )
        max_active = max(2, int(request.tracker.settings.get("MAX_ACTIVE_BRANCHES", 8)))
        total = scope.end_frame - scope.start_frame

        active: dict[str, _ForwardBranchState] = {
            seed_detection.detection_id: _ForwardBranchState(None, seed_detection)
        }
        accepted: dict[str, Detection] = {
            seed_detection.detection_id: seed_detection
        }
        edges: list[TrackEdge] = []
        warnings: list[str] = []

        for done, frame in enumerate(
            range(scope.start_frame + 1, scope.end_frame + 1), start=1
        ):
            _check_cancelled(cancelled)
            stack = np.asarray(image_provider.get_stack(frame, channel))
            candidate_by_id: dict[str, Detection] = {}
            predictions: dict[str, tuple[float, float, float]] = {}
            for branch_id in sorted(active):
                branch = active[branch_id]
                predicted = _predict_position(branch.previous, branch.last, frame)
                predictions[branch_id] = predicted
                crop, offset_zyx = _crop_around(
                    stack,
                    predicted,
                    calibration,
                    search_radius_um,
                )
                for candidate in detector.detect(
                    crop,
                    frame,
                    calibration,
                    request.detector.settings,
                    offset_zyx=offset_zyx,
                ):
                    current = candidate_by_id.get(candidate.detection_id)
                    if current is None or (
                        candidate.quality,
                        -candidate.z_um,
                        -candidate.y_um,
                        -candidate.x_um,
                    ) > (
                        current.quality,
                        -current.z_um,
                        -current.y_um,
                        -current.x_um,
                    ):
                        candidate_by_id[candidate.detection_id] = candidate

            detected_candidates = tuple(
                sorted(candidate_by_id.values(), key=lambda item: item.detection_id)
            )
            candidates, collided_ids = _exclude_existing_detections(
                detected_candidates,
                nuclei_record,
                frame,
                calibration,
            )
            candidates = tuple(candidates)

            # Solve each same-time lineage frontier together.  Calling a
            # division tracker once per branch would make two ordinary sister
            # continuations look like an excess-target split to each sister.
            # A frontier-aware tracker keeps source/target roles explicit even
            # when one branch is behind after a missing frame. Older splitting
            # plugins can use the ordinary API while every source shares a frame.
            candidate_ids = {candidate.detection_id for candidate in candidates}
            proposed_by_source: dict[str, tuple[TrackEdge, ...]] = {}
            frontier_tracker = getattr(tracker, "track_frontier", None)
            if callable(frontier_tracker):
                frontier = tuple(
                    active[branch_id].last for branch_id in sorted(active)
                )
                proposed = frontier_tracker(
                    frontier,
                    candidates,
                    request.tracker.settings,
                )
                for source in frontier:
                    source_id = source.detection_id
                    proposed_by_source[source_id] = tuple(
                        edge
                        for edge in proposed
                        if edge.source_id == source_id and edge.target_id in candidate_ids
                    )
            else:
                active_by_last_frame: dict[int, list[_ForwardBranchState]] = {}
                for branch in active.values():
                    active_by_last_frame.setdefault(branch.last.frame, []).append(branch)
                if len(active_by_last_frame) > 1:
                    raise ValueError(
                        "This splitting tracker cannot solve a sparse frontier with "
                        "branches separated by missing frames"
                    )
                for last_frame in sorted(active_by_last_frame):
                    frontier = sorted(
                        active_by_last_frame[last_frame],
                        key=lambda branch: branch.last.detection_id,
                    )
                    source_ids = {branch.last.detection_id for branch in frontier}
                    proposed = tracker.track(
                        tuple(branch.last for branch in frontier) + candidates,
                        request.tracker.settings,
                    )
                    for source_id in source_ids:
                        proposed_by_source[source_id] = tuple(
                            edge
                            for edge in proposed
                            if edge.source_id == source_id
                            and edge.target_id in candidate_ids
                        )

            # Build one atomic event (continuation or two-daughter split) per
            # branch.  Ambiguous non-division branches pause independently;
            # other branches remain useful and can continue.
            events: list[tuple[float, str, tuple[TrackEdge, ...]]] = []
            paused: dict[str, str] = {}
            for branch_id in sorted(active):
                branch = active[branch_id]
                ranked_before_conflicts = _rank_candidates(
                    branch.last, detected_candidates, request
                )
                if (
                    ranked_before_conflicts
                    and ranked_before_conflicts[0][1] in collided_ids
                ):
                    paused[branch_id] = "overlaps an existing curated nucleus"
                    continue

                ranked = _rank_candidates(branch.last, candidates, request)
                probable_division = _looks_like_division(
                    candidates,
                    ranked,
                    predictions[branch_id],
                )
                outgoing = sorted(
                    proposed_by_source.get(branch.last.detection_id, ()),
                    key=lambda edge: (edge.cost, edge.kind, edge.target_id),
                )
                split_edges = tuple(edge for edge in outgoing if edge.kind == "split")
                if split_edges:
                    if len(split_edges) != 2:
                        paused[branch_id] = "tracker returned an incomplete division"
                        continue
                    event_edges = split_edges
                elif probable_division and len(candidates) > len(active):
                    paused[branch_id] = (
                        "candidates form a probable division but the tracker did not "
                        "return both daughters"
                    )
                    continue
                elif len(ranked) > 1 and _costs_are_ambiguous(
                    ranked,
                    ambiguity_ratio,
                ):
                    paused[branch_id] = "two candidates had similar assignment costs"
                    continue
                elif outgoing:
                    event_edges = (outgoing[0],)
                else:
                    event_edges = ()
                if event_edges:
                    events.append(
                        (
                            float(sum(edge.cost for edge in event_edges)),
                            branch_id,
                            event_edges,
                        )
                    )

            # Lowest event cost wins any shared target.  A split is selected or
            # rejected as a pair, never half-applied.
            selected_events: dict[str, tuple[TrackEdge, ...]] = {}
            claimed_targets: set[str] = set()
            for _cost, branch_id, event_edges in sorted(
                events,
                key=lambda item: (
                    item[0],
                    item[1],
                    tuple(edge.target_id for edge in item[2]),
                ),
            ):
                targets = {edge.target_id for edge in event_edges}
                if targets & claimed_targets:
                    paused[branch_id] = "another selected branch claimed the same candidate"
                    continue
                selected_events[branch_id] = event_edges
                claimed_targets.update(targets)

            by_candidate_id = {candidate.detection_id: candidate for candidate in candidates}
            next_active: dict[str, _ForwardBranchState] = {}
            for branch_id in sorted(active):
                branch = active[branch_id]
                event_edges = selected_events.get(branch_id, ())
                if event_edges:
                    for edge in event_edges:
                        target = by_candidate_id.get(edge.target_id)
                        if target is None:
                            raise RuntimeError(
                                "Tracker returned an edge to an unknown sparse candidate"
                            )
                        kind = "gap" if target.frame - branch.last.frame > 1 else edge.kind
                        edges.append(replace(edge, kind=kind))
                        accepted[target.detection_id] = target
                        next_active[target.detection_id] = _ForwardBranchState(
                            branch.last,
                            target,
                            0,
                        )
                    continue

                if branch_id in paused:
                    warnings.append(
                        f"Paused branch {branch.last.detection_id} at t={frame}: "
                        f"{paused[branch_id]}"
                    )
                    continue
                missing = branch.missing + 1
                if allow_gap and missing <= max_missing:
                    next_active[branch_id] = _ForwardBranchState(
                        branch.previous,
                        branch.last,
                        missing,
                    )
                else:
                    warnings.append(
                        f"Stopped branch {branch.last.detection_id} at t={frame}: "
                        "no candidate passed the distance gate"
                    )

            if len(next_active) > max_active:
                keep = sorted(
                    next_active.values(),
                    key=lambda branch: (
                        -branch.last.quality,
                        branch.last.detection_id,
                    ),
                )[:max_active]
                dropped = len(next_active) - len(keep)
                next_active = {branch.last.detection_id: branch for branch in keep}
                warnings.append(
                    f"Paused {dropped} branch(es) at t={frame}: the sparse tracking "
                    f"limit is {max_active} active branches"
                )
            active = next_active
            if progress is not None:
                progress(
                    done,
                    total,
                    f"Tracking {len(active)} selected branch(es) at time {frame}",
                )
            if not active:
                break

        ordered_detections = tuple(
            sorted(accepted.values(), key=lambda item: (item.frame, item.detection_id))
        )
        ordered_edges = tuple(
            sorted(
                edges,
                key=lambda edge: (
                    accepted[edge.source_id].frame,
                    edge.source_id,
                    edge.target_id,
                    edge.kind,
                ),
            )
        )
        outcome = None
        if not warnings and active and all(
            branch.last.frame == scope.end_frame for branch in active.values()
        ):
            outcome = TrackingOutcome(
                code="completed",
                stop_frame=None,
                last_accepted_frame=scope.end_frame,
                predicted_position_um=None,
                search_radius_um=search_radius_um,
            )
        elif not warnings:
            warnings.append("Sparse forward tracking ended before every branch reached the end")

        return TrackingResult(
            request=request,
            detections=ordered_detections,
            edges=ordered_edges,
            existing_anchors={
                seed_detection.detection_id: scope.seed_anchors[0]
            },
            warnings=tuple(warnings),
            provenance=_provenance(
                self.registry,
                request,
                mode="selected_forward_follow_both",
            ),
            outcome=outcome,
        )


def _target_channel(request: TrackingRequest) -> int:
    """Translate TrackMate's 1-based TARGET_CHANNEL to ImageProvider's 0-based API."""
    return _target_channel_from_settings(request.detector.settings)


def _target_channel_from_settings(settings) -> int:
    """Translate detector settings' 1-based channel to the provider index."""

    channel_1based = int(settings.get("TARGET_CHANNEL", 1))
    if channel_1based < 1:
        raise ValueError("TARGET_CHANNEL must be a positive 1-based channel number")
    return channel_1based - 1


def _validate_source_bounds(
    image_provider: ImageProvider,
    request: TrackingRequest,
) -> None:
    if request.scope.end_frame > image_provider.num_timepoints:
        raise ValueError(
            f"Tracking ends at t={request.scope.end_frame}, but the image source "
            f"has {image_provider.num_timepoints} timepoint(s)"
        )
    channel = _target_channel(request)
    if channel >= image_provider.num_channels:
        raise ValueError(
            f"TARGET_CHANNEL {channel + 1} is unavailable; the image source has "
            f"{image_provider.num_channels} channel(s)"
        )


def _check_cancelled(cancelled: CancelCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise TrackingCancelled("Tracking analysis was cancelled")


def _get_alive_nucleus(
    nuclei_record: list[list[Nucleus]], time: int, index: int
) -> Nucleus:
    if time < 1 or time > len(nuclei_record):
        raise ValueError(f"Seed time {time} is outside the nuclei record")
    nuclei = nuclei_record[time - 1]
    if index < 1 or index > len(nuclei):
        raise ValueError(f"Seed index {index} is invalid at t={time}")
    nucleus = nuclei[index - 1]
    if not nucleus.is_alive:
        raise ValueError("The selected seed nucleus is not alive")
    if nucleus.successor1 > 0 or nucleus.successor2 > 0:
        raise ValueError(
            "The selected nucleus already has a successor; select the end of a track"
        )
    return nucleus


def _predict_position(
    previous: Detection | None, last: Detection, frame: int
) -> tuple[float, float, float]:
    if previous is None or last.frame == previous.frame:
        return last.x_um, last.y_um, last.z_um
    scale = (frame - last.frame) / (last.frame - previous.frame)
    return (
        last.x_um + (last.x_um - previous.x_um) * scale,
        last.y_um + (last.y_um - previous.y_um) * scale,
        last.z_um + (last.z_um - previous.z_um) * scale,
    )


def _crop_around(
    stack: np.ndarray,
    center_um: tuple[float, float, float],
    calibration: Calibration,
    radius_um: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    if stack.ndim != 3:
        raise ValueError(f"Detector input must be ZYX; received shape {stack.shape}")
    if radius_um <= 0:
        raise ValueError("Selected-forward ROI radius must be positive")
    x_px, y_px, z_plane = calibration.physical_to_pixel(*center_um)
    z_px = z_plane - calibration.plane_start
    rx = max(1, math.ceil(radius_um / calibration.xy_um))
    rz = max(1, math.ceil(radius_um / calibration.z_um))
    z0 = max(0, math.floor(z_px - rz))
    z1 = min(stack.shape[0], math.ceil(z_px + rz) + 1)
    y0 = max(0, math.floor(y_px - rx))
    y1 = min(stack.shape[1], math.ceil(y_px + rx) + 1)
    x0 = max(0, math.floor(x_px - rx))
    x1 = min(stack.shape[2], math.ceil(x_px + rx) + 1)
    return stack[z0:z1, y0:y1, x0:x1], (z0, y0, x0)


def _rank_candidates(
    source: Detection,
    candidates: Sequence[Detection],
    request: TrackingRequest,
) -> list[tuple[float, str]]:
    """Return admissible one-source LAP costs for ambiguity inspection."""
    delta = candidates[0].frame - source.frame if candidates else 1
    if delta <= 1:
        maximum = float(request.tracker.settings.get("LINKING_MAX_DISTANCE", 15.0))
    else:
        if not bool(request.tracker.settings.get("ALLOW_GAP_CLOSING", True)):
            return []
        maximum = float(
            request.tracker.settings.get("GAP_CLOSING_MAX_DISTANCE", 15.0)
        )
    ranked = []
    for candidate in candidates:
        distance2 = (
            (candidate.x_um - source.x_um) ** 2
            + (candidate.y_um - source.y_um) ** 2
            + (candidate.z_um - source.z_um) ** 2
        )
        if distance2 <= maximum * maximum:
            ranked.append((distance2, candidate.detection_id))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked


def _exclude_existing_detections(
    candidates: Sequence[Detection],
    nuclei_record: list[list[Nucleus]],
    frame: int,
    calibration: Calibration,
) -> tuple[list[Detection], set[str]]:
    """Protect curated records from duplicate selected-forward detections."""
    if frame < 1 or frame > len(nuclei_record):
        return list(candidates), set()
    existing = []
    for nucleus in nuclei_record[frame - 1]:
        if not nucleus.is_alive:
            continue
        x_um, y_um, z_um = calibration.pixel_to_physical(
            nucleus.x, nucleus.y, nucleus.z
        )
        existing.append(
            (x_um, y_um, z_um, max(calibration.xy_um, nucleus.size * calibration.xy_um / 2))
        )
    kept = []
    collisions: set[str] = set()
    for candidate in candidates:
        overlaps = any(
            (candidate.x_um - x_um) ** 2
            + (candidate.y_um - y_um) ** 2
            + (candidate.z_um - z_um) ** 2
            <= max(candidate.radius_um, radius_um) ** 2
            for x_um, y_um, z_um, radius_um in existing
        )
        if overlaps:
            collisions.add(candidate.detection_id)
        else:
            kept.append(candidate)
    return kept, collisions


def _costs_are_ambiguous(costs: Sequence[tuple[float, str]], ratio: float) -> bool:
    if len(costs) < 2:
        return False
    best = max(float(costs[0][0]), 1e-12)
    return float(costs[1][0]) / best < ratio


def _ordered_review_candidates(
    candidates: Sequence[Detection],
) -> tuple[Detection, ...]:
    """Canonicalize unaccepted detector observations for review/persistence."""
    return tuple(
        sorted(candidates, key=lambda candidate: (candidate.frame, candidate.detection_id))
    )


def _ranked_review_candidates(
    candidates: Sequence[Detection],
    ranked: Sequence[tuple[float, str]],
    limit: int = 2,
) -> tuple[Detection, ...]:
    """Return only the ranked observations that triggered a stop decision."""

    by_id = {candidate.detection_id: candidate for candidate in candidates}
    return tuple(by_id[detection_id] for _cost, detection_id in ranked[:limit])


def _stopped_outcome(
    code: str,
    stop_frame: int,
    last: Detection,
    predicted: tuple[float, float, float],
    search_radius_um: float,
    candidates: Sequence[Detection],
) -> TrackingOutcome:
    return TrackingOutcome(
        code=code,
        stop_frame=stop_frame,
        last_accepted_frame=last.frame,
        predicted_position_um=predicted,
        search_radius_um=search_radius_um,
        review_candidates=_ordered_review_candidates(candidates),
    )


def _looks_like_division(
    candidates: Sequence[Detection],
    ranked: Sequence[tuple[float, str]],
    predicted: tuple[float, float, float],
) -> bool:
    """Conservatively flag a close daughter pair for manual branch review."""
    if len(ranked) < 2:
        return False
    by_id = {candidate.detection_id: candidate for candidate in candidates}
    first = by_id[ranked[0][1]]
    second = by_id[ranked[1][1]]
    separation = math.dist(first.position_um, second.position_um)
    radius = max(first.radius_um, second.radius_um)
    midpoint = tuple(
        (a + b) / 2.0 for a, b in zip(first.position_um, second.position_um)
    )
    midpoint_error = math.dist(midpoint, predicted)
    quality_ratio = min(first.quality, second.quality) / max(
        first.quality,
        second.quality,
        1e-12,
    )
    return (
        separation <= 3.0 * radius
        and midpoint_error <= 1.5 * radius
        and quality_ratio >= 0.25
    )


def _provenance(
    registry: TrackingRegistry, request: TrackingRequest, *, mode: str
) -> dict[str, Any]:
    return {
        "schema": "acetree.tracking/v1",
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "detector": _component_provenance(
            registry, "detector", request.detector.plugin_id
        ),
        "tracker": _component_provenance(
            registry, "tracker", request.tracker.plugin_id
        ),
    }


def _component_provenance(
    registry: TrackingRegistry,
    expected_kind: str,
    plugin_id: str,
) -> dict[str, str]:
    descriptor = registry.get_descriptor(plugin_id)
    if descriptor.kind != expected_kind:
        raise ValueError(
            f"Plugin {plugin_id!r} is a {descriptor.kind}, not a {expected_kind}"
        )
    return {
        "plugin_id": descriptor.plugin_id,
        "api_version": descriptor.api_version,
        "implementation_version": descriptor.implementation_version,
    }
