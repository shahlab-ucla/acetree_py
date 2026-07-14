"""Orchestration for global and selected-cell image-analysis runs.

The pipeline is intentionally side-effect free: it reads image stacks and an
optional nuclei record, then returns a :class:`TrackingResult` proposal.  The
editing layer owns preview/acceptance and is the only layer allowed to mutate
AceTree nuclei.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.nucleus import Nucleus
from .api import (
    Calibration,
    Detection,
    TrackEdge,
    TrackingRequest,
    TrackingResult,
)
from .registry import TrackingRegistry, get_default_registry

if TYPE_CHECKING:
    from ..io.image_provider import ImageProvider


class TrackingCancelled(RuntimeError):
    """Raised when a caller cancels an analysis run."""


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


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
        edges = tracker.track(tuple(detections), request.tracker.settings)
        if progress is not None:
            progress(total, total, "Linking detections")
        return TrackingResult(
            request=request,
            detections=tuple(detections),
            edges=tuple(edges),
            existing_anchors={},
            warnings=(),
            provenance=_provenance(self.registry, request, mode="global"),
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
        total = request.scope.end_frame - seed_time

        for done, frame in enumerate(range(seed_time + 1, request.scope.end_frame + 1), start=1):
            _check_cancelled(cancelled)
            stack = np.asarray(image_provider.get_stack(frame, channel))
            predicted = _predict_position(previous, last, frame)
            crop, offset_zyx = _crop_around(
                stack,
                predicted,
                calibration,
                request.scope.roi_radius_um
                if request.scope.roi_radius_um is not None else 12.0,
            )
            candidates = list(
                detector.detect(
                    crop,
                    frame,
                    calibration,
                    request.detector.settings,
                    offset_zyx=offset_zyx,
                )
            )
            ranked_before_conflicts = _rank_candidates(last, candidates, request)
            candidates, collided_ids = _exclude_existing_detections(
                candidates,
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
                break

            ranked = _rank_candidates(last, candidates, request)
            if len(ranked) > 1 and _costs_are_ambiguous(ranked, ambiguity_ratio):
                warnings.append(
                    f"Stopped at t={frame}: two candidates had similar assignment costs"
                )
                break
            if _looks_like_division(candidates, ranked, predicted):
                warnings.append(
                    f"Stopped at t={frame}: two candidates form a probable division; "
                    "Simple LAP does not create daughter branches"
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

            if not outgoing:
                missing += 1
                if not allow_gap or missing > max_missing:
                    warnings.append(
                        f"Stopped at t={frame}: no unique candidate passed the distance gate"
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

        return TrackingResult(
            request=request,
            detections=tuple(accepted),
            edges=tuple(edges),
            existing_anchors={seed_id: (seed_time, seed_index)},
            warnings=tuple(warnings),
            provenance=_provenance(self.registry, request, mode="selected_forward"),
        )


def _target_channel(request: TrackingRequest) -> int:
    """Translate TrackMate's 1-based TARGET_CHANNEL to ImageProvider's 0-based API."""
    channel_1based = int(request.detector.settings.get("TARGET_CHANNEL", 1))
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
