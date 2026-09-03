"""Pure transformations of immutable tracking proposals.

These helpers deliberately operate below the GUI layer.  A review window can
therefore derive the exact proposal it will commit without mutating either the
original analysis result or the AceTree lineage document.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .api import TrackingOutcome, TrackingResult


_WARNING_FRAME = re.compile(r"\bt\s*=\s*(\d+)\b", re.IGNORECASE)
_SUPERSEDED_FRAMELESS_WARNINGS = (
    "ended before every branch reached the end",
    "ended before the requested end",
    "did not reach the requested end",
)


def trim_selected_forward_result(
    result: TrackingResult,
    end_frame: int,
) -> TrackingResult:
    """Return the reviewed portion of a selected-forward result through a frame.

    ``end_frame`` is inclusive and must coincide with a retained proposal
    detection (the seed is also a detection).  Requiring a real endpoint keeps
    the operation honest for gap edges: this helper never invents a detection
    at an interpolated preview position.

    The returned request is narrowed to the accepted endpoint and receives a
    valid ``completed`` outcome.  Stopped-frame candidates and other diagnostics
    beyond the endpoint are recorded in provenance rather than presented as if
    they still applied to the shorter proposal.  The source result is immutable
    and is never changed.
    """

    if result.request.scope.kind != "selected_forward":
        raise ValueError("Only selected-forward tracking results can be trimmed")
    if isinstance(end_frame, bool) or not isinstance(end_frame, int):
        raise TypeError("end_frame must be an integer")

    scope = result.request.scope
    if not scope.start_frame <= end_frame <= scope.end_frame:
        raise ValueError(
            "end_frame must be inside the selected-forward tracking scope"
        )

    retained_detections = tuple(
        detection for detection in result.detections if detection.frame <= end_frame
    )
    if not retained_detections or not any(
        detection.frame == end_frame for detection in retained_detections
    ):
        raise ValueError(
            "end_frame must coincide with a retained proposal detection; "
            "interpolated gap frames cannot be trim endpoints"
        )

    retained_ids = {
        detection.detection_id for detection in retained_detections
    }
    missing_anchors = set(result.existing_anchors) - retained_ids
    if missing_anchors:
        raise ValueError("Trimming would remove an existing seed anchor")
    if any(anchor[0] > end_frame for anchor in result.request.scope.seed_anchors):
        raise ValueError("Trimming would move the request before a seed anchor")

    retained_edges = tuple(
        edge
        for edge in result.edges
        if edge.source_id in retained_ids and edge.target_id in retained_ids
    )
    # A completed proposal already ending at this detection is an exact no-op.
    # Returning it directly avoids inventing review provenance for an action
    # that did not actually shorten or resolve anything.
    if (
        end_frame == scope.end_frame
        and len(retained_detections) == len(result.detections)
        and len(retained_edges) == len(result.edges)
        and result.outcome is not None
        and result.outcome.code == "completed"
    ):
        return result

    retained_warnings, discarded_warnings = _partition_trim_warnings(
        result.warnings,
        end_frame,
    )

    narrowed_scope = replace(scope, end_frame=end_frame)
    narrowed_request = replace(result.request, scope=narrowed_scope)
    search_radius_um = (
        result.outcome.search_radius_um
        if result.outcome is not None
        else (scope.roi_radius_um if scope.roi_radius_um is not None else 12.0)
    )
    completed_outcome = TrackingOutcome(
        code="completed",
        stop_frame=None,
        last_accepted_frame=end_frame,
        predicted_position_um=None,
        search_radius_um=search_radius_um,
    )

    provenance = dict(result.provenance)
    provenance["review_trim"] = {
        "inclusive_end_frame": end_frame,
        "original_end_frame": scope.end_frame,
        "original_outcome": (
            None if result.outcome is None else result.outcome.to_dict()
        ),
        "discarded_detection_count": (
            len(result.detections) - len(retained_detections)
        ),
        "discarded_edge_count": len(result.edges) - len(retained_edges),
        "discarded_warnings": list(discarded_warnings),
    }

    return TrackingResult(
        request=narrowed_request,
        detections=retained_detections,
        edges=retained_edges,
        existing_anchors=result.existing_anchors,
        warnings=retained_warnings,
        provenance=provenance,
        outcome=completed_outcome,
    )


def _partition_trim_warnings(
    warnings: tuple[str, ...],
    end_frame: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split warnings into those still applicable and downstream diagnostics."""

    retained: list[str] = []
    discarded: list[str] = []
    for warning in warnings:
        frames = tuple(int(value) for value in _WARNING_FRAME.findall(warning))
        lower_warning = warning.casefold()
        still_applies = (
            all(frame <= end_frame for frame in frames)
            if frames
            else not any(
                phrase in lower_warning
                for phrase in _SUPERSEDED_FRAMELESS_WARNINGS
            )
        )
        (retained if still_applies else discarded).append(warning)
    return tuple(retained), tuple(discarded)
