"""Tests for deriving a reviewed prefix from a selected-forward proposal."""

from __future__ import annotations

import json

import pytest

from acetree_py.tracking import trim_selected_forward_result
from acetree_py.tracking.api import (
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)


def _detection(detection_id: str, frame: int) -> Detection:
    return Detection(
        detection_id,
        frame,
        float(frame),
        2.0,
        3.0,
        2.5,
        1.0,
    )


def _request(*, end_frame: int = 5, kind: str = "selected_forward") -> TrackingRequest:
    return TrackingRequest(
        detector=ComponentSpec("example.detector", {"THRESHOLD": 2.0}),
        tracker=ComponentSpec("example.tracker", {"MAX_LINKING_DISTANCE": 9.0}),
        scope=TrackingScope(
            kind,
            1,
            end_frame,
            seed_anchors=((1, 7),) if kind == "selected_forward" else (),
            roi_radius_um=8.0,
        ),
    )


def _completed_result() -> TrackingResult:
    detections = tuple(_detection(f"d{frame}", frame) for frame in range(1, 6))
    edges = tuple(
        TrackEdge(f"d{frame}", f"d{frame + 1}", float(frame))
        for frame in range(1, 5)
    )
    return TrackingResult(
        request=_request(),
        detections=detections,
        edges=edges,
        existing_anchors={"d1": (1, 7)},
        warnings=(
            "Paused branch at t=2: local note",
            "Stopped at t=5: downstream diagnostic",
            "Detector used a calibrated threshold",
            "Sparse forward tracking ended before every branch reached the end",
        ),
        provenance={"run_id": "original"},
        outcome=TrackingOutcome(
            "completed",
            None,
            5,
            None,
            8.0,
        ),
    )


def test_trim_narrows_result_and_preserves_anchor_without_mutating_source():
    source = _completed_result()
    source_before = source.to_dict()

    trimmed = trim_selected_forward_result(source, 3)

    assert source.to_dict() == source_before
    assert [item.detection_id for item in trimmed.detections] == ["d1", "d2", "d3"]
    assert [(edge.source_id, edge.target_id) for edge in trimmed.edges] == [
        ("d1", "d2"),
        ("d2", "d3"),
    ]
    assert dict(trimmed.existing_anchors) == {"d1": (1, 7)}
    assert trimmed.request.scope.end_frame == 3
    assert trimmed.request.scope.seed_anchors == ((1, 7),)
    assert trimmed.outcome is not None
    assert trimmed.outcome.code == "completed"
    assert trimmed.outcome.last_accepted_frame == 3
    assert trimmed.outcome.stop_frame is None
    assert trimmed.outcome.review_candidates == ()
    assert trimmed.warnings == (
        "Paused branch at t=2: local note",
        "Detector used a calibrated threshold",
    )
    assert trimmed.provenance["run_id"] == "original"
    assert trimmed.to_dict()["provenance"]["review_trim"] == {
        "inclusive_end_frame": 3,
        "original_end_frame": 5,
        "original_outcome": source.outcome.to_dict(),
        "discarded_detection_count": 2,
        "discarded_edge_count": 2,
        "discarded_warnings": [
            "Stopped at t=5: downstream diagnostic",
            "Sparse forward tracking ended before every branch reached the end",
        ],
    }
    restored = TrackingResult.from_dict(json.loads(json.dumps(trimmed.to_dict())))
    assert restored.to_dict() == trimmed.to_dict()


def test_trim_replaces_stopped_diagnostics_with_completed_prefix():
    seed = _detection("seed", 1)
    accepted = _detection("accepted", 2)
    candidate = _detection("candidate", 3)
    source = TrackingResult(
        request=_request(end_frame=5),
        detections=(seed, accepted),
        edges=(TrackEdge("seed", "accepted", 1.0),),
        existing_anchors={"seed": (1, 7)},
        warnings=("Stopped at t=3: two candidates were ambiguous",),
        outcome=TrackingOutcome(
            "ambiguity",
            3,
            2,
            (3.0, 2.0, 3.0),
            8.0,
            (candidate,),
        ),
    )

    trimmed = trim_selected_forward_result(source, 2)

    assert trimmed.detections == source.detections
    assert trimmed.edges == source.edges
    assert trimmed.warnings == ()
    assert trimmed.request.scope.end_frame == 2
    assert trimmed.outcome is not None
    assert trimmed.outcome.to_dict() == {
        "code": "completed",
        "stop_frame": None,
        "last_accepted_frame": 2,
        "predicted_position_um": None,
        "search_radius_um": 8.0,
        "review_candidates": [],
    }
    original = trimmed.provenance["review_trim"]["original_outcome"]
    assert original["code"] == "ambiguity"
    assert original["review_candidates"][0]["detection_id"] == "candidate"


def test_trim_keeps_a_complete_division_event_at_the_endpoint():
    seed = _detection("seed", 1)
    daughter_a = _detection("daughter-a", 2)
    daughter_b = _detection("daughter-b", 2)
    later_a = _detection("later-a", 3)
    source = TrackingResult(
        request=_request(end_frame=3),
        detections=(seed, daughter_a, daughter_b, later_a),
        edges=(
            TrackEdge("seed", "daughter-a", 1.0, "split"),
            TrackEdge("seed", "daughter-b", 1.1, "split"),
            TrackEdge("daughter-a", "later-a", 1.0),
        ),
        existing_anchors={"seed": (1, 7)},
        outcome=TrackingOutcome("completed", None, 3, None, 8.0),
    )

    trimmed = trim_selected_forward_result(source, 2)

    assert [item.detection_id for item in trimmed.detections] == [
        "seed",
        "daughter-a",
        "daughter-b",
    ]
    assert [(edge.source_id, edge.target_id, edge.kind) for edge in trimmed.edges] == [
        ("seed", "daughter-a", "split"),
        ("seed", "daughter-b", "split"),
    ]


def test_trim_rejects_unsupported_or_inexact_endpoints():
    source = _completed_result()
    global_result = TrackingResult(
        request=_request(end_frame=5, kind="global"),
        detections=source.detections,
        edges=source.edges,
    )

    with pytest.raises(ValueError, match="selected-forward"):
        trim_selected_forward_result(global_result, 3)
    with pytest.raises(TypeError, match="integer"):
        trim_selected_forward_result(source, True)
    with pytest.raises(ValueError, match="inside"):
        trim_selected_forward_result(source, 6)

    gap_result = TrackingResult(
        request=_request(end_frame=3),
        detections=(_detection("seed", 1), _detection("target", 3)),
        edges=(TrackEdge("seed", "target", 1.0, "gap"),),
        existing_anchors={"seed": (1, 7)},
        outcome=TrackingOutcome("completed", None, 3, None, 8.0),
    )
    with pytest.raises(ValueError, match="interpolated gap"):
        trim_selected_forward_result(gap_result, 2)


def test_trim_to_existing_completed_end_is_an_identity_operation():
    source = _completed_result()

    assert trim_selected_forward_result(source, 5) is source
