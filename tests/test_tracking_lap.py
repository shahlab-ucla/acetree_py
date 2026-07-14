"""Focused tests for the Simple LAP temporal linker."""

from __future__ import annotations

import pytest

from acetree_py.tracking import Detection, SimpleLAPTracker


def _detection(
    detection_id: str,
    frame: int,
    x: float,
    *,
    quality: float = 1.0,
    radius: float = 2.0,
) -> Detection:
    return Detection(detection_id, frame, x, 0.0, 0.0, radius, quality)


def test_links_adjacent_frames_by_global_optimum_not_greedy_order():
    detections = (
        _detection("s0", 1, 0),
        _detection("s4", 1, 4),
        _detection("t3", 2, 3),
        _detection("t5", 2, 5),
    )

    edges = SimpleLAPTracker().track(
        detections,
        {"LINKING_MAX_DISTANCE": 10, "ALLOW_GAP_CLOSING": False},
    )

    assert {(edge.source_id, edge.target_id) for edge in edges} == {
        ("s0", "t3"),
        ("s4", "t5"),
    }


def test_distance_gate_leaves_detections_unlinked():
    edges = SimpleLAPTracker().track(
        (_detection("a", 1, 0), _detection("b", 2, 100)),
        {"LINKING_MAX_DISTANCE": 5, "ALLOW_GAP_CLOSING": False},
    )
    assert edges == ()


def test_gap_closing_uses_trackmate_frame_delta_semantics():
    detections = (_detection("a", 1, 0), _detection("b", 3, 2))
    tracker = SimpleLAPTracker()

    edges = tracker.track(
        detections,
        {
            "LINKING_MAX_DISTANCE": 1,
            "GAP_CLOSING_MAX_DISTANCE": 3,
            "MAX_FRAME_GAP": 2,
        },
    )

    assert len(edges) == 1
    assert edges[0].kind == "gap"
    assert edges[0].features["FRAME_DELTA"] == 2
    assert edges[0].features["MISSED_FRAMES"] == 1
    assert tracker.track(
        detections,
        {
            "LINKING_MAX_DISTANCE": 1,
            "GAP_CLOSING_MAX_DISTANCE": 3,
            "MAX_FRAME_GAP": 1,
        },
    ) == ()


def test_gap_closing_can_be_disabled_or_distance_gated():
    detections = (_detection("a", 1, 0), _detection("b", 3, 4))
    tracker = SimpleLAPTracker()
    assert tracker.track(
        detections,
        {"LINKING_MAX_DISTANCE": 1, "ALLOW_GAP_CLOSING": False},
    ) == ()
    assert tracker.track(
        detections,
        {
            "LINKING_MAX_DISTANCE": 1,
            "GAP_CLOSING_MAX_DISTANCE": 3,
            "MAX_FRAME_GAP": 2,
        },
    ) == ()


def test_simple_lap_never_splits_or_merges():
    tracker = SimpleLAPTracker()
    split_edges = tracker.track(
        (
            _detection("parent", 1, 0),
            _detection("left", 2, -1),
            _detection("right", 2, 1),
        ),
        {"LINKING_MAX_DISTANCE": 3, "ALLOW_GAP_CLOSING": False},
    )
    merge_edges = tracker.track(
        (
            _detection("left", 1, -1),
            _detection("right", 1, 1),
            _detection("child", 2, 0),
        ),
        {"LINKING_MAX_DISTANCE": 3, "ALLOW_GAP_CLOSING": False},
    )
    assert len(split_edges) == 1
    assert len(merge_edges) == 1
    with pytest.raises(ValueError, match="does not support track splitting"):
        tracker.track((), {"ALLOW_TRACK_SPLITTING": True})
    with pytest.raises(ValueError, match="does not support track merging"):
        tracker.track((), {"ALLOW_TRACK_MERGING": True})


def test_feature_penalty_can_change_assignment():
    detections = (
        _detection("small", 1, 0, radius=1),
        _detection("large", 1, 5, radius=5),
        _detection("near-large", 2, 1, radius=5),
        _detection("near-small", 2, 4, radius=1),
    )
    tracker = SimpleLAPTracker()
    no_penalty = tracker.track(
        detections,
        {"LINKING_MAX_DISTANCE": 10, "ALLOW_GAP_CLOSING": False},
    )
    with_penalty = tracker.track(
        detections,
        {
            "LINKING_MAX_DISTANCE": 10,
            "LINKING_FEATURE_PENALTIES": {"RADIUS": 2.0},
            "ALLOW_GAP_CLOSING": False,
        },
    )
    assert {(e.source_id, e.target_id) for e in no_penalty} == {
        ("small", "near-large"),
        ("large", "near-small"),
    }
    assert {(e.source_id, e.target_id) for e in with_penalty} == {
        ("small", "near-small"),
        ("large", "near-large"),
    }


def test_output_is_deterministic_for_reordered_input():
    detections = [
        _detection("a", 1, 0),
        _detection("b", 1, 10),
        _detection("c", 2, 1),
        _detection("d", 2, 9),
    ]
    settings = {"LINKING_MAX_DISTANCE": 5, "ALLOW_GAP_CLOSING": False}
    tracker = SimpleLAPTracker()
    first = tracker.track(detections, settings)
    second = tracker.track(tuple(reversed(detections)), settings)
    assert first == second


def test_duplicate_ids_and_bad_settings_rejected():
    tracker = SimpleLAPTracker()
    duplicate = (_detection("same", 1, 0), _detection("same", 2, 1))
    with pytest.raises(ValueError, match="unique"):
        tracker.track(duplicate, {})
    with pytest.raises(ValueError, match="Unsupported Simple LAP"):
        tracker.track((), {"UNKNOWN": True})
