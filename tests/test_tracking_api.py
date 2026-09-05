"""Focused contracts for structured tracking outcomes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
    WholeMoviePreflightContext,
)


def _candidate(frame: int = 4) -> Detection:
    return Detection(
        detection_id="candidate-a",
        frame=frame,
        x_um=10.0,
        y_um=12.0,
        z_um=3.0,
        radius_um=2.0,
        quality=8.0,
        features={"reason": "review-only"},
    )


def test_tracking_scope_branch_policy_defaults_and_round_trips() -> None:
    legacy_payload = {
        "kind": "selected_forward",
        "start_frame": 2,
        "end_frame": 5,
        "seed_anchors": [[2, 1]],
        "roi_radius_um": 8.0,
        "ambiguity_ratio": 1.2,
    }

    legacy_scope = TrackingScope.from_dict(legacy_payload)
    assert legacy_scope.branch_policy == "stop"

    follow_both = TrackingScope(
        "selected_forward",
        2,
        5,
        seed_anchors=((2, 1),),
        roi_radius_um=8.0,
        branch_policy="follow_both",
    )
    assert follow_both.to_dict()["branch_policy"] == "follow_both"
    assert TrackingScope.from_dict(follow_both.to_dict()) == follow_both


@pytest.mark.parametrize("branch_policy", ["stop", "follow_best", "follow_both"])
def test_tracking_scope_accepts_supported_branch_policies(branch_policy: str) -> None:
    scope = TrackingScope("global", 1, 2, branch_policy=branch_policy)
    assert scope.branch_policy == branch_policy


def test_tracking_scope_rejects_unknown_branch_policy() -> None:
    with pytest.raises(ValueError, match="branch policy"):
        TrackingScope("global", 1, 2, branch_policy="guess")


def test_whole_movie_preflight_context_is_immutable_and_identifies_full_scope() -> None:
    detector = ComponentSpec("example.detector", {"TARGET_CHANNEL": 1})
    context = WholeMoviePreflightContext(
        detector_spec=detector,
        calibration=Calibration(0.25, 1.0),
        scope=TrackingScope("global", 1, 4),
        source_num_timepoints=4,
        source_num_channels=2,
        target_channel=0,
    )

    assert context.covers_complete_global_movie
    with pytest.raises(FrozenInstanceError):
        context.source_num_timepoints = 3  # type: ignore[misc]
    partial = WholeMoviePreflightContext(
        detector_spec=detector,
        calibration=context.calibration,
        scope=TrackingScope("global", 1, 3),
        source_num_timepoints=4,
        source_num_channels=2,
        target_channel=0,
    )
    assert not partial.covers_complete_global_movie
    with pytest.raises(ValueError, match="Target channel"):
        WholeMoviePreflightContext(
            detector_spec=detector,
            calibration=context.calibration,
            scope=context.scope,
            source_num_timepoints=4,
            source_num_channels=2,
            target_channel=2,
        )


def test_tracking_outcome_is_immutable_and_round_trips() -> None:
    outcome = TrackingOutcome(
        code="ambiguity",
        stop_frame=4,
        last_accepted_frame=3,
        predicted_position_um=(9.5, 12.0, 3.0),
        search_radius_um=7.5,
        review_candidates=(_candidate(),),
    )

    assert TrackingOutcome.from_dict(outcome.to_dict()) == outcome
    assert outcome.stopped_early
    assert outcome.frame == outcome.stop_frame
    assert outcome.candidates == outcome.review_candidates
    with pytest.raises(FrozenInstanceError):
        outcome.code = "lost"  # type: ignore[misc]


def test_completed_outcome_has_no_stopped_frame_observations() -> None:
    outcome = TrackingOutcome(
        code="completed",
        stop_frame=None,
        last_accepted_frame=8,
        predicted_position_um=None,
        search_radius_um=12.0,
    )

    assert not outcome.stopped_early
    assert outcome.review_candidates == ()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"code": "unknown"}, "outcome code"),
        ({"stop_frame": None}, "requires a stop_frame"),
        ({"predicted_position_um": None}, "requires predicted_position_um"),
        ({"search_radius_um": 0.0}, "must be positive"),
        ({"review_candidates": (_candidate(frame=5),)}, "stop frame"),
    ],
)
def test_stopped_outcome_rejects_inconsistent_diagnostics(changes, message) -> None:
    values = {
        "code": "lost",
        "stop_frame": 4,
        "last_accepted_frame": 3,
        "predicted_position_um": (9.5, 12.0, 3.0),
        "search_radius_um": 7.5,
        "review_candidates": (),
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError), match=message):
        TrackingOutcome(**values)


def _request(kind: str = "selected_forward") -> TrackingRequest:
    anchors = ((1, 1),) if kind == "selected_forward" else ()
    return TrackingRequest(
        detector=ComponentSpec("detector"),
        tracker=ComponentSpec("tracker"),
        scope=TrackingScope(kind, 1, 4, seed_anchors=anchors),
    )


def _accepted(frame: int) -> Detection:
    return Detection(
        detection_id=f"accepted-{frame}",
        frame=frame,
        x_um=float(frame),
        y_um=2.0,
        z_um=1.0,
        radius_um=2.0,
        quality=5.0,
    )


def test_result_rejects_outcome_for_global_scope() -> None:
    outcome = TrackingOutcome(
        code="completed",
        stop_frame=None,
        last_accepted_frame=4,
        predicted_position_um=None,
        search_radius_um=5.0,
    )

    with pytest.raises(ValueError, match="selected-forward"):
        TrackingResult(_request("global"), (_accepted(4),), (), outcome=outcome)


@pytest.mark.parametrize(
    ("outcome", "final_frame", "message"),
    [
        (
            TrackingOutcome("completed", None, 3, None, 5.0),
            4,
            "final proposal detection",
        ),
        (
            TrackingOutcome("completed", None, 2, None, 5.0),
            2,
            "scope end frame",
        ),
        (
            TrackingOutcome("lost", 5, 4, (4.0, 2.0, 1.0), 5.0),
            4,
            "stop_frame",
        ),
    ],
)
def test_result_rejects_outcome_inconsistent_with_scope(
    outcome, final_frame, message
) -> None:
    detections = (_accepted(1), _accepted(final_frame))

    with pytest.raises(ValueError, match=message):
        TrackingResult(_request(), detections, (), outcome=outcome)


def test_proposal_nested_inputs_are_immutable_and_export_as_detached_json():
    nested = {"weights": {"values": [0.1, 0.2]}}
    request = TrackingRequest(
        ComponentSpec("test.detector", nested),
        ComponentSpec("test.tracker", nested),
        TrackingScope("global", 1, 2),
    )
    first = Detection("first", 1, 1, 2, 3, 1, 1, nested)
    second = Detection("second", 2, 1, 2, 3, 1, 1)
    edge = TrackEdge("first", "second", 1, features=nested)
    result = TrackingResult(request, (first, second), (edge,), provenance=nested)
    nested["weights"]["values"].append(99)
    nested["weights"]["extra"] = True

    for value in (request.detector.settings, request.tracker.settings,
                  first.features, edge.features, result.provenance):
        assert value["weights"]["values"] == (0.1, 0.2)
        assert "extra" not in value["weights"]
        with pytest.raises(TypeError):
            value["weights"]["extra"] = True
        with pytest.raises(TypeError):
            value["weights"]["values"][0] = 99

    payload = result.to_dict()
    assert TrackingResult.from_dict(json.loads(json.dumps(payload))) == result
    payload["request"]["detector"]["settings"]["weights"]["values"].append(99)
    payload["detections"][0]["features"]["weights"]["values"].clear()
    payload["edges"][0]["features"]["weights"].clear()
    payload["provenance"]["weights"]["values"].clear()
    assert result.to_dict()["provenance"] == {"weights": {"values": [0.1, 0.2]}}
    assert request.detector.settings["weights"]["values"] == (0.1, 0.2)
    assert first.features["weights"]["values"] == edge.features["weights"]["values"]
