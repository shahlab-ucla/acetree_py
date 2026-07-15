"""Focused contracts for structured tracking outcomes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from acetree_py.tracking.api import (
    ComponentSpec,
    Detection,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
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
