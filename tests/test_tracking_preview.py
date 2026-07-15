"""Tests for the exact, non-mutating representation shown before acceptance."""

from __future__ import annotations

import pytest

from acetree_py.gui.tracking_preview import (
    expand_detector_preview,
    expand_tracking_preview,
)
from acetree_py.tracking.api import (
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)


def _gap_result() -> TrackingResult:
    request = TrackingRequest(
        detector=ComponentSpec("acetree.dog3d", {}),
        tracker=ComponentSpec("acetree.simple_lap", {}),
        scope=TrackingScope(
            "selected_forward",
            1,
            3,
            seed_anchors=((1, 1),),
            roi_radius_um=10.0,
        ),
    )
    seed = Detection("seed", 1, 2.0, 4.0, 6.0, 2.0, 5.0)
    target = Detection("target", 3, 6.0, 8.0, 10.0, 4.0, 9.0)
    return TrackingResult(
        request=request,
        detections=(seed, target),
        edges=(TrackEdge("seed", "target", 4.0, kind="gap"),),
        existing_anchors={"seed": (1, 1)},
    )


def _diagnostic_result() -> TrackingResult:
    request = TrackingRequest(
        detector=ComponentSpec("acetree.dog3d", {}),
        tracker=ComponentSpec("acetree.simple_lap", {}),
        scope=TrackingScope(
            "selected_forward",
            1,
            3,
            seed_anchors=((1, 1),),
            roi_radius_um=9.0,
        ),
    )
    seed = Detection("seed", 1, 2.0, 4.0, 6.0, 2.0, 5.0)
    accepted = Detection("accepted", 2, 3.0, 4.0, 6.0, 2.0, 7.0)
    candidates = (
        Detection("candidate-a", 3, 4.0, 3.0, 6.0, 1.5, 8.0),
        Detection("candidate-b", 3, 4.0, 5.0, 6.0, 1.5, 7.5),
    )
    return TrackingResult(
        request=request,
        detections=(seed, accepted),
        edges=(TrackEdge("seed", "accepted", 1.0),),
        existing_anchors={"seed": (1, 1)},
        warnings=("Stopped at t=3: two candidates had similar assignment costs",),
        outcome=TrackingOutcome(
            code="ambiguity",
            stop_frame=3,
            last_accepted_frame=2,
            predicted_position_um=(4.0, 4.0, 6.0),
            search_radius_um=9.0,
            review_candidates=candidates,
        ),
    )


def test_gap_preview_matches_materialized_interpolation():
    preview = expand_tracking_preview(_gap_result())

    assert [(spot.frame, spot.kind) for spot in preview.spots] == [
        (1, "seed"),
        (2, "interpolated"),
        (3, "detection"),
    ]
    gap = preview.spots[1]
    assert gap.x_um == pytest.approx(4.0)
    assert gap.y_um == pytest.approx(6.0)
    assert gap.z_um == pytest.approx(8.0)
    assert gap.radius_um == pytest.approx(3.0)
    assert gap.quality == pytest.approx(7.0)
    assert preview.proposed_count == 2
    assert preview.interpolated_count == 1
    assert [link.kind for link in preview.links] == ["gap", "gap"]


def test_expanding_preview_does_not_change_result():
    result = _gap_result()
    before = result.to_dict()

    first = expand_tracking_preview(result)
    second = expand_tracking_preview(result)

    assert first == second
    assert result.to_dict() == before


def test_preview_includes_review_only_candidates_and_search_region():
    result = _diagnostic_result()

    preview = expand_tracking_preview(result)

    assert preview.outcome_code == "ambiguity"
    assert preview.candidate_count == 2
    assert preview.proposed_count == 1
    candidates = preview.candidates
    assert [spot.detection_id for spot in candidates] == [
        "candidate-a",
        "candidate-b",
    ]
    assert all(
        spot.preview_id.startswith("__review_candidate__:")
        for spot in candidates
    )
    assert all(
        candidate.detection_id not in {detection.detection_id for detection in result.detections}
        for candidate in result.outcome.review_candidates
    )
    assert all(spot.kind != "candidate" for spot in preview.spots)
    assert preview.review_spots == (*preview.spots, *preview.candidates)
    assert set(preview.by_id) == {
        spot.preview_id for spot in preview.review_spots
    }
    assert preview.search_region is not None
    assert preview.search_region.frame == 3
    assert preview.search_region.outcome_code == "ambiguity"
    assert preview.search_region.radius_um == pytest.approx(9.0)
    assert (
        preview.search_region.x_um,
        preview.search_region.y_um,
        preview.search_region.z_um,
    ) == pytest.approx((4.0, 4.0, 6.0))
    # Diagnostic candidates explain the stop; no proposal link may target them.
    assert all("__review_candidate__" not in link.target_id for link in preview.links)


def test_completed_preview_has_no_diagnostic_search_region():
    result = _gap_result()
    completed = TrackingResult(
        request=result.request,
        detections=result.detections,
        edges=result.edges,
        existing_anchors=result.existing_anchors,
        outcome=TrackingOutcome(
            code="completed",
            stop_frame=None,
            last_accepted_frame=3,
            predicted_position_um=None,
            search_radius_um=10.0,
        ),
    )

    preview = expand_tracking_preview(completed)

    assert preview.outcome_code == "completed"
    assert preview.candidate_count == 0
    assert preview.search_region is None


def test_detector_preview_is_transient_and_has_no_committable_positions():
    detections = (
        Detection("b", 4, 8.0, 7.0, 6.0, 2.0, 5.0),
        Detection("a", 4, 3.0, 4.0, 5.0, 1.5, 9.0),
    )

    preview = expand_detector_preview(detections)

    assert preview.proposed_count == 0
    assert preview.links == ()
    assert preview.candidates == ()
    assert [spot.detection_id for spot in preview.spots] == ["b", "a"]
    assert all(spot.kind == "detector_test" for spot in preview.spots)
    assert all(
        spot.preview_id.startswith("__detector_test__:")
        for spot in preview.spots
    )
