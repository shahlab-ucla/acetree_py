"""Focused parity tests for StarryNite's pre-classifier tracking stages."""

from __future__ import annotations

import pytest

from acetree_py.tracking.starrynite.legacy_early import (
    LegacyEarlyTrackingError,
    LegacyEarlyTrackingParameters,
    run_legacy_early_tracking,
    summarize_legacy_early_stages,
)
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
)


def _nucleus(
    nucleus_id: str,
    frame: int,
    row: int,
    x: float,
    *,
    diameter: float = 10.0,
) -> LegacyNucleus:
    return LegacyNucleus(
        nucleus_id=nucleus_id,
        frame=frame,
        matlab_row=row,
        position_xyz=(x, 0.0, 0.0),
        diameter=diameter,
        total_gfp=100.0,
        avg_gfp=10.0,
        aspect_ratio=1.0,
        log_odds_sum=1.0,
        slice_count=2,
        xy_principal_variance=2.0,
        xy_secondary_variance=1.0,
    )


def _context(
    nuclei: list[LegacyNucleus],
    end_frame: int,
    *,
    candidate_cutoff: float = 2.0,
) -> LegacyTrackingContext:
    return LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        (),
        LegacyFeatureParameters(
            interval=1.0,
            candidate_cutoff=candidate_cutoff,
            temporal_cutoff=6,
            temporal_cutoff_start=2,
            small_cutoff=4.0,
            anisotropy_xyz=(1.0, 1.0, 1.0),
            end_frame=end_frame,
        ),
    )


def _parameters(
    end_frame: int,
    **overrides,
) -> LegacyEarlyTrackingParameters:
    values = {
        "start_frame": 1,
        "end_frame": end_frame,
        "safe_filter": True,
        "safe_factor": 2.0,
        "conflict_filter": True,
        "backward_nn_count": 2,
        "forward_nn_count": 4,
        "min_nondivision_score": -1.0,
        "nondivision_score_step": 1.0,
        "max_nondivision_score": -1.0,
        "min_division_score": -1.0,
        "division_score_step": 1.0,
        "max_division_score": -1.0,
        "nondivision_cost": "distance",
        "division_cost": "distance",
        "complete_divisions": False,
    }
    values.update(overrides)
    return LegacyEarlyTrackingParameters(**values)


def test_easy_links_use_mutual_row_order_and_strict_safety() -> None:
    context = _context(
        [
            _nucleus("a0", 1, 0, 0.0),
            _nucleus("a1", 1, 1, 100.0),
            _nucleus("b0", 2, 0, 1.0),
            _nucleus("b1", 2, 1, 101.0),
        ],
        2,
    )

    result = run_legacy_early_tracking(context, _parameters(2), None)

    assert result.context.successor_slots("a0") == ("b0", None)
    assert result.context.successor_slots("a1") == ("b1", None)
    assert result.stages[1].label == "easy_links"
    assert result.stages[1].link_count == 2


def test_candidate_forward_pruning_preserves_matlab_stale_backward_lists() -> None:
    context = _context(
        [
            _nucleus("s0", 1, 0, 0.0),
            _nucleus("s1", 1, 1, 100.0),
            _nucleus("t0", 2, 0, 1.0),
            _nucleus("t1", 2, 1, 2.0),
            _nucleus("t2", 2, 2, 3.0),
        ],
        2,
        candidate_cutoff=20.0,
    )
    parameters = _parameters(
        2,
        safe_factor=1000.0,
        conflict_filter=False,
        forward_nn_count=1,
    )

    result = run_legacy_early_tracking(context, parameters, None)

    assert result.candidates.forward("s0") == ("t0",)
    assert result.candidates.forward("s1") == ("t2",)
    assert result.candidates.backward("t0") == ("s0", "s1")
    assert result.candidates.backward("t1") == ("s0", "s1")
    assert result.candidates.backward("t2") == ("s0", "s1")
    candidate_stage = next(item for item in result.stages if item.label == "candidates")
    assert len(candidate_stage.forward_candidates) == 2
    assert len(candidate_stage.backward_candidates) == 6


def test_nondivision_threshold_sweep_links_at_first_admitting_threshold() -> None:
    context = _context(
        [
            _nucleus("s0", 1, 0, 0.0),
            _nucleus("s1", 1, 1, 10.0),
            _nucleus("t0", 2, 0, 4.0),
            _nucleus("t1", 2, 1, 6.0),
        ],
        2,
        candidate_cutoff=2.0,
    )
    parameters = _parameters(
        2,
        safe_factor=10.0,
        min_nondivision_score=0.3,
        nondivision_score_step=0.2,
        max_nondivision_score=0.5,
    )

    result = run_legacy_early_tracking(context, parameters, None)

    nondivision_stages = [
        item for item in result.stages if item.label == "nondivision"
    ]
    first, second = nondivision_stages
    assert first.threshold == pytest.approx(0.3)
    assert second.threshold == pytest.approx(0.5)
    assert first.link_count == 0
    assert second.link_count == 2
    assert result.context.successor_slots("s0") == ("t0", None)
    assert result.context.successor_slots("s1") == ("t1", None)


def test_bounded_stage_provenance_matches_full_parity_snapshots() -> None:
    context = _context(
        [
            _nucleus("s0", 1, 0, 0.0),
            _nucleus("s1", 1, 1, 10.0),
            _nucleus("t0", 2, 0, 4.0),
            _nucleus("t1", 2, 1, 6.0),
        ],
        2,
        candidate_cutoff=2.0,
    )
    parameters = _parameters(
        2,
        safe_factor=10.0,
        min_nondivision_score=0.3,
        nondivision_score_step=0.2,
        max_nondivision_score=0.5,
    )

    parity = run_legacy_early_tracking(context, parameters, None)
    bounded = run_legacy_early_tracking(
        context,
        parameters,
        None,
        capture_snapshots=False,
        capture_summaries=True,
    )

    assert bounded.stages == ()
    assert bounded.stage_summaries == summarize_legacy_early_stages(parity.stages)
    summaries = tuple(item.as_provenance() for item in bounded.stage_summaries)
    assert summaries[0] == {
        "name": "initialized",
        "threshold": None,
        "link_count": 0,
        "division_count": 0,
        "deleted_count": 0,
        "forward_candidate_count": 0,
        "backward_candidate_count": 0,
        "added_pointer_count": 0,
        "removed_pointer_count": 0,
        "newly_deleted_count": 0,
        "restored_row_count": 0,
    }
    assert summaries[-2]["name"] == "division"
    assert summaries[-1]["name"] == "geometry_final"
    assert any(item["added_pointer_count"] == 2 for item in summaries)


def test_division_sweep_places_new_candidate_in_first_successor_slot() -> None:
    context = _context(
        [
            _nucleus("s", 1, 0, 0.0),
            _nucleus("q", 1, 1, 100.0),
            _nucleus("d1", 2, 0, 1.0),
            _nucleus("d2", 2, 1, -1.0),
            _nucleus("q1", 2, 2, 101.0),
        ],
        2,
    )
    parameters = _parameters(
        2,
        conflict_filter=False,
        min_division_score=0.0,
        max_division_score=0.0,
    )

    result = run_legacy_early_tracking(context, parameters, None)

    assert result.context.successor_slots("s") == ("d2", "d1")
    assert result.context.predecessor("d1") == "s"
    assert result.context.predecessor("d2") == "s"
    division_stage = next(
        item
        for item in result.stages
        if item.label == "division" and item.threshold == 0.0
    )
    assert division_stage.division_count == 1


def test_polar_filter_deletes_bright_small_track_and_unlinks_it() -> None:
    nuclei: list[LegacyNucleus] = []
    disk_max: dict[str, float] = {}
    for frame in range(1, 12):
        small = f"small{frame}"
        large = f"large{frame}"
        nuclei.extend(
            (
                _nucleus(small, frame, 0, 0.0, diameter=1.0),
                _nucleus(large, frame, 1, 100.0, diameter=9.0),
            )
        )
        disk_max[small] = 100.0
        disk_max[large] = 0.0
    context = _context(nuclei, 11)
    parameters = _parameters(
        11,
        polar_body_filter=True,
        polar_end_frame=11,
        polar_threshold=10.0,
        polar_threshold_high=20.0,
        polar_threshold2_time=100,
        polar_threshold2=10.0,
        polar_threshold2_high=20.0,
    )

    result = run_legacy_early_tracking(
        context,
        parameters,
        None,
        disk_max_by_id=disk_max,
    )

    assert all(f"small{frame}" in result.context.deleted_ids for frame in range(1, 12))
    assert all(
        result.context.successor_slots(f"small{frame}") == (None, None)
        for frame in range(1, 11)
    )
    assert result.context.successor_slots("large1") == ("large2", None)


def test_optional_measurement_stages_fail_closed() -> None:
    context = _context(
        [
            _nucleus("a0", 1, 0, 0.0),
            _nucleus("a1", 1, 1, 10.0),
            _nucleus("b0", 2, 0, 1.0),
            _nucleus("b1", 2, 1, 11.0),
        ],
        2,
    )
    parameters = _parameters(
        2,
        hysteresis=True,
        hysteresis_intensity_high=5.0,
    )

    with pytest.raises(LegacyEarlyTrackingError, match="local_maximum_by_id"):
        run_legacy_early_tracking(context, parameters, None)


def test_model_loader_rejects_opaque_cost_identity_instead_of_guessing() -> None:
    model = {
        "trackingparameters": {
            "starttime": 1,
            "endtime": 2,
            "safefilter": True,
            "safefactor": 2,
            "conflictfilter": True,
            "nnnumber": 2,
            "forwardnnnumber": 4,
            "minnondivscore": 0.125,
            "nondivscorestep": 0.125,
            "maxnondivscore": 0.875,
            "mindivscore": -20,
            "divscorestep": 2,
            "maxdivscore": 5,
            "nonDivCostFunction": object(),
            "DivCostFunction": object(),
        }
    }

    with pytest.raises(TypeError, match="nondivision_cost"):
        LegacyEarlyTrackingParameters.from_model(model)
    loaded = LegacyEarlyTrackingParameters.from_model(
        model,
        nondivision_cost="distanceCostFunction",
        division_cost="divScoreModelCostFunction",
    )
    assert loaded.nondivision_cost == "distance"
    assert loaded.division_cost == "model"
