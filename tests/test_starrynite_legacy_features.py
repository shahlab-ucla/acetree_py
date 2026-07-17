"""Focused conformance tests for exact legacy bifurcation features."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

import acetree_py.tracking.starrynite.legacy_features as legacy_feature_module
from acetree_py.tracking.api import TrackEdge
from acetree_py.tracking.starrynite.bifurcation import SingleModelLineageRequest
from acetree_py.tracking.starrynite.classifier import (
    SingleModelFeatureLayout,
    assemble_single_model_features,
)
from acetree_py.tracking.starrynite.legacy_features import (
    BACKWARD_FEATURE_NAMES,
    DAUGHTER_FEATURE_NAMES,
    FORWARD_FEATURE_NAMES,
    LegacyFeatureExtractionError,
    LegacyTrackingStatistics,
    calculate_legacy_division_pair,
    calculate_legacy_division_triple,
    calculate_legacy_nondivision_pair,
    calculate_legacy_nondivision_scores,
    extract_legacy_bifurcation_features,
    extract_retained_legacy_bifurcations,
)
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
    legacy_single_round,
)
from acetree_py.tracking.starrynite.repair_candidates import (
    BackwardRepairCandidates,
)


def _nucleus(
    nucleus_id: str,
    frame: int,
    row: int,
    xyz: tuple[float, float, float],
    *,
    total: float = 100.0,
    average: float = 10.0,
    principal: float = 8.0,
    secondary: float = 2.0,
) -> LegacyNucleus:
    return LegacyNucleus(
        nucleus_id=nucleus_id,
        frame=frame,
        matlab_row=row,
        position_xyz=xyz,
        diameter=2.0,
        total_gfp=total,
        avg_gfp=average,
        aspect_ratio=1.5,
        log_odds_sum=2.0,
        slice_count=2,
        xy_principal_variance=principal,
        xy_secondary_variance=secondary,
    )


def _edge(source: str, target: str, *, split: bool = False) -> TrackEdge:
    return TrackEdge(
        source_id=source,
        target_id=target,
        cost=0.0,
        kind="split" if split else "link",
    )


def _statistics() -> LegacyTrackingStatistics:
    def identity(size: int) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(1.0 if row == column else 0.0 for column in range(size))
            for row in range(size)
        )

    return LegacyTrackingStatistics(
        division_pair_mean=(0.0, 0.0),
        division_pair_covariance=identity(2),
        division_triple_mean=(0.0,) * 10,
        division_triple_covariance=identity(10),
        nondivision_mean=(0.0,) * 4,
        nondivision_covariance=identity(4),
    )


def _parameters(
    *,
    interval: float = 1.0,
    candidate_cutoff: float = 2.0,
    end_frame: int = 6,
) -> LegacyFeatureParameters:
    return LegacyFeatureParameters(
        interval=interval,
        candidate_cutoff=candidate_cutoff,
        temporal_cutoff=6,
        temporal_cutoff_start=2,
        small_cutoff=4.0,
        anisotropy_xyz=(1.0, 1.0, 2.0),
        end_frame=end_frame,
    )


def _base_context(*, add_backward_candidate: bool = False) -> LegacyTrackingContext:
    nuclei = [
        _nucleus("a1", 1, 0, (0.0, 0.0, 2.0)),
        _nucleus("c1", 1, 1, (10.0, 0.0, 2.0)),
        _nucleus("p", 2, 0, (0.0, 0.0, 2.0), total=100.0, average=10.0),
        _nucleus("c2", 2, 1, (10.0, 0.0, 2.0)),
    ]
    if add_backward_candidate:
        nuclei.append(_nucleus("back", 1, 2, (1.0, 0.0, 3.0)))

    for frame in range(3, 7):
        suffix = "" if frame == 3 else str(frame)
        nuclei.extend(
            (
                _nucleus(
                    f"d1{suffix}",
                    frame,
                    0,
                    (1.0, 0.0, 3.0),
                    total=60.0,
                    average=6.0,
                    principal=6.0,
                ),
                _nucleus(
                    f"d2{suffix}",
                    frame,
                    1,
                    (-1.0, 0.0, 3.0),
                    total=50.0,
                    average=5.0,
                    principal=4.0,
                ),
                _nucleus(f"c{frame}", frame, 2, (10.0, 0.0, 3.0)),
            )
        )

    edges = [
        _edge("a1", "p"),
        _edge("c1", "c2"),
        _edge("p", "d1", split=True),
        _edge("p", "d2", split=True),
        _edge("c2", "c3"),
    ]
    for frame in range(3, 6):
        current = "" if frame == 3 else str(frame)
        following = str(frame + 1)
        edges.extend(
            (
                _edge(f"d1{current}", f"d1{following}"),
                _edge(f"d2{current}", f"d2{following}"),
                _edge(f"c{frame}", f"c{frame + 1}"),
            )
        )
    return LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        edges,
        _parameters(candidate_cutoff=0.1 if add_backward_candidate else 2.0),
    )


def _forward_context() -> LegacyTrackingContext:
    nuclei = [
        _nucleus("a1", 1, 0, (0.0, 0.0, 2.0)),
        _nucleus("c1", 1, 1, (10.0, 0.0, 2.0)),
        _nucleus("p", 2, 0, (0.0, 0.0, 2.0)),
        _nucleus("c2", 2, 1, (10.0, 0.0, 2.0)),
        _nucleus("d1", 3, 0, (1.0, 0.0, 3.0), principal=6.0),
        _nucleus("d2", 3, 1, (-1.0, 0.0, 3.0), principal=4.0),
        _nucleus("c3", 3, 2, (10.0, 0.0, 3.0)),
        _nucleus("d24", 4, 0, (-1.0, 0.0, 3.0), principal=4.0),
        _nucleus("c4", 4, 1, (10.0, 0.0, 3.0)),
        _nucleus("forward", 5, 0, (1.0, 0.0, 3.0), principal=6.0),
        _nucleus("d25", 5, 1, (-1.0, 0.0, 3.0), principal=4.0),
        _nucleus("c5", 5, 2, (10.0, 0.0, 3.0)),
        _nucleus("forward6", 6, 0, (1.0, 0.0, 3.0), principal=6.0),
        _nucleus("d26", 6, 1, (-1.0, 0.0, 3.0), principal=4.0),
        _nucleus("c6", 6, 2, (10.0, 0.0, 3.0)),
    ]
    edges = (
        _edge("a1", "p"),
        _edge("c1", "c2"),
        _edge("p", "d1", split=True),
        _edge("p", "d2", split=True),
        _edge("c2", "c3"),
        _edge("d2", "d24"),
        _edge("c3", "c4"),
        _edge("d24", "d25"),
        _edge("c4", "c5"),
        _edge("forward", "forward6"),
        _edge("d25", "d26"),
        _edge("c5", "c6"),
    )
    return LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        edges,
        _parameters(interval=2.0, candidate_cutoff=2.0),
    )


def test_feature_name_contract_is_exactly_22_11_13() -> None:
    assert len(DAUGHTER_FEATURE_NAMES) == 22
    assert len(BACKWARD_FEATURE_NAMES) == 11
    assert len(FORWARD_FEATURE_NAMES) == 13
    assert (
        len(
            set(
                (
                    *DAUGHTER_FEATURE_NAMES,
                    *BACKWARD_FEATURE_NAMES,
                    *FORWARD_FEATURE_NAMES,
                )
            )
        )
        == 46
    )


def test_tracking_statistics_load_exact_matlab_model_field_names() -> None:
    source = {
        "div_mean": [[0.0, 1.0]],
        "div_std": [[2.0, 0.0], [0.0, 3.0]],
        "div_triple_mean": [[float(index) for index in range(10)]],
        "div_triple_std": [
            [2.0 if row == column else 0.0 for column in range(10)]
            for row in range(10)
        ],
        "nodiv_mean": [0.0, 1.0, 2.0, 3.0],
        "nodiv_std": [
            [1.0 if row == column else 0.0 for column in range(4)]
            for row in range(4)
        ],
        "unrelated_legacy_field": "ignored",
    }
    statistics = LegacyTrackingStatistics.from_model(source)
    assert statistics.division_pair_mean == (0.0, 1.0)
    assert statistics.division_pair_covariance[1][1] == 3.0
    assert statistics.division_triple_mean[-1] == 9.0
    assert statistics.nondivision_mean == (0.0, 1.0, 2.0, 3.0)
    assert LegacyTrackingStatistics.from_model(
        {"trackingparameters": {"model": source}}
    ) == statistics

    del source["nodiv_std"]
    with pytest.raises(LegacyFeatureExtractionError, match="nodiv_std"):
        LegacyTrackingStatistics.from_model(source)


def test_division_pair_reproduces_matlab_nonfinite_to_zero_rule() -> None:
    parent = _nucleus("p", 1, 0, (0.0, 0.0, 0.0), total=0.0, average=0.0)
    daughter = _nucleus("d", 2, 0, (0.0, 0.0, 0.0), total=10.0, average=5.0)
    assert calculate_legacy_division_pair(parent, daughter) == (0.0, 0.0)


def test_nondivision_mixed_cat_quantizes_the_entire_vector_to_single() -> None:
    vector = calculate_legacy_nondivision_pair(_base_context(), "p", "d1")

    assert vector[0] == legacy_single_round(0.6)
    assert vector[1] == legacy_single_round(0.6)
    assert vector[0] != 0.6


def test_triple_preserves_mixed_z_coordinate_quirk_and_feature_order() -> None:
    context = _base_context()
    triple = calculate_legacy_division_triple(context, "p", "d1", "d2")
    assert triple == pytest.approx(
        (0.5, 0.0, 0.2, 0.0, 1.2, 1.2, 4.0, 0.75, 0.5, 1.5)
    )
    assert triple[4] == legacy_single_round(1.2)
    assert triple[5] == legacy_single_round(1.2)


def test_complete_long_division_extracts_named_raw_blocks() -> None:
    context = _base_context()
    result = extract_legacy_bifurcation_features(
        context,
        "p",
        _statistics(),
    )

    assert result.daughter_ids == ("d1", "d2")
    assert result.daughter_lengths == (4.0, 4.0)
    assert result.backward_candidate_present == (False, False)
    assert result.best_forward_lengths == (-1.0, -1.0)
    assert result.backward_features == (-1.0, -1.0, -1.0, *(0.0,) * 6, -1.0, -1.0)
    assert result.forward_features == (-1.0, -1.0, -1.0, *(0.0,) * 6, -1.0, -1.0, -1.0, -1.0)
    assert result.daughter_features[:14] == pytest.approx(
        (
            math.log(1.6),
            0.6,
            math.log(1.5),
            0.5,
            0.5,
            0.0,
            0.2,
            0.0,
            1.2,
            1.2,
            4.0,
            0.75,
            0.5,
            1.5,
        )
    )
    assert result.daughter_features[20:] == pytest.approx((4.0, 5.0))
    pair = calculate_legacy_division_pair(
        context.nucleus("p"),
        context.nucleus("d1"),
    )
    assert pair[0] != legacy_single_round(pair[0])
    assert result.daughter_features[0] == legacy_single_round(pair[0])
    assert isinstance(result.repair_result, BackwardRepairCandidates)
    assert result.false_negative_plan is None
    assert result.features is result.feature_input
    assert len(result.named_daughter_features) == 22

    layout = SingleModelFeatureLayout(
        daughter_keep=(True,) * 22,
        backward_keep=(False,) * 11,
        forward_keep=(False,) * 13,
    )
    assembled = assemble_single_model_features(result.feature_input, layout)
    assert assembled.topology_class == 2
    assert assembled.topology_case == "fully_division_looking"

    lineage_state = context.to_lineage_graph_state()
    request = SingleModelLineageRequest.from_legacy_extraction(
        lineage_state,
        result,
    )
    assert request.features is result.feature_input
    assert request.parent_id == "p"
    assert (request.daughter1_id, request.daughter2_id) == ("d1", "d2")
    assert request.false_negative_plan is result.false_negative_plan

    retained = extract_retained_legacy_bifurcations(context, _statistics())
    assert tuple(item.parent_id for item in retained) == ("p",)


def test_retained_snapshot_uses_frame_row_order_not_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nuclei = (
        _nucleus("second_d2", 2, 3, (3.0, 0.0, 0.0)),
        _nucleus("second", 1, 1, (2.0, 0.0, 0.0)),
        _nucleus("first_d1", 2, 0, (0.0, 0.0, 0.0)),
        _nucleus("first", 1, 0, (0.0, 0.0, 0.0)),
        _nucleus("second_d1", 2, 2, (2.0, 0.0, 0.0)),
        _nucleus("first_d2", 2, 1, (1.0, 0.0, 0.0)),
    )
    context = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        (
            _edge("second", "second_d1", split=True),
            _edge("second", "second_d2", split=True),
            _edge("first", "first_d1", split=True),
            _edge("first", "first_d2", split=True),
        ),
        _parameters(end_frame=2),
    )
    observed: list[str] = []

    def record_parent(
        _context: LegacyTrackingContext,
        parent_id: str,
        _statistics_value: LegacyTrackingStatistics,
        **_kwargs: object,
    ) -> object:
        observed.append(parent_id)
        return object()

    monkeypatch.setattr(
        legacy_feature_module,
        "extract_legacy_bifurcation_features",
        record_parent,
    )

    extract_retained_legacy_bifurcations(context, _statistics())

    assert observed == ["first", "second"]


def test_nondivision_scores_use_probability_domain_gaussian_pdf() -> None:
    context = _base_context()
    result = extract_legacy_bifurcation_features(
        context,
        "p",
        _statistics(),
    )
    assert calculate_legacy_nondivision_scores(
        context,
        "p",
        _statistics(),
    ) == result.nondivision_scores
    normalizer = 0.5 * 4.0 * math.log(2.0 * math.pi)
    assert result.nondivision_scores[0] == pytest.approx(
        normalizer + 0.5 * (0.6**2 + 0.6**2 + 0.1**2 + 0.2**2)
    )
    assert result.nondivision_scores[1] == pytest.approx(
        normalizer + 0.5 * (0.5**2 + 0.5**2 + 0.1**2 + 0.2**2)
    )


def test_backward_candidate_order_and_measurements_match_legacy_rules() -> None:
    result = extract_legacy_bifurcation_features(
        _base_context(add_backward_candidate=True),
        "p",
        _statistics(),
    )
    assert result.backward_candidate_present == (True, False)
    assert result.diagnostics["backward_candidate_ids"] == (("back",), ())
    assert result.diagnostics["best_backward_candidate_id"] == "back"
    assert result.backward_features[0] == 2.0
    assert result.backward_features[2] == 1.0
    assert result.backward_features[9:] == pytest.approx((0.0, 0.0))
    assert result.false_negative_plan is not None


def test_forward_d1_keeps_unscaled_gap_offset_and_recursive_lengths() -> None:
    result = extract_legacy_bifurcation_features(
        _forward_context(),
        "p",
        _statistics(),
    )
    assert result.daughter_lengths == (1.0, 4.0)
    assert result.best_forward_lengths == (1.0, -1.0)
    assert result.diagnostics["best_forward_candidate_ids"] == ("forward", None)
    # MATLAB line 236 does not divide d1's offset by interval (2), while its
    # target branch length and recursive lengths are interval-normalised.
    assert result.forward_features[0] == 2.0
    assert result.forward_features[2] == 1.0
    assert result.forward_features[11:] == pytest.approx((1.0, 1.0))


def test_record_answers_mode_normalizes_daughter_lengths_after_traversal() -> None:
    result = extract_legacy_bifurcation_features(
        _forward_context(),
        "p",
        _statistics(),
        record_answers=True,
    )

    assert result.diagnostics["raw_daughter_lengths"] == (1.0, 4.0)
    assert result.daughter_lengths == (0.5, 2.0)
    assert result.daughter_features[20] == 0.5


def test_extraction_is_deeply_immutable() -> None:
    result = extract_legacy_bifurcation_features(
        _base_context(),
        "p",
        _statistics(),
    )
    with pytest.raises(TypeError):
        result.diagnostics["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result.named_daughter_features[DAUGHTER_FEATURE_NAMES[0]] = 0.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.parent_id = "changed"  # type: ignore[misc]


def test_degenerate_pca_infinity_is_checked_after_model_masking() -> None:
    context = _base_context()
    nuclei = tuple(
        _nucleus(
            item.nucleus_id,
            item.frame,
            item.matlab_row,
            item.position_xyz,
            total=item.total_gfp,
            average=item.avg_gfp,
            principal=item.xy_principal_variance,
            secondary=(0.0 if item.nucleus_id == "p" else item.xy_secondary_variance),
        )
        for item in context.nuclei
    )
    degenerate = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        context.edges,
        context.parameters,
    )
    assert math.isinf(calculate_legacy_division_triple(degenerate, "p", "d1", "d2")[6])
    result = extract_legacy_bifurcation_features(degenerate, "p", _statistics())
    assert math.isinf(result.daughter_features[10])

    excluding = SingleModelFeatureLayout(
        daughter_keep=(False,) * 22,
        backward_keep=(False,) * 11,
        forward_keep=(False,) * 13,
    )
    assert assemble_single_model_features(result.feature_input, excluding).values == (2.0,)

    selecting = SingleModelFeatureLayout(
        daughter_keep=(False,) * 10 + (True,) + (False,) * 11,
        backward_keep=(False,) * 11,
        forward_keep=(False,) * 13,
    )
    assert assemble_single_model_features(
        result.feature_input,
        selecting,
    ).values == (2.0, math.inf)


def test_gaussian_tail_scores_preserve_matlab_nonfinite_and_log_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Calls are two nondivision PDFs, two daughter-pair PDFs, then one triple
    # PDF.  These values exercise both tails and a zero inverse-PDF product.
    pdfs = iter((0.0, math.inf, math.inf, 1.0, 1.0))
    monkeypatch.setattr(
        legacy_feature_module,
        "_mvn_pdf",
        lambda *_args, **_kwargs: next(pdfs),
    )

    result = extract_legacy_bifurcation_features(
        _base_context(),
        "p",
        _statistics(),
    )

    assert result.nondivision_scores == (math.inf, -math.inf)
    assert result.diagnostics["division_score"] == -math.inf
