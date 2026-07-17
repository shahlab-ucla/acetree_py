"""Tests for immutable state shared by exact StarryNite extractors."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from acetree_py.tracking.api import Detection, TrackEdge
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyStateError,
    LegacyTrackingContext,
    legacy_gram_distance,
    legacy_single_log,
    legacy_single_mean,
)


def _parameters(
    *,
    end_frame: int = 3,
    anisotropy_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> LegacyFeatureParameters:
    return LegacyFeatureParameters(
        interval=1.0,
        candidate_cutoff=1.2,
        temporal_cutoff=3,
        temporal_cutoff_start=2,
        small_cutoff=4.0,
        anisotropy_xyz=anisotropy_xyz,
        end_frame=end_frame,
    )


def _nucleus(
    nucleus_id: str,
    frame: int,
    row: int,
    position_xyz: tuple[float, float, float],
    *,
    diameter: float = 10.0,
    total_gfp: float = 20.0,
    avg_gfp: float = 4.0,
    aspect_ratio: float = 1.0,
    log_odds_sum: float = 6.0,
    slice_count: int = 3,
    xy_principal_variance: float = 2.0,
    xy_secondary_variance: float = 1.0,
) -> LegacyNucleus:
    return LegacyNucleus(
        nucleus_id=nucleus_id,
        frame=frame,
        matlab_row=row,
        position_xyz=position_xyz,
        diameter=diameter,
        total_gfp=total_gfp,
        avg_gfp=avg_gfp,
        aspect_ratio=aspect_ratio,
        log_odds_sum=log_odds_sum,
        slice_count=slice_count,
        xy_principal_variance=xy_principal_variance,
        xy_secondary_variance=xy_secondary_variance,
    )


def test_nucleus_from_detection_preserves_exact_scalars_and_optional_nan() -> None:
    detection = Detection(
        "n1",
        2,
        100.0,
        200.0,
        30.0,
        5.0,
        9.0,
        {
            "LEGACY_ROW_INDEX": 0,
            "VOXEL_X": 10.0,
            "VOXEL_Y": 20.0,
            "VOXEL_Z": 3.0,
            "LEGACY_DIAMETER_XY_PX": 11.0,
            "LEGACY_TOTAL_GFP": 120.0,
            "LEGACY_AVG_GFP": 8.0,
            "LEGACY_ASPECT_RATIO": 1.25,
            "LEGACY_SLICE_COUNT": 4,
            "LEGACY_XY_PRINCIPAL_VARIANCE": 6.5,
            "LEGACY_XY_SECONDARY_VARIANCE": 2.5,
        },
    )

    nucleus = LegacyNucleus.from_detection(detection)

    assert nucleus.nucleus_id == detection.detection_id
    assert nucleus.position_xyz == (10.0, 20.0, 3.0)
    assert nucleus.diameter == 11.0
    assert nucleus.total_gfp == 120.0
    assert nucleus.avg_gfp == 8.0
    assert nucleus.aspect_ratio == 1.25
    assert nucleus.slice_count == 4
    assert nucleus.xy_principal_variance == 6.5
    assert nucleus.xy_secondary_variance == 2.5
    assert math.isnan(nucleus.log_odds_sum)
    with pytest.raises(LegacyStateError, match="log_odds_sum"):
        LegacyNucleus.from_detection(
            detection,
            required_measurements=("log_odds_sum",),
        )


def test_nucleus_from_detection_quantizes_only_legacy_geometry_to_matlab_single() -> None:
    raw_x = 16_777_217.0
    raw_y = 16_777_219.0
    raw_z = 0.1
    raw_diameter = 1.0000000894069672
    generic_position = (1.234567890123, 2.345678901234, 3.456789012345)
    detection = Detection(
        "single-boundary",
        1,
        *generic_position,
        4.567890123456,
        9.0,
        {
            "LEGACY_ROW_INDEX": 0,
            "VOXEL_X": raw_x,
            "VOXEL_Y": raw_y,
            "VOXEL_Z": raw_z,
            "LEGACY_DIAMETER_XY_PX": raw_diameter,
        },
    )

    nucleus = LegacyNucleus.from_detection(detection)

    assert nucleus.position_xyz == (
        16_777_217.0,
        16_777_219.0,
        0.10000002384185791,
    )
    assert nucleus.diameter == 1.0000001192092896
    # A direct zero-based float32 cast would instead produce 16_777_216 and
    # 0.10000000149011612.  MATLAB adds the one-based offset before its cast.
    assert nucleus.position_xyz[0] != 16_777_216.0
    assert nucleus.position_xyz[2] != 0.10000000149011612
    # Detector/AT-facing state remains full precision and is not mutated by
    # conversion into the MATLAB-compatible legacy representation.
    assert detection.position_um == generic_position
    assert detection.radius_um == 4.567890123456
    assert detection.features["VOXEL_X"] == raw_x
    assert detection.features["VOXEL_Y"] == raw_y
    assert detection.features["VOXEL_Z"] == raw_z
    assert detection.features["LEGACY_DIAMETER_XY_PX"] == raw_diameter


def test_legacy_distance_preserves_one_based_single_gram_cancellation() -> None:
    # Exact Euclidean distance is one. In MATLAB's one-based single Gram form,
    # 4097^2 rounds down by one and the squared-distance expression cancels.
    first = (4095.0,)
    second = (4096.0,)

    assert math.dist(first, second) == 1.0
    assert legacy_gram_distance(first, second) == 0.0


def test_legacy_distance_uses_matlab_native_single_dot_accumulation() -> None:
    # Captured from upstream distance.m under live R2025a. Sequentially
    # rounding each multiply/add produces the adjacent value 147570.265625.
    first = (-81_332.8671875, 93_557.109375, 51_126.23828125)
    second = (516.5201416015625, 41_034.15625, -59_864.734375)

    assert legacy_gram_distance(first, second, zero_based=False) == 147_570.25


def test_single_log_uses_matlab_float32_ufunc_boundary() -> None:
    # Captured directly from the live confidence-vector oracle.
    assert legacy_single_log(1.179023265838623) == 0.16468635201454163


def test_nucleus_from_detection_never_substitutes_generic_geometry() -> None:
    detection = Detection("n1", 1, 1.0, 2.0, 3.0, 4.0, 5.0)

    with pytest.raises(LegacyStateError, match="LEGACY_ROW_INDEX"):
        LegacyNucleus.from_detection(detection)


def test_context_orders_frames_and_successor_slots_by_matlab_row() -> None:
    parent = _nucleus("parent", 1, 0, (0.0, 0.0, 0.0))
    first = _nucleus("first", 2, 0, (-1.0, 0.0, 0.0))
    second = _nucleus("second", 2, 1, (1.0, 0.0, 0.0))
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (second, parent, first),
        (
            TrackEdge("parent", "second", 0.2, kind="split"),
            TrackEdge("parent", "first", 0.2, kind="split"),
        ),
        _parameters(),
    )

    assert context.frame_ids(2) == ("first", "second")
    assert context.successor_slots("parent") == ("first", "second")
    assert context.successors("parent") == ("first", "second")
    assert context.predecessor("first") == "parent"


def test_explicit_legacy_successor_slots_preserve_rewired_daughter_order() -> None:
    parent = _nucleus("parent", 1, 0, (0.0, 0.0, 0.0))
    lower_row = _nucleus("lower", 2, 0, (-1.0, 0.0, 0.0))
    higher_row = _nucleus("higher", 2, 1, (1.0, 0.0, 0.0))
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (parent, lower_row, higher_row),
        (
            TrackEdge(
                "parent",
                "higher",
                0.0,
                kind="split",
                features={"LEGACY_SUCCESSOR_SLOT": 0},
            ),
            TrackEdge(
                "parent",
                "lower",
                0.0,
                kind="split",
                features={"LEGACY_SUCCESSOR_SLOT": 1},
            ),
        ),
        _parameters(),
    )

    assert context.successor_slots("parent") == ("higher", "lower")


def test_matlab_rows_must_be_unique_contiguous_and_zero_based() -> None:
    duplicate = (
        _nucleus("a", 1, 0, (0.0, 0.0, 0.0)),
        _nucleus("b", 1, 0, (1.0, 0.0, 0.0)),
    )
    with pytest.raises(LegacyStateError, match="unique"):
        LegacyTrackingContext.from_nuclei_and_edges(
            duplicate,
            (),
            _parameters(),
        )

    noncontiguous = (
        _nucleus("a", 1, 0, (0.0, 0.0, 0.0)),
        _nucleus("b", 1, 2, (1.0, 0.0, 0.0)),
    )
    with pytest.raises(LegacyStateError, match="contiguous"):
        LegacyTrackingContext.from_nuclei_and_edges(
            noncontiguous,
            (),
            _parameters(),
        )


def test_nearest_neighbor_ties_and_suitors_use_matlab_row_order() -> None:
    nuclei = (
        _nucleus("s0", 1, 0, (0.0, 0.0, 0.0)),
        _nucleus("s1", 1, 1, (10.0, 0.0, 0.0)),
        _nucleus("t0", 2, 0, (5.0, 0.0, 0.0)),
        _nucleus("t1", 2, 1, (5.0, 0.0, 0.0)),
    )
    context = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        (),
        _parameters(),
    )

    assert context.f_nn("s0") == "t0"
    assert context.f_nn("s1") == "t0"
    assert context.b_nn("t0") == "s0"
    assert context.b_nn("t1") == "s0"
    assert context.predecessor_suitors("t0") == ("s0", "s1")
    assert context.successor_suitors("s0") == ("t0", "t1")
    assert context.predecessor_suitors("t1") == ()
    assert context.successor_suitors("s1") == ()


def test_self_distance_and_forward_cutoff_apply_anisotropy() -> None:
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (
            _nucleus("a", 1, 0, (0.0, 0.0, 0.0)),
            _nucleus("b", 1, 1, (0.0, 0.0, 2.0)),
        ),
        (),
        _parameters(anisotropy_xyz=(1.0, 1.0, 3.0)),
    )

    assert context.self_distance("a") == pytest.approx(6.0)
    assert context.self_distance("b") == pytest.approx(6.0)
    assert context.mean_self_distance(1) == pytest.approx(6.0)
    assert context.forward_cutoff(1) == pytest.approx(7.2)
    assert math.isnan(context.mean_self_distance(3))
    assert context.forward_cutoff(3) == -1.0


def test_absolute_candidate_cutoff_skips_spacing_normalization() -> None:
    parameters = LegacyFeatureParameters(
        interval=1.0,
        candidate_cutoff=3.25,
        temporal_cutoff=3,
        temporal_cutoff_start=2,
        small_cutoff=4.0,
        anisotropy_xyz=(1.0, 1.0, 1.0),
        end_frame=1,
        absolute_cutoff=True,
    )
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (
            _nucleus("a", 1, 0, (0.0, 0.0, 0.0)),
            _nucleus("b", 1, 1, (100.0, 0.0, 0.0)),
        ),
        (),
        parameters,
    )

    assert context.mean_self_distance(1) == 100.0
    assert context.forward_cutoff(1) == 3.25


def test_deleted_nuclei_do_not_mutate_frozen_nn_state() -> None:
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (
            _nucleus("active0", 1, 0, (0.0, 0.0, 0.0)),
            _nucleus("deleted", 1, 1, (0.1, 0.0, 0.0)),
            _nucleus("active2", 1, 2, (10.0, 0.0, 0.0)),
        ),
        (),
        _parameters(),
        deleted_ids=("deleted",),
    )

    assert context.frame_ids(1) == ("active0", "active2")
    assert context.frame_ids(1, include_deleted=True) == (
        "active0",
        "deleted",
        "active2",
    )
    near_distance = legacy_gram_distance(
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
    )
    far_distance = legacy_gram_distance(
        (0.1, 0.0, 0.0),
        (10.0, 0.0, 0.0),
    )
    assert context.self_distance("active0") == near_distance
    assert context.self_distance("deleted") == near_distance
    assert context.self_distance("active2") == far_distance
    assert context.mean_self_distance(1) == legacy_single_mean(
        (near_distance, near_distance, far_distance)
    )
    assert context.active_ids == frozenset({"active0", "active2"})


def test_deleted_nuclei_remain_in_frozen_cross_frame_nn_and_suitor_state() -> None:
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (
            _nucleus("source", 1, 0, (0.0, 0.0, 0.0)),
            _nucleus("deleted_source", 1, 1, (10.0, 0.0, 0.0)),
            _nucleus("target", 2, 0, (0.1, 0.0, 0.0)),
            _nucleus("deleted_target", 2, 1, (10.1, 0.0, 0.0)),
        ),
        (),
        _parameters(),
        deleted_ids=("deleted_source", "deleted_target"),
    )

    assert context.f_nn("deleted_source") == "deleted_target"
    assert context.b_nn("deleted_target") == "deleted_source"
    assert context.predecessor_suitors("deleted_target") == ("deleted_source",)
    assert context.successor_suitors("deleted_source") == ("deleted_target",)


def test_deleted_rows_retain_raw_successor_slots_for_candidate_parity() -> None:
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (
            _nucleus("source", 1, 0, (0.0, 0.0, 0.0)),
            _nucleus("deleted_target", 2, 0, (0.0, 0.0, 0.0)),
        ),
        (TrackEdge("source", "deleted_target", 0.0),),
        _parameters(),
        deleted_ids=("deleted_target",),
    )

    assert context.active_ids == frozenset({"source"})
    assert context.frame_ids(2) == ()
    assert context.successor_slots("source") == ("deleted_target", None)
    assert context.predecessor("deleted_target") == "source"

    lineage = context.to_lineage_graph_state()
    assert lineage.deleted_ids == frozenset({"deleted_target"})
    assert lineage.edges == ()


def test_confidence_vector_reproduces_frame_normalization() -> None:
    first = _nucleus(
        "a",
        1,
        0,
        (0.0, 0.0, 0.0),
        total_gfp=10.0,
        avg_gfp=2.0,
        aspect_ratio=1.0,
        log_odds_sum=4.0,
        slice_count=2,
    )
    second = _nucleus(
        "b",
        1,
        1,
        (3.0, 4.0, 2.0),
        total_gfp=30.0,
        avg_gfp=6.0,
        aspect_ratio=3.0,
        log_odds_sum=12.0,
        slice_count=3,
    )
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (first, second),
        (),
        _parameters(),
    )

    distance = math.sqrt(29.0)
    assert context.confidence_vector("a") == pytest.approx(
        (
            0.5,
            math.log(2.0 / distance + 1.0),
            math.log(5.0 / distance + 1.0),
            0.5,
            2.0 / 3.0,
            0.5,
        )
    )
    assert context.confidence_vector("b") == pytest.approx(
        (
            1.5,
            math.log(2.0 / distance + 1.0),
            math.log(5.0 / distance + 1.0),
            1.5,
            4.0 / 3.0,
            1.5,
        )
    )


def test_confidence_double_means_follow_matlab_accumulation_order() -> None:
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (
            _nucleus(
                "large",
                1,
                0,
                (0.0, 0.0, 0.0),
                total_gfp=1e16,
            ),
            _nucleus("small1", 1, 1, (1.0, 0.0, 0.0), total_gfp=1.0),
            _nucleus("small2", 1, 2, (2.0, 0.0, 0.0), total_gfp=1.0),
        ),
        (),
        _parameters(),
    )

    # MATLAB mean([1e16, 1, 1]) is 3333333333333333.5. math.fsum would
    # instead retain both unit terms and make this ratio the lower neighbor.
    assert context.confidence_vector("large")[0] == 3.0


def test_confidence_density_stays_raw_when_frame_mean_is_zero() -> None:
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (
            _nucleus(
                "negative",
                1,
                0,
                (0.0, 0.0, 0.0),
                log_odds_sum=-2.0,
                slice_count=1,
            ),
            _nucleus(
                "positive",
                1,
                1,
                (1.0, 0.0, 0.0),
                log_odds_sum=2.0,
                slice_count=1,
            ),
        ),
        (),
        _parameters(),
    )

    assert context.confidence_vector("negative")[4] == pytest.approx(-2.0)
    assert context.confidence_vector("positive")[4] == pytest.approx(2.0)


def test_traversal_follows_slot_zero_and_can_stop_before_a_division() -> None:
    nuclei = (
        _nucleus("root", 1, 0, (0.0, 0.0, 0.0)),
        _nucleus("parent", 2, 0, (0.0, 0.0, 0.0)),
        _nucleus("d1", 3, 0, (-1.0, 0.0, 0.0)),
        _nucleus("d2", 3, 1, (1.0, 0.0, 0.0)),
        _nucleus("d1_next", 4, 0, (-2.0, 0.0, 0.0)),
    )
    edges = (
        TrackEdge("root", "parent", 0.0),
        TrackEdge("parent", "d2", 0.0, kind="split"),
        TrackEdge("parent", "d1", 0.0, kind="split"),
        TrackEdge("d1", "d1_next", 0.0),
    )
    context = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        edges,
        _parameters(end_frame=4),
    )

    assert context.traverse_forward("root") == ("root", "parent", "d1", "d1_next")
    assert context.forward_depth("d1") == 2
    assert context.traverse_backward("d1_next") == (
        "d1_next",
        "d1",
        "parent",
        "root",
    )
    assert context.traverse_backward(
        "d1_next",
        stop_at_division=True,
    ) == ("d1_next", "d1")
    assert context.backward_depth("d1_next", stop_at_division=True) == 2


def test_context_and_nested_collections_are_immutable() -> None:
    nucleus = _nucleus("a", 1, 0, (0.0, 0.0, 0.0))
    context = LegacyTrackingContext.from_nuclei_and_edges(
        (nucleus,),
        (),
        _parameters(),
    )

    with pytest.raises(FrozenInstanceError):
        nucleus.frame = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.edges = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        context.nuclei_by_id["b"] = nucleus  # type: ignore[index]
    with pytest.raises(TypeError):
        context.successor_slots_by_id["a"] = ("b", None)  # type: ignore[index]


def test_graph_and_parameter_validation_fail_closed() -> None:
    a = _nucleus("a", 1, 0, (0.0, 0.0, 0.0))
    a2 = _nucleus("a2", 1, 1, (1.0, 0.0, 0.0))
    b = _nucleus("b", 2, 0, (0.0, 0.0, 0.0))
    with pytest.raises(LegacyStateError, match="merge"):
        LegacyTrackingContext.from_nuclei_and_edges(
            (a, a2, b),
            (TrackEdge("a", "b", 0.0), TrackEdge("a2", "b", 0.0)),
            _parameters(),
        )
    with pytest.raises(LegacyStateError, match="temporal_cutoff_start"):
        LegacyFeatureParameters(
            interval=1.0,
            candidate_cutoff=1.2,
            temporal_cutoff=2,
            temporal_cutoff_start=3,
            small_cutoff=4.0,
            anisotropy_xyz=(1.0, 1.0, 2.0),
            end_frame=3,
        )


def test_feature_parameters_load_exact_matlab_tracking_fields() -> None:
    parameters = LegacyFeatureParameters.from_model(
        {
            "trackingparameters": {
                "interval": 2.0,
                "candidateCutoff": 7.5,
                "temporalcutoff": 8,
                "temporalcutoffstart": 3,
                "smallcutoff": 5.0,
                "anisotropyvector": (1.0, 1.0, 4.0),
                "endtime": 20,
                "abscutoff": 1.0,
            }
        },
        end_frame=12,
    )

    assert parameters.interval == 2.0
    assert parameters.candidate_cutoff == 7.5
    assert parameters.temporal_cutoff_start == 3
    assert parameters.anisotropy_xyz == (1.0, 1.0, 4.0)
    assert parameters.end_frame == 12
    assert parameters.absolute_cutoff is True
