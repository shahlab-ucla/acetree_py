"""Native StarryNite detector, linker, and lineage-integration tests."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.core.nucleus import Nucleus
from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)
from acetree_py.tracking.integration import ApplyTrackingProposal
from acetree_py.tracking.registry import build_default_registry
from acetree_py.tracking.starrynite.detector import (
    LegacyRadialGeometry,
    StarryNiteDetector,
    _canonical_maxima,
    _legacy_slice_maxima,
    _regional_maxima_2d,
    _xy_principal_variances,
    legacy_dog_filter_parameters,
)
from acetree_py.tracking.starrynite.tracker import StarryNiteDivisionTracker
from acetree_py.tracking.starrynite.oracle.synthetic import (
    default_synthetic_suite,
    lineage_synthetic_suite,
)


def _detection(identifier: str, frame: int, x: float, radius: float = 4.0):
    return Detection(identifier, frame, x, 10.0, 3.0, radius, 10.0)


def test_starrynite_components_are_combined_with_existing_registry():
    registry = build_default_registry(discover_plugins=False)

    assert {item.plugin_id for item in registry.detector_descriptors()} == {
        "acetree.dog3d",
        "acetree.log3d",
        "acetree.starrynite_detector",
    }
    assert {item.plugin_id for item in registry.tracker_descriptors()} == {
        "acetree.simple_lap",
        "acetree.starrynite_division",
        "acetree.starrynite_legacy_exact",
    }
    descriptor = registry.get_descriptor("acetree.starrynite_division")
    assert "splitting" in descriptor.capabilities
    assert registry.default_settings(descriptor.plugin_id)["ALLOW_TRACK_SPLITTING"] is True
    assert registry.default_settings("acetree.starrynite_detector")[
        "DO_SUBPIXEL_LOCALIZATION"
    ] is False
    detector_defaults = registry.default_settings("acetree.starrynite_detector")
    assert detector_defaults["RANGE_THRESHOLD"] == 1.0
    assert detector_defaults["SPLIT_THRESHOLD"] == 100.0
    assert detector_defaults["MERGE_LOWER"] == -300.0
    assert detector_defaults["MERGE_SPLIT"] == 1.0
    assert detector_defaults["ROI_X_MAX"] == 0.0
    assert detector_defaults["ROI_Y_MAX"] == 0.0
    assert detector_defaults["STARRYNITE_USE_STATIC_DIAMETER"] is False
    assert detector_defaults["STARRYNITE_DISTRIBUTION_SOURCE_SHA256"] == ""
    assert registry.default_settings("acetree.dog3d")[
        "DO_SUBPIXEL_LOCALIZATION"
    ] is True


def test_starrynite_tracker_emits_exactly_two_split_edges():
    parent = _detection("parent", 1, 10.0)
    left = _detection("left", 2, 8.5, radius=3.0)
    right = _detection("right", 2, 11.5, radius=3.0)

    edges = StarryNiteDivisionTracker().track(
        (parent, left, right),
        {
            "ALLOW_TRACK_SPLITTING": True,
            "ALLOW_GAP_CLOSING": False,
            "LINKING_MAX_DISTANCE": 8.0,
            "DIVISION_MAX_DAUGHTER_DISTANCE": 8.0,
            "DIVISION_MAX_DAUGHTER_SEPARATION": 8.0,
            "DIVISION_MAX_MIDPOINT_ERROR": 4.0,
        },
    )

    assert [(edge.source_id, edge.target_id, edge.kind) for edge in edges] == [
        ("parent", "left", "split"),
        ("parent", "right", "split"),
    ]
    assert all(edge.features["SCORER"].endswith("/v1") for edge in edges)


def test_starrynite_tracker_can_be_used_as_one_to_one_linker():
    parent = _detection("parent", 1, 10.0)
    target = _detection("target", 2, 11.0)

    edges = StarryNiteDivisionTracker().track(
        (parent, target),
        {
            "ALLOW_TRACK_SPLITTING": False,
            "ALLOW_GAP_CLOSING": False,
            "LINKING_MAX_DISTANCE": 4.0,
        },
    )

    assert len(edges) == 1
    assert edges[0].kind == "link"
    assert edges[0].features["STARRYNITE_HYPOTHESIS"] == "continuation"


def test_starrynite_detector_finds_an_anisotropic_gaussian_blob():
    z, y, x = np.indices((9, 31, 31), dtype=float)
    image = 100.0 * np.exp(
        -(
            ((z - 4.0) / 1.0) ** 2
            + ((y - 15.0) / 2.0) ** 2
            + ((x - 16.0) / 2.0) ** 2
        )
        / 2.0
    )

    detections = StarryNiteDetector().detect(
        image.astype(np.float32),
        3,
        Calibration(0.5, 1.0),
        {
            "RADIUS": 2.0,
            "SIGMA": 1.0,
            "INTENSITY_THRESHOLD": 1.0,
            "BOUNDARY_PERCENT": 0.3,
        },
    )

    assert len(detections) == 1
    found = detections[0]
    assert abs(found.x_um - 8.0) < 0.6
    assert abs(found.y_um - 7.5) < 0.6
    assert abs(found.z_um - 4.0) < 0.8
    assert found.features["SUPPORT_VOXELS"] > 0


def test_legacy_dog_kernel_uses_cell_diameter_and_absolute_threshold():
    parameters = legacy_dog_filter_parameters(
        radius_um=5.0,
        sigma_factor=1.0,
        calibration=Calibration(0.25, 1.0),
    )

    denominator = 4.0 * np.sqrt(2.0 * np.log(2.0))
    assert parameters.inner_support_zyx == (10, 40, 40)
    assert parameters.outer_support_zyx == (16, 64, 64)
    assert parameters.inner_sigma_zyx == pytest.approx(
        (10 / denominator, 40 / denominator, 40 / denominator)
    )
    assert parameters.outer_sigma_zyx == pytest.approx(
        (16 / denominator, 64 / denominator, 64 / denominator)
    )


def test_single_slice_xy_pca_repeats_points_like_matlab() -> None:
    endpoints = tuple((float(index), 0.0) for index in range(16))
    geometry = LegacyRadialGeometry(
        center_zyx_px=(0.0, 0.0, 0.0),
        diameter_xy_px=15.0,
        valid_ray_count=16,
        ray_endpoints_xy_px=endpoints,
        peak_response=1.0,
    )

    principal, secondary = _xy_principal_variances((geometry,), 0.35)

    repeated_x = np.repeat(np.arange(16, dtype=float), 2)
    assert principal == pytest.approx(float(np.var(repeated_x, ddof=1)))
    assert secondary == 0.0


def test_detector_measurement_weight_uses_legacy_integrated_gfp():
    z, y, x = np.indices((9, 31, 31), dtype=float)
    blob = 100.0 * np.exp(
        -(
            ((z - 4.0) / 1.0) ** 2
            + ((y - 15.0) / 2.0) ** 2
            + ((x - 16.0) / 2.0) ** 2
        )
        / 2.0
    )
    detector = StarryNiteDetector()
    settings = {
        "RADIUS": 2.0,
        "SIGMA": 1.0,
        "INTENSITY_THRESHOLD": 1.0,
        "BOUNDARY_PERCENT": 0.3,
    }

    baseline = detector.detect(
        blob.astype(np.float32),
        1,
        Calibration(0.5, 1.0),
        settings,
    )[0]
    offset = detector.detect(
        (blob + 1_000.0).astype(np.float32),
        1,
        Calibration(0.5, 1.0),
        settings,
    )[0]

    support_voxels = baseline.features["SUPPORT_VOXELS"]
    assert offset.features["SUPPORT_VOXELS"] == support_voxels
    assert offset.features["TOTAL_INTENSITY"] - baseline.features[
        "TOTAL_INTENSITY"
    ] == pytest.approx(1_000.0 * support_voxels)
    # The native support sum remains available for scoring, but StarryNite's
    # nuclei-file weight is uint16(integrateGFP(...).totalGFP / 256).
    assert offset.features["LEGACY_TOTAL_GFP"] != pytest.approx(
        offset.features["TOTAL_INTENSITY"]
    )
    expected_weight = int(
        np.floor(offset.features["LEGACY_TOTAL_GFP"] / 256.0 + 0.5)
    )
    assert offset.features["ACETREE_WEIGHT"] == min(expected_weight, 65_535)
    raw_support_weight = int(
        np.floor(offset.features["TOTAL_INTENSITY"] / 256.0 + 0.5)
    )
    assert offset.features["ACETREE_WEIGHT"] != min(raw_support_weight, 65_535)


def test_canonical_maxima_handles_many_disconnected_peaks_in_one_pass():
    response = np.zeros((32, 64, 64), dtype=np.float32)
    response[1::4, 1::4, 1::4] = 10.0

    peaks = _canonical_maxima(
        response,
        threshold=1.0,
        footprint=np.ones((3, 3, 3), dtype=bool),
    )

    assert len(peaks) == 8 * 16 * 16
    assert peaks[0] == (1, 1, 1)
    assert peaks[-1] == (29, 61, 61)


def test_legacy_regional_maxima_rejects_an_entire_shelf_touching_a_higher_value():
    plane = np.zeros((5, 9), dtype=np.float32)
    plane[2, 2:6] = 5.0
    plane[2, 6] = 6.0

    maxima = _regional_maxima_2d(plane)

    assert np.argwhere(maxima).tolist() == [[2, 6]]
    assert _legacy_slice_maxima(plane[np.newaxis], 4.0) == ((0, 2, 6),)


def test_legacy_regional_maxima_preserves_every_pixel_of_a_true_plateau():
    plane = np.zeros((7, 9), dtype=np.float32)
    plane[1:3, 1:3] = 5.0
    plane[5, 7] = 6.0

    maxima = _legacy_slice_maxima(plane[np.newaxis], 4.0)

    assert maxima == (
        (0, 1, 1),
        (0, 2, 1),
        (0, 1, 2),
        (0, 2, 2),
        (0, 5, 7),
    )


def test_legacy_regional_maxima_uses_matlab_eight_connected_plateaus():
    plane = np.zeros((7, 7), dtype=np.float32)
    plane[1, 1] = 5.0
    plane[2, 2] = 5.0
    plane[3, 3] = 5.0
    plane[4, 4] = 6.0

    assert _legacy_slice_maxima(plane[np.newaxis], 4.0) == ((0, 4, 4),)


def test_legacy_neighbor_maxima_preserve_close_daughters_without_explicit_suppression():
    movie = default_synthetic_suite(seed=1731)[-1]
    image = movie.frames_tzyx[3]
    settings = {
        "RADIUS": 2.0,
        "SIGMA": 1.0,
        "INTENSITY_THRESHOLD": 18.0,
        "BOUNDARY_PERCENT": 0.35,
        "DO_SUBPIXEL_LOCALIZATION": False,
    }

    legacy = StarryNiteDetector().detect(
        image,
        4,
        movie.calibration,
        settings,
    )
    suppressed = StarryNiteDetector().detect(
        image,
        4,
        movie.calibration,
        {**settings, "MIN_SEPARATION": 2.0},
    )

    assert [item.features["PEAK_VOXEL_X"] for item in legacy] == [21.0, 25.0]
    assert [item.features["VOXEL_X"] for item in legacy] == [20.0, 26.0]
    assert [item.features["LEGACY_DIAMETER_XY_PX"] for item in legacy] == [
        item.radius_um * 2.0 / movie.calibration.xy_um for item in legacy
    ]
    assert len(suppressed) == 1


def test_legacy_boundary_transition_uses_disk_claims_before_geometric_merge():
    movie = default_synthetic_suite(seed=1731)[-1]
    detector = StarryNiteDetector()
    common = {
        "RADIUS": 2.0,
        "SIGMA": 1.0,
        "INTENSITY_THRESHOLD": 18.0,
    }

    observed = {}
    for boundary in (0.18, 0.19, 0.20, 0.27, 0.28, 0.35, 0.65):
        detections = detector.detect(
            movie.frames_tzyx[3],
            4,
            movie.calibration,
            {**common, "BOUNDARY_PERCENT": boundary},
        )
        observed[boundary] = tuple(
            (
                item.features["VOXEL_X"],
                item.features["LEGACY_DIAMETER_XY_PX"],
                item.features["LEGACY_MERGED_CANDIDATE_COUNT"],
            )
            for item in detections
        )

    assert len(observed[0.18]) == 1
    assert observed[0.18][0] == pytest.approx((22.1666666667, 13.0, 2))
    assert observed[0.19] == ((22.0, 13.0, 2),)
    assert observed[0.20] == ((22.0, 13.0, 2),)
    assert observed[0.27] == ((23.0, 13.0, 2),)
    assert observed[0.28] == ((20.0, 9.0, 1), (25.0, 8.0, 1))
    assert observed[0.35] == ((20.0, 7.0, 1), (26.0, 7.0, 1))
    assert observed[0.65] == ((21.0, 5.0, 1), (25.0, 8.0, 1))


def test_model_positive_division_fixture_is_available_to_tuned_native_tracker():
    movie = lineage_synthetic_suite(seed=2718)[3]
    detector = StarryNiteDetector()
    detections = tuple(
        detection
        for frame, image in enumerate(movie.frames_tzyx, start=1)
        for detection in detector.detect(
            image,
            frame,
            movie.calibration,
            {
                "RADIUS": 4.0,
                "SIGMA": 0.5,
                "INTENSITY_THRESHOLD": 0.25,
                "BOUNDARY_PERCENT": 0.35,
                "DO_SUBPIXEL_LOCALIZATION": False,
            },
        )
    )
    tracker = StarryNiteDivisionTracker()

    default_edges = tracker.track(detections, {})
    tuned_edges = tracker.track(
        detections,
        {"DIVISION_MAX_DAUGHTER_SEPARATION": 20.0},
    )
    split_edges = tuple(edge for edge in tuned_edges if edge.kind == "split")

    assert [sum(item.frame == frame for item in detections) for frame in range(1, 8)] == [
        4,
        4,
        5,
        5,
        5,
        5,
        5,
    ]
    assert not any(edge.kind == "split" for edge in default_edges)
    assert len(split_edges) == 2
    assert {edge.source_id for edge in split_edges} == {split_edges[0].source_id}
    assert split_edges[0].features["MIDPOINT_ERROR_UM"] == pytest.approx(1.0)
    assert split_edges[0].features["DAUGHTER_SEPARATION_UM"] == pytest.approx(16.0)


def test_two_daughter_proposal_commits_to_legacy_nucleus_links_and_undoes():
    seed = _detection("seed", 1, 10.0)
    left = _detection("left", 2, 8.5, radius=3.0)
    right = _detection("right", 2, 11.5, radius=3.0)
    request = TrackingRequest(
        ComponentSpec("acetree.starrynite_detector"),
        ComponentSpec("acetree.starrynite_division"),
        TrackingScope("selected_forward", 1, 2, seed_anchors=((1, 1),)),
    )
    result = TrackingResult(
        request,
        (seed, left, right),
        (
            TrackEdge("seed", "left", 0.1, kind="split"),
            TrackEdge("seed", "right", 0.1, kind="split"),
        ),
        existing_anchors={"seed": (1, 1)},
    )
    record = [[Nucleus(index=1, x=10, y=10, z=4.0, size=8, status=1)], []]
    command = ApplyTrackingProposal(result, Calibration(1.0, 1.0))

    command.execute(record)

    assert (record[0][0].successor1, record[0][0].successor2) == (1, 2)
    assert [item.predecessor for item in record[1]] == [1, 1]
    command.undo(record)
    assert record[0][0].successor1 < 0
    assert record[0][0].successor2 < 0
    assert record[1] == []
