"""Exact learned StarryNite detector-tail and sequential-state tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from acetree_py.tracking.api import Calibration
from acetree_py.tracking.starrynite.detector import (
    LegacyRadialGeometry,
    LegacyResolvedCandidate,
    StarryNiteDetector,
)
from acetree_py.tracking.starrynite.legacy_detector_tail import (
    LegacyDetectorNucleus,
    LegacyDetectorTailError,
    LegacyDetectorTailResult,
    LegacyDiskDistributionModel,
    legacy_adapt_cell_diameter,
    legacy_calculate_disk_log_odds,
    legacy_disk_feature_vectors,
    legacy_maximal_disk_range,
    load_legacy_disk_distributions,
)


def _identity_model(
    *, source_path: Path | None = None, source_sha256: str | None = None
) -> LegacyDiskDistributionModel:
    mean = np.zeros(7, dtype=np.float64)
    covariance = np.eye(7, dtype=np.float64)
    return LegacyDiskDistributionModel(
        mean,
        covariance,
        mean,
        covariance,
        mean,
        covariance,
        mean,
        covariance,
        source_path,
        source_sha256,
    )


def _geometry(z: float, x: float, *, coverage: int = 16) -> LegacyRadialGeometry:
    return LegacyRadialGeometry(
        center_zyx_px=(z, 5.0, x),
        diameter_xy_px=5.0,
        valid_ray_count=coverage,
        peak_response=10.0,
    )


def test_disk_feature_order_and_low_coverage_log_odds_match_legacy_rules() -> None:
    geometries = (_geometry(0, 4), _geometry(1, 5), _geometry(2, 6))
    features = legacy_disk_feature_vectors(
        (0, 1, 2),
        1,
        geometries,
        (5.0, 10.0, 20.0),
        (5.0, 5.0, 5.0),
        2.0,
    )

    np.testing.assert_allclose(
        features,
        np.asarray(
            (
                (0.2, 0.2, 0.0),
                (1.0, 1.0, 0.0),
                (0.5, 2.0, 0.0),
                (-0.4, 0.4, 0.0),
                (0.0, 0.0, 0.0),
                (0.5, 1.0, 0.0),
                (0.2, 0.2, 0.0),
            )
        ),
        rtol=0.0,
        atol=1e-7,
    )
    assert legacy_calculate_disk_log_odds(
        (0, 1, 2),
        1,
        geometries,
        (5.0, 10.0, 20.0),
        (5.0, 5.0, 5.0),
        2.0,
        _identity_model(),
    ) == (0.0, 0.0, 0.0)

    low_coverage = (_geometry(0, 4, coverage=13), geometries[1], geometries[2])
    nucleus = LegacyDetectorNucleus(
        center_disk_index=1,
        disk_indices=(0, 1, 2),
        log_odds=(-100.0, 0.0, 0.0),
        retained_indices=(),
    )
    # Coverage <= 13 is retained even when its learned score is far below the
    # threshold, exactly as vcalculateMaximalRange.m specifies.
    assert legacy_maximal_disk_range(nucleus, low_coverage, 1.0) == (1, 0, 2)


def test_distribution_loader_is_source_bound_and_rejects_bad_covariance(
    tmp_path: Path,
) -> None:
    scipy_io = pytest.importorskip("scipy.io")
    path = tmp_path / "clean_distributions.mat"
    mean = np.arange(7, dtype=np.float64)[np.newaxis, :]
    covariance = np.eye(7, dtype=np.float64)
    scipy_io.savemat(
        path,
        {
            "allbadlm": mean,
            "allbadlc": covariance,
            "allgoodlm": mean,
            "allgoodlc": covariance,
            "allbadrm": mean,
            "allbadrc": covariance,
            "allgoodrm": mean,
            "allgoodrc": covariance,
        },
    )

    model = load_legacy_disk_distributions(path)

    assert model.source_path == path.resolve()
    assert model.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert model.good_right_mean.tolist() == list(range(7))

    bad = tmp_path / "bad.mat"
    scipy_io.savemat(
        bad,
        {
            "allbadlm": mean,
            "allbadlc": np.zeros((7, 7)),
            "allgoodlm": mean,
            "allgoodlc": covariance,
            "allbadrm": mean,
            "allbadrc": covariance,
            "allgoodrm": mean,
            "allgoodrc": covariance,
        },
    )
    with pytest.raises(LegacyDetectorTailError, match="positive definite"):
        load_legacy_disk_distributions(bad)


def test_previous_candidate_median_diameter_requires_more_than_ten_values() -> None:
    assert legacy_adapt_cell_diameter(
        8.0, previous_candidate_diameters_xy_px=(6.0,) * 10
    ) == 8.0
    assert legacy_adapt_cell_diameter(
        8.0, previous_candidate_diameters_xy_px=(6.0,) * 11
    ) == 6.0
    assert legacy_adapt_cell_diameter(
        8.0,
        previous_candidate_diameters_xy_px=(6.0,) * 11,
        use_static_diameter=True,
    ) == 8.0


def test_exact_detector_uses_prior_final_count_and_candidate_median(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import acetree_py.tracking.starrynite.legacy_detector_tail as tail

    distribution = tmp_path / "distribution.mat"
    distribution.write_bytes(b"source-bound-test-model")
    digest = hashlib.sha256(distribution.read_bytes()).hexdigest()
    model = _identity_model(source_path=distribution.resolve(), source_sha256=digest)
    monkeypatch.setattr(tail, "load_legacy_disk_distributions", lambda _path: model)
    calls: list[tuple[float, int]] = []

    def fake_exact_tail(
        _response,
        _threshold,
        *,
        expected_diameter_xy_px,
        model=None,
        numcells,
        **_kwargs,
    ) -> LegacyDetectorTailResult:
        calls.append((float(expected_diameter_xy_px), int(numcells)))
        count = 3 if len(calls) == 1 else 1
        candidates = tuple(
            LegacyResolvedCandidate(
                center_zyx_px=(2.0, 6.0, 5.0 + index),
                diameter_xy_px=5.0,
                representative_peak_zyx=(2, 6, 5 + index),
                valid_ray_count=16,
                claimed_slice_count=0,
                merged_candidate_count=1,
                log_odds_sum=0.0,
            )
            for index in range(count)
        )
        return LegacyDetectorTailResult(
            candidates=candidates,
            nuclei=(),
            initial_center_count=count,
            recovery_round_count=0,
            disk_count=count,
            candidate_diameters_xy_px=(6.0,) * 11,
            merge_pairs=(),
        )

    monkeypatch.setattr(tail, "legacy_resolve_candidates_exact", fake_exact_tail)
    detector = StarryNiteDetector()
    image = np.zeros((5, 15, 15), dtype=np.float32)
    settings = {
        "RADIUS": 4.0,
        "SIGMA": 1.0,
        "INTENSITY_THRESHOLD": 1.0,
        "STARRYNITE_CELL_COUNT": 7,
        "STARRYNITE_STAGE_INDEX": 2,
        "STARRYNITE_DISTRIBUTION_FILE": str(distribution),
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": digest,
    }

    first = detector.detect(image, 1, Calibration(1.0, 1.0), settings)
    second = detector.detect(image, 2, Calibration(1.0, 1.0), settings)
    reset = detector.detect(image, 4, Calibration(1.0, 1.0), settings)

    assert calls == [(8.0, 7), (6.0, 3), (8.0, 7)]
    assert first[0].features["LEGACY_EFFECTIVE_DIAMETER_XY_PX"] == 8.0
    assert second[0].features["LEGACY_EFFECTIVE_DIAMETER_XY_PX"] == 6.0
    assert second[0].features["STARRYNITE_CELL_COUNT"] == 3
    assert reset[0].features["STARRYNITE_CELL_COUNT"] == 7
    assert second[0].features["STARRYNITE_DISTRIBUTION_FILE"] == str(
        distribution.resolve()
    )
    assert second[0].features["STARRYNITE_DISTRIBUTION_SHA256"] == digest
    assert second[0].features["STARRYNITE_DISTRIBUTION_SOURCE_SHA256"] == digest
    assert second[0].features["LEGACY_CANDIDATE_DIAMETER_COUNT"] == 11
    assert second[0].features["LEGACY_CANDIDATE_DIAMETER_MEDIAN_XY_PX"] == 6.0
    assert second[0].features["LEGACY_PREVIOUS_CANDIDATE_DIAMETER_COUNT"] == 11
    assert second[0].features[
        "LEGACY_PREVIOUS_CANDIDATE_DIAMETER_MEDIAN_XY_PX"
    ] == 6.0
    assert second[0].features["STARRYNITE_PARAMETER_FILE"] == ""
    assert second[0].features["STARRYNITE_PARAMETER_SHA256"] == ""


def test_exact_detector_static_diameter_ignores_prior_candidate_median(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import acetree_py.tracking.starrynite.legacy_detector_tail as tail

    distribution = tmp_path / "distribution.mat"
    distribution.write_bytes(b"static-source")
    digest = hashlib.sha256(distribution.read_bytes()).hexdigest()
    monkeypatch.setattr(
        tail,
        "load_legacy_disk_distributions",
        lambda _path: _identity_model(
            source_path=distribution.resolve(), source_sha256=digest
        ),
    )
    diameters: list[float] = []

    def fake_tail(
        _response,
        _threshold,
        *,
        expected_diameter_xy_px,
        model=None,
        **_kwargs,
    ) -> LegacyDetectorTailResult:
        diameters.append(float(expected_diameter_xy_px))
        candidate = LegacyResolvedCandidate(
            center_zyx_px=(2.0, 6.0, 6.0),
            diameter_xy_px=5.0,
            representative_peak_zyx=(2, 6, 6),
            valid_ray_count=16,
            claimed_slice_count=0,
            merged_candidate_count=1,
            log_odds_sum=0.0,
        )
        return LegacyDetectorTailResult(
            (candidate,), (), 1, 0, 1, (6.0,) * 11, ()
        )

    monkeypatch.setattr(tail, "legacy_resolve_candidates_exact", fake_tail)
    detector = StarryNiteDetector()
    settings = {
        "RADIUS": 4.0,
        "INTENSITY_THRESHOLD": 1.0,
        "STARRYNITE_DISTRIBUTION_FILE": str(distribution),
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": digest,
        "STARRYNITE_USE_STATIC_DIAMETER": True,
    }
    image = np.zeros((5, 15, 15), dtype=np.float32)

    detector.detect(image, 1, Calibration(1.0, 1.0), settings)
    second = detector.detect(image, 2, Calibration(1.0, 1.0), settings)

    assert diameters == [8.0, 8.0]
    assert second[0].features["STARRYNITE_USE_STATIC_DIAMETER"] is True


def test_exact_detector_preserves_empty_frame_state_and_unique_legacy_row_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import acetree_py.tracking.starrynite.legacy_detector_tail as tail

    distribution = tmp_path / "distribution.mat"
    distribution.write_bytes(b"empty-state-source")
    digest = hashlib.sha256(distribution.read_bytes()).hexdigest()
    monkeypatch.setattr(
        tail,
        "load_legacy_disk_distributions",
        lambda _path: _identity_model(
            source_path=distribution.resolve(), source_sha256=digest
        ),
    )
    calls = 0

    def fake_tail(*_args, **_kwargs) -> LegacyDetectorTailResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return LegacyDetectorTailResult((), (), 0, 0, 11, (6.0,) * 11, ())
        candidate = LegacyResolvedCandidate(
            center_zyx_px=(2.0, 6.0, 6.0),
            diameter_xy_px=5.0,
            representative_peak_zyx=(2, 6, 6),
            valid_ray_count=16,
            claimed_slice_count=0,
            merged_candidate_count=1,
            log_odds_sum=0.0,
        )
        # Equal legacy coordinates used to collide because the coordinate hash
        # alone was treated as row identity.
        return LegacyDetectorTailResult(
            (candidate, candidate), (), 2, 0, 2, (5.0, 5.0), ()
        )

    monkeypatch.setattr(tail, "legacy_resolve_candidates_exact", fake_tail)
    detector = StarryNiteDetector()
    settings = {
        "RADIUS": 4.0,
        "INTENSITY_THRESHOLD": 1.0,
        "STARRYNITE_DISTRIBUTION_FILE": str(distribution),
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": digest,
    }
    image = np.zeros((5, 15, 15), dtype=np.float32)

    assert detector.detect(image, 1, Calibration(1.0, 1.0), settings) == ()
    second = detector.detect(image, 2, Calibration(1.0, 1.0), settings)

    assert len(second) == 2
    assert len({item.detection_id for item in second}) == 2
    assert ":r0:" in second[0].detection_id
    assert ":r1:" in second[1].detection_id
    assert [item.features["LEGACY_ROW_INDEX"] for item in second] == [0, 1]
    for item in second:
        assert item.features["LEGACY_EFFECTIVE_DIAMETER_XY_PX"] == 6.0
        assert item.features["STARRYNITE_CELL_COUNT"] == 0
        assert item.features["LEGACY_PREVIOUS_CANDIDATE_DIAMETER_COUNT"] == 11
        assert item.features["LEGACY_PREVIOUS_CANDIDATE_DIAMETER_MEDIAN_XY_PX"] == 6.0


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf))
def test_exact_detector_rejects_nonfinite_voxels_without_native_normalization(
    tmp_path: Path, bad_value: float
) -> None:
    distribution = tmp_path / "distribution.mat"
    distribution.write_bytes(b"finite-input-source")
    digest = hashlib.sha256(distribution.read_bytes()).hexdigest()
    image = np.zeros((5, 15, 15), dtype=np.float32)
    image[2, 7, 7] = bad_value

    with pytest.raises(ValueError, match="requires finite image voxels"):
        StarryNiteDetector().detect(
            image,
            1,
            Calibration(1.0, 1.0),
            {
                "RADIUS": 4.0,
                "INTENSITY_THRESHOLD": 1.0,
                "STARRYNITE_DISTRIBUTION_FILE": str(distribution),
                "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": digest,
            },
        )

    # Native-fast mode retains its documented defensive normalization.
    assert StarryNiteDetector().detect(
        image,
        1,
        Calibration(1.0, 1.0),
        {"RADIUS": 4.0, "INTENSITY_THRESHOLD": 1.0},
    ) == ()


def test_exact_detector_requires_request_time_distribution_identity(
    tmp_path: Path,
) -> None:
    distribution = tmp_path / "distribution.mat"
    distribution.write_bytes(b"unbound-source")

    with pytest.raises(ValueError, match="request-time.*binding"):
        StarryNiteDetector().detect(
            np.zeros((5, 15, 15), dtype=np.float32),
            1,
            Calibration(1.0, 1.0),
            {"STARRYNITE_DISTRIBUTION_FILE": str(distribution)},
        )


def test_exact_detector_caches_source_bound_models_but_rehashes_each_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import acetree_py.tracking.starrynite.legacy_detector_tail as tail
    import acetree_py.tracking.starrynite.parameter_view as parameter_view
    import acetree_py.tracking.starrynite.parameters as parameter_module

    distribution = tmp_path / "distribution.mat"
    distribution.write_bytes(b"cache-test-distribution")
    distribution_hash = hashlib.sha256(distribution.read_bytes()).hexdigest()
    parameter_file = tmp_path / "parameters.m"
    parameter_file.write_text(
        "parameters.staging=[100];\n"
        "parameters.sigma=1;\n"
        "parameters.intensitythreshold=1;\n"
        "downsampling=1;\n",
        encoding="utf-8",
    )
    parameter_hash = hashlib.sha256(parameter_file.read_bytes()).hexdigest()

    load_calls: list[Path] = []

    def load_distribution(path: str | Path) -> LegacyDiskDistributionModel:
        load_calls.append(Path(path))
        return _identity_model(
            source_path=distribution.resolve(), source_sha256=distribution_hash
        )

    parse_calls: list[Path] = []
    original_read = parameter_module.read_parameter_file

    def read_parameters(path: str | Path):
        parse_calls.append(Path(path))
        return original_read(path)

    region_calls: list[object] = []
    original_regions = parameter_view.build_legacy_region_table

    def build_regions(parameters, *, strict=False):
        region_calls.append(parameters)
        return original_regions(parameters, strict=strict)

    candidate = LegacyResolvedCandidate(
        center_zyx_px=(2.0, 6.0, 6.0),
        diameter_xy_px=5.0,
        representative_peak_zyx=(2, 6, 6),
        valid_ray_count=16,
        claimed_slice_count=0,
        merged_candidate_count=1,
        log_odds_sum=0.0,
    )
    monkeypatch.setattr(tail, "load_legacy_disk_distributions", load_distribution)
    monkeypatch.setattr(parameter_module, "read_parameter_file", read_parameters)
    monkeypatch.setattr(parameter_view, "build_legacy_region_table", build_regions)
    monkeypatch.setattr(
        tail,
        "legacy_resolve_candidates_exact",
        lambda *_args, **_kwargs: LegacyDetectorTailResult(
            (candidate,), (), 1, 0, 1, (5.0,), ()
        ),
    )

    detector = StarryNiteDetector()
    settings = {
        "RADIUS": 4.0,
        "STARRYNITE_PARAMETER_FILE": str(parameter_file),
        "STARRYNITE_PARAMETER_SHA256": parameter_hash,
        "STARRYNITE_DISTRIBUTION_FILE": str(distribution),
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": distribution_hash,
    }
    image = np.zeros((5, 15, 15), dtype=np.float32)
    detector.detect(image, 1, Calibration(1.0, 1.0), settings)
    detector.detect(image, 2, Calibration(1.0, 1.0), settings)

    assert load_calls == [distribution.resolve()]
    assert parse_calls == [parameter_file.resolve()]
    assert len(region_calls) == 1

    # A discontinuous frame starts a fresh exact-movie state and therefore
    # rebuilds both source-bound caches.
    detector.detect(image, 4, Calibration(1.0, 1.0), settings)
    assert load_calls == [distribution.resolve(), distribution.resolve()]
    assert parse_calls == [parameter_file.resolve(), parameter_file.resolve()]
    assert len(region_calls) == 2

    distribution.write_bytes(b"mutated-distribution")
    with pytest.raises(ValueError, match="changed after the detector request"):
        detector.detect(image, 5, Calibration(1.0, 1.0), settings)
    assert load_calls == [distribution.resolve()] * 2
    assert parse_calls == [parameter_file.resolve()] * 2


def test_exact_detector_crops_legacy_roi_and_restores_global_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import acetree_py.tracking.starrynite.legacy_detector_tail as tail

    distribution = tmp_path / "distribution.mat"
    distribution.write_bytes(b"roi-distribution")
    distribution_hash = hashlib.sha256(distribution.read_bytes()).hexdigest()
    parameter_file = tmp_path / "parameters.m"
    parameter_file.write_text(
        "parameters.staging=[100];\n"
        "parameters.sigma=1;\n"
        "parameters.intensitythreshold=1;\n"
        "downsampling=1;\n"
        "ROI=true;\n"
        "ROIxmin=5; ROIxmax=10;\n"
        "ROIymin=4; ROIymax=9;\n"
        "ROIpoints=[5 4; 10 4; 10 9; 5 9;];\n",
        encoding="utf-8",
    )
    parameter_hash = hashlib.sha256(parameter_file.read_bytes()).hexdigest()
    monkeypatch.setattr(
        tail,
        "load_legacy_disk_distributions",
        lambda _path: _identity_model(
            source_path=distribution.resolve(), source_sha256=distribution_hash
        ),
    )
    observed: dict[str, object] = {}

    def fake_tail(response, _threshold, **kwargs) -> LegacyDetectorTailResult:
        observed["shape"] = response.shape
        observed["roi_points"] = kwargs["roi_points_xy"]
        observed["roi_cropped"] = kwargs["roi_cropped"]
        observed["roi_offset"] = kwargs["roi_offset_xy"]
        candidate = LegacyResolvedCandidate(
            center_zyx_px=(2.0, 2.0, 3.0),
            diameter_xy_px=5.0,
            representative_peak_zyx=(2, 2, 3),
            valid_ray_count=16,
            claimed_slice_count=0,
            merged_candidate_count=1,
            log_odds_sum=0.0,
        )
        return LegacyDetectorTailResult((candidate,), (), 1, 0, 1, (5.0,), ())

    monkeypatch.setattr(tail, "legacy_resolve_candidates_exact", fake_tail)
    detector = StarryNiteDetector()
    settings = {
        "RADIUS": 4.0,
        "STARRYNITE_PARAMETER_FILE": str(parameter_file),
        "STARRYNITE_PARAMETER_SHA256": parameter_hash,
        "STARRYNITE_DISTRIBUTION_FILE": str(distribution),
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": distribution_hash,
    }
    detection = detector.detect(
        np.zeros((5, 15, 20), dtype=np.float32),
        1,
        Calibration(1.0, 1.0),
        settings,
    )[0]

    assert observed == {
        "shape": (5, 6, 6),
        "roi_points": ((5, 4), (10, 4), (10, 9), (5, 9)),
        "roi_cropped": True,
        "roi_offset": (5.0, 4.0),
    }
    assert detection.features["VOXEL_Z"] == 2.0
    assert detection.features["VOXEL_Y"] == 5.0
    assert detection.features["VOXEL_X"] == 7.0
    assert detection.features["LEGACY_ROI_BOUNDS_XY_1BASED"] == (
        5.0,
        10.0,
        4.0,
        9.0,
    )
    assert detection.features["STARRYNITE_PARAMETER_FILE"] == str(
        parameter_file.resolve()
    )
    assert detection.features["STARRYNITE_PARAMETER_SHA256"] == parameter_hash

    with pytest.raises(ValueError, match="outside the supplied full image"):
        StarryNiteDetector().detect(
            np.zeros((5, 15, 20), dtype=np.float32),
            1,
            Calibration(1.0, 1.0),
            {
                "RADIUS": 4.0,
                "STARRYNITE_DISTRIBUTION_FILE": str(distribution),
                "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": distribution_hash,
                "ROI_CROPPED": True,
                "ROI_X_OFFSET": 5,
                "ROI_X_MAX": 21,
                "ROI_Y_OFFSET": 4,
                "ROI_Y_MAX": 9,
            },
        )
