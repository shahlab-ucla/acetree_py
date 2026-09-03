"""Executable and fail-closed tests for the whole-movie exact backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acetree_py.io.image_provider import NumpyProvider
from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    Detection,
    TrackingRequest,
    TrackingScope,
    WholeMoviePreflightContext,
)
from acetree_py.tracking.pipeline import TrackingPipeline
from acetree_py.tracking.registry import build_default_registry
from acetree_py.tracking.starrynite import (
    CategoricalFeatureDistribution,
    GaussianFeatureDistribution,
    LEGACY_EXACT_REFINEMENT_BACKEND,
    NeutralNaiveBayesClassifier,
    SingleModelFeatureLayout,
    StarryNiteLegacyExactError,
    StarryNiteLegacyExactTracker,
    load_tuning_profile,
    save_neutral_classifier,
    sha256_file,
)
from acetree_py.tracking.starrynite.legacy_exact_tracker import (
    _validate_detections,
)


def _classifier(source_hash: str) -> NeutralNaiveBayesClassifier:
    layout = SingleModelFeatureLayout(
        daughter_keep=(True,) * 12 + (False,) * 10,
        backward_keep=(False,) * 11,
        forward_keep=(True,) * 8 + (False,) * 5,
    )
    continuous = GaussianFeatureDistribution(
        means=(0.0, 1.0, 2.0, 3.0),
        standard_deviations=(1.0, 1.0, 1.0, 1.0),
    )
    topology = CategoricalFeatureDistribution(
        categories=(1.0, 2.0, 3.0, 4.0, 5.0),
        probabilities=((0.2,) * 5,) * 4,
    )
    return NeutralNaiveBayesClassifier(
        source_model_sha256=source_hash,
        classifier_family="new_classifier",
        feature_layout=layout,
        feature_names=("topology_class",)
        + tuple(f"feature_{index}" for index in range(layout.selected_feature_count)),
        class_labels=(0, 1, 2, 3),
        class_priors=(0.25, 0.25, 0.25, 0.25),
        misclassification_costs=(
            (0.0, 1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 0.0),
        ),
        distributions=(topology,) + (continuous,) * layout.selected_feature_count,
    )


def _runtime_files(tmp_path: Path):
    scipy_io = pytest.importorskip("scipy.io")
    model_path = tmp_path / "tracking.mat"
    scipy_io.savemat(
        model_path,
        {
            "trackingparameters": {
                "model": {
                    "div_mean": np.zeros(2),
                    "div_std": np.eye(2),
                    "div_triple_mean": np.zeros(10),
                    "div_triple_std": np.eye(10),
                    "nodiv_mean": np.zeros(4),
                    "nodiv_std": np.eye(4),
                },
                "interval": 1,
                "candidateCutoff": 1.2,
                "temporalcutoff": 1,
                "temporalcutoffstart": 1,
                "smallcutoff": 4,
                "endtime": 2,
                "anisotropyvector": np.asarray([1, 1, 4]),
                "starttime": 1,
                "safefilter": False,
                "safefactor": 2,
                "conflictfilter": False,
                "nnnumber": 2,
                "forwardnnnumber": 4,
                "minnondivscore": 0,
                "nondivscorestep": 1,
                "maxnondivscore": 0,
                "mindivscore": 0,
                "divscorestep": 1,
                "maxdivscore": 0,
                "polarbodyfilter": False,
                "hysteresis": False,
                "deleteisolated": False,
            }
        },
    )
    distribution_path = tmp_path / "distribution.mat"
    distribution_mean = np.zeros((1, 7), dtype=np.float64)
    distribution_covariance = np.eye(7, dtype=np.float64)
    scipy_io.savemat(
        distribution_path,
        {
            "allbadlm": distribution_mean,
            "allbadlc": distribution_covariance,
            "allgoodlm": distribution_mean,
            "allgoodlc": distribution_covariance,
            "allbadrm": distribution_mean,
            "allbadrc": distribution_covariance,
            "allgoodrm": distribution_mean,
            "allgoodrc": distribution_covariance,
        },
    )
    parameter_path = tmp_path / "parameters.m"
    parameter_path.write_text(
        "parameters.staging=[25,80];\n"
        "parameters.sigma=1;\n"
        "parameters.intensitythreshold=5;\n"
        "parameters.rangethreshold=10;\n"
        "parameters.boundary_percent=.5;\n"
        "parameters.large_ray_threshold=1.5;\n"
        "parameters.small_ray_threshold=.333333333333;\n"
        "parameters.mergelower=-20;\n"
        "parameters.mergesplit=1;\n"
        "parameters.split=20;\n"
        "parameters.nndist_merge=.8;\n"
        "parameters.armerge=1.6;\n"
        "firsttimestepdiam=8;\n"
        "firsttimestepnumcells=1;\n"
        "downsampling=1;\n"
        "xyres=.25;\n"
        "zres=1;\n"
        "distribution_file='distribution.mat';\n"
        "load 'tracking.mat';\n"
        "trackingparameters.nonDivCostFunction=@distanceCostFunction;\n"
        "trackingparameters.DivCostFunction=@divScoreModelCostFunction;\n",
        encoding="utf-8",
    )
    profile = load_tuning_profile(parameter_path)
    classifier_path = tmp_path / "classifier.json"
    save_neutral_classifier(classifier_path, _classifier(profile.model_sha256 or ""))
    settings = {
        "STARRYNITE_COMPATIBILITY_MODE": LEGACY_EXACT_REFINEMENT_BACKEND,
        "STARRYNITE_PARAMETER_FILE": str(parameter_path.resolve()),
        "STARRYNITE_PARAMETER_SHA256": profile.parameter_sha256,
        "STARRYNITE_MODEL_FILE": str(model_path.resolve()),
        "STARRYNITE_MODEL_SHA256": profile.model_sha256,
        "STARRYNITE_NEUTRAL_CLASSIFIER_FILE": str(classifier_path.resolve()),
        "STARRYNITE_NEUTRAL_CLASSIFIER_SHA256": sha256_file(classifier_path),
        "STARRYNITE_REQUIRE_EXACT_DETECTOR_TAIL": True,
        "STARRYNITE_USE_STATIC_DIAMETER": False,
    }
    return profile, distribution_path.resolve(), settings


def _detector_spec(profile) -> ComponentSpec:
    return ComponentSpec(
        "acetree.starrynite_detector",
        dict(profile.detector_settings),
    )


def _preflight_context(
    profile,
    *,
    scope: TrackingScope | None = None,
    source_num_timepoints: int = 2,
    calibration: Calibration | None = None,
) -> WholeMoviePreflightContext:
    return WholeMoviePreflightContext(
        detector_spec=_detector_spec(profile),
        calibration=calibration or Calibration(0.25, 1.0),
        scope=scope or TrackingScope("global", 1, source_num_timepoints),
        source_num_timepoints=source_num_timepoints,
        source_num_channels=1,
        target_channel=0,
    )


def _detection(
    identifier: str,
    frame: int,
    x: float,
    *,
    row: int,
    profile,
    distribution_path: Path,
    cell_count: int = 1,
    stage_index: int = 0,
    effective_diameter: float = 8.0,
    candidate_diameters: tuple[float, ...] = (8.0,),
    previous_candidate_diameters: tuple[float, ...] | None = None,
    static_diameter: bool = False,
) -> Detection:
    parameter_path = profile.parameters.source_path
    assert parameter_path is not None
    if previous_candidate_diameters is None:
        previous_candidate_diameters = () if frame == 1 else candidate_diameters
    candidate_median = (
        None if not candidate_diameters else float(np.median(candidate_diameters))
    )
    previous_candidate_median = (
        None
        if not previous_candidate_diameters
        else float(np.median(previous_candidate_diameters))
    )
    distribution_sha256 = sha256_file(distribution_path)
    return Detection(
        identifier,
        frame,
        x * 0.25,
        0.0,
        0.0,
        1.0,
        10.0,
        {
            "LEGACY_ROW_INDEX": row,
            "VOXEL_X": x,
            "VOXEL_Y": 0.0,
            "VOXEL_Z": 0.0,
            "LEGACY_DIAMETER_XY_PX": 8.0,
            "LEGACY_EFFECTIVE_DIAMETER_XY_PX": effective_diameter,
            "LEGACY_CANDIDATE_DIAMETERS_XY_PX": candidate_diameters,
            "LEGACY_CANDIDATE_DIAMETER_COUNT": len(candidate_diameters),
            "LEGACY_CANDIDATE_DIAMETER_MEDIAN_XY_PX": candidate_median,
            "LEGACY_PREVIOUS_CANDIDATE_DIAMETER_COUNT": len(
                previous_candidate_diameters
            ),
            "LEGACY_PREVIOUS_CANDIDATE_DIAMETER_MEDIAN_XY_PX": (
                previous_candidate_median
            ),
            "LEGACY_EXACT_TAIL": True,
            "STARRYNITE_CELL_COUNT": cell_count,
            "STARRYNITE_STAGE_INDEX": stage_index,
            "STARRYNITE_USE_STATIC_DIAMETER": static_diameter,
            "STARRYNITE_PARAMETER_FILE": str(parameter_path.resolve()),
            "STARRYNITE_PARAMETER_SHA256": profile.parameter_sha256,
            "STARRYNITE_DISTRIBUTION_FILE": str(distribution_path),
            "STARRYNITE_DISTRIBUTION_SHA256": distribution_sha256,
            "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": distribution_sha256,
        },
    )


def test_exact_tracker_executes_early_and_classifier_movie_boundary(tmp_path: Path):
    profile, distribution_path, settings = _runtime_files(tmp_path)
    detections = (
        _detection(
            "first",
            1,
            10.0,
            row=0,
            profile=profile,
            distribution_path=distribution_path,
        ),
        _detection(
            "second",
            2,
            10.5,
            row=0,
            profile=profile,
            distribution_path=distribution_path,
        ),
    )

    result = StarryNiteLegacyExactTracker().refine_movie(
        detections,
        settings,
        detector_spec=_detector_spec(profile),
        calibration=Calibration(0.25, 1.0),
        start_frame=1,
        end_frame=2,
    )

    assert result.detections == detections
    assert [(item.source_id, item.target_id) for item in result.edges] == [
        ("first", "second")
    ]
    assert not result.rejected_detection_ids
    assert result.provenance["backend"] == LEGACY_EXACT_REFINEMENT_BACKEND
    assert result.provenance["event_order_validated"] is True
    assert result.provenance["classifier_source_model_sha256"] == profile.model_sha256
    assert result.provenance["early_stages"][-1]["name"] == "geometry_final"


def test_exact_tracker_preflight_is_read_only_and_requires_complete_movie(
    tmp_path: Path,
):
    profile, _distribution_path, settings = _runtime_files(tmp_path)
    tracker = StarryNiteLegacyExactTracker()
    settings_before = dict(settings)
    context = _preflight_context(profile)
    detector_settings_before = dict(context.detector_spec.settings)
    source_paths = (
        Path(settings["STARRYNITE_PARAMETER_FILE"]),
        Path(settings["STARRYNITE_MODEL_FILE"]),
        Path(settings["STARRYNITE_NEUTRAL_CLASSIFIER_FILE"]),
        Path(context.detector_spec.settings["STARRYNITE_DISTRIBUTION_FILE"]),
    )
    source_hashes = {path: sha256_file(path) for path in source_paths}

    assert tracker.preflight_movie(
        settings,
        context=context,
    ) is None

    assert settings == settings_before
    assert dict(context.detector_spec.settings) == detector_settings_before
    assert {path: sha256_file(path) for path in source_paths} == source_hashes
    assert vars(tracker) == {}
    with pytest.raises(StarryNiteLegacyExactError, match="complete image source"):
        tracker.preflight_movie(
            settings,
            context=_preflight_context(
                profile,
                scope=TrackingScope("global", 1, 1),
                source_num_timepoints=2,
            ),
        )


def test_exact_tracker_preflight_rejects_changed_classifier_source(tmp_path: Path):
    profile, _distribution_path, settings = _runtime_files(tmp_path)
    classifier_path = Path(settings["STARRYNITE_NEUTRAL_CLASSIFIER_FILE"])
    classifier_path.write_text(
        classifier_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StarryNiteLegacyExactError, match="classifier changed"):
        StarryNiteLegacyExactTracker().preflight_movie(
            settings,
            context=_preflight_context(profile),
        )


def test_exact_pipeline_rejects_partial_movie_before_reading_frame_one(tmp_path: Path):
    profile, _distribution_path, settings = _runtime_files(tmp_path)

    class RecordingProvider(NumpyProvider):
        def __init__(self, data):
            super().__init__(data)
            self.calls = []

        def get_stack(self, time, channel=0):
            self.calls.append((time, channel))
            return super().get_stack(time, channel)

    provider = RecordingProvider(np.zeros((2, 3, 8, 8), dtype=np.float32))
    request = TrackingRequest(
        detector=_detector_spec(profile),
        tracker=ComponentSpec(StarryNiteLegacyExactTracker.plugin_id, settings),
        scope=TrackingScope("global", 1, 1),
    )

    with pytest.raises(StarryNiteLegacyExactError, match="complete image source"):
        TrackingPipeline(build_default_registry(discover_plugins=False)).run(
            provider,
            Calibration(0.25, 1.0),
            request,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "feature_name",
    (
        "STARRYNITE_DISTRIBUTION_SHA256",
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256",
    ),
)
def test_exact_tracker_rejects_mixed_detector_distribution_provenance(
    tmp_path: Path,
    feature_name: str,
):
    profile, distribution_path, settings = _runtime_files(tmp_path)
    detection = _detection(
        "mixed",
        1,
        10.0,
        row=0,
        profile=profile,
        distribution_path=distribution_path,
    )
    detection = Detection(
        **{
            **detection.to_dict(),
            "features": {
                **dict(detection.features),
                feature_name: "f" * 64,
            },
        }
    )

    with pytest.raises(StarryNiteLegacyExactError, match="same parameter source"):
        StarryNiteLegacyExactTracker().refine_movie(
            (detection,),
            settings,
            detector_spec=_detector_spec(profile),
            calibration=Calibration(0.25, 1.0),
            start_frame=1,
            end_frame=1,
        )


def test_exact_tracker_requires_full_history_and_matching_calibration(tmp_path: Path):
    profile, _distribution_path, settings = _runtime_files(tmp_path)
    tracker = StarryNiteLegacyExactTracker()

    with pytest.raises(StarryNiteLegacyExactError, match="start at frame 1"):
        tracker.refine_movie(
            (),
            settings,
            detector_spec=_detector_spec(profile),
            calibration=Calibration(0.25, 1.0),
            start_frame=2,
            end_frame=2,
        )
    with pytest.raises(StarryNiteLegacyExactError, match="dataset calibration"):
        tracker.refine_movie(
            (),
            settings,
            detector_spec=_detector_spec(profile),
            calibration=Calibration(0.5, 1.0),
            start_frame=1,
            end_frame=2,
        )


def test_exact_detector_dynamic_state_is_source_bound_across_empty_frame(
    tmp_path: Path,
) -> None:
    profile, distribution_path, settings = _runtime_files(tmp_path)
    prior_diameters = tuple(float(value) for value in range(8, 19))
    first_rows = tuple(
        _detection(
            f"first-{row}",
            1,
            10.0 + row,
            row=row,
            profile=profile,
            distribution_path=distribution_path,
            candidate_diameters=prior_diameters,
        )
        for row in range(11)
    )
    median_row = _detection(
        "median",
        2,
        30.0,
        row=0,
        profile=profile,
        distribution_path=distribution_path,
        cell_count=11,
        effective_diameter=13.0,
        previous_candidate_diameters=prior_diameters,
    )

    _validate_detections(
        (*first_rows, median_row),
        profile=profile,
        values=settings,
        start_frame=1,
        end_frame=2,
        require_exact_tail=True,
    )

    after_empty = _detection(
        "after-empty",
        3,
        40.0,
        row=0,
        profile=profile,
        distribution_path=distribution_path,
        cell_count=0,
        effective_diameter=6.0,
        previous_candidate_diameters=(6.0,) * 11,
    )
    _validate_detections(
        (*first_rows, after_empty),
        profile=profile,
        values=settings,
        start_frame=1,
        end_frame=3,
        require_exact_tail=True,
    )

    forged_row = _detection(
        "forged-row",
        1,
        10.0,
        row=1,
        profile=profile,
        distribution_path=distribution_path,
    )
    with pytest.raises(StarryNiteLegacyExactError, match="contiguous"):
        _validate_detections(
            (forged_row,),
            profile=profile,
            values=settings,
            start_frame=1,
            end_frame=1,
            require_exact_tail=True,
        )

    mixed_state = _detection(
        "mixed-state",
        1,
        11.0,
        row=1,
        profile=profile,
        distribution_path=distribution_path,
        cell_count=2,
    )
    with pytest.raises(StarryNiteLegacyExactError, match="disagree on dynamic"):
        _validate_detections(
            (first_rows[0], mixed_state),
            profile=profile,
            values=settings,
            start_frame=1,
            end_frame=1,
            require_exact_tail=True,
        )


def test_dynamic_provenance_scans_shared_candidate_tuple_once_per_frame(
    tmp_path: Path,
) -> None:
    profile, distribution_path, settings = _runtime_files(tmp_path)

    class CountingTuple(tuple):
        scans = 0

        def __iter__(self):
            type(self).scans += 1
            return super().__iter__()

    candidates = CountingTuple((8.0,) * 11)
    rows = tuple(
        _detection(
            f"row-{row}",
            1,
            10.0 + row,
            row=row,
            profile=profile,
            distribution_path=distribution_path,
            candidate_diameters=candidates,
        )
        for row in range(11)
    )
    CountingTuple.scans = 0

    _validate_detections(
        rows,
        profile=profile,
        values=settings,
        start_frame=1,
        end_frame=1,
        require_exact_tail=True,
    )

    assert CountingTuple.scans == 1
