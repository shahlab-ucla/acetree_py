"""Compatibility-report and fail-closed backend-selection tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from acetree_py.tracking.starrynite import (
    CategoricalFeatureDistribution,
    LEGACY_EXACT_REFINEMENT_BACKEND,
    NATIVE_FAST_BACKEND,
    GaussianFeatureDistribution,
    NeutralNaiveBayesClassifier,
    SingleModelFeatureLayout,
    StarryNiteDivisionTracker,
    StarryNiteCompatibilityError,
    build_compatibility_report,
    load_tuning_profile,
    save_neutral_classifier,
    select_compatibility_backend,
)


def _neutral_model(source_hash: str) -> NeutralNaiveBayesClassifier:
    layout = SingleModelFeatureLayout(
        daughter_keep=(True,) * 12 + (False,) * 10,
        backward_keep=(False,) * 11,
        forward_keep=(True,) * 8 + (False,) * 5,
    )
    distribution = GaussianFeatureDistribution(
        means=(0.0, 1.0, 2.0, 3.0),
        standard_deviations=(1.0, 1.0, 1.0, 1.0),
    )
    topology = CategoricalFeatureDistribution(
        categories=(1.0, 2.0, 3.0, 4.0, 5.0),
        probabilities=(
            (0.2, 0.2, 0.2, 0.2, 0.2),
            (0.2, 0.2, 0.2, 0.2, 0.2),
            (0.2, 0.2, 0.2, 0.2, 0.2),
            (0.2, 0.2, 0.2, 0.2, 0.2),
        ),
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
        distributions=(topology,) + (distribution,) * layout.selected_feature_count,
    )


def _profile(tmp_path: Path, *, opaque: bool = False):
    scipy_io = pytest.importorskip("scipy.io")
    model_path = tmp_path / "tracking-model.mat"
    distribution_mean = np.zeros((1, 7), dtype=np.float64)
    distribution_covariance = np.eye(7, dtype=np.float64)
    scipy_io.savemat(
        tmp_path / "distribution.mat",
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
                "temporalcutoff": 2,
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
                "nonDivCostFunction": "distanceCostFunction",
                "DivCostFunction": "divScoreModelCostFunction",
                "polarbodyfilter": False,
                "hysteresis": False,
                "deleteisolated": False,
            }
        },
    )
    parameter_path = tmp_path / "parameters.m"
    parameter_path.write_text(
        "parameters.staging=[25,80];\n"
        "parameters.sigma=1;\n"
        "parameters.intensitythreshold=5;\n"
        "parameters.intensitythreshold=7;\n"
        "parameters.rangethreshold=10;\n"
        "parameters.boundary_percent=.5;\n"
        "parameters.large_ray_threshold=1.5;\n"
        "parameters.small_ray_threshold=.333333333333;\n"
        "parameters.mergelower=-20;\n"
        "parameters.mergesplit=1;\n"
        "parameters.split=20;\n"
        "parameters.nndist_merge=.8;\n"
        "parameters.armerge=1.6;\n"
        "firsttimestepdiam=32;\n"
        "firsttimestepnumcells=30;\n"
        "downsampling=1;\n"
        "xyres=.25;\n"
        "zres=1;\n"
        "distribution_file='distribution.mat';\n"
        "load 'tracking-model.mat';\n"
        + ("danger=system('never execute');\n" if opaque else ""),
        encoding="utf-8",
    )
    return load_tuning_profile(parameter_path), model_path


def test_report_explains_native_behavior_and_exact_blockers(tmp_path: Path) -> None:
    profile, model_path = _profile(tmp_path, opaque=True)

    report = build_compatibility_report(profile)

    assert report.backend(NATIVE_FAST_BACKEND).runnable
    exact = report.backend(LEGACY_EXACT_REFINEMENT_BACKEND)
    assert not exact.runnable
    assert {
        "exact_runtime_unavailable",
        "opaque_parameter_statement",
        "neutral_classifier_not_selected",
    } <= {issue.code for issue in exact.blockers}
    intensity = next(
        item
        for item in report.effective_parameters
        if item.normalized_name == "parameters.intensitythreshold"
    )
    assert intensity.value == 7
    assert report.model_references[0].resolved_path == model_path.resolve()
    assert report.model_references[0].exists
    assert "provenance only" in report.format_text()
    # Reports are persistence/diagnostic artifacts, so strict JSON encoding is
    # part of the public boundary.
    json.dumps(report.to_dict(), allow_nan=False)


def test_exact_report_rejects_corrupt_distribution_payload(tmp_path: Path) -> None:
    profile, _model_path = _profile(tmp_path)
    distribution_path = Path(
        profile.detector_settings["STARRYNITE_DISTRIBUTION_FILE"]
    )
    distribution_path.write_bytes(b"not a MATLAB payload")

    report = build_compatibility_report(profile)

    assert "legacy_distribution_invalid" in {
        issue.code
        for issue in report.backend(LEGACY_EXACT_REFINEMENT_BACKEND).blockers
    }


def test_exact_report_rejects_distribution_changed_after_profile_load(
    tmp_path: Path,
) -> None:
    scipy_io = pytest.importorskip("scipy.io")
    profile, _model_path = _profile(tmp_path)
    distribution_path = Path(
        profile.detector_settings["STARRYNITE_DISTRIBUTION_FILE"]
    )
    changed_mean = np.ones((1, 7), dtype=np.float64)
    covariance = np.eye(7, dtype=np.float64)
    scipy_io.savemat(
        distribution_path,
        {
            "allbadlm": changed_mean,
            "allbadlc": covariance,
            "allgoodlm": changed_mean,
            "allgoodlc": covariance,
            "allbadrm": changed_mean,
            "allbadrc": covariance,
            "allgoodrm": changed_mean,
            "allgoodrc": covariance,
        },
    )

    report = build_compatibility_report(profile)

    assert "legacy_distribution_changed" in {
        issue.code
        for issue in report.backend(LEGACY_EXACT_REFINEMENT_BACKEND).blockers
    }


def test_exact_backend_requires_source_bound_neutral_export(tmp_path: Path) -> None:
    profile, _model_path = _profile(tmp_path)
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(neutral_path, _neutral_model(profile.model_sha256 or ""))

    report = build_compatibility_report(
        profile,
        neutral_classifier_path=neutral_path,
        runtime_capabilities=(LEGACY_EXACT_REFINEMENT_BACKEND,),
    )

    assert select_compatibility_backend(
        report, LEGACY_EXACT_REFINEMENT_BACKEND
    ) == LEGACY_EXACT_REFINEMENT_BACKEND
    assert report.neutral_classifier_family == "new_classifier"
    assert report.neutral_source_model_sha256 == profile.model_sha256


def test_exact_source_allows_zero_initial_count_but_not_implicit_downsampling(
    tmp_path: Path,
) -> None:
    profile, _model_path = _profile(tmp_path)
    parameter_path = profile.parameters.source_path
    assert parameter_path is not None
    source = parameter_path.read_text(encoding="utf-8")
    parameter_path.write_text(
        source.replace("firsttimestepnumcells=30", "firsttimestepnumcells=0").replace(
            "downsampling=1;\n", ""
        ),
        encoding="utf-8",
    )
    missing_profile = load_tuning_profile(parameter_path)
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(
        neutral_path,
        _neutral_model(missing_profile.model_sha256 or ""),
    )

    missing = build_compatibility_report(
        missing_profile,
        neutral_classifier_path=neutral_path,
        runtime_capabilities=(LEGACY_EXACT_REFINEMENT_BACKEND,),
    )
    blocker_codes = {
        issue.code
        for issue in missing.backend(LEGACY_EXACT_REFINEMENT_BACKEND).blockers
    }
    assert "legacy_downsampling_missing" in blocker_codes
    assert not any(
        "firsttimestepnumcells" in issue.message
        for issue in missing.backend(LEGACY_EXACT_REFINEMENT_BACKEND).blockers
    )

    parameter_path.write_text(
        parameter_path.read_text(encoding="utf-8") + "downsampling=1;\n",
        encoding="utf-8",
    )
    explicit_profile = load_tuning_profile(parameter_path)
    explicit = build_compatibility_report(
        explicit_profile,
        neutral_classifier_path=neutral_path,
        runtime_capabilities=(LEGACY_EXACT_REFINEMENT_BACKEND,),
    )
    assert explicit.backend(LEGACY_EXACT_REFINEMENT_BACKEND).runnable


def test_exact_source_blocks_a_missing_production_detector_stage(
    tmp_path: Path,
) -> None:
    profile, _model_path = _profile(tmp_path)
    parameter_path = profile.parameters.source_path
    assert parameter_path is not None
    parameter_path.write_text(
        parameter_path.read_text(encoding="utf-8").replace(
            "parameters.split=20;\n", ""
        ),
        encoding="utf-8",
    )
    incomplete_profile = load_tuning_profile(parameter_path)
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(
        neutral_path,
        _neutral_model(incomplete_profile.model_sha256 or ""),
    )

    report = build_compatibility_report(
        incomplete_profile,
        neutral_classifier_path=neutral_path,
        runtime_capabilities=(LEGACY_EXACT_REFINEMENT_BACKEND,),
    )
    blockers = report.backend(LEGACY_EXACT_REFINEMENT_BACKEND).blockers

    assert any(
        issue.code == "legacy_exact_detector_parameter_missing"
        and "parameters.split" in issue.message
        for issue in blockers
    )
    assert not report.backend(LEGACY_EXACT_REFINEMENT_BACKEND).runnable


def test_exact_report_accepts_only_implemented_static_cost_dispatch(
    tmp_path: Path,
) -> None:
    profile, _model_path = _profile(tmp_path)
    parameter_path = profile.parameters.source_path
    assert parameter_path is not None
    source = parameter_path.read_text(encoding="utf-8")
    parameter_path.write_text(
        source
        + "trackingparameters.nonDivCostFunction=@nondivScoreModelCostFunction;\n"
        + "trackingparameters.DivCostFunction=@divDistanceCostFunction;\n",
        encoding="utf-8",
    )
    supported = build_compatibility_report(load_tuning_profile(parameter_path))
    assert not supported.opaque_issues

    parameter_path.write_text(
        source + "trackingparameters.nonDivCostFunction=@unknownCost;\n",
        encoding="utf-8",
    )
    unsupported = build_compatibility_report(load_tuning_profile(parameter_path))
    assert any(
        issue.code == "opaque_parameter_statement"
        for issue in unsupported.opaque_issues
    )

    wrong_path = tmp_path / "wrong-classifier.json"
    save_neutral_classifier(wrong_path, _neutral_model("f" * 64))
    mismatched = build_compatibility_report(
        profile,
        neutral_classifier_path=wrong_path,
        runtime_capabilities=(LEGACY_EXACT_REFINEMENT_BACKEND,),
    )
    with pytest.raises(StarryNiteCompatibilityError, match="not usable"):
        select_compatibility_backend(
            mismatched,
            LEGACY_EXACT_REFINEMENT_BACKEND,
        )


def test_report_rehashes_parameter_and_model_sources_at_validation_time(
    tmp_path: Path,
) -> None:
    profile, model_path = _profile(tmp_path)
    parameter_path = profile.parameters.source_path
    assert parameter_path is not None
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(neutral_path, _neutral_model(profile.model_sha256 or ""))

    parameter_path.write_text(
        parameter_path.read_text(encoding="utf-8")
        + "% changed after profile load\n",
        encoding="utf-8",
    )
    model_path.write_bytes(b"changed model identity")

    report = build_compatibility_report(
        profile,
        neutral_classifier_path=neutral_path,
        runtime_capabilities=(LEGACY_EXACT_REFINEMENT_BACKEND,),
    )
    blocker_codes = {
        issue.code
        for issue in report.backend(LEGACY_EXACT_REFINEMENT_BACKEND).blockers
    }

    assert {"parameter_source_changed", "tracking_model_changed"} <= blocker_codes
    assert "neutral_classifier_unbound" in blocker_codes
    assert not report.neutral_classifier_source_bound
    assert report.current_parameter_sha256 != report.parameter_sha256
    assert report.model_references[0].sha256 != profile.model_sha256
    assert report.to_dict()["neutral_classifier"]["source_bound"] is False


def test_ambiguous_active_models_cannot_claim_a_source_bound_export(
    tmp_path: Path,
) -> None:
    first_model = tmp_path / "first.mat"
    second_model = tmp_path / "second.mat"
    first_model.write_bytes(b"first model")
    second_model.write_bytes(b"second model")
    parameter_path = tmp_path / "ambiguous.m"
    parameter_path.write_text(
        "load 'first.mat';\nload 'second.mat';\n",
        encoding="utf-8",
    )
    profile = load_tuning_profile(parameter_path)
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(neutral_path, _neutral_model(profile.model_sha256 or ""))

    report = build_compatibility_report(
        profile,
        neutral_classifier_path=neutral_path,
        runtime_capabilities=(LEGACY_EXACT_REFINEMENT_BACKEND,),
    )
    blocker_codes = {
        issue.code
        for issue in report.backend(LEGACY_EXACT_REFINEMENT_BACKEND).blockers
    }

    assert "tracking_model_ambiguous" in blocker_codes
    assert "neutral_classifier_unbound" in blocker_codes
    assert not report.neutral_classifier_source_bound


def test_backend_selection_never_silently_falls_back(tmp_path: Path) -> None:
    profile, _model_path = _profile(tmp_path)
    report = build_compatibility_report(profile)

    assert select_compatibility_backend(report, NATIVE_FAST_BACKEND) == NATIVE_FAST_BACKEND
    with pytest.raises(StarryNiteCompatibilityError, match="not available"):
        select_compatibility_backend(report, LEGACY_EXACT_REFINEMENT_BACKEND)
    with pytest.raises(StarryNiteCompatibilityError, match="Unknown"):
        select_compatibility_backend(report, "auto")

    with pytest.raises(ValueError, match="separately validated backend"):
        StarryNiteDivisionTracker().track(
            (),
            {"STARRYNITE_COMPATIBILITY_MODE": LEGACY_EXACT_REFINEMENT_BACKEND},
        )


def test_report_accepts_safe_regional_records_but_blocks_dynamic_regions(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tracking-model.mat"
    model_path.write_bytes(b"source model identity")
    safe_path = tmp_path / "safe-regions.m"
    safe_path.write_text(
        "parameters.staging=[25];\n"
        "parameters.intensitythreshold=[4,5];\n"
        "parameters.regions=cell(2,1);\n"
        "parameters.regions{1}.area=[0,10,0,10,0,10];\n"
        "parameters.regions{1}.intensitythreshold=7;\n"
        "load 'tracking-model.mat';\n",
        encoding="utf-8",
    )

    safe = build_compatibility_report(load_tuning_profile(safe_path))

    assert safe.regional_definition_count == 1
    assert len(safe.regional_consumed_record_indices) == 3
    assert not any(
        issue.code == "opaque_parameter_statement"
        and issue.record_index in safe.regional_consumed_record_indices
        for issue in safe.opaque_issues
    )
    json.dumps(safe.to_dict(), allow_nan=False)

    dynamic_path = tmp_path / "dynamic-regions.m"
    dynamic_path.write_text(
        "parameters.staging=[25];\n"
        "parameters.intensitythreshold=[4,5];\n"
        "parameters.regions=cell(size(parameters.staging));\n"
        "load 'tracking-model.mat';\n",
        encoding="utf-8",
    )
    dynamic = build_compatibility_report(load_tuning_profile(dynamic_path))
    assert any(
        issue.code == "regional_parameter_unresolved"
        for issue in dynamic.opaque_issues
    )


def test_report_does_not_require_commented_selection_distance(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tracking-model.mat"
    model_path.write_bytes(b"source model identity")
    parameter_path = tmp_path / "missing-selection-distance.m"
    parameter_path.write_text(
        "parameters.staging=[25];\n"
        "parameters.sigma=1;\n"
        "parameters.intensitythreshold=[4,5];\n"
        "parameters.rangethreshold=[10,8];\n"
        "parameters.boundary_percent=.5;\n"
        "parameters.large_ray_threshold=1.5;\n"
        "parameters.small_ray_threshold=1/3;\n"
        "parameters.mergelower=[-20,-10];\n"
        "parameters.mergesplit=[1,.5];\n"
        "parameters.split=[20,10];\n"
        "parameters.nndist_merge=[.8,.6];\n"
        "parameters.armerge=[1.6,1.3];\n"
        "load 'tracking-model.mat';\n",
        encoding="utf-8",
    )

    report = build_compatibility_report(load_tuning_profile(parameter_path))
    missing = tuple(
        issue
        for issue in report.backend(NATIVE_FAST_BACKEND).issues
        if issue.code == "legacy_detector_parameter_missing"
    )

    assert report.backend(NATIVE_FAST_BACKEND).runnable
    assert missing == ()
    assert "parameters.selection_dist" not in report.format_text()


def test_missing_model_finding_does_not_search_or_substitute_by_basename(
    tmp_path: Path,
) -> None:
    parameter_directory = tmp_path / "parameters"
    unrelated_directory = tmp_path / "distribution_lineaging"
    parameter_directory.mkdir()
    unrelated_directory.mkdir()
    (unrelated_directory / "tracking-model.mat").write_bytes(b"wrong search hit")
    parameter_path = parameter_directory / "parameters.m"
    parameter_path.write_text("load 'tracking-model.mat';\n", encoding="utf-8")

    report = build_compatibility_report(load_tuning_profile(parameter_path))
    exact = report.backend(LEGACY_EXACT_REFINEMENT_BACKEND)
    missing = tuple(
        issue for issue in exact.blockers if issue.code == "tracking_model_missing"
    )

    assert len(missing) == 1
    expected = (parameter_directory / "tracking-model.mat").resolve()
    assert report.model_references[0].resolved_path == expected
    assert not report.model_references[0].exists
    assert "no directory search or basename substitution" in missing[0].message
