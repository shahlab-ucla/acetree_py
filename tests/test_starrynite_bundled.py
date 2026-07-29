from __future__ import annotations

from pathlib import Path

from acetree_py.tracking.starrynite import (
    build_compatibility_report,
    bundled_classifier_for_profile,
    bundled_legacy_model_sources,
    bundled_parameter_presets,
    bundled_tracking_models,
    legacy_model_export_script,
    load_tuning_profile,
    native_sparse_detector_settings,
    sha256_file,
    validate_bundled_assets,
)
from acetree_py.tracking.workflows import (
    FORWARD_TRACKING_WORKFLOWS,
    GLOBAL_TRACKING_WORKFLOWS,
    LEGACY_STARRYNITE_EXACT,
    MODERN_STARRYNITE,
    workflow_for_components,
)


EXPECTED_NEWMATLAB_PARAMETERS = {
    "dispim_singleview_param.txt",
    "dispim_mipav_decon_param.txt",
    "dispim_mipav_decon_param_dispim_model.txt",
    "iSIM_red_40xs.txt",
    "SD_red_40x.txt",
    "SD_red_60xSI.txt",
}


def test_all_upstream_newmatlab_presets_are_install_ready() -> None:
    presets = bundled_parameter_presets()

    assert validate_bundled_assets() == ()
    assert {preset.filename for preset in presets} == EXPECTED_NEWMATLAB_PARAMETERS
    assert len({preset.preset_id for preset in presets}) == len(presets)

    for preset in presets:
        source = preset.parameter_file.read_text(encoding="utf-8")
        assert "l:\\" not in source.lower()

        profile = load_tuning_profile(preset.parameter_file)
        assert profile.model_path is not None and profile.model_path.is_file()
        distribution = Path(
            profile.detector_settings["STARRYNITE_DISTRIBUTION_FILE"]
        )
        assert distribution.is_file()
        sparse_settings = native_sparse_detector_settings(profile.detector_settings)
        assert sparse_settings["STARRYNITE_DISTRIBUTION_FILE"] == ""
        assert sparse_settings["STARRYNITE_DISTRIBUTION_SOURCE_SHA256"] == ""
        assert sparse_settings["STARRYNITE_PARAMETER_FILE"] == str(
            preset.parameter_file.resolve()
        )
        assert sparse_settings["STARRYNITE_PARAMETER_SHA256"]

        runtime_model = bundled_classifier_for_profile(profile)
        assert runtime_model is not None and runtime_model.suffix == ".atpy-model"
        report = build_compatibility_report(
            profile,
            neutral_classifier_path=runtime_model,
        )
        assert report.neutral_classifier_source_bound


def test_every_compatible_upstream_model_source_is_preserved() -> None:
    ready_models = bundled_tracking_models()
    assert {model.mat_filename for model in ready_models} == {
        "2019TrackingModelv2.mat",
        "gaussianlatedispimmodel_withoptimizedfeatures_ignoringFPstillpoorFN.mat",
    }
    for model in ready_models:
        assert sha256_file(model.mat_file) == model.source_sha256
        assert model.runtime_file.is_file()

    legacy_models = bundled_legacy_model_sources()
    assert {model.mat_filename for model in legacy_models} == {
        "clean_red_singlemodel_red_normal.mat"
    }
    for model in legacy_models:
        assert sha256_file(model.mat_file) == model.source_sha256
        assert "older MATLAB" in model.compatibility_note
    assert legacy_model_export_script().is_file()


def test_named_workflows_expose_exact_only_for_whole_movie_tracking() -> None:
    assert MODERN_STARRYNITE in GLOBAL_TRACKING_WORKFLOWS
    assert MODERN_STARRYNITE in FORWARD_TRACKING_WORKFLOWS
    assert LEGACY_STARRYNITE_EXACT in GLOBAL_TRACKING_WORKFLOWS
    assert LEGACY_STARRYNITE_EXACT not in FORWARD_TRACKING_WORKFLOWS
    assert workflow_for_components(
        "acetree.starrynite_detector",
        "acetree.starrynite_division",
    ) is MODERN_STARRYNITE
