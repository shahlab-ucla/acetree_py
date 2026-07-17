"""Practical compatibility reporting and parameter-session UI tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from acetree_py.gui.auto_tracking_dialog import AutoTrackForwardDialog
from acetree_py.gui.global_tracking_dialog import GlobalTrackingDialog
from acetree_py.tracking.api import Calibration, ComponentSpec
from acetree_py.tracking.registry import build_default_registry
from acetree_py.tracking.starrynite import (
    CategoricalFeatureDistribution,
    GaussianFeatureDistribution,
    LEGACY_EXACT_REFINEMENT_BACKEND,
    NATIVE_FAST_BACKEND,
    NeutralNaiveBayesClassifier,
    SingleModelFeatureLayout,
    save_neutral_classifier,
)


def _parameter_file(tmp_path: Path) -> Path:
    (tmp_path / "tracking.mat").write_bytes(b"model provenance")
    path = tmp_path / "parameters.m"
    path.write_text(
        "firsttimestepnumcells=30;\n"
        "xyres=.25;\n"
        "firsttimestepdiam=32;\n"
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[5,8,12];\n"
        "trackingparameters.temporalcutoff=2;\n"
        "load 'tracking.mat';\n",
        encoding="utf-8",
    )
    return path


def _neutral_model(source_hash: str) -> NeutralNaiveBayesClassifier:
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


def _exact_parameter_file(tmp_path: Path) -> Path:
    scipy_io = pytest.importorskip("scipy.io")
    model_path = tmp_path / "exact-tracking.mat"
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
    covariance = np.eye(7, dtype=np.float64)
    mean = np.zeros((1, 7), dtype=np.float64)
    scipy_io.savemat(
        tmp_path / "exact-distribution.mat",
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
    path = tmp_path / "exact-parameters.m"
    path.write_text(
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
        "firsttimestepnumcells=1;\n"
        "firsttimestepdiam=8;\n"
        "downsampling=1;\n"
        "xyres=.25;\n"
        "zres=1;\n"
        "distribution_file='exact-distribution.mat';\n"
        "load 'exact-tracking.mat';\n"
        "trackingparameters.nonDivCostFunction=@distanceCostFunction;\n"
        "trackingparameters.DivCostFunction=@divScoreModelCostFunction;\n",
        encoding="utf-8",
    )
    return path


def test_global_parameter_wizard_exposes_report_and_native_run_identity(
    qtbot,
    tmp_path: Path,
) -> None:
    dialog = GlobalTrackingDialog(
        1,
        3,
        registry=build_default_registry(discover_plugins=False),
    )
    qtbot.addWidget(dialog)
    dialog.load_starrynite_parameter_file(str(_parameter_file(tmp_path)))

    report = dialog._starrynite_compatibility_report
    assert report is not None
    assert report.backend(NATIVE_FAST_BACKEND).runnable
    assert not report.backend(LEGACY_EXACT_REFINEMENT_BACKEND).runnable
    assert dialog._starrynite_report_button.isEnabled()
    assert dialog._starrynite_neutral_button.isEnabled()
    assert (
        "another legacy model"
        in dialog._starrynite_neutral_button.text().lower()
    )
    assert "native tracker remains unchanged" in (
        dialog._starrynite_neutral_button.toolTip().lower()
    )
    assert (
        dialog.get_request().tracker.settings["STARRYNITE_COMPATIBILITY_MODE"]
        == NATIVE_FAST_BACKEND
    )
    assert dialog.export_settings()["starrynite_compatibility_backend"] == NATIVE_FAST_BACKEND


def test_global_wizard_builds_source_bound_exact_request_and_hides_it_from_auto(
    qtbot,
    tmp_path: Path,
) -> None:
    registry = build_default_registry(discover_plugins=False)
    current_frame = [2]
    dialog = GlobalTrackingDialog(
        1,
        2,
        registry=registry,
        calibration=Calibration(0.25, 1.0),
        current_frame_getter=lambda: current_frame[0],
    )
    qtbot.addWidget(dialog)
    dialog.load_starrynite_parameter_file(str(_exact_parameter_file(tmp_path)))
    profile = dialog._starrynite_profile
    assert profile is not None
    dialog._select_combo_value(
        dialog._tracker_combo,
        "acetree.starrynite_legacy_exact",
    )
    assert "Exact mode is blocked" in dialog._starrynite_file_label.text()
    assert "Choose a bundled preset" in dialog._starrynite_file_label.text()
    assert (
        "another legacy model"
        in dialog._starrynite_neutral_button.text().lower()
    )
    assert not dialog._detector_preview_button.isEnabled()
    assert "sequential" in dialog._detector_preview_button.toolTip().lower()
    assert not dialog._preview_button.isEnabled()

    classifier_path = tmp_path / "exact-classifier.json"
    save_neutral_classifier(
        classifier_path,
        _neutral_model(profile.model_sha256 or ""),
    )
    dialog.attach_starrynite_neutral_classifier(classifier_path)

    assert dialog._settings_validation_error() == ""
    assert dialog._preview_button.isEnabled()
    assert not dialog._detector_preview_button.isEnabled()
    current_frame[0] = 1
    dialog.sync_viewer_position(1)
    assert dialog._detector_preview_button.isEnabled()
    current_frame[0] = 2
    dialog.sync_viewer_position(2)
    assert not dialog._detector_preview_button.isEnabled()
    request = dialog.get_request()
    assert request.detector.plugin_id == "acetree.starrynite_detector"
    assert request.tracker.plugin_id == "acetree.starrynite_legacy_exact"
    assert (
        request.tracker.settings["STARRYNITE_COMPATIBILITY_MODE"]
        == LEGACY_EXACT_REFINEMENT_BACKEND
    )
    assert request.tracker.settings["STARRYNITE_NEUTRAL_CLASSIFIER_FILE"] == str(
        classifier_path.resolve()
    )
    assert (
        dialog.export_settings()["starrynite_compatibility_backend"]
        == LEGACY_EXACT_REFINEMENT_BACKEND
    )
    assert (
        "another legacy model"
        in dialog._starrynite_neutral_button.text().lower()
    )

    auto = AutoTrackForwardDialog(1, 2, seed_anchor=(1, 1))
    qtbot.addWidget(auto)
    assert auto._tracker_combo.findData("acetree.starrynite_legacy_exact") == -1


def test_auto_forward_restores_editable_parameter_session_without_losing_tuning(
    qtbot,
    tmp_path: Path,
) -> None:
    parameter_path = _parameter_file(tmp_path)
    first = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
    qtbot.addWidget(first)
    first.load_starrynite_parameter_file(str(parameter_path), cell_count=30)
    first._radius_spin.setValue(6.25)
    first._threshold_spin.setValue(17.5)
    settings = first.export_settings()

    restored = AutoTrackForwardDialog(
        1,
        3,
        seed_anchor=(1, 1),
        initial_settings=settings,
    )
    qtbot.addWidget(restored)

    assert restored._starrynite_profile is not None
    assert restored._starrynite_compatibility_report is not None
    assert restored._starrynite_save_button.isEnabled()
    assert restored._starrynite_report_button.isEnabled()
    assert restored._radius_spin.value() == pytest.approx(6.25)
    assert restored._threshold_spin.value() == pytest.approx(17.5)
    assert "Restored parameter session" in restored._starrynite_file_label.text()

    bad_export = tmp_path / "not-a-classifier.json"
    bad_export.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not usable"):
        restored.attach_starrynite_neutral_classifier(bad_export)
    assert restored._starrynite_neutral_classifier_path is None
    assert restored._starrynite_compatibility_report is not None

    restored._set_running(True)
    assert not restored._starrynite_neutral_button.isEnabled()
    assert not restored._starrynite_report_button.isEnabled()
    assert not restored._starrynite_save_button.isEnabled()
    restored._set_running(False)
    assert restored._starrynite_neutral_button.isEnabled()
    assert restored._starrynite_report_button.isEnabled()
    assert restored._starrynite_save_button.isEnabled()

    restored._restore_defaults()
    assert restored._starrynite_profile is None
    assert restored._starrynite_compatibility_report is None
    assert not restored._starrynite_neutral_button.isEnabled()
    assert not restored._starrynite_report_button.isEnabled()
    assert not restored._starrynite_save_button.isEnabled()


def test_auto_forward_rebases_source_metadata_but_keeps_visible_tuning(
    qtbot,
    tmp_path: Path,
) -> None:
    parameter_path = _parameter_file(tmp_path)
    first = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
    qtbot.addWidget(first)
    first.load_starrynite_parameter_file(str(parameter_path), cell_count=30)
    first._radius_spin.setValue(6.25)
    first._threshold_spin.setValue(17.5)
    settings = first.export_settings()

    parameter_path.write_text(
        parameter_path.read_text(encoding="utf-8").replace(
            "firsttimestepnumcells=30",
            "firsttimestepnumcells=90",
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "tracking.mat"
    model_path.write_bytes(b"model provenance changed")

    restored = AutoTrackForwardDialog(
        1,
        3,
        seed_anchor=(1, 1),
        initial_settings=settings,
    )
    qtbot.addWidget(restored)
    request = restored.get_request()

    assert restored._radius_spin.value() == pytest.approx(6.25)
    assert restored._threshold_spin.value() == pytest.approx(17.5)
    assert request.detector.settings["STARRYNITE_CELL_COUNT"] == 90
    assert request.detector.settings["STARRYNITE_STAGE_INDEX"] == 2
    assert request.detector.settings["STARRYNITE_PARAMETER_SHA256"] == (
        hashlib.sha256(parameter_path.read_bytes()).hexdigest()
    )
    assert request.tracker.settings["STARRYNITE_MODEL_SHA256"] == (
        hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    assert "Source/model changed" in restored._starrynite_file_label.text()


def test_auto_forward_unavailable_profile_never_reenables_save(qtbot, tmp_path: Path) -> None:
    parameter_path = _parameter_file(tmp_path)
    first = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
    qtbot.addWidget(first)
    first.load_starrynite_parameter_file(str(parameter_path), cell_count=30)
    settings = first.export_settings()
    parameter_path.unlink()

    restored = AutoTrackForwardDialog(
        1,
        3,
        seed_anchor=(1, 1),
        initial_settings=settings,
    )
    qtbot.addWidget(restored)
    assert restored._starrynite_profile is None
    assert not restored._starrynite_save_button.isEnabled()
    restored._set_running(True)
    restored._set_running(False)
    assert not restored._starrynite_save_button.isEnabled()
    assert not restored._starrynite_neutral_button.isEnabled()
    assert not restored._starrynite_report_button.isEnabled()


def test_classifier_association_restores_only_while_source_bound(
    qtbot,
    tmp_path: Path,
) -> None:
    parameter_path = _parameter_file(tmp_path)
    first = GlobalTrackingDialog(
        1,
        3,
        registry=build_default_registry(discover_plugins=False),
    )
    qtbot.addWidget(first)
    first.load_starrynite_parameter_file(str(parameter_path))
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(
        neutral_path,
        _neutral_model(first._starrynite_profile.model_sha256 or ""),
    )
    first.attach_starrynite_neutral_classifier(neutral_path)

    restored = GlobalTrackingDialog(
        1,
        3,
        registry=build_default_registry(discover_plugins=False),
    )
    qtbot.addWidget(restored)
    restored.load_starrynite_parameter_file(str(parameter_path))
    assert restored._starrynite_neutral_classifier_path == neutral_path.resolve()
    assert "reporting only" in restored._starrynite_file_label.text()

    neutral_path.unlink()
    stale = GlobalTrackingDialog(
        1,
        3,
        registry=build_default_registry(discover_plugins=False),
    )
    qtbot.addWidget(stale)
    stale.load_starrynite_parameter_file(str(parameter_path))
    assert stale._starrynite_neutral_classifier_path is None
    assert "Classifier not restored" in stale._starrynite_file_label.text()


def test_ambiguous_model_family_and_controls_fail_closed(qtbot, tmp_path: Path) -> None:
    first_model = tmp_path / "first.mat"
    second_model = tmp_path / "second.mat"
    first_model.write_bytes(b"first UI model")
    second_model.write_bytes(b"second UI model")
    parameter_path = tmp_path / "ambiguous.m"
    parameter_path.write_text(
        "load 'first.mat';\nload 'second.mat';\n",
        encoding="utf-8",
    )
    dialog = GlobalTrackingDialog(
        1,
        3,
        registry=build_default_registry(discover_plugins=False),
    )
    qtbot.addWidget(dialog)
    dialog.load_starrynite_parameter_file(str(parameter_path))
    neutral_path = tmp_path / "ambiguous-classifier.json"
    save_neutral_classifier(
        neutral_path,
        _neutral_model(dialog._starrynite_profile.model_sha256 or ""),
    )

    with pytest.raises(ValueError, match="Multiple active tracking model"):
        dialog.attach_starrynite_neutral_classifier(neutral_path)
    assert dialog._starrynite_neutral_classifier_path is None

    dialog._set_running(True)
    assert not dialog._starrynite_neutral_button.isEnabled()
    assert not dialog._starrynite_report_button.isEnabled()
    assert not dialog._starrynite_save_button.isEnabled()
    dialog._set_running(False)
    assert dialog._starrynite_neutral_button.isEnabled()
    assert dialog._starrynite_report_button.isEnabled()
    assert dialog._starrynite_save_button.isEnabled()

    dialog._restore_defaults()
    assert dialog._starrynite_profile is None
    assert dialog._starrynite_compatibility_report is None
    assert dialog._starrynite_neutral_classifier_path is None
    assert not dialog._starrynite_neutral_button.isEnabled()
    assert not dialog._starrynite_report_button.isEnabled()
    assert not dialog._starrynite_save_button.isEnabled()


def test_persisted_exact_backend_is_rejected_instead_of_coerced_to_native(
    qtbot,
    tmp_path: Path,
) -> None:
    parameter_path = _parameter_file(tmp_path)
    registry = build_default_registry(discover_plugins=False)
    source = GlobalTrackingDialog(1, 3, registry=registry)
    qtbot.addWidget(source)
    source.load_starrynite_parameter_file(str(parameter_path))
    native_request = source.get_request()
    exact_settings = dict(native_request.tracker.settings)
    exact_settings["STARRYNITE_COMPATIBILITY_MODE"] = (
        LEGACY_EXACT_REFINEMENT_BACKEND
    )
    exact_request = replace(
        native_request,
        tracker=ComponentSpec(native_request.tracker.plugin_id, exact_settings),
    )

    with pytest.raises(ValueError, match="supports only.*native_fast"):
        GlobalTrackingDialog(
            1,
            3,
            registry=registry,
            initial_request=exact_request,
        )

    auto_source = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
    qtbot.addWidget(auto_source)
    auto_settings = {
        **auto_source.export_settings(),
        "starrynite_compatibility_backend": LEGACY_EXACT_REFINEMENT_BACKEND,
    }
    with pytest.raises(ValueError, match="supports only.*native_fast"):
        AutoTrackForwardDialog(
            1,
            3,
            seed_anchor=(1, 1),
            initial_settings=auto_settings,
        )
    nested_exact_settings = {
        **auto_source.export_settings(),
        "starrynite_tracker_settings": {
            "STARRYNITE_COMPATIBILITY_MODE": LEGACY_EXACT_REFINEMENT_BACKEND,
        },
    }
    with pytest.raises(ValueError, match="supports only.*native_fast"):
        AutoTrackForwardDialog(
            1,
            3,
            seed_anchor=(1, 1),
            initial_settings=nested_exact_settings,
        )


@pytest.mark.parametrize("dialog_kind", ("global", "auto"))
def test_invalid_classifier_selection_keeps_previous_valid_attachment_transactionally(
    qtbot,
    tmp_path: Path,
    dialog_kind: str,
) -> None:
    parameter_path = _parameter_file(tmp_path)
    (tmp_path / "tracking.mat").write_bytes(
        f"transactional classifier {dialog_kind}".encode()
    )
    if dialog_kind == "global":
        dialog = GlobalTrackingDialog(
            1,
            3,
            registry=build_default_registry(discover_plugins=False),
        )
    else:
        dialog = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog.load_starrynite_parameter_file(str(parameter_path))
    source_hash = dialog._starrynite_profile.model_sha256 or ""
    first_path = tmp_path / "first-valid.json"
    second_path = tmp_path / "second-valid.json"
    invalid_path = tmp_path / "invalid.json"
    save_neutral_classifier(first_path, _neutral_model(source_hash))
    save_neutral_classifier(second_path, _neutral_model(source_hash))
    invalid_path.write_text("{}", encoding="utf-8")
    dialog.attach_starrynite_neutral_classifier(first_path)

    with pytest.raises(ValueError, match="not usable"):
        dialog.attach_starrynite_neutral_classifier(invalid_path)

    assert dialog._starrynite_neutral_classifier_path == first_path.resolve()
    assert dialog._remembered_neutral_classifier(
        dialog._starrynite_profile
    ) == first_path.resolve()
    assert first_path.name in dialog._starrynite_file_label.text()
    assert invalid_path.name not in dialog._starrynite_file_label.text()

    dialog.attach_starrynite_neutral_classifier(second_path)
    assert dialog._starrynite_neutral_classifier_path == second_path.resolve()
    assert second_path.name in dialog._starrynite_file_label.text()
    assert first_path.name not in dialog._starrynite_file_label.text()


@pytest.mark.parametrize("dialog_kind", ("global", "auto"))
def test_run_export_and_details_revalidate_compatibility_sources(
    qtbot,
    tmp_path: Path,
    monkeypatch,
    dialog_kind: str,
) -> None:
    parameter_path = _parameter_file(tmp_path)
    (tmp_path / "tracking.mat").write_bytes(
        f"live compatibility source {dialog_kind}".encode()
    )
    if dialog_kind == "global":
        dialog = GlobalTrackingDialog(
            1,
            3,
            registry=build_default_registry(discover_plugins=False),
        )
        message_box_target = (
            "acetree_py.gui.global_tracking_dialog.QMessageBox.exec"
        )
    else:
        dialog = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
        message_box_target = "acetree_py.gui.auto_tracking_dialog.QMessageBox.exec"
    qtbot.addWidget(dialog)
    dialog.load_starrynite_parameter_file(str(parameter_path))
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(
        neutral_path,
        _neutral_model(dialog._starrynite_profile.model_sha256 or ""),
    )
    dialog.attach_starrynite_neutral_classifier(neutral_path)
    (tmp_path / "tracking.mat").write_bytes(b"changed after attachment")

    dialog.get_request()

    assert dialog._starrynite_neutral_classifier_path is None
    assert "Classifier binding changed" in dialog._starrynite_file_label.text()
    exported = dialog.export_settings()
    assert exported["starrynite_neutral_classifier_file"] == ""
    blockers = {
        issue.code
        for issue in dialog._starrynite_compatibility_report.backend(
            LEGACY_EXACT_REFINEMENT_BACKEND
        ).blockers
    }
    assert "tracking_model_changed" in blockers

    parameter_path.write_text(
        parameter_path.read_text(encoding="utf-8") + "% changed on disk\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(message_box_target, lambda _message: 0)
    dialog._show_starrynite_compatibility_report()
    refreshed_blockers = {
        issue.code
        for issue in dialog._starrynite_compatibility_report.backend(
            LEGACY_EXACT_REFINEMENT_BACKEND
        ).blockers
    }
    assert "parameter_source_changed" in refreshed_blockers


@pytest.mark.parametrize("dialog_kind", ("global", "auto"))
def test_public_export_detaches_classifier_file_modified_after_validation(
    qtbot,
    tmp_path: Path,
    dialog_kind: str,
) -> None:
    parameter_path = _parameter_file(tmp_path)
    (tmp_path / "tracking.mat").write_bytes(
        f"classifier export refresh {dialog_kind}".encode()
    )
    if dialog_kind == "global":
        dialog = GlobalTrackingDialog(
            1,
            3,
            registry=build_default_registry(discover_plugins=False),
        )
    else:
        dialog = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog.load_starrynite_parameter_file(str(parameter_path))
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(
        neutral_path,
        _neutral_model(dialog._starrynite_profile.model_sha256 or ""),
    )
    dialog.attach_starrynite_neutral_classifier(neutral_path)
    neutral_path.write_text("{}", encoding="utf-8")

    exported = dialog.export_settings()

    assert exported["starrynite_neutral_classifier_file"] == ""
    assert dialog._starrynite_neutral_classifier_path is None
    assert "Classifier binding changed" in dialog._starrynite_file_label.text()


def test_ambiguous_profile_does_not_erase_single_model_classifier_association(
    qtbot,
    tmp_path: Path,
) -> None:
    first_model = tmp_path / "first.mat"
    second_model = tmp_path / "second.mat"
    first_model.write_bytes(b"unique first association model")
    second_model.write_bytes(b"unique second association model")
    single_path = tmp_path / "single.m"
    single_path.write_text("load 'first.mat';\n", encoding="utf-8")
    ambiguous_path = tmp_path / "ambiguous.m"
    ambiguous_path.write_text(
        "load 'first.mat';\nload 'second.mat';\n",
        encoding="utf-8",
    )
    registry = build_default_registry(discover_plugins=False)
    first = GlobalTrackingDialog(1, 3, registry=registry)
    qtbot.addWidget(first)
    first.load_starrynite_parameter_file(str(single_path))
    neutral_path = tmp_path / "classifier.json"
    save_neutral_classifier(
        neutral_path,
        _neutral_model(first._starrynite_profile.model_sha256 or ""),
    )
    first.attach_starrynite_neutral_classifier(neutral_path)

    ambiguous = GlobalTrackingDialog(1, 3, registry=registry)
    qtbot.addWidget(ambiguous)
    ambiguous.load_starrynite_parameter_file(str(ambiguous_path))
    assert ambiguous._neutral_classifier_settings_key(
        ambiguous._starrynite_profile
    ) is None

    restored = GlobalTrackingDialog(1, 3, registry=registry)
    qtbot.addWidget(restored)
    restored.load_starrynite_parameter_file(str(single_path))
    assert restored._starrynite_neutral_classifier_path == neutral_path.resolve()


@pytest.mark.parametrize("dialog_kind", ("global", "auto"))
def test_loaded_preset_is_neutralized_when_non_starrynite_components_are_selected(
    qtbot,
    tmp_path: Path,
    dialog_kind: str,
) -> None:
    parameter_path = _parameter_file(tmp_path)
    registry = build_default_registry(discover_plugins=False)
    if dialog_kind == "global":
        dialog = GlobalTrackingDialog(1, 3, registry=registry)
    else:
        dialog = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
        registry = dialog._registry
    qtbot.addWidget(dialog)
    dialog.load_starrynite_parameter_file(str(parameter_path))
    non_starry_detector = next(
        descriptor.plugin_id
        for descriptor in registry.detector_descriptors()
        if descriptor.plugin_id != "acetree.starrynite_detector"
    )
    non_starry_tracker = next(
        descriptor.plugin_id
        for descriptor in registry.tracker_descriptors()
        if descriptor.plugin_id != "acetree.starrynite_division"
    )
    dialog._select_combo_value(dialog._detector_combo, non_starry_detector)
    dialog._select_combo_value(dialog._tracker_combo, non_starry_tracker)

    assert "Inactive preset" in dialog._starrynite_file_label.text()
    assert "native geometry scoring remains active" not in (
        dialog._starrynite_file_label.text()
    )
    assert dialog.export_settings()["starrynite_compatibility_backend"] is None
    if dialog_kind == "global":
        assert dialog._starrynite_behavior_label.isHidden()


def test_compatibility_backend_export_is_scoped_to_starrynite_tracker(qtbot) -> None:
    registry = build_default_registry(discover_plugins=False)
    global_dialog = GlobalTrackingDialog(1, 3, registry=registry)
    qtbot.addWidget(global_dialog)
    non_starry = next(
        descriptor.plugin_id
        for descriptor in registry.tracker_descriptors()
        if descriptor.plugin_id != "acetree.starrynite_division"
    )
    global_dialog._select_combo_value(global_dialog._tracker_combo, non_starry)
    assert global_dialog.export_settings()["starrynite_compatibility_backend"] is None

    auto_dialog = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
    qtbot.addWidget(auto_dialog)
    auto_dialog._select_combo_value(auto_dialog._tracker_combo, non_starry)
    assert auto_dialog.export_settings()["starrynite_compatibility_backend"] is None
    assert "report only" in auto_dialog._starrynite_neutral_button.text()
    assert "â" not in global_dialog._starrynite_neutral_button.text()
