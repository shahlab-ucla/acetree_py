"""App-level contracts for the asynchronous initial tracking review."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QWidget
from qtpy.QtCore import QSettings

from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.config import AceTreeConfig
from acetree_py.io.image_provider import NumpyProvider
from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    TrackingRequest,
    TrackingScope,
)
from acetree_py.tracking.registry import build_default_registry
from acetree_py.tracking.starrynite import (
    CategoricalFeatureDistribution,
    GaussianFeatureDistribution,
    NeutralNaiveBayesClassifier,
    SingleModelFeatureLayout,
    save_neutral_classifier,
)


def _blob_movie() -> np.ndarray:
    z, y, x = np.indices((7, 25, 25), dtype=float)
    frames = []
    for cx in (10.0, 11.0):
        frames.append(
            100.0
            * np.exp(
                -((z - 3.0) ** 2 + (y - 10.0) ** 2 + (x - cx) ** 2) / 2.0
            )
        )
    return np.asarray(frames, dtype=np.float32)


def _request() -> TrackingRequest:
    return TrackingRequest(
        detector=ComponentSpec(
            "acetree.dog3d",
            {
                "TARGET_CHANNEL": 1,
                "RADIUS": 1.7,
                "THRESHOLD": 1.0,
                "DO_SUBPIXEL_LOCALIZATION": True,
                "DO_MEDIAN_FILTERING": False,
            },
        ),
        tracker=ComponentSpec(
            "acetree.simple_lap",
            {
                "LINKING_MAX_DISTANCE": 3.0,
                "ALLOW_GAP_CLOSING": True,
                "GAP_CLOSING_MAX_DISTANCE": 4.0,
                "MAX_FRAME_GAP": 2,
                "ALLOW_TRACK_SPLITTING": False,
                "ALLOW_TRACK_MERGING": False,
            },
        ),
        scope=TrackingScope("global", 1, 2),
    )


def _exact_classifier(source_hash: str) -> NeutralNaiveBayesClassifier:
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


def _exact_workbench_assets(tmp_path: Path) -> Path:
    scipy_io = pytest.importorskip("scipy.io")
    covariance = np.eye(7, dtype=np.float64)
    mean = np.zeros((1, 7), dtype=np.float64)
    scipy_io.savemat(
        tmp_path / "detector-distribution.mat",
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
    scipy_io.savemat(
        tmp_path / "tracking-model.mat",
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
                "anisotropyvector": np.asarray([1, 1, 1]),
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
    parameter_path = tmp_path / "exact-parameters.m"
    parameter_path.write_text(
        "xyres=1;\n"
        "zres=1;\n"
        "firsttimestepdiam=8;\n"
        "firsttimestepnumcells=1;\n"
        "downsampling=1;\n"
        "parameters.staging=[25,80];\n"
        "parameters.sigma=.5;\n"
        "parameters.intensitythreshold=.25;\n"
        "parameters.rangethreshold=1;\n"
        "parameters.boundary_percent=.35;\n"
        "parameters.large_ray_threshold=1.5;\n"
        "parameters.small_ray_threshold=.333333333333;\n"
        "parameters.mergelower=-300;\n"
        "parameters.mergesplit=1;\n"
        "parameters.split=100;\n"
        "parameters.nndist_merge=.8;\n"
        "parameters.armerge=1.6;\n"
        "distribution_file='detector-distribution.mat';\n"
        "load 'tracking-model.mat';\n"
        "trackingparameters.nonDivCostFunction=@distanceCostFunction;\n"
        "trackingparameters.DivCostFunction=@divScoreModelCostFunction;\n",
        encoding="utf-8",
    )
    return parameter_path


class _PreviewSpy:
    def __init__(self, app: AceTreeApp) -> None:
        self.app = app
        self.shown = []
        self.cleared = 0
        self.detector_shown = []
        self.detector_cleared = 0

    def show_tracking_preview(self, proposal, calibration, **state) -> None:
        self.shown.append((proposal, calibration, state))

    def clear_tracking_preview(self) -> None:
        self.cleared += 1

    def set_tracking_preview_visible(self, _visible: bool) -> None:
        pass

    def highlight_tracking_preview(self, _preview_id) -> None:
        pass

    def show_detector_preview(self, detections, calibration, **state) -> None:
        self.detector_shown.append((tuple(detections), calibration, state))

    def clear_detector_preview(self) -> None:
        self.detector_cleared += 1

    def set_detector_preview_visible(self, _visible: bool) -> None:
        pass

    def update_overlays(self) -> None:
        pass


def test_exact_workbench_fails_closed_when_movie_and_record_lengths_differ(
    qtbot,
) -> None:
    provider = NumpyProvider(_blob_movie())
    manager = NucleiManager.new_empty(AceTreeConfig(), num_timepoints=3)
    app = AceTreeApp(manager, provider)
    window = QWidget()
    qtbot.addWidget(window)
    app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
    app._image_layers = [SimpleNamespace(data=None, scale=(1.0, 1.0), visible=True)]
    app._viewer_integration = _PreviewSpy(app)

    dialog = app.open_global_tracking_workbench()
    assert dialog is not None
    qtbot.addWidget(dialog)
    dialog._tracker_combo.setCurrentIndex(
        dialog._tracker_combo.findData("acetree.starrynite_legacy_exact")
    )

    error = dialog._settings_validation_error()
    assert "nuclei record has 3 timepoint" in error
    assert "image source has 2" in error
    assert not dialog._preview_button.isEnabled()

    exact_request = TrackingRequest(
        detector=ComponentSpec("acetree.starrynite_detector", {}),
        tracker=ComponentSpec(
            "acetree.starrynite_legacy_exact",
            {"STARRYNITE_COMPATIBILITY_MODE": "legacy_exact_refinement"},
        ),
        scope=TrackingScope("global", 1, 2),
    )
    with pytest.raises(ValueError, match="same number of timepoints"):
        app.prepare_tracking_analysis(exact_request)

    dialog.reject()
    qtbot.waitUntil(lambda: app._global_tracking_dialog is None)


def test_initial_global_analysis_remains_empty_until_explicit_accept(qtbot) -> None:
    provider = NumpyProvider(_blob_movie())
    manager = NucleiManager.new_empty(
        AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=7),
        num_timepoints=2,
    )
    app = AceTreeApp(manager, provider)
    window = QWidget()
    qtbot.addWidget(window)
    app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
    app._image_layers = [
        SimpleNamespace(data=None, scale=(1.0, 1.0), visible=True)
    ]
    app._viewer_integration = _PreviewSpy(app)

    dialog = app.open_global_tracking_workbench(initial_request=_request())
    assert dialog is not None
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY, timeout=10_000)

    assert manager.nuclei_record == [[], []]
    assert app.edit_history.num_undoable == 0
    assert app._viewer_integration.shown

    dialog._accept_button.click()
    qtbot.waitUntil(lambda: app._global_tracking_dialog is None)
    qtbot.waitUntil(lambda: not app._global_tracking_jobs)

    assert sum(len(frame) for frame in manager.nuclei_record) == 2
    assert app.edit_history.num_undoable == 1
    assert len(app._tracking_results) == 1
    assert app._viewer_integration.cleared == 1


def test_exact_workbench_request_runs_accepts_and_restores_sources(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from acetree_py.gui.global_tracking_dialog import GlobalTrackingDialog

    settings_path = tmp_path / "workbench-settings.ini"

    def settings_store() -> QSettings:
        return QSettings(str(settings_path), QSettings.IniFormat)

    monkeypatch.setattr(
        GlobalTrackingDialog,
        "_settings_store",
        staticmethod(settings_store),
    )
    parameter_path = _exact_workbench_assets(tmp_path)
    provider = NumpyProvider(_blob_movie())
    manager = NucleiManager.new_empty(
        AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=7),
        num_timepoints=2,
    )
    app = AceTreeApp(manager, provider)
    window = QWidget()
    qtbot.addWidget(window)
    app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
    app._image_layers = [
        SimpleNamespace(data=None, scale=(1.0, 1.0), visible=True)
    ]
    app._viewer_integration = _PreviewSpy(app)

    dialog = app.open_global_tracking_workbench()
    assert dialog is not None
    qtbot.addWidget(dialog)
    dialog.load_starrynite_parameter_file(str(parameter_path))
    profile = dialog._starrynite_profile
    assert profile is not None and profile.model_sha256 is not None
    classifier_path = tmp_path / "tracking-model.neutral.json"
    save_neutral_classifier(
        classifier_path,
        _exact_classifier(profile.model_sha256),
    )
    dialog.attach_starrynite_neutral_classifier(classifier_path)
    dialog._select_combo_value(
        dialog._tracker_combo,
        "acetree.starrynite_legacy_exact",
    )

    request = dialog.get_request()
    assert request.detector.plugin_id == "acetree.starrynite_detector"
    assert request.tracker.plugin_id == "acetree.starrynite_legacy_exact"
    assert request.tracker.settings["STARRYNITE_MODEL_FILE"] == str(
        (tmp_path / "tracking-model.mat").resolve()
    )
    assert request.tracker.settings["STARRYNITE_NEUTRAL_CLASSIFIER_FILE"] == str(
        classifier_path.resolve()
    )

    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY, timeout=20_000)
    qtbot.waitUntil(lambda: not app._global_tracking_jobs, timeout=20_000)
    assert manager.nuclei_record == [[], []]
    assert dialog.proposal is not None
    refinement = dialog.proposal.provenance["graph_refinement"]
    assert refinement["backend"] == "legacy_exact_refinement"
    assert refinement["event_order_validated"] is True

    proposed_count = len(dialog.proposal.detections)
    assert proposed_count >= 2
    dialog._accept_button.click()
    qtbot.waitUntil(lambda: app._global_tracking_dialog is None, timeout=10_000)
    qtbot.waitUntil(lambda: not app._global_tracking_jobs, timeout=10_000)
    assert sum(len(frame) for frame in manager.nuclei_record) == proposed_count
    assert app.edit_history.num_undoable == 1
    assert len(app._tracking_results) == 1

    registry = build_default_registry(discover_plugins=False)
    restored = GlobalTrackingDialog(
        1,
        2,
        registry=registry,
        calibration=Calibration(1.0, 1.0),
    )
    qtbot.addWidget(restored)
    assert restored.recent_starrynite_parameter_file() == parameter_path.resolve()
    restored.load_starrynite_parameter_file(str(parameter_path))
    assert restored._starrynite_profile is not None
    assert restored._starrynite_profile.model_path == (
        tmp_path / "tracking-model.mat"
    ).resolve()
    assert restored._starrynite_neutral_classifier_path == classifier_path.resolve()
    model_key = restored._neutral_classifier_settings_key()
    assert model_key is not None
    assert Path(str(settings_store().value(model_key))).resolve() == (
        classifier_path.resolve()
    )
    assert refinement["event_order_validation_scope"] == (
        "structural_sequence_trace_coverage_and_unique_frame_row_order"
    )
    assert refinement["event_omission_completeness_proven"] is False


def test_async_current_frame_detector_preview_is_lightweight_and_non_mutating(
    qtbot,
) -> None:
    provider = NumpyProvider(_blob_movie())
    manager = NucleiManager.new_empty(
        AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=7),
        num_timepoints=2,
    )
    app = AceTreeApp(manager, provider)
    app.current_time = 2
    window = QWidget()
    qtbot.addWidget(window)
    app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
    app._image_layers = [
        SimpleNamespace(data=None, scale=(1.0, 1.0), visible=True)
    ]
    app._viewer_integration = _PreviewSpy(app)
    snapshot = app.prepare_detector_preview(_request().detector, 2)
    assert not hasattr(snapshot, "nuclei_record")

    dialog = app.open_global_tracking_workbench(initial_request=_request())
    assert dialog is not None
    qtbot.addWidget(dialog)
    dialog._detector_preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.DETECTOR_READY, timeout=10_000)
    qtbot.waitUntil(lambda: not app._global_tracking_jobs, timeout=10_000)

    assert manager.nuclei_record == [[], []]
    assert app.edit_history.num_undoable == 0
    assert app._tracking_results == []
    assert dialog.proposal is None
    assert not dialog._accept_button.isEnabled()
    assert len(app._viewer_integration.detector_shown) == 1
    detections, _calibration, state = app._viewer_integration.detector_shown[0]
    assert detections
    assert {detection.frame for detection in detections} == {2}
    assert state["visible"] is True

    dialog.reject()
    qtbot.waitUntil(lambda: app._global_tracking_dialog is None)
    assert manager.nuclei_record == [[], []]


def test_closing_global_workbench_cancels_worker_without_late_commit(qtbot) -> None:
    from acetree_py.tracking.pipeline import TrackingCancelled

    provider = NumpyProvider(_blob_movie())
    manager = NucleiManager.new_empty(AceTreeConfig(), num_timepoints=2)
    app = AceTreeApp(manager, provider)
    window = QWidget()
    qtbot.addWidget(window)
    app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
    app._image_layers = [SimpleNamespace(data=None, scale=(1.0, 1.0), visible=True)]
    app._viewer_integration = _PreviewSpy(app)
    started = threading.Event()

    def slow_analysis(_snapshot, *, progress=None, cancelled=None):
        started.set()
        while cancelled is None or not cancelled():
            time.sleep(0.005)
        raise TrackingCancelled("cancelled")

    app.analyze_prepared_tracking = slow_analysis
    dialog = app.open_global_tracking_workbench(initial_request=_request())
    assert dialog is not None
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(started.is_set)

    dialog.reject()
    qtbot.waitUntil(lambda: app._global_tracking_dialog is None)
    qtbot.waitUntil(lambda: not app._global_tracking_jobs, timeout=10_000)
    qtbot.waitUntil(lambda: not app._global_tracking_retiring_jobs, timeout=10_000)

    assert manager.nuclei_record == [[], []]
    assert app.edit_history.num_undoable == 0
    assert app._viewer_integration.cleared == 1


def test_canceled_global_worker_must_finish_before_reopen(qtbot, monkeypatch) -> None:
    from qtpy.QtWidgets import QMessageBox

    from acetree_py.tracking.pipeline import TrackingCancelled

    provider = NumpyProvider(_blob_movie())
    manager = NucleiManager.new_empty(AceTreeConfig(), num_timepoints=2)
    app = AceTreeApp(manager, provider)
    window = QWidget()
    qtbot.addWidget(window)
    app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
    app._image_layers = [SimpleNamespace(data=None, scale=(1.0, 1.0), visible=True)]
    app._viewer_integration = _PreviewSpy(app)
    started = threading.Event()
    release = threading.Event()

    def delayed_cancel(_snapshot, *, progress=None, cancelled=None):
        started.set()
        while cancelled is None or not cancelled():
            time.sleep(0.005)
        release.wait(2.0)
        raise TrackingCancelled("cancelled")

    app.analyze_prepared_tracking = delayed_cancel
    dialog = app.open_global_tracking_workbench(initial_request=_request())
    assert dialog is not None
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(started.is_set)
    dialog.reject()
    qtbot.waitUntil(lambda: app._global_tracking_dialog is None)

    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args: messages.append(args) or QMessageBox.Ok,
    )
    assert app.open_global_tracking_workbench(initial_request=_request()) is None
    assert messages

    release.set()
    qtbot.waitUntil(lambda: not app._global_tracking_jobs, timeout=10_000)
