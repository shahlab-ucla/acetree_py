"""App-level contracts for the asynchronous initial tracking review."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QWidget

from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.config import AceTreeConfig
from acetree_py.io.image_provider import NumpyProvider
from acetree_py.tracking.api import ComponentSpec, TrackingRequest, TrackingScope


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
