"""Regression tests for curated centroid markers in native 3D viewers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from acetree_py.core.nucleus import Nucleus  # noqa: E402
from acetree_py.editing.commands import MoveNucleus  # noqa: E402
from acetree_py.editing.history import EditHistory  # noqa: E402
from acetree_py.gui.app import AceTreeApp  # noqa: E402
from acetree_py.gui.viewer_3d_window import Viewer3DWindow  # noqa: E402


class _Layer:
    def __init__(self, data, **properties):
        self.data = np.asarray(data, dtype=float)
        self.editable = properties.pop("editable", True)
        self.selected_data = {0}
        self.mode = "select"
        self.mouse_drag_callbacks = []
        self.features = {}
        self.text = {}
        for name, value in properties.items():
            setattr(self, name, value)


class _Viewer:
    def __init__(self):
        self.layers = []

    def add_points(self, data, **properties):
        layer = _Layer(data, **properties)
        self.layers.append(layer)
        return layer


class _FailingAppend(list):
    def append(self, _value):
        raise RuntimeError("simulated callback registration failure")


class _ConfigurationFailureLayer(_Layer):
    def __init__(self, data, *, failure, **properties):
        self._configuration_failure = failure
        self._text = {}
        super().__init__(data, **properties)
        if failure == "callback":
            self.mouse_drag_callbacks = _FailingAppend()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        if self._configuration_failure == "text" and value:
            raise RuntimeError("simulated text configuration failure")


class _ConfigurationFailureViewer(_Viewer):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    def add_points(self, data, **properties):
        layer = _ConfigurationFailureLayer(
            data,
            failure=self.failure,
            **properties,
        )
        self.layers.append(layer)
        return layer


class _Manager:
    def __init__(self, nuclei):
        self.nuclei = nuclei
        self.z_pix_res = 2.0
        self.config = SimpleNamespace(plane_start=1)

    def alive_nuclei_at(self, _time):
        return list(self.nuclei)


class _ColorEngine:
    @staticmethod
    def colors_for_frame(nuclei, *_args, **_kwargs):
        return [[0.5, 0.25, 1.0, 1.0] for _ in nuclei]


def _main_harness(nuclei):
    callback = object()
    trail_calls = []
    harness = SimpleNamespace(
        viewer=_Viewer(),
        manager=_Manager(nuclei),
        current_time=1,
        current_cell_name="",
        _viz_mode=False,
        _points_layer=None,
        _viewer_integration=SimpleNamespace(
            _shown_labels=set(),
            _labels_global_visible=True,
        ),
        _on_3d_click=callback,
        _update_3d_trail=lambda: trail_calls.append("trail"),
        stack_z_from_plane=lambda value: float(value) - 1.0,
        _make_curated_points_read_only=AceTreeApp._make_curated_points_read_only,
    )
    return harness, callback, trail_calls


def test_main_curated_points_remain_locked_across_centroid_redraws():
    nucleus = Nucleus(index=1, x=10, y=20, z=3.0, size=8, status=1)
    app, callback, trail_calls = _main_harness([nucleus])

    AceTreeApp._update_3d_points(app)

    layer = app._points_layer
    assert layer.editable is False
    assert layer.selected_data == set()
    assert layer.mode == "pan_zoom"
    assert layer.mouse_drag_callbacks == [callback]
    np.testing.assert_allclose(layer.data, [[2.0, 20.0, 10.0]])

    # Re-lock on every redraw so an extension or native napari action cannot
    # leave this record-derived layer in an editing mode.
    layer.editable = True
    layer.selected_data = {0}
    layer.mode = "select"
    nucleus.x = 14
    AceTreeApp._update_3d_points(app)

    assert layer.editable is False
    assert layer.selected_data == set()
    assert layer.mode == "pan_zoom"
    assert layer.mouse_drag_callbacks == [callback]
    assert len(layer.data) == 1
    np.testing.assert_allclose(layer.data, [[2.0, 20.0, 14.0]])
    assert trail_calls == ["trail", "trail"]


def test_main_empty_frame_clears_points_and_still_updates_trail():
    app, _callback, trail_calls = _main_harness([])
    app._points_layer = _Layer([[4.0, 5.0, 6.0]])

    AceTreeApp._update_3d_points(app)

    assert app._points_layer.data.shape == (0, 3)
    assert app._points_layer.editable is False
    assert app._points_layer.mode == "pan_zoom"
    assert trail_calls == ["trail"]


@pytest.mark.parametrize("empty_frame", [False, True])
def test_main_failed_atomic_replacement_is_a_refresh_failure(
    monkeypatch,
    empty_frame,
):
    nucleus = Nucleus(index=1, x=10, y=20, z=3.0, size=8, status=1)
    app, _callback, trail_calls = _main_harness([nucleus])
    AceTreeApp._update_3d_points(app)
    previous = app._points_layer.data.copy()
    app.manager.nuclei = [] if empty_frame else [nucleus]

    monkeypatch.setattr(
        "acetree_py.gui.app.replace_points_layer",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="previous complete marker set"):
        AceTreeApp._update_3d_points(app)

    np.testing.assert_allclose(app._points_layer.data, previous)
    assert app._points_layer.editable is False
    assert app._points_layer.mode == "pan_zoom"
    assert trail_calls == ["trail"]


@pytest.mark.parametrize("failure", ["text", "callback"])
def test_main_initial_configuration_failure_still_locks_layer(failure):
    nucleus = Nucleus(index=1, x=10, y=20, z=3.0, size=8, status=1)
    app, callback, trail_calls = _main_harness([nucleus])
    app.viewer = _ConfigurationFailureViewer(failure)

    with pytest.raises(RuntimeError, match=failure):
        AceTreeApp._update_3d_points(app)

    layer = app._points_layer
    assert layer is not None
    assert layer.editable is False
    assert layer.selected_data == set()
    assert layer.mode == "pan_zoom"
    assert trail_calls == []

    # A later refresh must retry the missing setup instead of treating the
    # partially configured layer as permanently complete.
    layer._configuration_failure = None
    if failure == "callback":
        layer.mouse_drag_callbacks = []
    AceTreeApp._update_3d_points(app)
    assert layer.mouse_drag_callbacks == [callback]
    assert layer.text["string"] == "{name}"
    assert trail_calls == ["trail"]


def test_detached_curated_points_are_locked_without_losing_label_callback():
    nucleus = Nucleus(index=1, x=10, y=20, z=3.0, size=8, status=1)
    callback = object()
    trail_calls = []
    window = SimpleNamespace(
        app=SimpleNamespace(
            manager=_Manager([nucleus]),
            color_engine=_ColorEngine(),
            current_cell_name="",
        ),
        _viewer=_Viewer(),
        _points_layer=None,
        _labels_visible=True,
        _shown_labels=set(),
        _on_click=callback,
        _update_trail=lambda time: trail_calls.append(time),
        _stack_z_from_plane=lambda value: float(value) - 1.0,
        _make_curated_points_read_only=(
            Viewer3DWindow._make_curated_points_read_only
        ),
    )

    Viewer3DWindow._update_points(window, 1)

    layer = window._points_layer
    assert layer.editable is False
    assert layer.selected_data == set()
    assert layer.mode == "pan_zoom"
    assert layer.mouse_drag_callbacks == [callback]

    layer.editable = True
    layer.selected_data = {0}
    layer.mode = "select"
    nucleus.y = 24
    Viewer3DWindow._update_points(window, 1)

    assert layer.editable is False
    assert layer.selected_data == set()
    assert layer.mode == "pan_zoom"
    assert layer.mouse_drag_callbacks == [callback]
    assert len(layer.data) == 1
    np.testing.assert_allclose(layer.data, [[2.0, 24.0, 10.0]])
    assert trail_calls == [1, 1]


@pytest.mark.parametrize("empty_frame", [False, True])
def test_detached_failed_atomic_replacement_is_a_refresh_failure(
    monkeypatch,
    empty_frame,
):
    nucleus = Nucleus(index=1, x=10, y=20, z=3.0, size=8, status=1)
    trail_calls = []
    window = SimpleNamespace(
        app=SimpleNamespace(
            manager=_Manager([nucleus]),
            color_engine=_ColorEngine(),
            current_cell_name="",
        ),
        _viewer=_Viewer(),
        _points_layer=None,
        _labels_visible=True,
        _shown_labels=set(),
        _on_click=object(),
        _update_trail=lambda time: trail_calls.append(time),
        _stack_z_from_plane=lambda value: float(value) - 1.0,
        _make_curated_points_read_only=(
            Viewer3DWindow._make_curated_points_read_only
        ),
    )
    Viewer3DWindow._update_points(window, 1)
    previous = window._points_layer.data.copy()
    window.app.manager.nuclei = [] if empty_frame else [nucleus]

    monkeypatch.setattr(
        "acetree_py.gui.viewer_3d_window.replace_points_layer",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="previous complete marker set"):
        Viewer3DWindow._update_points(window, 1)

    np.testing.assert_allclose(window._points_layer.data, previous)
    assert window._points_layer.editable is False
    assert window._points_layer.mode == "pan_zoom"
    assert trail_calls == [1]


@pytest.mark.parametrize("failure", ["text", "callback"])
def test_detached_initial_configuration_failure_still_locks_layer(failure):
    nucleus = Nucleus(index=1, x=10, y=20, z=3.0, size=8, status=1)
    viewer = _ConfigurationFailureViewer(failure)
    window = SimpleNamespace(
        app=SimpleNamespace(
            manager=_Manager([nucleus]),
            color_engine=_ColorEngine(),
            current_cell_name="",
        ),
        _viewer=viewer,
        _points_layer=None,
        _labels_visible=True,
        _shown_labels=set(),
        _on_click=object(),
        _update_trail=lambda _time: None,
        _stack_z_from_plane=lambda value: float(value) - 1.0,
        _make_curated_points_read_only=(
            Viewer3DWindow._make_curated_points_read_only
        ),
    )

    with pytest.raises(RuntimeError, match=failure):
        Viewer3DWindow._update_points(window, 1)

    layer = window._points_layer
    assert layer is not None
    assert layer.editable is False
    assert layer.selected_data == set()
    assert layer.mode == "pan_zoom"

    layer._configuration_failure = None
    if failure == "callback":
        layer.mouse_drag_callbacks = []
    Viewer3DWindow._update_points(window, 1)
    assert layer.mouse_drag_callbacks == [window._on_click]
    assert layer.text["string"] == "{name}"


class _Control:
    def __init__(self, value, *, checked=False):
        self._value = value
        self._checked = checked

    def blockSignals(self, _blocked):
        pass

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value

    def isChecked(self):
        return self._checked


def test_detached_same_frame_refreshes_for_move_undo_redo_and_stays_unsynced():
    record = [[Nucleus(index=1, x=10, y=20, z=3.0, size=8, status=1)]]
    history = EditHistory(record)
    calls = []

    class _RefreshWindow:
        view_time = Viewer3DWindow.view_time

    window = _RefreshWindow()
    window.app = SimpleNamespace(current_time=9, edit_history=history)
    window._viewer = object()
    window._chk_sync = _Control(checked=False, value=2)
    window._time_spin = _Control(2)
    window._time_slider = _Control(2)
    window._local_time = 2
    window._last_time = 2
    window._last_change_counter = history.change_counter
    window._load_stacks = lambda time: calls.append(("stack", time))
    window._update_points = lambda time: calls.append(("points", time))
    window._update_tracking_preview = lambda: calls.append(("preview", 2))

    Viewer3DWindow.refresh(window)
    assert calls == []

    history.do(MoveNucleus(time=1, index=1, new_x=12))
    Viewer3DWindow.refresh(window)
    history.undo()
    Viewer3DWindow.refresh(window)
    history.redo()
    Viewer3DWindow.refresh(window)

    assert calls == [
        ("stack", 2),
        ("points", 2),
        ("preview", 2),
    ] * 3
    assert window._local_time == 2
    assert window._time_spin.value() == 2
    assert window._time_slider.value() == 2
    assert window.app.current_time == 9


def test_detached_refresh_caches_only_after_every_update_succeeds():
    history = SimpleNamespace(change_counter=4)
    calls = []
    point_attempts = 0

    class _RefreshWindow:
        view_time = Viewer3DWindow.view_time

    window = _RefreshWindow()
    window.app = SimpleNamespace(current_time=2, edit_history=history)
    window._viewer = object()
    window._chk_sync = _Control(checked=False, value=2)
    window._time_spin = _Control(2)
    window._time_slider = _Control(2)
    window._local_time = 2
    window._last_time = 2
    window._last_change_counter = 3
    window._load_stacks = lambda time: calls.append(("stack", time))

    def update_points(time):
        nonlocal point_attempts
        point_attempts += 1
        calls.append(("points", time))
        if point_attempts == 1:
            raise RuntimeError("simulated rolled-back points redraw")

    window._update_points = update_points
    window._update_tracking_preview = lambda: calls.append(("preview", 2))

    with pytest.raises(RuntimeError, match="rolled-back points redraw"):
        Viewer3DWindow.refresh(window)

    assert window._last_time == 2
    assert window._last_change_counter == 3

    Viewer3DWindow.refresh(window)

    assert calls == [
        ("stack", 2),
        ("points", 2),
        ("stack", 2),
        ("points", 2),
        ("preview", 2),
    ]
    assert window._last_time == 2
    assert window._last_change_counter == 4
