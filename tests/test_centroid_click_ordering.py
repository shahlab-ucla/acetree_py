"""Mouse-release ordering contracts for curated centroid interactions."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from acetree_py.core.lineage import build_lineage_tree  # noqa: E402
from acetree_py.core.movie import Movie  # noqa: E402
from acetree_py.core.nucleus import Nucleus  # noqa: E402
from acetree_py.core.nuclei_manager import NucleiManager  # noqa: E402
from acetree_py.gui.app import AceTreeApp  # noqa: E402
from acetree_py.gui.viewer_3d_window import Viewer3DWindow  # noqa: E402
from acetree_py.gui.viewer_integration import ViewerIntegration  # noqa: E402


def _event(*, button: int = 1, position=(11.0, 13.0)) -> SimpleNamespace:
    return SimpleNamespace(
        type="mouse_press",
        button=button,
        position=position,
        pos=position[-2:],
        view_direction=None,
        dims_displayed=None,
    )


def _release(generator, event) -> None:
    next(generator)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(generator)
    assert generator.gi_frame is None


def _nucleus(index: int, name: str = "AB") -> Nucleus:
    return Nucleus(
        index=index,
        x=10 * index,
        y=20,
        z=3.0,
        size=8,
        identity=name,
        status=1,
    )


def _two_dimensional_harness():
    first = _nucleus(1, "AB")
    second = _nucleus(2, "P1")
    hit = {"nucleus": first}
    calls: list[tuple] = []
    record = [[first, second]]
    manager = SimpleNamespace(
        nuclei_record=record,
        find_closest_nucleus=lambda *_args, **_kwargs: hit["nucleus"],
    )
    app = SimpleNamespace(
        viewer=None,
        manager=manager,
        edit_history=SimpleNamespace(change_counter=0),
        current_time=1,
        current_plane=3,
        current_cell_name="AB",
        selection_anchor=(1, 1),
        _relink_pick_mode=False,
        _relink_pick_callback=None,
        _add_mode=False,
        _placement_mode=False,
        _placement_parent_name=None,
        _placement_parent_anchor=None,
        _placement_default_size=20,
    )
    app._nucleus_at_anchor = lambda anchor: record[anchor[0] - 1][anchor[1] - 1]
    app._handle_add_click = lambda x, y: calls.append(("add", x, y))
    app._handle_placement_click = lambda x, y: calls.append(("track", x, y))
    app._set_selection_from_nucleus = lambda time, nuc: calls.append(
        ("select", time, nuc.index)
    )
    app.update_display = lambda: calls.append(("redraw",))
    app.deselect_cell = lambda: calls.append(("deselect",))

    def exit_relink() -> None:
        app._relink_pick_mode = False
        app._relink_pick_callback = None

    app.exit_relink_pick_mode = exit_relink
    integration = ViewerIntegration(app)
    layer = object()
    integration._shapes_layer = layer
    integration.update_overlays = lambda: calls.append(("overlay",))
    return integration, app, layer, first, second, hit, calls


def test_2d_add_and_track_run_after_release_with_press_coordinates(monkeypatch):
    integration, app, layer, _first, _second, _hit, calls = (
        _two_dimensional_harness()
    )
    queued: list = []
    monkeypatch.setattr(
        "acetree_py.gui.viewer_integration.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )

    app._add_mode = True
    event = _event(button=1, position=(11.0, 13.0))
    generator = integration._on_click(layer, event)
    next(generator)
    assert calls == []
    event.position = (99.0, 101.0)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(generator)
    assert calls == []
    queued.pop(0)()
    assert calls == [("add", 13.0, 11.0)]

    calls.clear()
    app._add_mode = False
    app._placement_mode = True
    event = _event(button=2, position=(17.0, 19.0))
    generator = integration._on_click(layer, event)
    _release(generator, event)
    assert calls == []
    queued.pop(0)()
    assert calls == [("track", 19.0, 17.0)]


def test_2d_pan_drag_and_stale_view_cancel_click_action(monkeypatch):
    integration, app, layer, _first, _second, _hit, calls = (
        _two_dimensional_harness()
    )
    queued: list = []
    monkeypatch.setattr(
        "acetree_py.gui.viewer_integration.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )
    app._add_mode = True

    drag_event = _event(button=1)
    generator = integration._on_click(layer, drag_event)
    next(generator)
    drag_event.type = "mouse_move"
    drag_event.pos = (100.0, 100.0)
    next(generator)
    drag_event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(generator)
    assert queued == []
    assert calls == []

    click_event = _event(button=1)
    generator = integration._on_click(layer, click_event)
    _release(generator, click_event)
    app.current_time = 2
    queued.pop(0)()
    assert calls == []


def test_2d_release_displacement_without_move_event_is_treated_as_drag(
    monkeypatch,
):
    integration, app, layer, _first, _second, _hit, calls = (
        _two_dimensional_harness()
    )
    queued: list = []
    monkeypatch.setattr(
        "acetree_py.gui.viewer_integration.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )
    app._add_mode = True

    event = _event(button=1)
    generator = integration._on_click(layer, event)
    next(generator)
    # Some backends coalesce the move and report only a distant release.
    event.type = "mouse_release"
    event.pos = (100.0, 100.0)
    with pytest.raises(StopIteration):
        next(generator)

    assert queued == []
    assert calls == []


def test_normal_2d_actions_remain_available_in_unchanged_edit_mode(monkeypatch):
    integration, app, layer, first, _second, _hit, calls = (
        _two_dimensional_harness()
    )
    queued: list = []
    monkeypatch.setattr(
        "acetree_py.gui.viewer_integration.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )

    # Add mode uses left click, so right-click selection remains useful for
    # choosing a different parent without leaving the workflow.
    app._add_mode = True
    event = _event(button=2)
    generator = integration._on_click(layer, event)
    _release(generator, event)
    queued.pop(0)()
    assert calls == [("select", 1, first.index), ("redraw",)]

    # Track mode uses right click, so left-click label toggling remains valid.
    calls.clear()
    app._add_mode = False
    app._placement_mode = True
    event = _event(button=1)
    generator = integration._on_click(layer, event)
    _release(generator, event)
    queued.pop(0)()
    assert integration._shown_labels == {"AB"}
    assert calls == [("overlay",)]

    # A mode transition after release invalidates the captured intent.
    calls.clear()
    app._placement_mode = False
    event = _event(button=2)
    generator = integration._on_click(layer, event)
    _release(generator, event)
    app._add_mode = True
    queued.pop(0)()
    assert calls == []


def test_2d_selection_and_relink_use_press_anchor_after_generator_closes(
    monkeypatch,
):
    integration, app, layer, first, second, hit, calls = (
        _two_dimensional_harness()
    )
    queued: list = []
    monkeypatch.setattr(
        "acetree_py.gui.viewer_integration.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )

    event = _event(button=2)
    generator = integration._on_click(layer, event)
    _release(generator, event)
    hit["nucleus"] = second
    queued.pop(0)()
    assert calls[:2] == [("select", 1, first.index), ("redraw",)]

    calls.clear()
    picked: list[tuple] = []
    routed_actions: list = []

    def run_edit_action(action, *args):
        routed_actions.append(action)
        return action(*args)

    app._run_edit_action = run_edit_action
    app._relink_pick_mode = True
    app._relink_pick_callback = lambda time, nuc: picked.append((time, nuc.index))
    hit["nucleus"] = first
    event = _event(button=2)
    generator = integration._on_click(layer, event)
    next(generator)
    assert app._relink_pick_mode is True
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(generator)
    assert generator.gi_frame is None
    assert picked == []
    queued.pop(0)()
    assert picked == [(1, first.index)]
    assert len(routed_actions) == 1
    assert app._relink_pick_mode is False

    # Escape/cancel after release but before the queued action must win.
    app._relink_pick_mode = True
    callback = lambda time, nuc: picked.append((time, nuc.index))
    app._relink_pick_callback = callback
    event = _event(button=2)
    generator = integration._on_click(layer, event)
    _release(generator, event)
    app.exit_relink_pick_mode()
    queued.pop(0)()
    assert picked == [(1, first.index)]


class _PickLayer:
    def __init__(self, *, time: int, indices: list[int], picked: int = 0) -> None:
        self.features = {
            "acetree_time": [time] * len(indices),
            "acetree_index": indices,
            "full_name": [f"Nuc{index}" for index in indices],
        }
        self.picked = picked

    def get_value(self, *_args, **_kwargs):
        return self.picked


def _main_3d_app() -> tuple[AceTreeApp, Nucleus, Nucleus]:
    manager = NucleiManager()
    manager.movie = Movie(xy_res=1.0, z_res=1.0, num_planes=8)
    first = _nucleus(1, "AB")
    second = _nucleus(2, "P1")
    manager.nuclei_record = [[first, second]]
    manager.lineage_tree = build_lineage_tree(
        manager.nuclei_record,
        starting_index=0,
        ending_index=1,
        create_dummy_ancestors=False,
    )
    app = AceTreeApp(manager)
    app.current_time = 1
    app.current_plane = 3
    app._viewer_integration = SimpleNamespace(_shown_labels=set())
    app._lineage_widgets = []
    app._lineage_list = None
    return app, first, second


def test_main_3d_selection_waits_for_release_and_camera_drag_is_ignored(
    monkeypatch,
):
    app, _first, second = _main_3d_app()
    layer = _PickLayer(time=1, indices=[1, 2], picked=1)
    app._points_layer = layer
    redraws: list[str] = []
    app._update_3d_points = lambda: redraws.append("redraw")
    queued: list = []
    monkeypatch.setattr(
        "qtpy.QtCore.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )

    event = _event(button=2, position=(3.0, 20.0, 20.0))
    generator = app._on_3d_click(layer, event)
    next(generator)
    assert app.selection_anchor is None
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(generator)
    assert app.selection_anchor is None
    queued.pop(0)()
    assert app.selection_anchor == (1, second.index)
    assert redraws == ["redraw"]

    app.selection_anchor = None
    redraws.clear()
    event = _event(button=2, position=(3.0, 20.0, 20.0))
    generator = app._on_3d_click(layer, event)
    next(generator)
    event.type = "mouse_move"
    event.pos = (100.0, 100.0)
    next(generator)
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(generator)
    assert queued == []
    assert app.selection_anchor is None
    assert redraws == []

    event = _event(button=2, position=(3.0, 20.0, 20.0))
    generator = app._on_3d_click(layer, event)
    next(generator)
    event.type = "mouse_release"
    event.pos = (100.0, 100.0)
    with pytest.raises(StopIteration):
        next(generator)
    assert queued == []
    assert app.selection_anchor is None


def test_main_3d_label_and_relink_are_deferred_and_relink_uses_safe_runner(
    monkeypatch,
):
    app, first, _second = _main_3d_app()
    layer = _PickLayer(time=1, indices=[1, 2], picked=0)
    app._points_layer = layer
    redraws: list[str] = []
    queued: list = []
    app._update_3d_points = lambda: redraws.append("redraw")
    monkeypatch.setattr(
        "qtpy.QtCore.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )

    event = _event(button=1, position=(3.0, 20.0, 10.0))
    generator = app._on_3d_click(layer, event)
    _release(generator, event)
    assert app._viewer_integration._shown_labels == set()
    queued.pop(0)()
    assert app._viewer_integration._shown_labels == {"AB"}
    assert redraws == ["redraw"]

    # Escape after release wins over the deferred relink callback.
    picked: list[tuple[int, int]] = []
    callback = lambda time, nuc: picked.append((time, nuc.index))
    app.enter_relink_pick_mode(callback)
    event = _event(button=2, position=(3.0, 20.0, 10.0))
    generator = app._on_3d_click(layer, event)
    _release(generator, event)
    app.exit_relink_pick_mode()
    queued.pop(0)()
    assert picked == []

    routed: list = []

    def run_edit_action(action, *args):
        routed.append(action)
        return action(*args)

    monkeypatch.setattr(app, "_run_edit_action", run_edit_action)
    app.enter_relink_pick_mode(callback)
    event = _event(button=2, position=(3.0, 20.0, 10.0))
    generator = app._on_3d_click(layer, event)
    _release(generator, event)
    queued.pop(0)()
    assert picked == [(1, first.index)]
    assert routed == [callback]
    assert app._relink_pick_mode is False


def test_add_and_track_switch_main_view_from_3d_to_2d(monkeypatch):
    app, _first, _second = _main_3d_app()
    switches: list[bool] = []
    modes_during_switch: list[tuple[bool, bool]] = []
    messages: list[str] = []

    def switch(enabled: bool) -> None:
        switches.append(enabled)
        modes_during_switch.append((app._add_mode, app._placement_mode))
        app._3d_mode = enabled

    monkeypatch.setattr(app, "set_3d_mode", switch)
    monkeypatch.setattr(app, "_say", messages.append)

    app._3d_mode = True
    app.enter_add_mode()
    assert switches == [False]
    assert modes_during_switch == [(True, False)]
    assert app._add_mode is True
    assert "2D slice" in messages[-1]

    app._3d_mode = True
    app.enter_placement_mode(parent_name="AB")
    assert switches == [False, False]
    assert modes_during_switch == [(True, False), (False, True)]
    assert app._placement_mode is True
    assert app._add_mode is False
    assert "2D slice" in messages[-1]


@pytest.mark.parametrize("active_mode", ["add", "track"])
def test_entering_3d_cancels_active_2d_placement_mode(monkeypatch, active_mode):
    app, _first, _second = _main_3d_app()
    app.viewer = SimpleNamespace(dims=SimpleNamespace(ndisplay=2))
    messages: list[str] = []
    checked = {"add": True, "track": True}
    app._edit_panel = SimpleNamespace(
        _btn_add=SimpleNamespace(
            setChecked=lambda value: checked.__setitem__("add", value)
        ),
        _btn_track=SimpleNamespace(
            setChecked=lambda value: checked.__setitem__("track", value)
        ),
    )

    def enter_3d() -> None:
        app.viewer.dims.ndisplay = 3

    monkeypatch.setattr(app, "_enter_3d", enter_3d)
    monkeypatch.setattr(app, "_say", messages.append)
    app._add_mode = active_mode == "add"
    app._placement_mode = active_mode == "track"

    app.set_3d_mode(True)

    assert app._3d_mode is True
    assert app.viewer.dims.ndisplay == 3
    assert app._add_mode is False
    assert app._placement_mode is False
    assert checked == {"add": False, "track": False}
    assert "placement uses the 2D slice view" in messages[-1]


def test_detached_3d_label_waits_for_release_and_stale_local_time_cancels(
    monkeypatch,
):
    nucleus = _nucleus(1, "AB")
    layer = _PickLayer(time=2, indices=[1])
    queued: list = []
    redraws: list[str] = []
    app = SimpleNamespace(
        manager=SimpleNamespace(alive_nuclei_at=lambda _time: [nucleus]),
        edit_history=SimpleNamespace(change_counter=0),
        _nucleus_at_anchor=lambda anchor: nucleus if anchor == (2, 1) else None,
    )
    window = SimpleNamespace(
        app=app,
        view_time=2,
        _points_layer=layer,
        _shown_labels=set(),
        _update_label_display=lambda: redraws.append("labels"),
    )
    window._apply_deferred_label_click = MethodType(
        Viewer3DWindow._apply_deferred_label_click,
        window,
    )
    monkeypatch.setattr(
        "acetree_py.gui.viewer_3d_window.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )

    event = _event(button=1, position=(3.0, 20.0, 10.0))
    generator = Viewer3DWindow._on_click(window, layer, event)
    _release(generator, event)
    assert window._shown_labels == set()
    queued.pop(0)()
    assert window._shown_labels == {"AB"}
    assert redraws == ["labels"]

    window.view_time = 2
    event = _event(button=1, position=(3.0, 20.0, 10.0))
    generator = Viewer3DWindow._on_click(window, layer, event)
    next(generator)
    event.type = "mouse_release"
    event.pos = (100.0, 100.0)
    with pytest.raises(StopIteration):
        next(generator)
    assert queued == []
    assert window._shown_labels == {"AB"}

    event = _event(button=1, position=(3.0, 20.0, 10.0))
    generator = Viewer3DWindow._on_click(window, layer, event)
    _release(generator, event)
    window.view_time = 3
    queued.pop(0)()
    assert window._shown_labels == {"AB"}
    assert redraws == ["labels"]
