"""Physical-selection regressions for duplicate and unnamed centroid labels."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from acetree_py.core.lineage import build_lineage_tree
from acetree_py.core.movie import Movie
from acetree_py.core.nucleus import Nucleus
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.gui.app import AceTreeApp
from acetree_py.gui.viewer_3d_window import Viewer3DWindow
from acetree_py.gui.viewer_integration import ViewerIntegration


def _nucleus(
    index: int,
    x: int,
    y: int,
    *,
    identity: str,
    assigned_id: str,
    predecessor: int = -1,
    successor1: int = -1,
) -> Nucleus:
    return Nucleus(
        index=index,
        x=x,
        y=y,
        z=5.0,
        size=20,
        status=1,
        identity=identity,
        assigned_id=assigned_id,
        predecessor=predecessor,
        successor1=successor1,
    )


def _duplicate_name_app() -> tuple[AceTreeApp, Nucleus, Nucleus]:
    manager = NucleiManager()
    manager.movie = Movie(xy_res=1.0, z_res=1.0, num_planes=12)
    left = _nucleus(
        1,
        40,
        50,
        identity="LeftSeed",
        assigned_id="Dup",
        successor1=1,
    )
    right = _nucleus(
        2,
        140,
        150,
        identity="RightSeed",
        assigned_id="Dup",
        successor1=2,
    )
    manager.nuclei_record = [
        [left, right],
        [
            _nucleus(
                1,
                42,
                51,
                identity="LeftSeed",
                assigned_id="Dup",
                predecessor=1,
            ),
            _nucleus(
                2,
                142,
                151,
                identity="RightSeed",
                assigned_id="Dup",
                predecessor=2,
            ),
        ],
    ]
    manager.set_all_successors()
    manager.lineage_tree = build_lineage_tree(
        manager.nuclei_record,
        starting_index=0,
        ending_index=2,
        create_dummy_ancestors=False,
    )
    app = AceTreeApp(manager)
    app.current_time = 1
    app.current_plane = 5
    app._set_selection_from_nucleus(1, right)
    return app, left, right


class _PointsLayer:
    def __init__(self, data, **properties) -> None:
        self.data = np.asarray(data, dtype=float)
        self.size = np.asarray(properties.pop("size", np.ones(len(self.data))))
        self.face_color = np.asarray(
            properties.pop("face_color", np.zeros((len(self.data), 4)))
        )
        self.features = properties.pop("features", {})
        self.mouse_drag_callbacks = []
        self.text = {}
        self.selected_data = set()
        self.editable = True
        self.mode = "select"
        for name, value in properties.items():
            setattr(self, name, value)


class _Viewer:
    def __init__(self) -> None:
        self.layers = []

    def add_points(self, data, **properties):
        layer = _PointsLayer(data, **properties)
        self.layers.append(layer)
        return layer


def _attach_main_3d(app: AceTreeApp) -> _PointsLayer:
    app.viewer = _Viewer()
    app._viewer_integration = SimpleNamespace(
        _shown_labels=set(),
        _labels_global_visible=True,
        trails_visible=False,
    )
    app._update_3d_points()
    return app._points_layer


def test_duplicate_forced_name_selects_exactly_one_2d_marker_in_both_modes() -> None:
    app, _left, right = _duplicate_name_app()

    editing = app.get_nucleus_overlay_data()
    assert editing["names"] == ["Dup", "Dup"]
    assert editing["selected_idx"] == right.index - 1
    assert np.allclose(editing["colors"][1], [1.0, 1.0, 1.0, 1.0])
    assert not np.allclose(editing["colors"][0], [1.0, 1.0, 1.0, 1.0])

    app._viz_mode = True
    visual = app.get_nucleus_overlay_data()
    assert visual["selected_idx"] == right.index - 1
    assert np.allclose(visual["colors"][1], app.color_engine.selected_color)
    assert visual["colors"][0, 3] < visual["colors"][1, 3]


def test_duplicate_forced_name_selects_exactly_one_main_3d_marker() -> None:
    app, _left, right = _duplicate_name_app()

    editing_layer = _attach_main_3d(app)
    assert np.allclose(
        editing_layer.face_color[right.index - 1],
        [1.0, 1.0, 1.0, 1.0],
    )
    assert not np.allclose(
        editing_layer.face_color[0],
        [1.0, 1.0, 1.0, 1.0],
    )
    assert list(editing_layer.features["acetree_time"]) == [1, 1]
    assert list(editing_layer.features["acetree_index"]) == [1, 2]

    app._viz_mode = True
    app._update_3d_points()
    assert np.allclose(
        editing_layer.face_color[right.index - 1],
        app.color_engine.selected_color,
    )
    assert editing_layer.face_color[0, 3] < editing_layer.face_color[1, 3]


def test_detached_unsynced_3d_resolves_selected_track_at_local_time() -> None:
    app, _left, _right = _duplicate_name_app()
    app.current_time = 1
    viewer = _Viewer()
    window = SimpleNamespace(
        app=app,
        _viewer=viewer,
        _points_layer=None,
        _labels_visible=True,
        _shown_labels=set(),
        _on_click=lambda *_args: None,
        _update_trail=lambda _time: None,
        _stack_z_from_plane=lambda plane: plane - 1.0,
        _make_curated_points_read_only=(
            Viewer3DWindow._make_curated_points_read_only
        ),
    )

    Viewer3DWindow._update_points(window, 2)

    layer = window._points_layer
    selected_at_two = app.get_selected_nucleus(2)
    assert selected_at_two is not None
    selected_index = selected_at_two[2] - 1
    assert np.allclose(
        layer.face_color[selected_index],
        app.color_engine.selected_color,
    )
    other_index = 1 - selected_index
    assert layer.face_color[other_index, 3] < layer.face_color[selected_index, 3]
    assert list(layer.features["acetree_time"]) == [2, 2]
    assert app.current_time == 1


def test_unnamed_physical_selection_is_highlighted_without_name_matching() -> None:
    manager = NucleiManager()
    manager.movie = Movie(xy_res=1.0, z_res=1.0, num_planes=12)
    unnamed = _nucleus(1, 20, 30, identity="", assigned_id="")
    manager.nuclei_record = [[unnamed]]
    manager.lineage_tree = build_lineage_tree(
        manager.nuclei_record,
        starting_index=0,
        ending_index=1,
        create_dummy_ancestors=False,
    )
    app = AceTreeApp(manager)
    app.current_time = 1
    app.current_plane = 5
    app._set_selection_from_nucleus(1, unnamed)

    overlay = app.get_nucleus_overlay_data()

    assert overlay["selected_idx"] == 0
    assert np.allclose(overlay["colors"][0], [1.0, 1.0, 1.0, 1.0])
    assert app.current_cell_name.startswith("idx=1:")


def test_cell_info_uses_selected_duplicate_track_not_name_lookup() -> None:
    app, _left, right = _duplicate_name_app()

    info = app.get_cell_info_text()

    assert f"Position: ({right.x}, {right.y}, {right.z:.1f})" in info


def test_color_rules_resolve_duplicate_names_by_hash_and_refresh_after_rebuild() -> None:
    app, left, right = _duplicate_name_app()
    engine = app.color_engine
    old_left_cell = app._cell_for_nucleus(1, left)
    old_right_cell = app._cell_for_nucleus(1, right)

    assert old_left_cell is not None
    assert old_right_cell is not None
    assert old_left_cell is not old_right_cell
    assert engine._lookup_cell(left, app.manager, 1) is old_left_cell
    assert engine._lookup_cell(right, app.manager, 1) is old_right_cell

    app.manager.lineage_tree = build_lineage_tree(
        app.manager.nuclei_record,
        starting_index=0,
        ending_index=2,
        create_dummy_ancestors=False,
    )
    rebuilt_right_cell = app._cell_for_nucleus(1, right)

    assert rebuilt_right_cell is not None
    assert rebuilt_right_cell is not old_right_cell
    assert engine._lookup_cell(right, app.manager, 1) is rebuilt_right_cell


class _HoverTimer:
    def __init__(self) -> None:
        self.active = False
        self.starts: list[int] = []

    def start(self, delay: int) -> None:
        self.active = True
        self.starts.append(delay)

    def stop(self) -> None:
        self.active = False

    def isActive(self) -> bool:
        return self.active


class _HoverTooltip:
    def __init__(self) -> None:
        self.visible = False

    def isVisible(self) -> bool:
        return self.visible

    def hide(self) -> None:
        self.visible = False


def test_hover_anchor_distinguishes_duplicate_forced_names() -> None:
    app, left, right = _duplicate_name_app()
    app.get_cell_info_text = lambda: "PHYSICALLY SELECTED"
    integration = ViewerIntegration(app)
    timer = _HoverTimer()
    integration._tooltip = _HoverTooltip()
    integration._tooltip_timer = timer

    integration._on_mouse_move(
        None,
        SimpleNamespace(position=(left.y, left.x)),
    )
    assert integration._last_hover_name == "Dup"
    assert integration._last_hover_anchor == (1, left.index)
    assert timer.starts == [integration._hover_delay_ms]

    # The second nucleus has the same forced name. Moving onto it must still
    # restart the delay and replace the physical hover target.
    integration._on_mouse_move(
        None,
        SimpleNamespace(position=(right.y, right.x)),
    )
    assert integration._last_hover_name == "Dup"
    assert integration._last_hover_anchor == (1, right.index)
    assert timer.starts == [
        integration._hover_delay_ms,
        integration._hover_delay_ms,
    ]

    left_info = integration._get_hover_info(
        "Dup",
        hover_anchor=(1, left.index),
    )
    right_info = integration._get_hover_info(
        "Dup",
        hover_anchor=(1, right.index),
    )

    assert f"Position: ({left.x}, {left.y}, {left.z:.1f})" in left_info
    assert f"Position: ({right.x}, {right.y}, {right.z:.1f})" not in left_info
    assert right_info == "PHYSICALLY SELECTED"

    integration._hide_tooltip()
    assert integration._last_hover_name is None
    assert integration._last_hover_anchor is None
