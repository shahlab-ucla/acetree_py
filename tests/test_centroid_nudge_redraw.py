"""Focused regressions for centroid nudge redraw ordering.

These tests use the real EditPanel and EditHistory while keeping the viewer
surface lightweight.  The recorded display events make redraw count and the
navigation state visible without requiring an OpenGL-backed napari canvas.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("qtpy")

from acetree_py.core.nucleus import Nucleus
from acetree_py.editing.history import EditHistory, PostCommitCallbackError
from acetree_py.gui.edit_panel import EditPanel


class _NudgeApp:
    def __init__(
        self,
        *,
        z: float = 10.0,
        current_plane: int = 10,
        num_planes: int = 30,
        on_render=None,
    ) -> None:
        self.nucleus = Nucleus(
            index=1,
            x=100,
            y=200,
            z=z,
            size=20,
            identity="AB",
            status=1,
        )
        record = [[self.nucleus]]
        self.manager = SimpleNamespace(nuclei_record=record)
        self.current_time = 1
        self.current_plane = current_plane
        self.current_cell_name = "AB"
        self.tracking = True
        self.image_provider = SimpleNamespace(num_planes=num_planes)

        self.viewer = None
        self._image_layer = None
        self._viewer_integration = None
        self._player_controls = None
        self._cell_info_panel = None
        self._contrast_tools = None
        self._edit_panel = None
        self._placement_mode = False
        self._add_mode = False
        self._viz_mode = False
        self._color_engine = None

        self.render_events: list[tuple[int, bool, float, int, int, int]] = []
        self.set_plane_calls: list[int] = []
        self._on_render = on_render
        self.edit_history = EditHistory(record, on_edit=self.update_display)

    def get_selected_nucleus(self):
        return self.nucleus, self.current_time, self.nucleus.index

    def update_display(self) -> None:
        self.render_events.append(
            (
                self.current_plane,
                self.tracking,
                self.nucleus.z,
                self.nucleus.x,
                self.nucleus.y,
                self.nucleus.size,
            )
        )
        if self._on_render is not None:
            self._on_render()

    def set_plane(self, plane: int) -> None:
        """Mirror AceTreeApp.set_plane so accidental second redraws are seen."""
        self.set_plane_calls.append(plane)
        plane = max(1, min(plane, self.image_provider.num_planes))
        if plane == self.current_plane:
            return
        self.current_plane = plane
        self.tracking = False
        self.update_display()


def _make_panel(qtbot, **app_kwargs):
    app = _NudgeApp(**app_kwargs)
    panel = EditPanel(app)
    qtbot.addWidget(panel)
    return app, panel


def test_z_nudge_renders_once_on_final_plane(qtbot):
    app, panel = _make_panel(qtbot, z=10.0, current_plane=10)

    panel._nudge(dz=1.0)

    assert app.nucleus.z == 11.0
    assert app.current_plane == 11
    assert app.tracking is False
    assert app.set_plane_calls == []
    assert app.render_events == [(11, False, 11.0, 100, 200, 20)]
    assert app.edit_history.num_undoable == 1


def test_z_nudge_clamps_plane_and_preserves_undo_redo(qtbot):
    app, panel = _make_panel(
        qtbot,
        z=29.0,
        current_plane=29,
        num_planes=30,
    )

    panel._nudge(dz=5.0)

    assert app.nucleus.z == 34.0
    assert app.current_plane == 30
    assert app.render_events == [(30, False, 34.0, 100, 200, 20)]

    app.edit_history.undo()
    assert app.nucleus.z == 29.0
    assert app.current_plane == 30
    assert app.render_events[-1] == (30, False, 29.0, 100, 200, 20)

    app.edit_history.redo()
    assert app.nucleus.z == 34.0
    assert app.current_plane == 30
    assert app.render_events[-1] == (30, False, 34.0, 100, 200, 20)
    assert len(app.render_events) == 3


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"dx": 5, "dy": -1}, (105, 199, 20)),
        ({"dsize": -5}, (100, 200, 15)),
    ],
)
def test_xy_and_size_nudges_leave_navigation_mode_unchanged(
    qtbot,
    delta,
    expected,
):
    app, panel = _make_panel(qtbot, z=12.0, current_plane=12)

    panel._nudge(**delta)

    assert (app.nucleus.x, app.nucleus.y, app.nucleus.size) == expected
    assert app.current_plane == 12
    assert app.tracking is True
    assert app.set_plane_calls == []
    assert app.render_events == [
        (12, True, 12.0, expected[0], expected[1], expected[2])
    ]


def test_z_nudge_at_lower_plane_keeps_tracking_when_plane_does_not_change(qtbot):
    app, panel = _make_panel(qtbot, z=0.0, current_plane=1)

    panel._nudge(dz=-1.0)

    assert app.nucleus.z == 0.0
    assert app.current_plane == 1
    assert app.tracking is True
    assert app.set_plane_calls == []
    assert app.render_events == [(1, True, 0.0, 100, 200, 20)]


def test_failed_z_nudge_restores_navigation_state(qtbot, monkeypatch):
    app, panel = _make_panel(qtbot, z=10.0, current_plane=10)
    monkeypatch.setattr(
        app,
        "get_selected_nucleus",
        lambda: (app.nucleus, 1, 99),
    )

    with pytest.raises(IndexError):
        panel._nudge(dz=1.0)

    assert app.nucleus.z == 10.0
    assert app.current_plane == 10
    assert app.tracking is True
    assert app.render_events == []
    assert app.edit_history.num_undoable == 0


def test_post_commit_render_failure_retains_final_z_navigation(qtbot):
    def fail_render():
        raise RuntimeError("simulated renderer failure")

    app, panel = _make_panel(
        qtbot,
        z=10.0,
        current_plane=10,
        on_render=fail_render,
    )

    with pytest.raises(PostCommitCallbackError):
        panel._nudge(dz=1.0)

    assert app.nucleus.z == 11.0
    assert app.current_plane == 11
    assert app.tracking is False
    assert app.render_events == [(11, False, 11.0, 100, 200, 20)]
    assert app.edit_history.num_undoable == 1
