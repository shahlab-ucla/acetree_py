from __future__ import annotations

from acetree_py.core.roi_manager import RoiManager
from acetree_py.core.subcellular_roi import (
    ContourSlice,
    ContourStack3D,
    ObjectClass,
    Polygon2D,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
)
from tests.test_gui_app import _make_app


class _RoiIntegration:
    def __init__(self, *, editing=True):
        self.editing = editing
        self.cancel_count = 0
        self.update_count = 0
        self.edit_args = None
        self.draw_kwargs = None

    def cancel_edit(self):
        self.cancel_count += 1
        self.editing = False

    def update_overlay(self):
        self.update_count += 1

    def enter_edit_mode(self, *args, **kwargs):
        self.edit_args = (args, kwargs)
        self.editing = True

    def begin_drawing(self, **kwargs):
        self.draw_kwargs = kwargs
        self.editing = True


class _RoiPanel:
    def __init__(self, mode="edit"):
        self.mode = mode
        self.mode_changes = []
        self.refresh_count = 0
        self.selected_class_id = None

    def set_mode(self, mode):
        self.mode = mode
        self.mode_changes.append(mode)
        return True

    def refresh(self):
        self.refresh_count += 1


class _Signal:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback


class _ScalarWindow:
    created = []

    def __init__(self, app, object_ids, parent):
        self.app = app
        self.object_ids = object_ids
        self.parent = parent
        self.destroyed = _Signal()
        self.shown = False
        self.snapshots = []

    @classmethod
    def from_app(cls, app, *, object_ids=None, parent=None, **_kwargs):
        window = cls(app, object_ids, parent)
        cls.created.append(window)
        return window

    def show(self):
        self.shown = True

    def on_measurements_updated(self, snapshot=None):
        self.snapshots.append(snapshot)


def _arm_roi_mode(app):
    integration = _RoiIntegration(editing=True)
    panel = _RoiPanel("edit")
    app._roi_viewer_integration = integration
    app._subcellular_objects_panel = panel
    return integration, panel


def test_navigation_cancels_roi_editor_but_keeps_object_selection():
    app = _make_app()
    app.current_roi_object_id = "stable-object"
    integration, panel = _arm_roi_mode(app)

    app.set_time(2)
    assert app.current_time == 2
    assert app.current_roi_object_id == "stable-object"
    assert integration.cancel_count == 1
    assert panel.mode == "inspect"

    integration.editing = True
    panel.mode = "edit"
    app.set_plane(16)
    assert app.current_plane == 16
    assert integration.cancel_count == 2
    assert panel.mode == "inspect"


def test_nuclear_modes_and_escape_cancel_roi_mode():
    app = _make_app()
    integration, panel = _arm_roi_mode(app)
    app.enter_add_mode()
    assert app._add_mode
    assert not integration.editing
    assert panel.mode == "inspect"

    integration.editing = True
    panel.mode = "edit"
    app.enter_placement_mode(parent_name="AB")
    assert app._placement_mode
    assert not app._add_mode
    assert not integration.editing

    integration.editing = True
    panel.mode = "edit"
    app._exit_all_modes()
    assert not integration.editing
    assert panel.mode == "inspect"


def test_space_does_not_deselect_cell_while_roi_editor_is_active():
    app = _make_app()
    integration, _panel = _arm_roi_mode(app)
    app.current_cell_name = "AB"
    app._handle_space_shortcut()
    assert app.current_cell_name == "AB"

    integration.editing = False
    app._handle_space_shortcut()
    assert app.current_cell_name == ""


def test_edit_action_enters_editor_and_updates_visible_mode():
    app = _make_app()
    object_class = ObjectClass(
        "Golgi",
        (0.2, 0.8, 1.0, 1.0),
        next_instance_index=2,
    )
    frame = RoiFrameRecord(
        timepoint=1,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.DRAFT,
        geometry=Polygon2D(15, ((1, 1), (4, 1), (1, 4))),
    )
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=1,
        frames={1: frame},
    )
    app.roi_manager = RoiManager(SubcellularRoiDocument(
        object_classes=(object_class,),
        objects=(track,),
    ))
    integration = _RoiIntegration(editing=False)
    panel = _RoiPanel("inspect")
    app._roi_viewer_integration = integration
    app._subcellular_objects_panel = panel

    app._on_roi_action_requested("edit", track.object_id)

    assert integration.edit_args is not None
    args, kwargs = integration.edit_args
    assert args[:2] == (track.object_id, 1)
    assert kwargs == {"z_plane": 15}
    assert panel.mode == "edit"


def test_draw_mode_uses_selected_class_without_mutating_manager():
    app = _make_app()
    object_class = app.roi_manager.create_class(
        "Golgi",
        (0.2, 0.8, 1.0, 1.0),
    )
    integration = _RoiIntegration(editing=False)
    panel = _RoiPanel("draw_polygon")
    panel.selected_class_id = object_class.class_id
    app._roi_viewer_integration = integration
    app._subcellular_objects_panel = panel

    app._on_roi_mode_changed("draw_polygon")

    assert app.roi_manager.objects == ()
    assert integration.draw_kwargs == {
        "class_id": object_class.class_id,
        "object_id": None,
        "timepoint": 1,
        "z_plane": 15,
        "kind": "draw_polygon",
        "cell_ref": None,
        "geometry": None,
    }


def test_contour_mode_continues_selected_stack_but_polygon_starts_new_object():
    app = _make_app()
    object_class = ObjectClass(
        "Membrane",
        (1.0, 0.5, 0.1, 1.0),
        next_instance_index=2,
    )
    stack = ContourStack3D((
        ContourSlice(14, ((1, 1), (4, 1), (1, 4))),
    ))
    frame = RoiFrameRecord(
        timepoint=1,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.DRAFT,
        geometry=stack,
    )
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=1,
        frames={1: frame},
    )
    app.roi_manager = RoiManager(SubcellularRoiDocument(
        object_classes=(object_class,),
        objects=(track,),
    ))
    app.current_roi_object_id = track.object_id
    app.current_roi_class_id = object_class.class_id
    integration = _RoiIntegration(editing=False)
    panel = _RoiPanel("draw_contour_stack")
    panel.selected_class_id = object_class.class_id
    app._roi_viewer_integration = integration
    app._subcellular_objects_panel = panel

    app._on_roi_mode_changed("draw_contour_stack")
    assert integration.draw_kwargs["object_id"] == track.object_id
    assert integration.draw_kwargs["geometry"] == stack

    integration.editing = False
    panel.mode = "draw_polygon"
    app._on_roi_mode_changed("draw_polygon")
    assert integration.draw_kwargs["object_id"] is None
    assert integration.draw_kwargs["geometry"] is None


def test_pick_cell_action_uses_same_frame_picker_and_is_undoable():
    app = _make_app()
    object_class = app.roi_manager.create_class(
        "Golgi",
        (0.2, 0.8, 1.0, 1.0),
    )
    track = app.roi_manager.create_object(object_class.class_id)
    app.roi_manager.set_frame(
        track.object_id,
        RoiFrameRecord(
            timepoint=1,
            presence=Presence.SEGMENTED,
            review_state=ReviewState.DRAFT,
            geometry=Polygon2D(15, ((1, 1), (4, 1), (1, 4))),
        ),
    )
    integration = _RoiIntegration(editing=False)
    panel = _RoiPanel("inspect")
    app._roi_viewer_integration = integration
    app._subcellular_objects_panel = panel

    app._on_roi_action_requested("pick_cell", track.object_id)
    assert app._relink_pick_mode
    assert app._relink_pick_callback is not None

    nucleus = app.manager.nuclei_at(1)[0]
    app._relink_pick_callback(1, nucleus)

    associated = app.roi_manager.require_object(track.object_id).frames[1]
    assert associated.cell_ref is not None
    assert associated.cell_ref.nucleus_anchor.timepoint == 1
    assert associated.cell_ref.nucleus_anchor.index == nucleus.index
    assert app.edit_history.undo_description == "Set ROI association at t=1"

    app.edit_history.undo()
    restored = app.roi_manager.require_object(track.object_id).frames[1]
    assert restored.cell_ref is None


def test_plot_track_opens_and_tracks_scalar_window(monkeypatch):
    app = _make_app()
    object_class = app.roi_manager.create_class(
        "Golgi",
        (0.2, 0.8, 1.0, 1.0),
    )
    track = app.roi_manager.create_object(object_class.class_id)
    _ScalarWindow.created.clear()
    monkeypatch.setattr(
        "acetree_py.gui.roi_scalar_plot_window.RoiScalarPlotWindow",
        _ScalarWindow,
    )

    app._on_roi_action_requested("plot_track", track.object_id)

    assert len(_ScalarWindow.created) == 1
    window = _ScalarWindow.created[0]
    assert window.object_ids == (track.object_id,)
    assert window.parent is None
    assert window.shown
    assert app._roi_scalar_plot_windows == [window]
    window.destroyed.callback()
    assert app._roi_scalar_plot_windows == []


def test_published_roi_measurements_refresh_live_scalar_windows():
    app = _make_app()
    first = _ScalarWindow(app, (), None)
    second = _ScalarWindow(app, (), None)
    app._roi_scalar_plot_windows = [first, second]
    snapshot = object()

    app._refresh_roi_scalar_plot_windows(snapshot)

    assert first.snapshots == [snapshot]
    assert second.snapshots == [snapshot]
