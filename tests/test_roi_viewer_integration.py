from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from acetree_py.core.subcellular_roi import (
    CellRef,
    ContourSlice,
    ContourStack3D,
    NucleusAnchor,
    ObjectClass,
    Polygon2D,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
)
from acetree_py.core.roi_manager import RoiManager
from acetree_py.editing.history import EditHistory
from acetree_py.gui.roi_viewer_integration import (
    EDITOR_LAYER_NAME,
    OVERLAY_LAYER_NAME,
    RoiViewerIntegration,
    model_xy_to_napari,
    napari_yx_to_model,
    roi_overlay_shapes,
)


class _Layer:
    def __init__(self, data, *, name, visible=True, **_kwargs):
        self.data = list(data)
        self.name = name
        self.visible = visible
        self.editable = True
        self.mode = "select"
        self.shape_type = []
        self.edge_color = []
        self.face_color = []
        self.edge_width = []
        self.properties = {}
        self.text = {}
        self.fail_next_add = False

    def add(self, data, *, shape_type, edge_color=None, face_color=None, edge_width=None):
        self.data.extend(data)
        if self.fail_next_add:
            self.fail_next_add = False
            raise RuntimeError("injected redraw failure")
        self.shape_type = shape_type
        self.edge_color = edge_color
        self.face_color = face_color
        self.edge_width = edge_width


class _Viewer:
    def __init__(self):
        self.layers = []
        self.layers_selection = SimpleNamespace(active=None)

    def add_shapes(self, data, **kwargs):
        layer = _Layer(data, **kwargs)
        self.layers.append(layer)
        return layer


class _Layers(list):
    def __init__(self):
        super().__init__()
        self.selection = SimpleNamespace(active=None)


class _RealishViewer:
    def __init__(self):
        self.layers = _Layers()

    def add_shapes(self, data, **kwargs):
        layer = _Layer(data, **kwargs)
        self.layers.append(layer)
        return layer


def _document():
    object_class = ObjectClass(
        "Golgi", (0.2, 0.8, 1.0, 1.0), next_instance_index=3
    )
    frame = RoiFrameRecord(
        timepoint=4,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.DRAFT,
        geometry=Polygon2D(12, ((10, 20), (18, 21), (14, 30))),
    )
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=2,
        frames={4: frame},
    )
    return SubcellularRoiDocument(
        object_classes=(object_class,),
        objects=(track,),
    ), track


def test_coordinate_conversion_swaps_only_at_boundary():
    napari = model_xy_to_napari(((1, 2), (3, 4)))
    np.testing.assert_array_equal(napari, np.array(((2, 1), (4, 3))))
    assert napari_yx_to_model(napari) == ((1.0, 2.0), (3.0, 4.0))


def test_projection_filters_absolute_time_and_z():
    document, track = _document()
    manager = SimpleNamespace(document=document)
    assert len(roi_overlay_shapes(manager, 4, 12)) == 1
    assert roi_overlay_shapes(manager, 3, 12) == ()
    assert roi_overlay_shapes(manager, 4, 11) == ()
    projected = roi_overlay_shapes(manager, 4, 12)[0]
    assert projected.object_id == track.object_id
    np.testing.assert_array_equal(projected.data_yx[0], (20, 10))


def test_overlay_editor_separation_atomic_redraw_and_3d_cleanup():
    document, track = _document()
    viewer = _RealishViewer()
    app = SimpleNamespace(viewer=viewer, current_time=4, current_plane=12)
    manager = SimpleNamespace(document=document)
    integration = RoiViewerIntegration(app, manager)
    integration.setup_layers()

    assert integration.overlay_layer.name == OVERLAY_LAYER_NAME
    assert integration.editor_layer.name == EDITOR_LAYER_NAME
    assert integration.overlay_layer.editable is False
    assert integration.editor_layer.visible is False
    assert len(integration.overlay_layer.data) == 1

    prior = [np.array(item, copy=True) for item in integration.overlay_layer.data]
    integration.overlay_layer.fail_next_add = True
    with pytest.raises(RuntimeError, match="prior complete projection"):
        integration.update_overlay()
    assert len(integration.overlay_layer.data) == len(prior)
    np.testing.assert_array_equal(integration.overlay_layer.data[0], prior[0])

    geometry = track.frames[4].geometry
    integration.enter_edit_mode(track.object_id, 4, geometry)
    assert integration.editing
    assert integration.editor_layer.visible
    integration.set_three_dimensional(True)
    assert not integration.editing
    assert not integration.editor_layer.visible
    assert not integration.overlay_layer.visible
    integration.set_three_dimensional(False)
    assert integration.overlay_layer.visible
    assert len(integration.overlay_layer.data) == 1


def test_editor_drag_commits_once_at_release():
    document, track = _document()
    viewer = _RealishViewer()
    app = SimpleNamespace(viewer=viewer, current_time=4, current_plane=12)
    submitted = []
    integration = RoiViewerIntegration(
        app,
        SimpleNamespace(document=document),
        command_sink=submitted.append,
    )
    integration.setup_layers()
    integration.enter_edit_mode(track.object_id, 4, track.frames[4].geometry)
    integration.editor_layer.data = [
        np.array(((21.0, 11.0), (22.0, 19.0), (31.0, 15.0)))
    ]
    event = SimpleNamespace(type="mouse_press")
    gesture = integration._on_editor_drag(integration.editor_layer, event)
    next(gesture)
    assert submitted == []
    event.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(gesture)
    assert len(submitted) == 1
    assert submitted[0].geometry.exterior_xy_px == (
        (11.0, 21.0),
        (19.0, 22.0),
        (15.0, 31.0),
    )
    assert not integration.editing


def test_new_object_drawing_commits_atomically_and_cancel_is_clean():
    object_class = ObjectClass("Golgi", (0.2, 0.8, 1.0, 1.0))
    manager = RoiManager(SubcellularRoiDocument(object_classes=(object_class,)))
    viewer = _RealishViewer()

    class _Panel:
        mode = "draw_polygon"

        def __init__(self):
            self.selected = None

        def set_mode(self, mode):
            self.mode = mode

        def refresh(self):
            pass

        def select_object(self, object_id):
            self.selected = object_id

    panel = _Panel()
    app = SimpleNamespace(
        viewer=viewer,
        current_time=4,
        current_plane=12,
        edit_history=EditHistory([]),
        _subcellular_objects_panel=panel,
    )

    def select_object(object_id):
        app.current_roi_object_id = object_id

    app._on_roi_object_selected = select_object
    integration = RoiViewerIntegration(app, manager)
    integration.setup_layers()
    cell_ref = CellRef(
        nucleus_anchor=NucleusAnchor(4, 7),
        name_snapshot="ABpl",
    )
    object_id = integration.begin_drawing(
        class_id=object_class.class_id,
        timepoint=4,
        z_plane=12,
        kind="draw_polygon",
        cell_ref=cell_ref,
    )
    assert manager.objects == ()
    assert integration.editor_layer.mode == "add_polygon"
    integration.editor_layer.data = [
        np.array(((20.0, 10.0), (21.0, 18.0), (30.0, 14.0)))
    ]
    integration.finish_edit()

    created = manager.get_object(object_id)
    assert created is not None
    assert created.class_id == object_class.class_id
    assert created.frames[4].geometry.z_plane == 12
    assert created.frames[4].review_state is ReviewState.DRAFT
    assert created.frames[4].cell_ref == cell_ref
    assert app.current_roi_object_id == object_id
    assert panel.selected == object_id
    assert panel.mode == "inspect"

    app.edit_history.undo()
    assert manager.get_object(object_id) is None
    app.edit_history.redo()
    assert manager.get_object(object_id).object_id == object_id

    before = manager.document
    integration.begin_drawing(
        class_id=object_class.class_id,
        timepoint=5,
        z_plane=12,
        kind="draw_polygon",
    )
    integration.cancel_edit()
    assert manager.document == before


def test_contour_drawing_appends_consecutive_slice_to_same_object():
    object_class = ObjectClass(
        "Membrane",
        (1.0, 0.5, 0.1, 1.0),
        next_instance_index=2,
    )
    geometry = ContourStack3D((
        ContourSlice(12, ((10, 20), (18, 21), (14, 30))),
    ))
    frame = RoiFrameRecord(
        timepoint=4,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.DRAFT,
        geometry=geometry,
    )
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=1,
        frames={4: frame},
    )
    manager = RoiManager(SubcellularRoiDocument(
        object_classes=(object_class,),
        objects=(track,),
    ))
    viewer = _RealishViewer()
    app = SimpleNamespace(
        viewer=viewer,
        current_time=4,
        current_plane=13,
        edit_history=EditHistory([]),
    )
    integration = RoiViewerIntegration(app, manager)
    integration.setup_layers()
    integration.begin_drawing(
        class_id=object_class.class_id,
        object_id=track.object_id,
        timepoint=4,
        z_plane=13,
        kind="draw_contour_stack",
        geometry=geometry,
    )
    integration.editor_layer.data = [
        np.array(((20.0, 11.0), (22.0, 19.0), (31.0, 15.0)))
    ]
    integration.finish_edit()

    assert len(manager.objects) == 1
    updated = manager.get_object(track.object_id).frames[4].geometry
    assert tuple(item.z_plane for item in updated.slices) == (12, 13)
    assert updated.slices[0] == geometry.slices[0]
    app.edit_history.undo()
    restored = manager.get_object(track.object_id).frames[4].geometry
    assert tuple(item.z_plane for item in restored.slices) == (12,)
    NucleusAnchor,
