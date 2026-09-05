from __future__ import annotations

from types import SimpleNamespace

import pytest

from acetree_py.core.subcellular_roi import (
    CellRef,
    NucleusAnchor,
    ObjectClass,
    Polygon2D,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
)
from acetree_py.gui.subcellular_objects_panel import (
    RoiInteractionMode,
    SubcellularObjectsPanel,
    filter_object_rows,
    object_browser_rows,
)


def _manager(*, protected=False):
    object_class = ObjectClass(
        "Membrane",
        (0.9, 0.5, 0.1, 1.0),
        next_instance_index=4,
    )
    cell_ref = CellRef(
        nucleus_anchor=NucleusAnchor(timepoint=42, index=7),
        name_snapshot="ABpl",
    )
    frame = RoiFrameRecord(
        timepoint=42,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.NEEDS_REVIEW,
        geometry=Polygon2D(15, ((1, 1), (4, 1), (1, 4))),
        cell_ref=cell_ref,
    )
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=3,
        expected_start_time=40,
        expected_end_time=45,
        frames={42: frame},
    )
    document = SubcellularRoiDocument(
        object_classes=(object_class,),
        objects=(track,),
    )
    return SimpleNamespace(
        document=document,
        classes=document.object_classes,
        objects=document.objects,
        is_write_protected=protected,
        read_only=False,
        is_dirty=True,
        sidecar_path="embryo.subcellular-rois.json",
        load_error="invalid checksum" if protected else None,
        get_class=lambda class_id: object_class if class_id == object_class.class_id else None,
        get_object=lambda object_id: track if object_id == track.object_id else None,
    ), object_class, track


def test_rows_include_identity_state_association_and_filters():
    manager, object_class, track = _manager()
    rows = object_browser_rows(manager, 42)
    assert len(rows) == 1
    assert rows[0].label == "Membrane #3"
    assert rows[0].association == "ABpl"
    assert rows[0].current_state == "needs_review"
    assert rows[0].status_text == "Needs review"
    assert filter_object_rows(rows, class_id=object_class.class_id) == rows
    assert filter_object_rows(rows, state="draft") == ()
    assert filter_object_rows(
        rows,
        cell_scope="current",
        selected_cell_name="ABpl",
        search="membrane",
    ) == rows


def test_panel_is_browse_only_and_accessible(qtbot):
    pytest.importorskip("qtpy")
    manager, object_class, track = _manager(protected=True)
    app = SimpleNamespace(
        roi_manager=manager,
        current_time=42,
        current_plane=15,
        current_cell_name="ABpl",
    )
    panel = SubcellularObjectsPanel(app, browse_only=False)
    qtbot.addWidget(panel)

    assert panel.browse_only
    assert panel._track_list.count() == 1
    assert not panel._btn_polygon.isEnabled()
    assert panel._btn_polygon.accessibleName()
    assert not panel.set_mode(RoiInteractionMode.DRAW_POLYGON)
    assert panel.mode is RoiInteractionMode.INSPECT
    assert panel.next_instance_index(object_class.class_id) == 4
    panel.select_object(track.object_id)
    assert panel.current_object_id == track.object_id


def test_plot_track_and_spatial_profiles_are_separate_actions(qtbot):
    pytest.importorskip("qtpy")
    manager, _object_class, track = _manager()
    app = SimpleNamespace(
        roi_manager=manager,
        current_time=42,
        current_plane=15,
        current_cell_name="ABpl",
    )
    panel = SubcellularObjectsPanel(app, browse_only=False)
    qtbot.addWidget(panel)
    panel.select_object(track.object_id)
    emitted = []
    panel.actionRequested.connect(
        lambda action, object_id: emitted.append((action, object_id))
    )

    panel._btn_plot.click()
    panel._btn_profiles.click()

    assert emitted == [
        ("plot_track", track.object_id),
        ("plot_profiles", track.object_id),
    ]
    assert "scalar" in panel._btn_plot.accessibleName().lower()
    assert "spatial" in panel._btn_profiles.accessibleName().lower()


def test_filters_clear_hidden_targets_and_drive_measurement_scope(qtbot):
    from acetree_py.core.roi_manager import RoiManager
    from acetree_py.gui.roi_measure_dialog import RoiMeasureDialog

    source, _first_class, first_track = _manager()
    manager = RoiManager(source.document)
    second_class = manager.create_class("Golgi", (0.2, 0.8, 1.0, 1.0))
    second_track = manager.create_object(second_class.class_id)
    overlay_filters = []
    cancelled_edits = []
    app = SimpleNamespace(
        roi_manager=manager,
        current_time=42,
        current_plane=15,
        current_cell_name="",
        current_roi_object_id=None,
        current_roi_class_id=None,
        image_provider=SimpleNamespace(num_channels=2),
        _roi_viewer_integration=SimpleNamespace(
            set_visible_object_ids=lambda ids: overlay_filters.append(ids),
            cancel_edit=lambda: cancelled_edits.append(True),
        ),
    )
    panel = SubcellularObjectsPanel(app, browse_only=False)
    app._subcellular_objects_panel = panel
    qtbot.addWidget(panel)

    def selection_changed(object_id):
        app.current_roi_object_id = object_id
        track = manager.get_object(object_id) if object_id else None
        app.current_roi_class_id = None if track is None else track.class_id

    panel.objectSelected.connect(selection_changed)
    panel.select_object(first_track.object_id)
    panel.select_class(second_class.class_id)
    assert panel.current_object_id is None
    assert app.current_roi_object_id is None
    assert panel._track_list.currentItem() is None
    assert not panel._btn_delete.isEnabled()
    assert overlay_filters[-1] == {second_track.object_id}

    dialog = RoiMeasureDialog(app)
    qtbot.addWidget(dialog)
    dialog._scope_combo.setCurrentIndex(dialog._scope_combo.findData("class"))
    assert dialog.build_request().object_ids == (str(second_track.object_id),)

    panel.select_class(None)
    panel._cell_combo.setCurrentIndex(panel._cell_combo.findData("current"))
    assert panel._track_list.count() == 0
    assert overlay_filters[-1] == set()
    assert "Select a cell" in panel._empty_label.text()
    app.current_cell_name = "ABpl"
    panel.refresh()
    assert overlay_filters[-1] == {first_track.object_id}
    panel.select_object(first_track.object_id)
    panel.set_mode("edit")
    panel._search_edit.setText("no matching object")
    panel.refresh()
    assert panel.current_object_id is None
    assert app.current_roi_object_id is None
    assert not panel._btn_edit.isEnabled()
    assert cancelled_edits == [True]
    assert panel.mode is RoiInteractionMode.INSPECT
