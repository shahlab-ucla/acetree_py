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
