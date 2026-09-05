from dataclasses import replace

import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtGui import QColor
from qtpy.QtWidgets import QDialog

from acetree_py.core.roi_manager import RoiManager
from acetree_py.core.subcellular_roi import Polygon2D, Presence, ReviewState, RoiFrameRecord
from acetree_py.editing.history import EditHistory
from acetree_py.editing.roi_commands import UpdateRoiObjectSpan
from acetree_py.gui.roi_object_dialogs import RoiClassDialog, RoiSpanDialog
from acetree_py.gui.subcellular_objects_panel import ObjectBrowserModel


def test_class_management_preserves_identity_and_undo(qtbot, monkeypatch):
    manager = RoiManager()
    original = manager.create_class("Golgi", (0.2, 0.8, 1.0, 1.0))
    track = manager.create_object(original.class_id)
    original = manager.require_class(original.class_id)
    history = EditHistory([])
    dialog = RoiClassDialog(manager, history.do, selected_class_id=original.class_id)
    qtbot.addWidget(dialog)
    assert not dialog._delete_button.isEnabled()
    dialog._name_edit.setText("Membrane")
    monkeypatch.setattr(
        "acetree_py.gui.roi_object_dialogs.QColorDialog.getColor",
        lambda *_args: QColor("#cc7733"),
    )
    dialog._color_button.click()
    dialog._save_button.click()
    updated = manager.require_class(original.class_id)
    assert updated.name == "Membrane"
    assert updated.color_rgba != original.color_rgba
    assert manager.require_object(track.object_id).class_id == original.class_id
    history.undo()
    assert manager.require_class(original.class_id) == original
    history.redo()
    assert manager.require_class(original.class_id) == updated

    dialog._class_combo.setCurrentIndex(0)
    dialog._name_edit.setText("Unused class")
    dialog._save_button.click()
    created_id = dialog.selected_class_id
    assert manager.require_class(created_id).name == "Unused class"
    assert dialog._delete_button.isEnabled()
    dialog._delete_button.click()
    assert manager.get_class(created_id) is None
    history.undo()
    assert manager.require_class(created_id).name == "Unused class"


def test_expected_span_validates_observations_and_counts_reviewed_absence(qtbot):
    manager = RoiManager()
    object_class = manager.create_class("Golgi", (0.2, 0.8, 1.0, 1.0))
    segmented = RoiFrameRecord(
        timepoint=2,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.REVIEWED,
        geometry=Polygon2D(1, ((1, 1), (4, 1), (1, 4))),
    )
    absent = RoiFrameRecord(3, Presence.ABSENT, ReviewState.REVIEWED)
    track = manager.create_object(object_class.class_id, frames={2: segmented, 3: absent})
    history = EditHistory([])
    dialog = RoiSpanDialog(track)
    qtbot.addWidget(dialog)
    dialog._start.setValue(3)
    dialog._accept_if_valid()
    assert dialog.result() == QDialog.Rejected
    assert "precedes" in dialog._status.text()
    dialog._start.setValue(1)
    dialog._end.setValue(3)
    dialog._accept_if_valid()
    assert dialog.result() == QDialog.Accepted
    history.do(UpdateRoiObjectSpan(manager, track.object_id, **dialog.get_values()))
    assert manager.require_object(track.object_id).expected_start_time == 1
    history.undo()
    assert manager.require_object(track.object_id).expected_start_time is None
    history.redo()

    model = ObjectBrowserModel()
    rows = model.rows(manager, 2)
    assert rows[0].status_text == "In progress"
    assert rows[0].reviewed_decision_count == 2
    assert rows[0].expected_count == 3
    assert model.rows(manager, 2) is rows
    assert model.rows(manager, 3)[0].current_state == "absent"
    undecided_absence = RoiFrameRecord(1, Presence.ABSENT, ReviewState.DRAFT)
    manager.set_frame(track.object_id, undecided_absence)
    assert model.rows(manager, 3)[0].status_text == "In progress"
    manager.set_frame(track.object_id, undecided_absence.mark_reviewed())
    assert model.rows(manager, 3)[0].status_text == "Complete"
    manager.set_frame(track.object_id, replace(absent, review_state=ReviewState.NEEDS_REVIEW))
    assert model.rows(manager, 3)[0].status_text == "Needs review"
