from __future__ import annotations

import pytest

from acetree_py.core.roi_manager import RoiManager, RoiWriteProtectedError
from acetree_py.core.subcellular_roi import (
    ObjectClass,
    Polygon2D,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
)
from acetree_py.editing.commands import (
    CompositeCommand,
    EditEffect,
    MoveNucleus,
)
from acetree_py.editing.history import EditHistory
from acetree_py.editing.roi_commands import (
    AssociateRoiFrame,
    CopyRoiFrameDraft,
    CreateObjectClass,
    CreateRoiObject,
    DeleteObjectClass,
    DeleteRoiFrame,
    DeleteRoiObject,
    MarkRoiFrameAbsent,
    MarkRoiFrameReviewed,
    ReclassifyRoiObject,
    ReindexRoiObject,
    SetRoiFrameGeometry,
    UpdateObjectClass,
)


def _manager_and_track():
    object_class = ObjectClass(
        "Golgi", (0.2, 0.8, 1.0, 1.0), next_instance_index=2
    )
    frame = RoiFrameRecord(
        timepoint=1,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.REVIEWED,
        geometry=Polygon2D(2, ((1, 1), (4, 1), (1, 4))),
    )
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=1,
        frames={1: frame},
    )
    manager = RoiManager(SubcellularRoiDocument(
        object_classes=(object_class,),
        objects=(track,),
    ))
    return manager, track


def test_effect_defaults_and_composite_union_remain_compatible():
    nucleus_move = MoveNucleus(time=1, index=1, new_x=3)
    assert nucleus_move.effects == {EditEffect.NUCLEI_TOPOLOGY}
    manager, track = _manager_and_track()
    roi_command = ReindexRoiObject(manager, track.object_id, 2)
    assert not roi_command.structural
    assert roi_command.effects == {EditEffect.ROI_METADATA}
    composite = CompositeCommand([nucleus_move, roi_command])
    assert composite.effects == {
        EditEffect.NUCLEI_TOPOLOGY,
        EditEffect.ROI_METADATA,
    }


def test_roi_commands_share_one_history_and_preserve_exact_undo_redo():
    manager, track = _manager_and_track()
    history = EditHistory([])
    revised = Polygon2D(2, ((2, 2), (5, 2), (2, 5)))
    command = SetRoiFrameGeometry(manager, track.object_id, 1, revised)
    history.do(command)
    assert manager.get_object(track.object_id).frames[1].geometry == revised
    assert manager.get_object(track.object_id).frames[1].review_state is ReviewState.NEEDS_REVIEW
    changed_revision = manager.roi_revision
    assert history.last_effects == {EditEffect.ROI_GEOMETRY}

    history.undo()
    assert manager.get_object(track.object_id).frames[1].geometry == track.frames[1].geometry
    assert manager.roi_revision > changed_revision
    history.redo()
    assert manager.get_object(track.object_id).frames[1].geometry == revised


def test_review_absence_copy_and_association_are_undoable():
    manager, track = _manager_and_track()
    history = EditHistory([])
    history.do(CopyRoiFrameDraft(manager, track.object_id, 1, 2))
    copied = manager.get_object(track.object_id).frames[2]
    assert copied.review_state is ReviewState.DRAFT
    history.do(MarkRoiFrameReviewed(manager, track.object_id, 2))
    assert manager.get_object(track.object_id).frames[2].review_state is ReviewState.REVIEWED
    history.do(AssociateRoiFrame(manager, track.object_id, 2, None))
    history.do(MarkRoiFrameAbsent(manager, track.object_id, 2))
    assert manager.get_object(track.object_id).frames[2].presence is Presence.ABSENT
    history.undo()
    assert manager.get_object(track.object_id).frames[2].presence is Presence.SEGMENTED


def test_write_protection_error_is_not_masked_by_rollback():
    manager, track = _manager_and_track()
    protected = RoiManager(manager.document, read_only=True)
    history = EditHistory([])
    with pytest.raises(RoiWriteProtectedError, match="newer schema"):
        history.do(ReindexRoiObject(protected, track.object_id, 2))
    assert protected.document == manager.document
    assert not history.can_undo


def test_class_and_object_lifecycle_commands_preserve_uuids():
    manager = RoiManager()
    history = EditHistory([])
    create_class = CreateObjectClass(manager, "Golgi", (0.2, 0.8, 1.0, 1.0))
    history.do(create_class)
    class_id = create_class.created_class_id
    assert manager.get_class(class_id).name == "Golgi"

    history.do(UpdateObjectClass(manager, class_id, name="cis-Golgi"))
    assert manager.get_class(class_id).name == "cis-Golgi"
    create_object = CreateRoiObject(manager, class_id)
    history.do(create_object)
    object_id = create_object.created_object_id
    assert manager.get_object(object_id).class_id == class_id

    history.do(DeleteRoiObject(manager, object_id))
    assert manager.get_object(object_id) is None
    history.undo()
    assert manager.get_object(object_id).object_id == object_id
    history.redo()
    assert manager.get_object(object_id) is None
    history.undo()

    history.do(DeleteRoiObject(manager, object_id))
    history.do(DeleteObjectClass(manager, class_id))
    assert manager.get_class(class_id) is None
    history.undo()
    assert manager.get_class(class_id).class_id == class_id


def test_reclass_delete_frame_and_undo_restore_exact_track():
    manager, original = _manager_and_track()
    history = EditHistory([])
    second = manager.create_class("Membrane", (1.0, 0.5, 0.1, 1.0))
    history.mark_saved()
    command = ReclassifyRoiObject(manager, original.object_id, second.class_id)
    history.do(command)
    reclassified = manager.get_object(original.object_id)
    assert reclassified.object_id == original.object_id
    assert reclassified.class_id == second.class_id
    assert reclassified.instance_index == 1
    history.undo()
    assert manager.get_object(original.object_id) == original

    history.do(DeleteRoiFrame(manager, original.object_id, 1))
    assert manager.get_object(original.object_id).frames == {}
    history.undo()
    assert manager.get_object(original.object_id).frames[1] == original.frames[1]
