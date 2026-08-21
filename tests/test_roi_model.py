"""Focused contract tests for the subcellular ROI domain and manager."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from acetree_py.core.roi_manager import RoiManager, RoiManagerError, RoiWriteProtectedError
from acetree_py.core.subcellular_roi import (
    AssociationStatus,
    CellRef,
    CellResolution,
    ContourSlice,
    ContourStack3D,
    CoordinateSpaceSnapshot,
    NucleusAnchor,
    ObjectClass,
    Polygon2D,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    RoiValidationError,
    SamplingMode,
    SubcellularRoiDocument,
    ThickPolyline2D,
    Thickness,
)


def _space() -> CoordinateSpaceSnapshot:
    return CoordinateSpaceSnapshot(
        plane_start=3,
        xy_res=0.2,
        z_res=1.0,
        image_width_px=100,
        image_height_px=80,
        plane_count=5,
        time_start=1,
        time_end=10,
    )


def _polygon(z: int = 3) -> Polygon2D:
    return Polygon2D(z, ((1, 1), (5, 1), (5, 5), (1, 5), (1, 1)))


def test_polygon_is_frozen_removes_closing_vertex_and_canonicalizes_winding():
    polygon = Polygon2D(3, ((1, 1), (1, 5), (5, 5), (5, 1), (1, 1)))

    assert len(polygon.exterior_xy_px) == 4
    assert polygon.exterior_xy_px[0] == (1.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        polygon.z_plane = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "points,match",
    [
        (((0, 0), (1, 1), (0, 1), (1, 0)), "self-intersect"),
        (((0, 0), (1, 0), (2, 0)), "non-zero area"),
        (((0, 0), (1, float("nan")), (0, 1)), "finite"),
    ],
)
def test_invalid_polygon_geometry_is_rejected(points, match):
    with pytest.raises(RoiValidationError, match=match):
        Polygon2D(3, points)


def test_polyline_and_contour_stack_validation():
    line = ThickPolyline2D(3, ((1, 1), (1, 1), (4, 5)), Thickness(0.8, "um"))
    assert line.points_xy_px[-1] == (4.0, 5.0)
    shell = ContourStack3D(
        (
            ContourSlice(4, ((1, 1), (5, 1), (3, 5))),
            ContourSlice(3, ((1, 1), (4, 1), (3, 4))),
        ),
        SamplingMode.INNER_SHELL,
        0.6,
    )
    assert [item.z_plane for item in shell.slices] == [3, 4]
    with pytest.raises(RoiValidationError, match="consecutive"):
        ContourStack3D(
            (
                ContourSlice(3, ((1, 1), (4, 1), (3, 4))),
                ContourSlice(5, ((1, 1), (4, 1), (3, 4))),
            )
        )


def test_presence_and_review_state_transitions_are_explicit():
    frame = RoiFrameRecord(2, Presence.SEGMENTED, ReviewState.REVIEWED, _polygon())
    edited = frame.with_geometry(Polygon2D(3, ((2, 2), (6, 2), (4, 6))))
    assert edited.review_state is ReviewState.NEEDS_REVIEW
    assert edited.revision == frame.revision + 1
    absent = edited.mark_absent()
    assert absent.presence is Presence.ABSENT
    assert absent.geometry is None
    assert absent.review_state is ReviewState.REVIEWED
    redrawn = absent.with_geometry(_polygon())
    assert redrawn.presence is Presence.SEGMENTED
    assert redrawn.review_state is ReviewState.DRAFT
    with pytest.raises(RoiValidationError, match="absent frame"):
        RoiFrameRecord(2, Presence.ABSENT, ReviewState.REVIEWED, _polygon())


def test_cell_anchor_must_be_same_frame_and_association_edit_needs_review():
    ref = CellRef(
        NucleusAnchor(2, 7),
        NucleusAnchor(1, 3),
        "ABa",
        (10, 11, 3.0),
    )
    frame = RoiFrameRecord(2, "segmented", "reviewed", _polygon(), ref)
    assert frame.with_association(None).review_state is ReviewState.NEEDS_REVIEW
    with pytest.raises(RoiValidationError, match="frame timepoint"):
        replace(frame, cell_ref=CellRef(NucleusAnchor(3, 7)))


def test_document_enforces_class_index_uniqueness_and_allocator_floor():
    object_class = ObjectClass("Golgi", (0.1, 0.2, 0.8, 1.0), next_instance_index=2)
    first = RoiObjectTrack(object_class.class_id, 1)
    with pytest.raises(RoiValidationError, match="duplicate class/instance"):
        SubcellularRoiDocument(
            coordinate_space=_space(),
            object_classes=(object_class,),
            objects=(first, replace(first, object_id=uuid4())),
        )
    with pytest.raises(RoiValidationError, match="allocator"):
        SubcellularRoiDocument(
            coordinate_space=_space(),
            object_classes=(replace(object_class, next_instance_index=1),),
            objects=(first,),
        )


def test_manager_allocates_monotonically_and_never_reuses_deleted_index():
    manager = RoiManager(SubcellularRoiDocument.empty(_space()))
    object_class = manager.create_class("Golgi", (0.1, 0.2, 0.8, 1.0))
    one = manager.create_object(object_class.class_id)
    manager.delete_object(one.object_id)
    two = manager.create_object(object_class.class_id)

    assert (one.instance_index, two.instance_index) == (1, 2)
    assert manager.roi_revision == 4
    assert manager.is_dirty


def test_manager_rejects_index_collision_and_snapshot_revision_goes_forward():
    manager = RoiManager(SubcellularRoiDocument.empty(_space()))
    object_class = manager.create_class("Golgi", (0.1, 0.2, 0.8, 1.0))
    first = manager.create_object(object_class.class_id)
    snapshot = manager.document
    manager.create_object(object_class.class_id)
    revision = manager.roi_revision
    manager.replace_document(snapshot)

    assert manager.roi_revision == revision + 1
    assert len(manager.objects) == 1
    with pytest.raises(RoiManagerError, match="already in use"):
        manager.create_object(object_class.class_id, instance_index=first.instance_index)


def test_reconciliation_is_exact_and_hint_changes_do_not_relink():
    manager = RoiManager(SubcellularRoiDocument.empty(_space()))
    object_class = manager.create_class("Golgi", (0.1, 0.2, 0.8, 1.0))
    ref = CellRef(NucleusAnchor(2, 7), name_snapshot="ABa")
    frame = RoiFrameRecord(2, "segmented", "reviewed", _polygon(), ref)
    track = manager.create_object(object_class.class_id, frames={2: frame})

    changed = manager.reconcile_cells(
        lambda time, index: CellResolution(object(), name="ABp")
    )
    orphaned = manager.reconcile_cells(lambda time, index: None)

    assert changed[0].object_id == track.object_id
    assert changed[0].status is AssociationStatus.HINT_CHANGED
    assert orphaned[0].status is AssociationStatus.ORPHANED
    assert manager.get_frame(track.object_id, 2).cell_ref == ref


def test_coordinate_mismatch_marks_geometry_and_blocks_physical_normalization():
    manager = RoiManager(SubcellularRoiDocument.empty(_space()))
    object_class = manager.create_class("Golgi", (0.1, 0.2, 0.8, 1.0))
    frame = RoiFrameRecord(2, "segmented", "reviewed", _polygon())
    track = manager.create_object(object_class.class_id, frames={2: frame})
    changed_space = replace(_space(), xy_res=0.3)

    assert manager.reconcile_coordinate_space(changed_space) == ("xy_res",)
    assert manager.physical_normalization_blocked
    assert manager.get_frame(track.object_id, 2).review_state is ReviewState.NEEDS_REVIEW


def test_newer_read_only_manager_rejects_every_mutation_funnel():
    manager = RoiManager(SubcellularRoiDocument.empty(_space()), read_only=True)
    with pytest.raises(RoiWriteProtectedError, match="read-only"):
        manager.create_class("Golgi", (0.1, 0.2, 0.8, 1.0))
    with pytest.raises(RoiWriteProtectedError, match="read-only"):
        manager.replace_document(manager.document)
    with pytest.raises(RoiWriteProtectedError, match="read-only"):
        manager.bump_revision()


def test_coordinate_space_from_config_uses_provider_dimensions():
    from acetree_py.core.roi_manager import coordinate_space_from_config

    config = SimpleNamespace(
        plane_start=4,
        plane_end=20,
        xy_res=0.2,
        z_res=1.0,
        starting_index=1,
        ending_index=99,
        split=1,
        flip=0,
    )
    provider = SimpleNamespace(num_planes=6, num_timepoints=12, image_shape=(80, 100))
    space = coordinate_space_from_config(config, image_provider=provider)
    assert (space.image_width_px, space.image_height_px) == (100, 80)
    assert space.plane_count == 6
    assert space.time_end == 12
