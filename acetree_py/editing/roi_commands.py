"""Undoable commands for the subcellular-ROI annotation stream.

The commands deliberately use the same :class:`EditHistory` protocol as
nucleus edits.  They capture immutable document boundaries instead of
individual mutable fields; this preserves UUIDs, allocators, and frame
revisions exactly across undo/redo while :meth:`RoiManager.replace_document`
keeps the manager's in-memory revision token monotonic.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import logging
from typing import Any, ClassVar
from uuid import UUID

from .commands import EditCommand, EditEffect, NucleiRecord

logger = logging.getLogger(__name__)


def _snapshot_document(manager: Any) -> Any:
    document = getattr(manager, "document")
    try:
        return deepcopy(document)
    except (TypeError, ValueError):
        # The production document is immutable.  MappingProxyType fields are
        # intentionally not deepcopyable on some Python versions, so the
        # immutable reference itself is a safe command boundary.
        return document


def _replace_document(manager: Any, document: Any) -> None:
    replacer = getattr(manager, "replace_document", None)
    if callable(replacer):
        replacer(document)
        return
    # Useful for small headless test doubles and older transitional managers.
    try:
        manager.document = document
    except (AttributeError, TypeError):
        manager._document = document


class RoiCommand(EditCommand):
    """Base class implementing exact immutable-document undo/redo."""

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset()

    _before_document: Any = None
    _after_document: Any = None
    _executed_once: bool = False
    _noop: bool = False

    @property
    def structural(self) -> bool:
        """ROI commands never trigger the legacy nuclear naming rebuild."""

        return False

    @property
    def effects(self) -> frozenset[EditEffect]:
        return self.EFFECTS

    @property
    def is_noop(self) -> bool:
        return self._noop

    def execute(self, nuclei_record: NucleiRecord) -> None:  # noqa: ARG002
        if self._executed_once:
            self._mark_rollback_ready()
            _replace_document(self.manager, self._after_document)
            return

        self._before_document = _snapshot_document(self.manager)
        self._mark_rollback_ready()
        try:
            self._apply()
        except Exception:
            # Some higher-level commands legitimately compose two manager
            # mutations (for example allocator advance plus reclassification).
            # Restore their complete scientific boundary before propagating.
            current = getattr(self.manager, "document", None)
            if current != self._before_document:
                try:
                    _replace_document(self.manager, self._before_document)
                except Exception:
                    # A protection transition can reject rollback as well.
                    # Preserve the original command failure; it is the useful
                    # reason the user's edit was not accepted.
                    logger.exception("Could not restore a failed ROI command boundary")
            self._failed_execute_rollback_ready = False
            raise
        self._after_document = _snapshot_document(self.manager)
        self._noop = self._before_document == self._after_document
        self._executed_once = True

    def undo(self, nuclei_record: NucleiRecord) -> None:  # noqa: ARG002
        if self._before_document is not None:
            _replace_document(self.manager, self._before_document)

    def _apply(self) -> None:
        raise NotImplementedError


@dataclass
class CreateObjectClass(RoiCommand):
    manager: Any
    name: str
    color_rgba: tuple[float, float, float, float]
    default_geometry_kind: Any = None

    created_class_id: UUID | None = field(default=None, init=False)

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_METADATA,)
    )

    def _apply(self) -> None:
        created = self.manager.create_class(
            self.name,
            self.color_rgba,
            default_geometry_kind=self.default_geometry_kind,
        )
        self.created_class_id = getattr(created, "class_id", created)

    @property
    def description(self) -> str:
        return f"Create ROI class {self.name}"


@dataclass
class UpdateObjectClass(RoiCommand):
    manager: Any
    class_id: UUID
    name: str | None = None
    color_rgba: tuple[float, float, float, float] | None = None
    default_geometry_kind: Any = None
    update_default_geometry_kind: bool = False

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_METADATA,)
    )

    def _apply(self) -> None:
        changes: dict[str, Any] = {}
        if self.name is not None:
            changes["name"] = self.name
        if self.color_rgba is not None:
            changes["color_rgba"] = self.color_rgba
        if self.update_default_geometry_kind:
            changes["default_geometry_kind"] = self.default_geometry_kind
        self.manager.update_class(
            self.class_id,
            name=changes.get("name"),
            color_rgba=changes.get("color_rgba"),
            default_geometry_kind=changes.get("default_geometry_kind"),
            update_default_geometry_kind=self.update_default_geometry_kind,
        )

    @property
    def description(self) -> str:
        return f"Update ROI class {self.class_id}"


@dataclass
class DeleteObjectClass(RoiCommand):
    manager: Any
    class_id: UUID

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_METADATA,)
    )

    def _apply(self) -> None:
        self.manager.delete_class(self.class_id)

    @property
    def description(self) -> str:
        return f"Delete ROI class {self.class_id}"


@dataclass
class CreateRoiObject(RoiCommand):
    manager: Any
    class_id: UUID
    instance_index: int | None = None
    expected_start_time: int | None = None
    expected_end_time: int | None = None
    object_id: UUID | None = None

    created_object_id: UUID | None = field(default=None, init=False)

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_METADATA,)
    )

    def _apply(self) -> None:
        kwargs = {
            "instance_index": self.instance_index,
            "expected_start_time": self.expected_start_time,
            "expected_end_time": self.expected_end_time,
        }
        if self.object_id is not None:
            kwargs["object_id"] = self.object_id
        try:
            created = self.manager.create_object(self.class_id, **kwargs)
        except TypeError:
            kwargs.pop("object_id", None)
            created = self.manager.create_object(self.class_id, **kwargs)
        if self.object_id is not None and getattr(created, "object_id", None) != self.object_id:
            document = self.manager.document
            updated = replace(created, object_id=self.object_id)
            self.manager.replace_document(
                replace(
                    document,
                    objects=tuple(
                        updated if item.object_id == created.object_id else item
                        for item in document.objects
                    ),
                )
            )
            created = updated
        self.created_object_id = getattr(created, "object_id", created)

    @property
    def description(self) -> str:
        suffix = (
            f" #{self.instance_index}" if self.instance_index is not None else ""
        )
        return f"Create ROI object{suffix}"


@dataclass
class UpdateRoiObjectSpan(RoiCommand):
    manager: Any
    object_id: UUID
    expected_start_time: int | None
    expected_end_time: int | None

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_METADATA,)
    )

    def _apply(self) -> None:
        track = self.manager.get_object(self.object_id)
        self.manager.update_object(
            replace(
                track,
                expected_start_time=self.expected_start_time,
                expected_end_time=self.expected_end_time,
            )
        )

    @property
    def description(self) -> str:
        return f"Set expected span for ROI object {self.object_id}"


@dataclass
class DeleteRoiObject(RoiCommand):
    manager: Any
    object_id: UUID

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (
            EditEffect.ROI_GEOMETRY,
            EditEffect.ROI_ASSOCIATION,
            EditEffect.ROI_METADATA,
        )
    )

    def _apply(self) -> None:
        self.manager.delete_object(self.object_id)

    @property
    def description(self) -> str:
        return f"Delete ROI object {self.object_id}"


@dataclass
class ReclassifyRoiObject(RoiCommand):
    manager: Any
    object_id: UUID
    class_id: UUID
    instance_index: int | None = None

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_METADATA,)
    )

    def _apply(self) -> None:
        track = self.manager.get_object(self.object_id)
        index = self.instance_index
        if index is None:
            index = self.manager.allocate_instance_index(self.class_id)
        self.manager.update_object(
            replace(track, class_id=self.class_id, instance_index=index)
        )

    @property
    def description(self) -> str:
        return f"Reclassify ROI object {self.object_id}"


@dataclass
class ReindexRoiObject(RoiCommand):
    manager: Any
    object_id: UUID
    instance_index: int

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_METADATA,)
    )

    def _apply(self) -> None:
        track = self.manager.get_object(self.object_id)
        self.manager.update_object(replace(track, instance_index=self.instance_index))

    @property
    def description(self) -> str:
        return f"Set ROI object {self.object_id} index to {self.instance_index}"


def _new_segmented_frame(timepoint: int, geometry: Any, cell_ref: Any = None) -> Any:
    from ..core.subcellular_roi import Presence, ReviewState, RoiFrameRecord

    return RoiFrameRecord(
        timepoint=timepoint,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.DRAFT,
        geometry=geometry,
        cell_ref=cell_ref,
    )


def _new_absent_frame(timepoint: int, cell_ref: Any = None) -> Any:
    from ..core.subcellular_roi import Presence, ReviewState, RoiFrameRecord

    return RoiFrameRecord(
        timepoint=timepoint,
        presence=Presence.ABSENT,
        review_state=ReviewState.REVIEWED,
        geometry=None,
        cell_ref=cell_ref,
    )


@dataclass
class SetRoiFrameGeometry(RoiCommand):
    manager: Any
    object_id: UUID
    timepoint: int
    geometry: Any
    cell_ref: Any = None
    preserve_association: bool = True

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_GEOMETRY,)
    )

    @property
    def effects(self) -> frozenset[EditEffect]:
        effects = set(self.EFFECTS)
        if self.cell_ref is not None or not self.preserve_association:
            effects.add(EditEffect.ROI_ASSOCIATION)
        return frozenset(effects)

    def _apply(self) -> None:
        track = self.manager.get_object(self.object_id)
        current = track.frames.get(self.timepoint)
        if current is None:
            frame = _new_segmented_frame(
                self.timepoint,
                self.geometry,
                self.cell_ref,
            )
        else:
            frame = current.with_geometry(self.geometry)
            if not self.preserve_association and frame.cell_ref != self.cell_ref:
                frame = frame.with_association(self.cell_ref)
        self.manager.set_frame(self.object_id, frame)

    @property
    def description(self) -> str:
        return f"Edit ROI geometry at t={self.timepoint}"


@dataclass
class DeleteRoiFrame(RoiCommand):
    manager: Any
    object_id: UUID
    timepoint: int

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_GEOMETRY, EditEffect.ROI_ASSOCIATION)
    )

    def _apply(self) -> None:
        self.manager.delete_frame(self.object_id, self.timepoint)

    @property
    def description(self) -> str:
        return f"Delete ROI frame at t={self.timepoint}"


@dataclass
class AssociateRoiFrame(RoiCommand):
    manager: Any
    object_id: UUID
    timepoint: int
    cell_ref: Any

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_ASSOCIATION,)
    )

    def _apply(self) -> None:
        track = self.manager.get_object(self.object_id)
        current = track.frames.get(self.timepoint)
        if current is None:
            raise ValueError(f"ROI object has no frame at t={self.timepoint}")
        self.manager.set_frame(
            self.object_id,
            current.with_association(self.cell_ref),
        )

    @property
    def description(self) -> str:
        action = "Clear" if self.cell_ref is None else "Set"
        return f"{action} ROI association at t={self.timepoint}"


@dataclass
class MarkRoiFrameReviewed(RoiCommand):
    manager: Any
    object_id: UUID
    timepoint: int

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_METADATA,)
    )

    def _apply(self) -> None:
        track = self.manager.get_object(self.object_id)
        current = track.frames.get(self.timepoint)
        if current is None:
            raise ValueError(f"ROI object has no frame at t={self.timepoint}")
        self.manager.set_frame(self.object_id, current.mark_reviewed())

    @property
    def description(self) -> str:
        return f"Mark ROI frame reviewed at t={self.timepoint}"


@dataclass
class MarkRoiFrameAbsent(RoiCommand):
    manager: Any
    object_id: UUID
    timepoint: int
    cell_ref: Any = None
    preserve_association: bool = True

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_GEOMETRY,)
    )

    @property
    def effects(self) -> frozenset[EditEffect]:
        effects = set(self.EFFECTS)
        if self.cell_ref is not None or not self.preserve_association:
            effects.add(EditEffect.ROI_ASSOCIATION)
        return frozenset(effects)

    def _apply(self) -> None:
        track = self.manager.get_object(self.object_id)
        current = track.frames.get(self.timepoint)
        if current is None:
            frame = _new_absent_frame(self.timepoint, self.cell_ref)
        else:
            frame = current.mark_absent()
            if not self.preserve_association and frame.cell_ref != self.cell_ref:
                frame = frame.with_association(self.cell_ref)
        self.manager.set_frame(self.object_id, frame)

    @property
    def description(self) -> str:
        return f"Mark ROI absent at t={self.timepoint}"


def _translated_geometry(geometry: Any, dx: float, dy: float) -> Any:
    """Return a translated geometry without depending on concrete unions."""

    if not dx and not dy:
        return geometry

    def points(values: Any) -> tuple[tuple[float, float], ...]:
        return tuple((float(x) + dx, float(y) + dy) for x, y in values)

    if hasattr(geometry, "exterior_xy_px"):
        return replace(geometry, exterior_xy_px=points(geometry.exterior_xy_px))
    if hasattr(geometry, "points_xy_px"):
        return replace(geometry, points_xy_px=points(geometry.points_xy_px))
    if hasattr(geometry, "slices"):
        return replace(
            geometry,
            slices=tuple(
                replace(item, exterior_xy_px=points(item.exterior_xy_px))
                for item in geometry.slices
            ),
        )
    if isinstance(geometry, dict):
        result = deepcopy(geometry)
        if "exterior_xy_px" in result:
            result["exterior_xy_px"] = points(result["exterior_xy_px"])
        elif "points_xy_px" in result:
            result["points_xy_px"] = points(result["points_xy_px"])
        elif "slices" in result:
            for item in result["slices"]:
                item["exterior_xy_px"] = points(item["exterior_xy_px"])
        return result
    raise TypeError(f"Unsupported ROI geometry: {type(geometry).__name__}")


@dataclass
class CopyRoiFrameDraft(RoiCommand):
    manager: Any
    object_id: UUID
    source_timepoint: int
    target_timepoint: int
    dx: float = 0.0
    dy: float = 0.0
    copy_association: bool = True

    EFFECTS: ClassVar[frozenset[EditEffect]] = frozenset(
        (EditEffect.ROI_GEOMETRY, EditEffect.ROI_ASSOCIATION)
    )

    def _apply(self) -> None:
        track = self.manager.get_object(self.object_id)
        source = track.frames.get(self.source_timepoint)
        if source is None or source.geometry is None:
            raise ValueError(
                f"ROI object has no segmented geometry at t={self.source_timepoint}"
            )
        geometry = _translated_geometry(source.geometry, self.dx, self.dy)
        cell_ref = source.cell_ref if self.copy_association else None
        # A cell reference is explicitly same-frame.  Carrying an unchanged
        # anchor to another time would violate the model and mis-associate the
        # draft, so only keep it when a test/model supplies a compatible ref.
        anchor = getattr(cell_ref, "nucleus_anchor", None)
        if anchor is not None and getattr(anchor, "timepoint", None) != self.target_timepoint:
            cell_ref = None
        frame = _new_segmented_frame(self.target_timepoint, geometry, cell_ref)
        self.manager.set_frame(self.object_id, frame)

    @property
    def description(self) -> str:
        return (
            f"Copy ROI draft t={self.source_timepoint} to "
            f"t={self.target_timepoint}"
        )


# Explicit command-suffix aliases make call sites self-documenting while the
# shorter names remain pleasant in menus and tests.
CreateObjectClassCommand = CreateObjectClass
UpdateObjectClassCommand = UpdateObjectClass
DeleteObjectClassCommand = DeleteObjectClass
CreateRoiObjectCommand = CreateRoiObject
UpdateRoiObjectSpanCommand = UpdateRoiObjectSpan
DeleteRoiObjectCommand = DeleteRoiObject
ReclassifyRoiObjectCommand = ReclassifyRoiObject
ReindexRoiObjectCommand = ReindexRoiObject
SetRoiFrameGeometryCommand = SetRoiFrameGeometry
DeleteRoiFrameCommand = DeleteRoiFrame
AssociateRoiFrameCommand = AssociateRoiFrame
MarkRoiFrameReviewedCommand = MarkRoiFrameReviewed
MarkRoiFrameAbsentCommand = MarkRoiFrameAbsent
CopyRoiFrameDraftCommand = CopyRoiFrameDraft

# Compact semantic aliases used by early design prototypes.
ReclassRoiObject = ReclassifyRoiObject
SetRoiAssociation = AssociateRoiFrame
MarkRoiReviewed = MarkRoiFrameReviewed
MarkRoiAbsent = MarkRoiFrameAbsent
CreateRoiClass = CreateObjectClass
UpdateRoiClass = UpdateObjectClass
DeleteRoiClass = DeleteObjectClass
EditRoiFrameGeometry = SetRoiFrameGeometry
SetRoiFrameAssociation = AssociateRoiFrame
ReviewRoiFrame = MarkRoiFrameReviewed
SetRoiFrameAbsent = MarkRoiFrameAbsent
CopyRoiFrame = CopyRoiFrameDraft


__all__ = [
    "AssociateRoiFrame",
    "CopyRoiFrame",
    "CopyRoiFrameDraft",
    "CreateObjectClass",
    "CreateRoiClass",
    "CreateRoiObject",
    "DeleteObjectClass",
    "DeleteRoiClass",
    "DeleteRoiFrame",
    "DeleteRoiObject",
    "EditRoiFrameGeometry",
    "MarkRoiFrameAbsent",
    "MarkRoiFrameReviewed",
    "ReclassifyRoiObject",
    "ReindexRoiObject",
    "ReviewRoiFrame",
    "RoiCommand",
    "SetRoiFrameGeometry",
    "SetRoiFrameAssociation",
    "SetRoiFrameAbsent",
    "UpdateObjectClass",
    "UpdateRoiClass",
    "UpdateRoiObjectSpan",
]
