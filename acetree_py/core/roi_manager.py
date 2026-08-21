"""Authoritative manager for subcellular ROI records and persistence state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import UUID

from .subcellular_roi import (
    AssociationResolution,
    AssociationStatus,
    CellRef,
    CellResolution,
    CellResolver,
    CoordinateSpaceSnapshot,
    Geometry,
    GeometryKind,
    NucleusAnchor,
    ObjectClass,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    RoiValidationError,
    SubcellularRoiDocument,
    validate_geometry_in_coordinate_space,
)
from ..io.roi_sidecar import (
    DEFAULT_ROI_JSON_LIMITS,
    RoiJsonLimits,
    RoiSidecarError,
    RoiSidecarLoad,
    RoiSidecarToken,
    StagedRoiSidecar,
    commit_staged_roi_sidecar,
    discard_staged_roi_sidecar,
    read_roi_sidecar,
    roi_sidecar_candidates,
    roi_sidecar_path,
    stage_roi_sidecar,
    utc_now_iso,
)


class RoiManagerError(RuntimeError):
    """Base class for manager operation failures."""


class RoiNotFoundError(RoiManagerError, KeyError):
    """A referenced class, object, or frame does not exist."""


class RoiWriteProtectedError(RoiManagerError):
    """Saving is blocked until an invalid/newer sidecar is explicitly handled."""


@dataclass(frozen=True)
class RoiManagerSaveStage:
    sidecar: StagedRoiSidecar
    source_roi_revision: int
    saved_document: SubcellularRoiDocument
    replaces_protected_sidecar: bool = False

    @property
    def destination(self) -> Path:
        return self.sidecar.destination

    @property
    def temp_path(self) -> Path:
        return self.sidecar.temp_path

    @property
    def staged_path(self) -> Path:
        return self.sidecar.temp_path


class RoiManager:
    """Mutable facade over immutable ROI document snapshots.

    Every scientific mutation installs a fresh document and advances the
    monotonic ``roi_revision`` token.  Persisted ``file_revision`` only
    advances after an installed staged save is finalized.
    """

    def __init__(
        self,
        document: SubcellularRoiDocument | None = None,
        *,
        sidecar_path: str | Path | None = None,
        loaded_token: RoiSidecarToken | None = None,
        load_error: str | None = None,
        read_only: bool = False,
        load_warnings: Iterable[str] = (),
        limits: RoiJsonLimits = DEFAULT_ROI_JSON_LIMITS,
    ) -> None:
        self._document = document or SubcellularRoiDocument.empty()
        self._roi_revision = self._document.roi_revision
        self._document = replace(self._document, roi_revision=self._roi_revision)
        self._saved_roi_revision = self._roi_revision
        self._sidecar_path = None if sidecar_path is None else Path(sidecar_path)
        self._loaded_token = loaded_token
        self._load_error = load_error
        self._read_only = bool(read_only)
        self._load_warnings = tuple(str(item) for item in load_warnings)
        self._limits = limits
        self._replace_invalid_authorized = False
        self._coordinate_mismatch: tuple[str, ...] = ()
        self._physical_normalization_acknowledged = True
        self._rebuild_indexes()

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        image_provider: Any | None = None,
        num_timepoints: int | None = None,
        limits: RoiJsonLimits = DEFAULT_ROI_JSON_LIMITS,
    ) -> RoiManager:
        """Load beside the XML without preventing a nuclear dataset open."""
        coordinate_space = coordinate_space_from_config(
            config,
            image_provider=image_provider,
            num_timepoints=num_timepoints,
        )
        config_path = getattr(config, "config_file", None)
        zip_path = getattr(config, "zip_file", None)
        has_xml = (
            config_path is not None
            and Path(config_path).name not in {"", ".", ".."}
            and Path(config_path).suffix.lower() == ".xml"
        )
        candidates = roi_sidecar_candidates(config_path if has_xml else None, zip_path)
        if not candidates:
            return cls(SubcellularRoiDocument.empty(coordinate_space), limits=limits)
        authoritative = candidates[0]
        warnings: list[str] = []
        if len(candidates) > 1 and candidates[0].exists() and candidates[1].exists():
            warnings.append(
                f"Both XML-derived and ZIP-derived ROI sidecars exist; using {candidates[0]}"
            )
        # When an XML is known its sibling is authoritative even when absent.
        # The ZIP fallback is only for a genuinely headless/no-XML manager.
        try:
            loaded = read_roi_sidecar(authoritative, limits=limits)
        except RoiSidecarError as exc:
            return cls(
                SubcellularRoiDocument.empty(coordinate_space),
                sidecar_path=authoritative,
                load_error=str(exc),
                load_warnings=warnings,
                limits=limits,
            )
        if loaded is None:
            return cls(
                SubcellularRoiDocument.empty(coordinate_space),
                sidecar_path=authoritative,
                load_warnings=warnings,
                limits=limits,
            )
        manager = cls.from_load(loaded, load_warnings=warnings, limits=limits)
        manager.reconcile_coordinate_space(coordinate_space)
        # Opening and mismatch marking establish the baseline; no user edit has
        # occurred merely because the current dataset view differs.
        manager._saved_roi_revision = manager._roi_revision
        return manager

    @classmethod
    def from_load(
        cls,
        loaded: RoiSidecarLoad,
        *,
        load_warnings: Iterable[str] = (),
        limits: RoiJsonLimits = DEFAULT_ROI_JSON_LIMITS,
    ) -> RoiManager:
        return cls(
            loaded.document,
            sidecar_path=loaded.path,
            loaded_token=loaded.token,
            read_only=loaded.read_only,
            load_warnings=(*load_warnings, *loaded.warnings),
            limits=limits,
        )

    @property
    def document(self) -> SubcellularRoiDocument:
        return self._document

    @property
    def classes(self) -> tuple[ObjectClass, ...]:
        return self._document.object_classes

    @property
    def objects(self) -> tuple[RoiObjectTrack, ...]:
        return self._document.objects

    @property
    def roi_revision(self) -> int:
        return self._roi_revision

    @property
    def file_revision(self) -> int:
        return self._document.file_revision

    @property
    def is_dirty(self) -> bool:
        return self._roi_revision != self._saved_roi_revision

    @property
    def sidecar_path(self) -> Path | None:
        return self._sidecar_path

    @property
    def loaded_token(self) -> RoiSidecarToken | None:
        return self._loaded_token

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def load_warnings(self) -> tuple[str, ...]:
        return self._load_warnings

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def is_write_protected(self) -> bool:
        return self._read_only or (
            self._load_error is not None and not self._replace_invalid_authorized
        )

    @property
    def coordinate_mismatch(self) -> tuple[str, ...]:
        return self._coordinate_mismatch

    @property
    def physical_normalization_blocked(self) -> bool:
        return bool(self._coordinate_mismatch) and not self._physical_normalization_acknowledged

    def _rebuild_indexes(self) -> None:
        self._classes_by_id = {
            object_class.class_id: object_class
            for object_class in self._document.object_classes
        }
        self._objects_by_id = {track.object_id: track for track in self._document.objects}
        self._objects_by_display_key = {
            (track.class_id, track.instance_index): track
            for track in self._document.objects
        }

    def get_class(self, class_id: UUID) -> ObjectClass | None:
        return self._classes_by_id.get(class_id)

    def require_class(self, class_id: UUID) -> ObjectClass:
        result = self.get_class(class_id)
        if result is None:
            raise RoiNotFoundError(f"unknown ROI class {class_id}")
        return result

    def get_object(self, object_id: UUID) -> RoiObjectTrack | None:
        return self._objects_by_id.get(object_id)

    def require_object(self, object_id: UUID) -> RoiObjectTrack:
        result = self.get_object(object_id)
        if result is None:
            raise RoiNotFoundError(f"unknown ROI object {object_id}")
        return result

    def get_by_class_index(
        self, class_id: UUID, instance_index: int
    ) -> RoiObjectTrack | None:
        return self._objects_by_display_key.get((class_id, instance_index))

    def get_frame(self, object_id: UUID, timepoint: int) -> RoiFrameRecord | None:
        track = self.get_object(object_id)
        return None if track is None else track.frames.get(timepoint)

    def bump_revision(self) -> int:
        """Advance the token for transitional callers that mutate externally."""
        self._assert_editable()
        self._roi_revision += 1
        self._document = replace(self._document, roi_revision=self._roi_revision)
        return self._roi_revision

    def _install_document(self, document: SubcellularRoiDocument) -> None:
        self._assert_editable()
        self._roi_revision += 1
        self._document = replace(document, roi_revision=self._roi_revision)
        self._rebuild_indexes()

    def _assert_editable(self) -> None:
        if self._read_only:
            raise RoiWriteProtectedError(
                "ROI document uses a newer schema version and is read-only"
            )
        if self._load_error is not None and not self._replace_invalid_authorized:
            raise RoiWriteProtectedError(
                "ROI sidecar is malformed and protected; explicitly authorize replacement first"
            )

    def replace_document(self, document: SubcellularRoiDocument) -> None:
        """Install an undo/redo snapshot while retaining a monotonic revision."""
        if document.document_id != self._document.document_id:
            raise RoiManagerError("cannot replace a manager with a different document identity")
        self._install_document(document)

    def create_class(
        self,
        name: str,
        color_rgba: tuple[float, float, float, float],
        *,
        default_geometry_kind: GeometryKind | str | None = None,
    ) -> ObjectClass:
        object_class = ObjectClass(
            name=name,
            color_rgba=color_rgba,
            default_geometry_kind=default_geometry_kind,
        )
        self._install_document(
            replace(
                self._document,
                object_classes=(*self._document.object_classes, object_class),
            )
        )
        return object_class

    def update_class(
        self,
        class_id: UUID,
        *,
        name: str | None = None,
        color_rgba: tuple[float, float, float, float] | None = None,
        default_geometry_kind: GeometryKind | str | None = None,
        update_default_geometry_kind: bool = False,
    ) -> ObjectClass:
        current = self.require_class(class_id)
        updated = replace(
            current,
            name=current.name if name is None else name,
            color_rgba=current.color_rgba if color_rgba is None else color_rgba,
            default_geometry_kind=(
                default_geometry_kind
                if update_default_geometry_kind
                else current.default_geometry_kind
            ),
        )
        if updated == current:
            return current
        classes = tuple(updated if item.class_id == class_id else item for item in self.classes)
        self._install_document(replace(self._document, object_classes=classes))
        return updated

    def delete_class(self, class_id: UUID) -> ObjectClass:
        current = self.require_class(class_id)
        if any(track.class_id == class_id for track in self.objects):
            raise RoiManagerError("cannot delete an ROI class while objects still use it")
        self._install_document(
            replace(
                self._document,
                object_classes=tuple(
                    item for item in self.classes if item.class_id != class_id
                ),
            )
        )
        return current

    def allocate_instance_index(self, class_id: UUID) -> int:
        object_class = self.require_class(class_id)
        index = object_class.next_instance_index
        updated = replace(object_class, next_instance_index=index + 1)
        classes = tuple(updated if item.class_id == class_id else item for item in self.classes)
        self._install_document(replace(self._document, object_classes=classes))
        return index

    def create_object(
        self,
        class_id: UUID,
        *,
        instance_index: int | None = None,
        expected_start_time: int | None = None,
        expected_end_time: int | None = None,
        frames: dict[int, RoiFrameRecord] | None = None,
    ) -> RoiObjectTrack:
        object_class = self.require_class(class_id)
        if instance_index is None:
            instance_index = object_class.next_instance_index
        if self.get_by_class_index(class_id, instance_index) is not None:
            raise RoiManagerError(f"ROI class index {instance_index} is already in use")
        track = RoiObjectTrack(
            class_id=class_id,
            instance_index=instance_index,
            expected_start_time=expected_start_time,
            expected_end_time=expected_end_time,
            frames=frames or {},
        )
        updated_class = replace(
            object_class,
            next_instance_index=max(object_class.next_instance_index, instance_index + 1),
        )
        classes = tuple(
            updated_class if item.class_id == class_id else item for item in self.classes
        )
        self._install_document(
            replace(
                self._document,
                object_classes=classes,
                objects=(*self.objects, track),
            )
        )
        return track

    def update_object(self, track: RoiObjectTrack) -> RoiObjectTrack:
        current = self.require_object(track.object_id)
        self.require_class(track.class_id)
        collision = self.get_by_class_index(track.class_id, track.instance_index)
        if collision is not None and collision.object_id != track.object_id:
            raise RoiManagerError(
                f"ROI class/index {track.class_id}/{track.instance_index} is already in use"
            )
        if track == current:
            return current
        object_class = self.require_class(track.class_id)
        classes = self.classes
        if object_class.next_instance_index <= track.instance_index:
            updated_class = replace(
                object_class, next_instance_index=track.instance_index + 1
            )
            classes = tuple(
                updated_class if item.class_id == track.class_id else item
                for item in classes
            )
        objects = tuple(track if item.object_id == track.object_id else item for item in self.objects)
        self._install_document(
            replace(self._document, object_classes=classes, objects=objects)
        )
        return track

    def reclass_object(
        self,
        object_id: UUID,
        class_id: UUID,
        *,
        instance_index: int | None = None,
    ) -> RoiObjectTrack:
        track = self.require_object(object_id)
        target_class = self.require_class(class_id)
        if instance_index is None:
            instance_index = target_class.next_instance_index
        frames = {
            timepoint: frame.mark_needs_review()
            for timepoint, frame in track.frames.items()
        }
        return self.update_object(
            replace(
                track,
                class_id=class_id,
                instance_index=instance_index,
                frames=frames,
            )
        )

    def reindex_object(self, object_id: UUID, instance_index: int) -> RoiObjectTrack:
        return self.update_object(
            replace(self.require_object(object_id), instance_index=instance_index)
        )

    def delete_object(self, object_id: UUID) -> RoiObjectTrack:
        current = self.require_object(object_id)
        self._install_document(
            replace(
                self._document,
                objects=tuple(item for item in self.objects if item.object_id != object_id),
            )
        )
        return current

    def set_frame(self, object_id: UUID, frame: RoiFrameRecord) -> RoiFrameRecord:
        track = self.require_object(object_id)
        if not self._document.coordinate_space.contains_time(frame.timepoint):
            raise RoiValidationError("frame timepoint is outside dataset bounds")
        if frame.geometry is not None:
            validate_geometry_in_coordinate_space(
                frame.geometry, self._document.coordinate_space
            )
        if track.frames.get(frame.timepoint) == frame:
            return frame
        self.update_object(track.with_frame(frame))
        return frame

    def delete_frame(self, object_id: UUID, timepoint: int) -> RoiFrameRecord:
        track = self.require_object(object_id)
        frame = track.frames.get(timepoint)
        if frame is None:
            raise RoiNotFoundError(f"ROI object has no frame at timepoint {timepoint}")
        self.update_object(track.without_frame(timepoint))
        return frame

    def update_frame_geometry(
        self, object_id: UUID, timepoint: int, geometry: Geometry
    ) -> RoiFrameRecord:
        frame = self.get_frame(object_id, timepoint)
        if frame is None:
            frame = RoiFrameRecord(
                timepoint=timepoint,
                presence=Presence.SEGMENTED,
                review_state=ReviewState.DRAFT,
                geometry=geometry,
            )
        else:
            frame = frame.with_geometry(geometry)
        return self.set_frame(object_id, frame)

    def associate_frame(
        self, object_id: UUID, timepoint: int, cell_ref: CellRef | None
    ) -> RoiFrameRecord:
        frame = self.get_frame(object_id, timepoint)
        if frame is None:
            raise RoiNotFoundError(f"ROI object has no frame at timepoint {timepoint}")
        return self.set_frame(object_id, frame.with_association(cell_ref))

    def mark_frame_reviewed(self, object_id: UUID, timepoint: int) -> RoiFrameRecord:
        frame = self.get_frame(object_id, timepoint)
        if frame is None:
            raise RoiNotFoundError(f"ROI object has no frame at timepoint {timepoint}")
        return self.set_frame(object_id, frame.mark_reviewed())

    def mark_frame_absent(
        self,
        object_id: UUID,
        timepoint: int,
        *,
        cell_ref: CellRef | None = None,
    ) -> RoiFrameRecord:
        frame = self.get_frame(object_id, timepoint)
        if frame is None:
            frame = RoiFrameRecord(
                timepoint=timepoint,
                presence=Presence.ABSENT,
                review_state=ReviewState.REVIEWED,
                cell_ref=cell_ref,
            )
        else:
            frame = frame.mark_absent()
            if cell_ref is not None and cell_ref != frame.cell_ref:
                frame = frame.with_association(cell_ref).mark_reviewed()
        return self.set_frame(object_id, frame)

    def objects_at_time(
        self,
        timepoint: int,
        *,
        include_absent: bool = False,
        class_id: UUID | None = None,
    ) -> tuple[tuple[RoiObjectTrack, RoiFrameRecord], ...]:
        result: list[tuple[RoiObjectTrack, RoiFrameRecord]] = []
        for track in self.objects:
            if class_id is not None and track.class_id != class_id:
                continue
            frame = track.frames.get(timepoint)
            if frame is None or (
                frame.presence is Presence.ABSENT and not include_absent
            ):
                continue
            result.append((track, frame))
        return tuple(result)

    def reconcile_cells(
        self, resolver: CellResolver
    ) -> tuple[AssociationResolution, ...]:
        """Resolve exact physical anchors and report hint changes fail-closed."""
        results: list[AssociationResolution] = []
        for track in self.objects:
            for frame in track.frames.values():
                cell_ref = frame.cell_ref
                if cell_ref is None:
                    results.append(
                        AssociationResolution(
                            track.object_id,
                            frame.frame_id,
                            frame.timepoint,
                            AssociationStatus.UNASSOCIATED,
                            None,
                        )
                    )
                    continue
                anchor = cell_ref.nucleus_anchor
                resolved_value = resolver(anchor.timepoint, anchor.index)
                if resolved_value is None:
                    results.append(
                        AssociationResolution(
                            track.object_id,
                            frame.frame_id,
                            frame.timepoint,
                            AssociationStatus.ORPHANED,
                            cell_ref,
                            warnings=("same-frame nucleus anchor is unavailable",),
                        )
                    )
                    continue
                resolved = _coerce_cell_resolution(resolved_value)
                warnings: list[str] = []
                if (
                    cell_ref.name_snapshot is not None
                    and resolved.name is not None
                    and cell_ref.name_snapshot != resolved.name
                ):
                    warnings.append(
                        f"cell name changed from {cell_ref.name_snapshot!r} to {resolved.name!r}"
                    )
                if (
                    cell_ref.cell_birth_anchor is not None
                    and resolved.birth_anchor is not None
                    and cell_ref.cell_birth_anchor != resolved.birth_anchor
                ):
                    warnings.append("cell birth anchor changed")
                results.append(
                    AssociationResolution(
                        track.object_id,
                        frame.frame_id,
                        frame.timepoint,
                        (
                            AssociationStatus.HINT_CHANGED
                            if warnings
                            else AssociationStatus.EXACT
                        ),
                        cell_ref,
                        resolved=resolved,
                        warnings=tuple(warnings),
                    )
                )
        return tuple(results)

    def reconcile_coordinate_space(
        self, current: CoordinateSpaceSnapshot
    ) -> tuple[str, ...]:
        mismatch = self._document.coordinate_space.mismatch_fields(current)
        self._coordinate_mismatch = mismatch
        self._physical_normalization_acknowledged = not mismatch
        if mismatch and not self.is_write_protected:
            updated_objects: list[RoiObjectTrack] = []
            changed = False
            for track in self.objects:
                frames: dict[int, RoiFrameRecord] = {}
                for timepoint, frame in track.frames.items():
                    updated = (
                        frame.mark_needs_review()
                        if frame.geometry is not None
                        else frame
                    )
                    frames[timepoint] = updated
                    changed = changed or updated != frame
                updated_objects.append(replace(track, frames=frames))
            if changed:
                self._install_document(
                    replace(self._document, objects=tuple(updated_objects))
                )
        return mismatch

    def acknowledge_coordinate_space(
        self, current: CoordinateSpaceSnapshot
    ) -> tuple[str, ...]:
        self._assert_editable()
        for track in self.objects:
            for frame in track.frames.values():
                if frame.geometry is not None:
                    validate_geometry_in_coordinate_space(frame.geometry, current)
        previous = self._coordinate_mismatch
        if current != self._document.coordinate_space:
            self._install_document(replace(self._document, coordinate_space=current))
        self._coordinate_mismatch = ()
        self._physical_normalization_acknowledged = True
        return previous

    def authorize_replace_invalid_sidecar(self) -> None:
        """Explicitly allow replacing a malformed protected destination."""
        if self._read_only:
            raise RoiWriteProtectedError(
                "a newer-version sidecar cannot be replaced in place; use a recovery copy"
            )
        if self._load_error is None:
            return
        if self._replace_invalid_authorized:
            return
        self._replace_invalid_authorized = True
        # Make the explicit replacement observable to coordinated Save even
        # when the recovery document is intentionally empty.
        self._roi_revision += 1
        self._document = replace(self._document, roi_revision=self._roi_revision)

    discard_invalid_sidecar = authorize_replace_invalid_sidecar

    def stage_save(self, path: str | Path | None = None) -> RoiManagerSaveStage:
        destination = Path(path) if path is not None else self._sidecar_path
        if destination is None:
            raise RoiManagerError("no ROI sidecar path is configured")
        if self.is_write_protected:
            detail = self._load_error or "the sidecar uses a newer schema version"
            raise RoiWriteProtectedError(f"ROI sidecar is write-protected: {detail}")
        now = utc_now_iso()
        saved_document = replace(
            self._document,
            file_revision=self._document.file_revision + 1,
            roi_revision=self._roi_revision,
            created_at=self._document.created_at or now,
            saved_at=now,
        )
        same_destination = (
            self._sidecar_path is not None
            and destination.resolve(strict=False)
            == self._sidecar_path.resolve(strict=False)
        )
        replacing_protected = self._load_error is not None and same_destination
        staged = stage_roi_sidecar(
            destination,
            saved_document,
            # Save As creates a new independent destination.  A token loaded
            # from the source sidecar is not a valid precondition for it.
            expected_token=self._loaded_token if same_destination else None,
            limits=self._limits,
            force=replacing_protected,
        )
        return RoiManagerSaveStage(
            sidecar=staged,
            source_roi_revision=self._roi_revision,
            saved_document=saved_document,
            replaces_protected_sidecar=replacing_protected,
        )

    def finalize_external_commit(self, staged: RoiManagerSaveStage) -> RoiSidecarToken:
        """Update manager state after a coordinator installed ``staged`` once."""
        if staged.saved_document.document_id != self._document.document_id:
            raise RoiManagerError("staged save belongs to a different ROI document")
        self._document = replace(
            self._document,
            file_revision=staged.saved_document.file_revision,
            created_at=staged.saved_document.created_at,
            saved_at=staged.saved_document.saved_at,
            roi_revision=self._roi_revision,
        )
        self._loaded_token = staged.sidecar.token
        self._sidecar_path = staged.sidecar.destination
        self._saved_roi_revision = staged.source_roi_revision
        self._load_error = None
        self._replace_invalid_authorized = False
        self._rebuild_indexes()
        return staged.sidecar.token

    finalize_save = finalize_external_commit

    def commit_save(self, staged: RoiManagerSaveStage) -> RoiSidecarToken:
        token = commit_staged_roi_sidecar(
            staged.sidecar,
            limits=self._limits,
            force=staged.replaces_protected_sidecar,
        )
        self.finalize_external_commit(staged)
        return token

    def discard_save(self, staged: RoiManagerSaveStage) -> None:
        discard_staged_roi_sidecar(staged.sidecar)

    def save(self, path: str | Path | None = None) -> RoiSidecarToken:
        staged = self.stage_save(path)
        try:
            return self.commit_save(staged)
        finally:
            self.discard_save(staged)

    def save_recovery_copy(self, path: str | Path) -> RoiSidecarToken:
        """Write current data elsewhere without clearing authoritative protection."""
        destination = Path(path)
        if destination == self._sidecar_path:
            raise RoiManagerError("recovery copy must use a different path")
        now = utc_now_iso()
        recovery = replace(
            self._document,
            file_revision=self._document.file_revision + 1,
            created_at=self._document.created_at or now,
            saved_at=now,
        )
        staged = stage_roi_sidecar(
            destination,
            recovery,
            expected_token=None,
            limits=self._limits,
        )
        try:
            return commit_staged_roi_sidecar(staged, limits=self._limits)
        finally:
            discard_staged_roi_sidecar(staged)


def coordinate_space_from_config(
    config: Any,
    *,
    image_provider: Any | None = None,
    num_timepoints: int | None = None,
) -> CoordinateSpaceSnapshot:
    plane_start = int(getattr(config, "plane_start", 1))
    plane_end = getattr(config, "plane_end", None)
    plane_count = None
    if plane_end is not None and int(plane_end) >= plane_start:
        plane_count = int(plane_end) - plane_start + 1
    image_width_px = None
    image_height_px = None
    if image_provider is not None:
        provider_planes = _positive_optional_int(
            getattr(image_provider, "num_planes", None)
        )
        if provider_planes is not None:
            plane_count = provider_planes
        shape = getattr(image_provider, "image_shape", None)
        if shape is not None:
            try:
                height, width = shape
            except (TypeError, ValueError):
                pass
            else:
                image_height_px = _positive_optional_int(height)
                image_width_px = _positive_optional_int(width)
        if num_timepoints is None:
            num_timepoints = _positive_optional_int(
                getattr(image_provider, "num_timepoints", None)
            )
    resolved_time_end = (
        _positive_optional_int(num_timepoints)
        if num_timepoints is not None
        else _positive_optional_int(getattr(config, "ending_index", None))
    )
    return CoordinateSpaceSnapshot(
        plane_start=plane_start,
        xy_res=_positive_optional_float(getattr(config, "xy_res", None)),
        z_res=_positive_optional_float(getattr(config, "z_res", None)),
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        plane_count=plane_count,
        time_start=max(1, int(getattr(config, "starting_index", 1))),
        time_end=resolved_time_end,
        split=int(getattr(config, "split", 1)),
        flip=int(getattr(config, "flip", 1)),
    )


def _positive_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if result > 0 else None


def _positive_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    return result if result > 0 else None


def _coerce_cell_resolution(value: Any) -> CellResolution:
    if isinstance(value, CellResolution):
        return value
    name = getattr(value, "effective_name", None) or getattr(value, "name", None)
    birth_anchor = getattr(value, "birth_anchor", None)
    if birth_anchor is not None and not isinstance(birth_anchor, NucleusAnchor):
        try:
            birth_anchor = NucleusAnchor(*birth_anchor)
        except (TypeError, ValueError, RoiValidationError):
            birth_anchor = None
    centroid = getattr(value, "centroid_xyz_px", None)
    if centroid is None and all(hasattr(value, name) for name in ("x", "y", "z")):
        centroid = (value.x, value.y, value.z)
    return CellResolution(
        value=value,
        name=None if name is None else str(name),
        birth_anchor=birth_anchor,
        centroid_xyz_px=centroid,
    )


__all__ = [
    "RoiManager",
    "RoiManagerError",
    "RoiManagerSaveStage",
    "RoiNotFoundError",
    "RoiWriteProtectedError",
    "coordinate_space_from_config",
    "roi_sidecar_path",
]
