"""Strict, versioned persistence for subcellular ROI annotations.

The sidecar is authoritative annotation data.  Loading is deliberately
fail-closed, while individual invalid class/object/frame records are
quarantined by the domain decoder.  Writes use same-directory private staging
and expose commit/finalize primitives for a multi-artifact save coordinator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from ..core.subcellular_roi import (
    CellRef,
    ContourSlice,
    ContourStack3D,
    CoordinateSpaceSnapshot,
    Geometry,
    GeometryKind,
    NucleusAnchor,
    ObjectClass,
    Polygon2D,
    Presence,
    QuarantinedRoiRecord,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    RoiValidationError,
    SamplingMode,
    SubcellularRoiDocument,
    ThickPolyline2D,
    Thickness,
)


ROI_SIDECAR_SCHEMA = "acetree.subcellular-rois"
ROI_SIDECAR_VERSION = 1
ROI_SIDECAR_SUFFIX = ".subcellular-rois.json"
CHECKSUM_ALGORITHM = "sha256"
QUARANTINE_EXTENSION_KEY = "acetree.quarantined-records"


class RoiSidecarError(ValueError):
    """Base class for safe ROI persistence failures."""


class RoiSidecarFormatError(RoiSidecarError):
    """The sidecar is malformed, invalid, or violates configured limits."""


class RoiSidecarChecksumError(RoiSidecarFormatError):
    """The checksum does not match the exact persisted document payload."""


class RoiSidecarConflictError(RoiSidecarError):
    """The destination changed since it was loaded or staged."""


class RoiSidecarUnsupportedVersionError(RoiSidecarError):
    """A sidecar is older than the registered migration chain supports."""


@dataclass(frozen=True)
class RoiJsonLimits:
    max_file_bytes: int = 32 * 1024 * 1024
    max_depth: int = 64
    max_items_per_collection: int = 250_000
    max_total_nodes: int = 1_000_000
    max_string_chars: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_file_bytes",
            "max_depth",
            "max_items_per_collection",
            "max_total_nodes",
            "max_string_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_ROI_JSON_LIMITS = RoiJsonLimits()


@dataclass(frozen=True)
class RoiSidecarToken:
    document_id: UUID
    file_revision: int
    checksum: str

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, UUID):
            raise ValueError("document_id must be a UUID")
        if (
            isinstance(self.file_revision, bool)
            or not isinstance(self.file_revision, int)
            or self.file_revision < 0
        ):
            raise ValueError("file_revision must be a non-negative integer")
        if (
            not isinstance(self.checksum, str)
            or len(self.checksum) != 64
            or any(character not in "0123456789abcdef" for character in self.checksum)
        ):
            raise ValueError("checksum must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class RoiSidecarLoad:
    path: Path
    document: SubcellularRoiDocument
    token: RoiSidecarToken
    source_version: int
    read_only: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class StagedRoiSidecar:
    destination: Path
    temp_path: Path
    document: SubcellularRoiDocument
    expected_token: RoiSidecarToken | None
    token: RoiSidecarToken
    byte_count: int

    @property
    def staged_path(self) -> Path:
        """Compatibility name used by dataset transaction coordinators."""
        return self.temp_path


Migration = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ROI_MIGRATIONS: dict[int, Migration] = {}


def register_roi_migration(from_version: int, migration: Migration) -> None:
    """Register a pure ``version -> version + 1`` envelope migration."""
    if (
        isinstance(from_version, bool)
        or not isinstance(from_version, int)
        or from_version < 0
    ):
        raise ValueError("from_version must be a non-negative integer")
    if not callable(migration):
        raise TypeError("migration must be callable")
    ROI_MIGRATIONS[from_version] = migration


def migrate_roi_payload(
    payload: Mapping[str, Any],
    *,
    target_version: int = ROI_SIDECAR_VERSION,
) -> dict[str, Any]:
    """Return a migrated deep copy; never mutate or rewrite the source."""
    current = _plain_json(payload)
    version = _integer(current.get("schema_version"), "schema_version", minimum=0)
    if version > target_version:
        raise RoiSidecarUnsupportedVersionError(
            f"cannot migrate newer schema version {version} to {target_version}"
        )
    while version < target_version:
        migration = ROI_MIGRATIONS.get(version)
        if migration is None:
            raise RoiSidecarUnsupportedVersionError(
                f"no ROI sidecar migration is registered for schema version {version}"
            )
        migrated = migration(deepcopy(current))
        current = _mapping(_plain_json(migrated), f"migration from version {version}")
        next_version = _integer(
            current.get("schema_version"),
            "migrated schema_version",
            minimum=0,
        )
        if next_version != version + 1:
            raise RoiSidecarUnsupportedVersionError(
                f"migration {version} must produce schema version {version + 1}"
            )
        version = next_version
    return dict(current)


def roi_sidecar_path(
    xml_path: str | Path | None,
    zip_path: str | Path | None = None,
) -> Path:
    """Derive the XML-authoritative path, falling back to the nuclei ZIP."""
    if xml_path is not None:
        candidate = Path(xml_path)
        if candidate.name and candidate.name not in {".", ".."}:
            return candidate.with_suffix(ROI_SIDECAR_SUFFIX)
    if zip_path is None:
        raise ValueError("an XML or nuclei ZIP path is required")
    candidate = Path(zip_path)
    return candidate.with_suffix(ROI_SIDECAR_SUFFIX)


def roi_sidecar_candidates(
    xml_path: str | Path | None,
    zip_path: str | Path | None,
) -> tuple[Path, ...]:
    """Return XML then ZIP candidates, de-duplicated in precedence order."""
    paths: list[Path] = []
    if xml_path is not None and Path(xml_path).name not in {"", ".", ".."}:
        paths.append(roi_sidecar_path(xml_path))
    if zip_path is not None:
        fallback = roi_sidecar_path(None, zip_path)
        if fallback not in paths:
            paths.append(fallback)
    return tuple(paths)


def canonical_document_bytes(document_data: Mapping[str, Any]) -> bytes:
    """Return the stable RFC-JSON byte representation used for checksums."""
    try:
        text = json.dumps(
            _plain_json(document_data),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RoiSidecarFormatError(f"document is not strict JSON: {exc}") from exc
    return text.encode("utf-8")


def document_checksum(document_data: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_document_bytes(document_data)).hexdigest()


def roi_document_to_dict(document: SubcellularRoiDocument) -> dict[str, Any]:
    if not isinstance(document, SubcellularRoiDocument):
        raise TypeError("document must be a SubcellularRoiDocument")
    extensions = _plain_json(document.extensions)
    if document.quarantine:
        extensions[QUARANTINE_EXTENSION_KEY] = [
            {"path": item.path, "reason": item.reason, "raw": _plain_json(item.raw)}
            for item in document.quarantine
        ]
    return {
        "document_id": str(document.document_id),
        "file_revision": document.file_revision,
        "roi_revision": document.roi_revision,
        "created_at": document.created_at,
        "saved_at": document.saved_at,
        "producer_version": document.producer_version,
        "dataset_fingerprint": document.dataset_fingerprint,
        "coordinate_space": _coordinate_space_to_dict(document.coordinate_space),
        "object_classes": [_object_class_to_dict(item) for item in document.object_classes],
        "objects": [_track_to_dict(item) for item in document.objects],
        "extensions": extensions,
    }


def roi_envelope_to_dict(document: SubcellularRoiDocument) -> dict[str, Any]:
    document_data = roi_document_to_dict(document)
    digest = document_checksum(document_data)
    return {
        "schema": ROI_SIDECAR_SCHEMA,
        "schema_version": ROI_SIDECAR_VERSION,
        "checksum": {"algorithm": CHECKSUM_ALGORITHM, "sha256": digest},
        "document": document_data,
    }


def read_roi_sidecar(
    path: str | Path,
    *,
    limits: RoiJsonLimits = DEFAULT_ROI_JSON_LIMITS,
) -> RoiSidecarLoad | None:
    """Read, checksum, migrate, validate, and partially quarantine a sidecar.

    An absent path returns ``None``.  Malformed or checksum-invalid content
    raises a typed error so callers can protect the existing path from writes.
    A newer envelope is decoded on a best-effort v1-compatible basis and is
    explicitly read-only.
    """
    source = Path(path)
    try:
        size = source.stat().st_size
    except FileNotFoundError:
        return None
    if size > limits.max_file_bytes:
        raise RoiSidecarFormatError(
            f"ROI sidecar exceeds {limits.max_file_bytes} byte limit: {source}"
        )
    try:
        raw_bytes = source.read_bytes()
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RoiSidecarFormatError(f"ROI sidecar is not valid UTF-8: {source}") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise RoiSidecarFormatError(
            f"Invalid ROI sidecar JSON in {source}: {exc.msg}"
        ) from exc
    _validate_json_limits(payload, limits)
    envelope = _mapping(payload, "ROI sidecar")
    schema = _string(envelope.get("schema"), "schema")
    if schema != ROI_SIDECAR_SCHEMA:
        raise RoiSidecarFormatError(f"unsupported ROI sidecar schema {schema!r}")
    version = _integer(envelope.get("schema_version"), "schema_version", minimum=0)
    document_data = _mapping(envelope.get("document"), "document")
    checksum_data = _mapping(envelope.get("checksum"), "checksum")
    algorithm = _string(checksum_data.get("algorithm"), "checksum.algorithm")
    if algorithm != CHECKSUM_ALGORITHM:
        raise RoiSidecarFormatError(f"unsupported checksum algorithm {algorithm!r}")
    expected_digest = _sha256(checksum_data.get("sha256"), "checksum.sha256")
    actual_digest = document_checksum(document_data)
    if actual_digest != expected_digest:
        raise RoiSidecarChecksumError(
            f"ROI sidecar checksum mismatch in {source}; expected {expected_digest}, "
            f"calculated {actual_digest}"
        )
    document_id = _uuid(document_data.get("document_id"), "document.document_id")
    file_revision = _integer(
        document_data.get("file_revision"),
        "document.file_revision",
        minimum=0,
    )
    token = RoiSidecarToken(document_id, file_revision, actual_digest)
    warnings: list[str] = []
    read_only = version > ROI_SIDECAR_VERSION
    decoded_envelope: Mapping[str, Any] = envelope
    if version < ROI_SIDECAR_VERSION:
        decoded_envelope = migrate_roi_payload(envelope)
        warnings.append(
            f"Loaded schema version {version} through an in-memory migration to "
            f"version {ROI_SIDECAR_VERSION}"
        )
    elif read_only:
        warnings.append(
            f"Schema version {version} is newer than supported version "
            f"{ROI_SIDECAR_VERSION}; ROI features are read-only"
        )
    decoded_document_data = _mapping(decoded_envelope.get("document"), "document")
    document = roi_document_from_dict(
        decoded_document_data,
        allow_unknown=read_only,
    )
    if document.quarantine:
        warnings.append(
            f"Quarantined {len(document.quarantine)} invalid ROI record(s) while loading"
        )
    return RoiSidecarLoad(
        path=source,
        document=document,
        token=token,
        source_version=version,
        read_only=read_only,
        warnings=tuple(warnings),
    )


def stage_roi_sidecar(
    path: str | Path,
    document: SubcellularRoiDocument,
    *,
    expected_token: RoiSidecarToken | None = None,
    limits: RoiJsonLimits = DEFAULT_ROI_JSON_LIMITS,
    force: bool = False,
) -> StagedRoiSidecar:
    """Create a complete private sibling file without installing it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force:
        _assert_destination_token(destination, expected_token, limits)
    envelope = roi_envelope_to_dict(document)
    try:
        text = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise RoiSidecarFormatError(f"ROI sidecar is not strict JSON: {exc}") from exc
    encoded = text.encode("utf-8")
    if len(encoded) > limits.max_file_bytes:
        raise RoiSidecarFormatError(
            f"serialized ROI sidecar exceeds {limits.max_file_bytes} byte limit"
        )
    digest = envelope["checksum"]["sha256"]
    token = RoiSidecarToken(document.document_id, document.file_revision, digest)
    fd, temp_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, _replacement_mode(destination))
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise
    return StagedRoiSidecar(
        destination=destination,
        temp_path=temp_path,
        document=document,
        expected_token=expected_token,
        token=token,
        byte_count=len(encoded),
    )


def commit_staged_roi_sidecar(
    staged: StagedRoiSidecar,
    *,
    limits: RoiJsonLimits = DEFAULT_ROI_JSON_LIMITS,
    force: bool = False,
) -> RoiSidecarToken:
    """Install a staged file after an immediate external-change check."""
    if not staged.temp_path.exists():
        raise RoiSidecarError(f"staged ROI file no longer exists: {staged.temp_path}")
    if not force:
        _assert_destination_token(staged.destination, staged.expected_token, limits)
    os.replace(staged.temp_path, staged.destination)
    _fsync_directory(staged.destination.parent)
    return staged.token


def discard_staged_roi_sidecar(staged: StagedRoiSidecar) -> None:
    staged.temp_path.unlink(missing_ok=True)


def write_roi_sidecar(
    path: str | Path,
    document: SubcellularRoiDocument,
    *,
    expected_token: RoiSidecarToken | None = None,
    limits: RoiJsonLimits = DEFAULT_ROI_JSON_LIMITS,
    force: bool = False,
) -> RoiSidecarToken:
    """Stage and atomically replace a sidecar, cleaning up on failure."""
    staged = stage_roi_sidecar(
        path,
        document,
        expected_token=expected_token,
        limits=limits,
        force=force,
    )
    try:
        return commit_staged_roi_sidecar(staged, limits=limits, force=force)
    finally:
        discard_staged_roi_sidecar(staged)


def roi_document_from_dict(
    value: Any,
    *,
    allow_unknown: bool = False,
) -> SubcellularRoiDocument:
    data = _mapping(value, "document")
    if not allow_unknown:
        _reject_unknown(
            data,
            {
                "document_id",
                "file_revision",
                "roi_revision",
                "created_at",
                "saved_at",
                "producer_version",
                "dataset_fingerprint",
                "coordinate_space",
                "object_classes",
                "objects",
                "extensions",
            },
            "document",
        )
    coordinate_space = _coordinate_space_from_dict(
        data.get("coordinate_space"), allow_unknown=allow_unknown
    )
    quarantine: list[QuarantinedRoiRecord] = []
    extensions = dict(_mapping(data.get("extensions", {}), "document.extensions"))
    persisted_quarantine = extensions.pop(QUARANTINE_EXTENSION_KEY, [])
    for index, item in enumerate(
        _sequence(persisted_quarantine, f"extensions.{QUARANTINE_EXTENSION_KEY}")
    ):
        try:
            item_data = _mapping(item, "persisted quarantine record")
            quarantine.append(
                QuarantinedRoiRecord(
                    path=_string(item_data.get("path"), "quarantine.path"),
                    reason=_string(item_data.get("reason"), "quarantine.reason"),
                    raw=item_data.get("raw"),
                )
            )
        except (RoiSidecarError, RoiValidationError, TypeError, ValueError) as exc:
            quarantine.append(
                QuarantinedRoiRecord(
                    path=f"document.extensions.{QUARANTINE_EXTENSION_KEY}[{index}]",
                    reason=str(exc),
                    raw=item,
                )
            )
    classes: list[ObjectClass] = []
    class_ids: set[UUID] = set()
    class_names: set[str] = set()
    for index, raw_class in enumerate(
        _sequence(data.get("object_classes", []), "document.object_classes")
    ):
        path = f"document.object_classes[{index}]"
        try:
            object_class = _object_class_from_dict(raw_class, allow_unknown=allow_unknown)
            if object_class.class_id in class_ids:
                raise RoiValidationError(f"duplicate class_id {object_class.class_id}")
            if object_class.name.casefold() in class_names:
                raise RoiValidationError(f"duplicate class name {object_class.name!r}")
            classes.append(object_class)
            class_ids.add(object_class.class_id)
            class_names.add(object_class.name.casefold())
        except (RoiSidecarError, RoiValidationError, TypeError, ValueError) as exc:
            quarantine.append(QuarantinedRoiRecord(path, str(exc), raw_class))
    objects: list[RoiObjectTrack] = []
    object_ids: set[UUID] = set()
    display_keys: set[tuple[UUID, int]] = set()
    maximum_indices: dict[UUID, int] = {}
    for index, raw_track in enumerate(
        _sequence(data.get("objects", []), "document.objects")
    ):
        path = f"document.objects[{index}]"
        try:
            track, frame_quarantine = _track_from_dict(
                raw_track,
                path=path,
                coordinate_space=coordinate_space,
                allow_unknown=allow_unknown,
            )
            quarantine.extend(frame_quarantine)
            if track.class_id not in class_ids:
                raise RoiValidationError(f"track refers to unknown class_id {track.class_id}")
            if track.object_id in object_ids:
                raise RoiValidationError(f"duplicate object_id {track.object_id}")
            display_key = (track.class_id, track.instance_index)
            if display_key in display_keys:
                raise RoiValidationError(
                    f"duplicate class/index {track.class_id}/{track.instance_index}"
                )
            objects.append(track)
            object_ids.add(track.object_id)
            display_keys.add(display_key)
            maximum_indices[track.class_id] = max(
                maximum_indices.get(track.class_id, 0), track.instance_index
            )
        except (RoiSidecarError, RoiValidationError, TypeError, ValueError) as exc:
            quarantine.append(QuarantinedRoiRecord(path, str(exc), raw_track))
    # A stale allocator is repaired monotonically and reported rather than
    # quarantining every otherwise-valid object in that class.
    repaired_classes: list[ObjectClass] = []
    for object_class in classes:
        minimum_next = maximum_indices.get(object_class.class_id, 0) + 1
        if object_class.next_instance_index < minimum_next:
            quarantine.append(
                QuarantinedRoiRecord(
                    f"document.object_classes[{classes.index(object_class)}].next_instance_index",
                    f"allocator advanced from {object_class.next_instance_index} to {minimum_next}",
                    object_class.next_instance_index,
                )
            )
            object_class = replace(object_class, next_instance_index=minimum_next)
        repaired_classes.append(object_class)
    try:
        return SubcellularRoiDocument(
            document_id=_uuid(data.get("document_id"), "document.document_id"),
            file_revision=_integer(
                data.get("file_revision"), "document.file_revision", minimum=0
            ),
            roi_revision=_integer(
                data.get("roi_revision", 0), "document.roi_revision", minimum=0
            ),
            coordinate_space=coordinate_space,
            object_classes=tuple(repaired_classes),
            objects=tuple(objects),
            created_at=_optional_string(data.get("created_at"), "document.created_at"),
            saved_at=_optional_string(data.get("saved_at"), "document.saved_at"),
            producer_version=_string(
                data.get("producer_version", "unknown"), "document.producer_version"
            ),
            dataset_fingerprint=_optional_string(
                data.get("dataset_fingerprint"), "document.dataset_fingerprint"
            ),
            extensions=extensions,
            quarantine=tuple(quarantine),
        )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(f"invalid ROI document: {exc}") from exc


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _coordinate_space_to_dict(value: CoordinateSpaceSnapshot) -> dict[str, Any]:
    return {
        "coordinate_space_version": value.coordinate_space_version,
        "plane_start": value.plane_start,
        "xy_res": value.xy_res,
        "z_res": value.z_res,
        "image_width_px": value.image_width_px,
        "image_height_px": value.image_height_px,
        "plane_count": value.plane_count,
        "time_start": value.time_start,
        "time_end": value.time_end,
        "split": value.split,
        "flip": value.flip,
    }


def _coordinate_space_from_dict(
    value: Any, *, allow_unknown: bool = False
) -> CoordinateSpaceSnapshot:
    data = _mapping(value, "document.coordinate_space")
    if not allow_unknown:
        _reject_unknown(
            data,
            {
                "coordinate_space_version",
                "plane_start",
                "xy_res",
                "z_res",
                "image_width_px",
                "image_height_px",
                "plane_count",
                "time_start",
                "time_end",
                "split",
                "flip",
            },
            "document.coordinate_space",
        )
    try:
        return CoordinateSpaceSnapshot(
            coordinate_space_version=_integer(
                data.get("coordinate_space_version", 1),
                "coordinate_space_version",
                minimum=1,
            ),
            plane_start=_integer(data.get("plane_start", 1), "plane_start", minimum=1),
            xy_res=_optional_number(data.get("xy_res"), "xy_res"),
            z_res=_optional_number(data.get("z_res"), "z_res"),
            image_width_px=_optional_integer(data.get("image_width_px"), "image_width_px"),
            image_height_px=_optional_integer(
                data.get("image_height_px"), "image_height_px"
            ),
            plane_count=_optional_integer(data.get("plane_count"), "plane_count"),
            time_start=_integer(data.get("time_start", 1), "time_start", minimum=1),
            time_end=_optional_integer(data.get("time_end"), "time_end"),
            split=_integer(data.get("split", 1), "split"),
            flip=_integer(data.get("flip", 1), "flip"),
        )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(f"invalid coordinate space: {exc}") from exc


def _object_class_to_dict(value: ObjectClass) -> dict[str, Any]:
    return {
        "class_id": str(value.class_id),
        "name": value.name,
        "color_rgba": list(value.color_rgba),
        "next_instance_index": value.next_instance_index,
        "default_geometry_kind": (
            None if value.default_geometry_kind is None else value.default_geometry_kind.value
        ),
    }


def _object_class_from_dict(value: Any, *, allow_unknown: bool) -> ObjectClass:
    data = _mapping(value, "object class")
    if not allow_unknown:
        _reject_unknown(
            data,
            {
                "class_id",
                "name",
                "color_rgba",
                "next_instance_index",
                "default_geometry_kind",
            },
            "object class",
        )
    color = _sequence(data.get("color_rgba"), "object class color_rgba")
    kind = data.get("default_geometry_kind")
    try:
        return ObjectClass(
            class_id=_uuid(data.get("class_id"), "object class class_id"),
            name=_string(data.get("name"), "object class name"),
            color_rgba=tuple(_number(item, "color component") for item in color),
            next_instance_index=_integer(
                data.get("next_instance_index", 1),
                "object class next_instance_index",
                minimum=1,
            ),
            default_geometry_kind=(
                None if kind is None else _string(kind, "default_geometry_kind")
            ),
        )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(str(exc)) from exc


def _track_to_dict(value: RoiObjectTrack) -> dict[str, Any]:
    return {
        "object_id": str(value.object_id),
        "class_id": str(value.class_id),
        "instance_index": value.instance_index,
        "expected_start_time": value.expected_start_time,
        "expected_end_time": value.expected_end_time,
        "frames": [_frame_to_dict(frame) for frame in value.frames.values()],
    }


def _track_from_dict(
    value: Any,
    *,
    path: str,
    coordinate_space: CoordinateSpaceSnapshot,
    allow_unknown: bool,
) -> tuple[RoiObjectTrack, list[QuarantinedRoiRecord]]:
    data = _mapping(value, "ROI object")
    if not allow_unknown:
        _reject_unknown(
            data,
            {
                "object_id",
                "class_id",
                "instance_index",
                "expected_start_time",
                "expected_end_time",
                "frames",
            },
            "ROI object",
        )
    frames: dict[int, RoiFrameRecord] = {}
    quarantine: list[QuarantinedRoiRecord] = []
    for index, raw_frame in enumerate(_sequence(data.get("frames", []), "object frames")):
        frame_path = f"{path}.frames[{index}]"
        try:
            frame = _frame_from_dict(raw_frame, allow_unknown=allow_unknown)
            if frame.timepoint in frames:
                raise RoiValidationError(f"duplicate frame timepoint {frame.timepoint}")
            if not coordinate_space.contains_time(frame.timepoint):
                raise RoiValidationError(
                    f"frame timepoint {frame.timepoint} is outside dataset bounds"
                )
            if frame.geometry is not None:
                from ..core.subcellular_roi import validate_geometry_in_coordinate_space

                validate_geometry_in_coordinate_space(frame.geometry, coordinate_space)
            frames[frame.timepoint] = frame
        except (RoiSidecarError, RoiValidationError, TypeError, ValueError) as exc:
            quarantine.append(QuarantinedRoiRecord(frame_path, str(exc), raw_frame))
    try:
        return (
            RoiObjectTrack(
                object_id=_uuid(data.get("object_id"), "object_id"),
                class_id=_uuid(data.get("class_id"), "class_id"),
                instance_index=_integer(
                    data.get("instance_index"), "instance_index", minimum=1
                ),
                expected_start_time=_optional_integer(
                    data.get("expected_start_time"), "expected_start_time"
                ),
                expected_end_time=_optional_integer(
                    data.get("expected_end_time"), "expected_end_time"
                ),
                frames=frames,
            ),
            quarantine,
        )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(str(exc)) from exc


def _frame_to_dict(value: RoiFrameRecord) -> dict[str, Any]:
    return {
        "frame_id": str(value.frame_id),
        "revision": value.revision,
        "timepoint": value.timepoint,
        "presence": value.presence.value,
        "review_state": value.review_state.value,
        "cell_ref": None if value.cell_ref is None else _cell_ref_to_dict(value.cell_ref),
        "geometry": None if value.geometry is None else _geometry_to_dict(value.geometry),
    }


def _frame_from_dict(value: Any, *, allow_unknown: bool) -> RoiFrameRecord:
    data = _mapping(value, "ROI frame")
    if not allow_unknown:
        _reject_unknown(
            data,
            {
                "frame_id",
                "revision",
                "timepoint",
                "presence",
                "review_state",
                "cell_ref",
                "geometry",
            },
            "ROI frame",
        )
    geometry_data = data.get("geometry")
    cell_data = data.get("cell_ref")
    try:
        return RoiFrameRecord(
            frame_id=_uuid(data.get("frame_id"), "frame_id"),
            revision=_integer(data.get("revision", 0), "frame revision", minimum=0),
            timepoint=_integer(data.get("timepoint"), "frame timepoint", minimum=1),
            presence=_string(data.get("presence"), "frame presence"),
            review_state=_string(data.get("review_state"), "frame review_state"),
            cell_ref=(
                None
                if cell_data is None
                else _cell_ref_from_dict(cell_data, allow_unknown=allow_unknown)
            ),
            geometry=(
                None
                if geometry_data is None
                else _geometry_from_dict(geometry_data, allow_unknown=allow_unknown)
            ),
        )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(str(exc)) from exc


def _cell_ref_to_dict(value: CellRef) -> dict[str, Any]:
    return {
        "nucleus_anchor": _anchor_to_dict(value.nucleus_anchor),
        "cell_birth_anchor": (
            None if value.cell_birth_anchor is None else _anchor_to_dict(value.cell_birth_anchor)
        ),
        "name_snapshot": value.name_snapshot,
        "centroid_snapshot_xyz_px": (
            None
            if value.centroid_snapshot_xyz_px is None
            else list(value.centroid_snapshot_xyz_px)
        ),
    }


def _cell_ref_from_dict(value: Any, *, allow_unknown: bool = False) -> CellRef:
    data = _mapping(value, "cell_ref")
    if not allow_unknown:
        _reject_unknown(
            data,
            {
                "nucleus_anchor",
                "cell_birth_anchor",
                "name_snapshot",
                "centroid_snapshot_xyz_px",
            },
            "cell_ref",
        )
    birth = data.get("cell_birth_anchor")
    centroid = data.get("centroid_snapshot_xyz_px")
    try:
        return CellRef(
            nucleus_anchor=_anchor_from_dict(
                data.get("nucleus_anchor"),
                "nucleus_anchor",
                allow_unknown=allow_unknown,
            ),
            cell_birth_anchor=(
                None
                if birth is None
                else _anchor_from_dict(
                    birth, "cell_birth_anchor", allow_unknown=allow_unknown
                )
            ),
            name_snapshot=_optional_string(data.get("name_snapshot"), "name_snapshot"),
            centroid_snapshot_xyz_px=(
                None
                if centroid is None
                else tuple(
                    _number(item, "centroid coordinate")
                    for item in _sequence(centroid, "centroid_snapshot_xyz_px")
                )
            ),
        )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(str(exc)) from exc


def _anchor_to_dict(value: NucleusAnchor) -> dict[str, int]:
    return {"timepoint": value.timepoint, "index": value.index}


def _anchor_from_dict(
    value: Any, label: str, *, allow_unknown: bool = False
) -> NucleusAnchor:
    data = _mapping(value, label)
    if not allow_unknown:
        _reject_unknown(data, {"timepoint", "index"}, label)
    try:
        return NucleusAnchor(
            _integer(data.get("timepoint"), f"{label}.timepoint", minimum=1),
            _integer(data.get("index"), f"{label}.index", minimum=1),
        )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(str(exc)) from exc


def _geometry_to_dict(value: Geometry) -> dict[str, Any]:
    if isinstance(value, Polygon2D):
        return {
            "kind": GeometryKind.POLYGON_2D.value,
            "z_plane": value.z_plane,
            "exterior_xy_px": [list(point) for point in value.exterior_xy_px],
        }
    if isinstance(value, ThickPolyline2D):
        return {
            "kind": GeometryKind.THICK_POLYLINE_2D.value,
            "z_plane": value.z_plane,
            "points_xy_px": [list(point) for point in value.points_xy_px],
            "thickness": {
                "value": value.thickness.value,
                "unit": value.thickness.unit.value,
            },
            "cap_style": value.cap_style,
            "join_style": value.join_style,
        }
    if isinstance(value, ContourStack3D):
        return {
            "kind": GeometryKind.CONTOUR_STACK_3D.value,
            "sampling_mode": value.sampling_mode.value,
            "shell_thickness_um": value.shell_thickness_um,
            "slices": [
                {
                    "z_plane": contour.z_plane,
                    "exterior_xy_px": [list(point) for point in contour.exterior_xy_px],
                }
                for contour in value.slices
            ],
        }
    raise RoiSidecarFormatError(f"unsupported geometry type {type(value).__name__}")


def _geometry_from_dict(value: Any, *, allow_unknown: bool = False) -> Geometry:
    data = _mapping(value, "geometry")
    kind = _string(data.get("kind"), "geometry.kind")
    try:
        if kind == GeometryKind.POLYGON_2D.value:
            if not allow_unknown:
                _reject_unknown(
                    data, {"kind", "z_plane", "exterior_xy_px"}, "polygon"
                )
            return Polygon2D(
                z_plane=_integer(data.get("z_plane"), "polygon z_plane", minimum=1),
                exterior_xy_px=_points(data.get("exterior_xy_px"), "exterior_xy_px"),
            )
        if kind == GeometryKind.THICK_POLYLINE_2D.value:
            if not allow_unknown:
                _reject_unknown(
                    data,
                    {
                        "kind",
                        "z_plane",
                        "points_xy_px",
                        "thickness",
                        "cap_style",
                        "join_style",
                    },
                    "polyline",
                )
            thickness_data = _mapping(data.get("thickness"), "polyline thickness")
            if not allow_unknown:
                _reject_unknown(
                    thickness_data, {"value", "unit"}, "polyline thickness"
                )
            return ThickPolyline2D(
                z_plane=_integer(data.get("z_plane"), "polyline z_plane", minimum=1),
                points_xy_px=_points(data.get("points_xy_px"), "points_xy_px"),
                thickness=Thickness(
                    _number(thickness_data.get("value"), "thickness.value"),
                    _string(thickness_data.get("unit"), "thickness.unit"),
                ),
                cap_style=_string(data.get("cap_style", "round"), "cap_style"),
                join_style=_string(data.get("join_style", "round"), "join_style"),
            )
        if kind == GeometryKind.CONTOUR_STACK_3D.value:
            if not allow_unknown:
                _reject_unknown(
                    data,
                    {"kind", "sampling_mode", "shell_thickness_um", "slices"},
                    "contour stack",
                )
            slices = tuple(
                _contour_slice_from_dict(item, allow_unknown=allow_unknown)
                for item in _sequence(data.get("slices"), "contour stack slices")
            )
            return ContourStack3D(
                slices=slices,
                sampling_mode=_string(data.get("sampling_mode"), "sampling_mode"),
                shell_thickness_um=_optional_number(
                    data.get("shell_thickness_um"), "shell_thickness_um"
                ),
            )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(str(exc)) from exc
    raise RoiSidecarFormatError(f"unsupported geometry kind {kind!r}")


def _contour_slice_from_dict(
    value: Any, *, allow_unknown: bool = False
) -> ContourSlice:
    data = _mapping(value, "contour slice")
    if not allow_unknown:
        _reject_unknown(data, {"z_plane", "exterior_xy_px"}, "contour slice")
    try:
        return ContourSlice(
            z_plane=_integer(data.get("z_plane"), "contour z_plane", minimum=1),
            exterior_xy_px=_points(data.get("exterior_xy_px"), "exterior_xy_px"),
        )
    except RoiValidationError as exc:
        raise RoiSidecarFormatError(str(exc)) from exc


def _points(value: Any, label: str) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for index, point in enumerate(_sequence(value, label)):
        coordinates = _sequence(point, f"{label}[{index}]")
        if len(coordinates) != 2:
            raise RoiSidecarFormatError(f"{label}[{index}] must have two coordinates")
        result.append(
            (
                _number(coordinates[0], f"{label}[{index}].x"),
                _number(coordinates[1], f"{label}[{index}].y"),
            )
        )
    return tuple(result)


def _assert_destination_token(
    destination: Path,
    expected: RoiSidecarToken | None,
    limits: RoiJsonLimits,
) -> None:
    current_load = read_roi_sidecar(destination, limits=limits)
    current = None if current_load is None else current_load.token
    if current != expected:
        expected_text = "absent" if expected is None else repr(expected)
        current_text = "absent" if current is None else repr(current)
        raise RoiSidecarConflictError(
            f"ROI sidecar changed externally at {destination}; expected {expected_text}, "
            f"found {current_text}"
        )


def _replacement_mode(destination: Path) -> int:
    try:
        return stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        previous = os.umask(0)
        os.umask(previous)
        return 0o666 & ~previous


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RoiSidecarFormatError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise RoiSidecarFormatError(f"invalid non-finite JSON number {value}")


def _validate_json_limits(value: Any, limits: RoiJsonLimits) -> None:
    stack: list[tuple[Any, int]] = [(value, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_total_nodes:
            raise RoiSidecarFormatError("ROI JSON exceeds the total-node limit")
        if depth > limits.max_depth:
            raise RoiSidecarFormatError("ROI JSON exceeds the nesting-depth limit")
        if isinstance(current, str):
            if len(current) > limits.max_string_chars:
                raise RoiSidecarFormatError("ROI JSON string exceeds the length limit")
        elif isinstance(current, Mapping):
            if len(current) > limits.max_items_per_collection:
                raise RoiSidecarFormatError("ROI JSON object exceeds the collection limit")
            stack.extend((key, depth + 1) for key in current)
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            if len(current) > limits.max_items_per_collection:
                raise RoiSidecarFormatError("ROI JSON array exceeds the collection limit")
            stack.extend((child, depth + 1) for child in current)
        elif isinstance(current, float) and not math.isfinite(current):
            raise RoiSidecarFormatError("ROI JSON contains a non-finite number")


def _plain_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise RoiSidecarFormatError("JSON value is non-finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise RoiSidecarFormatError("JSON mapping keys must be strings")
            result[key] = _plain_json(child)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(child) for child in value]
    raise RoiSidecarFormatError(f"unsupported JSON value {type(value).__name__}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoiSidecarFormatError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RoiSidecarFormatError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise RoiSidecarFormatError(f"{label} must be a string")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoiSidecarFormatError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RoiSidecarFormatError(f"{label} must be at least {minimum}")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    return None if value is None else _integer(value, label, minimum=1)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoiSidecarFormatError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise RoiSidecarFormatError(f"{label} must be finite")
    return result


def _optional_number(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _uuid(value: Any, label: str) -> UUID:
    if not isinstance(value, str):
        raise RoiSidecarFormatError(f"{label} must be a UUID string")
    try:
        return UUID(value)
    except (ValueError, AttributeError) as exc:
        raise RoiSidecarFormatError(f"{label} must be a valid UUID") from exc


def _sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if (
        len(text) != 64
        or text.lower() != text
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise RoiSidecarFormatError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise RoiSidecarFormatError(f"{label} contains unknown field(s): {', '.join(unknown)}")


# Concise aliases for callers and transaction implementations.
read_subcellular_rois = read_roi_sidecar
write_subcellular_rois = write_roi_sidecar
stage_subcellular_rois = stage_roi_sidecar
install_staged_roi_sidecar = commit_staged_roi_sidecar


__all__ = [
    "CHECKSUM_ALGORITHM",
    "DEFAULT_ROI_JSON_LIMITS",
    "ROI_MIGRATIONS",
    "ROI_SIDECAR_SCHEMA",
    "ROI_SIDECAR_SUFFIX",
    "ROI_SIDECAR_VERSION",
    "Migration",
    "RoiJsonLimits",
    "RoiSidecarChecksumError",
    "RoiSidecarConflictError",
    "RoiSidecarError",
    "RoiSidecarFormatError",
    "RoiSidecarLoad",
    "RoiSidecarToken",
    "RoiSidecarUnsupportedVersionError",
    "StagedRoiSidecar",
    "canonical_document_bytes",
    "commit_staged_roi_sidecar",
    "discard_staged_roi_sidecar",
    "document_checksum",
    "install_staged_roi_sidecar",
    "migrate_roi_payload",
    "read_roi_sidecar",
    "register_roi_migration",
    "roi_document_from_dict",
    "roi_document_to_dict",
    "roi_envelope_to_dict",
    "roi_sidecar_candidates",
    "roi_sidecar_path",
    "stage_roi_sidecar",
    "utc_now_iso",
    "write_roi_sidecar",
]
