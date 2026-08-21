"""Headless domain model for manually curated subcellular ROIs.

The model deliberately contains no Qt, napari, image-provider, or lineage
dependencies.  Records are frozen and normalize mutable inputs to immutable
tuples/mappings so command objects can safely retain before/after snapshots.
UUIDs are the persistence identity; class names and instance indices are
editable display metadata.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, TypeAlias
from uuid import UUID, uuid4


class RoiValidationError(ValueError):
    """Raised when an ROI domain record violates a scientific invariant."""


class GeometryKind(str, Enum):
    POLYGON_2D = "polygon_2d"
    THICK_POLYLINE_2D = "thick_polyline_2d"
    CONTOUR_STACK_3D = "contour_stack_3d"


class ThicknessUnit(str, Enum):
    MICROMETERS = "um"
    PIXELS = "px"


class SamplingMode(str, Enum):
    FILLED_VOLUME = "filled_volume"
    INNER_SHELL = "inner_shell"


class Presence(str, Enum):
    SEGMENTED = "segmented"
    ABSENT = "absent"


class ReviewState(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    NEEDS_REVIEW = "needs_review"


class AssociationStatus(str, Enum):
    UNASSOCIATED = "unassociated"
    EXACT = "exact"
    HINT_CHANGED = "hint_changed"
    ORPHANED = "orphaned"


Point2D: TypeAlias = tuple[float, float]
Point3D: TypeAlias = tuple[float, float, float]
ColorRgba: TypeAlias = tuple[float, float, float, float]


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RoiValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoiValidationError(f"{label} must be a non-negative integer")
    return value


def _finite_number(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoiValidationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise RoiValidationError(f"{label} must be finite")
    return result


def _optional_positive_number(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, label)
    if result <= 0:
        raise RoiValidationError(f"{label} must be positive")
    return result


def _uuid(value: UUID, label: str) -> UUID:
    if not isinstance(value, UUID):
        raise RoiValidationError(f"{label} must be a UUID")
    return value


def _enum(value: Any, enum_type: type[Enum], label: str) -> Any:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(repr(member.value) for member in enum_type)
        raise RoiValidationError(f"{label} must be one of {allowed}") from exc


def _point2(value: Sequence[float], label: str) -> Point2D:
    if isinstance(value, (str, bytes, bytearray)) or len(value) != 2:
        raise RoiValidationError(f"{label} must contain exactly two coordinates")
    return (_finite_number(value[0], f"{label}.x"), _finite_number(value[1], f"{label}.y"))


def _point3(value: Sequence[float], label: str) -> Point3D:
    if isinstance(value, (str, bytes, bytearray)) or len(value) != 3:
        raise RoiValidationError(f"{label} must contain exactly three coordinates")
    return (
        _finite_number(value[0], f"{label}.x"),
        _finite_number(value[1], f"{label}.y"),
        _finite_number(value[2], f"{label}.z"),
    )


def _immutable_json(value: Any, label: str = "extensions") -> Any:
    """Return a recursively immutable, finite JSON-compatible value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RoiValidationError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise RoiValidationError(f"{label} keys must be strings")
            result[key] = _immutable_json(child, f"{label}.{key}")
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _immutable_json(child, f"{label}[{index}]")
            for index, child in enumerate(value)
        )
    raise RoiValidationError(
        f"{label} must contain only JSON-compatible values, got {type(value).__name__}"
    )


def _signed_area(points: Sequence[Point2D]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]))
    )


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point2D, b: Point2D, point: Point2D, eps: float = 1e-12) -> bool:
    return (
        min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps
        and abs(_orientation(a, b, point)) <= eps
    )


def _segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    eps = 1e-12
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return (
        (abs(o1) <= eps and _on_segment(a, b, c))
        or (abs(o2) <= eps and _on_segment(a, b, d))
        or (abs(o3) <= eps and _on_segment(c, d, a))
        or (abs(o4) <= eps and _on_segment(c, d, b))
    )


def _canonical_ring(values: Sequence[Sequence[float]], label: str) -> tuple[Point2D, ...]:
    points = tuple(_point2(value, f"{label}[{index}]") for index, value in enumerate(values))
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(set(points)) < 3:
        raise RoiValidationError(f"{label} must have at least three distinct vertices")
    if len(points) < 3:
        raise RoiValidationError(f"{label} must have at least three vertices")
    edge_count = len(points)
    for first in range(edge_count):
        a, b = points[first], points[(first + 1) % edge_count]
        if a == b:
            raise RoiValidationError(f"{label} cannot contain a zero-length edge")
        for second in range(first + 1, edge_count):
            if second in {first, (first + 1) % edge_count}:
                continue
            if first == 0 and second == edge_count - 1:
                continue
            c, d = points[second], points[(second + 1) % edge_count]
            if _segments_intersect(a, b, c, d):
                raise RoiValidationError(f"{label} cannot self-intersect")
    area = _signed_area(points)
    if abs(area) <= 1e-12:
        raise RoiValidationError(f"{label} must have non-zero area")
    # Persist a counter-clockwise ring while retaining the authored first point.
    if area < 0:
        points = (points[0], *reversed(points[1:]))
    return points


@dataclass(frozen=True)
class CoordinateSpaceSnapshot:
    """Dataset geometry/calibration captured when annotations are authored."""

    coordinate_space_version: int = 1
    plane_start: int = 1
    xy_res: float | None = None
    z_res: float | None = None
    image_width_px: int | None = None
    image_height_px: int | None = None
    plane_count: int | None = None
    time_start: int = 1
    time_end: int | None = None
    split: int = 1
    flip: int = 1

    def __post_init__(self) -> None:
        _positive_integer(self.coordinate_space_version, "coordinate_space_version")
        _positive_integer(self.plane_start, "plane_start")
        _positive_integer(self.time_start, "time_start")
        if self.time_end is not None:
            _positive_integer(self.time_end, "time_end")
            if self.time_end < self.time_start:
                raise RoiValidationError("time_end cannot precede time_start")
        for label in ("image_width_px", "image_height_px", "plane_count"):
            value = getattr(self, label)
            if value is not None:
                _positive_integer(value, label)
        object.__setattr__(self, "xy_res", _optional_positive_number(self.xy_res, "xy_res"))
        object.__setattr__(self, "z_res", _optional_positive_number(self.z_res, "z_res"))
        if isinstance(self.split, bool) or not isinstance(self.split, int):
            raise RoiValidationError("split must be an integer")
        if isinstance(self.flip, bool) or not isinstance(self.flip, int):
            raise RoiValidationError("flip must be an integer")

    @property
    def plane_end(self) -> int | None:
        return None if self.plane_count is None else self.plane_start + self.plane_count - 1

    @property
    def has_physical_calibration(self) -> bool:
        return self.xy_res is not None and self.z_res is not None

    def contains_time(self, timepoint: int) -> bool:
        return timepoint >= self.time_start and (
            self.time_end is None or timepoint <= self.time_end
        )

    def contains_plane(self, z_plane: int) -> bool:
        return z_plane >= self.plane_start and (
            self.plane_end is None or z_plane <= self.plane_end
        )

    def z_index(self, z_plane: int) -> int:
        if not self.contains_plane(z_plane):
            raise RoiValidationError(f"z plane {z_plane} is outside the coordinate space")
        return z_plane - self.plane_start

    def xy_um(self, point_xy_px: Point2D) -> Point2D:
        if self.xy_res is None:
            raise RoiValidationError("XY physical calibration is unavailable")
        return (point_xy_px[0] * self.xy_res, point_xy_px[1] * self.xy_res)

    def mismatch_fields(self, other: CoordinateSpaceSnapshot) -> tuple[str, ...]:
        fields = (
            "coordinate_space_version",
            "plane_start",
            "xy_res",
            "z_res",
            "image_width_px",
            "image_height_px",
            "plane_count",
            "split",
            "flip",
        )
        return tuple(name for name in fields if getattr(self, name) != getattr(other, name))


@dataclass(frozen=True)
class Thickness:
    value: float
    unit: ThicknessUnit | str = ThicknessUnit.MICROMETERS

    def __post_init__(self) -> None:
        value = _finite_number(self.value, "thickness.value")
        if value <= 0:
            raise RoiValidationError("thickness.value must be positive")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", _enum(self.unit, ThicknessUnit, "thickness.unit"))


@dataclass(frozen=True)
class Polygon2D:
    z_plane: int
    exterior_xy_px: tuple[Point2D, ...]
    kind: GeometryKind = field(default=GeometryKind.POLYGON_2D, init=False)

    def __post_init__(self) -> None:
        _positive_integer(self.z_plane, "polygon z_plane")
        object.__setattr__(
            self,
            "exterior_xy_px",
            _canonical_ring(self.exterior_xy_px, "polygon exterior_xy_px"),
        )


@dataclass(frozen=True)
class ThickPolyline2D:
    z_plane: int
    points_xy_px: tuple[Point2D, ...]
    thickness: Thickness
    cap_style: str = "round"
    join_style: str = "round"
    kind: GeometryKind = field(default=GeometryKind.THICK_POLYLINE_2D, init=False)

    def __post_init__(self) -> None:
        _positive_integer(self.z_plane, "polyline z_plane")
        points = tuple(
            _point2(value, f"polyline points_xy_px[{index}]")
            for index, value in enumerate(self.points_xy_px)
        )
        if len(points) < 2 or len(set(points)) < 2:
            raise RoiValidationError("polyline must contain at least two distinct points")
        if not isinstance(self.thickness, Thickness):
            raise RoiValidationError("polyline thickness must be a Thickness")
        if self.cap_style != "round" or self.join_style != "round":
            raise RoiValidationError("V1 polylines require round cap and join styles")
        object.__setattr__(self, "points_xy_px", points)


@dataclass(frozen=True)
class ContourSlice:
    z_plane: int
    exterior_xy_px: tuple[Point2D, ...]

    def __post_init__(self) -> None:
        _positive_integer(self.z_plane, "contour z_plane")
        object.__setattr__(
            self,
            "exterior_xy_px",
            _canonical_ring(self.exterior_xy_px, "contour exterior_xy_px"),
        )


@dataclass(frozen=True)
class ContourStack3D:
    slices: tuple[ContourSlice, ...]
    sampling_mode: SamplingMode | str = SamplingMode.FILLED_VOLUME
    shell_thickness_um: float | None = None
    kind: GeometryKind = field(default=GeometryKind.CONTOUR_STACK_3D, init=False)

    def __post_init__(self) -> None:
        mode = _enum(self.sampling_mode, SamplingMode, "sampling_mode")
        slices = tuple(sorted(self.slices, key=lambda contour: contour.z_plane))
        if not slices or any(not isinstance(item, ContourSlice) for item in slices):
            raise RoiValidationError("contour stack must contain one or more ContourSlice records")
        planes = tuple(item.z_plane for item in slices)
        if len(set(planes)) != len(planes):
            raise RoiValidationError("contour stack z planes must be unique")
        if any(right != left + 1 for left, right in zip(planes, planes[1:])):
            raise RoiValidationError("contour stack z planes must be consecutive")
        shell = self.shell_thickness_um
        if mode is SamplingMode.INNER_SHELL:
            shell = _optional_positive_number(shell, "shell_thickness_um")
            if shell is None:
                raise RoiValidationError("inner-shell geometry requires shell_thickness_um")
        elif shell is not None:
            raise RoiValidationError("filled-volume geometry cannot have shell_thickness_um")
        object.__setattr__(self, "sampling_mode", mode)
        object.__setattr__(self, "slices", slices)
        object.__setattr__(self, "shell_thickness_um", shell)


Geometry: TypeAlias = Polygon2D | ThickPolyline2D | ContourStack3D


def geometry_planes(geometry: Geometry) -> tuple[int, ...]:
    if isinstance(geometry, (Polygon2D, ThickPolyline2D)):
        return (geometry.z_plane,)
    if isinstance(geometry, ContourStack3D):
        return tuple(contour.z_plane for contour in geometry.slices)
    raise RoiValidationError(f"unsupported geometry type {type(geometry).__name__}")


def validate_geometry_in_coordinate_space(
    geometry: Geometry,
    coordinate_space: CoordinateSpaceSnapshot,
) -> None:
    for plane in geometry_planes(geometry):
        if not coordinate_space.contains_plane(plane):
            raise RoiValidationError(f"geometry z plane {plane} is outside dataset bounds")
    if isinstance(geometry, ThickPolyline2D):
        if (
            geometry.thickness.unit is ThicknessUnit.PIXELS
            and coordinate_space.xy_res is not None
        ):
            raise RoiValidationError(
                "pixel thickness is allowed only when XY physical calibration is unavailable"
            )
    if (
        isinstance(geometry, ContourStack3D)
        and geometry.sampling_mode is SamplingMode.INNER_SHELL
        and not coordinate_space.has_physical_calibration
    ):
        raise RoiValidationError("inner-shell geometry requires physical calibration")


@dataclass(frozen=True)
class NucleusAnchor:
    timepoint: int
    index: int

    def __post_init__(self) -> None:
        _positive_integer(self.timepoint, "nucleus anchor timepoint")
        _positive_integer(self.index, "nucleus anchor index")


@dataclass(frozen=True)
class CellRef:
    nucleus_anchor: NucleusAnchor
    cell_birth_anchor: NucleusAnchor | None = None
    name_snapshot: str | None = None
    centroid_snapshot_xyz_px: Point3D | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.nucleus_anchor, NucleusAnchor):
            raise RoiValidationError("cell_ref.nucleus_anchor must be a NucleusAnchor")
        if self.cell_birth_anchor is not None and not isinstance(
            self.cell_birth_anchor, NucleusAnchor
        ):
            raise RoiValidationError("cell_ref.cell_birth_anchor must be a NucleusAnchor")
        if self.name_snapshot is not None:
            if not isinstance(self.name_snapshot, str) or not self.name_snapshot.strip():
                raise RoiValidationError("cell_ref.name_snapshot must be nonblank when present")
        if self.centroid_snapshot_xyz_px is not None:
            object.__setattr__(
                self,
                "centroid_snapshot_xyz_px",
                _point3(self.centroid_snapshot_xyz_px, "centroid_snapshot_xyz_px"),
            )


@dataclass(frozen=True)
class ObjectClass:
    name: str
    color_rgba: ColorRgba
    class_id: UUID = field(default_factory=uuid4)
    next_instance_index: int = 1
    default_geometry_kind: GeometryKind | str | None = None

    def __post_init__(self) -> None:
        _uuid(self.class_id, "class_id")
        if not isinstance(self.name, str) or not self.name.strip():
            raise RoiValidationError("object class name must be nonblank")
        if any(character in self.name for character in "\r\n"):
            raise RoiValidationError("object class name cannot contain line breaks")
        color = tuple(
            _finite_number(value, f"color_rgba[{index}]")
            for index, value in enumerate(self.color_rgba)
        )
        if len(color) != 4 or any(value < 0 or value > 1 for value in color):
            raise RoiValidationError("color_rgba must contain four values between 0 and 1")
        _positive_integer(self.next_instance_index, "next_instance_index")
        kind = self.default_geometry_kind
        if kind is not None:
            kind = _enum(kind, GeometryKind, "default_geometry_kind")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "color_rgba", color)
        object.__setattr__(self, "default_geometry_kind", kind)


def _edited_review_state(review_state: ReviewState) -> ReviewState:
    return (
        ReviewState.NEEDS_REVIEW
        if review_state is ReviewState.REVIEWED
        else review_state
    )


@dataclass(frozen=True)
class RoiFrameRecord:
    timepoint: int
    presence: Presence | str
    review_state: ReviewState | str
    geometry: Geometry | None = None
    cell_ref: CellRef | None = None
    frame_id: UUID = field(default_factory=uuid4)
    revision: int = 0

    def __post_init__(self) -> None:
        _uuid(self.frame_id, "frame_id")
        _positive_integer(self.timepoint, "frame timepoint")
        _nonnegative_integer(self.revision, "frame revision")
        presence = _enum(self.presence, Presence, "presence")
        review_state = _enum(self.review_state, ReviewState, "review_state")
        if self.cell_ref is not None:
            if not isinstance(self.cell_ref, CellRef):
                raise RoiValidationError("cell_ref must be a CellRef")
            if self.cell_ref.nucleus_anchor.timepoint != self.timepoint:
                raise RoiValidationError(
                    "cell_ref nucleus anchor must refer to the frame timepoint"
                )
        if presence is Presence.ABSENT and self.geometry is not None:
            raise RoiValidationError("an absent frame cannot have geometry")
        if presence is Presence.SEGMENTED and not isinstance(
            self.geometry, (Polygon2D, ThickPolyline2D, ContourStack3D)
        ):
            raise RoiValidationError("a segmented frame requires supported geometry")
        object.__setattr__(self, "presence", presence)
        object.__setattr__(self, "review_state", review_state)

    def with_geometry(self, geometry: Geometry) -> RoiFrameRecord:
        state = (
            ReviewState.DRAFT
            if self.presence is Presence.ABSENT
            else _edited_review_state(self.review_state)
        )
        return replace(
            self,
            geometry=geometry,
            presence=Presence.SEGMENTED,
            review_state=state,
            revision=self.revision + 1,
        )

    def with_association(self, cell_ref: CellRef | None) -> RoiFrameRecord:
        if cell_ref == self.cell_ref:
            return self
        return replace(
            self,
            cell_ref=cell_ref,
            review_state=_edited_review_state(self.review_state),
            revision=self.revision + 1,
        )

    def mark_reviewed(self) -> RoiFrameRecord:
        if self.review_state is ReviewState.REVIEWED:
            return self
        return replace(self, review_state=ReviewState.REVIEWED, revision=self.revision + 1)

    def mark_absent(self) -> RoiFrameRecord:
        if self.presence is Presence.ABSENT and self.review_state is ReviewState.REVIEWED:
            return self
        return replace(
            self,
            presence=Presence.ABSENT,
            geometry=None,
            review_state=ReviewState.REVIEWED,
            revision=self.revision + 1,
        )

    def mark_needs_review(self) -> RoiFrameRecord:
        if self.review_state is ReviewState.NEEDS_REVIEW:
            return self
        return replace(
            self,
            review_state=ReviewState.NEEDS_REVIEW,
            revision=self.revision + 1,
        )


def _immutable_frames(
    values: Mapping[int, RoiFrameRecord] | Sequence[RoiFrameRecord],
) -> Mapping[int, RoiFrameRecord]:
    if isinstance(values, Mapping):
        pairs = values.items()
    else:
        pairs = ((frame.timepoint, frame) for frame in values)
    result: dict[int, RoiFrameRecord] = {}
    for timepoint, frame in pairs:
        if not isinstance(frame, RoiFrameRecord):
            raise RoiValidationError("track frames must contain RoiFrameRecord values")
        _positive_integer(timepoint, "frame map key")
        if timepoint != frame.timepoint:
            raise RoiValidationError("frame map key must equal frame.timepoint")
        if timepoint in result:
            raise RoiValidationError(f"track contains duplicate frame timepoint {timepoint}")
        result[timepoint] = frame
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class RoiObjectTrack:
    class_id: UUID
    instance_index: int
    object_id: UUID = field(default_factory=uuid4)
    expected_start_time: int | None = None
    expected_end_time: int | None = None
    frames: Mapping[int, RoiFrameRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _uuid(self.object_id, "object_id")
        _uuid(self.class_id, "class_id")
        _positive_integer(self.instance_index, "instance_index")
        if self.expected_start_time is not None:
            _positive_integer(self.expected_start_time, "expected_start_time")
        if self.expected_end_time is not None:
            _positive_integer(self.expected_end_time, "expected_end_time")
        if (
            self.expected_start_time is not None
            and self.expected_end_time is not None
            and self.expected_end_time < self.expected_start_time
        ):
            raise RoiValidationError("expected_end_time cannot precede expected_start_time")
        frames = _immutable_frames(self.frames)
        for timepoint in frames:
            if self.expected_start_time is not None and timepoint < self.expected_start_time:
                raise RoiValidationError("frame precedes track expected_start_time")
            if self.expected_end_time is not None and timepoint > self.expected_end_time:
                raise RoiValidationError("frame follows track expected_end_time")
        object.__setattr__(self, "frames", frames)

    def with_frame(self, frame: RoiFrameRecord) -> RoiObjectTrack:
        updated = dict(self.frames)
        updated[frame.timepoint] = frame
        return replace(self, frames=updated)

    def without_frame(self, timepoint: int) -> RoiObjectTrack:
        if timepoint not in self.frames:
            return self
        updated = dict(self.frames)
        del updated[timepoint]
        return replace(self, frames=updated)

    @property
    def segmented_times(self) -> tuple[int, ...]:
        return tuple(
            timepoint
            for timepoint, frame in self.frames.items()
            if frame.presence is Presence.SEGMENTED
        )


@dataclass(frozen=True)
class QuarantinedRoiRecord:
    """Raw load record rejected without discarding the rest of the document."""

    path: str
    reason: str
    raw: Any = None

    def __post_init__(self) -> None:
        if not self.path or not self.reason:
            raise RoiValidationError("quarantine path and reason must be nonblank")
        object.__setattr__(self, "raw", _immutable_json(self.raw, "quarantine.raw"))


@dataclass(frozen=True)
class SubcellularRoiDocument:
    coordinate_space: CoordinateSpaceSnapshot = field(default_factory=CoordinateSpaceSnapshot)
    object_classes: tuple[ObjectClass, ...] = ()
    objects: tuple[RoiObjectTrack, ...] = ()
    document_id: UUID = field(default_factory=uuid4)
    file_revision: int = 0
    roi_revision: int = 0
    created_at: str | None = None
    saved_at: str | None = None
    producer_version: str = "0.2.0"
    dataset_fingerprint: str | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)
    quarantine: tuple[QuarantinedRoiRecord, ...] = ()

    def __post_init__(self) -> None:
        _uuid(self.document_id, "document_id")
        _nonnegative_integer(self.file_revision, "file_revision")
        _nonnegative_integer(self.roi_revision, "roi_revision")
        if not isinstance(self.coordinate_space, CoordinateSpaceSnapshot):
            raise RoiValidationError("coordinate_space must be a CoordinateSpaceSnapshot")
        classes = tuple(self.object_classes)
        objects = tuple(self.objects)
        if any(not isinstance(item, ObjectClass) for item in classes):
            raise RoiValidationError("object_classes must contain ObjectClass records")
        if any(not isinstance(item, RoiObjectTrack) for item in objects):
            raise RoiValidationError("objects must contain RoiObjectTrack records")
        class_by_id: dict[UUID, ObjectClass] = {}
        class_names: set[str] = set()
        for object_class in classes:
            if object_class.class_id in class_by_id:
                raise RoiValidationError(f"duplicate class_id {object_class.class_id}")
            folded = object_class.name.casefold()
            if folded in class_names:
                raise RoiValidationError(f"duplicate object class name {object_class.name!r}")
            class_by_id[object_class.class_id] = object_class
            class_names.add(folded)
        object_ids: set[UUID] = set()
        display_keys: set[tuple[UUID, int]] = set()
        max_indices: dict[UUID, int] = {}
        for track in objects:
            if track.object_id in object_ids:
                raise RoiValidationError(f"duplicate object_id {track.object_id}")
            if track.class_id not in class_by_id:
                raise RoiValidationError(f"object {track.object_id} refers to an unknown class")
            key = (track.class_id, track.instance_index)
            if key in display_keys:
                raise RoiValidationError(
                    f"duplicate class/instance index {track.class_id}/{track.instance_index}"
                )
            for frame in track.frames.values():
                if not self.coordinate_space.contains_time(frame.timepoint):
                    raise RoiValidationError(
                        f"frame timepoint {frame.timepoint} is outside dataset bounds"
                    )
                if frame.geometry is not None:
                    validate_geometry_in_coordinate_space(frame.geometry, self.coordinate_space)
            object_ids.add(track.object_id)
            display_keys.add(key)
            max_indices[track.class_id] = max(
                max_indices.get(track.class_id, 0), track.instance_index
            )
        for class_id, maximum in max_indices.items():
            if class_by_id[class_id].next_instance_index <= maximum:
                raise RoiValidationError(
                    f"class allocator for {class_id} must exceed existing index {maximum}"
                )
        quarantine = tuple(self.quarantine)
        if any(not isinstance(item, QuarantinedRoiRecord) for item in quarantine):
            raise RoiValidationError("quarantine must contain QuarantinedRoiRecord values")
        object.__setattr__(self, "object_classes", classes)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "extensions", _immutable_json(self.extensions))
        object.__setattr__(self, "quarantine", quarantine)

    @classmethod
    def empty(
        cls,
        coordinate_space: CoordinateSpaceSnapshot | None = None,
        *,
        dataset_fingerprint: str | None = None,
    ) -> SubcellularRoiDocument:
        return cls(
            coordinate_space=coordinate_space or CoordinateSpaceSnapshot(),
            dataset_fingerprint=dataset_fingerprint,
        )

    def get_class(self, class_id: UUID) -> ObjectClass | None:
        return next((item for item in self.object_classes if item.class_id == class_id), None)

    def get_object(self, object_id: UUID) -> RoiObjectTrack | None:
        return next((item for item in self.objects if item.object_id == object_id), None)

    def with_revision(self, roi_revision: int) -> SubcellularRoiDocument:
        return replace(self, roi_revision=roi_revision)


@dataclass(frozen=True)
class CellResolution:
    """Resolver output used by :class:`RoiManager` without lineage coupling."""

    value: Any
    name: str | None = None
    birth_anchor: NucleusAnchor | None = None
    centroid_xyz_px: Point3D | None = None

    def __post_init__(self) -> None:
        if self.birth_anchor is not None and not isinstance(self.birth_anchor, NucleusAnchor):
            raise RoiValidationError("birth_anchor must be a NucleusAnchor")
        if self.centroid_xyz_px is not None:
            object.__setattr__(
                self,
                "centroid_xyz_px",
                _point3(self.centroid_xyz_px, "centroid_xyz_px"),
            )


@dataclass(frozen=True)
class AssociationResolution:
    object_id: UUID
    frame_id: UUID
    timepoint: int
    status: AssociationStatus
    cell_ref: CellRef | None
    resolved: CellResolution | None = None
    warnings: tuple[str, ...] = ()


CellResolver: TypeAlias = Callable[[int, int], CellResolution | Any | None]


__all__ = [
    "AssociationResolution",
    "AssociationStatus",
    "CellRef",
    "CellResolution",
    "CellResolver",
    "ColorRgba",
    "ContourSlice",
    "ContourStack3D",
    "CoordinateSpaceSnapshot",
    "Geometry",
    "GeometryKind",
    "NucleusAnchor",
    "ObjectClass",
    "Point2D",
    "Point3D",
    "Polygon2D",
    "Presence",
    "QuarantinedRoiRecord",
    "ReviewState",
    "RoiFrameRecord",
    "RoiObjectTrack",
    "RoiValidationError",
    "SamplingMode",
    "SubcellularRoiDocument",
    "ThickPolyline2D",
    "Thickness",
    "ThicknessUnit",
    "geometry_planes",
    "validate_geometry_in_coordinate_space",
]
