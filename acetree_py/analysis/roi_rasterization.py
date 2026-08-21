"""Deterministic, cropped rasterization for subcellular ROI geometries.

The functions in this module deliberately know nothing about Qt, napari, or
the ROI manager.  Geometry objects are consumed through the public fields
defined in :mod:`acetree_py.core.subcellular_roi`; mappings with the same
fields are also accepted to keep persistence and tests at a clean boundary.

Coordinates are pixel-centre coordinates in ``(x, y)`` order.  Returned
arrays use NumPy order and are read-only.  Two-dimensional results therefore
have ``(Y, X)`` masks and three-dimensional results have ``(Z, Y, X)`` masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon as draw_polygon
from skimage.measure import marching_cubes, mesh_surface_area


ROI_RASTERIZATION_VERSION = 1
_MAX_RASTER_ELEMENTS = 100_000_000


class RoiRasterizationError(ValueError):
    """Raised when a geometry cannot be rasterized safely."""

    def __init__(self, message: str, *, reason: str = "invalid_geometry") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class RoiCalibration:
    """Physical calibration needed by ROI raster and measurement services."""

    xy_res: float | None = None
    z_res: float | None = None
    plane_start: int = 1

    @classmethod
    def from_value(cls, value: Any | None) -> "RoiCalibration":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        xy_res = _field(value, "xy_res", None)
        z_res = _field(value, "z_res", None)
        plane_start = _field(value, "plane_start", 1)
        return cls(
            xy_res=None if xy_res is None else float(xy_res),
            z_res=None if z_res is None else float(z_res),
            plane_start=int(plane_start),
        )

    @property
    def has_xy(self) -> bool:
        return self.xy_res is not None and math.isfinite(self.xy_res) and self.xy_res > 0

    @property
    def has_3d(self) -> bool:
        return (
            self.has_xy
            and self.z_res is not None
            and math.isfinite(self.z_res)
            and self.z_res > 0
        )

    def cache_key(self) -> tuple[float | None, float | None, int]:
        return (self.xy_res, self.z_res, self.plane_start)


@dataclass(frozen=True, slots=True)
class RasterizedRoi:
    """A minimal in-bounds ROI mask and its provenance/support metadata."""

    geometry_kind: str
    mask: np.ndarray
    slices: tuple[slice, ...]
    nominal_sample_count: int
    in_bounds_sample_count: int
    coverage_fraction: float
    clipped: bool
    parent_mask: np.ndarray | None = None
    surface_area_um2: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mask = _readonly_bool_array(self.mask)
        if mask.ndim not in (2, 3):
            raise ValueError("ROI masks must be 2D or 3D")
        if len(self.slices) != mask.ndim:
            raise ValueError("The number of bounding slices must match mask.ndim")
        object.__setattr__(self, "mask", mask)
        if self.parent_mask is not None:
            parent = _readonly_bool_array(self.parent_mask)
            if parent.shape != mask.shape:
                raise ValueError("parent_mask must have the same shape as mask")
            object.__setattr__(self, "parent_mask", parent)
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def bounding_slices(self) -> tuple[slice, ...]:
        """Alias documenting that ``slices`` locate this crop in the image."""

        return self.slices

    @property
    def sample_count(self) -> int:
        return self.in_bounds_sample_count

    @property
    def is_empty(self) -> bool:
        return self.in_bounds_sample_count == 0

    def extract_from(self, image: np.ndarray) -> np.ndarray:
        """Return the image crop corresponding to this mask.

        An already-cropped image is accepted as a convenience for reducers.
        """

        array = np.asarray(image)
        if array.shape == self.mask.shape:
            return array
        if array.ndim != self.mask.ndim:
            raise ValueError(
                f"Expected a {self.mask.ndim}D image or crop, got shape {array.shape}"
            )
        crop = array[self.slices]
        if crop.shape != self.mask.shape:
            raise ValueError(
                f"Image crop {crop.shape} does not match ROI mask {self.mask.shape}"
            )
        return crop


# Compatibility/readability alias used by callers that prefer the service name.
CroppedRoiMask = RasterizedRoi


class RoiMaskRasterizer:
    """Convert supported ROI geometries to deterministic cropped masks."""

    def __init__(self, calibration: Any | None = None) -> None:
        self.calibration = RoiCalibration.from_value(calibration)

    def rasterize(
        self,
        geometry: Any,
        image_shape: Sequence[int],
        *,
        calibration: Any | None = None,
    ) -> RasterizedRoi:
        cal = self.calibration if calibration is None else RoiCalibration.from_value(calibration)
        kind = geometry_kind(geometry)
        if kind == "polygon_2d":
            return rasterize_polygon(geometry, image_shape)
        if kind == "thick_polyline_2d":
            return rasterize_thick_polyline(geometry, image_shape, calibration=cal)
        if kind == "contour_stack_3d":
            return rasterize_contour_stack(geometry, image_shape, calibration=cal)
        raise RoiRasterizationError(f"Unsupported ROI geometry kind: {kind!r}")

    __call__ = rasterize


def rasterize_roi(
    geometry: Any,
    image_shape: Sequence[int],
    *,
    calibration: Any | None = None,
) -> RasterizedRoi:
    """Functional entry point matching :class:`RoiMaskRasterizer`."""

    return RoiMaskRasterizer(calibration).rasterize(geometry, image_shape)


def rasterize_polygon(geometry: Any, image_shape: Sequence[int]) -> RasterizedRoi:
    """Rasterize a closed polygon using integer pixel centres."""

    height, width = _plane_shape(image_shape)
    points = _points(geometry, "exterior_xy_px")
    _validate_polygon(points)
    nominal, y0, x0 = _polygon_nominal_mask(points)
    return _crop_2d_result(
        "polygon_2d",
        nominal,
        y0=y0,
        x0=x0,
        height=height,
        width=width,
    )


def rasterize_thick_polyline(
    geometry: Any,
    image_shape: Sequence[int],
    *,
    calibration: Any | None = None,
) -> RasterizedRoi:
    """Rasterize the union of round-capped line segments."""

    height, width = _plane_shape(image_shape)
    points = _points(geometry, "points_xy_px")
    points = _validate_polyline(points)
    radius_px = _line_radius_px(geometry, RoiCalibration.from_value(calibration))

    min_x = math.floor(min(point[0] for point in points) - radius_px)
    max_x = math.ceil(max(point[0] for point in points) + radius_px)
    min_y = math.floor(min(point[1] for point in points) - radius_px)
    max_y = math.ceil(max(point[1] for point in points) + radius_px)
    shape = (max_y - min_y + 1, max_x - min_x + 1)
    _guard_raster_size(shape)
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    nominal = np.zeros(shape, dtype=bool)
    radius_squared = radius_px * radius_px
    tolerance = max(1.0, radius_squared) * 1e-12

    for first, second in zip(points, points[1:]):
        ax, ay = first
        bx, by = second
        dx = bx - ax
        dy = by - ay
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            continue
        projection = ((xx - ax) * dx + (yy - ay) * dy) / length_squared
        projection = np.clip(projection, 0.0, 1.0)
        closest_x = ax + projection * dx
        closest_y = ay + projection * dy
        distance_squared = (xx - closest_x) ** 2 + (yy - closest_y) ** 2
        nominal |= distance_squared <= radius_squared + tolerance

    return _crop_2d_result(
        "thick_polyline_2d",
        nominal,
        y0=min_y,
        x0=min_x,
        height=height,
        width=width,
    )


def rasterize_contour_stack(
    geometry: Any,
    image_shape: Sequence[int],
    *,
    calibration: Any | None = None,
) -> RasterizedRoi:
    """Rasterize explicit consecutive contours as a volume or inner shell."""

    depth, height, width = _stack_shape(image_shape)
    cal = RoiCalibration.from_value(calibration)
    slices_value = tuple(_field(geometry, "slices", ()))
    if not slices_value:
        raise RoiRasterizationError("A contour stack must contain at least one slice")
    planes = [int(_field(contour, "z_plane")) for contour in slices_value]
    if planes != sorted(planes) or len(set(planes)) != len(planes):
        raise RoiRasterizationError("Contour planes must be unique and sorted")
    if any(second != first + 1 for first, second in zip(planes, planes[1:])):
        raise RoiRasterizationError(
            "Contour planes must be consecutive; implicit interpolation is not allowed"
        )

    contour_points = [_points(contour, "exterior_xy_px") for contour in slices_value]
    for points in contour_points:
        _validate_polygon(points)
    min_x = min(math.floor(min(point[0] for point in points)) for points in contour_points)
    max_x = max(math.ceil(max(point[0] for point in points)) for points in contour_points)
    min_y = min(math.floor(min(point[1] for point in points)) for points in contour_points)
    max_y = max(math.ceil(max(point[1] for point in points)) for points in contour_points)
    nominal_shape = (len(planes), max_y - min_y + 1, max_x - min_x + 1)
    _guard_raster_size(nominal_shape)
    parent_nominal = np.zeros(nominal_shape, dtype=bool)
    for z_offset, points in enumerate(contour_points):
        shifted = tuple((x - min_x, y - min_y) for x, y in points)
        rr, cc = draw_polygon(
            np.asarray([point[1] for point in shifted], dtype=np.float64),
            np.asarray([point[0] for point in shifted], dtype=np.float64),
            shape=nominal_shape[1:],
        )
        parent_nominal[z_offset, rr, cc] = True

    sampling_mode = _enum_value(_field(geometry, "sampling_mode", "filled_volume"))
    if sampling_mode in {"filled", "volume"}:
        sampling_mode = "filled_volume"
    if sampling_mode == "filled_volume":
        sampled_nominal = parent_nominal
    elif sampling_mode == "inner_shell":
        if not cal.has_3d:
            raise RoiRasterizationError(
                "Inner-shell rasterization requires positive XY and Z calibration",
                reason="calibration_unavailable",
            )
        thickness = _field(geometry, "shell_thickness_um", None)
        if thickness is None or not math.isfinite(float(thickness)) or float(thickness) <= 0:
            raise RoiRasterizationError("Inner-shell thickness must be positive and finite")
        sampled_nominal = _inner_shell(
            parent_nominal,
            thickness_um=float(thickness),
            sampling=(float(cal.z_res), float(cal.xy_res), float(cal.xy_res)),
        )
    else:
        raise RoiRasterizationError(f"Unsupported contour-stack sampling mode: {sampling_mode!r}")

    z0_local = planes[0] - cal.plane_start
    z1_local = planes[-1] - cal.plane_start + 1
    nominal_bounds = (
        (z0_local, z1_local),
        (min_y, max_y + 1),
        (min_x, max_x + 1),
    )
    crop_slices_image: list[slice] = []
    crop_slices_nominal: list[slice] = []
    for (nominal_start, nominal_stop), image_size in zip(
        nominal_bounds, (depth, height, width)
    ):
        image_slice, nominal_slice = _axis_crop(
            nominal_start,
            nominal_stop,
            image_size,
        )
        crop_slices_image.append(image_slice)
        crop_slices_nominal.append(nominal_slice)
    sampled_crop = np.array(sampled_nominal[tuple(crop_slices_nominal)], copy=True)
    parent_crop = np.array(parent_nominal[tuple(crop_slices_nominal)], copy=True)
    nominal_count = int(np.count_nonzero(sampled_nominal))
    in_bounds_count = int(np.count_nonzero(sampled_crop))
    clipped = in_bounds_count < nominal_count or _touches_stack_boundary(
        parent_crop,
        tuple(crop_slices_image),
        (depth, height, width),
    )
    coverage = 1.0 if nominal_count == 0 else in_bounds_count / nominal_count
    surface = None
    if cal.has_3d and np.any(parent_crop):
        surface = estimate_surface_area(
            parent_crop,
            xy_res=float(cal.xy_res),
            z_res=float(cal.z_res),
        )
    return RasterizedRoi(
        geometry_kind="contour_stack_3d",
        mask=sampled_crop,
        slices=tuple(crop_slices_image),
        nominal_sample_count=nominal_count,
        in_bounds_sample_count=in_bounds_count,
        coverage_fraction=float(coverage),
        clipped=clipped,
        parent_mask=parent_crop,
        surface_area_um2=surface,
        warnings=("clipped_to_image",) if clipped else (),
    )


def estimate_surface_area(
    volume_mask: np.ndarray,
    *,
    xy_res: float,
    z_res: float,
) -> float:
    """Estimate the surface of a binary voxel object with physical spacing."""

    mask = np.asarray(volume_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("Surface area requires a (Z, Y, X) mask")
    if not np.any(mask):
        return 0.0
    if not all(math.isfinite(v) and v > 0 for v in (xy_res, z_res)):
        raise ValueError("Surface-area calibration must be positive and finite")
    # Padding supplies an explicit exterior even for a one-voxel object or an
    # object touching the image boundary.  Such cases retain a clipping warning
    # on RasterizedRoi rather than failing marching cubes as an open surface.
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    vertices, faces, _normals, _values = marching_cubes(
        padded,
        level=0.5,
        spacing=(float(z_res), float(xy_res), float(xy_res)),
        allow_degenerate=False,
    )
    return float(mesh_surface_area(vertices, faces))


def geometry_kind(geometry: Any) -> str:
    kind = _field(geometry, "kind", None)
    if kind is not None:
        return str(_enum_value(kind))
    name = type(geometry).__name__.lower()
    if "polyline" in name:
        return "thick_polyline_2d"
    if "contour" in name and "stack" in name:
        return "contour_stack_3d"
    if "polygon" in name:
        return "polygon_2d"
    raise RoiRasterizationError(
        f"Cannot determine geometry kind from {type(geometry).__name__}"
    )


def polyline_width_px(geometry: Any, calibration: Any | None = None) -> float:
    """Return a thick line's full width in pixels."""

    return 2.0 * _line_radius_px(geometry, RoiCalibration.from_value(calibration))


def _inner_shell(
    parent: np.ndarray,
    *,
    thickness_um: float,
    sampling: tuple[float, float, float],
) -> np.ndarray:
    padded = np.pad(np.asarray(parent, dtype=bool), 1, mode="constant")
    distance = distance_transform_edt(padded, sampling=sampling)
    shell = padded & (distance <= thickness_um)
    return shell[1:-1, 1:-1, 1:-1]


def _crop_2d_result(
    kind: str,
    nominal: np.ndarray,
    *,
    y0: int,
    x0: int,
    height: int,
    width: int,
) -> RasterizedRoi:
    y1 = y0 + nominal.shape[0]
    x1 = x0 + nominal.shape[1]
    image_y, nominal_y = _axis_crop(y0, y1, height)
    image_x, nominal_x = _axis_crop(x0, x1, width)
    crop = np.array(nominal[nominal_y, nominal_x], copy=True)
    nominal_count = int(np.count_nonzero(nominal))
    in_bounds_count = int(np.count_nonzero(crop))
    clipped = in_bounds_count < nominal_count
    coverage = 1.0 if nominal_count == 0 else in_bounds_count / nominal_count
    return RasterizedRoi(
        geometry_kind=kind,
        mask=crop,
        slices=(image_y, image_x),
        nominal_sample_count=nominal_count,
        in_bounds_sample_count=in_bounds_count,
        coverage_fraction=float(coverage),
        clipped=clipped,
        warnings=("clipped_to_image",) if clipped else (),
    )


def _axis_crop(
    nominal_start: int,
    nominal_stop: int,
    image_size: int,
) -> tuple[slice, slice]:
    overlap_start = max(nominal_start, 0)
    overlap_stop = min(nominal_stop, image_size)
    if overlap_start < overlap_stop:
        return (
            slice(overlap_start, overlap_stop),
            slice(overlap_start - nominal_start, overlap_stop - nominal_start),
        )
    image_position = 0 if nominal_stop <= 0 else image_size
    return slice(image_position, image_position), slice(0, 0)


def _touches_stack_boundary(
    mask: np.ndarray,
    image_slices: tuple[slice, ...],
    image_shape: tuple[int, int, int],
) -> bool:
    if not np.any(mask):
        return False
    for axis, (image_slice, image_size) in enumerate(zip(image_slices, image_shape)):
        if image_slice.start == 0 and np.any(np.take(mask, 0, axis=axis)):
            return True
        if image_slice.stop == image_size and np.any(np.take(mask, -1, axis=axis)):
            return True
    return False


def _polygon_nominal_mask(
    points: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, int, int]:
    min_x = math.floor(min(point[0] for point in points))
    max_x = math.ceil(max(point[0] for point in points))
    min_y = math.floor(min(point[1] for point in points))
    max_y = math.ceil(max(point[1] for point in points))
    shape = (max_y - min_y + 1, max_x - min_x + 1)
    _guard_raster_size(shape)
    shifted_x = np.asarray([point[0] - min_x for point in points], dtype=np.float64)
    shifted_y = np.asarray([point[1] - min_y for point in points], dtype=np.float64)
    rr, cc = draw_polygon(shifted_y, shifted_x, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask, min_y, min_x


def _validate_polygon(
    points: tuple[tuple[float, float], ...],
) -> None:
    if len(points) >= 2 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3 or len(set(points)) < 3:
        raise RoiRasterizationError("A polygon needs at least three distinct vertices")
    count = len(points)
    edges = [(points[index], points[(index + 1) % count]) for index in range(count)]
    for left_index, left in enumerate(edges):
        for right_index in range(left_index + 1, count):
            if right_index in {
                left_index,
                (left_index + 1) % count,
                (left_index - 1) % count,
            }:
                continue
            if _segments_intersect(left[0], left[1], edges[right_index][0], edges[right_index][1]):
                raise RoiRasterizationError("A polygon must not self-intersect")
    area_twice = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    )
    if math.isclose(area_twice, 0.0, abs_tol=1e-12):
        raise RoiRasterizationError("A polygon must have nonzero area")


def _validate_polyline(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if len(points) < 2 or len(set(points)) < 2:
        raise RoiRasterizationError("A polyline needs at least two distinct vertices")
    # Repeated adjacent points are harmless but should not create zero-length
    # segments in the distance and profile algorithms.
    compact: list[tuple[float, float]] = []
    for point in points:
        if not compact or point != compact[-1]:
            compact.append(point)
    return tuple(compact)


def _line_radius_px(geometry: Any, calibration: RoiCalibration) -> float:
    thickness = _field(geometry, "thickness", None)
    if thickness is None:
        # Tolerate flattened request/test data while keeping the persisted
        # dataclass contract authoritative.
        value = _field(geometry, "thickness_value", None)
        unit = _field(geometry, "thickness_unit", "px")
    else:
        value = _field(thickness, "value", None)
        unit = _field(thickness, "unit", None)
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise RoiRasterizationError("Polyline thickness must be positive and finite")
    unit = str(_enum_value(unit)).lower()
    if unit in {"pixel", "pixels"}:
        unit = "px"
    if unit in {"micron", "microns", "micrometre", "micrometres", "µm"}:
        unit = "um"
    if unit == "um":
        if not calibration.has_xy:
            raise RoiRasterizationError(
                "Physical line thickness requires positive XY calibration",
                reason="calibration_unavailable",
            )
        width_px = float(value) / float(calibration.xy_res)
    elif unit == "px":
        width_px = float(value)
    else:
        raise RoiRasterizationError(f"Unsupported polyline thickness unit: {unit!r}")
    return width_px / 2.0


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(
        first: tuple[float, float],
        second: tuple[float, float],
        third: tuple[float, float],
    ) -> float:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

    def on_segment(
        first: tuple[float, float],
        second: tuple[float, float],
        point: tuple[float, float],
    ) -> bool:
        return (
            min(first[0], second[0]) - 1e-12
            <= point[0]
            <= max(first[0], second[0]) + 1e-12
            and min(first[1], second[1]) - 1e-12
            <= point[1]
            <= max(first[1], second[1]) + 1e-12
        )

    values = (
        orientation(a, b, c),
        orientation(a, b, d),
        orientation(c, d, a),
        orientation(c, d, b),
    )
    if (values[0] > 0 > values[1] or values[0] < 0 < values[1]) and (
        values[2] > 0 > values[3] or values[2] < 0 < values[3]
    ):
        return True
    for value, first, second, point in (
        (values[0], a, b, c),
        (values[1], a, b, d),
        (values[2], c, d, a),
        (values[3], c, d, b),
    ):
        if math.isclose(value, 0.0, abs_tol=1e-12) and on_segment(first, second, point):
            return True
    return False


def _points(value: Any, field_name: str) -> tuple[tuple[float, float], ...]:
    raw = _field(value, field_name, ())
    points: list[tuple[float, float]] = []
    for point in raw:
        if len(point) != 2:
            raise RoiRasterizationError(f"{field_name} points must contain exactly x and y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise RoiRasterizationError("ROI coordinates must be finite")
        points.append((x, y))
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    return tuple(points)


def _plane_shape(image_shape: Sequence[int]) -> tuple[int, int]:
    shape = tuple(int(value) for value in image_shape)
    if len(shape) == 3:
        shape = shape[-2:]
    if len(shape) != 2 or any(value < 0 for value in shape):
        raise ValueError(f"Expected image shape (Y, X), got {tuple(image_shape)!r}")
    return shape


def _stack_shape(image_shape: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in image_shape)
    if len(shape) != 3 or any(value < 0 for value in shape):
        raise ValueError(f"Expected stack shape (Z, Y, X), got {tuple(image_shape)!r}")
    return shape


def _guard_raster_size(shape: Sequence[int]) -> None:
    size = math.prod(int(value) for value in shape)
    if size > _MAX_RASTER_ELEMENTS:
        raise RoiRasterizationError(
            f"ROI raster would contain {size:,} elements; refusing unsafe allocation"
        )


def _field(value: Any, name: str, default: Any = ...):
    if isinstance(value, Mapping):
        if default is ...:
            return value[name]
        return value.get(name, default)
    if default is ...:
        return getattr(value, name)
    return getattr(value, name, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _readonly_bool_array(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=bool, copy=True, order="C")
    array.setflags(write=False)
    return array


__all__ = [
    "ROI_RASTERIZATION_VERSION",
    "CroppedRoiMask",
    "RasterizedRoi",
    "RoiCalibration",
    "RoiMaskRasterizer",
    "RoiRasterizationError",
    "estimate_surface_area",
    "geometry_kind",
    "polyline_width_px",
    "rasterize_contour_stack",
    "rasterize_polygon",
    "rasterize_roi",
    "rasterize_thick_polyline",
]
