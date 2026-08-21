"""Finite-only scalar reducers and spatial profiles for subcellular ROIs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from .roi_rasterization import (
    RasterizedRoi,
    RoiCalibration,
    RoiRasterizationError,
    geometry_kind,
    polyline_width_px,
)


ROI_MEASUREMENT_ALGORITHM_VERSION = 1


@dataclass(frozen=True, slots=True)
class RoiMetricValue:
    """One scalar value with explicit missing-data provenance."""

    value: float | None
    reason: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("A measurement value must be finite or None")
        if self.value is None and not self.reason:
            raise ValueError("A missing measurement value requires a reason")
        if self.value is not None and self.reason is not None:
            raise ValueError("A valid measurement value cannot have a missing reason")

    @property
    def is_valid(self) -> bool:
        return self.value is not None


@dataclass(frozen=True, slots=True)
class RoiDistribution:
    """Optional finite-pixel distribution output."""

    histogram_counts: tuple[int, ...]
    histogram_edges: tuple[float, ...]
    quantiles: Mapping[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "histogram_counts", tuple(self.histogram_counts))
        object.__setattr__(self, "histogram_edges", tuple(self.histogram_edges))
        object.__setattr__(self, "quantiles", MappingProxyType(dict(self.quantiles)))


@dataclass(frozen=True, slots=True)
class RoiIntensityResult:
    """All scalar outputs for one geometry/image/channel sample."""

    status: str
    metrics: Mapping[str, RoiMetricValue]
    nominal_sample_count: int
    in_bounds_sample_count: int
    finite_sample_count: int
    coverage_fraction: float
    clipped: bool
    warnings: tuple[str, ...] = ()
    distribution: RoiDistribution | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def metric(self, key: str) -> RoiMetricValue:
        return self.metrics.get(key, RoiMetricValue(None, "metric_unavailable"))

    def value(self, key: str) -> float | None:
        return self.metric(key).value

    def missing_reason(self, key: str) -> str | None:
        return self.metric(key).reason

    @property
    def is_valid(self) -> bool:
        return self.status in {"valid", "valid_clipped"}


@dataclass(frozen=True, slots=True)
class RoiSpatialProfile:
    """Immutable arclength profile sampled across a thick polyline."""

    distance_um: tuple[float, ...]
    mean: tuple[float | None, ...]
    median: tuple[float | None, ...]
    sample_count: tuple[int, ...]
    sum: tuple[float | None, ...] | None = None
    missing_reasons: tuple[str | None, ...] = ()
    step_um: float = 0.0
    width_step_um: float = 0.0

    def __post_init__(self) -> None:
        distances = tuple(float(value) for value in self.distance_um)
        means = tuple(None if value is None else float(value) for value in self.mean)
        medians = tuple(None if value is None else float(value) for value in self.median)
        counts = tuple(int(value) for value in self.sample_count)
        sums = None if self.sum is None else tuple(
            None if value is None else float(value) for value in self.sum
        )
        reasons = tuple(self.missing_reasons) or tuple(
            None if value is not None else "no_finite_pixels" for value in means
        )
        lengths = {len(distances), len(means), len(medians), len(counts), len(reasons)}
        if sums is not None:
            lengths.add(len(sums))
        if len(lengths) != 1:
            raise ValueError("All spatial-profile vectors must have equal length")
        object.__setattr__(self, "distance_um", distances)
        object.__setattr__(self, "mean", means)
        object.__setattr__(self, "median", medians)
        object.__setattr__(self, "sample_count", counts)
        object.__setattr__(self, "sum", sums)
        object.__setattr__(self, "missing_reasons", reasons)

    @property
    def distances_um(self) -> tuple[float, ...]:
        return self.distance_um

    def __len__(self) -> int:
        return len(self.distance_um)


class RoiIntensityReducer:
    """Reduce a cropped ROI to raw and physically normalized scalars."""

    def reduce(
        self,
        image: np.ndarray,
        raster: RasterizedRoi | np.ndarray,
        *,
        geometry: Any | None = None,
        calibration: Any | None = None,
        physical_normalization_blocked: bool = False,
        include_distribution: bool = False,
        histogram_bins: int | Sequence[float] = 32,
        quantiles: Iterable[float] = (0.25, 0.5, 0.75),
    ) -> RoiIntensityResult:
        cal = RoiCalibration.from_value(calibration)
        if isinstance(raster, RasterizedRoi):
            rasterized = raster
        else:
            mask = np.asarray(raster, dtype=bool)
            kind = geometry_kind(geometry) if geometry is not None else (
                "mask_2d" if mask.ndim == 2 else "mask_3d"
            )
            rasterized = RasterizedRoi(
                geometry_kind=kind,
                mask=mask,
                slices=tuple(slice(0, size) for size in mask.shape),
                nominal_sample_count=int(np.count_nonzero(mask)),
                in_bounds_sample_count=int(np.count_nonzero(mask)),
                coverage_fraction=1.0,
                clipped=False,
            )

        if rasterized.in_bounds_sample_count == 0:
            return _missing_intensity_result(
                rasterized,
                "empty_mask",
                geometry,
                cal,
                physical_normalization_blocked=physical_normalization_blocked,
            )
        crop = rasterized.extract_from(np.asarray(image))
        if crop.shape != rasterized.mask.shape:
            raise ValueError("Image and ROI mask shapes do not match")
        selected = np.asarray(crop[rasterized.mask])
        finite_values = selected[np.isfinite(selected)].astype(np.float64, copy=False)
        finite_count = int(finite_values.size)
        if finite_count == 0:
            return _missing_intensity_result(
                rasterized,
                "no_finite_pixels",
                geometry,
                cal,
                physical_normalization_blocked=physical_normalization_blocked,
            )

        total = float(np.sum(finite_values, dtype=np.float64))
        metrics: dict[str, RoiMetricValue] = {
            "intensity.sum": RoiMetricValue(total, unit="a.u."),
            "intensity.mean": RoiMetricValue(total / finite_count, unit="a.u."),
            "intensity.median": RoiMetricValue(
                float(np.median(finite_values)), unit="a.u."
            ),
        }
        _add_geometry_and_normalized_metrics(
            metrics,
            total=total,
            finite_count=finite_count,
            raster=rasterized,
            geometry=geometry,
            calibration=cal,
            physical_normalization_blocked=physical_normalization_blocked,
        )
        distribution = None
        if include_distribution:
            probability_points = tuple(float(value) for value in quantiles)
            if any(not 0 <= value <= 1 for value in probability_points):
                raise ValueError("Quantiles must be between zero and one")
            counts, edges = np.histogram(finite_values, bins=histogram_bins)
            distribution = RoiDistribution(
                histogram_counts=tuple(int(value) for value in counts),
                histogram_edges=tuple(float(value) for value in edges),
                quantiles={
                    value: float(np.quantile(finite_values, value))
                    for value in probability_points
                },
            )
        return RoiIntensityResult(
            status="valid_clipped" if rasterized.clipped else "valid",
            metrics=metrics,
            nominal_sample_count=rasterized.nominal_sample_count,
            in_bounds_sample_count=rasterized.in_bounds_sample_count,
            finite_sample_count=finite_count,
            coverage_fraction=rasterized.coverage_fraction,
            clipped=rasterized.clipped,
            warnings=rasterized.warnings,
            distribution=distribution,
        )

    __call__ = reduce


class RoiProfileSampler:
    """Sample a thick line along arclength using bilinear interpolation."""

    def sample(
        self,
        image_plane: np.ndarray,
        geometry: Any,
        *,
        calibration: Any | None = None,
        step_um: float | None = None,
        width_step_um: float | None = None,
        include_sum: bool = True,
    ) -> RoiSpatialProfile:
        image = np.asarray(image_plane)
        if image.ndim != 2:
            raise ValueError("Line profiles require one (Y, X) image plane")
        if geometry_kind(geometry) != "thick_polyline_2d":
            raise ValueError("Line profiles require a thick polyline geometry")
        cal = RoiCalibration.from_value(calibration)
        if not cal.has_xy:
            raise RoiRasterizationError(
                "Line profiles require positive XY calibration",
                reason="calibration_unavailable",
            )
        points = _polyline_points(geometry)
        width_px = polyline_width_px(geometry, cal)
        requested_step = float(cal.xy_res) if step_um is None else float(step_um)
        requested_width_step = (
            float(cal.xy_res) if width_step_um is None else float(width_step_um)
        )
        if not math.isfinite(requested_step) or requested_step <= 0:
            raise ValueError("Profile step must be positive and finite")
        if not math.isfinite(requested_width_step) or requested_width_step <= 0:
            raise ValueError("Profile width step must be positive and finite")
        # Never sample across the line more coarsely than one XY pixel.
        requested_width_step = min(requested_width_step, float(cal.xy_res))

        segment_vectors = np.diff(points, axis=0)
        segment_lengths_px = np.linalg.norm(segment_vectors, axis=1)
        nonzero = segment_lengths_px > 0
        segment_vectors = segment_vectors[nonzero]
        segment_lengths_px = segment_lengths_px[nonzero]
        if segment_lengths_px.size == 0:
            raise ValueError("A profile polyline needs at least two distinct points")
        start_points = points[:-1][nonzero]
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths_px)))
        total_px = float(cumulative[-1])
        total_um = total_px * float(cal.xy_res)

        # Equal spacing (rather than a fixed-step remainder) makes vertex
        # reversal produce the exact reversed sample grid.
        along_count = max(2, int(math.ceil(total_um / requested_step)) + 1)
        along_um = np.linspace(0.0, total_um, along_count, dtype=np.float64)
        along_px = along_um / float(cal.xy_res)
        segment_indices = np.searchsorted(cumulative[1:], along_px, side="right")
        segment_indices = np.minimum(segment_indices, len(segment_lengths_px) - 1)
        local = along_px - cumulative[segment_indices]
        unit_tangents = segment_vectors / segment_lengths_px[:, None]
        centres = start_points[segment_indices] + unit_tangents[segment_indices] * local[:, None]
        tangents = unit_tangents[segment_indices]
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

        width_um = width_px * float(cal.xy_res)
        across_count = max(2, int(math.ceil(width_um / requested_width_step)) + 1)
        across_um = np.linspace(-width_um / 2.0, width_um / 2.0, across_count)
        across_px = across_um / float(cal.xy_res)
        sample_xy = centres[:, None, :] + normals[:, None, :] * across_px[None, :, None]
        coordinates = np.stack((sample_xy[..., 1], sample_xy[..., 0]), axis=0)
        values = map_coordinates(
            image.astype(np.float64, copy=False),
            coordinates,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        finite = np.isfinite(values)
        counts = np.count_nonzero(finite, axis=1).astype(int)
        means: list[float | None] = []
        medians: list[float | None] = []
        sums: list[float | None] = []
        reasons: list[str | None] = []
        for row, row_finite, count in zip(values, finite, counts):
            if count == 0:
                means.append(None)
                medians.append(None)
                sums.append(None)
                reasons.append("no_finite_pixels")
                continue
            selected = row[row_finite]
            total = float(np.sum(selected, dtype=np.float64))
            means.append(total / int(count))
            medians.append(float(np.median(selected)))
            sums.append(total)
            reasons.append(None)
        actual_step = 0.0 if along_count == 1 else total_um / (along_count - 1)
        actual_width_step = width_um / (across_count - 1)
        return RoiSpatialProfile(
            distance_um=tuple(float(value) for value in along_um),
            mean=tuple(means),
            median=tuple(medians),
            sum=tuple(sums) if include_sum else None,
            sample_count=tuple(int(value) for value in counts),
            missing_reasons=tuple(reasons),
            step_um=float(actual_step),
            width_step_um=float(actual_width_step),
        )

    __call__ = sample


def reduce_roi_intensity(
    image: np.ndarray,
    raster: RasterizedRoi | np.ndarray,
    *,
    geometry: Any | None = None,
    calibration: Any | None = None,
    **kwargs: Any,
) -> RoiIntensityResult:
    return RoiIntensityReducer().reduce(
        image,
        raster,
        geometry=geometry,
        calibration=calibration,
        **kwargs,
    )


def sample_roi_profile(
    image_plane: np.ndarray,
    geometry: Any,
    *,
    calibration: Any | None = None,
    **kwargs: Any,
) -> RoiSpatialProfile:
    return RoiProfileSampler().sample(
        image_plane,
        geometry,
        calibration=calibration,
        **kwargs,
    )


def _missing_intensity_result(
    raster: RasterizedRoi,
    reason: str,
    geometry: Any | None,
    calibration: RoiCalibration,
    *,
    physical_normalization_blocked: bool = False,
) -> RoiIntensityResult:
    metrics: dict[str, RoiMetricValue] = {
        "intensity.sum": RoiMetricValue(None, reason, "a.u."),
        "intensity.mean": RoiMetricValue(None, reason, "a.u."),
        "intensity.median": RoiMetricValue(None, reason, "a.u."),
    }
    _add_geometry_and_normalized_metrics(
        metrics,
        total=None,
        finite_count=0,
        raster=raster,
        geometry=geometry,
        calibration=calibration,
        physical_normalization_blocked=physical_normalization_blocked,
        missing_reason=reason,
    )
    return RoiIntensityResult(
        status=reason,
        metrics=metrics,
        nominal_sample_count=raster.nominal_sample_count,
        in_bounds_sample_count=raster.in_bounds_sample_count,
        finite_sample_count=0,
        coverage_fraction=raster.coverage_fraction,
        clipped=raster.clipped,
        warnings=raster.warnings,
    )


def _add_geometry_and_normalized_metrics(
    metrics: dict[str, RoiMetricValue],
    *,
    total: float | None,
    finite_count: int,
    raster: RasterizedRoi,
    geometry: Any | None,
    calibration: RoiCalibration,
    physical_normalization_blocked: bool = False,
    missing_reason: str | None = None,
) -> None:
    kind = raster.geometry_kind
    calibration_reason = (
        "calibration_mismatch"
        if physical_normalization_blocked
        else "calibration_unavailable"
    )
    data_reason = missing_reason or "no_finite_pixels"
    if kind in {"polygon_2d", "thick_polyline_2d", "mask_2d"}:
        if physical_normalization_blocked or not calibration.has_xy:
            area_value = RoiMetricValue(None, calibration_reason, "um2")
            normalized = RoiMetricValue(None, calibration_reason, "a.u./um2")
        elif finite_count == 0:
            area_value = RoiMetricValue(None, data_reason, "um2")
            normalized = RoiMetricValue(None, data_reason, "a.u./um2")
        else:
            finite_area = finite_count * float(calibration.xy_res) ** 2
            area_value = RoiMetricValue(finite_area, unit="um2")
            normalized = RoiMetricValue(float(total) / finite_area, unit="a.u./um2")
        metrics["geometry.area_um2"] = area_value
        metrics["intensity.sum_per_area_um2"] = normalized

    if kind == "thick_polyline_2d":
        length_um = (
            None
            if physical_normalization_blocked
            else _polyline_length_um(geometry, calibration)
        )
        metrics["geometry.length_um"] = (
            RoiMetricValue(length_um, unit="um")
            if length_um is not None
            else RoiMetricValue(None, calibration_reason, "um")
        )
        if length_um is None:
            metrics["intensity.sum_per_length_um"] = RoiMetricValue(
                None, calibration_reason, "a.u./um"
            )
        elif finite_count == 0 or raster.nominal_sample_count == 0:
            metrics["intensity.sum_per_length_um"] = RoiMetricValue(
                None, data_reason, "a.u./um"
            )
        else:
            finite_length = length_um * finite_count / raster.nominal_sample_count
            metrics["intensity.sum_per_length_um"] = RoiMetricValue(
                float(total) / finite_length,
                unit="a.u./um",
            )

    if kind in {"contour_stack_3d", "mask_3d"}:
        if physical_normalization_blocked or not calibration.has_3d:
            volume = RoiMetricValue(None, calibration_reason, "um3")
            normalized_volume = RoiMetricValue(None, calibration_reason, "a.u./um3")
        elif finite_count == 0:
            volume = RoiMetricValue(None, data_reason, "um3")
            normalized_volume = RoiMetricValue(None, data_reason, "a.u./um3")
        else:
            finite_volume = (
                finite_count
                * float(calibration.xy_res) ** 2
                * float(calibration.z_res)
            )
            volume = RoiMetricValue(finite_volume, unit="um3")
            normalized_volume = RoiMetricValue(
                float(total) / finite_volume,
                unit="a.u./um3",
            )
        metrics["geometry.volume_um3"] = volume
        metrics["intensity.sum_per_volume_um3"] = normalized_volume
        surface = None if physical_normalization_blocked else raster.surface_area_um2
        metrics["geometry.surface_area_um2"] = (
            RoiMetricValue(float(surface), unit="um2")
            if surface is not None
            else RoiMetricValue(None, calibration_reason, "um2")
        )
        sampling_mode = _geometry_field(geometry, "sampling_mode", None)
        if getattr(sampling_mode, "value", sampling_mode) == "inner_shell":
            if surface is None:
                metrics["intensity.sum_per_surface_area_um2"] = RoiMetricValue(
                    None, calibration_reason, "a.u./um2"
                )
            elif finite_count == 0 or raster.in_bounds_sample_count == 0 or surface <= 0:
                metrics["intensity.sum_per_surface_area_um2"] = RoiMetricValue(
                    None, data_reason, "a.u./um2"
                )
            else:
                finite_surface = surface * finite_count / raster.in_bounds_sample_count
                metrics["intensity.sum_per_surface_area_um2"] = RoiMetricValue(
                    float(total) / finite_surface,
                    unit="a.u./um2",
                )


def _polyline_length_um(
    geometry: Any | None,
    calibration: RoiCalibration,
) -> float | None:
    if geometry is None or not calibration.has_xy:
        return None
    points = _polyline_points(geometry)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.sum(lengths, dtype=np.float64) * float(calibration.xy_res))


def _polyline_points(geometry: Any) -> np.ndarray:
    raw = geometry.get("points_xy_px", ()) if isinstance(geometry, Mapping) else getattr(
        geometry, "points_xy_px", ()
    )
    points = np.asarray(tuple(raw), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError("A profile polyline needs at least two (x, y) points")
    if not np.all(np.isfinite(points)):
        raise ValueError("Polyline points must be finite")
    compact = np.concatenate(([True], np.any(np.diff(points, axis=0) != 0, axis=1)))
    points = points[compact]
    if len(points) < 2:
        raise ValueError("A profile polyline needs at least two distinct points")
    return points


def _geometry_field(geometry: Any | None, name: str, default: Any) -> Any:
    if geometry is None:
        return default
    if isinstance(geometry, Mapping):
        return geometry.get(name, default)
    return getattr(geometry, name, default)


__all__ = [
    "ROI_MEASUREMENT_ALGORITHM_VERSION",
    "RoiDistribution",
    "RoiIntensityReducer",
    "RoiIntensityResult",
    "RoiMetricValue",
    "RoiProfileSampler",
    "RoiSpatialProfile",
    "reduce_roi_intensity",
    "sample_roi_profile",
]
