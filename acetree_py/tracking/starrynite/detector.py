"""Native, stage-aware 3-D nucleus detector for StarryNite workflows.

The detector follows the documented StarryNite stages (anisotropic smoothing,
local maxima, bounded nucleus support, and per-candidate measurements) while
remaining an independent implementation.  Its output is the ordinary AceTree
``Detection`` contract, so it can be paired with Simple LAP, the native
division tracker, or third-party trackers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from ..api import Calibration, Detection
from .models import sha256_file


_DEFAULT_SETTINGS = MappingProxyType(
    {
        "TARGET_CHANNEL": 1,
        "RADIUS": 4.0,
        "THRESHOLD": 0.0,
        # StarryNite's production point is the integer, ray-recentered slice
        # maximum.  The support-weighted centroid is a native opt-in because
        # it can pull two close daughters toward the same midpoint.
        "DO_SUBPIXEL_LOCALIZATION": False,
        "DO_MEDIAN_FILTERING": False,
        # Legacy StarryNite detector controls. SIGMA multiplies the expected
        # XY cell diameter; INTENSITY_THRESHOLD is an absolute DoG response.
        "SIGMA": 1.0,
        "INTENSITY_THRESHOLD": 4.0,
        "MIN_LOCAL_CONTRAST": 0.0,
        "BOUNDARY_PERCENT": 0.5,
        "LARGE_RAY_THRESHOLD": 1.5,
        "SMALL_RAY_THRESHOLD": 1.0 / 3.0,
        "NNDIST_MERGE": 0.8,
        "AR_MERGE": 1.6,
        "RANGE_THRESHOLD": 1.0,
        "SPLIT_THRESHOLD": 100.0,
        "MERGE_LOWER": -300.0,
        "MERGE_SPLIT": 1.0,
        "MIN_SEPARATION": 0.0,
        "DARK_NUCLEI": False,
        # ROI coordinates use StarryNite's one-based XY convention.  When the
        # supplied image is already cropped, ROI_X/Y_OFFSET are added before
        # the polygon test just like createDiskSet.m's ROI branch.
        "ROI_POINTS_XY": (),
        "ROI_CROPPED": False,
        "ROI_X_OFFSET": 0.0,
        "ROI_Y_OFFSET": 0.0,
        "ROI_X_MAX": 0.0,
        "ROI_Y_MAX": 0.0,
        "STARRYNITE_CELL_COUNT": 0,
        "STARRYNITE_STAGE_INDEX": 0,
        "STARRYNITE_PARAMETER_FILE": "",
        "STARRYNITE_PARAMETER_SHA256": "",
        "STARRYNITE_DISTRIBUTION_FILE": "",
        # Expected request-time identity.  The similarly named feature without
        # ``SOURCE`` remains the hash actually used for each emitted row.
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": "",
        "STARRYNITE_USE_STATIC_DIAMETER": False,
    }
)


def _number(name: str, value: Any, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(settings) - set(_DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(
            "Unsupported StarryNite detector setting(s): " + ", ".join(sorted(unknown))
        )
    values = dict(_DEFAULT_SETTINGS)
    values.update(settings)
    channel = values["TARGET_CHANNEL"]
    if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)):
        raise ValueError("TARGET_CHANNEL must be an integer")
    if int(channel) < 1:
        raise ValueError("TARGET_CHANNEL must be a positive 1-based channel")
    values["TARGET_CHANNEL"] = int(channel)

    for key in (
        "RADIUS",
        "THRESHOLD",
        "SIGMA",
        "INTENSITY_THRESHOLD",
        "MIN_LOCAL_CONTRAST",
        "BOUNDARY_PERCENT",
        "MIN_SEPARATION",
        "LARGE_RAY_THRESHOLD",
        "SMALL_RAY_THRESHOLD",
        "NNDIST_MERGE",
        "AR_MERGE",
        "RANGE_THRESHOLD",
        "SPLIT_THRESHOLD",
        "MERGE_SPLIT",
        "ROI_X_OFFSET",
        "ROI_Y_OFFSET",
        "ROI_X_MAX",
        "ROI_Y_MAX",
    ):
        values[key] = _number(key, values[key], minimum=0.0)
    values["MERGE_LOWER"] = _number("MERGE_LOWER", values["MERGE_LOWER"])
    if values["RADIUS"] <= 0:
        raise ValueError("RADIUS must be positive")
    if values["SIGMA"] <= 0:
        raise ValueError("SIGMA must be positive")
    if not 0 < values["BOUNDARY_PERCENT"] <= 1:
        raise ValueError("BOUNDARY_PERCENT must be in (0, 1]")
    if values["LARGE_RAY_THRESHOLD"] <= 0:
        raise ValueError("LARGE_RAY_THRESHOLD must be positive")
    if values["SMALL_RAY_THRESHOLD"] <= 0:
        raise ValueError("SMALL_RAY_THRESHOLD must be positive")
    if values["NNDIST_MERGE"] < 0:
        raise ValueError("NNDIST_MERGE cannot be negative")
    if values["AR_MERGE"] < 0:
        raise ValueError("AR_MERGE cannot be negative")

    for key in (
        "DO_SUBPIXEL_LOCALIZATION",
        "DO_MEDIAN_FILTERING",
        "DARK_NUCLEI",
        "ROI_CROPPED",
        "STARRYNITE_USE_STATIC_DIAMETER",
    ):
        if not isinstance(values[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be boolean")
        values[key] = bool(values[key])
    roi_points = values["ROI_POINTS_XY"]
    if roi_points is None or (isinstance(roi_points, str) and roi_points == ""):
        roi_points = ()
    try:
        roi_array = np.asarray(roi_points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ROI_POINTS_XY must be an Nx2 numeric polygon") from exc
    if roi_array.size == 0:
        values["ROI_POINTS_XY"] = ()
    else:
        if (
            roi_array.ndim != 2
            or roi_array.shape[1] != 2
            or roi_array.shape[0] < 3
            or not np.all(np.isfinite(roi_array))
        ):
            raise ValueError("ROI_POINTS_XY must be a finite Nx2 polygon with N >= 3")
        values["ROI_POINTS_XY"] = tuple(
            tuple(float(item) for item in row) for row in roi_array
        )
    for key in ("STARRYNITE_CELL_COUNT", "STARRYNITE_STAGE_INDEX"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{key} must be an integer")
        if int(value) < 0:
            raise ValueError(f"{key} cannot be negative")
        values[key] = int(value)
    for key in (
        "STARRYNITE_PARAMETER_FILE",
        "STARRYNITE_PARAMETER_SHA256",
        "STARRYNITE_DISTRIBUTION_FILE",
    ):
        value = values[key]
        if value is None:
            value = ""
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{key} must be a path string")
        values[key] = str(value)
    distribution_digest = values["STARRYNITE_DISTRIBUTION_SOURCE_SHA256"]
    if distribution_digest is None:
        distribution_digest = ""
    if not isinstance(distribution_digest, str):
        raise ValueError("STARRYNITE_DISTRIBUTION_SOURCE_SHA256 must be a string")
    values["STARRYNITE_DISTRIBUTION_SOURCE_SHA256"] = distribution_digest
    if distribution_digest and (
        len(distribution_digest) != 64
        or any(character not in "0123456789abcdef" for character in distribution_digest)
    ):
        raise ValueError(
            "STARRYNITE_DISTRIBUTION_SOURCE_SHA256 must be an empty string or "
            "a lowercase SHA-256 digest"
        )
    return values


def _select_image(stack_zyx: np.ndarray, channel_1based: int) -> np.ndarray:
    image = np.asarray(stack_zyx)
    if image.ndim == 4:
        channel = channel_1based - 1
        if not 0 <= channel < image.shape[0]:
            raise IndexError(
                f"TARGET_CHANNEL {channel_1based} is unavailable for {image.shape[0]} channels"
            )
        image = image[channel]
    elif image.ndim != 3:
        raise ValueError(f"Expected a ZYX or CZYX image, got shape {image.shape}")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("Detector input must be numeric")
    return np.array(image, dtype=np.float32, copy=True)


@dataclass(frozen=True, slots=True)
class StarryNiteDogFilter:
    """Resolved legacy DoG kernel parameters in NumPy ZYX order."""

    inner_sigma_zyx: tuple[float, float, float]
    outer_sigma_zyx: tuple[float, float, float]
    inner_support_zyx: tuple[int, int, int]
    outer_support_zyx: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class LegacyRadialGeometry:
    """One candidate's ray-recentered XY geometry from StarryNite."""

    center_zyx_px: tuple[float, float, float]
    diameter_xy_px: float
    valid_ray_count: int
    ray_endpoints_xy_px: tuple[tuple[float, float], ...] = ()
    peak_response: float = 0.0


@dataclass(frozen=True, slots=True)
class LegacyResolvedCandidate:
    """Candidate geometry after slice claims and geometric conflict merging."""

    center_zyx_px: tuple[float, float, float]
    diameter_xy_px: float
    representative_peak_zyx: tuple[int, int, int]
    valid_ray_count: int
    claimed_slice_count: int
    merged_candidate_count: int
    claimed_slices: tuple[LegacyRadialGeometry, ...] = ()
    aspect_ratio: float = 1.0
    xy_principal_variance: float = 0.0
    xy_secondary_variance: float = 0.0
    log_odds_sum: float | None = None
    claimed_log_odds: tuple[float, ...] = ()
    recovery_round: int = 0


def legacy_dog_filter_parameters(
    radius_um: float,
    sigma_factor: float,
    calibration: Calibration,
) -> StarryNiteDogFilter:
    """Reproduce ``processVolume.m`` kernel sizing from physical calibration."""

    radius = _number("RADIUS", radius_um, minimum=0.0)
    factor = _number("SIGMA", sigma_factor, minimum=0.0)
    if radius <= 0 or factor <= 0:
        raise ValueError("RADIUS and SIGMA must be positive")
    diameter_xy_px = 2.0 * radius / calibration.xy_um
    legacy_sigma = diameter_xy_px * factor
    anisotropy = calibration.z_um / calibration.xy_um

    def support(multiplier: float) -> tuple[int, int, int]:
        scaled = legacy_sigma * multiplier
        xy = 2 * math.floor(scaled / 2.0)
        z = max(3, 2 * math.floor(scaled / anisotropy / 2.0) + 1) - 1
        if xy < 2 or z < 2:
            raise ValueError(
                "The resolved StarryNite Gaussian support is smaller than two pixels"
            )
        return int(z), int(xy), int(xy)

    denominator = 4.0 * math.sqrt(2.0 * math.log(2.0))
    inner_support = support(1.0)
    outer_support = support(1.6)
    return StarryNiteDogFilter(
        inner_sigma_zyx=tuple(value / denominator for value in inner_support),
        outer_sigma_zyx=tuple(value / denominator for value in outer_support),
        inner_support_zyx=inner_support,
        outer_support_zyx=outer_support,
    )


def _separable_gaussian(
    image: np.ndarray,
    sigma_zyx: tuple[float, float, float],
    support_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Apply the same finite, normalized 1-D kernels and replicate boundary."""

    filtered = np.asarray(image, dtype=np.float32)
    # Keep the ZYX pass order stable.  SciPy and MATLAB use different
    # single-precision accumulation implementations; this ordering has the
    # smallest measured error across the oracle corpus.
    for axis in (0, 1, 2):
        sigma = sigma_zyx[axis]
        support = support_zyx[axis]
        radius = math.ceil(support / 2.0)
        coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
        kernel /= np.sum(kernel)
        filtered = ndimage.convolve1d(
            filtered,
            kernel.astype(np.float32),
            axis=axis,
            mode="nearest",
        )
    return np.asarray(filtered, dtype=np.float32)


def _legacy_dog_components(
    image_zyx: np.ndarray,
    radius_um: float,
    sigma_factor: float,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray, StarryNiteDogFilter]:
    """Return the DoG response, inner blur, and resolved legacy kernel."""

    image = np.asarray(image_zyx)
    if image.ndim != 3:
        raise ValueError(f"Expected a ZYX image, got shape {image.shape}")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("DoG input must be numeric")
    working = np.asarray(image, dtype=np.float32)
    parameters = legacy_dog_filter_parameters(radius_um, sigma_factor, calibration)
    inner = _separable_gaussian(
        working,
        parameters.inner_sigma_zyx,
        parameters.inner_support_zyx,
    )
    outer = _separable_gaussian(
        working,
        parameters.outer_sigma_zyx,
        parameters.outer_support_zyx,
    )
    return np.asarray(inner - outer, dtype=np.float32), inner, parameters


def legacy_dog_response(
    image_zyx: np.ndarray,
    radius_um: float,
    sigma_factor: float,
    calibration: Calibration,
) -> np.ndarray:
    """Filter one ZYX volume using the legacy ``processVolume.m`` DoG kernel.

    This public stage boundary is useful for differential validation against a
    locally installed MATLAB StarryNite checkout.  It deliberately performs
    only the finite-kernel DoG operation; detection thresholds and maxima are
    evaluated by later stages.
    """

    response, _inner, _parameters = _legacy_dog_components(
        image_zyx,
        radius_um,
        sigma_factor,
        calibration,
    )
    return response


def _footprint(radius_um: float, calibration: Calibration) -> np.ndarray:
    spacing = np.asarray(calibration.spacing_zyx, dtype=float)
    half_width = np.maximum(1, np.ceil(radius_um / spacing).astype(int))
    z, y, x = np.ogrid[
        -half_width[0] : half_width[0] + 1,
        -half_width[1] : half_width[1] + 1,
        -half_width[2] : half_width[2] + 1,
    ]
    return (
        (z * spacing[0]) ** 2
        + (y * spacing[1]) ** 2
        + (x * spacing[2]) ** 2
        <= radius_um**2 + np.finfo(float).eps
    )


def _canonical_maxima(response: np.ndarray, threshold: float, footprint: np.ndarray):
    maxima = ndimage.maximum_filter(response, footprint=footprint, mode="nearest")
    mask = np.isfinite(response) & (response >= threshold) & (response == maxima)
    if not np.any(mask):
        return []
    labels, _count = ndimage.label(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    coordinates = np.argwhere(mask)
    label_values = labels[mask]
    response_values = response[mask]
    best_by_label: dict[int, tuple[float, tuple[int, int, int]]] = {}
    for label_value, response_value, coordinate in zip(
        label_values,
        response_values,
        coordinates,
        strict=True,
    ):
        label_index = int(label_value)
        point = tuple(int(item) for item in coordinate)
        current = best_by_label.get(label_index)
        if current is None or float(response_value) > current[0]:
            best_by_label[label_index] = (float(response_value), point)
        elif float(response_value) == current[0] and point < current[1]:
            best_by_label[label_index] = (float(response_value), point)
    peaks = [item[1] for item in best_by_label.values()]
    peaks.sort()
    return peaks


_LEGACY_RAY_DIRECTIONS_YX = np.asarray(
    (
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.7071, 0.7071),
        (-0.7071, -0.7071),
        (0.7071, -0.7071),
        (-0.7071, 0.7071),
        (0.8409, 0.3483),
        (-0.8409, -0.3483),
        (-0.8409, 0.3483),
        (0.8409, -0.3483),
        (0.3483, 0.8409),
        (-0.3483, -0.8409),
        (0.3483, -0.8409),
        (-0.3483, 0.8409),
    ),
    dtype=np.float64,
)
_LEGACY_RAY_PREVIOUS = np.asarray(
    (15, 14, 8, 9, 12, 13, 11, 10, 4, 5, 3, 2, 0, 1, 6, 7),
    dtype=np.intp,
)
_LEGACY_RAY_NEXT = np.asarray(
    (12, 13, 11, 10, 8, 9, 14, 15, 2, 3, 7, 6, 4, 5, 1, 0),
    dtype=np.intp,
)
_LEGACY_RAY_CLOCKWISE_RANK = np.asarray(
    (0, 8, 4, 12, 2, 10, 6, 14, 3, 11, 13, 5, 1, 9, 7, 15),
    dtype=np.intp,
)


def _matlab_round(value: float | np.ndarray) -> int | np.ndarray:
    """Round halves away from zero, matching MATLAB rather than NumPy."""

    array = np.asarray(value, dtype=np.float64)
    rounded = np.sign(array) * np.floor(np.abs(array) + 0.5)
    if rounded.ndim == 0:
        return int(rounded)
    return rounded.astype(np.intp)


def legacy_radial_geometry(
    response_zyx: np.ndarray,
    peak_zyx: tuple[int, int, int],
    expected_diameter_xy_px: float,
    boundary_percent: float,
    *,
    large_ray_threshold: float = 1.5,
    small_ray_threshold: float = 1.0 / 3.0,
) -> LegacyRadialGeometry:
    """Port StarryNite's 16-ray XY diameter and integer recentering stage.

    This mirrors ``calculateSphereDiameters_geometric.m``: threshold crossings
    are measured on the filtered center slice, abrupt adjacent-ray changes are
    rejected, valley distances repair missing rays, a sufficiently complete
    polygon recenters the point, and the 80th-percentile radius is converted to
    the legacy odd-pixel diameter.
    """

    response = np.asarray(response_zyx)
    if response.ndim != 3 or not np.issubdtype(response.dtype, np.number):
        raise ValueError("response_zyx must be a numeric ZYX volume")
    if len(peak_zyx) != 3:
        raise ValueError("peak_zyx must have three coordinates")
    peak = tuple(int(item) for item in peak_zyx)
    if any(item < 0 or item >= response.shape[axis] for axis, item in enumerate(peak)):
        raise IndexError("peak_zyx lies outside response_zyx")
    expected = _number(
        "expected_diameter_xy_px", expected_diameter_xy_px, minimum=0.0
    )
    boundary = _number("boundary_percent", boundary_percent, minimum=0.0)
    larger = _number("large_ray_threshold", large_ray_threshold, minimum=0.0)
    smaller = _number("small_ray_threshold", small_ray_threshold, minimum=0.0)
    if expected <= 0 or not 0 < boundary <= 1 or larger <= 0 or smaller <= 0:
        raise ValueError(
            "diameter and ray thresholds must be positive and boundary_percent "
            "must be in (0, 1]"
        )

    z, center_y, center_x = peak
    maximum = float(response[peak])
    crossing_steps = np.zeros(len(_LEGACY_RAY_DIRECTIONS_YX), dtype=np.float64)
    valley_steps = np.zeros_like(crossing_steps)
    maximum_step = int(math.ceil(expected * 1.5))
    height, width = response.shape[1:]
    for ray_index, (direction_y, direction_x) in enumerate(
        _LEGACY_RAY_DIRECTIONS_YX
    ):
        minimum = maximum
        valley = 0
        crossed = 0
        for step in range(1, maximum_step + 1):
            sample_y = int(
                np.clip(_matlab_round(center_y + step * direction_y), 0, height - 1)
            )
            sample_x = int(
                np.clip(_matlab_round(center_x + step * direction_x), 0, width - 1)
            )
            value = float(response[z, sample_y, sample_x])
            if value < minimum:
                minimum = value
            elif not valley and not crossed and value > minimum:
                valley = step - 1
                valley_steps[ray_index] = valley
            if not crossed and value < maximum * boundary:
                crossed = step
                crossing_steps[ray_index] = step
                break

    ray_norms = np.linalg.norm(_LEGACY_RAY_DIRECTIONS_YX, axis=1)
    distances = crossing_steps * ray_norms
    valley_distances = valley_steps * ray_norms
    valid = np.flatnonzero(distances != 0)
    if valid.size:
        median_distance = float(np.median(distances[valid]))
        current = int(valid[np.argmin(np.abs(distances[valid] - median_distance))])
        for _ in range(len(_LEGACY_RAY_DIRECTIONS_YX) - 1):
            following = int(_LEGACY_RAY_NEXT[current])
            while distances[current] == 0:
                current = int(_LEGACY_RAY_PREVIOUS[current])
            ratio = distances[following] / distances[current]
            if ratio > larger or ratio < smaller:
                distances[following] = 0.0
            current = following
    repair = (distances == 0) & (valley_distances != 0)
    distances[repair] = valley_distances[repair]

    direction_y = _LEGACY_RAY_DIRECTIONS_YX[:, 0]
    direction_x = _LEGACY_RAY_DIRECTIONS_YX[:, 1]
    relative_x = distances * direction_x
    relative_y = distances * direction_y
    image_x = np.clip(relative_x + center_x, 0, width - 1)
    image_y = np.clip(relative_y + center_y, 0, height - 1)
    valid_mask = distances != 0
    coverage = int(np.count_nonzero(valid_mask))
    recentered_x = center_x
    recentered_y = center_y
    if coverage > 13:
        valid_indices = np.flatnonzero(valid_mask)
        clockwise = valid_indices[
            np.argsort(_LEGACY_RAY_CLOCKWISE_RANK[valid_indices], kind="stable")
        ]
        polygon_x = relative_x[clockwise]
        polygon_y = relative_y[clockwise]
        following_x = np.roll(polygon_x, -1)
        following_y = np.roll(polygon_y, -1)
        cross = polygon_x * following_y - polygon_y * following_x
        twice_area = float(np.sum(cross))
        if abs(twice_area) > np.finfo(float).eps:
            centroid_x = float(np.sum((polygon_x + following_x) * cross))
            centroid_y = float(np.sum((polygon_y + following_y) * cross))
            # Six times the signed polygon area equals three times twice_area.
            recentered_x += int(_matlab_round(centroid_x / (3.0 * twice_area)))
            recentered_y += int(_matlab_round(centroid_y / (3.0 * twice_area)))
            recentered_x = int(np.clip(recentered_x, 0, width - 1))
            recentered_y = int(np.clip(recentered_y, 0, height - 1))

    if coverage > 13:
        center_distances = np.sqrt(
            (image_x[valid_mask] - recentered_x) ** 2
            + (image_y[valid_mask] - recentered_y) ** 2
        )
        ordered_distances = np.sort(center_distances)
        percentile_index_1based = int(_matlab_round(len(ordered_distances) * 0.8))
        percentile_index = int(
            np.clip(percentile_index_1based - 1, 0, len(ordered_distances) - 1)
        )
        diameter = float(
            int(_matlab_round(float(ordered_distances[percentile_index]))) * 2 + 1
        )
    else:
        diameter = float(_matlab_round(expected))

    return LegacyRadialGeometry(
        center_zyx_px=(float(z), float(recentered_y), float(recentered_x)),
        diameter_xy_px=diameter,
        valid_ray_count=coverage,
        ray_endpoints_xy_px=tuple(
            (float(x_value), float(y_value))
            for x_value, y_value in zip(image_x, image_y, strict=True)
        ),
        peak_response=maximum,
    )


def _regional_maxima_2d(plane_yx: np.ndarray) -> np.ndarray:
    """Return MATLAB-``imregionalmax``-compatible 8-connected plateaus.

    Comparing a pixel with only its immediate neighbours is insufficient for
    a regional maximum.  A constant-valued shelf can extend for many pixels
    before touching a higher value; MATLAB rejects that *entire* connected
    plateau.  This implementation starts with the fast local-maximum mask and
    then removes every candidate component connected to an equal-valued,
    non-candidate shelf pixel.  The work is vectorized over each of the eight
    neighbour directions, so its cost remains linear in the plane size.

    Non-finite pixels are not candidates.  Production detector responses are
    finite already, but keeping that rule here makes the stage deterministic
    for diagnostic callers as well.
    """

    plane = np.asarray(plane_yx)
    if plane.ndim != 2 or not np.issubdtype(plane.dtype, np.number):
        raise ValueError("plane_yx must be a numeric 2-D array")
    finite = np.isfinite(plane)
    if not np.any(finite):
        return np.zeros(plane.shape, dtype=bool)
    working = plane if np.all(finite) else np.where(finite, plane, -np.inf)
    local = finite & (
        working == ndimage.maximum_filter(working, size=3, mode="nearest")
    )
    labels, count = ndimage.label(
        local,
        structure=np.ones((3, 3), dtype=bool),
    )
    if count == 0:
        return local

    rejected = np.zeros(count + 1, dtype=bool)

    def aligned_slices(offset: int, size: int) -> tuple[slice, slice]:
        if offset < 0:
            return slice(-offset, size), slice(0, size + offset)
        if offset > 0:
            return slice(0, size - offset), slice(offset, size)
        return slice(0, size), slice(0, size)

    height, width = plane.shape
    for y_offset in (-1, 0, 1):
        source_y, neighbor_y = aligned_slices(y_offset, height)
        for x_offset in (-1, 0, 1):
            if y_offset == 0 and x_offset == 0:
                continue
            source_x, neighbor_x = aligned_slices(x_offset, width)
            source = (source_y, source_x)
            neighbor = (neighbor_y, neighbor_x)
            touches_rejected_shelf = (
                local[source]
                & ~local[neighbor]
                & finite[neighbor]
                & (plane[source] == plane[neighbor])
            )
            rejected[labels[source][touches_rejected_shelf]] = True
    return local & ~rejected[labels]


def _legacy_slice_maxima(
    response: np.ndarray,
    threshold: float,
) -> tuple[tuple[int, int, int], ...]:
    """Return thresholded per-slice maxima in MATLAB ``find`` order.

    Z planes remain contiguous, and pixels within each plane are ordered with
    the first image dimension varying fastest.  That is the order produced by
    ``find(imregionalmax(...))`` and retained by ``createDiskSet.m``.
    """

    maxima: list[tuple[int, int, int]] = []
    for z in range(response.shape[0]):
        plane = response[z]
        # createDiskSet applies a strict greater-than threshold after
        # imregionalmax.  Every pixel in a true flat regional maximum remains
        # represented, matching MATLAB's plateau convention.
        mask = _regional_maxima_2d(plane) & (plane > threshold)
        maxima.extend(
            (z, int(y), int(x)) for x, y in np.argwhere(mask.T)
        )
    return tuple(maxima)


def _legacy_center_indices(
    response: np.ndarray,
    slice_maxima: tuple[tuple[int, int, int], ...],
) -> tuple[int, ...]:
    """Reproduce pickCenterIndicies' adjacent-plane 18-neighbor test."""

    shape = response.shape
    centers: list[int] = []
    for index, (z, y, x) in enumerate(slice_maxima):
        neighboring_values = []
        for z_offset in (-1, 1):
            sample_z = int(np.clip(z + z_offset, 0, shape[0] - 1))
            for y_offset in (-1, 0, 1):
                sample_y = int(np.clip(y + y_offset, 0, shape[1] - 1))
                for x_offset in (-1, 0, 1):
                    sample_x = int(np.clip(x + x_offset, 0, shape[2] - 1))
                    neighboring_values.append(
                        float(response[sample_z, sample_y, sample_x])
                    )
        if max(neighboring_values) <= float(response[z, y, x]):
            centers.append(index)
    return tuple(centers)


def _legacy_plane_claims(
    center: LegacyRadialGeometry,
    slice_geometry: tuple[LegacyRadialGeometry, ...],
    *,
    anisotropy: float,
    z_size: int,
) -> tuple[int, ...]:
    """Assign the contiguous cylinder of slice disks used by assign_planes.m."""

    center_z, center_y, center_x = center.center_zyx_px
    slice_centers = np.asarray(
        [item.center_zyx_px for item in slice_geometry], dtype=np.float64
    )
    xy_distance = np.sqrt(
        (slice_centers[:, 1] - center_y) ** 2
        + (slice_centers[:, 2] - center_x) ** 2
    )
    radius = center.diameter_xy_px / 2.0

    def on_plane(offset: int) -> tuple[int, ...]:
        plane = int(center_z + offset)
        if not 0 <= plane < z_size:
            return ()
        if abs(offset * anisotropy / radius) >= 3.0:
            return ()
        return tuple(
            int(index)
            for index in np.flatnonzero(
                (slice_centers[:, 0] == plane) & (xy_distance < radius)
            )
        )

    upper: list[int] = []
    offset = -1
    while True:
        current = on_plane(offset)
        if not current:
            break
        upper.extend(current)
        offset -= 1
    lower: list[int] = []
    offset = 1
    while True:
        current = on_plane(offset)
        if not current:
            break
        lower.extend(current)
        offset += 1
    return tuple((*upper, *on_plane(0), *lower))


def _connected_components(
    count: int,
    edges: set[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    parents = list(range(count))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in edges:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parents[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(root(index), []).append(index)
    return tuple(tuple(group) for group in groups.values())


def _xy_principal_variances(
    slices: tuple[LegacyRadialGeometry, ...],
    boundary_percent: float,
) -> tuple[float, float]:
    """Return MATLAB ``pca(..., 1:2)``'s two XY latent values.

    ``calculateCellTripleVector.m`` first keeps claimed slice disks whose DoG
    maximum is at least ``boundary_percent`` of the nucleus maximum, then
    concatenates every 16-ray boundary point.  Only the leading XY covariance
    eigenvalue ratios and the leading daughter values survive into the
    22-value classifier block, so retaining both scalars avoids carrying
    polygon masks through the tracking API.
    """

    if not slices:
        return 0.0, 0.0
    maximum = max(item.peak_response for item in slices)
    selected = tuple(
        item
        for item in slices
        if item.peak_response >= maximum * boundary_percent
    )
    positions = np.asarray(
        [point for item in selected for point in item.ray_endpoints_xy_px],
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 2:
        return 0.0, 0.0
    if len(selected) == 1:
        # calculateCellTripleVector appends a Z-shifted copy for one-slice
        # nuclei before PCA. Its XY coordinates are identical, but repeating
        # them changes MATLAB's N-1 covariance denominator.
        positions = np.concatenate((positions, positions), axis=0)
    centered = positions - np.mean(positions, axis=0)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return (
        float(max(0.0, eigenvalues[-1])),
        float(max(0.0, eigenvalues[-2])),
    )


def legacy_resolve_candidates(
    response_zyx: np.ndarray,
    threshold: float,
    expected_diameter_xy_px: float,
    boundary_percent: float,
    anisotropy: float,
    *,
    large_ray_threshold: float = 1.5,
    small_ray_threshold: float = 1.0 / 3.0,
    normalized_merge_distance: float = 0.8,
    aspect_ratio_merge_threshold: float = 1.6,
    minimum_separation_footprint: np.ndarray | None = None,
) -> tuple[LegacyResolvedCandidate, ...]:
    """Resolve slice maxima through StarryNite's geometric conflict stages.

    The legacy log-odds score predicates remain model-backed, but the two
    production geometric merge predicates are exact here: candidates must
    first claim at least one common slice disk, then either normalized center
    distance or union aspect ratio may join their conflict component.  This
    order is essential; applying a distance rule to every peak over-merges
    normally separated daughters.
    """

    response = np.asarray(response_zyx)
    slice_maxima = _legacy_slice_maxima(response, float(threshold))
    if not slice_maxima:
        return ()
    slice_geometry = tuple(
        legacy_radial_geometry(
            response,
            peak,
            expected_diameter_xy_px,
            boundary_percent,
            large_ray_threshold=large_ray_threshold,
            small_ray_threshold=small_ray_threshold,
        )
        for peak in slice_maxima
    )
    if minimum_separation_footprint is None:
        center_indices = _legacy_center_indices(response, slice_maxima)
    else:
        selected_peaks = set(
            _canonical_maxima(
                response,
                float(threshold),
                minimum_separation_footprint,
            )
        )
        center_indices = tuple(
            index for index, peak in enumerate(slice_maxima) if peak in selected_peaks
        )
    if not center_indices:
        return ()

    candidate_geometry = tuple(slice_geometry[index] for index in center_indices)
    claims = tuple(
        _legacy_plane_claims(
            geometry,
            slice_geometry,
            anisotropy=float(anisotropy),
            z_size=response.shape[0],
        )
        for geometry in candidate_geometry
    )
    claimants: dict[int, list[int]] = {}
    for candidate_index, claimed in enumerate(claims):
        for slice_index in claimed:
            claimants.setdefault(slice_index, []).append(candidate_index)
    overlapping_pairs = {
        (left, right)
        for candidates in claimants.values()
        for position, left in enumerate(candidates)
        for right in candidates[position + 1 :]
        if left != right
    }
    merge_edges: set[tuple[int, int]] = set()
    for left, right in overlapping_pairs:
        left_center = np.asarray(candidate_geometry[left].center_zyx_px)
        right_center = np.asarray(candidate_geometry[right].center_zyx_px)
        delta = left_center - right_center
        physical_pixel_distance = math.sqrt(
            float(delta[1] ** 2 + delta[2] ** 2 + (delta[0] * anisotropy) ** 2)
        )
        normalized_distance = physical_pixel_distance / expected_diameter_xy_px
        combined_claims = (*claims[left], *claims[right])
        claimed_z = [slice_geometry[index].center_zyx_px[0] for index in combined_claims]
        height = max(claimed_z) - min(claimed_z) + 1.0
        merged_aspect_ratio = (
            height
            * anisotropy
            / max(
                candidate_geometry[left].diameter_xy_px,
                candidate_geometry[right].diameter_xy_px,
            )
        )
        if (
            normalized_distance < normalized_merge_distance
            or merged_aspect_ratio < aspect_ratio_merge_threshold
        ):
            merge_edges.add((left, right))

    resolved: list[LegacyResolvedCandidate] = []
    for group in _connected_components(len(candidate_geometry), merge_edges):
        representative = max(
            group,
            key=lambda index: float(response[slice_maxima[center_indices[index]]]),
        )
        if len(group) == 1:
            geometry = candidate_geometry[group[0]]
            center = geometry.center_zyx_px
            claimed = claims[group[0]]
        else:
            claimed = tuple(
                slice_index
                for candidate_index in group
                for slice_index in claims[candidate_index]
            )
            points = np.asarray(
                [slice_geometry[index].center_zyx_px for index in claimed],
                dtype=np.float64,
            )
            center = tuple(float(value) for value in np.mean(points, axis=0))
        claimed_geometry = tuple(slice_geometry[index] for index in claimed)
        diameter = max(
            candidate_geometry[index].diameter_xy_px for index in group
        )
        claimed_z = [item.center_zyx_px[0] for item in claimed_geometry]
        aspect_ratio = (
            (max(claimed_z) - min(claimed_z) + 1.0) * anisotropy / diameter
            if claimed_z
            else 1.0
        )
        xy_principal_variance, xy_secondary_variance = (
            _xy_principal_variances(
                claimed_geometry,
                float(boundary_percent),
            )
        )
        resolved.append(
            LegacyResolvedCandidate(
                center_zyx_px=tuple(float(value) for value in center),
                diameter_xy_px=diameter,
                representative_peak_zyx=slice_maxima[center_indices[representative]],
                valid_ray_count=max(
                    candidate_geometry[index].valid_ray_count for index in group
                ),
                claimed_slice_count=len(claimed_geometry),
                merged_candidate_count=len(group),
                claimed_slices=claimed_geometry,
                aspect_ratio=float(aspect_ratio),
                xy_principal_variance=xy_principal_variance,
                xy_secondary_variance=xy_secondary_variance,
            )
        )
    resolved.sort(
        key=lambda item: (
            item.center_zyx_px,
            item.representative_peak_zyx,
        )
    )
    return tuple(resolved)


def _local_support(
    image: np.ndarray,
    peak: tuple[int, int, int],
    expected_radius_um: float,
    boundary_percent: float,
    calibration: Calibration,
    *,
    measurement_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    spacing = np.asarray(calibration.spacing_zyx, dtype=float)
    half_width = np.maximum(1, np.ceil(1.5 * expected_radius_um / spacing).astype(int))
    start = np.maximum(0, np.asarray(peak, dtype=int) - half_width)
    stop = np.minimum(np.asarray(image.shape), np.asarray(peak, dtype=int) + half_width + 1)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    local = image[slices]
    coordinates = np.indices(local.shape, dtype=float)
    for axis in range(3):
        coordinates[axis] += start[axis]
    distance_sq = np.zeros(local.shape, dtype=float)
    for axis in range(3):
        distance_sq += ((coordinates[axis] - peak[axis]) * spacing[axis]) ** 2
    sphere = distance_sq <= (1.5 * expected_radius_um) ** 2
    values = local[sphere]
    if values.size == 0:
        center = np.asarray(peak, dtype=float)
        return (
            center,
            np.zeros(local.shape, dtype=bool),
            expected_radius_um,
            0.0,
            0.0,
            0.0,
        )
    baseline = float(np.percentile(values, 20.0))
    peak_value = float(image[peak])
    boundary = baseline + boundary_percent * max(0.0, peak_value - baseline)
    support = sphere & (local >= boundary)
    if not np.any(support):
        support[tuple(np.asarray(peak) - start)] = True
    weights = np.where(support, np.maximum(local - baseline, 0.0), 0.0)
    total = float(np.sum(weights))
    if total > np.finfo(float).eps:
        center = np.asarray(
            [float(np.sum(coordinates[axis] * weights) / total) for axis in range(3)]
        )
    else:
        center = np.asarray(peak, dtype=float)
    physical_volume = float(np.count_nonzero(support) * np.prod(spacing))
    radius = (3.0 * physical_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    radius = float(np.clip(radius, 0.5 * expected_radius_um, 1.5 * expected_radius_um))
    measurement_local = (
        local if measurement_image is None else np.asarray(measurement_image)[slices]
    )
    raw_total = float(np.sum(measurement_local[support], dtype=np.float64))
    mean_intensity = float(np.mean(measurement_local[support]))
    return center, support, radius, total, raw_total, mean_intensity


def _points_in_polygon(
    x: np.ndarray,
    y: np.ndarray,
    vertices_xy: np.ndarray,
) -> np.ndarray:
    """Vectorized MATLAB-``inpolygon`` equivalent including the boundary."""

    inside = np.zeros(x.shape, dtype=bool)
    boundary = np.zeros(x.shape, dtype=bool)
    tolerance = 1e-10
    previous_x, previous_y = vertices_xy[-1]
    for current_x, current_y in vertices_xy:
        delta_x = current_x - previous_x
        delta_y = current_y - previous_y
        cross = (x - previous_x) * delta_y - (y - previous_y) * delta_x
        on_segment = (
            (np.abs(cross) <= tolerance)
            & (x >= min(previous_x, current_x) - tolerance)
            & (x <= max(previous_x, current_x) + tolerance)
            & (y >= min(previous_y, current_y) - tolerance)
            & (y <= max(previous_y, current_y) + tolerance)
        )
        boundary |= on_segment
        crosses = (previous_y > y) != (current_y > y)
        denominator = current_y - previous_y
        intersection_x = np.full(x.shape, np.inf, dtype=np.float64)
        np.divide(
            (current_x - previous_x) * (y - previous_y),
            denominator,
            out=intersection_x,
            where=crosses,
        )
        intersection_x += previous_x
        inside ^= crosses & (x < intersection_x)
        previous_x, previous_y = current_x, current_y
    return inside | boundary


def _legacy_polygon_indices(
    geometry: LegacyRadialGeometry,
    shape_yx: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    endpoints = np.asarray(geometry.ray_endpoints_xy_px, dtype=np.float64)
    if endpoints.shape != (16, 2):
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    order = np.argsort(_LEGACY_RAY_CLOCKWISE_RANK, kind="stable")
    vertices = endpoints[order]
    x_values = np.arange(
        max(0, math.floor(float(np.min(vertices[:, 0])))),
        min(shape_yx[1] - 1, math.ceil(float(np.max(vertices[:, 0])))) + 1,
        dtype=np.float64,
    )
    y_values = np.arange(
        max(0, math.floor(float(np.min(vertices[:, 1])))),
        min(shape_yx[0] - 1, math.ceil(float(np.max(vertices[:, 1])))) + 1,
        dtype=np.float64,
    )
    if not len(x_values) or not len(y_values):
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    included = _points_in_polygon(grid_x, grid_y, vertices)
    return (
        grid_y[included].astype(np.intp),
        grid_x[included].astype(np.intp),
    )


def _legacy_integrated_intensity(
    image_zyx: np.ndarray,
    candidate: LegacyResolvedCandidate,
    boundary_percent: float,
) -> tuple[float, int, float]:
    """Port ``calculateSliceGFP`` plus ``integrateGFP`` for one nucleus."""

    claimed = candidate.claimed_slices
    if not claimed:
        return 0.0, 0, 0.0
    maximum = max(item.peak_response for item in claimed)
    selected = tuple(
        item
        for item in claimed
        if item.peak_response >= maximum * boundary_percent
    )
    total = 0.0
    area = 0
    for geometry in selected:
        z = int(round(geometry.center_zyx_px[0]))
        y_indices, x_indices = _legacy_polygon_indices(
            geometry,
            image_zyx.shape[1:],
        )
        area += len(x_indices)
        if len(x_indices):
            total += float(
                np.sum(image_zyx[z, y_indices, x_indices], dtype=np.float64)
            )
    average = total / area if area else 0.0
    return total, area, float(average)


class StarryNiteDetector:
    """Anisotropic DoG/maxima detector with mapped StarryNite settings."""

    plugin_id = "acetree.starrynite_detector"
    display_name = "StarryNite nucleus detector"
    default_settings = _DEFAULT_SETTINGS

    def __init__(self) -> None:
        self._exact_previous_frame: int | None = None
        self._exact_previous_final_count = 0
        self._exact_previous_candidate_diameters: tuple[float, ...] = ()
        self._exact_source_token: tuple[object, ...] | None = None
        self._exact_parameter_cache_key: tuple[str, str] | None = None
        self._exact_parameter_cache_value: tuple[Any, Any] | None = None
        self._exact_distribution_cache_key: tuple[str, str] | None = None
        self._exact_distribution_cache_value: Any | None = None

    def _reset_exact_movie_state(self) -> None:
        self._exact_previous_frame = None
        self._exact_previous_final_count = 0
        self._exact_previous_candidate_diameters = ()
        self._exact_source_token = None
        self._exact_parameter_cache_key = None
        self._exact_parameter_cache_value = None
        self._exact_distribution_cache_key = None
        self._exact_distribution_cache_value = None

    def detect(
        self,
        stack_zyx: np.ndarray,
        frame: int,
        calibration: Calibration,
        settings: Mapping[str, Any],
        *,
        offset_zyx: tuple[float, float, float] = (0, 0, 0),
    ) -> tuple[Detection, ...]:
        if frame < 1:
            raise ValueError("frame must be positive and 1-based")
        if len(offset_zyx) != 3 or not all(math.isfinite(float(v)) for v in offset_zyx):
            raise ValueError("offset_zyx must contain three finite values")
        values = _settings(settings)
        exact_requested = bool(values["STARRYNITE_DISTRIBUTION_FILE"])
        parameter_resolver = None
        parsed_parameters = None
        resolved_parameter_path = ""
        resolved_parameter_sha256 = ""
        parameter_path: Path | None = None
        if values["STARRYNITE_PARAMETER_FILE"]:
            parameter_path = Path(values["STARRYNITE_PARAMETER_FILE"]).resolve(
                strict=False
            )
            if not parameter_path.is_file():
                raise FileNotFoundError(
                    f"StarryNite parameter file was not found: {parameter_path}"
                )
            current_hash = sha256_file(parameter_path)
            if values["STARRYNITE_PARAMETER_SHA256"]:
                if current_hash.lower() != values["STARRYNITE_PARAMETER_SHA256"].lower():
                    raise ValueError(
                        "StarryNite parameter file changed after the detector settings "
                        "were resolved"
                    )
            resolved_parameter_path = str(parameter_path)
            resolved_parameter_sha256 = current_hash

        resolved_distribution_path = ""
        resolved_distribution_sha256 = ""
        distribution_path: Path | None = None
        exact_source_token: tuple[object, ...] | None = None
        continuing = False
        if exact_requested:
            distribution_path = Path(
                values["STARRYNITE_DISTRIBUTION_FILE"]
            ).resolve(strict=False)
            if not distribution_path.is_file():
                raise FileNotFoundError(
                    f"StarryNite disk-distribution file was not found: {distribution_path}"
                )
            resolved_distribution_path = str(distribution_path)
            resolved_distribution_sha256 = sha256_file(distribution_path)
            expected_distribution_sha256 = values[
                "STARRYNITE_DISTRIBUTION_SOURCE_SHA256"
            ]
            if not expected_distribution_sha256:
                raise ValueError(
                    "Exact StarryNite detection requires a request-time "
                    "STARRYNITE_DISTRIBUTION_SOURCE_SHA256 binding"
                )
            if resolved_distribution_sha256 != expected_distribution_sha256:
                raise ValueError(
                    "StarryNite disk-distribution file changed after the detector "
                    "request was created"
                )
            exact_source_token = (
                resolved_parameter_path,
                resolved_parameter_sha256,
                resolved_distribution_path,
                resolved_distribution_sha256,
                calibration.xy_um,
                calibration.z_um,
                values["TARGET_CHANNEL"],
                values["RADIUS"],
                values["STARRYNITE_USE_STATIC_DIAMETER"],
                repr(values["ROI_POINTS_XY"]),
                values["ROI_CROPPED"],
                values["ROI_X_OFFSET"],
                values["ROI_Y_OFFSET"],
                values["ROI_X_MAX"],
                values["ROI_Y_MAX"],
            )
            continuing = (
                self._exact_previous_frame is not None
                and frame == self._exact_previous_frame + 1
                and exact_source_token == self._exact_source_token
            )
            if not continuing:
                # A discontinuity or source change starts a new exact-movie
                # state.  Source-bound caches are deliberately scoped to that
                # state so random access cannot inherit stale mutable inputs.
                self._reset_exact_movie_state()
        else:
            self._reset_exact_movie_state()

        if parameter_path is not None:
            from .parameter_view import (
                build_legacy_region_table,
                resolve_legacy_parameter,
            )
            from .parameters import read_parameter_file

            parameter_cache_key = (
                resolved_parameter_path,
                resolved_parameter_sha256,
            )
            cached_parameters = self._exact_parameter_cache_value
            if (
                exact_requested
                and self._exact_parameter_cache_key == parameter_cache_key
                and cached_parameters is not None
            ):
                parsed_parameters, region_table = cached_parameters
            else:
                parsed_parameters = read_parameter_file(parameter_path)
                region_table = build_legacy_region_table(
                    parsed_parameters, strict=True
                )
                if exact_requested:
                    self._exact_parameter_cache_key = parameter_cache_key
                    self._exact_parameter_cache_value = (
                        parsed_parameters,
                        region_table,
                    )

            def parameter_resolver(
                name: str,
                count: int,
                location: tuple[float, float, float] | None,
            ) -> float:
                resolution = resolve_legacy_parameter(
                    parsed_parameters,
                    name,
                    cell_count=count,
                    location=location,
                    region_table=region_table,
                )
                if isinstance(resolution.value, (bool, str, tuple)):
                    raise ValueError(
                        f"Legacy detector parameter {name} did not resolve to a scalar"
                    )
                return float(resolution.value)

        effective_diameter_xy_px = 2.0 * values["RADIUS"] / calibration.xy_um
        detector_cell_count = values["STARRYNITE_CELL_COUNT"]
        effective_stage_index = values["STARRYNITE_STAGE_INDEX"]
        previous_candidate_diameters: tuple[float, ...] = ()
        if exact_requested:
            from .legacy_detector_tail import legacy_adapt_cell_diameter

            if continuing:
                previous_candidate_diameters = (
                    self._exact_previous_candidate_diameters
                )
                detector_cell_count = self._exact_previous_final_count
                effective_diameter_xy_px = legacy_adapt_cell_diameter(
                    2.0 * values["RADIUS"] / calibration.xy_um,
                    previous_candidate_diameters_xy_px=previous_candidate_diameters,
                    use_static_diameter=values["STARRYNITE_USE_STATIC_DIAMETER"],
                )
            if parsed_parameters is not None:
                from .presets import legacy_stage_index

                effective_stage_index = legacy_stage_index(
                    parsed_parameters, detector_cell_count
                )

        effective_radius_um = effective_diameter_xy_px * calibration.xy_um / 2.0

        image = _select_image(stack_zyx, values["TARGET_CHANNEL"])
        if exact_requested and not np.all(np.isfinite(image)):
            raise ValueError(
                "Exact StarryNite detection requires finite image voxels; NaN and "
                "infinite values are not normalized"
            )
        roi_points = values["ROI_POINTS_XY"]
        roi_enabled = values["ROI_CROPPED"]
        roi_xmin = values["ROI_X_OFFSET"]
        roi_ymin = values["ROI_Y_OFFSET"]
        roi_xmax = values["ROI_X_MAX"]
        roi_ymax = values["ROI_Y_MAX"]
        if exact_requested and parsed_parameters is not None:
            from .parameters import normalize_parameter_name

            normalized = parsed_parameters.normalized_settings

            def parameter_value(name: str, fallback: Any) -> Any:
                # Preserve the spelling used by StarryNite's parameter files.
                # In particular, ``ROIxmin`` normalizes to ``ro_ixmin`` (not
                # ``roixmin``), so guessing normalized spellings silently
                # discarded valid legacy ROI bounds.
                if name in parsed_parameters.settings:
                    return parsed_parameters.settings[name]
                return normalized.get(normalize_parameter_name(name), fallback)

            downsample = parameter_value("downsampling", 1.0)
            if isinstance(downsample, (bool, str, tuple)):
                raise ValueError("Legacy downsampling must be scalar")
            if float(downsample) != 1.0:
                raise ValueError(
                    "Exact StarryNite image downsampling is not implemented; "
                    "downsampling must be 1"
                )
            parameter_roi = parameter_value("ROIpoints", None)
            if not roi_points and isinstance(parameter_roi, tuple):
                roi_points = parameter_roi
            roi_value = parameter_value("ROI", None)
            if roi_value is not None:
                if not isinstance(roi_value, bool):
                    raise ValueError("Legacy ROI must be boolean")
                roi_enabled = roi_value

            def roi_scalar(name: str, fallback: float) -> float:
                value = parameter_value(name, fallback)
                if isinstance(value, (bool, str, tuple)):
                    raise ValueError(f"Legacy {name} must be a scalar")
                result = float(value)
                if not math.isfinite(result):
                    raise ValueError(f"Legacy {name} must be finite")
                return result

            roi_xmin = roi_scalar("ROIxmin", roi_xmin)
            roi_ymin = roi_scalar("ROIymin", roi_ymin)
            roi_xmax = roi_scalar("ROIxmax", roi_xmax)
            roi_ymax = roi_scalar("ROIymax", roi_ymax)

        roi_output_offset = np.zeros(3, dtype=np.float64)
        if exact_requested and roi_enabled:
            if any(float(item) != 0.0 for item in offset_zyx):
                raise ValueError(
                    "Exact legacy ROI cropping requires the full global stack and "
                    "offset_zyx=(0, 0, 0)"
                )
            bounds = (roi_xmin, roi_xmax, roi_ymin, roi_ymax)
            if any(not float(item).is_integer() for item in bounds):
                raise ValueError("Legacy ROI bounds must be integer pixel indices")
            xmin, xmax, ymin, ymax = (int(item) for item in bounds)
            if xmin < 1 or ymin < 1 or xmax < xmin or ymax < ymin:
                raise ValueError(
                    "Legacy ROI bounds must be positive, one-based inclusive ranges"
                )
            if xmax > image.shape[2] or ymax > image.shape[1]:
                raise ValueError(
                    "Legacy ROI bounds lie outside the supplied full image"
                )
            image = image[:, ymin - 1 : ymax, xmin - 1 : xmax]
            roi_output_offset = np.asarray((0.0, ymin - 1.0, xmin - 1.0))
        if any(size == 0 for size in image.shape):
            return ()
        if not exact_requested and not np.all(np.isfinite(image)):
            image[~np.isfinite(image)] = 0.0
        measurement_image = image.copy()
        if values["DARK_NUCLEI"]:
            image = float(np.max(image)) - image
        if values["DO_MEDIAN_FILTERING"]:
            image = ndimage.median_filter(image, size=(1, 3, 3), mode="nearest")

        sigma_factor = values["SIGMA"]
        if parameter_resolver is not None:
            sigma_factor = parameter_resolver("sigma", detector_cell_count, None)
        response, inner, filter_parameters = _legacy_dog_components(
            image,
            effective_radius_um if exact_requested else values["RADIUS"],
            sigma_factor,
            calibration,
        )
        threshold = values["THRESHOLD"]
        if threshold <= 0:
            threshold = (
                values["INTENSITY_THRESHOLD"]
                if parameter_resolver is None
                else parameter_resolver(
                    "intensitythreshold", detector_cell_count, None
                )
            )

        # A positive MIN_SEPARATION is an explicit native override.  Zero runs
        # the legacy slice-maxima, adjacent-plane, disk-claim, and geometric
        # conflict stages without an implicit nucleus-radius suppression.
        separation_footprint = (
            _footprint(values["MIN_SEPARATION"], calibration)
            if values["MIN_SEPARATION"] > 0
            else None
        )
        exact_tail = None
        if values["STARRYNITE_DISTRIBUTION_FILE"]:
            if values["DO_SUBPIXEL_LOCALIZATION"]:
                raise ValueError(
                    "DO_SUBPIXEL_LOCALIZATION is a native-only override and cannot "
                    "be used with the exact StarryNite detector tail"
                )
            if values["MIN_LOCAL_CONTRAST"] > 0:
                raise ValueError(
                    "MIN_LOCAL_CONTRAST is a native-only override and cannot be used "
                    "with the exact StarryNite detector tail"
                )
            if values["DO_MEDIAN_FILTERING"] or values["DARK_NUCLEI"]:
                raise ValueError(
                    "Native median/dark-nucleus preprocessing cannot be used with "
                    "the exact StarryNite detector tail"
                )
            if separation_footprint is not None:
                raise ValueError(
                    "MIN_SEPARATION is a native-only override and cannot be used "
                    "with the exact StarryNite disk-distribution tail"
                )
            from .legacy_detector_tail import (
                legacy_resolve_candidates_exact,
                load_legacy_disk_distributions,
            )

            distribution_cache_key = (
                resolved_distribution_path,
                resolved_distribution_sha256,
            )
            if (
                self._exact_distribution_cache_key == distribution_cache_key
                and self._exact_distribution_cache_value is not None
            ):
                distribution_model = self._exact_distribution_cache_value
            else:
                distribution_model = load_legacy_disk_distributions(
                    resolved_distribution_path
                )
                if (
                    distribution_model.source_sha256
                    != resolved_distribution_sha256
                ):
                    raise ValueError(
                        "StarryNite disk-distribution file changed while it was "
                        "being loaded"
                    )
                self._exact_distribution_cache_key = distribution_cache_key
                self._exact_distribution_cache_value = distribution_model
            exact_tail = legacy_resolve_candidates_exact(
                response,
                threshold,
                expected_diameter_xy_px=effective_diameter_xy_px,
                boundary_percent=values["BOUNDARY_PERCENT"],
                anisotropy=calibration.z_um / calibration.xy_um,
                model=distribution_model,
                numcells=detector_cell_count,
                range_threshold=values["RANGE_THRESHOLD"],
                split_threshold=values["SPLIT_THRESHOLD"],
                merge_lower=values["MERGE_LOWER"],
                merge_split=values["MERGE_SPLIT"],
                normalized_merge_distance=values["NNDIST_MERGE"],
                aspect_ratio_merge_threshold=values["AR_MERGE"],
                large_ray_threshold=values["LARGE_RAY_THRESHOLD"],
                small_ray_threshold=values["SMALL_RAY_THRESHOLD"],
                parameter_resolver=parameter_resolver,
                roi_points_xy=roi_points,
                roi_cropped=roi_enabled,
                # createDiskSet.m adds ROIxmin/ymin (not minus one) before
                # inpolygon; preserve that historical one-pixel convention.
                roi_offset_xy=(roi_xmin, roi_ymin),
            )
            candidates = exact_tail.candidates
        else:
            candidates = legacy_resolve_candidates(
                response,
                threshold,
                expected_diameter_xy_px=(
                    2.0 * values["RADIUS"] / calibration.xy_um
                ),
                boundary_percent=values["BOUNDARY_PERCENT"],
                anisotropy=calibration.z_um / calibration.xy_um,
                large_ray_threshold=values["LARGE_RAY_THRESHOLD"],
                small_ray_threshold=values["SMALL_RAY_THRESHOLD"],
                normalized_merge_distance=values["NNDIST_MERGE"],
                aspect_ratio_merge_threshold=values["AR_MERGE"],
                minimum_separation_footprint=separation_footprint,
            )
        previous_candidate_count = len(previous_candidate_diameters)
        previous_candidate_median = (
            None
            if not previous_candidate_diameters
            else float(np.median(previous_candidate_diameters))
        )
        current_candidate_diameters = (
            ()
            if exact_tail is None
            else exact_tail.candidate_diameters_xy_px
        )
        current_candidate_count = len(current_candidate_diameters)
        current_candidate_median = (
            None
            if not current_candidate_diameters
            else float(np.median(current_candidate_diameters))
        )
        offset = np.asarray(offset_zyx, dtype=float) + roi_output_offset
        spacing = np.asarray(calibration.spacing_zyx, dtype=float)
        detections: list[Detection] = []
        for legacy_candidate_row, candidate in enumerate(candidates):
            peak = candidate.representative_peak_zyx
            local_half_width = np.maximum(
                1,
                np.ceil(
                    (effective_radius_um if exact_tail is not None else values["RADIUS"])
                    / spacing
                ).astype(int),
            )
            start = np.maximum(0, np.asarray(peak) - local_half_width)
            stop = np.minimum(np.asarray(image.shape), np.asarray(peak) + local_half_width + 1)
            local = response[tuple(slice(int(a), int(b)) for a, b in zip(start, stop))]
            intensity_range = float(np.max(local) - np.min(local)) if local.size else 0.0
            if intensity_range < values["MIN_LOCAL_CONTRAST"]:
                continue
            (
                center,
                support,
                support_radius,
                filtered_total,
                raw_total,
                mean_intensity,
            ) = _local_support(
                response,
                peak,
                effective_radius_um if exact_tail is not None else values["RADIUS"],
                values["BOUNDARY_PERCENT"],
                calibration,
                measurement_image=measurement_image,
            )
            if not values["DO_SUBPIXEL_LOCALIZATION"]:
                center = np.asarray(candidate.center_zyx_px, dtype=float)
            radius = (
                candidate.diameter_xy_px * calibration.xy_um / 2.0
            )
            legacy_total_gfp, legacy_area, legacy_average_gfp = (
                _legacy_integrated_intensity(
                    measurement_image,
                    candidate,
                    values["BOUNDARY_PERCENT"],
                )
            )
            full_zyx = center + offset
            physical_zyx = full_zyx * spacing
            coordinate_key = ",".join(f"{float(value):.6f}" for value in full_zyx)
            digest = hashlib.blake2b(
                coordinate_key.encode("ascii"), digest_size=8
            ).hexdigest()
            row_identity = (
                f":r{legacy_candidate_row}" if exact_tail is not None else ""
            )
            detections.append(
                Detection(
                    detection_id=(
                        f"{self.plugin_id}:t{frame}{row_identity}:{digest}"
                    ),
                    frame=frame,
                    x_um=float(physical_zyx[2]),
                    y_um=float(physical_zyx[1]),
                    z_um=float(physical_zyx[0]),
                    radius_um=radius,
                    quality=float(response[peak]),
                    features={
                        "DETECTOR_ID": self.plugin_id,
                        "TARGET_CHANNEL": values["TARGET_CHANNEL"],
                        "VOXEL_Z": float(full_zyx[0]),
                        "VOXEL_Y": float(full_zyx[1]),
                        "VOXEL_X": float(full_zyx[2]),
                        "PEAK_VOXEL_Z": float(peak[0] + offset[0]),
                        "PEAK_VOXEL_Y": float(peak[1] + offset[1]),
                        "PEAK_VOXEL_X": float(peak[2] + offset[2]),
                        "PEAK_INTENSITY": float(inner[peak]),
                        "INTENSITY_RANGE": intensity_range,
                        "TOTAL_INTENSITY": raw_total,
                        "FILTERED_SUPPORT_SIGNAL": filtered_total,
                        # StarryNite's legacy exporter scales integrated GFP by
                        # 256 before storing the integer nuclei-file weight.
                        "ACETREE_WEIGHT": int(
                            np.clip(
                                math.floor(legacy_total_gfp / 256.0 + 0.5),
                                0,
                                65535,
                            )
                        ),
                        "MEAN_INTENSITY": mean_intensity,
                        "SUPPORT_VOXELS": int(np.count_nonzero(support)),
                        "SUPPORT_EQUIVALENT_RADIUS_UM": support_radius,
                        "LEGACY_DIAMETER_XY_PX": candidate.diameter_xy_px,
                        "LEGACY_XY_COVERAGE": candidate.valid_ray_count,
                        "LEGACY_CLAIMED_SLICE_COUNT": candidate.claimed_slice_count,
                        "LEGACY_MERGED_CANDIDATE_COUNT": (
                            candidate.merged_candidate_count
                        ),
                        "LEGACY_TOTAL_GFP": legacy_total_gfp,
                        "LEGACY_AVG_GFP": legacy_average_gfp,
                        "LEGACY_DISK_AREA": legacy_area,
                        "LEGACY_ASPECT_RATIO": candidate.aspect_ratio,
                        # Distribution-backed slice log odds are not yet part
                        # of the native detector.  ``None`` is explicit and
                        # JSON-safe; the exact extractor turns it into a
                        # missing value and supported models mask that cue.
                        "LEGACY_LOG_ODDS_SUM": candidate.log_odds_sum,
                        "LEGACY_CLAIMED_LOG_ODDS": candidate.claimed_log_odds,
                        "LEGACY_CLAIMED_XY_COVERAGES": tuple(
                            item.valid_ray_count for item in candidate.claimed_slices
                        ),
                        "LEGACY_RECOVERY_ROUND": candidate.recovery_round,
                        "LEGACY_EXACT_TAIL": exact_tail is not None,
                        "LEGACY_INITIAL_CENTER_COUNT": (
                            None
                            if exact_tail is None
                            else exact_tail.initial_center_count
                        ),
                        "LEGACY_RECOVERY_ROUND_COUNT": (
                            None
                            if exact_tail is None
                            else exact_tail.recovery_round_count
                        ),
                        "LEGACY_CANDIDATE_DIAMETERS_XY_PX": (
                            None
                            if exact_tail is None
                            else current_candidate_diameters
                        ),
                        "LEGACY_CANDIDATE_DIAMETER_COUNT": (
                            None if exact_tail is None else current_candidate_count
                        ),
                        "LEGACY_CANDIDATE_DIAMETER_MEDIAN_XY_PX": (
                            None if exact_tail is None else current_candidate_median
                        ),
                        # When an earlier frame emitted no final rows, this
                        # compact handoff is the only behaviorally relevant
                        # detector state needed to validate the next frame.
                        "LEGACY_PREVIOUS_CANDIDATE_DIAMETER_COUNT": (
                            None if exact_tail is None else previous_candidate_count
                        ),
                        "LEGACY_PREVIOUS_CANDIDATE_DIAMETER_MEDIAN_XY_PX": (
                            None if exact_tail is None else previous_candidate_median
                        ),
                        "LEGACY_SLICE_COUNT": candidate.claimed_slice_count,
                        "LEGACY_XY_PRINCIPAL_VARIANCE": (
                            candidate.xy_principal_variance
                        ),
                        "LEGACY_XY_SECONDARY_VARIANCE": (
                            candidate.xy_secondary_variance
                        ),
                        "LEGACY_EFFECTIVE_DIAMETER_XY_PX": (
                            effective_diameter_xy_px if exact_tail is not None else None
                        ),
                        "LEGACY_ROI_ENABLED": bool(roi_enabled),
                        "LEGACY_ROI_BOUNDS_XY_1BASED": (
                            (roi_xmin, roi_xmax, roi_ymin, roi_ymax)
                            if exact_tail is not None and roi_enabled
                            else None
                        ),
                        "STARRYNITE_CELL_COUNT": detector_cell_count,
                        "STARRYNITE_STAGE_INDEX": effective_stage_index,
                        "STARRYNITE_USE_STATIC_DIAMETER": values[
                            "STARRYNITE_USE_STATIC_DIAMETER"
                        ],
                        "STARRYNITE_PARAMETER_FILE": resolved_parameter_path,
                        "STARRYNITE_PARAMETER_SHA256": resolved_parameter_sha256,
                        "STARRYNITE_DISTRIBUTION_FILE": (
                            resolved_distribution_path if exact_tail is not None else ""
                        ),
                        "STARRYNITE_DISTRIBUTION_SHA256": (
                            resolved_distribution_sha256 if exact_tail is not None else ""
                        ),
                        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": (
                            values["STARRYNITE_DISTRIBUTION_SOURCE_SHA256"]
                            if exact_tail is not None
                            else ""
                        ),
                        "DOG_INNER_SIGMA_Z_PX": filter_parameters.inner_sigma_zyx[0],
                        "DOG_INNER_SIGMA_XY_PX": filter_parameters.inner_sigma_zyx[1],
                        "DOG_OUTER_SIGMA_Z_PX": filter_parameters.outer_sigma_zyx[0],
                        "DOG_OUTER_SIGMA_XY_PX": filter_parameters.outer_sigma_zyx[1],
                    },
                )
            )
        if exact_tail is None:
            detections.sort(
                key=lambda item: (item.z_um, item.y_um, item.x_um, item.detection_id)
            )
        detections = [
            replace(
                detection,
                features={
                    **dict(detection.features),
                    "LEGACY_ROW_INDEX": row_index,
                },
            )
            for row_index, detection in enumerate(detections)
        ]
        if exact_tail is not None:
            self._exact_previous_frame = frame
            self._exact_previous_final_count = len(detections)
            self._exact_previous_candidate_diameters = (
                exact_tail.candidate_diameters_xy_px
            )
            self._exact_source_token = exact_source_token
        return tuple(detections)


__all__ = [
    "LegacyRadialGeometry",
    "LegacyResolvedCandidate",
    "StarryNiteDetector",
    "StarryNiteDogFilter",
    "legacy_dog_filter_parameters",
    "legacy_dog_response",
    "legacy_radial_geometry",
    "legacy_resolve_candidates",
]
