"""Exact learned tail of StarryNite's legacy nucleus detector.

This module ports the stages between the XY disk geometry produced by
``createDiskSet.m`` and the final nuclei returned by ``processVolume.m``:

* the seven disk-to-disk features and left/right Gaussian log odds;
* maximal contiguous Z ranges;
* iterative recovery of overlooked, non-isolated maxima;
* overlap scoring, split-range repair, and learned/geometric merging; and
* the previous-frame median-diameter update used by the movie driver.

The Gaussian tables are data, not fitted Python substitutes.  They are loaded
from the exact ``clean_distributions*.mat`` selected by the parameter file and
validated before use.  Missing or malformed tables fail closed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import cholesky, solve_triangular


class LegacyDetectorTailError(RuntimeError):
    """Raised when an exact legacy detector-tail operation cannot continue."""


LegacyParameterResolver = Callable[
    [str, int, tuple[float, float, float] | None], float
]


@dataclass(frozen=True, slots=True)
class LegacyDiskDistributionModel:
    """The four good/bad, left/right Gaussian disk distributions."""

    bad_left_mean: np.ndarray
    bad_left_covariance: np.ndarray
    good_left_mean: np.ndarray
    good_left_covariance: np.ndarray
    bad_right_mean: np.ndarray
    bad_right_covariance: np.ndarray
    good_right_mean: np.ndarray
    good_right_covariance: np.ndarray
    source_path: Path | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "bad_left_mean",
            "good_left_mean",
            "bad_right_mean",
            "good_right_mean",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape != (7,) or not np.all(np.isfinite(value)):
                raise LegacyDetectorTailError(f"{name} must contain seven finite values")
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in (
            "bad_left_covariance",
            "good_left_covariance",
            "bad_right_covariance",
            "good_right_covariance",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (7, 7) or not np.all(np.isfinite(value)):
                raise LegacyDetectorTailError(f"{name} must be a finite 7x7 matrix")
            if not np.allclose(value, value.T, rtol=0.0, atol=1e-12):
                raise LegacyDetectorTailError(f"{name} must be symmetric")
            try:
                np.linalg.cholesky(value)
            except np.linalg.LinAlgError as exc:
                raise LegacyDetectorTailError(
                    f"{name} must be positive definite"
                ) from exc
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))


@dataclass(frozen=True, slots=True)
class LegacyDetectorNucleus:
    """One center disk, its assigned cylinder, log odds, and retained range."""

    center_disk_index: int
    disk_indices: tuple[int, ...]
    log_odds: tuple[float, ...]
    retained_indices: tuple[int, ...]
    recovery_round: int = 0


@dataclass(frozen=True, slots=True)
class LegacyDetectorTailResult:
    """Final exact candidates plus diagnostics needed by the movie driver."""

    candidates: tuple[object, ...]
    nuclei: tuple[LegacyDetectorNucleus, ...]
    initial_center_count: int
    recovery_round_count: int
    disk_count: int
    candidate_diameters_xy_px: tuple[float, ...]
    merge_pairs: tuple[tuple[int, int, float, float], ...]


def load_legacy_disk_distributions(
    path: str | Path,
) -> LegacyDiskDistributionModel:
    """Load and strictly validate StarryNite's Gaussian disk tables."""

    source = Path(path).expanduser().resolve(strict=False)
    if not source.is_file():
        raise LegacyDetectorTailError(
            f"StarryNite disk-distribution file was not found: {source}"
        )
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - project dependency guard
        raise LegacyDetectorTailError(
            "scipy.io is required to load StarryNite disk distributions"
        ) from exc
    try:
        payload = loadmat(source, squeeze_me=False, struct_as_record=False)
    except Exception as exc:
        raise LegacyDetectorTailError(
            f"Could not read StarryNite disk distributions from {source}: {exc}"
        ) from exc
    required = {
        "allbadlm",
        "allbadlc",
        "allgoodlm",
        "allgoodlc",
        "allbadrm",
        "allbadrc",
        "allgoodrm",
        "allgoodrc",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise LegacyDetectorTailError(
            "StarryNite disk-distribution file is missing: " + ", ".join(missing)
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return LegacyDiskDistributionModel(
        bad_left_mean=payload["allbadlm"],
        bad_left_covariance=payload["allbadlc"],
        good_left_mean=payload["allgoodlm"],
        good_left_covariance=payload["allgoodlc"],
        bad_right_mean=payload["allbadrm"],
        bad_right_covariance=payload["allbadrc"],
        good_right_mean=payload["allgoodrm"],
        good_right_covariance=payload["allgoodrc"],
        source_path=source,
        source_sha256=digest,
    )


def legacy_adapt_cell_diameter(
    first_timestep_diameter_xy_px: float,
    *,
    downsample: float = 1.0,
    previous_candidate_diameters_xy_px: Sequence[float] | None = None,
    use_static_diameter: bool = False,
) -> float:
    """Match ``processVolume.m``'s per-frame detector diameter selection.

    The legacy driver updates only when the preceding frame has more than ten
    *candidate-disk* diameters.  It uses their median, not final merged nucleus
    diameters.  Otherwise it returns ``firsttimestepdiam * downsampling``.
    """

    first = float(first_timestep_diameter_xy_px)
    scale = float(downsample)
    if not math.isfinite(first) or first <= 0:
        raise ValueError("first_timestep_diameter_xy_px must be positive and finite")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("downsample must be positive and finite")
    previous = np.asarray(
        () if previous_candidate_diameters_xy_px is None else previous_candidate_diameters_xy_px,
        dtype=np.float64,
    ).reshape(-1)
    if previous.size and (not np.all(np.isfinite(previous)) or np.any(previous <= 0)):
        raise ValueError("previous candidate diameters must be positive and finite")
    if not use_static_diameter and previous.size > 10:
        return float(np.median(previous))
    return first * scale


def _matlab_mvnpdf(
    features: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate the full-rank branch of MATLAB ``mvnpdf``."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("features must have shape (N, 7)")
    # MATLAB chol returns upper R and evaluates (X-mu)/R.  NumPy's lower L
    # gives the same triangular solve as L \ (X-mu)'.
    upper = cholesky(covariance, lower=False, check_finite=False)
    solved = solve_triangular(
        upper.T,
        (values - mean).T,
        lower=True,
        check_finite=False,
    )
    quadratic = np.sum(solved * solved, axis=0)
    denominator = (2.0 * math.pi) ** 3.5 * float(np.prod(np.diag(upper)))
    return np.exp(-0.5 * quadratic) / denominator


def legacy_disk_feature_vectors(
    disk_indices: Sequence[int],
    center_disk_index: int,
    geometries: Sequence[object],
    maxima_values: Sequence[float],
    diameters_xy_px: Sequence[float],
    anisotropy: float,
) -> np.ndarray:
    """Port ``calc_disk_feature_vector.m`` in its original column order."""

    indices = tuple(int(item) for item in disk_indices)
    if not indices:
        return np.empty((7, 0), dtype=np.float64)
    center_index = int(center_disk_index)
    center = geometries[center_index].center_zyx_px
    pivot = int(round(center[0]))
    center_intensity = np.float32(maxima_values[center_index])
    center_diameter = float(diameters_xy_px[center_index])
    if center_diameter <= 0:
        raise LegacyDetectorTailError("center disk diameter must be positive")
    if center_intensity == 0:
        raise LegacyDetectorTailError("center disk intensity must be nonzero")
    output = np.zeros((7, len(indices)), dtype=np.float64)
    z_by_index = {
        index: int(round(geometries[index].center_zyx_px[0])) for index in indices
    }
    top = min(min(z_by_index.values()), pivot)
    bottom = max(max(z_by_index.values()), pivot)
    counter = 0

    def xy_distance(first: int, second: int) -> float:
        first_center = geometries[first].center_zyx_px
        second_center = geometries[second].center_zyx_px
        return math.hypot(
            float(first_center[2]) - float(second_center[2]),
            float(first_center[1]) - float(second_center[1]),
        )

    def write(
        disk_index: int,
        neighbor_index: int | None,
        z_distance: float,
        diameter_delta: float,
        intensity_delta: float,
    ) -> None:
        nonlocal counter
        disk_intensity = np.float32(maxima_values[disk_index])
        disk_diameter = float(diameters_xy_px[disk_index])
        output[0, counter] = xy_distance(disk_index, center_index) / center_diameter
        output[1, counter] = disk_diameter / center_diameter
        output[2, counter] = float(
            np.float32(disk_intensity / center_intensity)
        )
        output[3, counter] = z_distance
        output[4, counter] = diameter_delta
        output[5, counter] = intensity_delta
        output[6, counter] = (
            xy_distance(disk_index, center_index if neighbor_index is None else neighbor_index)
            / center_diameter
        )
        counter += 1

    for plane in range(top, bottom):
        left = tuple(index for index in indices if z_by_index[index] == plane)
        right = tuple(index for index in indices if z_by_index[index] == plane + 1)
        if plane == pivot:
            for disk_index in right:
                disk_intensity = np.float32(maxima_values[disk_index])
                write(
                    disk_index,
                    None,
                    (plane + 1 - pivot) * float(anisotropy) / center_diameter,
                    (float(diameters_xy_px[disk_index]) - center_diameter)
                    / center_diameter,
                    float(
                        np.float32(
                            np.float32(disk_intensity - center_intensity)
                            / center_intensity
                        )
                    ),
                )
        elif plane == pivot - 1:
            for disk_index in left:
                disk_intensity = np.float32(maxima_values[disk_index])
                write(
                    disk_index,
                    None,
                    (plane - pivot) * float(anisotropy) / center_diameter,
                    (center_diameter - float(diameters_xy_px[disk_index]))
                    / center_diameter,
                    float(
                        np.float32(
                            np.float32(center_intensity - disk_intensity)
                            / center_intensity
                        )
                    ),
                )
        else:
            if plane < pivot:
                for disk_index in left:
                    neighbor = right[
                        int(np.argmin([xy_distance(disk_index, item) for item in right]))
                    ]
                    disk_intensity = np.float32(maxima_values[disk_index])
                    neighbor_intensity = np.float32(maxima_values[neighbor])
                    write(
                        disk_index,
                        neighbor,
                        (plane - pivot) * float(anisotropy) / center_diameter,
                        (
                            float(diameters_xy_px[neighbor])
                            - float(diameters_xy_px[disk_index])
                        )
                        / center_diameter,
                        float(
                            np.float32(
                                np.float32(neighbor_intensity - disk_intensity)
                                / center_intensity
                            )
                        ),
                    )
            if plane > pivot:
                for disk_index in right:
                    neighbor = left[
                        int(np.argmin([xy_distance(disk_index, item) for item in left]))
                    ]
                    disk_intensity = np.float32(maxima_values[disk_index])
                    neighbor_intensity = np.float32(maxima_values[neighbor])
                    write(
                        disk_index,
                        neighbor,
                        (plane + 1 - pivot) * float(anisotropy) / center_diameter,
                        (
                            float(diameters_xy_px[disk_index])
                            - float(diameters_xy_px[neighbor])
                        )
                        / center_diameter,
                        float(
                            np.float32(
                                np.float32(disk_intensity - neighbor_intensity)
                                / center_intensity
                            )
                        ),
                    )
    return output


def legacy_calculate_disk_log_odds(
    disk_indices: Sequence[int],
    center_disk_index: int,
    geometries: Sequence[object],
    maxima_values: Sequence[float],
    diameters_xy_px: Sequence[float],
    anisotropy: float,
    model: LegacyDiskDistributionModel,
) -> tuple[float, ...]:
    """Port ``calculateLogodds.m`` including its low-coverage zero rule."""

    indices = tuple(int(item) for item in disk_indices)
    features = legacy_disk_feature_vectors(
        indices,
        center_disk_index,
        geometries,
        maxima_values,
        diameters_xy_px,
        anisotropy,
    ).T
    good_left = _matlab_mvnpdf(
        features, model.good_left_mean, model.good_left_covariance
    )
    bad_left = _matlab_mvnpdf(features, model.bad_left_mean, model.bad_left_covariance)
    good_right = _matlab_mvnpdf(
        features, model.good_right_mean, model.good_right_covariance
    )
    bad_right = _matlab_mvnpdf(
        features, model.bad_right_mean, model.bad_right_covariance
    )
    result = np.zeros(len(indices), dtype=np.float64)
    center = geometries[int(center_disk_index)].center_zyx_px
    pivot = int(round(center[0]))
    z_values = tuple(int(round(geometries[index].center_zyx_px[0])) for index in indices)
    top = min(min(z_values), pivot)
    bottom = max(max(z_values), pivot)
    counter = 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for plane in range(top, bottom):
            left_positions = tuple(
                position for position, z_value in enumerate(z_values) if z_value == plane
            )
            right_positions = tuple(
                position
                for position, z_value in enumerate(z_values)
                if z_value == plane + 1
            )
            if plane < pivot:
                for position in left_positions:
                    disk_index = indices[position]
                    if geometries[disk_index].valid_ray_count > 13:
                        result[position] = math.log(good_left[counter] / bad_left[counter])
                    counter += 1
            if plane >= pivot:
                for position in right_positions:
                    disk_index = indices[position]
                    if geometries[disk_index].valid_ray_count > 13:
                        result[position] = math.log(good_right[counter] / bad_right[counter])
                    counter += 1
    return tuple(float(value) for value in result)


def legacy_maximal_disk_range(
    nucleus: LegacyDetectorNucleus,
    geometries: Sequence[object],
    threshold: float,
) -> tuple[int, ...]:
    """Port ``vcalculateMaximalRange.m`` with strict score comparisons."""

    limit = float(threshold)
    if not math.isfinite(limit):
        raise LegacyDetectorTailError("range threshold must be finite")
    center_z = int(round(geometries[nucleus.center_disk_index].center_zyx_px[0]))
    z_values = tuple(
        int(round(geometries[index].center_zyx_px[0]))
        for index in nucleus.disk_indices
    )
    best = [index for index, z_value in enumerate(z_values) if z_value == center_z]
    top = min(min(z_values), center_z)
    bottom = max(max(z_values), center_z)
    active = True
    for plane in range(center_z - 1, top - 1, -1):
        positions = tuple(
            index for index, z_value in enumerate(z_values) if z_value == plane
        )
        positive = tuple(
            index
            for index in positions
            if nucleus.log_odds[index] > -limit
            or geometries[nucleus.disk_indices[index]].valid_ray_count <= 13
        )
        if positive and active:
            best.extend(positive)
        else:
            active = False
    active = True
    for plane in range(center_z + 1, bottom + 1):
        positions = tuple(
            index for index, z_value in enumerate(z_values) if z_value == plane
        )
        positive = tuple(
            index
            for index in positions
            if nucleus.log_odds[index] > -limit
            or geometries[nucleus.disk_indices[index]].valid_ray_count <= 13
        )
        if positive and active:
            best.extend(positive)
        else:
            active = False
    return tuple(best)


def _resolve(
    resolver: LegacyParameterResolver | None,
    name: str,
    count: int,
    location: tuple[float, float, float] | None,
    fallback: float,
) -> float:
    value = fallback if resolver is None else resolver(name, count, location)
    result = float(value)
    if not math.isfinite(result):
        raise LegacyDetectorTailError(f"Legacy detector parameter {name} is not finite")
    return result


def _split_score(
    first: int,
    second: int,
    nuclei: Sequence[LegacyDetectorNucleus],
    geometries: Sequence[object],
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    n1 = nuclei[first]
    n2 = nuclei[second]
    z1 = int(round(geometries[n1.center_disk_index].center_zyx_px[0]))
    z2 = int(round(geometries[n2.center_disk_index].center_zyx_px[0]))
    retained1 = set(n1.retained_indices)
    retained2 = set(n2.retained_indices)
    disk_to_second = {
        disk: position
        for position, disk in enumerate(n2.disk_indices)
        if position in retained2
    }
    disk_to_first = {
        disk: position
        for position, disk in enumerate(n1.disk_indices)
        if position in retained1
    }

    def z(nucleus: LegacyDetectorNucleus, position: int) -> int:
        return int(round(geometries[nucleus.disk_indices[position]].center_zyx_px[0]))

    if z1 == z2:
        range1 = tuple(position for position in retained1 if z(n1, position) >= z2)
        range2 = tuple(position for position in retained2 if z(n2, position) <= z2)
        score = sum(n1.log_odds[position] for position in retained1 if z(n1, position) > z2)
        score += sum(n2.log_odds[position] for position in retained2 if z(n2, position) < z2)
        other1 = tuple(position for position in retained1 if z(n1, position) <= z2)
        other2 = tuple(position for position in retained2 if z(n2, position) >= z2)
        other_score = sum(
            n2.log_odds[position] for position in retained2 if z(n2, position) > z2
        )
        other_score += sum(
            n1.log_odds[position] for position in retained1 if z(n1, position) < z2
        )
        if other_score > score:
            range1, range2 = other1, other2
        return float(max(score, other_score)), tuple(sorted(range1)), tuple(sorted(range2))
    if z1 > z2:
        score, range2, range1 = _split_score(second, first, nuclei, geometries)
        return score, range1, range2

    # This deliberately retains the two upstream ``sum(find(...))`` terms;
    # they sum one-based row indices rather than disk log odds.
    range1 = [position for position in retained1 if z(n1, position) <= z1]
    range2 = [position for position in retained2 if z(n2, position) >= z2]
    score = float(sum(position + 1 for position in range1))
    score += float(sum(position + 1 for position in range2))
    uncertain1 = tuple(
        position for position in retained1 if z1 < z(n1, position) < z2
    )
    for position in uncertain1:
        other = disk_to_second.get(n1.disk_indices[position])
        if other is None or n2.log_odds[other] <= n1.log_odds[position]:
            score += n1.log_odds[position]
            range1.append(position)
    uncertain2 = tuple(
        position for position in retained2 if z1 < z(n2, position) < z2
    )
    for position in uncertain2:
        other = disk_to_first.get(n2.disk_indices[position])
        if other is None or n1.log_odds[other] < n2.log_odds[position]:
            score += n2.log_odds[position]
            range2.append(position)
    return float(score), tuple(sorted(range1)), tuple(sorted(range2))


def _merge_score(
    first: int,
    second: int,
    nuclei: Sequence[LegacyDetectorNucleus],
    geometries: Sequence[object],
    maxima_values: Sequence[float],
    diameters_xy_px: Sequence[float],
    anisotropy: float,
    model: LegacyDiskDistributionModel,
) -> float:
    merged = [
        nuclei[first].disk_indices[position]
        for position in nuclei[first].retained_indices
    ] + [
        nuclei[second].disk_indices[position]
        for position in nuclei[second].retained_indices
    ]
    # MATLAB unique sorts disk IDs and returns the first occurrence.
    unique_disks = tuple(sorted(set(merged)))
    first_z = geometries[nuclei[first].center_disk_index].center_zyx_px[0]
    second_z = geometries[nuclei[second].center_disk_index].center_zyx_px[0]
    center_plane = int(math.floor(abs((first_z + second_z) / 2.0) + 0.5))
    center_disk = next(
        (
            disk
            for disk in unique_disks
            if int(round(geometries[disk].center_zyx_px[0])) == center_plane
        ),
        None,
    )
    if center_disk is None:
        raise LegacyDetectorTailError("Merged disk union has no center-plane disk")
    log_odds = legacy_calculate_disk_log_odds(
        unique_disks,
        center_disk,
        geometries,
        maxima_values,
        diameters_xy_px,
        anisotropy,
        model,
    )
    return float(np.sum(np.asarray(log_odds, dtype=np.float64)))


def _merge_groups(pairs: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    remaining = list(pairs)
    groups: list[tuple[int, ...]] = []
    while remaining:
        first = remaining.pop(0)
        connected = {first[0], first[1]}
        changed = True
        while changed:
            changed = False
            keep: list[tuple[int, int]] = []
            for pair in remaining:
                if pair[0] in connected or pair[1] in connected:
                    connected.update(pair)
                    changed = True
                else:
                    keep.append(pair)
            remaining = keep
        groups.append(tuple(sorted(connected)))
    return tuple(groups)


def legacy_resolve_candidates_exact(
    response_zyx: np.ndarray,
    threshold: float,
    expected_diameter_xy_px: float,
    boundary_percent: float,
    anisotropy: float,
    model: LegacyDiskDistributionModel,
    *,
    numcells: int,
    range_threshold: float = 1.0,
    split_threshold: float = 100.0,
    merge_lower: float = -300.0,
    merge_split: float = 1.0,
    normalized_merge_distance: float = 0.8,
    aspect_ratio_merge_threshold: float = 1.6,
    large_ray_threshold: float = 1.5,
    small_ray_threshold: float = 1.0 / 3.0,
    parameter_resolver: LegacyParameterResolver | None = None,
    roi_points_xy: Sequence[Sequence[float]] | None = None,
    roi_cropped: bool = False,
    roi_offset_xy: tuple[float, float] = (0.0, 0.0),
) -> LegacyDetectorTailResult:
    """Run the learned/recovery/conflict detector tail used by StarryNite."""

    # Lazy import avoids a module cycle while keeping the existing public
    # detector geometry types stable.
    from .detector import (
        LegacyResolvedCandidate,
        _legacy_center_indices,
        _legacy_plane_claims,
        _legacy_slice_maxima,
        _points_in_polygon,
        _xy_principal_variances,
        legacy_radial_geometry,
    )

    response = np.asarray(response_zyx)
    if response.ndim != 3 or not np.issubdtype(response.dtype, np.number):
        raise ValueError("response_zyx must be a numeric ZYX volume")
    if not np.all(np.isfinite(response)):
        raise LegacyDetectorTailError("exact legacy detector response must be finite")
    diameter = float(expected_diameter_xy_px)
    z_scale = float(anisotropy)
    if diameter <= 0 or not math.isfinite(diameter):
        raise ValueError("expected_diameter_xy_px must be positive and finite")
    if z_scale <= 0 or not math.isfinite(z_scale):
        raise ValueError("anisotropy must be positive and finite")
    raw_peaks = _legacy_slice_maxima(response, float(threshold))
    if roi_points_xy:
        vertices = np.asarray(roi_points_xy, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise LegacyDetectorTailError("ROIpoints must be an Nx2 polygon with N >= 3")
        if not np.all(np.isfinite(vertices)):
            raise LegacyDetectorTailError("ROIpoints must be finite")
        x = np.asarray([peak[2] + 1.0 for peak in raw_peaks])
        y = np.asarray([peak[1] + 1.0 for peak in raw_peaks])
        if roi_cropped:
            x += float(roi_offset_xy[0])
            y += float(roi_offset_xy[1])
        mask = _points_in_polygon(x, y, vertices)
        raw_peaks = tuple(peak for peak, keep in zip(raw_peaks, mask, strict=True) if keep)
    if not raw_peaks:
        return LegacyDetectorTailResult((), (), 0, 0, 0, (), ())

    # calculateSphereDiameters_geometric resolves these three values using
    # the number of 2-D maxima, not the movie's previous-frame cell count.
    geometry_count = len(raw_peaks)
    boundary_percent = _resolve(
        parameter_resolver,
        "boundary_percent",
        geometry_count,
        None,
        boundary_percent,
    )
    large_ray_threshold = _resolve(
        parameter_resolver,
        "large_ray_threshold",
        geometry_count,
        None,
        large_ray_threshold,
    )
    small_ray_threshold = _resolve(
        parameter_resolver,
        "small_ray_threshold",
        geometry_count,
        None,
        small_ray_threshold,
    )

    geometry = tuple(
        legacy_radial_geometry(
            response,
            peak,
            diameter,
            boundary_percent,
            large_ray_threshold=large_ray_threshold,
            small_ray_threshold=small_ray_threshold,
        )
        for peak in raw_peaks
    )
    values = tuple(float(response[peak]) for peak in raw_peaks)
    diameters = tuple(float(item.diameter_xy_px) for item in geometry)
    initial_centers = _legacy_center_indices(response, raw_peaks)

    def make_nucleus(center_disk: int, recovery_round: int, count: int) -> LegacyDetectorNucleus:
        disks = _legacy_plane_claims(
            geometry[center_disk], geometry, anisotropy=z_scale, z_size=response.shape[0]
        )
        log_odds = legacy_calculate_disk_log_odds(
            disks, center_disk, geometry, values, diameters, z_scale, model
        )
        provisional = LegacyDetectorNucleus(
            center_disk, disks, log_odds, (), recovery_round
        )
        center = geometry[center_disk].center_zyx_px
        location_xyz = (float(center[2] + 1), float(center[1] + 1), float(center[0] + 1))
        limit = _resolve(
            parameter_resolver,
            "rangethreshold",
            count,
            location_xyz,
            range_threshold,
        )
        retained = legacy_maximal_disk_range(provisional, geometry, limit)
        return LegacyDetectorNucleus(
            center_disk, disks, log_odds, retained, recovery_round
        )

    nuclei = [
        make_nucleus(center, 0, len(initial_centers)) for center in initial_centers
    ]
    initial_count = len(nuclei)
    recovery_round = 0
    previous_unclaimed: set[int] | None = None
    while True:
        claimed = {
            nucleus.disk_indices[position]
            for nucleus in nuclei
            for position in nucleus.retained_indices
        }
        unclaimed = set(range(len(raw_peaks))) - claimed
        if previous_unclaimed is not None:
            unclaimed &= previous_unclaimed
        ordered = tuple(sorted(unclaimed))
        if not ordered:
            break
        scaled = np.asarray(
            [
                (
                    raw_peaks[index][2],
                    raw_peaks[index][1],
                    raw_peaks[index][0] * z_scale,
                )
                for index in ordered
            ],
            dtype=np.float64,
        )
        delta = scaled[:, np.newaxis, :] - scaled[np.newaxis, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        selected: list[int] = []
        for position, disk_index in enumerate(ordered):
            neighbors = np.flatnonzero(distances[position] < 0.75 * diameter)
            neighbors = neighbors[neighbors != position]
            if neighbors.size and values[disk_index] >= max(
                values[ordered[int(item)]] for item in neighbors
            ):
                selected.append(disk_index)
        if not selected:
            break
        recovery_round += 1
        nuclei.extend(
            make_nucleus(center, recovery_round, int(numcells)) for center in selected
        )
        previous_unclaimed = unclaimed

    # Build overlap pairs in the same candidate/disk scan order as
    # buildOverlapList.m.  Each higher candidate stores lower claimants once.
    claimants: list[list[int]] = [[] for _ in raw_peaks]
    for candidate_index, nucleus in enumerate(nuclei):
        for position in nucleus.retained_indices:
            claimants[nucleus.disk_indices[position]].append(candidate_index)
    overlaps: list[list[int]] = [[] for _ in nuclei]
    for owners in claimants:
        for left_position, left in enumerate(owners):
            for right in owners[left_position + 1 :]:
                high, low = max(left, right), min(left, right)
                if low not in overlaps[high]:
                    overlaps[high].append(low)
    pair_order = tuple(
        (candidate, other)
        for candidate, others in enumerate(overlaps)
        for other in others
    )
    ranges = [nucleus.retained_indices for nucleus in nuclei]
    scored_pairs: list[tuple[int, int, float, float]] = []
    for first, second in pair_order:
        # mergeDecision receives the original range cell array, while
        # filter_boundary stores each returned pair into rangesnew.  Since our
        # ``working`` reflects prior rangesnew entries this is only different
        # for nuclei in several overlaps; preserve upstream by scoring against
        # the immutable pre-filter ranges below.
        original = nuclei
        split_score, range_first, range_second = _split_score(
            first, second, original, geometry
        )
        merge_score = _merge_score(
            first,
            second,
            original,
            geometry,
            values,
            diameters,
            z_scale,
            model,
        )
        ranges[first] = range_first
        ranges[second] = range_second
        scored_pairs.append((first, second, split_score, merge_score))
    nuclei = [
        LegacyDetectorNucleus(
            item.center_disk_index,
            item.disk_indices,
            item.log_odds,
            ranges[index],
            item.recovery_round,
        )
        for index, item in enumerate(nuclei)
    ]

    merge_lower_value = _resolve(
        parameter_resolver, "mergelower", int(numcells), None, merge_lower
    )
    merge_split_value = _resolve(
        parameter_resolver, "mergesplit", int(numcells), None, merge_split
    )
    selected_pairs: list[tuple[int, int]] = []
    for first, second, split_score, merge_score in scored_pairs:
        first_center = geometry[nuclei[first].center_disk_index].center_zyx_px
        second_center = geometry[nuclei[second].center_disk_index].center_zyx_px
        location_xyz = (
            float(first_center[2] + 1),
            float(first_center[1] + 1),
            float(first_center[0] + 1),
        )
        split_value = _resolve(
            parameter_resolver,
            "split",
            int(numcells),
            location_xyz,
            split_threshold,
        )
        distance_value = _resolve(
            parameter_resolver,
            "nndist_merge",
            int(numcells),
            location_xyz,
            normalized_merge_distance,
        )
        aspect_value = _resolve(
            parameter_resolver,
            "armerge",
            int(numcells),
            location_xyz,
            aspect_ratio_merge_threshold,
        )
        delta = np.asarray(first_center) - np.asarray(second_center)
        normalized_distance = math.sqrt(
            float(delta[1] ** 2 + delta[2] ** 2 + (delta[0] * z_scale) ** 2)
        ) / diameter
        union_z = [
            geometry[nuclei[first].disk_indices[position]].center_zyx_px[0]
            for position in nuclei[first].retained_indices
        ] + [
            geometry[nuclei[second].disk_indices[position]].center_zyx_px[0]
            for position in nuclei[second].retained_indices
        ]
        merge_ar = (
            (max(union_z) - min(union_z) + 1.0)
            * z_scale
            / max(
                diameters[nuclei[first].center_disk_index],
                diameters[nuclei[second].center_disk_index],
            )
        )
        good = split_score < split_value and merge_score > merge_lower_value
        good = good or merge_score > split_score * merge_split_value
        good = good or normalized_distance < distance_value
        good = good or merge_ar < aspect_value
        if good:
            selected_pairs.append((first, second))

    groups = _merge_groups(selected_pairs)
    merged_indices = {index for group in groups for index in group}
    final: list[LegacyResolvedCandidate] = []

    def candidate_from_group(group: tuple[int, ...]) -> LegacyResolvedCandidate:
        claimed_disks = tuple(
            nuclei[index].disk_indices[position]
            for index in group
            for position in nuclei[index].retained_indices
        )
        points = np.asarray(
            [geometry[index].center_zyx_px for index in claimed_disks],
            dtype=np.float64,
        )
        center = tuple(float(value) for value in np.mean(points, axis=0))
        center_disks = tuple(nuclei[index].center_disk_index for index in group)
        representative = max(center_disks, key=lambda index: values[index])
        final_diameter = max(diameters[index] for index in center_disks)
        claimed_geometry = tuple(geometry[index] for index in claimed_disks)
        z_values = points[:, 0]
        principal, secondary = _xy_principal_variances(
            claimed_geometry, float(boundary_percent)
        )
        log_sum = float(
            np.sum(
                np.asarray(
                    [
                        nuclei[index].log_odds[position]
                        for index in group
                        for position in nuclei[index].retained_indices
                    ],
                    dtype=np.float64,
                )
            )
        )
        retained_log_odds = tuple(
            nuclei[index].log_odds[position]
            for index in group
            for position in nuclei[index].retained_indices
        )
        return LegacyResolvedCandidate(
            center_zyx_px=center,
            diameter_xy_px=float(final_diameter),
            representative_peak_zyx=raw_peaks[representative],
            valid_ray_count=max(geometry[index].valid_ray_count for index in center_disks),
            claimed_slice_count=len(claimed_disks),
            merged_candidate_count=len(group),
            claimed_slices=claimed_geometry,
            aspect_ratio=float(
                (float(np.max(z_values)) - float(np.min(z_values)) + 1.0)
                * z_scale
                / final_diameter
            ),
            xy_principal_variance=principal,
            xy_secondary_variance=secondary,
            log_odds_sum=log_sum,
            claimed_log_odds=retained_log_odds,
            recovery_round=max(nuclei[index].recovery_round for index in group),
        )

    for group in groups:
        final.append(candidate_from_group(group))
    for index, nucleus in enumerate(nuclei):
        if index not in merged_indices:
            final.append(candidate_from_group((index,)))
    return LegacyDetectorTailResult(
        candidates=tuple(final),
        nuclei=tuple(nuclei),
        initial_center_count=initial_count,
        recovery_round_count=recovery_round,
        disk_count=len(raw_peaks),
        candidate_diameters_xy_px=tuple(
            diameters[item.center_disk_index] for item in nuclei
        ),
        merge_pairs=tuple(scored_pairs),
    )


__all__ = [
    "LegacyDetectorNucleus",
    "LegacyDetectorTailError",
    "LegacyDetectorTailResult",
    "LegacyDiskDistributionModel",
    "LegacyParameterResolver",
    "legacy_adapt_cell_diameter",
    "legacy_calculate_disk_log_odds",
    "legacy_disk_feature_vectors",
    "legacy_maximal_disk_range",
    "legacy_resolve_candidates_exact",
    "load_legacy_disk_distributions",
]
