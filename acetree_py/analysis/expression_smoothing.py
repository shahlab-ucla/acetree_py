"""Gap-preserving smoothing shared by expression plotting workflows."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian_smooth_missing(
    values: Iterable[float | None],
    sigma_samples: float,
    *,
    truncate: float = 4.0,
) -> tuple[float | None, ...]:
    """Gaussian-smooth finite contiguous segments without crossing gaps.

    ``sigma_samples`` is expressed in sample bins. Missing/non-finite inputs
    remain missing, singleton segments remain unchanged, and sigma zero is an
    exact identity. ``mode='nearest'`` avoids depressing finite segment edges
    and preserves a constant trace exactly.
    """

    sigma = float(sigma_samples)
    truncate = float(truncate)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Gaussian smoothing sigma must be finite and non-negative")
    if not math.isfinite(truncate) or truncate <= 0:
        raise ValueError("Gaussian smoothing truncate must be finite and positive")

    normalized = tuple(_finite_or_none(value) for value in values)
    if sigma == 0 or not normalized:
        return normalized

    output = list(normalized)
    start = 0
    while start < len(normalized):
        while start < len(normalized) and normalized[start] is None:
            start += 1
        if start >= len(normalized):
            break
        end = start + 1
        while end < len(normalized) and normalized[end] is not None:
            end += 1
        segment = normalized[start:end]
        if len(segment) > 1:
            filtered = gaussian_filter1d(
                np.asarray(segment, dtype=float),
                sigma=sigma,
                mode="nearest",
                truncate=truncate,
            )
            output[start:end] = [float(value) for value in filtered]
        start = end
    return tuple(output)


def gaussian_smooth_on_grid(
    values: Iterable[float | None],
    sigma_axis_units: float,
    grid_step: float,
    *,
    truncate: float = 4.0,
) -> tuple[float | None, ...]:
    """Smooth using sigma expressed in the displayed axis's units."""

    step = float(grid_step)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Expression comparison grid step must be positive")
    return gaussian_smooth_missing(
        values,
        float(sigma_axis_units) / step,
        truncate=truncate,
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


__all__ = ["gaussian_smooth_missing", "gaussian_smooth_on_grid"]
