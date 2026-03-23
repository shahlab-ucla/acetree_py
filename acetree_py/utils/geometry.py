"""Geometry utilities for 3D vector math.

Provides helper functions used throughout the naming and analysis modules.
"""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector. Returns zero vector if input has zero length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    return v / norm


def distance_3d(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    ))


def distance_2d(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
