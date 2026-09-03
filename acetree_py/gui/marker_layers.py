"""Atomic helpers for record-derived napari marker layers."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def configure_curated_points_layer(layer, *, callback, lock) -> None:
    """Idempotently configure and lock a record-derived Points layer.

    Layer creation can succeed before napari configures text or a custom
    callback. Re-running this on every refresh makes such partial setup
    recoverable without duplicating callbacks or replacing the layer.
    """

    try:
        layer.text = {
            "string": "{name}",
            "color": "white",
            "size": 10,
        }
        callbacks = layer.mouse_drag_callbacks
        if callback not in callbacks:
            callbacks.append(callback)
    finally:
        lock(layer)


def pointer_position(event) -> np.ndarray:
    """Copy a mouse event's canvas position for later drag detection."""

    value = getattr(event, "pos", None)
    if value is None:
        value = getattr(event, "position", ())
    return np.asarray(value, dtype=float).reshape(-1).copy()


def passed_drag_threshold(event, press_position: np.ndarray) -> bool:
    """Return whether a mouse move is a drag rather than a click jitter."""

    current = pointer_position(event)
    dimensions = min(len(current), len(press_position), 2)
    if dimensions == 0:
        return False
    distance = float(
        np.linalg.norm(current[-dimensions:] - press_position[-dimensions:])
    )
    try:
        from qtpy.QtWidgets import QApplication

        threshold = float(QApplication.startDragDistance())
    except Exception:
        threshold = 3.0
    return distance >= max(1.0, threshold)


def point_anchor(layer, point_index: int) -> tuple[int, int] | None:
    """Read an immutable AceTree ``(time, index)`` from Points features."""

    try:
        features = layer.features
        time_values = features["acetree_time"]
        index_values = features["acetree_index"]
        if hasattr(time_values, "iloc"):
            time_value = time_values.iloc[point_index]
            index_value = index_values.iloc[point_index]
        else:
            time_value = time_values[point_index]
            index_value = index_values[point_index]
        return int(time_value), int(index_value)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def replace_points_layer(
    layer,
    *,
    data: np.ndarray,
    size: np.ndarray,
    face_color: np.ndarray,
    features: Mapping[str, Sequence],
) -> bool:
    """Replace all parallel Points fields or restore their prior state.

    napari updates ``data``, features, sizes, and colors through separate
    setters.  A failure in a later setter must not leave new centroids paired
    with stale styling or feature rows.  The existing layer object is retained
    so callbacks, visibility, layer order, and camera-mode state survive.

    Returns ``True`` after a complete replacement and ``False`` when a failed
    replacement was rolled back successfully.  A rollback failure is raised
    because the presentation can no longer be guaranteed internally
    consistent.
    """

    data_array = np.asarray(data, dtype=float)
    size_array = np.asarray(size, dtype=float)
    color_array = np.asarray(face_color, dtype=float)
    feature_values = {
        name: deepcopy(list(values)) for name, values in features.items()
    }

    if data_array.ndim != 2:
        raise ValueError("Points data must be a two-dimensional array")
    count = len(data_array)
    if size_array.ndim != 1 or len(size_array) != count:
        raise ValueError("Points sizes must contain one value per centroid")
    if color_array.shape != (count, 4):
        raise ValueError("Points colors must be an N x 4 RGBA array")
    mismatched = [
        name for name, values in feature_values.items() if len(values) != count
    ]
    if mismatched:
        raise ValueError(
            "Points features must contain one value per centroid: "
            + ", ".join(sorted(mismatched))
        )

    previous = {
        "data": np.array(layer.data, copy=True),
        "features": deepcopy(getattr(layer, "features", {})),
        "size": np.array(
            getattr(layer, "size", np.ones(len(layer.data))),
            copy=True,
        ),
        "face_color": np.array(
            getattr(layer, "face_color", np.zeros((len(layer.data), 4))),
            copy=True,
        ),
    }
    replacement = {
        "data": np.array(data_array, copy=True),
        "features": feature_values,
        "size": np.array(size_array, copy=True),
        # napari's color manager intentionally retains one default color row
        # when a Points layer is empty. Supplying a 0x4 array is treated as an
        # illegal color and emits a warning before being reset to white.
        "face_color": (
            np.array(color_array, copy=True)
            if count
            else np.ones((1, 4), dtype=float)
        ),
    }

    try:
        _assign_points_state(layer, replacement)
    except Exception as error:
        try:
            _assign_points_state(layer, previous)
        except Exception as restore_error:
            raise RuntimeError(
                "Centroid redraw and rollback both failed; refresh the view"
            ) from restore_error
        logger.warning(
            "Centroid Points redraw failed; restored the prior complete "
            "marker set on %s: %s",
            getattr(layer, "name", "Points"),
            error,
        )
        return False
    return True


def _assign_points_state(layer, state: Mapping[str, object]) -> None:
    """Assign Points fields in the order expected by napari's color model."""

    # Setting data can resize napari's feature and color arrays. Features can
    # in turn update the color manager, so the explicit curated colors win last.
    layer.data = deepcopy(state["data"])
    layer.features = deepcopy(state["features"])
    layer.size = deepcopy(state["size"])
    layer.face_color = deepcopy(state["face_color"])
