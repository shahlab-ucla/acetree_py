"""Contracts for atomic record-derived Points layer replacement."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from acetree_py.gui.marker_layers import replace_points_layer


class _FaultInjectingPoints:
    def __init__(self) -> None:
        self.name = "Nuclei 3D"
        self._data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._features = {"name": ["OldA", "OldB"]}
        self._size = np.array([8.0, 9.0])
        self._face_color = np.array(
            [[0.2, 0.3, 0.4, 1.0], [0.5, 0.6, 0.7, 1.0]]
        )
        self.fail_next: str | None = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value) -> None:
        self._data = np.asarray(value, dtype=float)
        self._maybe_fail("data")

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value) -> None:
        self._features = deepcopy(value)
        self._maybe_fail("features")

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value) -> None:
        self._size = np.asarray(value, dtype=float)
        self._maybe_fail("size")

    @property
    def face_color(self):
        return self._face_color

    @face_color.setter
    def face_color(self, value) -> None:
        self._face_color = np.asarray(value, dtype=float)
        self._maybe_fail("face_color")

    def _maybe_fail(self, field: str) -> None:
        if self.fail_next == field:
            self.fail_next = None
            raise RuntimeError(f"simulated {field} setter failure")


def _replacement(count: int) -> dict:
    return {
        "data": np.arange(count * 3, dtype=float).reshape(count, 3),
        "size": np.arange(1, count + 1, dtype=float),
        "face_color": np.tile([0.9, 0.2, 0.4, 1.0], (count, 1)),
        "features": {
            "name": [f"New{i}" for i in range(count)],
            "acetree_time": [3] * count,
            "acetree_index": list(range(1, count + 1)),
        },
    }


@pytest.mark.parametrize("failed_field", ["features", "size", "face_color"])
def test_partial_points_failure_restores_all_parallel_fields(
    failed_field,
    caplog,
) -> None:
    layer = _FaultInjectingPoints()
    previous = {
        "data": layer.data.copy(),
        "features": deepcopy(layer.features),
        "size": layer.size.copy(),
        "face_color": layer.face_color.copy(),
    }
    layer.fail_next = failed_field

    with caplog.at_level("WARNING"):
        completed = replace_points_layer(layer, **_replacement(1))

    assert completed is False
    np.testing.assert_allclose(layer.data, previous["data"])
    assert layer.features == previous["features"]
    np.testing.assert_allclose(layer.size, previous["size"])
    np.testing.assert_allclose(layer.face_color, previous["face_color"])
    assert "restored the prior complete marker set" in caplog.text


def test_points_replacement_keeps_every_field_aligned_through_empty() -> None:
    layer = _FaultInjectingPoints()

    for count in (1, 2, 0):
        assert replace_points_layer(layer, **_replacement(count)) is True
        assert layer.data.shape == (count, 3)
        assert layer.size.shape == (count,)
        expected_colors = count if count else 1
        assert layer.face_color.shape == (expected_colors, 4)
        assert all(len(values) == count for values in layer.features.values())


def test_points_payload_is_validated_before_layer_mutation() -> None:
    layer = _FaultInjectingPoints()
    previous = layer.data.copy()
    payload = _replacement(1)
    payload["features"]["name"] = ["one", "too many"]

    with pytest.raises(ValueError, match="one value per centroid"):
        replace_points_layer(layer, **payload)

    np.testing.assert_allclose(layer.data, previous)


def test_real_napari_points_accepts_atomic_two_to_one_to_empty() -> None:
    napari = pytest.importorskip("napari")
    viewer = napari.components.ViewerModel()
    initial = _replacement(2)
    layer = viewer.add_points(
        initial["data"],
        size=initial["size"],
        face_color=initial["face_color"],
        features=initial["features"],
    )

    for count in (1, 0):
        assert replace_points_layer(layer, **_replacement(count)) is True
        assert len(layer.data) == count
        assert len(layer.size) == count
        assert len(layer.face_color) == (count if count else 1)
        assert len(layer.features) == count
