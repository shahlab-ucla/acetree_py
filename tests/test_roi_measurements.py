from types import SimpleNamespace

import numpy as np
import pytest

from acetree_py.analysis.roi_measure import RoiIntensityReducer
from acetree_py.analysis.roi_rasterization import RoiCalibration, RoiMaskRasterizer


def polygon(points):
    return SimpleNamespace(kind="polygon_2d", z_plane=1, exterior_xy_px=tuple(points))


def test_exact_finite_only_aggregates_and_area_normalization():
    geometry = polygon(((1, 1), (3, 1), (3, 3), (1, 3)))
    raster = RoiMaskRasterizer().rasterize(geometry, (5, 5))
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    image[1, 1] = np.nan
    result = RoiIntensityReducer().reduce(
        image,
        raster,
        geometry=geometry,
        calibration=RoiCalibration(xy_res=0.5),
    )
    expected = image[raster.slices][raster.mask]
    expected = expected[np.isfinite(expected)].astype(np.float64)

    assert result.status == "valid"
    assert result.finite_sample_count == 8
    assert result.value("intensity.sum") == pytest.approx(np.sum(expected, dtype=np.float64))
    assert result.value("intensity.mean") == pytest.approx(expected.mean())
    assert result.value("intensity.median") == pytest.approx(np.median(expected))
    assert result.value("geometry.area_um2") == pytest.approx(8 * 0.25)
    assert result.value("intensity.sum_per_area_um2") == pytest.approx(
        expected.sum() / (8 * 0.25)
    )


def test_zero_is_valid_but_no_finite_pixels_is_missing():
    geometry = polygon(((0, 0), (2, 0), (2, 2), (0, 2)))
    raster = RoiMaskRasterizer().rasterize(geometry, (3, 3))
    zero = RoiIntensityReducer().reduce(np.zeros((3, 3)), raster, geometry=geometry)
    missing = RoiIntensityReducer().reduce(
        np.full((3, 3), np.nan), raster, geometry=geometry
    )

    assert zero.status == "valid"
    assert zero.value("intensity.sum") == 0
    assert missing.status == "no_finite_pixels"
    assert missing.value("intensity.sum") is None
    assert missing.missing_reason("intensity.sum") == "no_finite_pixels"


def test_float64_sum_does_not_overflow_integer_source():
    mask = np.ones((2, 2), dtype=bool)
    image = np.full((2, 2), np.iinfo(np.uint32).max, dtype=np.uint32)
    result = RoiIntensityReducer().reduce(image, mask)
    assert result.value("intensity.sum") == float(np.iinfo(np.uint32).max) * 4


def test_missing_calibration_only_blocks_physical_metrics():
    result = RoiIntensityReducer().reduce(np.ones((2, 2)), np.ones((2, 2), bool))
    assert result.value("intensity.mean") == 1
    assert result.metric("geometry.area_um2").reason == "calibration_unavailable"


def test_distribution_uses_only_finite_samples():
    image = np.array([[0.0, 1.0], [2.0, np.inf]])
    result = RoiIntensityReducer().reduce(
        image,
        np.ones_like(image, dtype=bool),
        include_distribution=True,
        histogram_bins=3,
    )
    assert sum(result.distribution.histogram_counts) == 3
    assert result.distribution.quantiles[0.5] == 1.0
