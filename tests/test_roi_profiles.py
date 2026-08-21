from types import SimpleNamespace

import numpy as np
import pytest

from acetree_py.analysis.roi_measure import RoiProfileSampler
from acetree_py.analysis.roi_rasterization import RoiCalibration


def line(points, width=2.0):
    return SimpleNamespace(
        kind="thick_polyline_2d",
        z_plane=1,
        points_xy_px=tuple(points),
        thickness=SimpleNamespace(value=width, unit="px"),
    )


def test_constant_and_linear_profiles_have_physical_distance():
    calibration = RoiCalibration(xy_res=0.5)
    constant = RoiProfileSampler().sample(
        np.full((10, 10), 7.0),
        line(((2, 4), (8, 4))),
        calibration=calibration,
    )
    gradient = np.tile(np.arange(10, dtype=float), (10, 1))
    linear = RoiProfileSampler().sample(
        gradient,
        line(((2, 4), (8, 4))),
        calibration=calibration,
    )

    assert constant.distance_um[-1] == pytest.approx(3.0)
    assert constant.mean == pytest.approx((7.0,) * len(constant))
    assert linear.mean == pytest.approx(np.linspace(2, 8, len(linear)))
    assert constant.width_step_um <= calibration.xy_res


def test_vertex_reversal_exactly_reverses_values():
    image = np.add.outer(np.arange(12) * 10.0, np.arange(12))
    geometry = line(((2, 2), (8, 2), (8, 8)), width=2)
    reverse = line(tuple(reversed(geometry.points_xy_px)), width=2)
    forward_result = RoiProfileSampler().sample(
        image, geometry, calibration=RoiCalibration(xy_res=1.0), step_um=0.7
    )
    reverse_result = RoiProfileSampler().sample(
        image, reverse, calibration=RoiCalibration(xy_res=1.0), step_um=0.7
    )
    assert reverse_result.mean == pytest.approx(tuple(reversed(forward_result.mean)))


def test_out_of_bounds_profile_has_explicit_gaps():
    image = np.ones((5, 5), dtype=float)
    result = RoiProfileSampler().sample(
        image,
        line(((-4, 2), (4, 2))),
        calibration=RoiCalibration(xy_res=1),
    )
    assert result.mean[0] is None
    assert result.missing_reasons[0] == "no_finite_pixels"
    assert result.mean[-1] == 1.0
