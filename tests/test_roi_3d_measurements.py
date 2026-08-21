from types import SimpleNamespace

import numpy as np
import pytest

from acetree_py.analysis.roi_measure import RoiIntensityReducer
from acetree_py.analysis.roi_rasterization import (
    RoiCalibration,
    RoiMaskRasterizer,
    RoiRasterizationError,
)


def contour(z, points=((1, 1), (4, 1), (4, 4), (1, 4))):
    return SimpleNamespace(z_plane=z, exterior_xy_px=tuple(points))


def stack(planes, mode="filled_volume", shell=None):
    return SimpleNamespace(
        kind="contour_stack_3d",
        slices=tuple(contour(z) for z in planes),
        sampling_mode=mode,
        shell_thickness_um=shell,
    )


def test_anisotropic_volume_and_surface_are_physical():
    calibration = RoiCalibration(xy_res=0.5, z_res=2.0, plane_start=5)
    geometry = stack((5, 6, 7))
    raster = RoiMaskRasterizer(calibration).rasterize(geometry, (3, 6, 6))
    result = RoiIntensityReducer().reduce(
        np.ones((3, 6, 6)), raster, geometry=geometry, calibration=calibration
    )
    assert result.value("geometry.volume_um3") == pytest.approx(
        raster.in_bounds_sample_count * 0.5 * 0.5 * 2.0
    )
    assert result.value("geometry.surface_area_um2") > 0
    assert result.value("intensity.sum_per_volume_um3") == pytest.approx(2.0)


def test_inner_shell_is_subset_and_respects_physical_thickness():
    calibration = RoiCalibration(xy_res=1, z_res=1, plane_start=1)
    filled = RoiMaskRasterizer(calibration).rasterize(stack((1, 2, 3)), (3, 6, 6))
    shell = RoiMaskRasterizer(calibration).rasterize(
        stack((1, 2, 3), mode="inner_shell", shell=1.0), (3, 6, 6)
    )
    assert np.all(~shell.mask | shell.parent_mask)
    assert shell.in_bounds_sample_count <= filled.in_bounds_sample_count


def test_contour_gap_is_not_implicitly_interpolated():
    with pytest.raises(RoiRasterizationError, match="consecutive"):
        RoiMaskRasterizer(RoiCalibration(xy_res=1, z_res=1)).rasterize(
            stack((1, 3)), (3, 6, 6)
        )


def test_boundary_touching_stack_is_valid_but_clipped():
    geometry = SimpleNamespace(
        kind="contour_stack_3d",
        slices=(contour(1, ((-2, 1), (2, 1), (2, 4), (-2, 4))),),
        sampling_mode="filled_volume",
        shell_thickness_um=None,
    )
    result = RoiMaskRasterizer(RoiCalibration(xy_res=1, z_res=1)).rasterize(
        geometry, (1, 6, 6)
    )
    assert result.clipped
    assert result.surface_area_um2 > 0
