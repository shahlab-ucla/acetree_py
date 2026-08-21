from types import SimpleNamespace

import numpy as np
import pytest

from acetree_py.analysis.roi_rasterization import (
    RoiCalibration,
    RoiMaskRasterizer,
    RoiRasterizationError,
)


def polygon(points, z=1):
    return SimpleNamespace(kind="polygon_2d", z_plane=z, exterior_xy_px=tuple(points))


def line(points, width=2.0, unit="px", z=1):
    return SimpleNamespace(
        kind="thick_polyline_2d",
        z_plane=z,
        points_xy_px=tuple(points),
        thickness=SimpleNamespace(value=width, unit=unit),
    )


def test_polygon_is_cropped_boundary_inclusive_and_winding_independent():
    rasterizer = RoiMaskRasterizer()
    points = ((2, 2), (5, 2), (5, 5), (2, 5))
    forward = rasterizer.rasterize(polygon(points), (10, 12))
    reverse = rasterizer.rasterize(polygon(tuple(reversed(points))), (10, 12))

    assert forward.slices == (slice(2, 6), slice(2, 6))
    assert forward.mask.shape == (4, 4)
    assert forward.in_bounds_sample_count == 16
    np.testing.assert_array_equal(forward.mask, reverse.mask)
    assert not forward.mask.flags.writeable


def test_concave_polygon_and_clipping_report_nominal_coverage():
    points = ((-1, 1), (3, 1), (3, 2), (1, 2), (1, 4), (-1, 4))
    result = RoiMaskRasterizer().rasterize(polygon(points), (6, 4))

    assert result.clipped
    assert result.coverage_fraction == pytest.approx(
        result.in_bounds_sample_count / result.nominal_sample_count
    )
    assert result.warnings == ("clipped_to_image",)


def test_self_intersecting_polygon_is_rejected():
    bow_tie = polygon(((1, 1), (4, 4), (1, 4), (4, 1)))
    with pytest.raises(RoiRasterizationError, match="self-intersect"):
        RoiMaskRasterizer().rasterize(bow_tie, (8, 8))


def test_round_thick_line_unions_bends_and_ignores_repeated_vertices():
    geometry = line(((2, 3), (5, 3), (5, 3), (5, 6)), width=2)
    result = RoiMaskRasterizer().rasterize(geometry, (10, 10))

    assert result.mask[result.slices[0].start - result.slices[0].start + 1, 0]
    assert result.in_bounds_sample_count == np.count_nonzero(result.mask)
    # The shared bend is counted once by the union, never once per segment.
    assert result.in_bounds_sample_count < 2 * 15


def test_physical_line_width_uses_xy_calibration():
    geometry = line(((2, 5), (8, 5)), width=2.0, unit="um")
    result = RoiMaskRasterizer(RoiCalibration(xy_res=0.5)).rasterize(
        geometry, (12, 12)
    )
    assert result.mask.shape[0] == 5
    with pytest.raises(RoiRasterizationError) as exc:
        RoiMaskRasterizer().rasterize(geometry, (12, 12))
    assert exc.value.reason == "calibration_unavailable"
