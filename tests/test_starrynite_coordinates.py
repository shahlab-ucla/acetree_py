"""Coordinate conformance checks for the legacy StarryNite boundary."""

from __future__ import annotations

import pytest

from acetree_py.tracking.api import Calibration
from acetree_py.tracking.starrynite import (
    LegacyCoordinateError,
    LegacyCoordinateTransform,
    matlab_round,
)


def test_launcher_export_transform_matches_roi_and_downsampling_contract() -> None:
    transform = LegacyCoordinateTransform(
        Calibration(0.16, 1.0),
        downsampling=0.5,
        roi_x_min=41,
        roi_y_min=3,
    )

    assert transform.matlab_local_to_acetree((5.0, 7.0, 4.0)) == (
        49.0,
        15.0,
        4.0,
    )
    assert transform.acetree_to_matlab_local((49.0, 15.0, 4.0)) == (
        5.0,
        7.0,
        4.0,
    )
    assert transform.tracking_anisotropy == pytest.approx(3.125)
    assert transform.diameter_to_acetree_pixels(10.0) == 20.0
    assert transform.diameter_to_radius_um(10.0) == pytest.approx(1.6)


def test_coordinate_round_trip_includes_at_physical_and_negative_offsets() -> None:
    transform = LegacyCoordinateTransform(
        Calibration(0.25, 1.5, plane_start=1),
        downsampling=1.25,
        roi_x_min=-7,
        roi_y_min=12,
    )
    local = (13.25, 8.75, 2.5)

    physical = transform.matlab_local_to_physical(local)
    assert transform.physical_to_matlab_local(physical) == pytest.approx(local)
    assert physical == pytest.approx((0.4, 4.25, 2.25))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2.5, 3),
        (1.5, 2),
        (1.49, 1),
        (-1.49, -1),
        (-1.5, -2),
        (-2.5, -3),
    ],
)
def test_matlab_round_uses_half_away_from_zero(value: float, expected: int) -> None:
    assert matlab_round(value) == expected


def test_rounded_export_applies_offset_before_rounding() -> None:
    transform = LegacyCoordinateTransform(
        Calibration(1.0, 1.0),
        downsampling=2.0,
        roi_x_min=1,
        roi_y_min=1,
    )

    # 3 / 2 + 1 - 2 = 0.5, which MATLAB writes as 1 rather than Python's 0.
    assert transform.rounded_acetree_export_xyz((3.0, -1.0, 2.5)) == (1, -2, 3)


def test_coordinate_contract_rejects_nonfinite_or_nonpositive_inputs() -> None:
    with pytest.raises(LegacyCoordinateError, match="downsampling must be positive"):
        LegacyCoordinateTransform(Calibration(1.0, 1.0), downsampling=0)
    transform = LegacyCoordinateTransform(Calibration(1.0, 1.0))
    with pytest.raises(LegacyCoordinateError, match="three coordinates"):
        transform.matlab_local_to_acetree((1.0, 2.0))
    with pytest.raises(LegacyCoordinateError, match="finite"):
        transform.matlab_local_to_acetree((1.0, 2.0, float("nan")))
