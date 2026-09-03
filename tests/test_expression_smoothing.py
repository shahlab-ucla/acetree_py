"""Gap-preserving Gaussian smoothing tests."""

from __future__ import annotations

import pytest

from acetree_py.analysis.expression_smoothing import (
    gaussian_smooth_missing,
    gaussian_smooth_on_grid,
)


def test_sigma_zero_is_exact_identity_and_normalizes_nonfinite():
    assert gaussian_smooth_missing([1.0, None, float("nan"), 2.0], 0) == (
        1.0,
        None,
        None,
        2.0,
    )


def test_constant_segments_and_gaps_are_preserved():
    result = gaussian_smooth_missing([5.0, 5.0, None, 100.0, 100.0], 2.0)
    assert result[:2] == pytest.approx((5.0, 5.0))
    assert result[2] is None
    assert result[3:] == pytest.approx((100.0, 100.0))


def test_gaussian_does_not_bleed_across_missing_gap():
    result = gaussian_smooth_missing([0.0, 0.0, None, 100.0, 100.0], 1.0)
    assert result[:2] == pytest.approx((0.0, 0.0))
    assert result[2] is None
    assert result[3:] == pytest.approx((100.0, 100.0))


def test_axis_sigma_is_converted_to_grid_bins():
    by_axis = gaussian_smooth_on_grid([0.0, 10.0, 0.0], 2.0, 2.0)
    by_sample = gaussian_smooth_missing([0.0, 10.0, 0.0], 1.0)
    assert by_axis == pytest.approx(by_sample)


@pytest.mark.parametrize("sigma", [-1.0, float("nan")])
def test_invalid_sigma_is_rejected(sigma: float):
    with pytest.raises(ValueError, match="sigma"):
        gaussian_smooth_missing([1.0], sigma)


def test_gaussian_truncation_is_applied_and_validated():
    values = [0.0, 0.0, 10.0, 0.0, 0.0]
    narrow = gaussian_smooth_missing(values, 1.0, truncate=0.5)
    wide = gaussian_smooth_missing(values, 1.0, truncate=4.0)

    assert narrow != wide
    with pytest.raises(ValueError, match="truncate"):
        gaussian_smooth_missing(values, 1.0, truncate=0.0)
