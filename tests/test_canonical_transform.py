"""Tests for acetree_py.naming.canonical_transform."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.naming.canonical_transform import (
    AP_CANONICAL,
    DV_CANONICAL,
    LR_CANONICAL,
    CanonicalTransform,
    TransformValidationError,
    build_v1_sign_matrix,
)


class TestCanonicalTransform:
    """Test the improved CanonicalTransform using scipy."""

    def test_identity_transform(self):
        """When AP/LR already point canonical, rotation should be identity-like."""
        ct = CanonicalTransform(
            ap_vec=np.array([-1.0, 0.0, 0.0]),
            lr_vec=np.array([0.0, 0.0, 1.0]),
        )
        assert ct.active

        # Applying to canonical vectors should return them unchanged
        result_ap = ct.apply(np.array([-1.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result_ap, AP_CANONICAL)

        result_lr = ct.apply(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_almost_equal(result_lr, LR_CANONICAL)

    def test_rotated_embryo(self):
        """Test with non-trivial AP/LR vectors that need rotation."""
        # AP along +y, LR along +x  (90° rotation from canonical)
        ct = CanonicalTransform(
            ap_vec=np.array([0.0, 1.0, 0.0]),
            lr_vec=np.array([1.0, 0.0, 0.0]),
        )
        assert ct.active

        # The AP vector should map to canonical AP
        result = ct.apply(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_almost_equal(result, AP_CANONICAL, decimal=4)

        # The LR vector should map to canonical LR
        result = ct.apply(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, LR_CANONICAL, decimal=4)

    def test_arbitrary_orientation(self):
        """Test with an arbitrary orthogonal orientation."""
        # Create a rotated frame (45° rotation around z)
        angle = np.pi / 4
        ap = np.array([-np.cos(angle), -np.sin(angle), 0.0])
        lr = np.array([-np.sin(angle), np.cos(angle), 0.0])

        ct = CanonicalTransform(ap_vec=ap, lr_vec=lr)
        assert ct.active

        result_ap = ct.apply(ap)
        np.testing.assert_array_almost_equal(result_ap, AP_CANONICAL, decimal=4)

        result_lr = ct.apply(lr)
        np.testing.assert_array_almost_equal(result_lr, LR_CANONICAL, decimal=4)

    def test_preserves_dv_direction(self):
        """DV should map to canonical DV after transform."""
        ct = CanonicalTransform(
            ap_vec=np.array([-1.0, 0.0, 0.0]),
            lr_vec=np.array([0.0, 0.0, 1.0]),
        )

        # DV = AP × LR
        dv = np.cross(np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        result_dv = ct.apply(dv)
        # DV_CANONICAL = cross(AP_CANONICAL, LR_CANONICAL) = [0, 1, 0]
        np.testing.assert_array_almost_equal(result_dv, DV_CANONICAL, decimal=4)

    def test_zero_vector_raises(self):
        """Zero-length vectors should raise an error."""
        with pytest.raises(TransformValidationError, match="too small"):
            CanonicalTransform(
                ap_vec=np.array([0.0, 0.0, 0.0]),
                lr_vec=np.array([0.0, 0.0, 1.0]),
            )

    def test_parallel_vectors_raises(self):
        """Parallel AP and LR should raise an error."""
        with pytest.raises(TransformValidationError, match="parallel"):
            CanonicalTransform(
                ap_vec=np.array([1.0, 0.0, 0.0]),
                lr_vec=np.array([2.0, 0.0, 0.0]),
            )

    def test_unnormalized_inputs(self):
        """Should handle unnormalized input vectors."""
        ct = CanonicalTransform(
            ap_vec=np.array([-5.0, 0.0, 0.0]),
            lr_vec=np.array([0.0, 0.0, 3.0]),
        )
        assert ct.active
        result = ct.apply(np.array([-1.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, AP_CANONICAL, decimal=4)

    def test_rmsd_is_small(self):
        """RMSD should be very small for a valid transform."""
        ct = CanonicalTransform(
            ap_vec=np.array([-1.0, 0.0, 0.0]),
            lr_vec=np.array([0.0, 0.0, 1.0]),
        )
        assert ct.rmsd < 1e-4

    def test_apply_division_vector(self):
        """Test applying the transform to a realistic division vector."""
        ct = CanonicalTransform(
            ap_vec=np.array([-1.0, 0.0, 0.0]),
            lr_vec=np.array([0.0, 0.0, 1.0]),
        )

        # A division vector along the AP axis
        div_vec = np.array([10.0, 0.0, 0.0])
        result = ct.apply(div_vec)
        # Should still be along x after identity-like transform
        assert abs(result[0]) > abs(result[1])
        assert abs(result[0]) > abs(result[2])


class TestBuildV1SignMatrix:
    """Test v1 axis string → sign matrix conversion."""

    def test_adl(self):
        """ADL = default orientation, all positive."""
        mat = build_v1_sign_matrix("ADL")
        expected = np.diag([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(mat, expected)

    def test_avr(self):
        """AVR = A(+x), V(-y), R(-z)."""
        mat = build_v1_sign_matrix("AVR")
        expected = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_array_equal(mat, expected)

    def test_pdr(self):
        """PDR = P(-x), D(+y), R(-z)."""
        mat = build_v1_sign_matrix("PDR")
        expected = np.diag([-1.0, 1.0, -1.0])
        np.testing.assert_array_equal(mat, expected)

    def test_pvl(self):
        """PVL = P(-x), V(-y), L(+z)."""
        mat = build_v1_sign_matrix("PVL")
        expected = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_array_equal(mat, expected)

    def test_short_string_defaults(self):
        """Short axis string should default to ADL."""
        mat = build_v1_sign_matrix("A")
        expected = np.diag([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(mat, expected)

    def test_all_eight_orientations(self):
        """All 8 valid orientations should produce valid diagonal matrices."""
        for ap in ("A", "P"):
            for dv in ("D", "V"):
                for lr in ("L", "R"):
                    axis = ap + dv + lr
                    mat = build_v1_sign_matrix(axis)
                    # Should be diagonal with +/-1 entries
                    assert mat.shape == (3, 3)
                    for i in range(3):
                        assert abs(mat[i, i]) == 1.0
                        for j in range(3):
                            if i != j:
                                assert mat[i, j] == 0.0
