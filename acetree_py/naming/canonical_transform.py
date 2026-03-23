"""Canonical coordinate transform for embryo orientation.

Rotates vectors from the embryo's measured orientation into a canonical
frame where:
    AP (anterior-posterior) → (-1, 0, 0)
    LR (left-right)        → ( 0, 0, 1)
    DV (dorsal-ventral)    → ( 0, 1, 0)   [= AP × LR]

This replaces ~300 lines of Java in CanonicalTransform.java with a robust
implementation using scipy's Rotation.align_vectors() (SVD-based Wahba's
problem solver), which handles ALL orientations correctly without manual
degenerate-case branches.

Ported from: org.rhwlab.snight.CanonicalTransform
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# Canonical frame vectors
AP_CANONICAL = np.array([-1.0, 0.0, 0.0])
LR_CANONICAL = np.array([0.0, 0.0, 1.0])
DV_CANONICAL = np.cross(AP_CANONICAL, LR_CANONICAL)  # [0, 1, 0]


class TransformValidationError(Exception):
    """Raised when the canonical transform cannot be validated."""


class CanonicalTransform:
    """3D rotation that maps embryo orientation vectors to canonical frame.

    Uses scipy.spatial.transform.Rotation.align_vectors() which solves
    Wahba's problem via SVD — handles all orientations robustly.

    Attributes:
        rotation: The computed Rotation object.
        active: True if the transform was successfully computed and validated.
        rmsd: Root-mean-square deviation of the alignment.
    """

    def __init__(
        self,
        ap_vec: np.ndarray,
        lr_vec: np.ndarray,
        tolerance: float = 1e-4,
    ) -> None:
        """Build a canonical transform from measured AP and LR vectors.

        Args:
            ap_vec: Measured anterior-posterior orientation vector.
            lr_vec: Measured left-right orientation vector.
            tolerance: Maximum RMSD for validation.

        Raises:
            TransformValidationError: If the rotation cannot be validated
                within the given tolerance.
        """
        self.active = False
        self.rmsd = float("inf")

        # Normalize input vectors
        ap = np.asarray(ap_vec, dtype=np.float64)
        lr = np.asarray(lr_vec, dtype=np.float64)

        ap_norm = np.linalg.norm(ap)
        lr_norm = np.linalg.norm(lr)

        if ap_norm < 1e-10 or lr_norm < 1e-10:
            raise TransformValidationError(
                f"Input vectors too small: |AP|={ap_norm:.2e}, |LR|={lr_norm:.2e}"
            )

        ap = ap / ap_norm
        lr = lr / lr_norm

        # Compute DV as cross product (completes the right-handed frame)
        dv = np.cross(ap, lr)
        dv_norm = np.linalg.norm(dv)
        if dv_norm < 1e-10:
            raise TransformValidationError(
                "AP and LR vectors are nearly parallel; cannot form a frame"
            )
        dv = dv / dv_norm

        # Build source and target basis frames (rows = basis vectors)
        source = np.stack([ap, dv, lr])
        target = np.stack([AP_CANONICAL, DV_CANONICAL, LR_CANONICAL])

        # Solve Wahba's problem: find rotation R such that target ≈ R @ source
        rotation, rmsd = Rotation.align_vectors(target, source)
        self.rotation = rotation
        self.rmsd = rmsd

        # Validate the result
        transformed_ap = rotation.apply(ap)
        transformed_lr = rotation.apply(lr)

        ap_ok = np.allclose(transformed_ap, AP_CANONICAL, atol=tolerance)
        lr_ok = np.allclose(transformed_lr, LR_CANONICAL, atol=tolerance)

        if not (ap_ok and lr_ok):
            msg = (
                f"Rotation validation failed: "
                f"transformed AP={transformed_ap} (expected {AP_CANONICAL}), "
                f"transformed LR={transformed_lr} (expected {LR_CANONICAL}), "
                f"rmsd={rmsd:.6f}"
            )
            raise TransformValidationError(msg)

        self.active = True
        logger.debug(
            "CanonicalTransform computed successfully (rmsd=%.6f)", rmsd
        )

    def apply(self, vec: np.ndarray) -> np.ndarray:
        """Apply the canonical rotation to a vector.

        Args:
            vec: A 3D vector in the embryo's measured coordinate frame.

        Returns:
            The vector rotated into the canonical frame.
        """
        return self.rotation.apply(np.asarray(vec, dtype=np.float64))

    def __repr__(self) -> str:
        return f"CanonicalTransform(active={self.active}, rmsd={self.rmsd:.6f})"


def build_v1_sign_matrix(axis_string: str) -> np.ndarray:
    """Build a sign-flip matrix from a v1 axis string.

    AuxInfo v1 encodes the embryo orientation as a 3-character string
    like "ADL" (Anterior-Dorsal-Left), which tells us which canonical
    direction each measured axis points toward.

    The Java code has 8 hardcoded if-branches for this. Instead, we
    compute the sign matrix directly from the axis string.

    Args:
        axis_string: A 3-character string like "ADL", "AVR", "PDR", "PVL", etc.
            Position 0: A(nterior) or P(osterior) → x-sign
            Position 1: D(orsal) or V(entral) → y-sign
            Position 2: L(eft) or R(ight) → z-sign

    Returns:
        A 3x3 diagonal matrix of signs (+1 or -1) for x, y, z.
    """
    if len(axis_string) < 3:
        logger.warning("Axis string '%s' too short; using default ADL", axis_string)
        axis_string = "ADL"

    axis_string = axis_string.upper()

    # AP axis: 'A' means measured axis aligns with canonical AP (-x direction)
    # So 'A' → x_sign = +1 (no flip needed), 'P' → x_sign = -1
    x_sign = 1.0 if axis_string[0] == "A" else -1.0

    # DV axis: 'D' means measured aligns with canonical DV (-y direction)
    # 'D' → y_sign = +1, 'V' → y_sign = -1
    y_sign = 1.0 if axis_string[1] == "D" else -1.0

    # LR axis: 'L' means measured aligns with canonical LR (+z direction)
    # 'L' → z_sign = +1, 'R' → z_sign = -1
    z_sign = 1.0 if axis_string[2] == "L" else -1.0

    return np.diag([x_sign, y_sign, z_sign])
