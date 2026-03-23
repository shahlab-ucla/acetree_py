"""Division caller — assigns daughter names when a cell divides.

Given a dividing parent and its two daughters, this module:
  1. Computes the division vector (daughter2 - daughter1)
  2. Applies corrections (z-scaling, embryo shape via measurement correction)
  3. Rotates to canonical frame (v2: CanonicalTransform, v1: angle + sign-flipping)
  4. Dots the corrected vector with the Rule's axis vector
  5. Assigns daughter1/daughter2 names based on the sign of the dot product

Enhanced with:
  - Multi-frame division vector averaging for noise reduction
  - Confidence scoring for each division classification
  - Support for founder-derived axes (no AuxInfo required)

Ported from: org.rhwlab.snight.DivisionCaller
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ..core.nucleus import NILLI, Nucleus
from .canonical_transform import CanonicalTransform, build_v1_sign_matrix
from .rules import Rule, RuleManager

logger = logging.getLogger(__name__)

# Confidence thresholds for division classification
HIGH_CONFIDENCE_ANGLE = 20.0    # degrees — very confident
LOW_CONFIDENCE_ANGLE = 40.0     # degrees — still trust the rule but uncertain
RULE_OVERRIDE_ANGLE = 55.0      # degrees — use observed dominant axis instead

# Number of frames to average division vector over
DEFAULT_AVG_FRAMES = 3


@dataclass
class DivisionClassification:
    """Result of classifying a single cell division.

    Attributes:
        parent_name: The dividing parent cell name.
        daughter1_name: Name assigned to successor1.
        daughter2_name: Name assigned to successor2.
        axis_used: Which axis the division was classified along ("ap", "lr", "dv").
        confidence: 0-1 confidence in this classification.
        angle_from_rule: Degrees deviation from the expected rule vector.
        dot_product: Raw dot product with the rule axis.
    """

    parent_name: str = ""
    daughter1_name: str = ""
    daughter2_name: str = ""
    axis_used: str = ""
    confidence: float = 1.0
    angle_from_rule: float = 0.0
    dot_product: float = 0.0


class DivisionCaller:
    """Assigns Sulston names to daughter cells upon division.

    Supports three modes:
        - v2 mode: Uses CanonicalTransform (from AuxInfo v2 orientation vectors)
        - v1 mode: Uses axis string + angle rotation (from AuxInfo v1)
        - founder mode: Uses axes derived from founder cell positions
    """

    def __init__(
        self,
        rule_manager: RuleManager,
        z_pix_res: float = 11.1,
        canonical_transform: CanonicalTransform | None = None,
        axis_string: str = "",
        angle: float = 0.0,
        founder_ap: np.ndarray | None = None,
        founder_lr: np.ndarray | None = None,
        founder_dv: np.ndarray | None = None,
    ) -> None:
        """Initialize the DivisionCaller.

        Args:
            rule_manager: The RuleManager providing division rules.
            z_pix_res: Z pixel resolution (z_res / xy_res).
            canonical_transform: For v2 mode — the CanonicalTransform.
            axis_string: For v1 mode — e.g. "ADL".
            angle: For v1 mode — embryo rotation angle in degrees.
            founder_ap: AP axis from founder identification (unit vector).
            founder_lr: LR axis from founder identification (unit vector).
            founder_dv: DV axis from founder identification (unit vector).
        """
        self.rule_manager = rule_manager
        self.z_pix_res = z_pix_res
        self.canonical_transform = canonical_transform
        self.axis_string = axis_string
        self.angle = angle

        # Founder-derived axes
        self.founder_ap = founder_ap
        self.founder_lr = founder_lr
        self.founder_dv = founder_dv

        # Precompute v1 sign matrix if needed
        self._v1_sign_matrix: np.ndarray | None = None
        if axis_string and not canonical_transform:
            self._v1_sign_matrix = build_v1_sign_matrix(axis_string)

        # Track classification results for validation
        self._classifications: list[DivisionClassification] = []

    @property
    def is_v2(self) -> bool:
        """True if using v2 mode (CanonicalTransform)."""
        return self.canonical_transform is not None and self.canonical_transform.active

    @property
    def is_founder_mode(self) -> bool:
        """True if using founder-derived axes."""
        return (
            self.founder_ap is not None
            and self.founder_lr is not None
            and not self.is_v2
            and not self.axis_string
        )

    @property
    def classifications(self) -> list[DivisionClassification]:
        """All division classifications made so far."""
        return self._classifications

    def assign_names(
        self,
        parent: Nucleus,
        daughter1: Nucleus,
        daughter2: Nucleus,
    ) -> tuple[str, str]:
        """Assign Sulston names to the two daughters of a dividing cell.

        Args:
            parent: The dividing parent nucleus.
            daughter1: First daughter nucleus (successor1).
            daughter2: Second daughter nucleus (successor2).

        Returns:
            (name1, name2) — the names to assign to daughter1 and daughter2.
        """
        parent_name = parent.effective_name
        if not parent_name:
            return "", ""

        rule = self.rule_manager.get_rule(parent_name)
        classification = self._classify_division(parent, daughter1, daughter2, rule)
        self._classifications.append(classification)

        return classification.daughter1_name, classification.daughter2_name

    def assign_names_multi_frame(
        self,
        parent: Nucleus,
        daughter1: Nucleus,
        daughter2: Nucleus,
        nuclei_record: list[list[Nucleus]],
        division_time: int,
        n_frames: int = DEFAULT_AVG_FRAMES,
    ) -> tuple[str, str]:
        """Assign names using multi-frame averaged division vector.

        Averages the division vector over n_frames after the division event
        for improved noise robustness.

        Args:
            parent: The dividing parent nucleus.
            daughter1: First daughter at division_time.
            daughter2: Second daughter at division_time.
            nuclei_record: Full nuclei record for multi-frame lookup.
            division_time: 0-based timepoint when daughters first appear.
            n_frames: Number of frames to average over.

        Returns:
            (name1, name2) — the names to assign.
        """
        parent_name = parent.effective_name
        if not parent_name:
            return "", ""

        rule = self.rule_manager.get_rule(parent_name)

        # Compute averaged division vector
        avg_diff, avg_confidence = self._compute_averaged_diff(
            daughter1, daughter2, nuclei_record, division_time, n_frames,
        )

        if avg_diff is None:
            # Fallback to single-frame
            return self.assign_names(parent, daughter1, daughter2)

        # Dot with rule axis
        dot = float(np.dot(avg_diff, rule.axis_vector))

        # Compute angle from rule
        diff_norm = np.linalg.norm(avg_diff)
        if diff_norm > 1e-6:
            cos_angle = abs(dot) / (diff_norm * np.linalg.norm(rule.axis_vector))
            cos_angle = min(1.0, cos_angle)
            angle_deg = math.degrees(math.acos(cos_angle))
        else:
            angle_deg = 90.0

        # Compute confidence
        confidence = _angle_to_confidence(angle_deg) * avg_confidence

        # Classify
        if dot >= 0:
            name1, name2 = rule.daughter1, rule.daughter2
        else:
            name1, name2 = rule.daughter2, rule.daughter1

        classification = DivisionClassification(
            parent_name=parent_name,
            daughter1_name=name1,
            daughter2_name=name2,
            axis_used=rule.sulston_letter,
            confidence=confidence,
            angle_from_rule=angle_deg,
            dot_product=dot,
        )
        self._classifications.append(classification)

        if confidence < 0.5:
            logger.warning(
                "%s: low confidence division (%.2f, angle=%.1f deg, avg_frames=%d)",
                parent_name, confidence, angle_deg, n_frames,
            )

        return name1, name2

    def _classify_division(
        self,
        parent: Nucleus,
        daughter1: Nucleus,
        daughter2: Nucleus,
        rule: Rule,
    ) -> DivisionClassification:
        """Classify a division with confidence scoring."""
        parent_name = parent.effective_name
        diff = self._diffs_corrected(daughter1, daughter2)

        # Dot product with rule axis
        dot = float(np.dot(diff, rule.axis_vector))

        # Compute angle from rule vector
        diff_norm = np.linalg.norm(diff)
        rule_norm = np.linalg.norm(rule.axis_vector)

        if diff_norm > 1e-6 and rule_norm > 1e-6:
            cos_angle = abs(dot) / (diff_norm * rule_norm)
            cos_angle = min(1.0, cos_angle)  # Clamp for numerical safety
            angle_deg = math.degrees(math.acos(cos_angle))
        else:
            angle_deg = 90.0

        confidence = _angle_to_confidence(angle_deg)

        # Determine names based on dot product sign
        if dot >= 0:
            name1, name2 = rule.daughter1, rule.daughter2
        else:
            name1, name2 = rule.daughter2, rule.daughter1

        if confidence < 0.5:
            logger.debug(
                "%s: moderate confidence (%.2f, angle=%.1f deg from rule)",
                parent_name, confidence, angle_deg,
            )

        return DivisionClassification(
            parent_name=parent_name,
            daughter1_name=name1,
            daughter2_name=name2,
            axis_used=rule.sulston_letter,
            confidence=confidence,
            angle_from_rule=angle_deg,
            dot_product=dot,
        )

    def _compute_averaged_diff(
        self,
        daughter1: Nucleus,
        daughter2: Nucleus,
        nuclei_record: list[list[Nucleus]],
        division_time: int,
        n_frames: int,
    ) -> tuple[np.ndarray | None, float]:
        """Compute averaged division vector over multiple frames.

        Returns:
            (averaged_corrected_diff, consistency_confidence) or (None, 0)
        """
        vectors = []

        # First frame: use the provided nuclei directly
        diff0 = self._diffs_corrected(daughter1, daughter2)
        diff0_norm = np.linalg.norm(diff0)
        if diff0_norm > 1e-6:
            vectors.append(diff0 / diff0_norm)

        # Subsequent frames: find the same cells by tracking successors
        d1_current = daughter1
        d2_current = daughter2

        for dt in range(1, n_frames):
            t = division_time + dt
            if t >= len(nuclei_record):
                break

            next_nuclei = nuclei_record[t]

            # Follow successor1 for both daughters
            d1_next = _follow_successor(d1_current, next_nuclei)
            d2_next = _follow_successor(d2_current, next_nuclei)

            if d1_next is None or d2_next is None:
                break  # One of the daughters divided or died

            diff = self._diffs_corrected(d1_next, d2_next)
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-6:
                vectors.append(diff / diff_norm)

            d1_current = d1_next
            d2_current = d2_next

        if not vectors:
            return None, 0.0

        # Average the unit vectors
        avg = np.mean(vectors, axis=0)
        avg_norm = np.linalg.norm(avg)

        # Consistency: if all vectors point the same way, avg_norm ≈ 1.0
        consistency = avg_norm

        if avg_norm > 1e-6:
            avg = avg / avg_norm

        # Scale back to approximate physical magnitude using first frame
        avg = avg * diff0_norm

        return avg, consistency

    def _compute_dot_product(
        self,
        parent: Nucleus,
        daughter1: Nucleus,
        daughter2: Nucleus,
        rule: Rule,
    ) -> float:
        """Compute the dot product of the corrected division vector with the rule axis.

        This is the core computation that determines which daughter gets which name.
        """
        # Compute corrected difference vector
        diff = self._diffs_corrected(daughter1, daughter2)

        # Dot with rule axis vector
        return float(np.dot(diff, rule.axis_vector))

    def _diffs_corrected(
        self,
        daughter1: Nucleus,
        daughter2: Nucleus,
    ) -> np.ndarray:
        """Compute the corrected division vector (d2 - d1).

        Steps:
            1. Raw difference: d2.xyz - d1.xyz
            2. Z-scaling: multiply z component by z_pix_res
            3. Measurement correction (rotation to canonical frame)
            4. V1 only: axis sign-flipping
        """
        # Raw difference vector (daughter2 - daughter1)
        da = np.array([
            daughter2.x - daughter1.x,
            daughter2.y - daughter1.y,
            daughter2.z - daughter1.z,
        ], dtype=np.float64)

        # Z-scaling
        da[2] *= self.z_pix_res

        # Measurement correction: rotate to canonical frame
        da = self._measurement_correction(da)

        return da

    def _measurement_correction(self, da: np.ndarray) -> np.ndarray:
        """Apply the measurement correction (rotation to canonical frame).

        For v2: applies the CanonicalTransform rotation.
        For v1: applies angle rotation in XY plane, then axis sign-flipping.
        For founder mode: applies the founder-derived rotation.
        """
        if self.is_v2:
            # V2: use CanonicalTransform
            return self.canonical_transform.apply(da)  # type: ignore[union-attr]
        elif self.is_founder_mode:
            # Founder mode: project onto founder axes
            return self._apply_founder_transform(da)
        else:
            # V1: angle rotation in XY plane
            da = self._handle_rotation_v1(da)
            # V1: axis sign-flipping based on axis string
            if self._v1_sign_matrix is not None:
                da = self._v1_sign_matrix @ da
            return da

    def _apply_founder_transform(self, da: np.ndarray) -> np.ndarray:
        """Transform a vector into canonical frame using founder-derived axes.

        The founder axes form an orthonormal basis. We project the vector
        onto this basis and map:
            AP projection -> canonical x (sign: AP points in -x direction)
            DV projection -> canonical y
            LR projection -> canonical z
        """
        if self.founder_ap is None or self.founder_lr is None or self.founder_dv is None:
            return da

        # Project onto founder basis
        ap_component = np.dot(da, self.founder_ap)
        lr_component = np.dot(da, self.founder_lr)
        dv_component = np.dot(da, self.founder_dv)

        # Map to canonical frame:
        # Canonical AP is [-1, 0, 0], so AP component -> -x
        # Canonical DV is [0, 1, 0], so DV component -> y
        # Canonical LR is [0, 0, 1], so LR component -> z
        return np.array([-ap_component, dv_component, lr_component])

    def _handle_rotation_v1(self, da: np.ndarray) -> np.ndarray:
        """Apply v1 angle rotation in the XY plane.

        Rotates the vector by -angle degrees in the XY plane.
        Matches Java DivisionCaller.handleRotation_V1().
        """
        if self.angle == 0.0:
            return da

        angle_rad = math.radians(-self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        x_new = da[0] * cos_a - da[1] * sin_a
        y_new = da[0] * sin_a + da[1] * cos_a
        da[0] = x_new
        da[1] = y_new

        return da


def _follow_successor(nuc: Nucleus, next_nuclei: list[Nucleus]) -> Nucleus | None:
    """Follow a nucleus to its successor in the next timepoint.

    Only follows non-dividing successors (successor1 only, successor2 == NILLI).
    Returns None if the cell divided, died, or has invalid links.
    """
    if nuc.successor1 == NILLI:
        return None

    if nuc.successor2 != NILLI:
        return None  # Cell is dividing

    s1_idx = nuc.successor1 - 1  # 1-based to 0-based
    if 0 <= s1_idx < len(next_nuclei):
        return next_nuclei[s1_idx]

    return None


def _angle_to_confidence(angle_deg: float) -> float:
    """Convert angle deviation from rule to a confidence score.

    Args:
        angle_deg: Angle in degrees between division vector and rule axis.

    Returns:
        Confidence score between 0 and 1.
    """
    if angle_deg <= HIGH_CONFIDENCE_ANGLE:
        return 1.0
    elif angle_deg <= LOW_CONFIDENCE_ANGLE:
        # Linear interpolation from 1.0 to 0.5
        t = (angle_deg - HIGH_CONFIDENCE_ANGLE) / (LOW_CONFIDENCE_ANGLE - HIGH_CONFIDENCE_ANGLE)
        return 1.0 - 0.5 * t
    elif angle_deg <= RULE_OVERRIDE_ANGLE:
        # Linear interpolation from 0.5 to 0.2
        t = (angle_deg - LOW_CONFIDENCE_ANGLE) / (RULE_OVERRIDE_ANGLE - LOW_CONFIDENCE_ANGLE)
        return 0.5 - 0.3 * t
    else:
        return max(0.1, 0.2 - (angle_deg - RULE_OVERRIDE_ANGLE) / 180.0)
