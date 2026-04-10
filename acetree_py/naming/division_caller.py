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
from .lineage_axes import axes_to_canonical, compute_local_axes
from .rules import Rule, RuleManager

logger = logging.getLogger(__name__)

# Confidence thresholds for division classification
HIGH_CONFIDENCE_ANGLE = 20.0    # degrees — very confident
LOW_CONFIDENCE_ANGLE = 40.0     # degrees — still trust the rule but uncertain
RULE_OVERRIDE_ANGLE = 55.0      # degrees — use observed dominant axis instead

# Number of frames to average division vector over
DEFAULT_AVG_FRAMES = 3

# LR axis quality thresholds
LR_QUALITY_THRESHOLD = 0.15  # Below this, LR axis is degenerate
_LR_HISTORY_MAX = 20         # Max entries in LR smoothing buffer
_LR_CONTINUITY_GAP = 3       # Max gap for continuity-based sign correction

# Deferred naming: re-evaluate with look-ahead when confidence is below this
_DEFERRED_CONFIDENCE_THRESHOLD = 0.3
_DEFERRED_LOOKAHEAD_FRAMES = 8  # Max frames to look ahead


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

    Supports four modes:
        - v2 mode: Uses CanonicalTransform (from AuxInfo v2 orientation vectors)
        - v1 mode: Uses axis string + angle rotation (from AuxInfo v1)
        - lineage mode: Per-timepoint axes from lineage centroids (rotation-invariant)
        - founder mode: Uses static axes derived from founder cell positions (legacy)
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
        lineage_map: list[list[str]] | None = None,
        nuclei_record: list[list] | None = None,
        seed_ap: np.ndarray | None = None,
        seed_lr: np.ndarray | None = None,
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
            lineage_map: For lineage mode — per-nucleus lineage labels
                (output of build_lineage_map).
            nuclei_record: For lineage mode — full nuclei record (needed
                to compute per-timepoint axes).
            seed_ap: Initial AP axis direction for sign anchoring
                (typically from 4-cell midpoint).
            seed_lr: Initial LR axis direction for sign anchoring.
        """
        self.rule_manager = rule_manager
        self.z_pix_res = z_pix_res
        self.canonical_transform = canonical_transform
        self.axis_string = axis_string
        self.angle = angle

        # Founder-derived axes (static, legacy)
        self.founder_ap = founder_ap
        self.founder_lr = founder_lr
        self.founder_dv = founder_dv

        # Lineage mode data
        self._lineage_map = lineage_map
        self._nuclei_record = nuclei_record

        # Cache for per-timepoint axes in lineage mode
        self._axes_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Anchor-based sign convention and temporal smoothing for LR.
        # Seed from the 4-cell midpoint axes for a reliable starting sign.
        self._lr_anchor: np.ndarray | None = (
            seed_lr.copy() if seed_lr is not None else None
        )
        self._lr_anchor_time: int = 0 if seed_lr is not None else -1
        self._ap_anchor: np.ndarray | None = (
            seed_ap.copy() if seed_ap is not None else None
        )
        self._lr_history: list[tuple[int, np.ndarray, float]] = []

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
    def is_lineage_mode(self) -> bool:
        """True if using per-timepoint lineage centroid axes."""
        return (
            self._lineage_map is not None
            and self._nuclei_record is not None
            and not self.is_v2
            and not self.axis_string
        )

    @property
    def is_founder_mode(self) -> bool:
        """True if using static founder-derived axes."""
        return (
            self.founder_ap is not None
            and self.founder_lr is not None
            and not self.is_v2
            and not self.axis_string
            and not self.is_lineage_mode
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
        timepoint: int = -1,
        nuclei_record: list[list] | None = None,
    ) -> tuple[str, str]:
        """Assign Sulston names to the two daughters of a dividing cell.

        When the initial classification has very low confidence (high angle
        from rule) and a *nuclei_record* is provided, looks ahead several
        frames to re-evaluate with more separated daughters and potentially
        recovered axes.

        Args:
            parent: The dividing parent nucleus.
            daughter1: First daughter nucleus (successor1).
            daughter2: Second daughter nucleus (successor2).
            timepoint: 0-based timepoint of the daughters (for lineage mode).
            nuclei_record: Full nuclei record for look-ahead re-evaluation.

        Returns:
            (name1, name2) — the names to assign to daughter1 and daughter2.
        """
        parent_name = parent.effective_name
        if not parent_name:
            return "", ""

        rule = self.rule_manager.get_rule(parent_name)
        classification = self._classify_division(
            parent, daughter1, daughter2, rule, timepoint=timepoint,
        )

        # If confidence is very low and we can look ahead, re-evaluate
        # using majority vote across future frames
        if (classification.confidence < _DEFERRED_CONFIDENCE_THRESHOLD
                and nuclei_record is not None
                and timepoint >= 0):
            deferred = self._deferred_evaluate(
                daughter1, daughter2, rule, nuclei_record, timepoint,
            )
            if deferred is not None and deferred.confidence >= classification.confidence:
                classification = deferred

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
            return self.assign_names(parent, daughter1, daughter2, timepoint=division_time)

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

    def _deferred_evaluate(
        self,
        daughter1: Nucleus,
        daughter2: Nucleus,
        rule: Rule,
        nuclei_record: list[list[Nucleus]],
        division_time: int,
    ) -> DivisionClassification | None:
        """Re-evaluate a division using later frames when initial confidence is low.

        Follows both daughters forward up to ``_DEFERRED_LOOKAHEAD_FRAMES``
        frames, re-classifying at each step.  Uses majority vote across
        all look-ahead frames to decide the assignment, which is more robust
        than any single frame when angles are near 90 degrees.
        """
        votes_positive = 0  # dot >= 0
        votes_negative = 0  # dot < 0
        best_confidence = 0.0
        d1_cur, d2_cur = daughter1, daughter2

        for dt in range(1, _DEFERRED_LOOKAHEAD_FRAMES + 1):
            t = division_time + dt
            if t >= len(nuclei_record):
                break

            d1_next = _follow_successor(d1_cur, nuclei_record[t])
            d2_next = _follow_successor(d2_cur, nuclei_record[t])
            if d1_next is None or d2_next is None:
                break

            cls = self._classify_division(
                daughter1, d1_next, d2_next, rule, timepoint=t,
            )

            if cls.dot_product >= 0:
                votes_positive += 1
            else:
                votes_negative += 1

            best_confidence = max(best_confidence, cls.confidence)
            d1_cur, d2_cur = d1_next, d2_next

        total_votes = votes_positive + votes_negative
        if total_votes == 0:
            return None

        # Majority vote determines assignment
        if votes_positive >= votes_negative:
            name1, name2 = rule.daughter1, rule.daughter2
        else:
            name1, name2 = rule.daughter2, rule.daughter1

        # Confidence is boosted by vote margin
        margin = abs(votes_positive - votes_negative) / total_votes
        vote_confidence = max(best_confidence, margin)

        return DivisionClassification(
            parent_name=daughter1.effective_name or rule.daughter1,
            daughter1_name=name1,
            daughter2_name=name2,
            axis_used=rule.sulston_letter,
            confidence=vote_confidence,
            angle_from_rule=0.0,  # not meaningful for vote
            dot_product=float(votes_positive - votes_negative),
        )

    def _get_local_axes(self, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Get body axes at timepoint t (lineage mode only).

        Uses a two-tier sign correction strategy:
        1. If a nearby (within ``_LR_CONTINUITY_GAP``) cached frame exists,
           use continuity correction against it (works well when cache is dense).
        2. Otherwise, fall back to the seed anchor from the 4-cell stage.

        When the LR axis quality is below ``LR_QUALITY_THRESHOLD``, substitutes
        a temporally smoothed LR from recent high-quality frames.
        """
        if t in self._axes_cache:
            return self._axes_cache[t]

        ap, lr, dv, lr_quality = compute_local_axes(
            self._nuclei_record, self._lineage_map, t, self.z_pix_res,
        )
        if ap is None or lr is None or dv is None:
            return None

        # --- AP sign consistency against nearby cache ---
        if self._axes_cache:
            candidates = [k for k in self._axes_cache if k < t]
            if candidates:
                prev_t = max(candidates)
                if t - prev_t <= _LR_CONTINUITY_GAP:
                    if np.dot(ap, self._axes_cache[prev_t][0]) < 0:
                        ap = -ap
                        dv = -dv

        # --- LR handling depends on quality ---
        if lr_quality < LR_QUALITY_THRESHOLD:
            # Low quality: prefer temporal smoothing over noisy fresh value
            smoothed = self._get_smoothed_lr(t)
            if smoothed is not None:
                lr = smoothed
                dv = np.cross(ap, lr)
                dv_norm = np.linalg.norm(dv)
                if dv_norm > 1e-6:
                    dv = dv / dv_norm
                else:
                    return None
            else:
                # No smoothing history — use any previous cache for sign
                lr, dv = self._correct_lr_sign(lr, dv, t, lr_quality)
        else:
            # High quality: only correct sign against nearby cache
            lr, dv = self._correct_lr_sign(lr, dv, t, lr_quality)

            self._lr_history.append((t, lr.copy(), lr_quality))
            if len(self._lr_history) > _LR_HISTORY_MAX:
                self._lr_history.pop(0)

        result = (ap, lr, dv)
        self._axes_cache[t] = result
        return result

    def _correct_lr_sign(
        self, lr: np.ndarray, dv: np.ndarray, t: int,
        lr_quality: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct LR sign using cached reference.

        The correction strategy depends on LR quality:
        - High quality (>= threshold): only correct against nearby cache
          (within ``_LR_CONTINUITY_GAP``) — trust the fresh computation
          when no nearby reference exists.
        - Low quality (< threshold): correct against ANY previous cache
          entry, since the fresh value is unreliable noise.
        """
        if not self._axes_cache:
            return lr, dv

        candidates = [k for k in self._axes_cache if k < t]
        if not candidates:
            return lr, dv

        prev_t = max(candidates)

        if lr_quality >= LR_QUALITY_THRESHOLD:
            # High quality: only use nearby reference
            if t - prev_t > _LR_CONTINUITY_GAP:
                return lr, dv
        # Low quality or nearby cache: apply correction
        reference_lr = self._axes_cache[prev_t][1]
        if np.dot(lr, reference_lr) < 0:
            lr = -lr
            dv = -dv

        return lr, dv

    def _get_smoothed_lr(self, t: int) -> np.ndarray | None:
        """Get a temporally smoothed LR axis from recent high-quality frames.

        Uses exponential weighting by recency and quality.
        """
        if not self._lr_history:
            return None

        weighted_sum = np.zeros(3)
        total_weight = 0.0

        for hist_t, hist_lr, hist_quality in self._lr_history:
            dt = abs(t - hist_t)
            # Exponential decay: half-life of 10 timepoints
            recency_weight = 0.5 ** (dt / 10.0)
            weight = recency_weight * hist_quality
            weighted_sum += weight * hist_lr
            total_weight += weight

        if total_weight < 1e-6:
            return None

        smoothed = weighted_sum / total_weight
        smoothed_norm = np.linalg.norm(smoothed)
        if smoothed_norm < 1e-6:
            return None

        return smoothed / smoothed_norm

    def _classify_division(
        self,
        parent: Nucleus,
        daughter1: Nucleus,
        daughter2: Nucleus,
        rule: Rule,
        timepoint: int = -1,
    ) -> DivisionClassification:
        """Classify a division with confidence scoring."""
        parent_name = parent.effective_name
        diff = self._diffs_corrected(daughter1, daughter2, timepoint=timepoint)

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

        Averages raw lab-frame vectors (z-scaled only), then transforms
        into canonical frame ONCE using the division-time axes.  This
        avoids mixing coordinate systems when axes change between frames.

        Returns:
            (averaged_corrected_diff, consistency_confidence) or (None, 0)
        """
        lab_vectors = []

        # First frame: raw z-scaled vector
        raw0 = np.array([
            daughter2.x - daughter1.x,
            daughter2.y - daughter1.y,
            (daughter2.z - daughter1.z) * self.z_pix_res,
        ], dtype=np.float64)
        raw0_norm = np.linalg.norm(raw0)
        if raw0_norm > 1e-6:
            lab_vectors.append(raw0 / raw0_norm)

        # Subsequent frames: follow successors, collect raw lab-frame vectors
        d1_current = daughter1
        d2_current = daughter2

        for dt in range(1, n_frames):
            t = division_time + dt
            if t >= len(nuclei_record):
                break

            next_nuclei = nuclei_record[t]

            d1_next = _follow_successor(d1_current, next_nuclei)
            d2_next = _follow_successor(d2_current, next_nuclei)

            if d1_next is None or d2_next is None:
                break  # One of the daughters divided or died

            raw = np.array([
                d2_next.x - d1_next.x,
                d2_next.y - d1_next.y,
                (d2_next.z - d1_next.z) * self.z_pix_res,
            ], dtype=np.float64)
            raw_norm = np.linalg.norm(raw)
            if raw_norm > 1e-6:
                lab_vectors.append(raw / raw_norm)

            d1_current = d1_next
            d2_current = d2_next

        if not lab_vectors:
            return None, 0.0

        # Average the unit vectors in lab frame
        avg_lab = np.mean(lab_vectors, axis=0)
        avg_norm = np.linalg.norm(avg_lab)

        # Consistency: if all vectors point the same way, avg_norm ~ 1.0
        consistency = avg_norm

        if avg_norm > 1e-6:
            avg_lab = avg_lab / avg_norm

        # Scale to first-frame magnitude
        avg_lab = avg_lab * raw0_norm

        # Transform ONCE using division-time axes
        avg_canonical = self._measurement_correction(avg_lab, timepoint=division_time)

        return avg_canonical, consistency

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
        timepoint: int = -1,
    ) -> np.ndarray:
        """Compute the corrected division vector (d2 - d1).

        Steps:
            1. Raw difference: d2.xyz - d1.xyz
            2. Z-scaling: multiply z component by z_pix_res
            3. Measurement correction (rotation to canonical frame)
            4. V1 only: axis sign-flipping

        Args:
            daughter1, daughter2: The two daughter nuclei.
            timepoint: 0-based timepoint of the daughters (used by lineage
                mode for per-timepoint axis computation).
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
        da = self._measurement_correction(da, timepoint=timepoint)

        return da

    def _measurement_correction(
        self, da: np.ndarray, timepoint: int = -1,
    ) -> np.ndarray:
        """Apply the measurement correction (rotation to canonical frame).

        For v2: applies the CanonicalTransform rotation.
        For v1: applies angle rotation in XY plane, then axis sign-flipping.
        For lineage mode: computes axes from lineage centroids at *timepoint*.
        For founder mode: applies the static founder-derived rotation.
        """
        if self.is_v2:
            # V2: use CanonicalTransform
            return self.canonical_transform.apply(da)  # type: ignore[union-attr]
        elif self.is_lineage_mode:
            return self._apply_lineage_transform(da, timepoint)
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

    def _apply_lineage_transform(self, da: np.ndarray, timepoint: int) -> np.ndarray:
        """Transform a vector into canonical frame using lineage centroids.

        Computes body axes at *timepoint* from the spatial distribution
        of ABa-lineage vs ABp-lineage cells, making this inherently
        robust to global embryo rotations around the AP axis.
        """
        axes = self._get_local_axes(timepoint)
        if axes is None:
            # Fallback to static founder axes if available
            if self.founder_ap is not None and self.founder_lr is not None:
                return self._apply_founder_transform(da)
            return da  # No correction possible

        ap_vec, lr_vec, dv_vec = axes
        return axes_to_canonical(da, ap_vec, lr_vec, dv_vec)

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
