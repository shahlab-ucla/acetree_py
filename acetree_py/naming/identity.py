"""Identity assignment pipeline — orchestrates the full naming system.

This module ties together InitialID (legacy) or FounderID (new topology-based),
DivisionCaller, CanonicalTransform, and Validation to assign Sulston names
to all nuclei in the dataset.

Ported from: org.rhwlab.snight.Identity3

Pipeline:
  1. Clear all non-forced names
  2. Determine embryo axes (from AuxInfo, or from founder cell positions)
  3. Identify early cells (P0, AB, P1, EMS, P2)
  4. Run canonical rules to name all divisions
  5. Validate naming consistency
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.nucleus import NILLI, Nucleus
from ..io.auxinfo import AuxInfo
from .canonical_transform import CanonicalTransform, TransformValidationError
from .division_caller import DivisionCaller, DivisionClassification
from .founder_id import FounderAssignment, identify_founders
from .initial_id import NUC, identify_initial_cells
from .lineage_axes import build_lineage_map
from .rules import RuleManager
from .validation import NamingWarning, validate_naming

logger = logging.getLogger(__name__)

# Naming method constants (matches NamingMethod enum in io/config.py)
MANUAL = 2
NEWCANONICAL = 3


class IdentityAssigner:
    """Orchestrates the full naming pipeline.

    Usage:
        assigner = IdentityAssigner(nuclei_record, auxinfo, ...)
        assigner.assign_identities()

    The pipeline supports two identification strategies:
      1. Legacy (InitialID): Uses cardinal-direction diamond pattern alignment.
         Requires AuxInfo with orientation data. Matches original Java behavior.
      2. Topology-based (FounderID): Uses division timing and topology only.
         Rotation-invariant; does not require AuxInfo.

    The topology-based approach is tried first. If it fails or produces
    low confidence, the legacy approach is used as fallback.
    """

    def __init__(
        self,
        nuclei_record: list[list[Nucleus]],
        auxinfo: AuxInfo | None = None,
        naming_method: int = NEWCANONICAL,
        starting_index: int = 0,
        ending_index: int = -1,
        z_pix_res: float = 11.1,
        use_multi_frame: bool = True,
        legacy_mode: bool = False,
    ) -> None:
        """Initialize the identity assigner.

        Args:
            nuclei_record: The full nuclei record.
            auxinfo: AuxInfo data (v1 or v2).
            naming_method: MANUAL (2) or NEWCANONICAL (3).
            starting_index: 0-based starting timepoint.
            ending_index: Ending timepoint (-1 for all).
            z_pix_res: Z pixel resolution.
            use_multi_frame: If True, use multi-frame division vector averaging.
            legacy_mode: If True, use the legacy InitialID fallback pipeline
                instead of the unified topology-based pipeline.  For backward
                compatibility testing only.
        """
        self.nuclei_record = nuclei_record
        self.auxinfo = auxinfo
        self.naming_method = naming_method
        self.starting_index = starting_index
        self.ending_index = ending_index if ending_index >= 0 else len(nuclei_record)
        self.z_pix_res = z_pix_res
        self.use_multi_frame = use_multi_frame
        self.legacy_mode = legacy_mode

        self.canonical_transform: CanonicalTransform | None = None
        self.rule_manager = RuleManager()
        self.division_caller: DivisionCaller | None = None
        self.founder_assignment: FounderAssignment | None = None
        self.warnings: list[NamingWarning] = []

    def assign_identities(self) -> None:
        """Run the full naming pipeline.

        This is the main entry point, corresponding to Identity3.identityAssignment().

        The unified pipeline uses topology-based founder identification with
        per-timepoint lineage centroid axes (rotation-invariant).  If AuxInfo
        is available, it is used for cross-validation diagnostics only.

        Set ``legacy_mode=True`` to use the original InitialID fallback
        pipeline for backward compatibility testing.
        """
        if self.naming_method == MANUAL:
            logger.info("Skipping naming due to MANUAL naming method")
            return

        if self.legacy_mode:
            self._clear_all_names()
            if self.auxinfo is not None and self.auxinfo.is_v2:
                self._build_canonical_transform()
            self._run_legacy_pipeline()
            return

        # Step 1: Clear all non-forced names
        self._clear_all_names()

        # Step 1b: Propagate forced names through successor/predecessor chains
        self._propagate_assigned_ids()

        # Step 2: Build CanonicalTransform if v2 (for cross-validation only)
        if self.auxinfo is not None and self.auxinfo.is_v2:
            self._build_canonical_transform()

        # Step 3: Topology-based identification (unified default)
        self.founder_assignment = identify_founders(
            self.nuclei_record,
            starting_index=self.starting_index,
            ending_index=self.ending_index,
            z_pix_res=self.z_pix_res,
        )

        if self.founder_assignment.success and self.founder_assignment.confidence >= 0.3:
            logger.info(
                "Topology-based founder ID succeeded (confidence=%.2f: "
                "timing=%.2f, size=%.2f, axis=%.2f)",
                self.founder_assignment.confidence,
                self.founder_assignment.timing_confidence,
                self.founder_assignment.size_confidence,
                self.founder_assignment.axis_confidence,
            )
            self._setup_division_caller_from_founders()
            if self.division_caller is not None:
                # Start canonical rules from the 4-cell midpoint, NOT from
                # start_index.  The back-trace already correctly named
                # everything between start_index and four_cell_time (P0, AB,
                # P1, ABa, ABp, EMS, P2 and their continuation cells).
                # Starting earlier would cause the forward pass to overwrite
                # those names via DivisionCaller, which fails for early cells
                # like P0 that have no precomputed rule.
                self._use_canonical_rules(self.founder_assignment.four_cell_time)
                self._cross_validate_with_auxinfo()
                return

        # Step 4: Founder ID failed — provide diagnostics instead of
        # silently falling back to the weaker legacy algorithm
        fa = self.founder_assignment
        logger.warning(
            "Topology-based founder identification failed "
            "(success=%s, confidence=%.2f, warnings=%s). "
            "Falling back to generic naming. Use legacy_mode=True "
            "to try the AuxInfo-dependent pipeline.",
            fa.success, fa.confidence, fa.warnings,
        )
        self._assign_generic_names(self.starting_index)

    def _run_legacy_pipeline(self) -> None:
        """Run the legacy InitialID-based pipeline."""
        import math

        angle_rad = 0.0
        axis_string = ""
        if self.auxinfo is not None:
            angle_rad = math.radians(-self.auxinfo.angle)
            axis_string = self.auxinfo.axis or ""

        result = identify_initial_cells(
            self.nuclei_record,
            starting_index=self.starting_index,
            ending_index=self.ending_index,
            canonical_transform=self.canonical_transform,
            angle=angle_rad,
            z_pix_res=self.z_pix_res,
        )

        # If axis found and NEWCANONICAL, use canonical rules
        if result.axis_found and self.naming_method == NEWCANONICAL:
            orientation = ""
            if self.auxinfo is not None and not self.auxinfo.is_v2:
                orientation = _compute_orientation(result.ap, result.dv, result.lr)

            self._setup_division_caller(orientation)
            self._use_canonical_rules(result.start_index)
            return

        # Fallback — assign generic Nuc names
        logger.info("No axis found or not NEWCANONICAL; assigning generic names")
        self._assign_generic_names(result.start_index)

    def _build_canonical_transform(self) -> None:
        """Build the CanonicalTransform from AuxInfo v2 orientation vectors."""
        if self.auxinfo is None:
            return

        ap_vec = self.auxinfo.ap_orientation
        lr_vec = self.auxinfo.lr_orientation

        if ap_vec is None or lr_vec is None:
            logger.warning("AuxInfo v2 but no orientation vectors; skipping transform")
            return

        try:
            self.canonical_transform = CanonicalTransform(ap_vec, lr_vec)
            logger.info("CanonicalTransform built successfully")
        except TransformValidationError as e:
            logger.warning("CanonicalTransform failed: %s; falling back to v1", e)
            self.canonical_transform = None

    def _cross_validate_with_auxinfo(self) -> None:
        """Cross-validate lineage centroid axes against AuxInfo if available.

        This is a diagnostic method — it compares the per-timepoint lineage
        centroid axes with AuxInfo-derived axes at a few sample timepoints
        and logs any disagreement. It does NOT change the division caller.
        """
        if self.auxinfo is None or self.division_caller is None:
            return
        if not self.division_caller.is_lineage_mode:
            return

        fa = self.founder_assignment
        if fa is None:
            return

        # Pick a few sample timepoints after the 4-cell stage
        sample_times = []
        start = fa.four_cell_time + 5
        end = min(self.ending_index, fa.four_cell_time + 50)
        for t in range(start, end, 10):
            sample_times.append(t)

        if not sample_times:
            return

        # Get AuxInfo-derived AP direction for comparison
        auxinfo_ap = None
        if self.auxinfo.is_v2 and self.canonical_transform is not None:
            # In v2, canonical_transform maps lab -> canonical.
            # AP in canonical is [-1, 0, 0], so lab AP = inverse(transform) @ [-1,0,0]
            # For comparison, we just check angle agreement, not exact direction.
            auxinfo_ap = self.canonical_transform.apply(np.array([-1.0, 0.0, 0.0]))
        elif not self.auxinfo.is_v2 and fa.ap_vector is not None:
            # v1: use the founder-derived AP as a proxy for "AuxInfo-informed" AP
            auxinfo_ap = fa.ap_vector

        if auxinfo_ap is None:
            return

        angles = []
        for t in sample_times:
            axes = self.division_caller._get_local_axes(t)
            if axes is None:
                continue
            lineage_ap = axes[0]
            cos_angle = np.clip(np.dot(lineage_ap, auxinfo_ap), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(abs(cos_angle)))
            angles.append(angle_deg)

        if angles:
            mean_angle = np.mean(angles)
            max_angle = max(angles)
            if mean_angle > 30:
                logger.warning(
                    "Lineage centroid axes disagree with AuxInfo "
                    "(mean AP angle=%.1f deg, max=%.1f deg at %d sample points). "
                    "This may indicate an issue with lineage tracking.",
                    mean_angle, max_angle, len(angles),
                )
            else:
                logger.info(
                    "Lineage centroid axes agree with AuxInfo "
                    "(mean AP angle=%.1f deg, max=%.1f deg at %d sample points)",
                    mean_angle, max_angle, len(angles),
                )

    def _setup_division_caller(self, orientation: str) -> None:
        """Create the DivisionCaller with v1/v2 settings (legacy path)."""
        angle = 0.0
        if self.auxinfo is not None:
            angle = self.auxinfo.angle

        self.division_caller = DivisionCaller(
            rule_manager=self.rule_manager,
            z_pix_res=self.z_pix_res,
            canonical_transform=self.canonical_transform,
            axis_string=orientation,
            angle=angle,
        )

    def _setup_division_caller_from_founders(self) -> None:
        """Create the DivisionCaller using per-timepoint lineage centroid axes.

        Always uses the rotation-invariant lineage centroid approach as the
        primary axis source. AuxInfo (v1 or v2) is used for cross-validation
        diagnostics only (see ``_cross_validate_with_auxinfo``).
        """
        fa = self.founder_assignment
        if fa is None:
            return

        # Always use per-timepoint lineage centroid axes (rotation-invariant).
        lineage_map = build_lineage_map(
            self.nuclei_record,
            four_cell_time=fa.four_cell_time,
            aba_idx=fa.aba_idx,
            abp_idx=fa.abp_idx,
            ems_idx=fa.ems_idx,
            p2_idx=fa.p2_idx,
        )

        # Compute axes at the 4-cell midpoint to seed the sign anchor.
        # This is typically the highest-quality frame for LR because ABa
        # and ABp are maximally separated at this stage.
        from .lineage_axes import compute_local_axes
        seed_ap, seed_lr, seed_dv, seed_q = compute_local_axes(
            self.nuclei_record, lineage_map, fa.four_cell_time, self.z_pix_res,
        )

        self.division_caller = DivisionCaller(
            rule_manager=self.rule_manager,
            z_pix_res=self.z_pix_res,
            lineage_map=lineage_map,
            nuclei_record=self.nuclei_record,
            seed_ap=seed_ap,
            seed_lr=seed_lr,
        )
        # Disable multi-frame averaging in lineage mode.  With per-timepoint
        # axes that may differ between frames, averaging the division vector
        # across frames blurs the signal.  Single-frame classification with
        # quality-aware axis smoothing gives better results empirically.
        self.use_multi_frame = False
        logger.info("Using per-timepoint lineage centroid axes (rotation-invariant)")

    def _clear_all_names(self) -> None:
        """Clear all non-forced names in the nuclei record.

        Names set via assigned_id are preserved (forced names survive renaming).
        """
        for t in range(self.starting_index, min(self.ending_index, len(self.nuclei_record))):
            for nuc in self.nuclei_record[t]:
                if nuc.assigned_id:
                    continue
                nuc.identity = ""

    def _propagate_assigned_ids(self) -> None:
        """Propagate forced names (assigned_id) through continuation chains.

        When a user renames a cell at a single timepoint, this method
        extends that forced name to every timepoint the cell exists:
        forward through successor1 (non-dividing continuations) and
        backward through the predecessor chain.

        This ensures that:
        1. The forced name is visible at all timepoints the cell exists.
        2. When the cell eventually divides, the division caller uses the
           forced name as the parent name for daughter naming.
        """
        nr = self.nuclei_record
        n_times = min(len(nr), self.ending_index)

        # Collect all (t, j) with assigned_id set
        seeds: list[tuple[int, int, str]] = []
        for t in range(self.starting_index, n_times):
            for j, nuc in enumerate(nr[t]):
                if nuc.assigned_id:
                    seeds.append((t, j, nuc.assigned_id))

        for seed_t, seed_j, forced_name in seeds:
            # Forward: follow successor1 chain (non-dividing only)
            t, idx = seed_t, seed_j
            while t + 1 < n_times:
                nuc = nr[t][idx]
                if nuc.successor1 <= 0:
                    break
                # Stop at divisions — daughters get names from division rules
                if nuc.successor2 > 0:
                    break
                s_idx = nuc.successor1 - 1
                if not (0 <= s_idx < len(nr[t + 1])):
                    break
                succ = nr[t + 1][s_idx]
                if succ.assigned_id and succ.assigned_id != forced_name:
                    break  # Different forced name — don't overwrite
                succ.assigned_id = forced_name
                succ.identity = forced_name
                t, idx = t + 1, s_idx

            # Backward: follow predecessor chain
            t, idx = seed_t, seed_j
            while t > self.starting_index:
                nuc = nr[t][idx]
                if nuc.predecessor == NILLI or nuc.predecessor <= 0:
                    break
                p_idx = nuc.predecessor - 1
                if not (0 <= p_idx < len(nr[t - 1])):
                    break
                pred = nr[t - 1][p_idx]
                # Stop if predecessor is dividing (has two successors) —
                # this cell is a daughter, not a continuation
                if pred.successor2 > 0:
                    break
                if pred.assigned_id and pred.assigned_id != forced_name:
                    break  # Different forced name — don't overwrite
                pred.assigned_id = forced_name
                pred.identity = forced_name
                t, idx = t - 1, p_idx

        if seeds:
            logger.info("Propagated %d forced name(s) through continuation chains", len(seeds))

    def _use_canonical_rules(self, start_index: int) -> None:
        """Apply canonical naming rules to all timepoints.

        For each timepoint and each nucleus:
        - Unnamed nuclei get a generated name (Nuc_time_z_x_y)
        - Non-dividing nuclei inherit the parent's name
        - Dividing nuclei get daughter names via DivisionCaller

        Supports both single-frame and multi-frame division classification.

        Corresponds to Identity3.useCanonicalRules() in Java.
        """
        if self.division_caller is None:
            logger.error("DivisionCaller not initialized")
            return

        m = min(len(self.nuclei_record), self.ending_index)

        for i in range(start_index, m):
            nuclei = self.nuclei_record[i]
            next_nuclei = self.nuclei_record[i + 1] if i + 1 < m else None

            for parent in nuclei:
                if parent.status < 1:
                    continue

                pname = parent.identity

                # Assign generic name if unnamed
                if not pname:
                    if parent.assigned_id:
                        pname = parent.assigned_id
                    else:
                        z = round(parent.z)
                        pname = f"{NUC}{i + 1:03d}_{z}_{parent.x}_{parent.y}"
                    parent.identity = pname

                # Process successors in next timepoint
                if next_nuclei is None:
                    continue

                has_two_successors = (
                    parent.successor1 > 0 and parent.successor2 > 0
                )

                if not has_two_successors:
                    # Not dividing — extend name to successor
                    if parent.successor1 > 0:
                        s1_idx = parent.successor1 - 1
                        if 0 <= s1_idx < len(next_nuclei):
                            succ = next_nuclei[s1_idx]
                            if not succ.assigned_id:
                                succ.identity = pname
                    continue

                # Dividing — use DivisionCaller
                s1_idx = parent.successor1 - 1
                s2_idx = parent.successor2 - 1
                if not (0 <= s1_idx < len(next_nuclei) and 0 <= s2_idx < len(next_nuclei)):
                    continue

                dau1 = next_nuclei[s1_idx]
                dau2 = next_nuclei[s2_idx]

                # Assign names (single-frame or multi-frame)
                # division_time = i + 1 (0-based timepoint of the daughters)
                if self.use_multi_frame:
                    name1, name2 = self.division_caller.assign_names_multi_frame(
                        parent, dau1, dau2,
                        self.nuclei_record, i + 1,
                    )
                else:
                    name1, name2 = self.division_caller.assign_names(
                        parent, dau1, dau2, timepoint=i + 1,
                        nuclei_record=self.nuclei_record,
                    )

                dau1.identity = name1
                dau2.identity = name2

                # Honor forced names (assigned_id takes priority)
                _use_preassigned_id(dau1, dau2)

    def _assign_generic_names(self, start_index: int) -> None:
        """Assign generic names when canonical naming isn't available.

        Non-dividing cells inherit parent name. Dividing cells get
        parent + "a" / parent + "p" as a simple fallback.
        """
        for i in range(start_index, min(self.ending_index, len(self.nuclei_record))):
            nuclei = self.nuclei_record[i]
            prev_nuclei = self.nuclei_record[i - 1] if i > 0 else None

            for nuc in nuclei:
                if nuc.status < 1:
                    continue

                if nuc.identity:
                    continue  # already named

                if prev_nuclei is not None and nuc.predecessor != NILLI:
                    pred_idx = nuc.predecessor - 1
                    if 0 <= pred_idx < len(prev_nuclei):
                        pred = prev_nuclei[pred_idx]
                        if pred.successor2 == NILLI:
                            nuc.identity = pred.identity
                        else:
                            # Dividing — simple a/p naming
                            if nuc.assigned_id:
                                nuc.identity = nuc.assigned_id
                            else:
                                nuc.identity = pred.identity + "a"
                                # Name the sister too
                                s2_idx = pred.successor2 - 1
                                if 0 <= s2_idx < len(nuclei):
                                    sister = nuclei[s2_idx]
                                    if not sister.identity and not sister.assigned_id:
                                        sister.identity = pred.identity + "p"
                        continue

                # First encounter of unnamed nucleus
                if nuc.assigned_id:
                    nuc.identity = nuc.assigned_id
                else:
                    z = round(nuc.z)
                    nuc.identity = f"{NUC}{i + 1:03d}_{z}_{nuc.x}_{nuc.y}"


def _use_preassigned_id(dau1: Nucleus, dau2: Nucleus) -> None:
    """Honor forced names (assigned_id) on daughter cells.

    If a daughter has an assigned_id, override its identity with it.
    If both daughters end up with the same name, append 'X' to distinguish.
    """
    if not dau1.assigned_id and not dau2.assigned_id:
        return

    if dau1.assigned_id:
        dau1.identity = dau1.assigned_id
    if dau2.assigned_id:
        dau2.identity = dau2.assigned_id

    # Resolve naming collision
    if dau1.identity == dau2.identity:
        dau2.identity = dau2.identity[:-1] + "X"


def _compute_orientation(ap: int, dv: int, lr: int) -> str:
    """Compute the 3-character orientation string from axis signs.

    Args:
        ap: +1 for anterior, -1 for posterior
        dv: +1 for dorsal, -1 for ventral
        lr: +1 for left, -1 for right

    Returns:
        3-character string like "ADL", "AVR", etc.
    """
    orientation = "A" if ap >= 0 else "P"
    orientation += "D" if dv > 0 else "V"
    orientation += "L" if lr > 0 else "R"
    return orientation
