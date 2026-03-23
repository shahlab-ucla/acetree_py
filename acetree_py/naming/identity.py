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
        """
        self.nuclei_record = nuclei_record
        self.auxinfo = auxinfo
        self.naming_method = naming_method
        self.starting_index = starting_index
        self.ending_index = ending_index if ending_index >= 0 else len(nuclei_record)
        self.z_pix_res = z_pix_res
        self.use_multi_frame = use_multi_frame

        self.canonical_transform: CanonicalTransform | None = None
        self.rule_manager = RuleManager()
        self.division_caller: DivisionCaller | None = None
        self.founder_assignment: FounderAssignment | None = None
        self.warnings: list[NamingWarning] = []

    def assign_identities(self) -> None:
        """Run the full naming pipeline.

        This is the main entry point, corresponding to Identity3.identityAssignment().
        """
        if self.naming_method == MANUAL:
            logger.info("Skipping naming due to MANUAL naming method")
            return

        # Step 1: Clear all non-forced names
        self._clear_all_names()

        # Step 2: Build CanonicalTransform if v2
        if self.auxinfo is not None and self.auxinfo.is_v2:
            self._build_canonical_transform()

        # Step 3: Try topology-based identification first
        self.founder_assignment = identify_founders(
            self.nuclei_record,
            starting_index=self.starting_index,
            ending_index=self.ending_index,
            z_pix_res=self.z_pix_res,
        )

        if self.founder_assignment.success and self.founder_assignment.confidence >= 0.3:
            logger.info(
                "Using topology-based founder ID (confidence=%.2f)",
                self.founder_assignment.confidence,
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
                return

        # Step 4: Fallback to legacy InitialID
        logger.info("Falling back to legacy InitialID")
        self._clear_all_names()  # Re-clear since founder ID may have set some names
        self._run_legacy_pipeline()

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
        """Create the DivisionCaller using founder-derived axes.

        If AuxInfo v2 is available, use it instead (more precise).
        If AuxInfo v1 is available, use it as well.
        Otherwise, use the founder-derived axes.
        """
        fa = self.founder_assignment
        if fa is None:
            return

        # Prefer AuxInfo v2 if available (externally measured, more precise)
        if self.canonical_transform is not None and self.canonical_transform.active:
            self.division_caller = DivisionCaller(
                rule_manager=self.rule_manager,
                z_pix_res=self.z_pix_res,
                canonical_transform=self.canonical_transform,
            )
            logger.info("Using AuxInfo v2 transform with topology-based founder ID")
            return

        # Try AuxInfo v1
        if self.auxinfo is not None and not self.auxinfo.is_v2:
            import math
            orientation = ""
            axis_str = self.auxinfo.axis or ""
            if axis_str and axis_str != "XXX":
                orientation = axis_str
                self.division_caller = DivisionCaller(
                    rule_manager=self.rule_manager,
                    z_pix_res=self.z_pix_res,
                    axis_string=orientation,
                    angle=self.auxinfo.angle,
                )
                logger.info("Using AuxInfo v1 transform with topology-based founder ID")
                return

        # Use founder-derived axes
        if fa.ap_vector is not None and fa.lr_vector is not None:
            # Build a CanonicalTransform from founder axes
            try:
                ct = CanonicalTransform(fa.ap_vector, fa.lr_vector)
                self.division_caller = DivisionCaller(
                    rule_manager=self.rule_manager,
                    z_pix_res=self.z_pix_res,
                    canonical_transform=ct,
                )
                logger.info("Using founder-derived axes for division calling")
                return
            except TransformValidationError:
                # Founder axes not orthogonal enough for CanonicalTransform
                # Use direct projection instead
                dv = fa.dv_vector
                if dv is None:
                    dv = np.cross(fa.ap_vector, fa.lr_vector)
                    dv_norm = np.linalg.norm(dv)
                    if dv_norm > 1e-6:
                        dv = dv / dv_norm

                self.division_caller = DivisionCaller(
                    rule_manager=self.rule_manager,
                    z_pix_res=self.z_pix_res,
                    founder_ap=fa.ap_vector,
                    founder_lr=fa.lr_vector,
                    founder_dv=dv,
                )
                logger.info("Using founder-derived axes (direct projection mode)")
                return

        logger.warning("No axes available for division calling")

    def _clear_all_names(self) -> None:
        """Clear all non-forced names in the nuclei record.

        Names set via assigned_id are preserved (forced names survive renaming).
        """
        for t in range(self.starting_index, min(self.ending_index, len(self.nuclei_record))):
            for nuc in self.nuclei_record[t]:
                if nuc.assigned_id:
                    continue
                nuc.identity = ""

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
                        pname = f"{NUC}{i + 1:04d}_{z}_{parent.x}_{parent.y}"
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
                if self.use_multi_frame:
                    name1, name2 = self.division_caller.assign_names_multi_frame(
                        parent, dau1, dau2,
                        self.nuclei_record, i + 1,  # division_time is 0-based next timepoint
                    )
                else:
                    name1, name2 = self.division_caller.assign_names(parent, dau1, dau2)

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
                    nuc.identity = f"{NUC}{i + 1:04d}_{z}_{nuc.x}_{nuc.y}"


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
