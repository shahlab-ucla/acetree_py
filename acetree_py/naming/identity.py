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
from .division_caller import DivisionCaller
from .founder_id import FounderAssignment, identify_founders
from .initial_id import NUC, identify_initial_cells
from .lineage_axes import build_lineage_map
from .rules import RuleManager
from .validation import NamingWarning

logger = logging.getLogger(__name__)

# Naming method constants (matches NamingMethod enum in io/config.py)
MANUAL = 2
NEWCANONICAL = 3

# Automatic identities that are only valid once the real four-cell stage has
# been established.  A common manual-initialisation failure mode is that two
# small polar-body detections make a two-cell embryo look like this stage.
_FOUR_CELL_AUTOMATIC_NAMES = frozenset({"ABa", "ABp", "EMS", "P2"})

# Size is deliberately a last-resort two-cell cue.  It is used only when the
# two rows removed from the four-object frame are substantially smaller than
# both survivors (the characteristic polar-body curation footprint), and the
# survivor diameters themselves differ by at least this fraction.
_POLAR_BODY_MAX_SIZE_RATIO = 0.8
_TWO_CELL_MIN_SIZE_SEPARATION = 0.05


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
            # Manual mode does not generate identities, but a forced name is
            # still cell-scoped.  Normalising explicit overrides here keeps a
            # saved file consistent across MANUAL and automatic modes.
            self._propagate_assigned_ids()
            logger.info("Skipping automatic naming due to MANUAL naming method")
            return

        if self.legacy_mode:
            self._clear_all_names()
            self._propagate_assigned_ids()
            if self.auxinfo is not None and self.auxinfo.is_v2:
                self._build_canonical_transform()
            self._run_legacy_pipeline()
            return

        # Keep the loaded/current automatic state until the pipeline proves
        # that it can establish a replacement founder frame.  Partial movies
        # and focused edits often contain no four-cell stage; erasing valid
        # names in those datasets on every rebuild is destructive.
        previous_identities = [
            [nuc.identity for nuc in nuclei]
            for nuclei in self.nuclei_record
        ]

        # Step 1: Clear all non-forced names
        self._clear_all_names()

        # Step 1b: Propagate forced names through successor/predecessor chains
        self._propagate_assigned_ids()

        # Step 2: Build CanonicalTransform if v2 (for cross-validation only)
        if self.auxinfo is not None and self.auxinfo.is_v2:
            self._build_canonical_transform()

        # Step 3: Topology-based identification (unified default)
        ap_hint = None
        if self.auxinfo is not None and self.auxinfo.is_v2 and self.auxinfo.has_orientation:
            ap_hint = self.auxinfo.ap_orientation

        self.founder_assignment = identify_founders(
            self.nuclei_record,
            starting_index=self.starting_index,
            ending_index=self.ending_index,
            z_pix_res=self.z_pix_res,
            ap_hint=ap_hint,
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

            # Founder topology can be trustworthy even when compression or
            # missing landmark groups make DV/LR unknowable.  Preserve those
            # founder identities, but do not let raw microscope coordinates
            # masquerade as a canonical body frame for downstream divisions.
            logger.warning(
                "Founder identities retained, but downstream canonical naming "
                "is deferred because no complete body frame is available"
            )
            downstream_start = self.founder_assignment.four_cell_time + 1
            self._restore_previous_identities(
                previous_identities, start_index=downstream_start,
            )
            self._propagate_assigned_ids()
            self._assign_neutral_names(downstream_start)
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
        # Founder probing may have written tentative ABa/ABp/EMS/P2 labels
        # before its composite confidence fell below the acceptance threshold.
        # Do not let those rejected guesses leak into the fallback result.
        self._clear_all_names()

        if fa.constraint_conflict:
            # Conflicting curator anchors are visible data that require manual
            # resolution.  Preserve those explicit values, but never replace
            # them with a different automatic founder hypothesis or restore
            # stale biological descendants.
            self._propagate_assigned_ids()
            self._assign_neutral_names(self.starting_index)
            return

        # Reconcile a curated false four-cell hypothesis before the generic
        # forced-anchor path.  A curator lock below a real division is useful
        # downstream evidence, but it must not prevent the stale ABa/ABp/EMS/P2
        # roots from first being corrected to AB/P1.
        two_cell_recovery = self._recover_curated_two_cell_stage(
            previous_identities,
        )
        if two_cell_recovery is not None:
            (
                invalidated_refs,
                ab_component,
                p1_component,
                recovery_time,
                recovery_source,
            ) = two_cell_recovery
            self._restore_previous_identities(
                previous_identities,
                excluded_refs=invalidated_refs,
            )
            self._propagate_assigned_ids()

            if ab_component is not None and p1_component is not None:
                self._set_automatic_component_name(ab_component, "AB")
                self._set_automatic_component_name(p1_component, "P1")
                fa.warnings.append(
                    "Reconciled a curated two-cell stage to AB/P1 using "
                    f"{recovery_source}"
                )
                logger.info(
                    "Reconciled stale four-cell identities at t=%d to AB/P1 "
                    "using %s",
                    recovery_time + 1,
                    recovery_source,
                )

                # A complete explicit body frame can safely continue the
                # normal daughter rules.  Without DV/LR, the recovered
                # AB/P1 pair still supplies enough AP information to preserve
                # each exact RuleManager daughter family.  Only the within-
                # pair ordering may fall back to stable successor order.
                if self.canonical_transform is not None and self.canonical_transform.active:
                    self._setup_division_caller("")
                elif (
                    self.auxinfo is not None
                    and not self.auxinfo.is_v2
                    and self.auxinfo.has_orientation
                ):
                    self._setup_division_caller(self.auxinfo.axis.upper())

                if self.division_caller is not None:
                    self._use_canonical_rules(recovery_time)
                else:
                    recovered_ap = self._recovered_ap_direction(
                        ab_component,
                        p1_component,
                    )
                    stable_order_divisions = self._assign_rule_safe_descendants(
                        invalidated_refs,
                        start_index=recovery_time,
                        ap_direction=recovered_ap,
                        previous_identities=previous_identities,
                    )
                    if stable_order_divisions:
                        fa.warnings.append(
                            "Preserved canonical daughter families for "
                            f"{stable_order_divisions} division(s) without a "
                            "complete body frame; ambiguous sister order used "
                            "stable successor order"
                        )
                    self._assign_neutral_names(self.starting_index)
                    recovered_frame = self._find_recovered_four_cell_frame(
                        invalidated_refs,
                        start_index=recovery_time,
                    )
                    if recovered_frame is not None:
                        four_cell_time, founder_indices = recovered_frame
                        fa.four_cell_time = four_cell_time
                        fa.aba_idx = founder_indices["ABa"]
                        fa.abp_idx = founder_indices["ABp"]
                        fa.ems_idx = founder_indices["EMS"]
                        fa.p2_idx = founder_indices["P2"]
                        fa.ap_vector = recovered_ap
                        fa.lr_vector = None
                        fa.dv_vector = None
                        self._setup_division_caller_from_founders()
                        if self.division_caller is not None:
                            self._use_canonical_rules(four_cell_time)
            else:
                fa.warnings.append(
                    "Discarded stale four-cell identities after a curated "
                    "four-to-two object correction; AB/P1 ordering remains ambiguous"
                )
                logger.warning(
                    "Discarded stale four-cell identities at t=%d; no trusted "
                    "timing, AP, or polar-size cue can order AB/P1",
                    recovery_time + 1,
                )
                self._assign_neutral_names(self.starting_index)
            return

        # A late-start or ablated dataset may not contain a usable four-cell
        # stage.  If the curator supplied both a trusted orientation and at
        # least one forced lineage anchor, continue canonical rules forward
        # from that anchor instead of discarding the useful manual context.
        has_forced_anchor = any(
            nuc.is_alive and bool(nuc.assigned_id)
            for nuclei in self.nuclei_record[self.starting_index:self.ending_index]
            for nuc in nuclei
        )
        if has_forced_anchor:
            if self.canonical_transform is not None and self.canonical_transform.active:
                self._setup_division_caller("")
            elif (
                self.auxinfo is not None
                and not self.auxinfo.is_v2
                and getattr(self.auxinfo, "has_orientation", False)
            ):
                self._setup_division_caller(self.auxinfo.axis.upper())
            if self.division_caller is not None:
                logger.info(
                    "Founder ID unavailable; continuing canonical naming from forced anchors"
                )
                self._use_canonical_rules(self.starting_index)
                return

        self._restore_previous_identities(previous_identities)
        self._propagate_assigned_ids()
        self._clear_unforced_descendants_of_forced_early_cells()
        self._assign_neutral_names(self.starting_index)

    def _restore_previous_identities(
        self,
        previous_identities: list[list[str]],
        start_index: int = 0,
        excluded_refs: set[tuple[int, int]] | None = None,
    ) -> None:
        """Restore valid loaded names when re-identification is unavailable.

        Manual overrides remain authoritative, dead records stay unnamed, and
        previously blank entries remain available for the generic fill pass.
        ``excluded_refs`` marks an automatic hypothesis invalidated by the
        current topology and prevents stale display state from becoming data.
        """
        excluded = excluded_refs or set()
        for t, nuclei in enumerate(self.nuclei_record):
            if t < start_index:
                continue
            if t >= len(previous_identities):
                break
            prior_at_time = previous_identities[t]
            for j, nuc in enumerate(nuclei):
                if (
                    not nuc.is_alive
                    or nuc.assigned_id
                    or j >= len(prior_at_time)
                    or (t, j) in excluded
                ):
                    continue
                if prior_at_time[j]:
                    nuc.identity = prior_at_time[j]

    def _recover_curated_two_cell_stage(
        self,
        previous_identities: list[list[str]],
    ) -> tuple[
        set[tuple[int, int]],
        set[tuple[int, int]] | None,
        set[tuple[int, int]] | None,
        int,
        str,
    ] | None:
        """Recognise and repair a false four-cell hypothesis after curation.

        Detector-assisted manual initialisation commonly produces four rows at
        the biological two-cell stage: AB, P1, and two polar bodies.  Once the
        two false detections are killed, founder probing correctly rejects the
        old four-cell frame, but the generic partial-movie fallback used to
        restore its surviving automatic ``ABa/ABp/EMS/P2`` labels verbatim.

        The repair is intentionally narrow.  It requires exactly four retained
        rows at one timepoint, exactly two live and two dead, two unforced live
        rows whose prior automatic names came from the four-cell family (or an
        already-reconciled automatic AB/P1 pair), two substantially smaller
        deleted rows, and disjoint continuation components.  The AB/P1 form
        upgrades datasets saved by the earlier root-only repair.  The positive
        polar-size footprint avoids reinterpreting ordinary two-cell partial
        or four-cell ablation movies.

        AB/P1 ordering uses the strongest available evidence in this order:
        future division timing, an explicit AP orientation, then blastomere
        size asymmetry.  The rejected four-cell labels are never reused as
        ordering evidence.  If no cue is available, the returned components
        are ``None`` so the caller can invalidate the impossible labels and
        fail closed with neutral root names.
        """
        end = min(self.ending_index, len(self.nuclei_record))
        for t in range(self.starting_index, end):
            nuclei = self.nuclei_record[t]
            if len(nuclei) != 4 or t >= len(previous_identities):
                continue

            alive = [(idx, nuc) for idx, nuc in enumerate(nuclei) if nuc.is_alive]
            dead = [(idx, nuc) for idx, nuc in enumerate(nuclei) if not nuc.is_alive]
            if len(alive) != 2 or len(dead) != 2:
                continue

            # A genuine four-cell stage can have exactly this live/dead row
            # shape after two blastomeres are ablated.  The retained
            # predecessor links still reveal two sister pairs descending from
            # two distinct dividing parents, even though set_all_successors()
            # no longer includes the dead rows.  That topology is stronger
            # evidence than size and vetoes polar-body recovery.
            if self._has_four_cell_division_topology(t, nuclei):
                continue

            live_sizes = [float(nucleus.size) for _idx, nucleus in alive]
            dead_sizes = [float(nucleus.size) for _idx, nucleus in dead]
            if min(live_sizes) <= 0 or min(dead_sizes) < 0:
                continue
            if (
                max(dead_sizes)
                > _POLAR_BODY_MAX_SIZE_RATIO * min(live_sizes)
            ):
                continue

            prior_at_time = previous_identities[t]
            if any(
                nuc.assigned_id or idx >= len(prior_at_time)
                for idx, nuc in alive
            ):
                continue
            prior_live_names = [prior_at_time[idx] for idx, _nuc in alive]
            stale_four_cell_hypothesis = (
                len(set(prior_live_names)) == 2
                and all(
                    name in _FOUR_CELL_AUTOMATIC_NAMES
                    for name in prior_live_names
                )
            )
            already_reconciled_pair = set(prior_live_names) == {"AB", "P1"}
            if not stale_four_cell_hypothesis and not already_reconciled_pair:
                continue
            if len(set(prior_live_names)) != 2:
                # Duplicate automatic founder labels are corrupt state, not a
                # trustworthy rejected four-cell hypothesis to reinterpret.
                continue

            first_component = self._continuation_component_refs(t, alive[0][0])
            second_component = self._continuation_component_refs(t, alive[1][0])
            if (
                not self._component_topology_is_valid(first_component)
                or not self._component_topology_is_valid(second_component)
            ):
                continue
            invalidated = self._descendant_refs(
                first_component | second_component,
            )
            if not first_component or not second_component:
                continue
            if first_component & second_component:
                return invalidated, None, None, t, "ambiguous topology"
            if any(
                self.nuclei_record[ref_t][ref_idx].assigned_id
                for ref_t, ref_idx in first_component | second_component
            ):
                # A continuation-scoped curator lock is authoritative.  It
                # should already have propagated to the candidate row, but
                # retain this guard for malformed/non-reciprocal input.
                continue

            if already_reconciled_pair:
                first_is_ab = prior_live_names[0] == "AB"
                source = "existing reconciled AB/P1 state"
            else:
                first_is_ab, source = self._order_two_cell_candidates(
                    t,
                    alive,
                    dead,
                )
            if first_is_ab is None:
                return invalidated, None, None, t, source
            if first_is_ab:
                return invalidated, first_component, second_component, t, source
            return invalidated, second_component, first_component, t, source

        return None

    def _has_four_cell_division_topology(
        self,
        time: int,
        nuclei: list[Nucleus],
    ) -> bool:
        """Return whether lineage evidence must block polar-body recovery.

        Trace all four retained rows back through continuation frames to their
        birth divisions.  Two pairs born from two parents identify a genuine
        four-cell stage, even several frames after those divisions.  Malformed
        claimed topology also blocks recovery: broken links are uncertainty,
        never positive evidence that deleted rows were polar bodies.
        """
        if len(nuclei) != 4:
            return False

        birth_parents: list[tuple[int, int] | None] = []
        for index, nucleus in enumerate(nuclei):
            birth_parent, topology_valid = self._birth_division_parent(
                time,
                index,
            )
            if not topology_valid:
                return True
            birth_parents.append(birth_parent)

        if any(parent is None for parent in birth_parents):
            return False
        groups: dict[tuple[int, int], int] = {}
        for parent in birth_parents:
            assert parent is not None
            groups[parent] = groups.get(parent, 0) + 1
        return len(groups) == 2 and set(groups.values()) == {2}

    def _birth_division_parent(
        self,
        time: int,
        index: int,
    ) -> tuple[tuple[int, int] | None, bool]:
        """Trace a row to its birth division and validate every link.

        The boolean is false for any claimed but non-reciprocal, out-of-range,
        or over-subscribed relationship.  Dead children remain in the retained
        rows and therefore still participate in the predecessor grouping.
        """
        current_time = time
        current_index = index
        while current_time > 0:
            current = self.nuclei_record[current_time][current_index]
            if current.predecessor <= 0:
                reverse_claim = any(
                    parent.is_alive
                    and current_index + 1
                    in (parent.successor1, parent.successor2)
                    for parent in self.nuclei_record[current_time - 1]
                )
                return None, not reverse_claim

            parent_index = current.predecessor - 1
            previous = self.nuclei_record[current_time - 1]
            if not (0 <= parent_index < len(previous)):
                return None, False
            parent = previous[parent_index]
            if not parent.is_alive:
                return None, False

            retained_children = {
                child_index
                for child_index, child in enumerate(
                    self.nuclei_record[current_time]
                )
                if child.predecessor == parent_index + 1
            }
            if current_index not in retained_children:
                return None, False
            if len(retained_children) not in (1, 2):
                return None, False

            declared_values = [
                successor
                for successor in (parent.successor1, parent.successor2)
                if successor > 0
            ]
            if (
                parent.successor2 > 0 and parent.successor1 <= 0
            ) or len(set(declared_values)) != len(declared_values):
                return None, False
            declared_children = {
                successor - 1 for successor in declared_values
            }
            if any(
                not (0 <= child < len(self.nuclei_record[current_time]))
                for child in declared_children
            ):
                return None, False
            if not declared_children <= retained_children:
                return None, False
            live_children = {
                child
                for child in retained_children
                if self.nuclei_record[current_time][child].is_alive
            }
            if not live_children <= declared_children:
                return None, False

            other_parent_claims = {
                other_index
                for other_index, other_parent in enumerate(previous)
                if other_parent.is_alive
                and current_index + 1
                in (other_parent.successor1, other_parent.successor2)
            }
            if other_parent_claims - {parent_index}:
                return None, False

            if len(retained_children) == 2:
                return (current_time - 1, parent_index), True

            current_time -= 1
            current_index = parent_index

        return None, True

    def _order_two_cell_candidates(
        self,
        time: int,
        alive: list[tuple[int, Nucleus]],
        dead: list[tuple[int, Nucleus]],
    ) -> tuple[bool | None, str]:
        """Return whether ``alive[0]`` is AB and the evidence provenance."""
        division_observations = [
            self._division_observation(time, idx)
            for idx, _nucleus in alive
        ]
        first_division, first_last_observed = division_observations[0]
        second_division, second_last_observed = division_observations[1]
        if first_division is not None and second_division is not None:
            if first_division != second_division:
                return first_division < second_division, "future division timing"
        elif first_division is not None:
            if second_last_observed > first_division:
                return True, "future division timing"
        elif second_division is not None:
            if first_last_observed > second_division:
                return False, "future division timing"

        ap = self._explicit_ap_direction()
        if ap is not None:
            projections = []
            for _idx, nucleus in alive:
                position = np.array(
                    [nucleus.x, nucleus.y, nucleus.z * self.z_pix_res],
                    dtype=float,
                )
                projections.append(float(np.dot(position, ap)))
            if not np.isclose(projections[0], projections[1], atol=1e-8):
                # AP points posterior -> anterior, so the larger projection is AB.
                return projections[0] > projections[1], "explicit AP orientation"

        live_sizes = [float(nucleus.size) for _idx, nucleus in alive]
        dead_sizes = [float(nucleus.size) for _idx, nucleus in dead]
        if min(live_sizes) > 0 and min(dead_sizes) >= 0:
            deleted_are_small = (
                max(dead_sizes)
                <= _POLAR_BODY_MAX_SIZE_RATIO * min(live_sizes)
            )
            relative_gap = (
                abs(live_sizes[0] - live_sizes[1]) / max(live_sizes)
            )
            if deleted_are_small and relative_gap >= _TWO_CELL_MIN_SIZE_SEPARATION:
                # AB is the larger blastomere at the two-cell stage.
                return live_sizes[0] > live_sizes[1], "polar-body size asymmetry"

        return None, "ambiguous evidence"

    def _explicit_ap_direction(self) -> np.ndarray | None:
        """Return a trusted posterior-to-anterior vector in physical space."""
        if self.auxinfo is None or not self.auxinfo.has_orientation:
            return None

        if self.auxinfo.is_v2:
            ap = self.auxinfo.ap_orientation
            if ap is None:
                return None
            vector = np.asarray(ap, dtype=float)
        else:
            axis = self.auxinfo.axis.upper()
            sign = 1.0 if axis[0] == "A" else -1.0
            angle = np.radians(self.auxinfo.angle)
            vector = np.array(
                [sign * np.cos(angle), sign * np.sin(angle), 0.0],
                dtype=float,
            )

        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return vector / norm

    def _division_observation(
        self,
        time: int,
        index: int,
    ) -> tuple[int | None, int]:
        """Return the next valid division and last reciprocally observed time.

        A missing division is right-censored, not automatically later.  The
        last-observed boundary lets the caller use one-sided evidence only
        when the intact sister continuation is actually seen beyond the other
        lineage's division.
        """
        end = min(self.ending_index, len(self.nuclei_record))
        t = time
        idx = index
        while 0 <= t < end and 0 <= idx < len(self.nuclei_record[t]):
            nucleus = self.nuclei_record[t][idx]
            if not nucleus.is_alive:
                return None, t - 1
            if nucleus.successor1 > 0 and nucleus.successor2 > 0:
                next_time = t + 1
                if next_time >= end:
                    return None, t
                successor_indices = (
                    nucleus.successor1 - 1,
                    nucleus.successor2 - 1,
                )
                if any(
                    not (0 <= successor < len(self.nuclei_record[next_time]))
                    for successor in successor_indices
                ):
                    return None, t
                daughters = [
                    self.nuclei_record[next_time][successor]
                    for successor in successor_indices
                ]
                if any(
                    not daughter.is_alive or daughter.predecessor != idx + 1
                    for daughter in daughters
                ):
                    return None, t
                return t, t
            if nucleus.successor1 <= 0 or t + 1 >= end:
                return None, t
            successor_idx = nucleus.successor1 - 1
            if not (0 <= successor_idx < len(self.nuclei_record[t + 1])):
                return None, t
            successor = self.nuclei_record[t + 1][successor_idx]
            if not successor.is_alive or successor.predecessor != idx + 1:
                return None, t
            t += 1
            idx = successor_idx
        return None, max(time, t - 1)

    def _continuation_component_refs(
        self,
        time: int,
        index: int,
    ) -> set[tuple[int, int]]:
        """Return the reciprocal, non-dividing continuation component."""
        end = min(self.ending_index, len(self.nuclei_record))
        if not (0 <= time < end and 0 <= index < len(self.nuclei_record[time])):
            return set()
        if not self.nuclei_record[time][index].is_alive:
            return set()

        refs = {(time, index)}

        t = time
        idx = index
        while t > self.starting_index:
            nucleus = self.nuclei_record[t][idx]
            if nucleus.predecessor <= 0:
                break
            pred_idx = nucleus.predecessor - 1
            if not (0 <= pred_idx < len(self.nuclei_record[t - 1])):
                break
            predecessor = self.nuclei_record[t - 1][pred_idx]
            if (
                not predecessor.is_alive
                or predecessor.successor2 > 0
                or predecessor.successor1 != idx + 1
            ):
                break
            t -= 1
            idx = pred_idx
            refs.add((t, idx))

        t = time
        idx = index
        while t + 1 < end:
            nucleus = self.nuclei_record[t][idx]
            if nucleus.successor1 <= 0 or nucleus.successor2 > 0:
                break
            successor_idx = nucleus.successor1 - 1
            if not (0 <= successor_idx < len(self.nuclei_record[t + 1])):
                break
            successor = self.nuclei_record[t + 1][successor_idx]
            if not successor.is_alive or successor.predecessor != idx + 1:
                break
            t += 1
            idx = successor_idx
            refs.add((t, idx))

        return refs

    def _set_automatic_component_name(
        self,
        component: set[tuple[int, int]],
        name: str,
    ) -> None:
        """Assign an automatic name without converting it into a curator lock."""
        for time, index in component:
            nucleus = self.nuclei_record[time][index]
            if nucleus.is_alive and not nucleus.assigned_id:
                nucleus.identity = name

    def _recovered_ap_direction(
        self,
        ab_component: set[tuple[int, int]],
        p1_component: set[tuple[int, int]],
    ) -> np.ndarray | None:
        """Estimate posterior-to-anterior direction from recovered AB/P1.

        AB and P1 roles have already been resolved by trusted timing, explicit
        AP metadata, or the polar/blastomere size footprint.  Their physical
        separation therefore supplies the partial AP frame needed to order the
        two early founder divisions without inventing DV or LR axes.
        """
        ab_by_time = {time: index for time, index in ab_component}
        p1_by_time = {time: index for time, index in p1_component}
        samples: list[np.ndarray] = []
        for time in sorted(ab_by_time.keys() & p1_by_time.keys()):
            ab = self.nuclei_record[time][ab_by_time[time]]
            p1 = self.nuclei_record[time][p1_by_time[time]]
            delta = np.array(
                [
                    float(ab.x - p1.x),
                    float(ab.y - p1.y),
                    float(ab.z - p1.z) * self.z_pix_res,
                ],
                dtype=float,
            )
            norm = float(np.linalg.norm(delta))
            if np.isfinite(norm) and norm > 1e-8:
                samples.append(delta / norm)

        if not samples:
            return None
        direction = np.mean(samples, axis=0)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-8:
            return None
        return direction / norm

    def _find_recovered_four_cell_frame(
        self,
        scope: set[tuple[int, int]],
        *,
        start_index: int,
    ) -> tuple[int, dict[str, int]] | None:
        """Find the first complete recovered founder quartet in one frame."""
        expected = {"ABa", "ABp", "EMS", "P2"}
        end = min(self.ending_index, len(self.nuclei_record))
        for time in range(max(0, start_index), end):
            found: dict[str, int] = {}
            duplicate = False
            for index, nucleus in enumerate(self.nuclei_record[time]):
                if (
                    (time, index) not in scope
                    or not nucleus.is_alive
                    or nucleus.effective_name not in expected
                ):
                    continue
                name = nucleus.effective_name
                if name in found:
                    duplicate = True
                    break
                found[name] = index
            if not duplicate and set(found) == expected:
                return time, found
        return None

    def _assign_rule_safe_descendants(
        self,
        scope: set[tuple[int, int]],
        *,
        start_index: int,
        ap_direction: np.ndarray | None,
        previous_identities: list[list[str]],
    ) -> int:
        """Repair the first AB/P1 divisions without changing daughter family.

        A complete AP/DV/LR frame determines *which* sister receives each name,
        but the unordered daughter pair is already fixed for recovered AB and
        P1.  This bridge deliberately stops after those founder divisions;
        later divisions return to the normal geometry-aware caller when a
        complete four-founder frame can be reconstructed.
        """
        end = min(self.ending_index, len(self.nuclei_record))
        stable_order_divisions = 0
        for time in range(max(0, start_index), max(0, end - 1)):
            next_nuclei = self.nuclei_record[time + 1]
            for index, parent in enumerate(self.nuclei_record[time]):
                if (time, index) not in scope or not parent.is_alive:
                    continue
                parent_name = parent.effective_name
                if parent_name not in {"AB", "P1"}:
                    continue

                successors = [
                    successor - 1
                    for successor in (parent.successor1, parent.successor2)
                    if successor > 0
                ]
                if not successors:
                    continue
                if any(
                    not (0 <= successor < len(next_nuclei))
                    or (time + 1, successor) not in scope
                    or not next_nuclei[successor].is_alive
                    or next_nuclei[successor].predecessor != index + 1
                    for successor in successors
                ):
                    continue

                if len(successors) == 1:
                    successor = next_nuclei[successors[0]]
                    if not successor.assigned_id:
                        successor.identity = parent_name
                    continue
                if len(successors) != 2 or successors[0] == successors[1]:
                    continue

                daughter1 = next_nuclei[successors[0]]
                daughter2 = next_nuclei[successors[1]]
                first_component = self._continuation_component_refs(
                    time + 1,
                    successors[0],
                )
                second_component = self._continuation_component_refs(
                    time + 1,
                    successors[1],
                )
                if (
                    not first_component
                    or not second_component
                    or first_component & second_component
                    or not self._component_topology_is_valid(first_component)
                    or not self._component_topology_is_valid(second_component)
                ):
                    warning = (
                        f"{parent_name} daughters remain neutral because their "
                        "continuation topology is malformed"
                    )
                    logger.warning("%s", warning)
                    if self.founder_assignment is not None:
                        self.founder_assignment.warnings.append(warning)
                    continue

                rule = self.rule_manager.get_rule(parent_name)
                prior1 = self._prior_component_rule_name(
                    first_component,
                    previous_identities,
                    {rule.daughter1, rule.daughter2},
                )
                prior2 = self._prior_component_rule_name(
                    second_component,
                    previous_identities,
                    {rule.daughter1, rule.daughter2},
                )
                if {prior1, prior2} != {rule.daughter1, rule.daughter2}:
                    prior1 = prior2 = ""
                ordered = self._order_recovered_daughter_components(
                    parent_name,
                    time + 1,
                    successors,
                    first_component,
                    second_component,
                    ap_direction,
                )
                if ordered is not None:
                    name1, name2, source = ordered
                else:
                    name1, name2, source = self._coerce_rule_daughter_pair(
                        parent_name,
                        prior1,
                        prior2,
                        allow_missing_fallback=True,
                    )
                expected = {name1, name2}
                pair_components = first_component | second_component
                collisions = [
                    nucleus.effective_name
                    for other_time, nuclei in enumerate(self.nuclei_record[:end])
                    for other_index, nucleus in enumerate(nuclei)
                    if nucleus.is_alive
                    and (other_time, other_index) not in pair_components
                    and nucleus.effective_name in expected
                ]
                if collisions:
                    warning = (
                        f"{parent_name} daughter family collides with existing "
                        f"name(s) {sorted(set(collisions))}; duplicate ownership "
                        "was retained for validation"
                    )
                    logger.warning("%s", warning)
                    if self.founder_assignment is not None:
                        self.founder_assignment.warnings.append(warning)

                daughter1.identity = name1
                daughter2.identity = name2
                _use_preassigned_id(daughter1, daughter2)
                if source == "stable successor order":
                    stable_order_divisions += 1

        return stable_order_divisions

    def _order_recovered_daughter_components(
        self,
        parent_name: str,
        daughter_time: int,
        successors: list[int],
        first_component: set[tuple[int, int]],
        second_component: set[tuple[int, int]],
        ap_direction: np.ndarray | None,
    ) -> tuple[str, str, str] | None:
        """Order a recovered founder pair from timing or averaged AP geometry."""
        rule = self.rule_manager.get_rule(parent_name)
        if parent_name == "P1":
            first_division, first_last = self._division_observation(
                daughter_time,
                successors[0],
            )
            second_division, second_last = self._division_observation(
                daughter_time,
                successors[1],
            )
            first_is_ems: bool | None = None
            if first_division is not None and second_division is not None:
                if first_division != second_division:
                    first_is_ems = first_division < second_division
            elif first_division is not None and second_last > first_division:
                first_is_ems = True
            elif second_division is not None and first_last > second_division:
                first_is_ems = False
            if first_is_ems is not None:
                if first_is_ems:
                    return rule.daughter1, rule.daughter2, "future division timing"
                return rule.daughter2, rule.daughter1, "future division timing"

        if ap_direction is None:
            return None
        first_by_time = {time: index for time, index in first_component}
        second_by_time = {time: index for time, index in second_component}
        projections: list[float] = []
        for time in sorted(first_by_time.keys() & second_by_time.keys()):
            first = self.nuclei_record[time][first_by_time[time]]
            second = self.nuclei_record[time][second_by_time[time]]
            separation = np.array(
                [
                    float(first.x - second.x),
                    float(first.y - second.y),
                    float(first.z - second.z) * self.z_pix_res,
                ],
                dtype=float,
            )
            projection = float(np.dot(separation, ap_direction))
            if np.isfinite(projection):
                projections.append(projection)
        if not projections:
            return None
        mean_projection = float(np.mean(projections))
        if np.isclose(mean_projection, 0.0, atol=1e-8):
            return None
        if mean_projection > 0:
            return rule.daughter1, rule.daughter2, "recovered AP"
        return rule.daughter2, rule.daughter1, "recovered AP"

    @staticmethod
    def _prior_component_rule_name(
        component: set[tuple[int, int]],
        previous_identities: list[list[str]],
        expected: set[str],
    ) -> str:
        """Return one consistent prior rule name from a continuation."""
        names = {
            previous_identities[time][index]
            for time, index in component
            if time < len(previous_identities)
            and index < len(previous_identities[time])
            and previous_identities[time][index]
            and not previous_identities[time][index].startswith(NUC)
        }
        if len(names) == 1 and names <= expected:
            return next(iter(names))
        return ""

    def _coerce_rule_daughter_pair(
        self,
        parent_name: str,
        proposed1: str,
        proposed2: str,
        *,
        allow_missing_fallback: bool = False,
    ) -> tuple[str, str, str]:
        """Return an exact RuleManager daughter pair and its provenance."""
        rule = self.rule_manager.get_rule(parent_name)
        expected = {rule.daughter1, rule.daughter2}

        if proposed1 != proposed2 and {proposed1, proposed2} == expected:
            return proposed1, proposed2, "division caller"
        if not proposed1 or not proposed2:
            if not allow_missing_fallback:
                return "", "", "deferred"
        return rule.daughter1, rule.daughter2, "stable successor order"

    def _component_topology_is_valid(
        self,
        component: set[tuple[int, int]],
    ) -> bool:
        """Reject malformed links instead of turning them into founder evidence."""
        end = min(self.ending_index, len(self.nuclei_record))
        for time, index in component:
            nucleus = self.nuclei_record[time][index]
            claimed_parents: set[int] = set()
            if time > 0:
                claimed_parents = {
                    parent_index
                    for parent_index, parent in enumerate(
                        self.nuclei_record[time - 1]
                    )
                    if parent.is_alive
                    and index + 1 in (parent.successor1, parent.successor2)
                }
            if nucleus.predecessor > 0:
                if time <= 0:
                    return False
                predecessor_idx = nucleus.predecessor - 1
                if not (
                    0 <= predecessor_idx < len(self.nuclei_record[time - 1])
                ):
                    return False
                predecessor = self.nuclei_record[time - 1][predecessor_idx]
                if (
                    not predecessor.is_alive
                    or index + 1
                    not in (predecessor.successor1, predecessor.successor2)
                    or claimed_parents != {predecessor_idx}
                ):
                    return False
            elif claimed_parents:
                return False

            successor_values = (nucleus.successor1, nucleus.successor2)
            positive_successors = [value for value in successor_values if value > 0]
            if nucleus.successor2 > 0 and nucleus.successor1 <= 0:
                return False
            if not positive_successors:
                if time + 1 < end and any(
                    child.is_alive and child.predecessor == index + 1
                    for child in self.nuclei_record[time + 1]
                ):
                    return False
                continue
            if time + 1 >= end:
                return False
            if len(set(positive_successors)) != len(positive_successors):
                return False
            declared_children = {
                successor - 1 for successor in positive_successors
            }
            claimed_children = {
                child_index
                for child_index, child in enumerate(
                    self.nuclei_record[time + 1]
                )
                if child.is_alive and child.predecessor == index + 1
            }
            if claimed_children != declared_children:
                return False
            for successor in positive_successors:
                successor_idx = successor - 1
                if not (
                    0 <= successor_idx < len(self.nuclei_record[time + 1])
                ):
                    return False
                child = self.nuclei_record[time + 1][successor_idx]
                if not child.is_alive or child.predecessor != index + 1:
                    return False
        return True

    def _descendant_refs(
        self,
        seeds: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        """Return live descendants so a rejected founder hypothesis stays gone."""
        end = min(self.ending_index, len(self.nuclei_record))
        refs = set(seeds)
        frontier = list(seeds)
        while frontier:
            time, index = frontier.pop()
            if not (0 <= time < end and 0 <= index < len(self.nuclei_record[time])):
                continue
            nucleus = self.nuclei_record[time][index]
            if not nucleus.is_alive or time + 1 >= end:
                continue
            for successor in (nucleus.successor1, nucleus.successor2):
                if successor <= 0:
                    continue
                successor_ref = (time + 1, successor - 1)
                if successor_ref in refs:
                    continue
                next_time, next_index = successor_ref
                if not (0 <= next_index < len(self.nuclei_record[next_time])):
                    continue
                child = self.nuclei_record[next_time][next_index]
                if not child.is_alive or child.predecessor != index + 1:
                    continue
                refs.add(successor_ref)
                frontier.append(successor_ref)
        return refs

    def _assign_neutral_names(self, start_index: int) -> None:
        """Fill unnamed records without asserting anatomical daughter order.

        Continuations inherit a known parent identity.  At a division with no
        complete body frame, each still-unnamed daughter receives a neutral
        ``Nuc...`` identifier instead of a biological ``a/p``, ``d/v``, or
        ``l/r`` suffix.  The curated two-cell bridge pre-fills the exact first
        AB/P1 daughter families before this pass; this helper handles roots or
        later axis-dependent divisions that remain unresolved.  Reprocessing
        after a curator supplies valid axes can replace those placeholders.
        """
        end = min(self.ending_index, len(self.nuclei_record))
        for t in range(max(0, start_index), end):
            previous = self.nuclei_record[t - 1] if t > 0 else None
            for nuc in self.nuclei_record[t]:
                if not nuc.is_alive or nuc.identity:
                    continue
                if nuc.assigned_id:
                    nuc.identity = nuc.assigned_id
                    continue

                if previous is not None and nuc.predecessor > 0:
                    pred_idx = nuc.predecessor - 1
                    if 0 <= pred_idx < len(previous):
                        pred = previous[pred_idx]
                        if pred.is_alive and pred.successor2 == NILLI:
                            nuc.identity = pred.effective_name
                            if nuc.identity:
                                continue

                z = round(nuc.z)
                nuc.identity = f"{NUC}{t + 1:03d}_{z}_{nuc.x}_{nuc.y}"

    def _clear_unforced_descendants_of_forced_early_cells(self) -> None:
        """Invalidate stale automatic progeny after an unresolved early edit.

        When a partial movie has no usable founder frame and no explicit body
        axes, canonical daughter order cannot be recomputed.  Restoring loaded
        identities below a changed forced P0/AB/P1/ABa/ABp/EMS/P2 anchor would
        present old automatic names as though they were concurrent with the
        edit.  Clear those unforced subtrees so ``_assign_neutral_names`` can
        fail closed with non-biological labels.  Explicit descendant overrides
        remain untouched.
        """
        early_names = {"P0", "AB", "P1", "ABa", "ABp", "EMS", "P2"}
        end = min(self.ending_index, len(self.nuclei_record))
        frontier: list[tuple[int, int]] = []

        for t in range(self.starting_index, end):
            for idx, nucleus in enumerate(self.nuclei_record[t]):
                if nucleus.is_alive and nucleus.assigned_id in early_names:
                    if nucleus.successor1 > 0 and t + 1 < end:
                        frontier.append((t + 1, nucleus.successor1 - 1))
                    if nucleus.successor2 > 0 and t + 1 < end:
                        frontier.append((t + 1, nucleus.successor2 - 1))

        visited: set[tuple[int, int]] = set()
        while frontier:
            t, idx = frontier.pop()
            if (t, idx) in visited or not (0 <= t < end):
                continue
            if not (0 <= idx < len(self.nuclei_record[t])):
                continue
            visited.add((t, idx))
            nucleus = self.nuclei_record[t][idx]
            if not nucleus.is_alive:
                continue
            if not nucleus.assigned_id:
                nucleus.identity = ""
            if t + 1 >= end:
                continue
            if nucleus.successor1 > 0:
                frontier.append((t + 1, nucleus.successor1 - 1))
            if nucleus.successor2 > 0:
                frontier.append((t + 1, nucleus.successor2 - 1))

    def _run_legacy_pipeline(self) -> None:
        """Run the legacy InitialID-based pipeline."""
        import math

        angle_rad = 0.0
        if self.auxinfo is not None:
            angle_rad = math.radians(-self.auxinfo.angle)

        result = identify_initial_cells(
            self.nuclei_record,
            starting_index=self.starting_index,
            ending_index=self.ending_index,
            canonical_transform=self.canonical_transform,
            angle=angle_rad,
            z_pix_res=self.z_pix_res,
        )
        # InitialID assigns its own early identities.  Reassert cell-scoped
        # curator overrides before those names become parents for canonical
        # daughter rules.
        self._propagate_assigned_ids()

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
            auxinfo_ap = self.canonical_transform.inverse_apply(np.array([-1.0, 0.0, 0.0]))
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
            angle_deg = np.degrees(np.arccos(cos_angle))
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
        """Create a DivisionCaller using the best trusted body-frame source.

        Explicit AuxInfo orientation is user/acquisition metadata and therefore
        outranks an inferred frame.  The lineage-centroid estimate is retained
        as the fallback for datasets without valid orientation metadata.
        """
        fa = self.founder_assignment
        if fa is None:
            return

        if self.canonical_transform is not None and self.canonical_transform.active:
            self.division_caller = DivisionCaller(
                rule_manager=self.rule_manager,
                z_pix_res=self.z_pix_res,
                canonical_transform=self.canonical_transform,
            )
            logger.info("Using explicit AuxInfo v2 body axes for division naming")
            return

        if (
            self.auxinfo is not None
            and not self.auxinfo.is_v2
            and getattr(self.auxinfo, "has_orientation", False)
        ):
            self._setup_division_caller(self.auxinfo.axis.upper())
            logger.info("Using explicit AuxInfo v1 orientation for division naming")
            return

        # No explicit orientation: infer a per-timepoint frame from lineages.
        lineage_map = build_lineage_map(
            self.nuclei_record,
            four_cell_time=fa.four_cell_time,
            aba_idx=fa.aba_idx,
            abp_idx=fa.abp_idx,
            ems_idx=fa.ems_idx,
            p2_idx=fa.p2_idx,
        )

        # Compute axes at the four-cell midpoint to seed temporal signs.
        # AP comes from P2->ABa and DV from EMS->ABp; LR is the right-handed
        # completion, not the ABa--ABp separation.  The secondary quality
        # value records when that DV geometry is weak or nearly collinear.
        from .lineage_axes import compute_local_axes
        seed_ap, seed_lr, seed_dv, _seed_quality = compute_local_axes(
            self.nuclei_record, lineage_map, fa.four_cell_time, self.z_pix_res,
        )

        complete_seed = all(axis is not None for axis in (seed_ap, seed_lr, seed_dv))
        complete_founder = all(
            axis is not None
            for axis in (fa.ap_vector, fa.lr_vector, fa.dv_vector)
        )
        if not complete_seed and not complete_founder:
            fa.warnings.append(
                "Founder topology identified, but no complete AP/DV/LR frame "
                "is available for downstream division naming"
            )
            self.division_caller = None
            return

        self.division_caller = DivisionCaller(
            rule_manager=self.rule_manager,
            z_pix_res=self.z_pix_res,
            founder_ap=fa.ap_vector,
            founder_lr=fa.lr_vector,
            founder_dv=fa.dv_vector,
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
                if nuc.is_alive and nuc.assigned_id:
                    nuc.identity = nuc.assigned_id
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
                if not succ.is_alive or succ.predecessor != idx + 1:
                    break
                if succ.assigned_id and succ.assigned_id != forced_name:
                    logger.warning(
                        "Conflicting forced names in one continuation at t=%d idx=%d: "
                        "'%s' vs '%s'; stopping propagation",
                        t + 2, s_idx + 1, forced_name, succ.assigned_id,
                    )
                    break
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
                if not pred.is_alive or pred.successor1 != idx + 1:
                    break
                if pred.assigned_id and pred.assigned_id != forced_name:
                    logger.warning(
                        "Conflicting forced names in one continuation at t=%d idx=%d: "
                        "'%s' vs '%s'; stopping propagation",
                        t, p_idx + 1, forced_name, pred.assigned_id,
                    )
                    break
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

                pname = parent.effective_name
                if parent.assigned_id:
                    parent.identity = pname

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

                safe_name1, safe_name2, source = self._coerce_rule_daughter_pair(
                    pname,
                    name1,
                    name2,
                )
                if source == "deferred":
                    logger.warning(
                        "%s division remains unnamed because the division "
                        "caller has no complete anatomical frame",
                        pname,
                    )
                    continue
                if source != "division caller":
                    logger.warning(
                        "%s division returned a foreign daughter family; "
                        "preserving the canonical rule pair using %s",
                        pname,
                        source,
                    )
                name1, name2 = safe_name1, safe_name2

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

    If a daughter has an assigned_id, override its identity with it.  When a
    single forced name selects the automatic name originally proposed for the
    sister, move that sister to the complementary automatic name.  Two equal
    forced names are an invalid manual conflict and are deliberately left
    visible for validation; changing only ``identity`` cannot disambiguate
    them because ``effective_name`` prioritises ``assigned_id``.
    """
    if not dau1.assigned_id and not dau2.assigned_id:
        return

    automatic1, automatic2 = dau1.identity, dau2.identity

    if dau1.assigned_id and dau2.assigned_id:
        dau1.identity = dau1.assigned_id
        dau2.identity = dau2.assigned_id
        if dau1.assigned_id == dau2.assigned_id:
            logger.error(
                "Both daughters carry the same forced name '%s'; manual correction required",
                dau1.assigned_id,
            )
        return

    if dau1.assigned_id:
        dau1.identity = dau1.assigned_id
        if dau1.assigned_id == automatic2:
            dau2.identity = automatic1
    elif dau2.assigned_id:
        dau2.identity = dau2.assigned_id
        if dau2.assigned_id == automatic1:
            dau1.identity = automatic2


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
