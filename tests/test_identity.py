"""Tests for acetree_py.naming.identity — the full naming pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.io.auxinfo import AuxInfo
from acetree_py.naming.identity import MANUAL, NEWCANONICAL, IdentityAssigner


def _make_nuc(
    index: int,
    x: int,
    y: int,
    z: float,
    identity: str = "",
    status: int = 1,
    pred: int = NILLI,
    succ1: int = NILLI,
    succ2: int = NILLI,
    assigned_id: str = "",
) -> Nucleus:
    return Nucleus(
        index=index, x=x, y=y, z=z, size=20,
        identity=identity, status=status,
        predecessor=pred, successor1=succ1, successor2=succ2,
        weight=5000, assigned_id=assigned_id,
    )


def _make_simple_lineage() -> list[list[Nucleus]]:
    """Build a simple 3-timepoint lineage: P0 -> AB + P1.

    Timepoint 0: P0 (dividing into two at t1)
    Timepoint 1: AB (succ1 at t0) and P1 (succ2 at t0)
    Timepoint 2: AB and P1 continuing (no division)
    """
    # T0: P0 dividing
    p0 = _make_nuc(1, 300, 250, 15.0, identity="P0", succ1=1, succ2=2)

    # T1: AB and P1
    ab = _make_nuc(1, 280, 240, 14.0, identity="AB", pred=1, succ1=1)
    p1 = _make_nuc(2, 320, 260, 16.0, identity="P1", pred=1, succ1=2)

    # T2: AB and P1 continuing
    ab2 = _make_nuc(1, 275, 235, 13.5, identity="", pred=1)
    p1_2 = _make_nuc(2, 325, 265, 16.5, identity="", pred=2)

    return [[p0], [ab, p1], [ab2, p1_2]]


class TestIdentityAssigner:
    """Test the full naming pipeline."""

    def test_manual_method_skips_naming(self):
        """MANUAL naming method should not change any names."""
        nuclei_record = _make_simple_lineage()
        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=MANUAL,
        )
        assigner.assign_identities()
        # Names should be unchanged
        assert nuclei_record[0][0].identity == "P0"

    def test_clear_names_preserves_assigned_id(self):
        """Clear should not remove names backed by assigned_id."""
        nuclei_record = _make_simple_lineage()
        nuclei_record[1][0].assigned_id = "ForcedAB"
        nuclei_record[1][0].identity = "ForcedAB"

        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
        )
        assigner._clear_all_names()

        # Forced name should survive
        assert nuclei_record[1][0].identity == "ForcedAB"
        # Non-forced name should be cleared
        assert nuclei_record[1][1].identity == ""

    def test_generic_naming_inherits_parent_name(self):
        """Non-dividing successors should inherit parent name."""
        nuclei_record = _make_simple_lineage()

        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
        )
        # Call generic naming directly
        assigner._assign_generic_names(0)

        # T2 nuclei should inherit from T1
        assert nuclei_record[2][0].identity == "AB"
        assert nuclei_record[2][1].identity == "P1"

    def test_preassigned_id_honored(self):
        """Forced names should override DivisionCaller assignments."""
        from acetree_py.naming.identity import _use_preassigned_id

        dau1 = _make_nuc(1, 280, 240, 14.0, identity="ABa", assigned_id="ForcedName")
        dau2 = _make_nuc(2, 320, 260, 16.0, identity="ABp")

        _use_preassigned_id(dau1, dau2)

        assert dau1.identity == "ForcedName"
        assert dau2.identity == "ABp"

    def test_preassigned_id_collision_resolved(self):
        """If both daughters get the same name after forcing, append X."""
        from acetree_py.naming.identity import _use_preassigned_id

        dau1 = _make_nuc(1, 280, 240, 14.0, identity="SameName", assigned_id="SameName")
        dau2 = _make_nuc(2, 320, 260, 16.0, identity="SameName", assigned_id="SameName")

        _use_preassigned_id(dau1, dau2)

        assert dau1.identity == "SameName"
        assert dau2.identity == "SameNamX"

    def test_orientation_string_computation(self):
        """Test _compute_orientation helper."""
        from acetree_py.naming.identity import _compute_orientation

        assert _compute_orientation(1, 1, 1) == "ADL"
        assert _compute_orientation(1, -1, -1) == "AVR"
        assert _compute_orientation(-1, 1, -1) == "PDR"
        assert _compute_orientation(-1, -1, 1) == "PVL"


class TestCanonicalNaming:
    """Test the canonical naming path with a synthetic lineage."""

    def _make_dividing_lineage(self) -> list[list[Nucleus]]:
        """Build a lineage where AB divides at t1->t2.

        T0: P0 -> (divides at T1)
        T1: AB, P1
        T2: ABa, ABp, P1 (AB divides)
        """
        p0 = _make_nuc(1, 300, 250, 15.0, identity="P0", succ1=1, succ2=2)

        ab = _make_nuc(1, 280, 240, 14.0, identity="AB", pred=1, succ1=1, succ2=2)
        p1 = _make_nuc(2, 320, 260, 16.0, identity="P1", pred=1, succ1=3)

        aba = _make_nuc(1, 260, 230, 13.0, identity="", pred=1)
        abp = _make_nuc(2, 300, 250, 15.0, identity="", pred=1)
        p1c = _make_nuc(3, 340, 270, 17.0, identity="", pred=2)

        return [[p0], [ab, p1], [aba, abp, p1c]]

    def test_canonical_names_daughters_of_dividing_cell(self):
        """Daughters of a dividing cell should get named via DivisionCaller.

        InitialID would assign early names (P0, AB, P1) before canonical rules run.
        We simulate this by starting canonical rules from T1 (after P0 division),
        with AB and P1 already named.
        """
        nuclei_record = self._make_dividing_lineage()

        # Create a v2 auxinfo-like setup
        auxinfo = AuxInfo(
            version=2,
            data={"AP_orientation": "-1 0 0", "LR_orientation": "0 0 1",
                  "zpixres": "11.1", "name": "test"},
        )

        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            auxinfo=auxinfo,
            naming_method=NEWCANONICAL,
            z_pix_res=11.1,
        )

        # Directly test canonical rules starting from T1 (after early cell ID)
        # T0=P0 already named, T1=AB/P1 already named by InitialID
        assigner._build_canonical_transform()
        assigner._setup_division_caller("")
        assigner._use_canonical_rules(1)  # Start from T1 where AB divides

        # T2 daughters of AB should be named ABa/ABp
        t2_names = {nuclei_record[2][0].identity, nuclei_record[2][1].identity}
        assert "ABa" in t2_names or "ABp" in t2_names

        # P1 should be inherited
        assert nuclei_record[2][2].identity == "P1"


class TestFullPipelineIntegration:
    """End-to-end tests for the naming + lineage tree pipeline.

    Verifies that the topology-based founder ID → back-trace → forward
    canonical rules pipeline correctly names all cells from P0 through
    two generations of divisions, and that the lineage tree has correct
    parent→child linkages with no orphaned cells.
    """

    @staticmethod
    def _build_full_lineage() -> list[list[Nucleus]]:
        """Build a realistic lineage: P0 → AB+P1 → ABa+ABp+EMS+P2 → ...

        Timeline (biologically realistic: AB divides BEFORE P1):
          T0-T1: P0 alone (continuing)
          T2:    P0 divides → AB + P1
          T3:    AB and P1 continue (AB about to divide)
          T4:    AB divides → ABa + ABp;  P1 still continuing  (3 cells)
          T5:    ABa, ABp, P1 continue (P1 about to divide)
          T6:    P1 divides → EMS + P2;  now 4 cells (ABa, ABp, EMS, P2)
          T7-T8: 4-cell stage continues
          T9:    EMS divides → E + MS;  now 5 cells
          T10:   ABa, ABp, E, MS, P2 continue

        Spatial layout:
          - x axis ≈ AP, y axis ≈ DV, z axis ≈ LR
          - AB is anterior (low x), P1 posterior (high x)
          - EMS is larger (size=25) than P2 (size=15) for reliable distinction
        """
        record: list[list[Nucleus]] = []

        # T0: P0
        record.append([
            _make_nuc(1, 300, 250, 15.0, succ1=1),
        ])
        # T1: P0 continuing
        record.append([
            _make_nuc(1, 300, 250, 15.0, pred=1, succ1=1, succ2=2),
        ])
        # T2: AB + P1  (P0 divided)
        record.append([
            _make_nuc(1, 260, 230, 12.0, pred=1, succ1=1),     # AB (anterior)
            _make_nuc(2, 350, 270, 18.0, pred=1, succ1=2),     # P1 (posterior)
        ])
        # T3: AB dividing, P1 continuing
        record.append([
            _make_nuc(1, 258, 228, 12.0, pred=1, succ1=1, succ2=2),  # AB dividing
            _make_nuc(2, 352, 272, 18.0, pred=2, succ1=3),           # P1 continuing
        ])
        # T4: ABa, ABp, P1  (3 cells — AB divided, P1 still alive)
        record.append([
            _make_nuc(1, 230, 215, 9.0, pred=1, succ1=1),     # ABa (anterior, dorsal, left)
            _make_nuc(2, 290, 245, 15.0, pred=1, succ1=2),    # ABp (posterior, ventral, right)
            _make_nuc(3, 354, 274, 18.0, pred=2, succ1=3, succ2=4),  # P1 dividing
        ])
        # T5: ABa, ABp, EMS, P2  (4 cells — P1 divided)
        # Note: EMS gets size=25, P2 gets size=15 for reliable size-based distinction
        record.append([
            _make_nuc(1, 228, 213, 9.0, pred=1, succ1=1),     # ABa
            _make_nuc(2, 292, 247, 15.0, pred=2, succ1=2),    # ABp
            _make_nuc(3, 330, 260, 16.0, pred=3, succ1=3),    # EMS (larger, size overridden below)
            _make_nuc(4, 380, 285, 21.0, pred=3, succ1=4),    # P2 (smaller, size overridden below)
        ])
        # Override sizes for EMS/P2 distinction
        record[5][2].size = 25  # EMS is larger
        record[5][3].size = 15  # P2 is smaller

        # T6: 4-cell stage continues
        record.append([
            _make_nuc(1, 226, 211, 9.0, pred=1, succ1=1),     # ABa
            _make_nuc(2, 294, 249, 15.0, pred=2, succ1=2),    # ABp
            _make_nuc(3, 328, 258, 16.0, pred=3, succ1=3),    # EMS
            _make_nuc(4, 382, 287, 21.0, pred=4, succ1=4),    # P2
        ])
        record[6][2].size = 25
        record[6][3].size = 15

        # T7: 4-cell stage continues; EMS about to divide
        record.append([
            _make_nuc(1, 224, 209, 9.0, pred=1, succ1=1),     # ABa
            _make_nuc(2, 296, 251, 15.0, pred=2, succ1=2),    # ABp
            _make_nuc(3, 326, 256, 16.0, pred=3, succ1=3, succ2=4),  # EMS dividing
            _make_nuc(4, 384, 289, 21.0, pred=4, succ1=5),    # P2
        ])
        record[7][2].size = 25
        record[7][3].size = 15

        # T8: ABa, ABp, E, MS, P2  (5 cells — EMS divided)
        record.append([
            _make_nuc(1, 222, 207, 9.0, pred=1),              # ABa
            _make_nuc(2, 298, 253, 15.0, pred=2),             # ABp
            _make_nuc(3, 318, 252, 15.0, pred=3),             # E (anterior daughter of EMS)
            _make_nuc(4, 340, 264, 17.0, pred=3),             # MS (posterior daughter of EMS)
            _make_nuc(5, 386, 291, 21.0, pred=4),             # P2
        ])

        return record

    def test_back_trace_names_early_cells(self):
        """The back-trace should correctly name P0, AB, P1 from the 4-cell stage."""
        nuclei_record = self._build_full_lineage()

        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
            z_pix_res=11.1,
        )
        assigner.assign_identities()

        # P0 should be named at T0-T1
        assert nuclei_record[0][0].identity == "P0", \
            f"T0 should be P0, got '{nuclei_record[0][0].identity}'"
        assert nuclei_record[1][0].identity == "P0"

        # AB and P1 at T2-T3
        t2_names = {nuclei_record[2][0].identity, nuclei_record[2][1].identity}
        assert "AB" in t2_names, f"T2 names: {t2_names}"
        assert "P1" in t2_names, f"T2 names: {t2_names}"

    def test_four_cell_stage_named(self):
        """ABa, ABp, EMS, P2 should all be named at the 4-cell midpoint."""
        nuclei_record = self._build_full_lineage()

        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
            z_pix_res=11.1,
        )
        assigner.assign_identities()

        # T5 is first 4-cell timepoint
        t5_names = {n.identity for n in nuclei_record[5]}
        assert "ABa" in t5_names, f"T5 names: {t5_names}"
        assert "ABp" in t5_names, f"T5 names: {t5_names}"
        assert "EMS" in t5_names, f"T5 names: {t5_names}"
        assert "P2" in t5_names, f"T5 names: {t5_names}"

    def test_no_nuc_filler_names_for_known_cells(self):
        """No cell that should have a Sulston name should be stuck with a Nuc filler."""
        nuclei_record = self._build_full_lineage()

        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
            z_pix_res=11.1,
        )
        assigner.assign_identities()

        # Check all timepoints — no alive nucleus should have a Nuc* name
        for t, nuclei in enumerate(nuclei_record):
            for j, nuc in enumerate(nuclei):
                if nuc.status >= 1:
                    assert not nuc.identity.startswith("Nuc"), \
                        f"T{t}[{j}] has filler name '{nuc.identity}'"

    def test_forward_pass_names_ems_daughters(self):
        """The forward pass from four_cell_time should correctly name E and MS."""
        nuclei_record = self._build_full_lineage()

        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
            z_pix_res=11.1,
        )
        assigner.assign_identities()

        # T8 should have E and MS (daughters of EMS division at T7→T8)
        t8_names = {n.identity for n in nuclei_record[8]}
        assert "E" in t8_names or "MS" in t8_names, \
            f"T8 names: {t8_names} — expected E and MS from EMS division"

    def test_lineage_tree_linkages(self):
        """The lineage tree should have correct parent→child chains."""
        from acetree_py.core.lineage import build_lineage_tree

        nuclei_record = self._build_full_lineage()

        # First run naming
        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
            z_pix_res=11.1,
        )
        assigner.assign_identities()

        # Then build lineage tree
        tree = build_lineage_tree(nuclei_record)

        # P0 should be root
        assert tree.root is not None
        assert tree.root.name == "P0"

        # P0 → AB + P1
        p0_children = {c.name for c in tree.root.children}
        assert "AB" in p0_children, f"P0 children: {p0_children}"
        assert "P1" in p0_children, f"P0 children: {p0_children}"

        # AB → ABa + ABp
        ab = tree.get_cell("AB")
        assert ab is not None
        ab_children = {c.name for c in ab.children}
        assert "ABa" in ab_children, f"AB children: {ab_children}"
        assert "ABp" in ab_children, f"AB children: {ab_children}"

        # P1 → EMS + P2
        p1 = tree.get_cell("P1")
        assert p1 is not None
        p1_children = {c.name for c in p1.children}
        assert "EMS" in p1_children, f"P1 children: {p1_children}"
        assert "P2" in p1_children, f"P1 children: {p1_children}"

        # EMS → E + MS
        ems = tree.get_cell("EMS")
        assert ems is not None
        ems_children = {c.name for c in ems.children}
        assert "E" in ems_children or "MS" in ems_children, \
            f"EMS children: {ems_children}"

    def test_lineage_tree_no_orphaned_real_cells(self):
        """Real cells with nuclei data should not be orphaned (parentless)."""
        from acetree_py.core.lineage import build_lineage_tree

        nuclei_record = self._build_full_lineage()

        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
            z_pix_res=11.1,
        )
        assigner.assign_identities()

        tree = build_lineage_tree(nuclei_record)

        # Every cell with real nuclei data (except root) should have a parent
        for name, cell in tree.cells_by_name.items():
            if len(cell.nuclei) > 0 and cell != tree.root:
                assert cell.parent is not None, \
                    f"Cell '{name}' has {len(cell.nuclei)} nuclei but no parent (orphaned)"
