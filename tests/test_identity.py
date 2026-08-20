"""Tests for acetree_py.naming.identity — the full naming pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import acetree_py.naming.identity as identity_module
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

    def test_partial_dataset_rebuild_preserves_existing_names(self):
        nuclei_record = _make_simple_lineage()
        assigner = IdentityAssigner(
            nuclei_record=nuclei_record,
            naming_method=NEWCANONICAL,
        )

        assigner.assign_identities()

        assert nuclei_record[0][0].identity == "P0"
        assert nuclei_record[1][0].identity == "AB"
        assert nuclei_record[1][1].identity == "P1"
        assert nuclei_record[2][0].identity == "AB"
        assert nuclei_record[2][1].identity == "P1"

    def test_founders_without_full_frame_do_not_use_lab_space_for_daughters(
        self, monkeypatch,
    ):
        import numpy as np
        import acetree_py.naming.identity as identity_module
        from acetree_py.naming.founder_id import FounderAssignment

        # Four trusted topology anchors are deliberately collinear, so AP is
        # available but DV/LR are anatomically unknowable.
        record = [[
            _make_nuc(1, 0, 0, 0, succ1=1),
            _make_nuc(2, 10, 0, 0, succ1=2),
            _make_nuc(3, 20, 0, 0, succ1=3, succ2=4),
            _make_nuc(4, 30, 0, 0, succ1=5),
        ], [
            _make_nuc(1, 0, 0, 0, pred=1),
            _make_nuc(2, 10, 0, 0, pred=2),
            _make_nuc(3, 18, 0, 0, identity="trusted-loaded-E", pred=3),
            _make_nuc(4, 22, 0, 0, pred=3),
            _make_nuc(5, 30, 0, 0, pred=4),
        ]]

        def fake_identify(nuclei_record, **_kwargs):
            for nuc, name in zip(
                nuclei_record[0], ("ABa", "ABp", "EMS", "P2"),
            ):
                nuc.identity = name
            return FounderAssignment(
                success=True,
                confidence=0.5,
                four_cell_time=0,
                aba_idx=0,
                abp_idx=1,
                ems_idx=2,
                p2_idx=3,
                ap_vector=np.array([-1.0, 0.0, 0.0]),
                lr_vector=None,
                dv_vector=None,
                timing_confidence=1.0,
                size_confidence=1.0,
                axis_confidence=0.0,
            )

        monkeypatch.setattr(identity_module, "identify_founders", fake_identify)
        assigner = IdentityAssigner(record, naming_method=NEWCANONICAL)

        assigner.assign_identities()

        assert assigner.division_caller is None
        assert [n.effective_name for n in record[0]] == [
            "ABa", "ABp", "EMS", "P2",
        ]
        assert record[1][2].effective_name == "trusted-loaded-E"
        assert record[1][3].effective_name.startswith("Nuc")
        assert not record[1][3].effective_name.startswith(("E", "MS"))
        assert any("no complete AP/DV/LR frame" in w for w in assigner.founder_assignment.warnings)

    def test_preassigned_id_honored(self):
        """Forced names should override DivisionCaller assignments."""
        from acetree_py.naming.identity import _use_preassigned_id

        dau1 = _make_nuc(1, 280, 240, 14.0, identity="ABa", assigned_id="ForcedName")
        dau2 = _make_nuc(2, 320, 260, 16.0, identity="ABp")

        _use_preassigned_id(dau1, dau2)

        assert dau1.identity == "ForcedName"
        assert dau2.identity == "ABp"

    def test_duplicate_forced_names_remain_visible_for_validation(self):
        """Automation must not hide an explicit conflict with an identity-only alias."""
        from acetree_py.naming.identity import _use_preassigned_id

        dau1 = _make_nuc(1, 280, 240, 14.0, identity="SameName", assigned_id="SameName")
        dau2 = _make_nuc(2, 320, 260, 16.0, identity="SameName", assigned_id="SameName")

        _use_preassigned_id(dau1, dau2)

        assert dau1.effective_name == "SameName"
        assert dau2.effective_name == "SameName"

    def test_one_forced_daughter_can_take_sisters_automatic_name(self):
        from acetree_py.naming.identity import _use_preassigned_id

        dau1 = _make_nuc(1, 280, 240, 14.0, identity="ABa", assigned_id="ABp")
        dau2 = _make_nuc(2, 320, 260, 16.0, identity="ABp")

        _use_preassigned_id(dau1, dau2)

        assert dau1.effective_name == "ABp"
        assert dau2.effective_name == "ABa"

    def test_forced_name_propagation_stops_at_dead_or_nonreciprocal_link(self):
        record = [
            [_make_nuc(1, 0, 0, 0, succ1=1, assigned_id="curated")],
            [_make_nuc(1, 0, 0, 0, pred=2, succ1=1)],
            [_make_nuc(1, 0, 0, 0, pred=1, status=-1)],
        ]
        assigner = IdentityAssigner(record, naming_method=MANUAL)

        assigner.assign_identities()

        assert record[0][0].assigned_id == "curated"
        assert record[1][0].assigned_id == ""
        assert record[2][0].assigned_id == ""

    def test_conflicting_forced_anchors_are_not_overwritten(self):
        record = [
            [_make_nuc(1, 0, 0, 0, succ1=1, assigned_id="first")],
            [_make_nuc(1, 0, 0, 0, pred=1, succ1=1)],
            [_make_nuc(1, 0, 0, 0, pred=1, assigned_id="second")],
        ]
        assigner = IdentityAssigner(record, naming_method=MANUAL)

        assigner.assign_identities()

        assert record[0][0].assigned_id == "first"
        assert record[2][0].assigned_id == "second"

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


def _axis_configuration(mode: str) -> AuxInfo | None:
    if mode == "inferred":
        return None
    if mode == "v2":
        return AuxInfo(
            version=2,
            data={
                "AP_orientation": "-1 0 0",
                "LR_orientation": "0 0 1",
                "zpixres": "11.1",
                "name": "forced-founder-test",
            },
        )
    return AuxInfo(version=1, data={"axis": "ADL", "ang": "0"})


def _false_four_object_two_cell_stage() -> list[list[Nucleus]]:
    """Two blastomeres plus two small polar-body false detections."""
    ab_candidate = _make_nuc(1, 80, 100, 10.0, identity="ABa")
    p1_candidate = _make_nuc(2, 160, 100, 10.0, identity="ABp")
    first_polar = _make_nuc(3, 110, 70, 10.0, identity="EMS")
    second_polar = _make_nuc(4, 120, 75, 10.0, identity="P2")
    ab_candidate.size = 30
    p1_candidate.size = 22
    first_polar.size = 6
    second_polar.size = 5
    return [[ab_candidate, p1_candidate, first_polar, second_polar]]


@pytest.mark.parametrize("axis_mode", ["inferred", "v2", "v1"])
def test_deleting_two_polar_objects_reconciles_false_four_cell_names(
    axis_mode: str,
):
    """A structural 4-object -> 2-cell correction must invalidate old labels."""
    from acetree_py.editing.commands import RemoveNucleus
    from acetree_py.editing.history import EditHistory

    record = _false_four_object_two_cell_stage()
    original_names = [
        (nucleus.identity, nucleus.assigned_id, nucleus.status)
        for nucleus in record[0]
    ]
    assignments: list[IdentityAssigner] = []

    def rebuild_names() -> None:
        assigner = IdentityAssigner(
            record,
            auxinfo=_axis_configuration(axis_mode),
            naming_method=NEWCANONICAL,
            z_pix_res=1.0,
        )
        assigner.assign_identities()
        assignments.append(assigner)

    history = EditHistory(record, on_edit=rebuild_names)
    history.do(RemoveNucleus(time=1, index=3))
    after_first_delete = [
        (nucleus.identity, nucleus.assigned_id, nucleus.status)
        for nucleus in record[0]
    ]
    history.do(RemoveNucleus(time=1, index=4))
    after_second_delete = [
        (nucleus.identity, nucleus.assigned_id, nucleus.status)
        for nucleus in record[0]
    ]

    alive = [nucleus for nucleus in record[0] if nucleus.is_alive]
    assert {nucleus.identity for nucleus in alive} == {"AB", "P1"}
    assert all(nucleus.assigned_id == "" for nucleus in alive)
    assert all(
        nucleus.identity == "" and nucleus.assigned_id == ""
        for nucleus in record[0][2:]
    )
    assert any(
        "curated two-cell stage" in warning
        for warning in assignments[-1].founder_assignment.warnings
    )

    # AP is posterior -> anterior.  v2 points toward -X; v1 ADL points +X.
    expected_ab_index = 1 if axis_mode == "v1" else 0
    assert record[0][expected_ab_index].identity == "AB"

    # Every automatic naming side effect belongs to the same history boundary.
    # Undoing both deletions therefore restores the exact persisted state, not
    # merely the original live/dead count.
    history.undo()
    assert [
        (nucleus.identity, nucleus.assigned_id, nucleus.status)
        for nucleus in record[0]
    ] == after_first_delete
    history.undo()
    assert [
        (nucleus.identity, nucleus.assigned_id, nucleus.status)
        for nucleus in record[0]
    ] == original_names
    assert not history.modified

    history.redo()
    assert [
        (nucleus.identity, nucleus.assigned_id, nucleus.status)
        for nucleus in record[0]
    ] == after_first_delete
    history.redo()
    assert [
        (nucleus.identity, nucleus.assigned_id, nucleus.status)
        for nucleus in record[0]
    ] == after_second_delete


def test_two_cell_recovery_prefers_future_division_timing_without_axes():
    """In inferred mode, the lineage that divides first is AB."""
    first = _make_nuc(1, 80, 100, 10.0, identity="ABa", succ1=1)
    second = _make_nuc(2, 160, 100, 10.0, identity="ABp", succ1=2)
    first.size = second.size = 20
    record = [[
        first,
        second,
        _make_nuc(3, 110, 70, 10.0, status=-1),
        _make_nuc(4, 120, 75, 10.0, status=-1),
    ], [
        _make_nuc(1, 75, 100, 10.0, pred=1, succ1=1, succ2=3),
        _make_nuc(2, 165, 100, 10.0, pred=2, succ1=2),
    ], [
        _make_nuc(1, 65, 100, 10.0, pred=1),
        _make_nuc(2, 170, 100, 10.0, pred=2),
        _make_nuc(3, 85, 100, 10.0, pred=1),
    ]]
    record[0][2].size = 5
    record[0][3].size = 5

    IdentityAssigner(record, naming_method=NEWCANONICAL).assign_identities()

    assert record[0][0].identity == "AB"
    assert record[0][1].identity == "P1"
    assert record[1][0].identity == "AB"
    assert record[1][1].identity == "P1"
    assert record[2][1].identity == "P1"
    assert record[2][0].identity.startswith("Nuc")
    assert record[2][2].identity.startswith("Nuc")


def test_two_cell_recovery_uses_symmetric_future_division_timing():
    """The second candidate is AB when its valid division occurs first."""
    first = _make_nuc(1, 80, 100, 10.0, identity="ABa", succ1=1)
    second = _make_nuc(2, 160, 100, 10.0, identity="ABp", succ1=2)
    first.size = second.size = 20
    record = [[
        first,
        second,
        _make_nuc(3, 110, 70, 10.0, status=-1),
        _make_nuc(4, 120, 75, 10.0, status=-1),
    ], [
        _make_nuc(1, 75, 100, 10.0, pred=1, succ1=1),
        _make_nuc(2, 165, 100, 10.0, pred=2, succ1=2, succ2=3),
    ], [
        _make_nuc(1, 70, 100, 10.0, pred=1),
        _make_nuc(2, 155, 100, 10.0, pred=2),
        _make_nuc(3, 175, 100, 10.0, pred=2),
    ]]
    record[0][2].size = 5
    record[0][3].size = 5

    IdentityAssigner(record, naming_method=NEWCANONICAL).assign_identities()

    assert record[0][0].identity == "P1"
    assert record[0][1].identity == "AB"


def test_right_censored_sister_is_not_assumed_to_divide_later():
    """A track ending at the observed division time supplies no ordering."""
    first = _make_nuc(1, 80, 100, 10.0, identity="ABa", succ1=1)
    second = _make_nuc(2, 160, 100, 10.0, identity="ABp", succ1=2)
    first.size = second.size = 20
    record = [[
        first,
        second,
        _make_nuc(3, 110, 70, 10.0, status=-1),
        _make_nuc(4, 120, 75, 10.0, status=-1),
    ], [
        _make_nuc(1, 75, 100, 10.0, pred=1, succ1=1, succ2=2),
        _make_nuc(2, 165, 100, 10.0, pred=2),
    ], [
        _make_nuc(1, 65, 100, 10.0, pred=1),
        _make_nuc(2, 85, 100, 10.0, pred=1),
    ]]
    record[0][2].size = 5
    record[0][3].size = 5

    assigner = IdentityAssigner(record, naming_method=NEWCANONICAL)
    assigner.assign_identities()

    assert record[0][0].identity.startswith("Nuc")
    assert record[0][1].identity.startswith("Nuc")
    assert any(
        "ordering remains ambiguous" in warning
        for warning in assigner.founder_assignment.warnings
    )


def _two_cell_stage_with_divisions() -> list[list[Nucleus]]:
    record = _false_four_object_two_cell_stage()
    record[0][2].status = -1
    record[0][2].identity = ""
    record[0][3].status = -1
    record[0][3].identity = ""
    record[0][0].successor1 = 1
    record[0][0].successor2 = 2
    record[0][1].successor1 = 3
    record[0][1].successor2 = 4
    record.append([
        _make_nuc(1, 60, 100, 10.0, pred=1),
        _make_nuc(2, 100, 100, 10.0, pred=1),
        _make_nuc(3, 140, 100, 10.0, pred=2),
        _make_nuc(4, 180, 100, 10.0, pred=2),
    ])
    return record


@pytest.mark.parametrize("axis_mode", ["v2", "v1"])
def test_explicit_axes_regenerate_descendants_after_two_cell_recovery(
    axis_mode: str,
):
    record = _two_cell_stage_with_divisions()

    IdentityAssigner(
        record,
        auxinfo=_axis_configuration(axis_mode),
        naming_method=NEWCANONICAL,
        z_pix_res=1.0,
    ).assign_identities()

    assert {record[0][0].identity, record[0][1].identity} == {"AB", "P1"}
    ab_parent = next(nucleus for nucleus in record[0] if nucleus.identity == "AB")
    p1_parent = next(nucleus for nucleus in record[0] if nucleus.identity == "P1")
    assert {
        record[1][successor - 1].identity
        for successor in (ab_parent.successor1, ab_parent.successor2)
    } == {"ABa", "ABp"}
    assert {
        record[1][successor - 1].identity
        for successor in (p1_parent.successor1, p1_parent.successor2)
    } == {"EMS", "P2"}
    assert all(
        nucleus.assigned_id == ""
        for timepoint in record
        for nucleus in timepoint
    )


def test_forced_post_division_descendant_survives_two_cell_recovery():
    record = _false_four_object_two_cell_stage()
    record[0][2].status = -1
    record[0][2].identity = ""
    record[0][3].status = -1
    record[0][3].identity = ""
    record[0][0].successor1 = 1
    record[0][0].successor2 = 2
    record[0][1].successor1 = 3
    record.append([
        _make_nuc(1, 60, 100, 10.0, pred=1),
        _make_nuc(2, 100, 100, 10.0, pred=1),
        _make_nuc(3, 160, 100, 10.0, pred=2),
    ])
    record[1][0].identity = "ABa"
    record[1][0].assigned_id = "ABa"

    assigner = IdentityAssigner(
        record,
        auxinfo=_axis_configuration("v2"),
        naming_method=NEWCANONICAL,
        z_pix_res=1.0,
    )
    assigner.assign_identities()

    assert {record[0][0].identity, record[0][1].identity} == {"AB", "P1"}
    assert record[1][0].identity == "ABa"
    assert record[1][0].assigned_id == "ABa"
    assert any(
        "curated two-cell stage" in warning
        for warning in assigner.founder_assignment.warnings
    )


def test_malformed_candidate_links_do_not_drive_two_cell_recovery():
    record = _false_four_object_two_cell_stage()
    record[0][0].successor1 = 99
    record[0][2].status = -1
    record[0][2].identity = ""
    record[0][3].status = -1
    record[0][3].identity = ""

    IdentityAssigner(record, naming_method=NEWCANONICAL).assign_identities()

    assert {record[0][0].identity, record[0][1].identity} == {"ABa", "ABp"}


def test_reverse_only_candidate_link_does_not_drive_two_cell_recovery():
    """A child claiming an undeclared predecessor makes topology invalid."""
    record = _false_four_object_two_cell_stage()
    record[0][2].status = -1
    record[0][2].identity = ""
    record[0][3].status = -1
    record[0][3].identity = ""
    record.append([
        _make_nuc(1, 75, 100, 10.0, pred=1),
    ])

    IdentityAssigner(record, naming_method=NEWCANONICAL).assign_identities()

    assert {record[0][0].identity, record[0][1].identity} == {"ABa", "ABp"}


def test_ambiguous_four_to_two_correction_discards_impossible_founder_names():
    record = _false_four_object_two_cell_stage()
    record[0][0].size = 20
    record[0][1].size = 20
    record[0][2].size = 5
    record[0][3].size = 5
    record[0][2].status = -1
    record[0][2].identity = ""
    record[0][3].status = -1
    record[0][3].identity = ""

    assigner = IdentityAssigner(record, naming_method=NEWCANONICAL)
    assigner.assign_identities()

    assert all(
        nucleus.identity.startswith("Nuc")
        for nucleus in record[0]
        if nucleus.is_alive
    )
    assert any(
        "ordering remains ambiguous" in warning
        for warning in assigner.founder_assignment.warnings
    )


@pytest.mark.parametrize("axis_mode", ["inferred", "v2", "v1"])
def test_real_four_cell_ablation_is_not_reinterpreted(axis_mode: str):
    """Normal-sized dead founder rows are not evidence of polar bodies."""
    record = [[
        _make_nuc(1, 80, 100, 10.0, identity="ABa"),
        _make_nuc(2, 120, 100, 10.0, identity="ABp", status=-1),
        _make_nuc(3, 160, 100, 10.0, identity="EMS"),
        _make_nuc(4, 200, 100, 10.0, identity="P2", status=-1),
    ]]
    for nucleus in record[0]:
        nucleus.size = 20
    record[0][1].identity = ""
    record[0][3].identity = ""

    IdentityAssigner(
        record,
        auxinfo=_axis_configuration(axis_mode),
        naming_method=NEWCANONICAL,
        z_pix_res=1.0,
    ).assign_identities()

    assert {record[0][0].identity, record[0][2].identity} == {"ABa", "EMS"}


@pytest.mark.parametrize("axis_mode", ["inferred", "v2", "v1"])
def test_true_four_cell_parent_topology_vetoes_small_ablation_recovery(
    axis_mode: str,
):
    """Two small killed founders remain a real four-cell-stage ablation."""
    record = [[
        # Successors model the post-delete rebuild: only live daughters remain
        # in these slots, while dead rows retain their predecessor values.
        _make_nuc(1, 100, 100, 10.0, identity="AB", succ1=1),
        _make_nuc(2, 180, 100, 10.0, identity="P1", succ1=3),
    ], [
        _make_nuc(1, 80, 100, 10.0, identity="ABa", pred=1),
        _make_nuc(2, 120, 100, 10.0, status=-1, pred=1),
        _make_nuc(3, 160, 100, 10.0, identity="EMS", pred=2),
        _make_nuc(4, 200, 100, 10.0, status=-1, pred=2),
    ]]
    record[1][0].size = 30
    record[1][1].size = 5
    record[1][2].size = 22
    record[1][3].size = 5

    IdentityAssigner(
        record,
        auxinfo=_axis_configuration(axis_mode),
        naming_method=NEWCANONICAL,
        z_pix_res=1.0,
    ).assign_identities()

    assert record[0][0].identity == "AB"
    assert record[0][1].identity == "P1"
    assert record[1][0].identity == "ABa"
    assert record[1][2].identity == "EMS"
    assert record[1][1].identity == ""
    assert record[1][3].identity == ""


def test_true_four_cell_topology_vetoes_an_all_dead_sister_pair():
    """A parent may have no rebuilt successors when both daughters were killed."""
    record = [[
        _make_nuc(1, 100, 100, 10.0, identity="AB", succ1=1, succ2=2),
        _make_nuc(2, 180, 100, 10.0, identity="P1"),
    ], [
        _make_nuc(1, 80, 100, 10.0, identity="ABa", pred=1),
        _make_nuc(2, 120, 100, 10.0, identity="ABp", pred=1),
        _make_nuc(3, 160, 100, 10.0, status=-1, pred=2),
        _make_nuc(4, 200, 100, 10.0, status=-1, pred=2),
    ]]
    record[1][0].size = 30
    record[1][1].size = 22
    record[1][2].size = 5
    record[1][3].size = 5

    IdentityAssigner(
        record,
        naming_method=NEWCANONICAL,
        starting_index=1,
    ).assign_identities()

    assert record[1][0].identity == "ABa"
    assert record[1][1].identity == "ABp"


def test_true_four_cell_continuations_veto_late_small_ablation_recovery():
    """The four-cell-stage birth topology remains authoritative in later frames."""
    record = [[
        _make_nuc(1, 100, 100, 10.0, identity="AB", succ1=1, succ2=2),
        _make_nuc(2, 180, 100, 10.0, identity="P1", succ1=3, succ2=4),
    ], [
        _make_nuc(1, 80, 100, 10.0, identity="ABa", pred=1, succ1=1),
        _make_nuc(2, 120, 100, 10.0, identity="ABp", pred=1, succ1=2),
        _make_nuc(3, 160, 100, 10.0, identity="EMS", pred=2),
        _make_nuc(4, 200, 100, 10.0, identity="P2", pred=2),
    ], [
        _make_nuc(1, 78, 100, 10.0, identity="ABa", pred=1),
        _make_nuc(2, 122, 100, 10.0, identity="ABp", pred=2),
        _make_nuc(3, 158, 100, 10.0, status=-1, pred=3),
        _make_nuc(4, 202, 100, 10.0, status=-1, pred=4),
    ]]
    record[2][0].size = 30
    record[2][1].size = 22
    record[2][2].size = 5
    record[2][3].size = 5

    IdentityAssigner(
        record,
        naming_method=NEWCANONICAL,
        starting_index=2,
    ).assign_identities()

    assert record[2][0].identity == "ABa"
    assert record[2][1].identity == "ABp"


def test_malformed_four_cell_parent_claim_blocks_polar_recovery():
    """Corrupt reciprocal metadata is ambiguity, not permission to relabel."""
    record = [[
        _make_nuc(1, 100, 100, 10.0, identity="AB", succ1=1, succ2=99),
        _make_nuc(2, 180, 100, 10.0, identity="P1", succ1=3, succ2=98),
    ], [
        _make_nuc(1, 80, 100, 10.0, identity="ABa", pred=1),
        _make_nuc(2, 120, 100, 10.0, status=-1, pred=1),
        _make_nuc(3, 160, 100, 10.0, identity="EMS", pred=2),
        _make_nuc(4, 200, 100, 10.0, status=-1, pred=2),
    ]]
    record[1][0].size = 30
    record[1][1].size = 5
    record[1][2].size = 22
    record[1][3].size = 5

    IdentityAssigner(record, naming_method=NEWCANONICAL).assign_identities()

    assert record[1][0].identity == "ABa"
    assert record[1][2].identity == "EMS"


def test_duplicate_prior_founder_labels_are_not_promoted_to_ab_and_p1():
    record = _false_four_object_two_cell_stage()
    record[0][0].identity = "ABa"
    record[0][1].identity = "ABa"
    record[0][2].status = -1
    record[0][2].identity = ""
    record[0][3].status = -1
    record[0][3].identity = ""

    IdentityAssigner(record, naming_method=NEWCANONICAL).assign_identities()

    assert {record[0][0].identity, record[0][1].identity} != {"AB", "P1"}


def test_polar_size_cue_outranks_rejected_four_cell_family_labels():
    record = _false_four_object_two_cell_stage()
    record[0][0].identity = "P2"
    record[0][1].identity = "ABa"
    record[0][2].status = -1
    record[0][2].identity = ""
    record[0][3].status = -1
    record[0][3].identity = ""

    IdentityAssigner(record, naming_method=NEWCANONICAL).assign_identities()

    assert record[0][0].size > record[0][1].size
    assert record[0][0].identity == "AB"
    assert record[0][1].identity == "P1"


def test_two_live_late_stage_names_are_not_reinterpreted_as_ab_and_p1():
    record = [[
        _make_nuc(1, 80, 100, 10.0, identity="E"),
        _make_nuc(2, 160, 100, 10.0, identity="MS"),
        _make_nuc(3, 110, 70, 10.0, status=-1),
        _make_nuc(4, 120, 75, 10.0, status=-1),
    ]]

    IdentityAssigner(record, naming_method=NEWCANONICAL).assign_identities()

    assert {record[0][0].identity, record[0][1].identity} == {"E", "MS"}


@pytest.mark.parametrize("axis_mode", ["inferred", "v2", "v1"])
def test_forced_early_parents_reconcile_progeny_for_every_axis_mode(axis_mode: str):
    record = TestFullPipelineIntegration._build_full_lineage()
    # Deliberately exchange the topology heuristic's AB/P1 roles.  These are
    # cell-scoped curator anchors on the two-cell-stage continuations.
    record[2][0].assigned_id = "P1"
    record[2][0].identity = "P1"
    record[2][1].assigned_id = "AB"
    record[2][1].identity = "AB"

    assigner = IdentityAssigner(
        record,
        auxinfo=_axis_configuration(axis_mode),
        naming_method=NEWCANONICAL,
        z_pix_res=11.1,
    )
    assigner.assign_identities()

    assignment = assigner.founder_assignment
    assert assignment is not None and assignment.success
    assert not assignment.constraint_conflict
    assert {assignment.ems_idx, assignment.p2_idx} == {0, 1}
    assert {assignment.aba_idx, assignment.abp_idx} == {2, 3}
    assert {record[4][0].identity, record[4][1].identity} == {"EMS", "P2"}
    assert {record[6][2].identity, record[6][3].identity} == {"ABa", "ABp"}

    expected = np.array([
        record[assignment.four_cell_time][assignment.aba_idx].x
        - record[assignment.four_cell_time][assignment.p2_idx].x,
        record[assignment.four_cell_time][assignment.aba_idx].y
        - record[assignment.four_cell_time][assignment.p2_idx].y,
        (
            record[assignment.four_cell_time][assignment.aba_idx].z
            - record[assignment.four_cell_time][assignment.p2_idx].z
        ) * 11.1,
    ])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(assignment.ap_vector, expected)

    assert assigner.division_caller is not None
    if axis_mode == "inferred":
        assert assigner.division_caller.is_lineage_mode
        lineage_map = assigner.division_caller._lineage_map
        assert lineage_map is not None
        assert lineage_map[assignment.four_cell_time][assignment.aba_idx] == "ABa"
        assert lineage_map[assignment.four_cell_time][assignment.ems_idx] == "EMS"
    elif axis_mode == "v2":
        assert assigner.division_caller.is_v2
        assert not assigner.division_caller.is_lineage_mode
    else:
        assert not assigner.division_caller.is_v2
        assert not assigner.division_caller.is_lineage_mode


def _partial_forced_parent(auxinfo: AuxInfo | None) -> list[list[Nucleus]]:
    parent = _make_nuc(
        1, 100, 100, 10.0, identity="AB", assigned_id="AB",
        succ1=1, succ2=2,
    )
    first = _make_nuc(1, 80, 100, 10.0, identity="EMS", pred=1)
    second = _make_nuc(2, 120, 100, 10.0, identity="P2", pred=1)
    record = [[parent], [first, second]]
    IdentityAssigner(record, auxinfo=auxinfo, z_pix_res=1.0).assign_identities()
    return record


def test_partial_movie_without_axes_clears_stale_automatic_progeny():
    record = _partial_forced_parent(None)

    assert record[0][0].effective_name == "AB"
    assert all(nuc.assigned_id == "" for nuc in record[1])
    assert all(nuc.identity.startswith("Nuc") for nuc in record[1])
    assert {nuc.identity for nuc in record[1]}.isdisjoint({"EMS", "P2", "ABa", "ABp"})


@pytest.mark.parametrize("axis_mode", ["v2", "v1"])
def test_partial_movie_explicit_axes_recompute_forced_parent_progeny(axis_mode: str):
    record = _partial_forced_parent(_axis_configuration(axis_mode))
    assert {nuc.identity for nuc in record[1]} == {"ABa", "ABp"}


def test_post_founder_forced_parent_still_drives_canonical_daughters():
    auxinfo = _axis_configuration("v2")
    parent = _make_nuc(
        1, 100, 100, 10.0, identity="ABal", assigned_id="ABal",
        succ1=1, succ2=2,
    )
    record = [[parent], [
        _make_nuc(1, 80, 100, 10.0, identity="OldA", pred=1),
        _make_nuc(2, 120, 100, 10.0, identity="OldB", pred=1),
    ]]

    IdentityAssigner(record, auxinfo=auxinfo, z_pix_res=1.0).assign_identities()

    assert {nuc.identity for nuc in record[1]} == {"ABala", "ABalp"}


def test_legacy_initial_id_cannot_overwrite_forced_parent_before_division(
    monkeypatch,
):
    parent = _make_nuc(
        1, 100, 100, 10.0, identity="AB", assigned_id="AB",
        succ1=1, succ2=2,
    )
    record = [[parent], [
        _make_nuc(1, 80, 100, 10.0, pred=1),
        _make_nuc(2, 120, 100, 10.0, pred=1),
    ]]

    def overwriting_initial_id(nuclei_record, **_kwargs):
        nuclei_record[0][0].identity = "P1"
        return SimpleNamespace(axis_found=True, start_index=0, ap=1, dv=1, lr=1)

    monkeypatch.setattr(identity_module, "identify_initial_cells", overwriting_initial_id)
    assigner = IdentityAssigner(
        record,
        auxinfo=_axis_configuration("v2"),
        z_pix_res=1.0,
        legacy_mode=True,
    )

    assigner.assign_identities()

    assert record[0][0].identity == "AB"
    assert {nuc.identity for nuc in record[1]} == {"ABa", "ABp"}
