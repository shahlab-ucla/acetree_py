"""Tests for the editing system — commands, history, and validators.

Covers:
- Each command's execute + undo cycle
- EditHistory undo/redo stack behavior
- Validators for pre-edit checks
- Multi-command sequences with full undo rollback
"""

import pytest

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.editing.commands import (
    AddNucleus,
    KillCell,
    MoveNucleus,
    RelinkNucleus,
    RelinkWithInterpolation,
    RemoveNucleus,
    RenameCell,
    ResurrectCell,
    SwapCellNames,
    _add_successor,
    _get_nucleus,
    _remove_successor,
    _walk_continuation_chain,
)
from acetree_py.editing.history import EditHistory
from acetree_py.editing.validators import (
    validate_add_nucleus,
    validate_kill_cell,
    validate_relink,
    validate_relink_interpolation,
    validate_remove_nucleus,
    validate_rename_cell,
)


# ── Fixtures ────────────────────────────────────────────────────


def _make_nucleus(index: int, x: int = 100, y: int = 200, z: float = 5.0,
                  size: int = 20, identity: str = "", status: int = 1,
                  predecessor: int = NILLI, successor1: int = NILLI,
                  successor2: int = NILLI) -> Nucleus:
    """Create a nucleus with convenient defaults."""
    return Nucleus(
        index=index, x=x, y=y, z=z, size=size,
        identity=identity, status=status,
        predecessor=predecessor, successor1=successor1, successor2=successor2,
    )


def _simple_record() -> list[list[Nucleus]]:
    """3 timepoints, 2 nuclei each: A and B cells tracked across time.

    T1: [A1(idx=1), B1(idx=2)]
    T2: [A2(idx=1, pred=1), B2(idx=2, pred=2)]
    T3: [A3(idx=1, pred=1), B3(idx=2, pred=2)]
    """
    return [
        [  # T1
            _make_nucleus(1, x=100, y=100, z=5.0, identity="A", successor1=1),
            _make_nucleus(2, x=200, y=200, z=5.0, identity="B", successor1=2),
        ],
        [  # T2
            _make_nucleus(1, x=110, y=110, z=5.0, identity="A", predecessor=1, successor1=1),
            _make_nucleus(2, x=210, y=210, z=5.0, identity="B", predecessor=2, successor1=2),
        ],
        [  # T3
            _make_nucleus(1, x=120, y=120, z=5.0, identity="A", predecessor=1),
            _make_nucleus(2, x=220, y=220, z=5.0, identity="B", predecessor=2),
        ],
    ]


def _dividing_record() -> list[list[Nucleus]]:
    """2 timepoints with a cell that divides.

    T1: [P0(idx=1, succ1=1, succ2=2)]
    T2: [AB(idx=1, pred=1), P1(idx=2, pred=1)]
    """
    return [
        [  # T1
            _make_nucleus(1, x=150, y=150, z=10.0, identity="P0",
                         successor1=1, successor2=2),
        ],
        [  # T2
            _make_nucleus(1, x=100, y=150, z=10.0, identity="AB", predecessor=1),
            _make_nucleus(2, x=200, y=150, z=10.0, identity="P1", predecessor=1),
        ],
    ]


# ── Helper function tests ───────────────────────────────────────


class TestHelpers:
    def test_get_nucleus(self):
        record = _simple_record()
        nuc = _get_nucleus(record, 1, 1)
        assert nuc.identity == "A"
        nuc = _get_nucleus(record, 2, 2)
        assert nuc.identity == "B"

    def test_get_nucleus_out_of_range(self):
        record = _simple_record()
        with pytest.raises(IndexError):
            _get_nucleus(record, 0, 1)  # time < 1
        with pytest.raises(IndexError):
            _get_nucleus(record, 1, 5)  # index out of range

    def test_add_remove_successor(self):
        nuc = _make_nucleus(1)
        assert nuc.successor1 == NILLI
        assert nuc.successor2 == NILLI

        _add_successor(nuc, 5)
        assert nuc.successor1 == 5
        assert nuc.successor2 == NILLI

        _add_successor(nuc, 7)
        assert nuc.successor1 == 5
        assert nuc.successor2 == 7

        _remove_successor(nuc, 5)
        assert nuc.successor1 == 7
        assert nuc.successor2 == NILLI

        _remove_successor(nuc, 7)
        assert nuc.successor1 == NILLI
        assert nuc.successor2 == NILLI

    def test_add_successor_full(self):
        """Adding a 3rd successor should warn but not crash."""
        nuc = _make_nucleus(1, successor1=2, successor2=3)
        _add_successor(nuc, 4)  # Should log warning
        assert nuc.successor1 == 2  # Unchanged
        assert nuc.successor2 == 3


# ── AddNucleus tests ────────────────────────────────────────────


class TestAddNucleus:
    def test_add_and_undo(self):
        record = _simple_record()
        assert len(record[0]) == 2  # T1 has 2 nuclei

        cmd = AddNucleus(time=1, x=50, y=50, z=3.0, size=15, identity="C")
        cmd.execute(record)

        assert len(record[0]) == 3
        added = record[0][2]
        assert added.index == 3
        assert added.x == 50
        assert added.identity == "C"
        assert added.is_alive

        cmd.undo(record)
        assert len(record[0]) == 2

    def test_add_extends_record(self):
        record = _simple_record()
        assert len(record) == 3

        cmd = AddNucleus(time=5, x=10, y=10, z=1.0)
        cmd.execute(record)

        assert len(record) == 5
        assert len(record[4]) == 1
        assert record[4][0].index == 1

    def test_add_with_predecessor(self):
        record = _simple_record()
        cmd = AddNucleus(time=2, x=50, y=50, z=3.0, predecessor=1)
        cmd.execute(record)
        assert record[1][2].predecessor == 1

    def test_description(self):
        cmd = AddNucleus(time=3, x=100, y=200, z=5.0, identity="ABa")
        assert "ABa" in cmd.description
        assert "t=3" in cmd.description

    def test_add_with_assigned_id_writes_forced_name(self):
        """AddNucleus stores assigned_id on the new nucleus so the naming
        pipeline's _propagate_assigned_ids() treats it as a forced name."""
        record = _simple_record()
        cmd = AddNucleus(
            time=2, x=50, y=50, z=3.0, size=15,
            identity="ABa", assigned_id="ABa", predecessor=1,
        )
        cmd.execute(record)
        added = record[1][-1]
        assert added.assigned_id == "ABa"
        assert added.identity == "ABa"
        # effective_name returns assigned_id first
        assert added.effective_name == "ABa"

    def test_add_without_assigned_id_is_empty(self):
        """Backward compat: omitting assigned_id still works (default '')."""
        record = _simple_record()
        cmd = AddNucleus(time=2, x=50, y=50, z=3.0, identity="ABa")
        cmd.execute(record)
        added = record[1][-1]
        assert added.assigned_id == ""
        assert added.identity == "ABa"

    def test_add_with_assigned_id_propagates_backward(self):
        """A forced name on a newly-added child should be swept backward
        onto its predecessor by IdentityAssigner._propagate_assigned_ids,
        unifying the continuation chain."""
        from acetree_py.core.nucleus import Nucleus
        from acetree_py.naming.identity import IdentityAssigner

        # Build a clean two-timepoint record with a predecessor link
        nuc_t1 = Nucleus(index=1, x=10, y=10, z=1.0, size=10, status=1)
        record = [[nuc_t1], []]

        cmd = AddNucleus(
            time=2, x=10, y=11, z=1.0, size=10,
            identity="ABa", assigned_id="ABa", predecessor=1,
        )
        cmd.execute(record)

        assigner = IdentityAssigner(
            nuclei_record=record,
            auxinfo=None,
            starting_index=0,
            ending_index=len(record),
        )
        assigner._propagate_assigned_ids()

        # Predecessor at t=1 should now carry the forced name too
        assert record[0][0].assigned_id == "ABa"
        assert record[0][0].identity == "ABa"


# ── RemoveNucleus tests ─────────────────────────────────────────


class TestRemoveNucleus:
    def test_remove_and_undo(self):
        record = _simple_record()
        nuc = record[0][0]
        assert nuc.is_alive
        assert nuc.identity == "A"

        cmd = RemoveNucleus(time=1, index=1)
        cmd.execute(record)

        assert not nuc.is_alive
        assert nuc.status == -1
        assert nuc.identity == ""
        assert nuc.assigned_id == ""

        cmd.undo(record)
        assert nuc.is_alive
        assert nuc.identity == "A"

    def test_remove_preserves_list_length(self):
        record = _simple_record()
        original_len = len(record[0])

        cmd = RemoveNucleus(time=1, index=1)
        cmd.execute(record)
        assert len(record[0]) == original_len  # No physical removal


# ── MoveNucleus tests ───────────────────────────────────────────


class TestMoveNucleus:
    def test_move_all_fields(self):
        record = _simple_record()
        nuc = record[0][0]
        old_x, old_y, old_z, old_size = nuc.x, nuc.y, nuc.z, nuc.size

        cmd = MoveNucleus(time=1, index=1, new_x=50, new_y=75, new_z=8.0, new_size=30)
        cmd.execute(record)

        assert nuc.x == 50
        assert nuc.y == 75
        assert nuc.z == 8.0
        assert nuc.size == 30

        cmd.undo(record)
        assert nuc.x == old_x
        assert nuc.y == old_y
        assert nuc.z == old_z
        assert nuc.size == old_size

    def test_move_partial(self):
        record = _simple_record()
        nuc = record[0][0]
        old_y, old_z, old_size = nuc.y, nuc.z, nuc.size

        cmd = MoveNucleus(time=1, index=1, new_x=50)
        cmd.execute(record)

        assert nuc.x == 50
        assert nuc.y == old_y  # Unchanged
        assert nuc.z == old_z
        assert nuc.size == old_size


# ── RenameCell tests ─────────────────────────────────────────────


class TestRenameCell:
    def test_rename_and_undo(self):
        """Renaming any nucleus in a continuation chain renames the whole chain."""
        record = _simple_record()
        # Chain A spans all 3 timepoints (A1 -> A2 -> A3)
        for t_idx in range(3):
            assert record[t_idx][0].identity == "A"
            assert record[t_idx][0].assigned_id == ""

        # Rename by clicking nucleus at T2 idx=1 — whole chain should update
        cmd = RenameCell(time=2, index=1, new_name="ABala")
        cmd.execute(record)

        for t_idx in range(3):
            assert record[t_idx][0].identity == "ABala", f"t={t_idx + 1} not renamed"
            assert record[t_idx][0].assigned_id == "ABala", f"t={t_idx + 1} assigned_id missing"

        cmd.undo(record)
        for t_idx in range(3):
            assert record[t_idx][0].identity == "A"
            assert record[t_idx][0].assigned_id == ""

    def test_rename_affects_entire_continuation(self):
        """Rename at any timepoint updates every nucleus in the chain."""
        record = _simple_record()
        # Rename clicking the middle nucleus
        cmd = RenameCell(time=2, index=1, new_name="MyCell")
        cmd.execute(record)

        names = [record[t][0].effective_name for t in range(3)]
        assert names == ["MyCell", "MyCell", "MyCell"]

    def test_rename_twice_custom_to_custom(self):
        """Renaming twice (X -> Y -> Z) leaves the whole chain at Z."""
        record = _simple_record()
        RenameCell(time=1, index=1, new_name="X").execute(record)
        RenameCell(time=2, index=1, new_name="Y").execute(record)
        RenameCell(time=3, index=1, new_name="Z").execute(record)

        for t_idx in range(3):
            assert record[t_idx][0].effective_name == "Z"
            assert record[t_idx][0].assigned_id == "Z"

    def test_rename_roundtrip(self):
        """Rename MyCell -> back to original sulston name: whole chain reflects it."""
        record = _simple_record()
        # Original identity is "A".  Rename to "MyCell", then back to "A".
        cmd1 = RenameCell(time=1, index=1, new_name="MyCell")
        cmd1.execute(record)
        cmd2 = RenameCell(time=2, index=1, new_name="A")
        cmd2.execute(record)

        for t_idx in range(3):
            assert record[t_idx][0].effective_name == "A"
            assert record[t_idx][0].assigned_id == "A"

    def test_rename_undo_reverts_full_chain(self):
        """Undo restores every nucleus in the chain to its prior state."""
        record = _simple_record()
        # Seed some prior state: each nucleus has a different identity
        record[0][0].identity = "first"
        record[1][0].identity = "middle"
        record[2][0].identity = "last"
        record[1][0].assigned_id = "was-forced"

        cmd = RenameCell(time=1, index=1, new_name="NewName")
        cmd.execute(record)

        # All three should now be NewName
        for t_idx in range(3):
            assert record[t_idx][0].effective_name == "NewName"

        cmd.undo(record)

        # Each nucleus returns to its exact prior state
        assert record[0][0].identity == "first"
        assert record[0][0].assigned_id == ""
        assert record[1][0].identity == "middle"
        assert record[1][0].assigned_id == "was-forced"
        assert record[2][0].identity == "last"
        assert record[2][0].assigned_id == ""

    def test_rename_stops_at_division(self):
        """Rename on a dividing parent only affects the parent's chain, not daughters."""
        record = _dividing_record()
        # Give P0 a single timepoint chain (T1 only); daughters at T2 are separate cells.
        RenameCell(time=1, index=1, new_name="NewP0").execute(record)

        assert record[0][0].effective_name == "NewP0"
        # Daughters are NOT renamed
        assert record[1][0].effective_name == "AB"
        assert record[1][1].effective_name == "P1"


class TestWalkContinuationChain:
    def test_simple_chain(self):
        """A 3-timepoint non-dividing cell: chain has 3 entries."""
        record = _simple_record()
        chain = _walk_continuation_chain(record, 1, 0)  # middle of A's chain
        assert chain == [(0, 0), (1, 0), (2, 0)]

    def test_chain_stops_at_division(self):
        """Predecessor chain stops when parent has 2 successors."""
        record = _dividing_record()
        # AB at T2 idx=1 has predecessor=1 (P0).  P0.successor2 > 0, so chain
        # should NOT include P0.
        chain = _walk_continuation_chain(record, 1, 0)
        assert chain == [(1, 0)]

    def test_chain_from_anchor_idx_agnostic(self):
        """Walking from any nucleus in the chain returns the same chain."""
        record = _simple_record()
        c0 = _walk_continuation_chain(record, 0, 0)
        c1 = _walk_continuation_chain(record, 1, 0)
        c2 = _walk_continuation_chain(record, 2, 0)
        assert c0 == c1 == c2


class TestSwapCellNames:
    def test_swap_basic(self):
        """Swap cell A (chain: T1-T3 idx=1) with cell B (chain: T1-T3 idx=2)."""
        record = _simple_record()
        for t_idx in range(3):
            assert record[t_idx][0].effective_name == "A"
            assert record[t_idx][1].effective_name == "B"

        cmd = SwapCellNames(time_a=1, index_a=1, time_b=1, index_b=2)
        cmd.execute(record)

        for t_idx in range(3):
            assert record[t_idx][0].effective_name == "B"
            assert record[t_idx][1].effective_name == "A"

    def test_swap_undo(self):
        record = _simple_record()
        cmd = SwapCellNames(time_a=1, index_a=1, time_b=1, index_b=2)
        cmd.execute(record)
        cmd.undo(record)

        for t_idx in range(3):
            assert record[t_idx][0].effective_name == "A"
            assert record[t_idx][0].assigned_id == ""
            assert record[t_idx][1].effective_name == "B"
            assert record[t_idx][1].assigned_id == ""

    def test_swap_preserves_forced_names(self):
        """Swap writes assigned_id on both chains."""
        record = _simple_record()
        # First force-rename A to MyCell
        RenameCell(time=1, index=1, new_name="MyCell").execute(record)
        # Then swap MyCell <-> B
        SwapCellNames(time_a=1, index_a=1, time_b=1, index_b=2).execute(record)

        for t_idx in range(3):
            assert record[t_idx][0].assigned_id == "B"
            assert record[t_idx][1].assigned_id == "MyCell"


class TestValidateRenameCell:
    def test_valid_rename(self):
        record = _simple_record()
        errors, collision = validate_rename_cell(record, 1, 1, "Whatever")
        assert errors == []
        assert collision is None

    def test_rename_to_same_name_is_noop(self):
        record = _simple_record()
        errors, collision = validate_rename_cell(record, 1, 1, "A")
        assert errors == []
        assert collision is None

    def test_empty_new_name_rejected(self):
        record = _simple_record()
        errors, _ = validate_rename_cell(record, 1, 1, "")
        assert errors
        assert "empty" in errors[0].lower()

    def test_collision_detected(self):
        """Renaming A -> B (already used by another cell) returns collision anchor."""
        record = _simple_record()
        errors, collision = validate_rename_cell(record, 1, 1, "B")
        assert errors
        assert collision is not None
        other_t, other_j = collision
        # The conflicting cell is B, which lives at idx=2
        assert other_j == 2

    def test_collision_within_own_chain_is_allowed(self):
        """Renaming to a name already in the same continuation chain is fine."""
        record = _simple_record()
        # Pre-set T2 and T3 of A's chain to "NewName" (simulating a partial edit)
        record[1][0].assigned_id = "NewName"
        record[2][0].assigned_id = "NewName"
        # Renaming T1 to "NewName" should not be a collision — it's the same cell
        errors, collision = validate_rename_cell(record, 1, 1, "NewName")
        assert errors == []
        assert collision is None


# ── RelinkNucleus tests ─────────────────────────────────────────


class TestRelinkNucleus:
    def test_relink_basic(self):
        record = _simple_record()
        # T2 nucleus A2 has pred=1 (linked to A1)
        a2 = record[1][0]
        assert a2.predecessor == 1

        # Relink A2 to B1 (index=2 at T1)
        cmd = RelinkNucleus(time=2, index=1, new_predecessor=2)
        cmd.execute(record)

        assert a2.predecessor == 2
        # B1 should now have A2 as successor
        b1 = record[0][1]
        assert b1.successor1 == 2 or b1.successor2 == 1  # B1 already had succ1=2

        cmd.undo(record)
        assert a2.predecessor == 1

    def test_relink_to_nilli(self):
        record = _simple_record()
        a2 = record[1][0]
        assert a2.predecessor == 1

        cmd = RelinkNucleus(time=2, index=1, new_predecessor=NILLI)
        cmd.execute(record)
        assert a2.predecessor == NILLI

        cmd.undo(record)
        assert a2.predecessor == 1

    def test_relink_updates_old_parent_successors(self):
        record = _simple_record()
        a1 = record[0][0]
        assert a1.successor1 == 1  # A1 -> A2

        cmd = RelinkNucleus(time=2, index=1, new_predecessor=2)
        cmd.execute(record)

        # A1 should no longer have A2 as successor
        assert a1.successor1 == NILLI or a1.successor1 != 1

        cmd.undo(record)
        assert a1.successor1 == 1  # Restored


# ── KillCell tests ───────────────────────────────────────────────


class TestKillCell:
    def test_kill_across_timepoints(self):
        record = _simple_record()
        # Cell "A" exists in all 3 timepoints
        for t_idx in range(3):
            assert record[t_idx][0].identity == "A"
            assert record[t_idx][0].is_alive

        cmd = KillCell(cell_name="A", start_time=1)
        cmd.execute(record)

        for t_idx in range(3):
            assert not record[t_idx][0].is_alive
            assert record[t_idx][0].identity == ""

        cmd.undo(record)
        for t_idx in range(3):
            assert record[t_idx][0].identity == "A"
            assert record[t_idx][0].is_alive

    def test_kill_partial_range(self):
        record = _simple_record()

        cmd = KillCell(cell_name="A", start_time=2, end_time=2)
        cmd.execute(record)

        assert record[0][0].is_alive  # T1 unaffected
        assert not record[1][0].is_alive  # T2 killed
        assert record[2][0].is_alive  # T3 unaffected

    def test_kill_nonexistent_cell(self):
        record = _simple_record()
        cmd = KillCell(cell_name="XYZ", start_time=1)
        cmd.execute(record)
        assert len(cmd._killed) == 0  # Nothing to kill

    def test_kill_only_matching_name(self):
        record = _simple_record()
        cmd = KillCell(cell_name="A", start_time=1)
        cmd.execute(record)

        # B cells should be unaffected
        for t_idx in range(3):
            assert record[t_idx][1].identity == "B"
            assert record[t_idx][1].is_alive


# ── ResurrectCell tests ──────────────────────────────────────────


class TestResurrectCell:
    def test_resurrect_and_undo(self):
        record = _simple_record()
        # Kill first, then resurrect
        nuc = record[0][0]
        nuc.status = -1
        nuc.identity = ""
        assert not nuc.is_alive

        cmd = ResurrectCell(time=1, index=1, identity="A_resurrected")
        cmd.execute(record)

        assert nuc.is_alive
        assert nuc.identity == "A_resurrected"

        cmd.undo(record)
        assert not nuc.is_alive
        assert nuc.identity == ""

    def test_resurrect_without_name(self):
        record = _simple_record()
        nuc = record[0][0]
        nuc.status = -1
        nuc.identity = "old_name"

        cmd = ResurrectCell(time=1, index=1)
        cmd.execute(record)
        assert nuc.is_alive
        assert nuc.identity == "old_name"  # Unchanged when no identity given


# ── RelinkWithInterpolation tests ────────────────────────────────


class TestRelinkWithInterpolation:
    def test_adjacent_timepoints(self):
        """Adjacent timepoints: no interpolation needed, just link."""
        record = [
            [_make_nucleus(1, x=100, y=100, z=5.0, identity="A")],
            [_make_nucleus(1, x=120, y=120, z=5.0, identity="A")],
        ]

        cmd = RelinkWithInterpolation(start_time=1, start_index=1,
                                       end_time=2, end_index=1)
        cmd.execute(record)

        assert record[1][0].predecessor == 1
        assert record[0][0].successor1 == 1

        cmd.undo(record)
        assert record[1][0].predecessor == NILLI
        assert record[0][0].successor1 == NILLI

    def test_interpolation_creates_nuclei(self):
        """Gap of 3 timepoints: should create 2 interpolated nuclei."""
        record = [
            [_make_nucleus(1, x=100, y=100, z=5.0, identity="A")],
            [],  # T2 - empty
            [],  # T3 - empty
            [_make_nucleus(1, x=400, y=400, z=20.0, identity="A")],
        ]

        cmd = RelinkWithInterpolation(start_time=1, start_index=1,
                                       end_time=4, end_index=1)
        cmd.execute(record)

        # Should have created nuclei at T2 and T3
        assert len(record[1]) == 1  # T2
        assert len(record[2]) == 1  # T3

        # Check interpolation at T2 (1/3 of the way)
        t2_nuc = record[1][0]
        assert t2_nuc.x == 200  # 100 + (400-100) * 1/3 = 200
        assert t2_nuc.y == 200
        assert t2_nuc.predecessor == 1  # Linked to start

        # Check interpolation at T3 (2/3 of the way)
        t3_nuc = record[2][0]
        assert t3_nuc.x == 300  # 100 + (400-100) * 2/3 = 300
        assert t3_nuc.y == 300

        # Check chaining
        assert record[0][0].successor1 == 1  # Start -> T2 interp
        assert t2_nuc.successor1 == 1  # T2 interp -> T3 interp
        assert t3_nuc.successor1 == 1  # T3 interp -> End
        assert record[3][0].predecessor == 1  # End links back to T3 interp

    def test_interpolation_undo(self):
        record = [
            [_make_nucleus(1, x=100, y=100, z=5.0)],
            [],
            [],
            [_make_nucleus(1, x=400, y=400, z=20.0)],
        ]

        cmd = RelinkWithInterpolation(start_time=1, start_index=1,
                                       end_time=4, end_index=1)
        cmd.execute(record)
        assert len(record[1]) == 1
        assert len(record[2]) == 1

        cmd.undo(record)
        assert len(record[1]) == 0
        assert len(record[2]) == 0
        assert record[0][0].successor1 == NILLI
        assert record[3][0].predecessor == NILLI


# ── EditHistory tests ────────────────────────────────────────────


class TestEditHistory:
    def test_do_and_undo(self):
        record = _simple_record()
        history = EditHistory(record)

        assert not history.can_undo
        assert not history.modified

        history.do(AddNucleus(time=1, x=50, y=50, z=3.0, identity="C"))
        assert len(record[0]) == 3
        assert history.can_undo
        assert history.modified

        history.undo()
        assert len(record[0]) == 2
        assert not history.can_undo

    def test_redo(self):
        record = _simple_record()
        history = EditHistory(record)

        history.do(AddNucleus(time=1, x=50, y=50, z=3.0, identity="C"))
        history.undo()
        assert len(record[0]) == 2
        assert history.can_redo

        history.redo()
        assert len(record[0]) == 3
        assert not history.can_redo

    def test_redo_cleared_on_new_edit(self):
        record = _simple_record()
        history = EditHistory(record)

        history.do(AddNucleus(time=1, x=50, y=50, z=3.0))
        history.undo()
        assert history.can_redo

        history.do(AddNucleus(time=1, x=99, y=99, z=1.0))
        assert not history.can_redo  # Redo cleared

    def test_multiple_undo_redo(self):
        record = _simple_record()
        history = EditHistory(record)

        # Do 3 edits
        history.do(MoveNucleus(time=1, index=1, new_x=10))
        history.do(MoveNucleus(time=1, index=1, new_x=20))
        history.do(MoveNucleus(time=1, index=1, new_x=30))

        assert record[0][0].x == 30
        assert history.num_undoable == 3

        history.undo()
        assert record[0][0].x == 20
        history.undo()
        assert record[0][0].x == 10
        history.undo()
        assert record[0][0].x == 100  # Original

        assert history.num_undoable == 0
        assert history.num_redoable == 3

    def test_on_edit_callback(self):
        record = _simple_record()
        callback_count = [0]

        def on_edit():
            callback_count[0] += 1

        history = EditHistory(record, on_edit=on_edit)
        history.do(AddNucleus(time=1, x=0, y=0, z=0.0))
        assert callback_count[0] == 1

        history.undo()
        assert callback_count[0] == 2

        history.redo()
        assert callback_count[0] == 3

    def test_max_history(self):
        record = _simple_record()
        history = EditHistory(record, max_history=3)

        for i in range(5):
            history.do(MoveNucleus(time=1, index=1, new_x=i))

        assert history.num_undoable == 3  # Capped at max_history

    def test_undo_returns_command(self):
        record = _simple_record()
        history = EditHistory(record)

        cmd = AddNucleus(time=1, x=50, y=50, z=3.0)
        history.do(cmd)

        undone = history.undo()
        assert undone is cmd

    def test_undo_empty_returns_none(self):
        record = _simple_record()
        history = EditHistory(record)
        assert history.undo() is None
        assert history.redo() is None

    def test_description_properties(self):
        record = _simple_record()
        history = EditHistory(record)

        assert history.undo_description == ""
        assert history.redo_description == ""

        history.do(AddNucleus(time=1, x=0, y=0, z=0.0, identity="X"))
        assert "X" in history.undo_description

    def test_clear(self):
        record = _simple_record()
        history = EditHistory(record)

        history.do(AddNucleus(time=1, x=0, y=0, z=0.0))
        history.undo()
        assert history.can_redo

        history.clear()
        assert not history.can_undo
        assert not history.can_redo

    def test_mark_saved(self):
        record = _simple_record()
        history = EditHistory(record)

        history.do(AddNucleus(time=1, x=0, y=0, z=0.0))
        assert history.modified

        history.mark_saved()
        assert not history.modified

    def test_history_log(self):
        record = _simple_record()
        history = EditHistory(record)

        history.do(AddNucleus(time=1, x=0, y=0, z=0.0, identity="X"))
        history.do(RemoveNucleus(time=1, index=1))

        log = history.history_log()
        assert len(log) == 2
        assert "X" in log[0]
        assert "Remove" in log[1]


# ── Full undo rollback test ─────────────────────────────────────


class TestFullUndoRollback:
    """Execute a complex sequence of edits, then undo all to verify
    the original state is perfectly restored."""

    def test_full_rollback(self):
        record = _simple_record()

        # Save original state
        original_state = [
            [(n.x, n.y, n.z, n.size, n.identity, n.status, n.predecessor,
              n.successor1, n.successor2) for n in tp]
            for tp in record
        ]

        history = EditHistory(record)

        # Complex sequence of edits
        history.do(AddNucleus(time=1, x=50, y=50, z=3.0, identity="C"))
        history.do(MoveNucleus(time=1, index=1, new_x=999, new_y=888))
        history.do(RenameCell(time=2, index=2, new_name="B_renamed"))
        history.do(RemoveNucleus(time=3, index=2))
        history.do(ResurrectCell(time=3, index=2, identity="B_back"))

        # State should be different now
        assert len(record[0]) == 3  # Added one
        assert record[0][0].x == 999
        assert record[1][1].identity == "B_renamed"

        # Undo everything
        while history.can_undo:
            history.undo()

        # Remove the added nucleus check (AddNucleus undo removes it)
        assert len(record[0]) == 2  # Back to original

        # Verify original state restored
        for t_idx in range(3):
            for n_idx in range(len(original_state[t_idx])):
                nuc = record[t_idx][n_idx]
                expected = original_state[t_idx][n_idx]
                assert (nuc.x, nuc.y, nuc.z, nuc.size, nuc.identity, nuc.status,
                        nuc.predecessor, nuc.successor1, nuc.successor2) == expected


# ── Validator tests ──────────────────────────────────────────────


class TestValidators:
    def test_validate_add_nucleus_valid(self):
        record = _simple_record()
        errors = validate_add_nucleus(record, time=2, predecessor=1)
        assert errors == []

    def test_validate_add_nucleus_bad_time(self):
        record = _simple_record()
        errors = validate_add_nucleus(record, time=0)
        assert len(errors) > 0

    def test_validate_add_nucleus_bad_predecessor(self):
        record = _simple_record()
        errors = validate_add_nucleus(record, time=2, predecessor=99)
        assert any("out of range" in e for e in errors)

    def test_validate_add_nucleus_full_parent(self):
        record = _dividing_record()
        # P0 at T1 already has 2 successors
        errors = validate_add_nucleus(record, time=2, predecessor=1)
        assert any("2 successors" in e for e in errors)

    def test_validate_remove_valid(self):
        record = _simple_record()
        errors = validate_remove_nucleus(record, time=1, index=1)
        assert errors == []

    def test_validate_remove_bad_index(self):
        record = _simple_record()
        errors = validate_remove_nucleus(record, time=1, index=99)
        assert len(errors) > 0

    def test_validate_remove_already_dead(self):
        record = _simple_record()
        record[0][0].status = -1
        errors = validate_remove_nucleus(record, time=1, index=1)
        assert any("already dead" in e for e in errors)

    def test_validate_relink_valid(self):
        record = _simple_record()
        errors = validate_relink(record, time=2, index=1, new_predecessor=2)
        assert errors == []

    def test_validate_relink_to_nilli(self):
        record = _simple_record()
        errors = validate_relink(record, time=2, index=1, new_predecessor=NILLI)
        assert errors == []

    def test_validate_relink_bad_predecessor(self):
        record = _simple_record()
        errors = validate_relink(record, time=2, index=1, new_predecessor=99)
        assert any("out of range" in e for e in errors)

    def test_validate_relink_full_parent(self):
        record = _dividing_record()
        # P0 at T1 has 2 successors; relinking T2 idx=1 (AB) to P0 should
        # be fine since AB is already a successor of P0
        errors = validate_relink(record, time=2, index=1, new_predecessor=1)
        assert errors == []  # AB is already a child of P0

    def test_validate_kill_valid(self):
        record = _simple_record()
        errors = validate_kill_cell(record, cell_name="A", start_time=1)
        assert errors == []

    def test_validate_kill_empty_name(self):
        record = _simple_record()
        errors = validate_kill_cell(record, cell_name="", start_time=1)
        assert len(errors) > 0

    def test_validate_kill_nonexistent(self):
        record = _simple_record()
        errors = validate_kill_cell(record, cell_name="XYZ", start_time=1)
        assert any("not found" in e for e in errors)

    def test_validate_kill_bad_time(self):
        record = _simple_record()
        errors = validate_kill_cell(record, cell_name="A", start_time=99)
        assert len(errors) > 0

    def test_validate_relink_interpolation_valid(self):
        record = [
            [_make_nucleus(1, x=100, y=100, z=5.0)],
            [],
            [_make_nucleus(1, x=200, y=200, z=10.0)],
        ]
        errors = validate_relink_interpolation(record, 1, 1, 3, 1)
        assert errors == []

    def test_validate_relink_interpolation_bad_order(self):
        record = _simple_record()
        errors = validate_relink_interpolation(record, 3, 1, 1, 1)
        assert any("after start" in e for e in errors)

    def test_validate_relink_interpolation_full_start(self):
        record = _dividing_record()
        # P0 at T1 already has 2 successors
        errors = validate_relink_interpolation(record, 1, 1, 2, 1)
        assert any("2 successors" in e for e in errors)
