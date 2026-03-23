"""Tests for acetree_py.core.lineage — lineage tree building."""

from __future__ import annotations

import pytest

from acetree_py.core.cell import Cell, CellFate
from acetree_py.core.lineage import (
    LineageTree,
    _make_hash_key,
    build_lineage_tree,
)
from acetree_py.core.nucleus import NILLI, Nucleus


def _nuc(
    index: int,
    x: int = 300,
    y: int = 250,
    z: float = 15.0,
    identity: str = "",
    status: int = 1,
    pred: int = NILLI,
    succ1: int = NILLI,
    succ2: int = NILLI,
) -> Nucleus:
    """Shorthand to create a Nucleus for testing."""
    return Nucleus(
        index=index, x=x, y=y, z=z, size=20,
        identity=identity, status=status,
        predecessor=pred, successor1=succ1, successor2=succ2,
        weight=5000,
    )


def _simple_lineage() -> list[list[Nucleus]]:
    """P0 -> AB + P1, then AB -> ABa + ABp.

    T0: P0 (dividing)
    T1: AB, P1
    T2: ABa, ABp, P1 (continuing)
    """
    return [
        # T0: P0
        [_nuc(1, identity="P0", succ1=1, succ2=2)],
        # T1: AB, P1
        [
            _nuc(1, x=280, identity="AB", pred=1, succ1=1, succ2=2),
            _nuc(2, x=320, identity="P1", pred=1, succ1=3),
        ],
        # T2: ABa, ABp, P1
        [
            _nuc(1, x=260, identity="ABa", pred=1),
            _nuc(2, x=300, identity="ABp", pred=1),
            _nuc(3, x=340, identity="P1", pred=2),
        ],
    ]


def _linear_lineage() -> list[list[Nucleus]]:
    """A single cell tracked across 3 timepoints with no divisions.

    T0: CellA
    T1: CellA (continues)
    T2: CellA (continues)
    """
    return [
        [_nuc(1, identity="CellA", succ1=1)],
        [_nuc(1, identity="CellA", pred=1, succ1=1)],
        [_nuc(1, identity="CellA", pred=1)],
    ]


class TestMakeHashKey:
    """Test hash key generation."""

    def test_basic_hash(self):
        assert _make_hash_key(1, 1) == "100001"
        assert _make_hash_key(1, 5) == "100005"
        assert _make_hash_key(10, 3) == "1000003"

    def test_unique_keys(self):
        keys = set()
        for t in range(1, 100):
            for n in range(1, 50):
                key = _make_hash_key(t, n)
                assert key not in keys
                keys.add(key)


class TestBuildLineageTree:
    """Test lineage tree construction."""

    def test_simple_lineage(self):
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        assert tree is not None
        assert tree.root is not None
        # P0 should be the root (from dummy ancestors)
        assert tree.root.name == "P0"

    def test_cell_counts(self):
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        # T0: 1 alive, T1: 2 alive, T2: 3 alive
        assert tree.cell_counts[0] == 1
        assert tree.cell_counts[1] == 2
        assert tree.cell_counts[2] == 3

    def test_cells_by_name_populated(self):
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        # Should have dummy ancestors + real cells
        assert "P0" in tree.cells_by_name
        assert "AB" in tree.cells_by_name
        assert "P1" in tree.cells_by_name

    def test_linear_cell_not_dividing(self):
        nuclei = _linear_lineage()
        tree = build_lineage_tree(nuclei)

        cell = tree.get_cell("CellA")
        assert cell is not None
        assert cell.start_time == 1
        assert cell.end_time == 3
        assert cell.end_fate == CellFate.ALIVE
        assert len(cell.children) == 0

    def test_linear_cell_has_nuclei(self):
        nuclei = _linear_lineage()
        tree = build_lineage_tree(nuclei)

        cell = tree.get_cell("CellA")
        assert cell is not None
        assert len(cell.nuclei) == 3
        assert cell.get_nucleus_at(1) is not None
        assert cell.get_nucleus_at(2) is not None
        assert cell.get_nucleus_at(3) is not None

    def test_division_creates_daughters(self):
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        ab = tree.get_cell("AB")
        assert ab is not None
        # AB should have been marked as dividing
        # (its children ABa, ABp should be in the tree)
        aba = tree.get_cell("ABa")
        abp = tree.get_cell("ABp")
        assert aba is not None
        assert abp is not None

    def test_no_dummy_ancestors(self):
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei, create_dummy_ancestors=False)

        # Should still build a tree
        assert tree is not None
        # P0 should exist as a real cell, not a dummy
        p0 = tree.get_cell("P0")
        assert p0 is not None
        assert p0.start_time == 1  # Real data, not dummy

    def test_dead_nuclei_skipped(self):
        """Dead nuclei (status < 1) should be skipped."""
        nuclei = [
            [
                _nuc(1, identity="Live", status=1, succ1=1),
                _nuc(2, identity="Dead", status=-1),
            ],
            [_nuc(1, identity="Live", pred=1)],
        ]
        tree = build_lineage_tree(nuclei)

        live_cell = tree.get_cell("Live")
        assert live_cell is not None
        # Dead cell should not appear as a real cell in the tree
        dead_cell = tree.get_cell("Dead")
        # It might exist as a dummy or not at all
        if dead_cell is not None:
            assert len(dead_cell.nuclei) == 0  # No real nucleus data

    def test_empty_nuclei_record(self):
        tree = build_lineage_tree([])
        assert tree is not None
        assert tree.num_cells > 0  # dummy ancestors

    def test_get_cell_icase(self):
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        cell = tree.get_cell_icase("ab")
        assert cell is not None
        assert cell.name == "AB"


class TestDummyAncestorMerging:
    """Test that real daughters merge with dummy ancestors instead of being orphaned."""

    def test_real_daughters_linked_to_parent(self):
        """When AB divides, real ABa and ABp should be children of AB, not orphaned."""
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        ab = tree.get_cell("AB")
        assert ab is not None

        # AB should have exactly 2 children
        assert len(ab.children) == 2

        # The children should have real data (start_time > 0)
        child_names = {c.name for c in ab.children}
        assert "ABa" in child_names
        assert "ABp" in child_names

        for child in ab.children:
            assert child.start_time > 0, f"{child.name} has no real data (start_time={child.start_time})"
            assert len(child.nuclei) > 0, f"{child.name} has no nuclei"

    def test_ab_division_marked(self):
        """AB should be marked as DIVIDED with correct end time."""
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        ab = tree.get_cell("AB")
        assert ab is not None
        assert ab.end_fate == CellFate.DIVIDED

    def test_aba_parent_is_ab(self):
        """ABa's parent should be AB, not None."""
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        aba = tree.get_cell("ABa")
        assert aba is not None
        assert aba.parent is not None
        assert aba.parent.name == "AB"

    def test_p0_to_ab_to_aba_traversal(self):
        """Full tree traversal: P0 → AB → ABa should work."""
        nuclei = _simple_lineage()
        tree = build_lineage_tree(nuclei)

        p0 = tree.root
        assert p0 is not None
        assert p0.name == "P0"

        # P0 should have AB and P1 as children
        child_names = {c.name for c in p0.children}
        assert "AB" in child_names

        # Find AB
        ab = next(c for c in p0.children if c.name == "AB")

        # AB should have ABa and ABp
        ab_child_names = {c.name for c in ab.children}
        assert "ABa" in ab_child_names
        assert "ABp" in ab_child_names

    def test_extended_lineage_ems_daughters(self):
        """EMS → E + MS linkage should work through dummy merging."""
        # Build a lineage that goes all the way to EMS dividing
        nuclei = [
            # T0: P0
            [_nuc(1, identity="P0", succ1=1, succ2=2)],
            # T1: AB, P1
            [
                _nuc(1, x=280, identity="AB", pred=1, succ1=1, succ2=2),
                _nuc(2, x=320, identity="P1", pred=1, succ1=2),
            ],
            # T2: ABa, ABp, P1 continuing
            [
                _nuc(1, x=260, identity="ABa", pred=1, succ1=1),
                _nuc(2, x=300, identity="ABp", pred=1, succ1=2),
                _nuc(3, x=340, identity="P1", pred=2, succ1=1, succ2=2),
            ],
            # T3: ABa, ABp, EMS, P2
            [
                _nuc(1, x=255, identity="ABa", pred=1),
                _nuc(2, x=305, identity="ABp", pred=2),
                _nuc(3, x=330, identity="EMS", pred=3, succ1=3, succ2=4),
                _nuc(4, x=370, identity="P2", pred=3),
            ],
            # T4: ABa, ABp, E, MS, P2
            [
                _nuc(1, x=250, identity="ABa", pred=1),
                _nuc(2, x=310, identity="ABp", pred=2),
                _nuc(3, x=325, identity="E", pred=3),
                _nuc(4, x=340, identity="MS", pred=3),
                _nuc(5, x=375, identity="P2", pred=4),
            ],
        ]
        tree = build_lineage_tree(nuclei)

        ems = tree.get_cell("EMS")
        assert ems is not None
        ems_child_names = {c.name for c in ems.children}
        assert "E" in ems_child_names, f"E not in EMS children: {ems_child_names}"
        assert "MS" in ems_child_names, f"MS not in EMS children: {ems_child_names}"

        # E and MS should have real data
        e = tree.get_cell("E")
        ms = tree.get_cell("MS")
        assert e is not None and e.start_time > 0
        assert ms is not None and ms.start_time > 0

        # E's parent should be EMS
        assert e.parent is not None
        assert e.parent.name == "EMS"


class TestLineageTreeLookup:
    """Test LineageTree lookup methods."""

    def test_get_cell_found(self):
        tree = LineageTree()
        cell = Cell(name="TestCell", start_time=1, end_time=5)
        tree.cells_by_name["TestCell"] = cell

        assert tree.get_cell("TestCell") is cell

    def test_get_cell_not_found(self):
        tree = LineageTree()
        assert tree.get_cell("NonExistent") is None

    def test_all_cells(self):
        tree = LineageTree()
        tree.cells_by_name["A"] = Cell(name="A")
        tree.cells_by_name["B"] = Cell(name="B")

        cells = tree.all_cells()
        assert len(cells) == 2

    def test_num_cells(self):
        tree = LineageTree()
        assert tree.num_cells == 0
        tree.cells_by_name["A"] = Cell(name="A")
        assert tree.num_cells == 1
