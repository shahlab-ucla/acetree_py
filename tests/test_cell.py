"""Tests for acetree_py.core.cell."""

from __future__ import annotations

import pytest

from acetree_py.core.cell import Cell, CellFate
from acetree_py.core.nucleus import Nucleus


class TestCellBasics:
    """Test Cell creation and basic properties."""

    def test_default_cell(self):
        cell = Cell(name="P0", start_time=1, end_time=10)
        assert cell.name == "P0"
        assert cell.start_time == 1
        assert cell.end_time == 10
        assert cell.end_fate == CellFate.ALIVE
        assert cell.parent is None
        assert cell.children == []
        assert cell.is_leaf
        assert cell.is_root

    def test_lifetime(self):
        cell = Cell(start_time=5, end_time=15)
        assert cell.lifetime == 11

    def test_add_child(self):
        parent = Cell(name="P0", start_time=1, end_time=10, end_fate=CellFate.DIVIDED)
        child1 = Cell(name="AB", start_time=11, end_time=20)
        child2 = Cell(name="P1", start_time=11, end_time=25)

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.children) == 2
        assert child1.parent is parent
        assert child2.parent is parent
        assert not parent.is_leaf
        assert child1.is_leaf
        assert not child1.is_root


class TestCellNucleiAccess:
    """Test nucleus snapshot access."""

    def test_add_and_get_nucleus(self):
        cell = Cell(name="ABa", start_time=5, end_time=7)
        nuc5 = Nucleus(index=1, x=100, y=200, z=15.0)
        nuc6 = Nucleus(index=1, x=105, y=205, z=15.5)
        nuc7 = Nucleus(index=1, x=110, y=210, z=16.0)

        cell.add_nucleus(5, nuc5)
        cell.add_nucleus(6, nuc6)
        cell.add_nucleus(7, nuc7)

        assert cell.get_nucleus_at(5) is nuc5
        assert cell.get_nucleus_at(6) is nuc6
        assert cell.get_nucleus_at(7) is nuc7
        assert cell.get_nucleus_at(8) is None


class TestCellTreeTraversal:
    """Test tree traversal methods."""

    @pytest.fixture
    def sample_tree(self) -> Cell:
        """Build a small lineage tree:
            P0
            ├── AB
            │   ├── ABa
            │   └── ABp
            └── P1
        """
        p0 = Cell(name="P0", start_time=1, end_time=5, end_fate=CellFate.DIVIDED)
        ab = Cell(name="AB", start_time=6, end_time=10, end_fate=CellFate.DIVIDED)
        p1 = Cell(name="P1", start_time=6, end_time=15, end_fate=CellFate.ALIVE)
        aba = Cell(name="ABa", start_time=11, end_time=20)
        abp = Cell(name="ABp", start_time=11, end_time=20)

        p0.add_child(ab)
        p0.add_child(p1)
        ab.add_child(aba)
        ab.add_child(abp)

        return p0

    def test_iter_descendants(self, sample_tree: Cell):
        names = [c.name for c in sample_tree.iter_descendants()]
        assert "AB" in names
        assert "P1" in names
        assert "ABa" in names
        assert "ABp" in names
        assert "P0" not in names

    def test_iter_subtree_preorder(self, sample_tree: Cell):
        names = [c.name for c in sample_tree.iter_subtree_preorder()]
        assert names[0] == "P0"
        assert len(names) == 5

    def test_iter_ancestors(self, sample_tree: Cell):
        aba = list(sample_tree.iter_descendants())[2]  # ABa
        ancestors = [c.name for c in aba.iter_ancestors()]
        assert ancestors == ["AB", "P0"]

    def test_iter_leaves(self, sample_tree: Cell):
        leaves = [c.name for c in sample_tree.iter_leaves()]
        assert set(leaves) == {"ABa", "ABp", "P1"}

    def test_depth(self, sample_tree: Cell):
        assert sample_tree.depth() == 0
        ab = sample_tree.children[0]
        assert ab.depth() == 1
        aba = ab.children[0]
        assert aba.depth() == 2
