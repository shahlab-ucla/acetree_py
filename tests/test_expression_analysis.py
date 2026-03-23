"""Tests for analysis/expression.py — expression analysis functions.

Tests cover:
- ExpressionTimeSeries properties (mean, max, min, onset, total)
- cell_expression_time_series() extraction
- Custom value functions
- SubtreeStats computation
- SisterComparison properties (ratio, difference, fold_change)
- compare_sisters() function
- all_sister_comparisons() with filtering
- expression_onset_map()
- Edge cases: zero expression, single timepoint, no nuclei
"""

from __future__ import annotations

import pytest

from acetree_py.core.cell import Cell, CellFate
from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.analysis.expression import (
    ExpressionTimeSeries,
    SisterComparison,
    SubtreeStats,
    all_sister_comparisons,
    cell_expression_time_series,
    compare_sisters,
    expression_onset_map,
    subtree_expression_stats,
)


# ── Helpers ──────────────────────────────────────────────────────


def _nuc(rweight=100, weight=5000, rwraw=120, z=10.0, index=1):
    return Nucleus(
        index=index, x=100, y=200, z=z, size=20,
        identity="test", status=1, predecessor=NILLI,
        weight=weight, rweight=rweight, rwraw=rwraw,
    )


def _make_cell(name="ABa", start=1, end=3, rweights=None):
    """Create a cell with nuclei at each timepoint with given rweights."""
    if rweights is None:
        rweights = [100, 150, 200]
    cell = Cell(name=name, start_time=start, end_time=end)
    for i, rw in enumerate(rweights):
        t = start + i
        nuc = _nuc(rweight=rw, index=1)
        cell.add_nucleus(t, nuc)
    return cell


def _make_tree():
    """
    Build a small tree:
        P0
        ├── AB  (rweights: [10, 20, 30])
        │   ├── ABa  (rweights: [100, 200])
        │   └── ABp  (rweights: [50, 60])
        └── P1  (rweights: [5, 10, 15])
    """
    p0 = _make_cell("P0", 1, 1, [0])
    ab = _make_cell("AB", 2, 4, [10, 20, 30])
    p1 = _make_cell("P1", 2, 4, [5, 10, 15])
    aba = _make_cell("ABa", 5, 6, [100, 200])
    abp = _make_cell("ABp", 5, 6, [50, 60])

    p0.add_child(ab)
    p0.add_child(p1)
    p0.end_fate = CellFate.DIVIDED
    ab.add_child(aba)
    ab.add_child(abp)
    ab.end_fate = CellFate.DIVIDED

    return p0


# ── ExpressionTimeSeries tests ──────────────────────────────────


class TestExpressionTimeSeries:

    def test_mean(self):
        ts = ExpressionTimeSeries("A", 1, 3, [1, 2, 3], [100.0, 200.0, 300.0])
        assert ts.mean == 200.0

    def test_max(self):
        ts = ExpressionTimeSeries("A", 1, 3, [1, 2, 3], [100.0, 300.0, 200.0])
        assert ts.max_value == 300.0

    def test_min(self):
        ts = ExpressionTimeSeries("A", 1, 3, [1, 2, 3], [100.0, 50.0, 200.0])
        assert ts.min_value == 50.0

    def test_onset_time(self):
        ts = ExpressionTimeSeries("A", 1, 3, [1, 2, 3], [0.0, 0.0, 100.0])
        assert ts.onset_time == 3

    def test_onset_time_never(self):
        ts = ExpressionTimeSeries("A", 1, 3, [1, 2, 3], [0.0, 0.0, 0.0])
        assert ts.onset_time is None

    def test_onset_time_first(self):
        ts = ExpressionTimeSeries("A", 1, 3, [1, 2, 3], [50.0, 100.0, 200.0])
        assert ts.onset_time == 1

    def test_total(self):
        ts = ExpressionTimeSeries("A", 1, 3, [1, 2, 3], [100.0, 200.0, 300.0])
        assert ts.total == 600.0

    def test_empty(self):
        ts = ExpressionTimeSeries("A", 1, 1, [], [])
        assert ts.mean == 0.0
        assert ts.max_value == 0.0
        assert ts.onset_time is None
        assert ts.total == 0.0
        assert ts.num_timepoints == 0


# ── cell_expression_time_series tests ───────────────────────────


class TestCellExpressionTimeSeries:

    def test_basic_extraction(self):
        cell = _make_cell("ABa", 1, 3, [100, 150, 200])
        ts = cell_expression_time_series(cell)

        assert ts.cell_name == "ABa"
        assert ts.start_time == 1
        assert ts.end_time == 3
        assert ts.timepoints == [1, 2, 3]
        assert ts.values == [100.0, 150.0, 200.0]
        assert ts.mean == 150.0

    def test_custom_value_fn(self):
        cell = _make_cell("ABa", 1, 3, [100, 150, 200])
        # Override nuclei to have specific weight values
        for t, nuc in cell.nuclei:
            nuc.weight = t * 1000

        ts = cell_expression_time_series(cell, value_fn=lambda n: float(n.weight))
        assert ts.values == [1000.0, 2000.0, 3000.0]

    def test_single_timepoint(self):
        cell = _make_cell("X", 5, 5, [42])
        ts = cell_expression_time_series(cell)
        assert ts.values == [42.0]
        assert ts.mean == 42.0

    def test_zero_expression(self):
        cell = _make_cell("X", 1, 3, [0, 0, 0])
        ts = cell_expression_time_series(cell)
        assert ts.mean == 0.0
        assert ts.onset_time is None


# ── SubtreeStats tests ──────────────────────────────────────────


class TestSubtreeExpressionStats:

    def test_single_cell(self):
        cell = _make_cell("ABa", 1, 3, [100, 200, 300])
        stats = subtree_expression_stats(cell)

        assert stats.root_name == "ABa"
        assert stats.num_cells == 1
        assert stats.num_expressing == 1
        assert stats.mean_expression == 200.0
        assert stats.max_expression == 300.0
        assert stats.earliest_onset == 1

    def test_subtree(self):
        root = _make_tree()
        stats = subtree_expression_stats(root)

        assert stats.root_name == "P0"
        assert stats.num_cells == 5  # P0, AB, P1, ABa, ABp
        assert stats.num_expressing == 4  # P0 has rweight=0
        assert stats.max_expression == 200.0  # ABa has 200

    def test_no_expression(self):
        cell = _make_cell("X", 1, 3, [0, 0, 0])
        stats = subtree_expression_stats(cell)

        assert stats.num_expressing == 0
        assert stats.earliest_onset is None
        assert stats.mean_expression == 0.0


# ── SisterComparison tests ──────────────────────────────────────


class TestSisterComparison:

    def test_ratio(self):
        comp = SisterComparison("P", "A", "B", 100.0, 50.0)
        assert comp.ratio == 2.0

    def test_ratio_zero_denominator(self):
        comp = SisterComparison("P", "A", "B", 100.0, 0.0)
        assert comp.ratio == float("inf")

    def test_ratio_both_zero(self):
        comp = SisterComparison("P", "A", "B", 0.0, 0.0)
        assert comp.ratio == 1.0

    def test_difference(self):
        comp = SisterComparison("P", "A", "B", 100.0, 30.0)
        assert comp.difference == 70.0

    def test_fold_change(self):
        comp = SisterComparison("P", "A", "B", 100.0, 50.0)
        assert comp.fold_change == 2.0

    def test_fold_change_symmetric(self):
        comp1 = SisterComparison("P", "A", "B", 100.0, 50.0)
        comp2 = SisterComparison("P", "A", "B", 50.0, 100.0)
        assert comp1.fold_change == comp2.fold_change


# ── compare_sisters tests ───────────────────────────────────────


class TestCompareSisters:

    def test_basic(self):
        root = _make_tree()
        ab = root.children[0]  # AB has children ABa, ABp
        comp = compare_sisters(ab)

        assert comp is not None
        assert comp.parent_name == "AB"
        assert comp.sister1_name == "ABa"
        assert comp.sister2_name == "ABp"
        assert comp.sister1_mean == 150.0  # (100+200)/2
        assert comp.sister2_mean == 55.0  # (50+60)/2

    def test_no_children(self):
        leaf = _make_cell("leaf", 1, 3, [100, 200, 300])
        assert compare_sisters(leaf) is None

    def test_one_child(self):
        parent = _make_cell("P", 1, 1, [0])
        child = _make_cell("C", 2, 3, [10, 20])
        parent.add_child(child)
        assert compare_sisters(parent) is None


# ── all_sister_comparisons tests ────────────────────────────────


class TestAllSisterComparisons:

    def test_finds_all_divisions(self):
        root = _make_tree()
        comps = all_sister_comparisons(root)
        # P0 divides (AB vs P1) and AB divides (ABa vs ABp) = 2 comparisons
        assert len(comps) == 2

    def test_sorted_by_fold_change(self):
        root = _make_tree()
        comps = all_sister_comparisons(root)
        # Fold changes should be descending
        for i in range(len(comps) - 1):
            assert comps[i].fold_change >= comps[i + 1].fold_change

    def test_min_fold_change_filter(self):
        root = _make_tree()
        comps = all_sister_comparisons(root, min_fold_change=2.5)
        # Only keep pairs with fold_change >= 2.5
        for comp in comps:
            assert comp.fold_change >= 2.5


# ── expression_onset_map tests ──────────────────────────────────


class TestExpressionOnsetMap:

    def test_basic(self):
        root = _make_tree()
        onset_map = expression_onset_map(root)

        assert "P0" in onset_map
        assert "ABa" in onset_map
        assert onset_map["P0"] is None  # rweight=0
        assert onset_map["AB"] == 2  # first timepoint with rweight>0
        assert onset_map["ABa"] == 5

    def test_all_keys_present(self):
        root = _make_tree()
        onset_map = expression_onset_map(root)
        assert len(onset_map) == 5  # P0, AB, P1, ABa, ABp
