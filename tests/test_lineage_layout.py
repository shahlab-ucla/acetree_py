"""Tests for the lineage tree layout algorithm.

Tests the pure layout function (no Qt dependency) to verify:
- Correct y-positions from cell lifetimes
- Non-overlapping x-positions
- Internal nodes centered between children
- Expression value extraction
- Color mapping
- Tree bounds computation
"""

import pytest

from acetree_py.core.cell import Cell, CellFate
from acetree_py.core.nucleus import Nucleus
from acetree_py.gui.lineage_layout import (
    LayoutNode,
    LayoutParams,
    compute_layout,
    compute_tree_bounds,
    expression_to_color,
)


# ── Helpers ───────────────────────────────────────────────────────


def _make_cell(name, start, end, fate=CellFate.ALIVE, nuclei=None):
    """Create a Cell with optional nuclei."""
    cell = Cell(name=name, start_time=start, end_time=end, end_fate=fate)
    if nuclei:
        for t, nuc in nuclei:
            cell.add_nucleus(t, nuc)
    return cell


def _make_nuc(rweight=0):
    return Nucleus(index=1, x=0, y=0, z=0, size=10, status=1, rweight=rweight)


def _single_cell_tree():
    """A tree with just one cell: P0 from t=1 to t=10."""
    return _make_cell("P0", 1, 10)


def _binary_tree():
    """P0 -> AB + P1, each with further children.

    P0: t=1-5 (divides)
      AB: t=6-15 (divides)
        ABa: t=16-25 (leaf)
        ABp: t=16-25 (leaf)
      P1: t=6-20 (leaf)
    """
    p0 = _make_cell("P0", 1, 5, CellFate.DIVIDED)
    ab = _make_cell("AB", 6, 15, CellFate.DIVIDED)
    p1 = _make_cell("P1", 6, 20, CellFate.ALIVE)
    aba = _make_cell("ABa", 16, 25, CellFate.ALIVE)
    abp = _make_cell("ABp", 16, 25, CellFate.ALIVE)

    p0.add_child(ab)
    p0.add_child(p1)
    ab.add_child(aba)
    ab.add_child(abp)

    return p0


def _deep_tree():
    """A deeper tree with 4 leaves.

    R: t=1-5 (divides)
      A: t=6-10 (divides)
        C: t=11-20 (leaf)
        D: t=11-20 (leaf)
      B: t=6-10 (divides)
        E: t=11-15 (leaf)
        F: t=11-25 (leaf)
    """
    r = _make_cell("R", 1, 5, CellFate.DIVIDED)
    a = _make_cell("A", 6, 10, CellFate.DIVIDED)
    b = _make_cell("B", 6, 10, CellFate.DIVIDED)
    c = _make_cell("C", 11, 20)
    d = _make_cell("D", 11, 20)
    e = _make_cell("E", 11, 15)
    f = _make_cell("F", 11, 25)

    r.add_child(a)
    r.add_child(b)
    a.add_child(c)
    a.add_child(d)
    b.add_child(e)
    b.add_child(f)

    return r


def _tree_with_expression():
    """Simple tree with nuclei that have rweight values."""
    p0 = _make_cell("P0", 1, 3, CellFate.DIVIDED, nuclei=[
        (1, _make_nuc(rweight=100)),
        (2, _make_nuc(rweight=200)),
        (3, _make_nuc(rweight=300)),
    ])
    ab = _make_cell("AB", 4, 5, nuclei=[
        (4, _make_nuc(rweight=500)),
        (5, _make_nuc(rweight=1000)),
    ])
    p1 = _make_cell("P1", 4, 5, nuclei=[
        (4, _make_nuc(rweight=0)),
        (5, _make_nuc(rweight=50)),
    ])
    p0.add_child(ab)
    p0.add_child(p1)
    return p0


# ── Single cell tests ─────────────────────────────────────────────


class TestSingleCell:
    def test_single_cell_layout(self):
        root = _single_cell_tree()
        nodes = compute_layout(root)
        assert "P0" in nodes
        assert nodes["P0"].cell_name == "P0"
        assert nodes["P0"].is_leaf

    def test_single_cell_y_range(self):
        root = _single_cell_tree()
        params = LayoutParams(y_scale=3.0, top_margin=20.0)
        nodes = compute_layout(root, params)
        node = nodes["P0"]
        # y_start = 20 + (1 - 1) * 3 = 20
        # y_end = 20 + (10 - 1) * 3 = 47
        assert node.y_start == pytest.approx(20.0)
        assert node.y_end == pytest.approx(47.0)

    def test_single_cell_x_is_zero(self):
        root = _single_cell_tree()
        nodes = compute_layout(root)
        # Single cell should be at x=0 (or wherever the initial x is)
        assert nodes["P0"].x >= 0


# ── Binary tree tests ─────────────────────────────────────────────


class TestBinaryTree:
    def test_all_cells_present(self):
        root = _binary_tree()
        nodes = compute_layout(root)
        assert set(nodes.keys()) == {"P0", "AB", "P1", "ABa", "ABp"}

    def test_leaf_classification(self):
        root = _binary_tree()
        nodes = compute_layout(root)
        assert not nodes["P0"].is_leaf
        assert not nodes["AB"].is_leaf
        assert nodes["P1"].is_leaf
        assert nodes["ABa"].is_leaf
        assert nodes["ABp"].is_leaf

    def test_leaves_have_distinct_x(self):
        root = _binary_tree()
        nodes = compute_layout(root)
        leaf_xs = [nodes[n].x for n in ["ABa", "ABp", "P1"]]
        # All leaves should have distinct x positions
        assert len(set(leaf_xs)) == 3

    def test_leaves_no_overlap(self):
        root = _binary_tree()
        params = LayoutParams(x_scale=20.0)
        nodes = compute_layout(root, params)
        leaf_xs = sorted(nodes[n].x for n in ["ABa", "ABp", "P1"])
        # Each leaf should be at least x_scale apart
        for i in range(len(leaf_xs) - 1):
            assert leaf_xs[i + 1] - leaf_xs[i] >= params.x_scale - 0.01

    def test_parent_centered_between_children(self):
        root = _binary_tree()
        nodes = compute_layout(root)

        # AB should be centered between ABa and ABp
        ab_x = nodes["AB"].x
        aba_x = nodes["ABa"].x
        abp_x = nodes["ABp"].x
        assert ab_x == pytest.approx((aba_x + abp_x) / 2.0)

        # P0 should be centered between AB and P1
        p0_x = nodes["P0"].x
        p1_x = nodes["P1"].x
        assert p0_x == pytest.approx((ab_x + p1_x) / 2.0)

    def test_children_x_recorded(self):
        root = _binary_tree()
        nodes = compute_layout(root)
        p0 = nodes["P0"]
        assert len(p0.children_x) == 2
        child_names = [c[0] for c in p0.children_x]
        assert "AB" in child_names
        assert "P1" in child_names

    def test_y_positions_reflect_time(self):
        root = _binary_tree()
        params = LayoutParams(y_scale=2.0, top_margin=10.0)
        nodes = compute_layout(root, params)

        # P0 starts at t=1: y_start = 10 + (1-1)*2 = 10
        assert nodes["P0"].y_start == pytest.approx(10.0)
        # P0 ends at t=5: y_end = 10 + (5-1)*2 = 18
        assert nodes["P0"].y_end == pytest.approx(18.0)

        # AB starts at t=6: y_start = 10 + (6-1)*2 = 20
        assert nodes["AB"].y_start == pytest.approx(20.0)

    def test_depth_values(self):
        root = _binary_tree()
        nodes = compute_layout(root)
        assert nodes["P0"].depth == 0
        assert nodes["AB"].depth == 1
        assert nodes["P1"].depth == 1
        assert nodes["ABa"].depth == 2
        assert nodes["ABp"].depth == 2


# ── Deep tree tests ───────────────────────────────────────────────


class TestDeepTree:
    def test_all_cells_present(self):
        root = _deep_tree()
        nodes = compute_layout(root)
        assert set(nodes.keys()) == {"R", "A", "B", "C", "D", "E", "F"}

    def test_four_leaves_no_overlap(self):
        root = _deep_tree()
        params = LayoutParams(x_scale=15.0)
        nodes = compute_layout(root, params)
        leaf_xs = sorted(nodes[n].x for n in ["C", "D", "E", "F"])
        for i in range(len(leaf_xs) - 1):
            assert leaf_xs[i + 1] - leaf_xs[i] >= params.x_scale - 0.01

    def test_left_subtree_left_of_right(self):
        root = _deep_tree()
        nodes = compute_layout(root)
        # Left subtree (C, D) should be left of right subtree (E, F)
        left_max = max(nodes["C"].x, nodes["D"].x)
        right_min = min(nodes["E"].x, nodes["F"].x)
        assert left_max < right_min

    def test_root_centered(self):
        root = _deep_tree()
        nodes = compute_layout(root)
        a_x = nodes["A"].x
        b_x = nodes["B"].x
        assert nodes["R"].x == pytest.approx((a_x + b_x) / 2.0)


# ── Expression tests ──────────────────────────────────────────────


class TestExpression:
    def test_expression_values_extracted(self):
        root = _tree_with_expression()
        nodes = compute_layout(root)
        p0 = nodes["P0"]
        assert len(p0.expression_values) == 3
        assert p0.expression_values[0] == pytest.approx(100.0)
        assert p0.expression_values[2] == pytest.approx(300.0)

    def test_expression_values_for_children(self):
        root = _tree_with_expression()
        nodes = compute_layout(root)
        ab = nodes["AB"]
        assert len(ab.expression_values) == 2
        assert ab.expression_values[0] == pytest.approx(500.0)

    def test_custom_expression_fn(self):
        root = _tree_with_expression()

        def custom_fn(cell, time):
            return time * 10.0  # Simple function of time

        nodes = compute_layout(root, expression_fn=custom_fn)
        p0 = nodes["P0"]
        assert p0.expression_values[0] == pytest.approx(10.0)
        assert p0.expression_values[1] == pytest.approx(20.0)
        assert p0.expression_values[2] == pytest.approx(30.0)


# ── Expression color mapping tests ────────────────────────────────


class TestExpressionColor:
    def test_below_min_is_gray(self):
        r, g, b = expression_to_color(-100, vmin=0, vmax=100)
        assert r == pytest.approx(0.5)
        assert g == pytest.approx(0.5)
        assert b == pytest.approx(0.5)

    def test_min_is_dark_green(self):
        r, g, b = expression_to_color(0, vmin=0, vmax=100)
        assert r == pytest.approx(0.0)
        assert g > 0.2  # Dark green
        assert b == pytest.approx(0.0)

    def test_max_is_bright_red(self):
        r, g, b = expression_to_color(100, vmin=0, vmax=100)
        assert r == pytest.approx(1.0)
        assert g == pytest.approx(0.0)
        assert b == pytest.approx(0.0)

    def test_midpoint_transitions(self):
        r1, g1, b1 = expression_to_color(25, vmin=0, vmax=100)
        r2, g2, b2 = expression_to_color(75, vmin=0, vmax=100)
        # 25% should be greenish, 75% should be reddish
        assert g1 > r1  # Green dominant
        assert r2 > g2  # Red dominant

    def test_equal_min_max(self):
        r, g, b = expression_to_color(50, vmin=50, vmax=50)
        assert g > 0  # Default green


# ── Tree bounds tests ─────────────────────────────────────────────


class TestTreeBounds:
    def test_empty_nodes(self):
        bounds = compute_tree_bounds({})
        assert bounds == (0.0, 0.0, 0.0, 0.0)

    def test_single_cell_bounds(self):
        root = _single_cell_tree()
        nodes = compute_layout(root)
        x_min, y_min, x_max, y_max = compute_tree_bounds(nodes)
        assert x_min <= x_max
        assert y_min <= y_max

    def test_binary_tree_bounds(self):
        root = _binary_tree()
        nodes = compute_layout(root)
        x_min, y_min, x_max, y_max = compute_tree_bounds(nodes)

        # Should span all cell positions
        for node in nodes.values():
            assert node.x >= x_min
            assert node.x <= x_max
            assert node.y_start >= y_min
            assert node.y_end <= y_max


# ── Layout params tests ──────────────────────────────────────────


class TestLayoutParams:
    def test_custom_x_scale(self):
        root = _binary_tree()
        params_wide = LayoutParams(x_scale=50.0)
        params_narrow = LayoutParams(x_scale=10.0)

        nodes_wide = compute_layout(root, params_wide)
        nodes_narrow = compute_layout(root, params_narrow)

        # Wide layout should have larger x spread
        wide_spread = max(n.x for n in nodes_wide.values()) - min(n.x for n in nodes_wide.values())
        narrow_spread = max(n.x for n in nodes_narrow.values()) - min(n.x for n in nodes_narrow.values())
        assert wide_spread > narrow_spread

    def test_custom_y_scale(self):
        root = _single_cell_tree()
        params = LayoutParams(y_scale=5.0, top_margin=0.0)
        nodes = compute_layout(root, params)
        # P0: t=1-10, y_end = 0 + (10-1)*5 = 45
        assert nodes["P0"].y_end == pytest.approx(45.0)

    def test_late_time_clips(self):
        root = _binary_tree()
        # Clip display at t=10 (before ABa/ABp start at t=16)
        params = LayoutParams(late_time=10)
        nodes = compute_layout(root, params)

        # ABa and ABp should exist but be clipped
        # AB ends at t=10 (not 15), so it becomes a "leaf" at the cutoff
        assert "AB" in nodes

    def test_custom_root_time(self):
        root = _binary_tree()
        params = LayoutParams(root_time=1, y_scale=2.0, top_margin=0.0)
        nodes = compute_layout(root, params)
        # P0 starts at t=1: y_start = 0 + (1-1)*2 = 0
        assert nodes["P0"].y_start == pytest.approx(0.0)


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_child(self):
        """A cell with only one child (error case, but should not crash)."""
        parent = _make_cell("P", 1, 5, CellFate.DIVIDED)
        child = _make_cell("C", 6, 10)
        parent.add_child(child)

        nodes = compute_layout(parent)
        assert "P" in nodes
        assert "C" in nodes
        # Single child should be at same x as parent
        assert nodes["P"].x == pytest.approx(nodes["C"].x)

    def test_zero_lifetime_cell(self):
        """Cell with start_time == end_time."""
        cell = _make_cell("X", 5, 5)
        nodes = compute_layout(cell)
        assert "X" in nodes
        assert nodes["X"].y_start == nodes["X"].y_end

    def test_large_tree(self):
        """Stress test with many leaves to verify no overlaps."""
        # Build a balanced binary tree with 16 leaves
        def _build_balanced(depth, t_start, prefix=""):
            name = prefix or "R"
            cell = _make_cell(name, t_start, t_start + 4)
            if depth > 0:
                cell.end_fate = CellFate.DIVIDED
                left = _build_balanced(depth - 1, t_start + 5, name + "a")
                right = _build_balanced(depth - 1, t_start + 5, name + "b")
                cell.add_child(left)
                cell.add_child(right)
            return cell

        root = _build_balanced(4, 1)
        params = LayoutParams(x_scale=15.0)
        nodes = compute_layout(root, params)

        # Should have 16 leaves + 15 internal = 31 nodes
        assert len(nodes) == 31

        # No leaf overlap
        leaf_xs = sorted(n.x for n in nodes.values() if n.is_leaf)
        assert len(leaf_xs) == 16
        for i in range(len(leaf_xs) - 1):
            assert leaf_xs[i + 1] - leaf_xs[i] >= params.x_scale - 0.01
