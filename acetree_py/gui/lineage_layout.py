"""Pure layout algorithm for the Sulston lineage tree.

Takes a Cell tree root + display parameters and returns a dictionary
of LayoutNode objects with (x, y_start, y_end) positions for each cell.

This module has ZERO Qt/GUI imports — it's a pure function that can be
tested independently.

Layout strategy (matching Java Cell.draw()):
  - Y-axis (vertical) = time. Branch length is proportional to cell lifetime.
  - X-axis (horizontal) = leaf ordering. Recursive depth-first traversal
    assigns x-positions left-to-right, with a global watermark (x_max)
    to prevent overlap.
  - Internal nodes are centered between their children.
  - Leaves are spaced by x_scale, with wider subtrees getting proportionally
    more horizontal space.

Ported from: org.rhwlab.tree.Cell.draw() layout logic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from ..core.cell import Cell, CellFate

logger = logging.getLogger(__name__)


@dataclass
class LayoutNode:
    """Layout information for a single cell in the tree.

    Attributes:
        cell_name: Name of the cell.
        x: Horizontal position (in layout units).
        y_start: Vertical start position (top of branch, = cell birth time).
        y_end: Vertical end position (bottom of branch, = cell end time).
        is_leaf: True if this cell has no children.
        depth: Number of divisions from root.
        children_x: List of (child_name, child_x) for drawing horizontal connectors.
        expression_values: Per-timepoint expression values for color mapping.
    """

    cell_name: str = ""
    x: float = 0.0
    y_start: float = 0.0
    y_end: float = 0.0
    is_leaf: bool = True
    depth: int = 0
    children_x: list[tuple[str, float]] = field(default_factory=list)
    expression_values: list[float] = field(default_factory=list)


@dataclass
class LayoutParams:
    """Parameters controlling the tree layout.

    Attributes:
        x_scale: Horizontal spacing between adjacent leaves (in pixels/units).
        y_scale: Vertical pixels per timepoint.
        top_margin: Top margin in pixels.
        bottom_margin: Bottom margin in pixels.
        late_time: Latest timepoint to display (cells beyond this are clipped).
        root_time: Earliest timepoint (typically the root cell's start_time).
    """

    x_scale: float = 20.0
    y_scale: float = 3.0
    top_margin: float = 20.0
    bottom_margin: float = 40.0
    late_time: int | None = None  # None = use tree's actual range
    root_time: int | None = None  # None = use root cell's start_time


def compute_layout(
    root: Cell,
    params: LayoutParams | None = None,
    expression_fn: Callable[[Cell, int], float] | None = None,
) -> dict[str, LayoutNode]:
    """Compute (x, y) positions for all cells in the lineage tree.

    This is the main entry point. It performs a recursive depth-first
    traversal to assign positions, matching the Java Cell.draw() algorithm.

    Args:
        root: Root cell of the lineage tree.
        params: Layout parameters (uses defaults if None).
        expression_fn: Optional function(cell, timepoint) -> float that
            returns an expression value for coloring. If None, uses
            Nucleus.rweight.

    Returns:
        Dictionary mapping cell name -> LayoutNode with positions.
    """
    if params is None:
        params = LayoutParams()

    # Determine time range
    root_time = params.root_time if params.root_time is not None else root.start_time
    late_time = params.late_time
    if late_time is None:
        late_time = _find_latest_time(root)

    if late_time <= root_time:
        late_time = root_time + 1

    # Compute y_scale from time range if not explicitly set
    y_scale = params.y_scale

    # Build layout via recursive traversal
    state = _LayoutState(
        x_max=-params.x_scale,  # Start below zero so first leaf lands at 0
        x_scale=params.x_scale,
        y_scale=y_scale,
        top_margin=params.top_margin,
        root_time=root_time,
        late_time=late_time,
        expression_fn=expression_fn,
        nodes={},
    )

    _layout_cell(root, 0.0, state)

    return state.nodes


@dataclass
class _LayoutState:
    """Mutable state passed through the recursive layout traversal."""

    x_max: float  # Global watermark: rightmost x assigned so far
    x_scale: float
    y_scale: float
    top_margin: float
    root_time: int
    late_time: int
    expression_fn: Callable[[Cell, int], float] | None
    nodes: dict[str, LayoutNode]


def _layout_cell(cell: Cell, x: float, state: _LayoutState) -> float:
    """Recursively assign layout position to a cell and its descendants.

    Args:
        cell: The cell to lay out.
        x: Proposed x-position for this cell.
        state: Mutable layout state.

    Returns:
        The actual x-position assigned to this cell.
    """
    # Compute y positions from time
    y_start = state.top_margin + (cell.start_time - state.root_time) * state.y_scale

    # Clamp end time to display range
    end_time = min(cell.end_time, state.late_time)
    y_end = state.top_margin + (end_time - state.root_time) * state.y_scale

    # Get expression values
    expr_values = _get_expression_values(cell, state)

    if not cell.children or end_time >= state.late_time:
        # Leaf node (or clipped at late_time)
        # Avoid overlap: ensure minimum x_scale gap from the last placed leaf
        min_x = state.x_max + state.x_scale
        if x < min_x:
            x = min_x
        state.x_max = x

        node = LayoutNode(
            cell_name=cell.name,
            x=x,
            y_start=y_start,
            y_end=y_end,
            is_leaf=True,
            depth=cell.depth(),
            expression_values=expr_values,
        )
        state.nodes[cell.name] = node
        return x

    # Internal node: lay out children recursively
    children_x = []

    if len(cell.children) == 1:
        # Single child — place directly below
        child_x = _layout_cell(cell.children[0], x, state)
        children_x.append((cell.children[0].name, child_x))
        actual_x = child_x

    elif len(cell.children) >= 2:
        # Two children — determine correct left/right ordering
        left_child, right_child = _order_daughters(cell)

        # Lay out left subtree
        left_x = _layout_cell(left_child, x, state)
        children_x.append((left_child.name, left_x))

        # Compute spacing: proportional to left subtree width
        left_leaf_count = _count_leaves(left_child)
        spacing = max(1, left_leaf_count // 2) * state.x_scale

        # Lay out right subtree with offset
        right_start_x = left_x + spacing
        right_x = _layout_cell(right_child, right_start_x, state)
        children_x.append((right_child.name, right_x))

        # Parent x is midpoint of children
        actual_x = (left_x + right_x) / 2.0

    else:
        actual_x = x

    node = LayoutNode(
        cell_name=cell.name,
        x=actual_x,
        y_start=y_start,
        y_end=y_end,
        is_leaf=False,
        depth=cell.depth(),
        children_x=children_x,
        expression_values=expr_values,
    )
    state.nodes[cell.name] = node
    return actual_x


def _order_daughters(parent: Cell) -> tuple[Cell, Cell]:
    """Determine which daughter goes left and which goes right.

    Matches Java AncesTree.checkDaughters() logic:

    1. Special-case parents (P0, P1, P2, P3, P4, EMS):
       - P0  -> AB (left), P1 (right)
       - P1  -> EMS (left), P2 (right)
       - P2  -> C (left), P3 (right)
       - P3  -> D (left), P4 (right)
       - EMS -> MS (left), E (right)
       - P4  -> Z2 (left), Z3 (right)

    2. For normal Sulston-named cells, compare the last character:
       - 'a', 'd', 'l' daughters go LEFT  (anterior, dorsal, left)
       - 'p', 'v', 'r' daughters go RIGHT (posterior, ventral, right)
       This works because ASCII: a(97) < d(100) < l(108) < p(112) < r(114) < v(118)
       So the daughter with the smaller last character goes left.

    Args:
        parent: The parent cell with exactly 2 children.

    Returns:
        (left_child, right_child) tuple.
    """
    d1, d2 = parent.children[0], parent.children[1]
    n1, n2 = d1.name, d2.name
    pn = parent.name

    # Special-case parents with uppercase daughter names
    SPECIAL_CASES = {
        "P0":  "AB",    # AB goes left
        "P1":  "EMS",   # EMS goes left
        "P2":  "C",     # C goes left
        "P3":  "D",     # D goes left
        "EMS": "MS",    # MS goes left
        "P4":  "Z2",    # Z2 goes left
    }

    if pn in SPECIAL_CASES:
        left_name = SPECIAL_CASES[pn]
        if n1 == left_name:
            return (d1, d2)
        elif n2 == left_name:
            return (d2, d1)
        # If neither matches (e.g. renamed), fall through to general logic

    # General case: compare last character of daughter names
    # Daughters ending in a/d/l go left; p/v/r go right
    if n1 and n2:
        c1 = n1[-1]
        c2 = n2[-1]

        # Only apply if both end in lowercase Sulston suffixes
        if c1.islower() and c2.islower():
            if c1 <= c2:
                return (d1, d2)  # d1 is left (smaller char = a/d/l)
            else:
                return (d2, d1)  # d2 is left

    # Fallback: keep original order
    return (d1, d2)


def _count_leaves(cell: Cell) -> int:
    """Count the number of leaf cells in a subtree."""
    if cell.is_leaf:
        return 1
    return sum(_count_leaves(c) for c in cell.children)


def _find_latest_time(root: Cell) -> int:
    """Find the latest end_time in the tree."""
    latest = root.end_time
    for cell in root.iter_descendants():
        if cell.end_time > latest:
            latest = cell.end_time
    return latest


def _get_expression_values(cell: Cell, state: _LayoutState) -> list[float]:
    """Get per-timepoint expression values for a cell."""
    values = []
    for t in range(cell.start_time, min(cell.end_time, state.late_time) + 1):
        if state.expression_fn is not None:
            values.append(state.expression_fn(cell, t))
        else:
            nuc = cell.get_nucleus_at(t)
            if nuc is not None:
                values.append(float(nuc.rweight))
            else:
                values.append(0.0)
    return values


# ── Expression color mapping ─────────────────────────────────────


def expression_to_color(
    value: float,
    vmin: float = 0.0,
    vmax: float = 5000.0,
) -> tuple[float, float, float]:
    """Map an expression value to an RGB color.

    Uses a green-to-red gradient matching Java's CMAP2:
    - Below vmin: gray (0.5, 0.5, 0.5)
    - vmin to midpoint: green gradient (dark to bright)
    - midpoint to vmax: red gradient (dark to bright)

    Args:
        value: The expression value.
        vmin: Minimum value (maps to dark green).
        vmax: Maximum value (maps to bright red).

    Returns:
        (r, g, b) tuple with values in [0, 1].
    """
    if value < vmin:
        return (0.5, 0.5, 0.5)  # Gray for sub-threshold

    if vmax <= vmin:
        return (0.0, 1.0, 0.0)  # Default green

    # Normalize to 0-1
    t = min(1.0, (value - vmin) / (vmax - vmin))

    if t < 0.5:
        # Green gradient: dark green -> bright green
        g = 0.3 + 0.7 * (t / 0.5)
        return (0.0, g, 0.0)
    else:
        # Red gradient: dark red -> bright red
        r = 0.3 + 0.7 * ((t - 0.5) / 0.5)
        return (r, 0.0, 0.0)


def compute_tree_bounds(nodes: dict[str, LayoutNode]) -> tuple[float, float, float, float]:
    """Compute the bounding box of a layout.

    Args:
        nodes: Layout node dictionary from compute_layout().

    Returns:
        (x_min, y_min, x_max, y_max) bounding box.
    """
    if not nodes:
        return (0.0, 0.0, 0.0, 0.0)

    x_vals = [n.x for n in nodes.values()]
    y_starts = [n.y_start for n in nodes.values()]
    y_ends = [n.y_end for n in nodes.values()]

    return (
        min(x_vals),
        min(y_starts),
        max(x_vals),
        max(y_ends),
    )
