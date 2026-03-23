"""Expression analysis — higher-level analysis of reporter expression data.

Builds on the low-level corrected_red() and compute_red_weights() in core/
to provide per-cell time series extraction, subtree summary statistics,
and sister-cell expression comparisons.

Ported from / consolidates: org.rhwlab.analyze.Analysis* (9 numbered variants)
into a single clean module.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

from ..core.cell import Cell, CellFate
from ..core.nucleus import Nucleus

logger = logging.getLogger(__name__)


@dataclass
class ExpressionTimeSeries:
    """Expression values for a single cell across its lifetime.

    Attributes:
        cell_name: Name of the cell.
        start_time: First timepoint (1-based).
        end_time: Last timepoint (1-based).
        timepoints: List of 1-based timepoints.
        values: List of expression values (rweight) at each timepoint.
    """

    cell_name: str
    start_time: int
    end_time: int
    timepoints: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        """Mean expression value across the cell's lifetime."""
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def max_value(self) -> float:
        """Maximum expression value."""
        return max(self.values) if self.values else 0.0

    @property
    def min_value(self) -> float:
        """Minimum expression value."""
        return min(self.values) if self.values else 0.0

    @property
    def onset_time(self) -> int | None:
        """First timepoint where expression exceeds threshold (> 0).

        Returns None if expression never exceeds 0.
        """
        for t, v in zip(self.timepoints, self.values):
            if v > 0:
                return t
        return None

    @property
    def total(self) -> float:
        """Sum of all expression values."""
        return sum(self.values)

    @property
    def num_timepoints(self) -> int:
        """Number of timepoints with data."""
        return len(self.values)


@dataclass
class SubtreeStats:
    """Summary expression statistics for a subtree rooted at a cell.

    Attributes:
        root_name: Name of the subtree root cell.
        num_cells: Total number of cells in the subtree.
        num_expressing: Number of cells with any expression > 0.
        mean_expression: Mean expression across all cells.
        max_expression: Maximum expression across all cells.
        earliest_onset: Earliest onset time in the subtree.
        total_expression: Sum of all expression values.
    """

    root_name: str
    num_cells: int = 0
    num_expressing: int = 0
    mean_expression: float = 0.0
    max_expression: float = 0.0
    earliest_onset: int | None = None
    total_expression: float = 0.0


@dataclass
class SisterComparison:
    """Expression comparison between two sister cells.

    Attributes:
        parent_name: Name of the parent cell.
        sister1_name: Name of the first sister.
        sister2_name: Name of the second sister.
        sister1_mean: Mean expression for sister 1.
        sister2_mean: Mean expression for sister 2.
        ratio: Ratio of sister1_mean / sister2_mean (or inf if sister2 = 0).
        difference: Absolute difference in mean expression.
    """

    parent_name: str
    sister1_name: str
    sister2_name: str
    sister1_mean: float
    sister2_mean: float

    @property
    def ratio(self) -> float:
        """Ratio of sister1_mean / sister2_mean."""
        if self.sister2_mean == 0:
            return float("inf") if self.sister1_mean > 0 else 1.0
        return self.sister1_mean / self.sister2_mean

    @property
    def difference(self) -> float:
        """Absolute difference in mean expression."""
        return abs(self.sister1_mean - self.sister2_mean)

    @property
    def fold_change(self) -> float:
        """Fold change (always >= 1): max/min ratio."""
        low = min(self.sister1_mean, self.sister2_mean)
        high = max(self.sister1_mean, self.sister2_mean)
        if low == 0:
            return float("inf") if high > 0 else 1.0
        return high / low


# ── Core analysis functions ──────────────────────────────────────


def cell_expression_time_series(
    cell: Cell,
    value_fn: Callable[[Nucleus], float] | None = None,
) -> ExpressionTimeSeries:
    """Extract expression values for a cell across its lifetime.

    Args:
        cell: The cell to analyze.
        value_fn: Optional function to extract a value from a Nucleus.
            Defaults to lambda nuc: nuc.rweight.

    Returns:
        An ExpressionTimeSeries with values at each timepoint.
    """
    if value_fn is None:
        value_fn = lambda nuc: float(nuc.rweight)

    ts = ExpressionTimeSeries(
        cell_name=cell.name,
        start_time=cell.start_time,
        end_time=cell.end_time,
    )

    for time, nuc in cell.nuclei:
        ts.timepoints.append(time)
        ts.values.append(value_fn(nuc))

    return ts


def subtree_expression_stats(
    root: Cell,
    value_fn: Callable[[Nucleus], float] | None = None,
) -> SubtreeStats:
    """Compute summary expression statistics for a subtree.

    Args:
        root: The root cell of the subtree.
        value_fn: Optional function to extract a value from a Nucleus.

    Returns:
        A SubtreeStats summarizing expression in the subtree.
    """
    stats = SubtreeStats(root_name=root.name)
    all_means: list[float] = []
    total = 0.0

    for cell in root.iter_subtree_preorder():
        ts = cell_expression_time_series(cell, value_fn)
        stats.num_cells += 1

        cell_mean = ts.mean
        all_means.append(cell_mean)
        total += ts.total

        if ts.max_value > stats.max_expression:
            stats.max_expression = ts.max_value

        if ts.max_value > 0:
            stats.num_expressing += 1

        onset = ts.onset_time
        if onset is not None:
            if stats.earliest_onset is None or onset < stats.earliest_onset:
                stats.earliest_onset = onset

    stats.total_expression = total
    if all_means:
        stats.mean_expression = sum(all_means) / len(all_means)

    return stats


def compare_sisters(
    parent: Cell,
    value_fn: Callable[[Nucleus], float] | None = None,
) -> SisterComparison | None:
    """Compare expression between two sister cells.

    Args:
        parent: The parent cell (must have exactly 2 children).
        value_fn: Optional function to extract a value from a Nucleus.

    Returns:
        A SisterComparison, or None if the parent doesn't have 2 children.
    """
    if len(parent.children) != 2:
        return None

    s1, s2 = parent.children
    ts1 = cell_expression_time_series(s1, value_fn)
    ts2 = cell_expression_time_series(s2, value_fn)

    return SisterComparison(
        parent_name=parent.name,
        sister1_name=s1.name,
        sister2_name=s2.name,
        sister1_mean=ts1.mean,
        sister2_mean=ts2.mean,
    )


def all_sister_comparisons(
    root: Cell,
    value_fn: Callable[[Nucleus], float] | None = None,
    min_fold_change: float = 0.0,
) -> list[SisterComparison]:
    """Compare expression for all sister pairs in a subtree.

    Args:
        root: Root of the subtree to analyze.
        value_fn: Optional function to extract a value from a Nucleus.
        min_fold_change: Only include pairs with fold change >= this value.

    Returns:
        List of SisterComparison objects, sorted by fold change (descending).
    """
    comparisons = []

    for cell in root.iter_subtree_preorder():
        if len(cell.children) == 2:
            comp = compare_sisters(cell, value_fn)
            if comp is not None:
                if comp.fold_change >= min_fold_change:
                    comparisons.append(comp)

    comparisons.sort(key=lambda c: c.fold_change, reverse=True)
    return comparisons


def expression_onset_map(
    root: Cell,
    value_fn: Callable[[Nucleus], float] | None = None,
) -> dict[str, int | None]:
    """Build a map of cell name → onset time for expression.

    Args:
        root: Root cell of the subtree.
        value_fn: Optional function to extract a value from a Nucleus.

    Returns:
        Dict mapping cell name to onset time (None if never expressed).
    """
    result = {}
    for cell in root.iter_subtree_preorder():
        ts = cell_expression_time_series(cell, value_fn)
        result[cell.name] = ts.onset_time
    return result
