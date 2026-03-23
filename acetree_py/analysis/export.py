"""Export utilities — CSV and Newick tree format export.

Provides functions to export cell lineage data, nucleus records, and
expression time series to standard file formats for downstream analysis.

Supported formats:
  - CSV: cell table (summary per cell) and nucleus table (per-timepoint)
  - Newick: standard phylogenetic tree format (for tree viewers like FigTree)
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import TextIO

from ..core.cell import Cell, CellFate
from ..core.lineage import LineageTree
from ..core.nucleus import Nucleus
from .expression import cell_expression_time_series

logger = logging.getLogger(__name__)


# ── CSV Export ───────────────────────────────────────────────────


def export_cell_table_csv(
    tree: LineageTree,
    output: str | Path | TextIO,
) -> None:
    """Export a cell summary table to CSV.

    One row per cell with: name, start_time, end_time, lifetime, fate,
    parent, num_children, depth, mean_expression, max_expression, onset_time.

    Args:
        tree: The lineage tree to export.
        output: Output file path or file-like object.
    """
    fieldnames = [
        "name", "start_time", "end_time", "lifetime", "fate",
        "parent", "num_children", "depth",
        "mean_expression", "max_expression", "onset_time",
    ]

    should_close = False
    if isinstance(output, (str, Path)):
        fh = open(output, "w", newline="", encoding="utf-8")
        should_close = True
    else:
        fh = output

    try:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for cell in _sorted_cells(tree):
            ts = cell_expression_time_series(cell)
            writer.writerow({
                "name": cell.name,
                "start_time": cell.start_time,
                "end_time": cell.end_time,
                "lifetime": cell.lifetime,
                "fate": cell.end_fate.name,
                "parent": cell.parent.name if cell.parent else "",
                "num_children": len(cell.children),
                "depth": cell.depth(),
                "mean_expression": f"{ts.mean:.2f}",
                "max_expression": f"{ts.max_value:.2f}",
                "onset_time": ts.onset_time if ts.onset_time is not None else "",
            })

        logger.info("Exported cell table: %d cells", tree.num_cells)
    finally:
        if should_close:
            fh.close()


def export_nucleus_table_csv(
    nuclei_record: list[list[Nucleus]],
    output: str | Path | TextIO,
) -> None:
    """Export a nucleus table to CSV.

    One row per alive nucleus per timepoint with all fields.

    Args:
        nuclei_record: The raw nuclei record.
        output: Output file path or file-like object.
    """
    fieldnames = [
        "timepoint", "index", "x", "y", "z", "size",
        "identity", "assigned_id", "status",
        "predecessor", "successor1", "successor2",
        "weight", "rweight", "rwraw",
        "rwcorr1", "rwcorr2", "rwcorr3", "rwcorr4",
    ]

    should_close = False
    if isinstance(output, (str, Path)):
        fh = open(output, "w", newline="", encoding="utf-8")
        should_close = True
    else:
        fh = output

    try:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        total = 0
        for t_idx, nuclei in enumerate(nuclei_record):
            time = t_idx + 1
            for nuc in nuclei:
                if not nuc.is_alive:
                    continue
                writer.writerow({
                    "timepoint": time,
                    "index": nuc.index,
                    "x": nuc.x,
                    "y": nuc.y,
                    "z": f"{nuc.z:.1f}",
                    "size": nuc.size,
                    "identity": nuc.identity,
                    "assigned_id": nuc.assigned_id,
                    "status": nuc.status,
                    "predecessor": nuc.predecessor,
                    "successor1": nuc.successor1,
                    "successor2": nuc.successor2,
                    "weight": nuc.weight,
                    "rweight": nuc.rweight,
                    "rwraw": nuc.rwraw,
                    "rwcorr1": nuc.rwcorr1,
                    "rwcorr2": nuc.rwcorr2,
                    "rwcorr3": nuc.rwcorr3,
                    "rwcorr4": nuc.rwcorr4,
                })
                total += 1

        logger.info("Exported nucleus table: %d rows", total)
    finally:
        if should_close:
            fh.close()


# ── Newick Tree Export ───────────────────────────────────────────


def export_newick(
    tree: LineageTree,
    output: str | Path | TextIO | None = None,
    include_branch_lengths: bool = True,
) -> str:
    """Export the lineage tree in Newick format.

    The Newick format is the standard for representing phylogenetic trees
    as a nested parenthesized string. Branch lengths represent cell lifetimes.

    Format: ((A:10,B:12)AB:8,(C:5,D:7)CD:6)root:1;

    Args:
        tree: The lineage tree to export.
        output: Optional output file path or file-like object.
            If None, returns the string only.
        include_branch_lengths: If True, include :length annotations.

    Returns:
        The Newick string.
    """
    if tree.root is None:
        return ";"

    newick = _cell_to_newick(tree.root, include_branch_lengths) + ";"

    if output is not None:
        should_close = False
        if isinstance(output, (str, Path)):
            fh = open(output, "w", encoding="utf-8")
            should_close = True
        else:
            fh = output

        try:
            fh.write(newick)
            fh.write("\n")
        finally:
            if should_close:
                fh.close()

    logger.info("Exported Newick tree")
    return newick


def _cell_to_newick(cell: Cell, include_lengths: bool) -> str:
    """Recursively convert a Cell subtree to Newick format."""
    name = cell.name.replace("(", "_").replace(")", "_").replace(",", "_")
    length = cell.lifetime

    if cell.children:
        children_str = ",".join(
            _cell_to_newick(c, include_lengths) for c in cell.children
        )
        if include_lengths:
            return f"({children_str}){name}:{length}"
        return f"({children_str}){name}"
    else:
        if include_lengths:
            return f"{name}:{length}"
        return name


# ── Expression CSV Export ────────────────────────────────────────


def export_expression_csv(
    tree: LineageTree,
    output: str | Path | TextIO,
) -> None:
    """Export per-cell expression summary to CSV.

    Combines cell info with expression time series statistics.

    Args:
        tree: The lineage tree.
        output: Output file path or file-like object.
    """
    fieldnames = [
        "name", "start_time", "end_time", "fate", "parent",
        "expr_mean", "expr_max", "expr_min", "expr_total",
        "onset_time", "num_timepoints",
    ]

    should_close = False
    if isinstance(output, (str, Path)):
        fh = open(output, "w", newline="", encoding="utf-8")
        should_close = True
    else:
        fh = output

    try:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for cell in _sorted_cells(tree):
            ts = cell_expression_time_series(cell)
            writer.writerow({
                "name": cell.name,
                "start_time": cell.start_time,
                "end_time": cell.end_time,
                "fate": cell.end_fate.name,
                "parent": cell.parent.name if cell.parent else "",
                "expr_mean": f"{ts.mean:.2f}",
                "expr_max": f"{ts.max_value:.2f}",
                "expr_min": f"{ts.min_value:.2f}",
                "expr_total": f"{ts.total:.2f}",
                "onset_time": ts.onset_time if ts.onset_time is not None else "",
                "num_timepoints": ts.num_timepoints,
            })

        logger.info("Exported expression CSV")
    finally:
        if should_close:
            fh.close()


# ── Helpers ──────────────────────────────────────────────────────


def _sorted_cells(tree: LineageTree) -> list[Cell]:
    """Return all cells sorted by start_time, then name."""
    cells = tree.all_cells()
    cells.sort(key=lambda c: (c.start_time, c.name))
    return cells
