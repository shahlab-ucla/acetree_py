"""Post-hoc validation of Sulston naming assignments.

After the naming pipeline runs, this module checks the consistency of
assignments and flags potential errors. It does NOT change names — it
only produces warnings that can be displayed in the GUI or logged.

Checks performed:
  1. Sister cells should have complementary Sulston suffixes
  2. Named cells should have unique names (no duplicates at same timepoint)
  3. Cell lifetimes should be biologically reasonable
  4. Position continuity (no teleportation)
  5. Division timing consistency for same-generation cells
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..core.cell import Cell, CellFate
from ..core.lineage import LineageTree
from ..core.nucleus import Nucleus
from .sulston_names import complement, FOUNDER_CELLS

logger = logging.getLogger(__name__)

# Thresholds
MIN_CELL_LIFETIME = 2           # Minimum frames a real cell should exist
MAX_POSITION_JUMP = 30.0        # Max pixels per frame for position continuity
MAX_DIVISION_TIMING_GAP = 8     # Max timing diff between same-depth cousin divisions


@dataclass
class NamingWarning:
    """A warning about a potentially incorrect name assignment.

    Attributes:
        cell_name: The name of the cell with the issue.
        category: Warning category (e.g. "sister_mismatch", "short_lifetime").
        message: Human-readable description.
        severity: "info", "warning", or "error".
        confidence_impact: Suggested reduction in confidence (0-1).
    """

    cell_name: str
    category: str
    message: str
    severity: str = "warning"
    confidence_impact: float = 0.1


def validate_naming(
    tree: LineageTree,
    nuclei_record: list[list[Nucleus]] | None = None,
) -> list[NamingWarning]:
    """Run all validation checks on a named lineage tree.

    Args:
        tree: The lineage tree with Sulston names assigned.
        nuclei_record: Optional nuclei record for position checks.

    Returns:
        List of NamingWarning objects.
    """
    warnings: list[NamingWarning] = []

    if tree.root is None:
        return warnings

    warnings.extend(_check_sister_suffixes(tree))
    warnings.extend(_check_duplicate_names(tree))
    warnings.extend(_check_cell_lifetimes(tree))
    warnings.extend(_check_division_timing(tree))

    if nuclei_record is not None:
        warnings.extend(_check_position_continuity(tree, nuclei_record))

    if warnings:
        logger.info(
            "Naming validation: %d warnings (%d errors, %d warnings, %d info)",
            len(warnings),
            sum(1 for w in warnings if w.severity == "error"),
            sum(1 for w in warnings if w.severity == "warning"),
            sum(1 for w in warnings if w.severity == "info"),
        )

    return warnings


def _check_sister_suffixes(tree: LineageTree) -> list[NamingWarning]:
    """Check that sister cells have complementary Sulston suffixes."""
    warnings = []

    for cell in tree.all_cells():
        if len(cell.children) != 2:
            continue

        c1, c2 = cell.children
        n1, n2 = c1.name, c2.name

        # Skip cells with auto-generated names
        if n1.startswith("Nuc") or n2.startswith("Nuc"):
            continue

        # Skip founder cells with special naming (e.g., EMS -> E, MS)
        if cell.name in FOUNDER_CELLS:
            continue

        # Both should share the parent prefix
        if not n1.startswith(cell.name) or not n2.startswith(cell.name):
            # Special cases: P0->AB/P1, P1->EMS/P2, etc.
            if cell.name in FOUNDER_CELLS:
                continue
            warnings.append(NamingWarning(
                cell.name, "naming_prefix",
                f"Daughters {n1}, {n2} don't share parent prefix {cell.name}",
                severity="warning",
                confidence_impact=0.2,
            ))
            continue

        # Extract suffixes
        suffix1 = n1[len(cell.name):]
        suffix2 = n2[len(cell.name):]

        if len(suffix1) == 1 and len(suffix2) == 1:
            expected2 = complement(suffix1)
            if suffix2 != expected2:
                warnings.append(NamingWarning(
                    cell.name, "sister_mismatch",
                    f"Daughter suffixes '{suffix1}' and '{suffix2}' are not "
                    f"complements (expected '{suffix1}' and '{expected2}')",
                    severity="error",
                    confidence_impact=0.5,
                ))

    return warnings


def _check_duplicate_names(tree: LineageTree) -> list[NamingWarning]:
    """Check for cells with duplicate names (excluding auto-generated)."""
    warnings = []
    name_counts: dict[str, int] = {}

    for cell in tree.all_cells():
        name = cell.name
        if name.startswith("Nuc") or not name:
            continue
        name_counts[name] = name_counts.get(name, 0) + 1

    for name, count in name_counts.items():
        if count > 1:
            warnings.append(NamingWarning(
                name, "duplicate_name",
                f"Name '{name}' appears {count} times in the lineage tree",
                severity="error",
                confidence_impact=0.3,
            ))

    return warnings


def _check_cell_lifetimes(tree: LineageTree) -> list[NamingWarning]:
    """Check that cell lifetimes are biologically reasonable."""
    warnings = []

    for cell in tree.all_cells():
        if cell.start_time < 0:
            continue  # Dummy cell

        lt = cell.lifetime
        if lt < MIN_CELL_LIFETIME and cell.end_fate == CellFate.DIVIDED:
            warnings.append(NamingWarning(
                cell.name, "short_lifetime",
                f"Cell exists for only {lt} timepoint(s) before dividing",
                severity="warning",
                confidence_impact=0.15,
            ))

    return warnings


def _check_division_timing(tree: LineageTree) -> list[NamingWarning]:
    """Check that same-generation cells divide at roughly the same time.

    In C. elegans, cells at the same depth in the lineage tree tend to
    divide near-synchronously. Large timing differences suggest naming errors.
    """
    warnings = []

    # Group dividing cells by depth
    depth_to_times: dict[int, list[tuple[str, int]]] = {}

    for cell in tree.all_cells():
        if cell.end_fate != CellFate.DIVIDED or cell.start_time < 0:
            continue
        d = cell.depth()
        if d not in depth_to_times:
            depth_to_times[d] = []
        depth_to_times[d].append((cell.name, cell.end_time))

    for depth, entries in depth_to_times.items():
        if len(entries) < 2:
            continue

        times = [t for _, t in entries]
        median_time = sorted(times)[len(times) // 2]

        for name, div_time in entries:
            gap = abs(div_time - median_time)
            if gap > MAX_DIVISION_TIMING_GAP:
                warnings.append(NamingWarning(
                    name, "timing_outlier",
                    f"Divides at t={div_time}, median for depth {depth} is {median_time} "
                    f"(gap={gap} frames)",
                    severity="info",
                    confidence_impact=0.05,
                ))

    return warnings


def _check_position_continuity(
    tree: LineageTree,
    nuclei_record: list[list[Nucleus]],
) -> list[NamingWarning]:
    """Check for large position jumps within a cell's lifetime."""
    warnings = []

    for cell in tree.all_cells():
        if len(cell.nuclei) < 2:
            continue

        for i in range(1, len(cell.nuclei)):
            t_prev, nuc_prev = cell.nuclei[i - 1]
            t_curr, nuc_curr = cell.nuclei[i]

            dt = t_curr - t_prev
            if dt <= 0:
                continue

            dx = nuc_curr.x - nuc_prev.x
            dy = nuc_curr.y - nuc_prev.y
            dz = nuc_curr.z - nuc_prev.z  # In planes, not physical
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            speed = dist / max(1, dt)
            if speed > MAX_POSITION_JUMP:
                warnings.append(NamingWarning(
                    cell.name, "position_jump",
                    f"Large position jump at t={t_curr}: {dist:.0f}px in {dt} frame(s) "
                    f"({speed:.0f}px/frame)",
                    severity="warning",
                    confidence_impact=0.1,
                ))

    return warnings
