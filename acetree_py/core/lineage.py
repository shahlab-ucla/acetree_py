"""Lineage tree builder — constructs Cell tree from nuclei records.

Processes the nuclei_record (list of timepoints, each a list of Nucleus)
to build a tree of Cell objects representing the full cell lineage.

Key concepts:
  - Each Cell spans from birth (division or first appearance) to
    death/division/end of dataset.
  - Cells are linked to their parent and children via the
    predecessor/successor fields in Nucleus objects.
  - A hash_key (timepoint * 100000 + nucleus_index) uniquely identifies
    each cell across timepoints.
  - Dummy ancestor nodes (P0, AB, P1, etc.) are created for the standard
    Sulston lineage display.

Ported from: org.rhwlab.tree.AncesTree
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .cell import Cell, CellFate
from .nucleus import NILLI, Nucleus

logger = logging.getLogger(__name__)


def _make_hash_key(timepoint: int, nucleus_index: int) -> str:
    """Generate a unique hash key for a nucleus at a timepoint.

    Matches Java AncesTree.makeHashKey(): index * 100000 + n.index.
    timepoint is 1-based, nucleus_index is 1-based.
    """
    return str(timepoint * 100000 + nucleus_index)


# ── Dummy ancestor definitions for Sulston lineage ──
# Each entry: (name, parent_name, start_time_offset, children)
# start_time_offset is relative to the first real cell's appearance
_DUMMY_ANCESTORS = [
    # (name, parent_name)
    ("P0", None),
    ("AB", "P0"),
    ("P1", "P0"),
    ("ABa", "AB"),
    ("ABp", "AB"),
    ("EMS", "P1"),
    ("P2", "P1"),
    ("MS", "EMS"),
    ("E", "EMS"),
    ("C", "P2"),
    ("P3", "P2"),
    ("D", "P3"),
    ("P4", "P3"),
]


@dataclass
class LineageTree:
    """The complete cell lineage tree.

    Attributes:
        root: The root Cell (P0 or first observed cell).
        cells_by_name: Lookup from cell name to Cell object.
        cells_by_hash: Lookup from hash_key to Cell object.
        cell_counts: Number of alive cells at each timepoint (0-indexed).
    """

    root: Cell | None = None
    cells_by_name: dict[str, Cell] = field(default_factory=dict)
    cells_by_hash: dict[str, Cell] = field(default_factory=dict)
    cell_counts: list[int] = field(default_factory=list)

    def get_cell(self, name: str) -> Cell | None:
        """Look up a cell by name (case-sensitive)."""
        return self.cells_by_name.get(name)

    def get_cell_icase(self, name: str) -> Cell | None:
        """Look up a cell by name (case-insensitive)."""
        name_lower = name.lower()
        for key, cell in self.cells_by_name.items():
            if key.lower() == name_lower:
                return cell
        return None

    def all_cells(self) -> list[Cell]:
        """Return all cells in the tree."""
        return list(self.cells_by_name.values())

    @property
    def num_cells(self) -> int:
        return len(self.cells_by_name)


def build_lineage_tree(
    nuclei_record: list[list[Nucleus]],
    starting_index: int = 0,
    ending_index: int = -1,
    create_dummy_ancestors: bool = True,
) -> LineageTree:
    """Build a lineage tree from processed nuclei records.

    This is the main entry point, corresponding to AncesTree's constructor
    and processEntries() in Java.

    Args:
        nuclei_record: The full nuclei record (list of timepoints, 0-indexed).
        starting_index: 0-based starting timepoint.
        ending_index: 0-based ending timepoint (-1 for all).
        create_dummy_ancestors: If True, create dummy P0/AB/P1 etc. nodes
            for the Sulston lineage display.

    Returns:
        A LineageTree with all cells linked.
    """
    if ending_index < 0:
        ending_index = len(nuclei_record)

    tree = LineageTree()
    tree.cell_counts = [0] * len(nuclei_record)

    # Maps hash_key -> Cell for tracking across timepoints
    cells_by_hash: dict[str, Cell] = {}

    # Create dummy ancestors if requested
    dummy_cells: dict[str, Cell] = {}
    if create_dummy_ancestors:
        dummy_cells = _create_dummy_ancestors()

    # Process each timepoint
    for t in range(starting_index, min(ending_index, len(nuclei_record))):
        time_1based = t + 1  # Convert to 1-based for hash keys and Cell times
        nuclei = nuclei_record[t]

        # Get next timepoint's nuclei (for detecting divisions)
        next_nuclei = nuclei_record[t + 1] if t + 1 < len(nuclei_record) else None

        for j, nuc in enumerate(nuclei):
            if nuc.status < 1:
                continue  # Skip dead nuclei

            nuc_index_1based = j + 1  # 1-based index within timepoint

            if t == starting_index or nuc.predecessor == NILLI:
                # Root cell or first appearance
                cell = _process_root_cell(
                    nuc, time_1based, nuc_index_1based,
                    dummy_cells, cells_by_hash,
                )
                if tree.root is None and cell.parent is None:
                    tree.root = cell
            else:
                # Has a predecessor — find parent cell
                prev_nuclei = nuclei_record[t - 1]
                pred_idx_0based = nuc.predecessor - 1

                if not (0 <= pred_idx_0based < len(prev_nuclei)):
                    # Invalid predecessor — treat as root
                    cell = _process_root_cell(
                        nuc, time_1based, nuc_index_1based,
                        dummy_cells, cells_by_hash,
                    )
                    continue

                prev_nuc = prev_nuclei[pred_idx_0based]

                # Get the parent cell using predecessor's hash_key
                parent_hash = prev_nuc.hash_key
                if parent_hash is None:
                    # Try to compute it
                    parent_hash = _make_hash_key(time_1based - 1, pred_idx_0based + 1)

                parent_cell = cells_by_hash.get(parent_hash)

                if parent_cell is None:
                    # Parent not found — treat as root
                    cell = _process_root_cell(
                        nuc, time_1based, nuc_index_1based,
                        dummy_cells, cells_by_hash,
                    )
                    continue

                # Check if this is a division or continuation
                if prev_nuc.successor2 == NILLI:
                    # Not dividing — update existing cell
                    hash_key = _make_hash_key(time_1based, nuc_index_1based)
                    nuc.hash_key = parent_cell.hash_key  # Inherit hash
                    parent_cell.end_time = time_1based
                    parent_cell.add_nucleus(time_1based, nuc)
                    # Also register with new hash for lookup
                    cells_by_hash[hash_key] = parent_cell
                    cell = parent_cell
                else:
                    # Division — create new daughter cell
                    hash_key = _make_hash_key(time_1based, nuc_index_1based)
                    nuc.hash_key = hash_key

                    daughter_name = nuc.effective_name or f"Nuc{time_1based}_{j}"

                    # Check if the parent already has a dummy child with this
                    # name (from _create_dummy_ancestors).  If so, merge the
                    # real data into the existing dummy Cell so the tree
                    # stays connected.  Without this, the dummy children
                    # occupy both child slots and real daughters become
                    # orphaned.
                    existing_dummy = None
                    for child in (parent_cell.children or []):
                        if child.name == daughter_name and child.start_time < 0:
                            existing_dummy = child
                            break

                    if existing_dummy is not None:
                        # Merge into the dummy child
                        existing_dummy.start_time = time_1based
                        existing_dummy.end_time = time_1based
                        existing_dummy.hash_key = hash_key
                        existing_dummy.add_nucleus(time_1based, nuc)
                        cell = existing_dummy
                        # Mark parent as divided
                        parent_cell.end_fate = CellFate.DIVIDED
                        if parent_cell.end_time < 0:
                            parent_cell.end_time = time_1based - 1
                    else:
                        cell = Cell(
                            name=daughter_name,
                            start_time=time_1based,
                            end_time=time_1based,
                            hash_key=hash_key,
                        )
                        cell.add_nucleus(time_1based, nuc)

                        # Link to parent — replace a dummy child if possible
                        _link_daughter_to_parent(parent_cell, cell, time_1based)

                    cells_by_hash[hash_key] = cell

            # Check for cell death (no successors and not at final timepoint)
            if next_nuclei is not None and nuc.successor1 == NILLI:
                hash_key = nuc.hash_key or _make_hash_key(time_1based, nuc_index_1based)
                cell_at_hash = cells_by_hash.get(hash_key)
                if cell_at_hash is not None:
                    cell_at_hash.end_time = time_1based
                    cell_at_hash.end_fate = CellFate.DIED

        # Count alive cells at this timepoint
        tree.cell_counts[t] = sum(1 for n in nuclei if n.status >= 1)

    # Update Cell names for cells whose nuclei have assigned_id.
    # Continuation cells reuse the same Cell object across timepoints,
    # so Cell.name is set at birth and never updated.  If the user
    # renames a cell (setting assigned_id), the naming pipeline
    # propagates assigned_id to all continuation nuclei, but the Cell
    # object's .name field stays stale.  Fix that here.
    _apply_assigned_id_names(cells_by_hash)

    # Build name lookup tables
    tree.cells_by_hash = cells_by_hash
    tree.cells_by_name = _build_name_lookup(cells_by_hash, dummy_cells)

    # If no root found from data, use the dummy P0
    if tree.root is None and "P0" in dummy_cells:
        tree.root = dummy_cells["P0"]

    # Adjust dummy ancestors based on actual data
    if create_dummy_ancestors and tree.root is not None:
        _adjust_dummy_timing(dummy_cells, tree)

    return tree


def _create_dummy_ancestors() -> dict[str, Cell]:
    """Create dummy ancestor cells for Sulston lineage display.

    These cells represent the pre-observed lineage (P0, AB, P1, etc.)
    and will be connected to real cells when they are found in the data.
    """
    cells: dict[str, Cell] = {}

    for name, parent_name in _DUMMY_ANCESTORS:
        cell = Cell(
            name=name,
            start_time=-1,  # Will be adjusted later
            end_time=-1,
        )
        cells[name] = cell

        if parent_name and parent_name in cells:
            cells[parent_name].add_child(cell)

    return cells


def _link_daughter_to_parent(
    parent_cell: Cell,
    daughter: Cell,
    time_1based: int,
) -> None:
    """Link a real daughter cell to its parent, handling dummy children.

    If the parent already has 2 children (e.g. from dummy ancestors),
    try to replace an uninitialized dummy child (start_time < 0) with
    the real daughter.  If no dummy slot is available and the parent
    already has 2 real children, log a warning and don't add.
    """
    if len(parent_cell.children) < 2:
        parent_cell.add_child(daughter)
    else:
        # Parent has 2 children already.  See if one is a stale dummy
        # that was never populated with real data.
        replaced = False
        for i, child in enumerate(parent_cell.children):
            if child.start_time < 0:
                # Stale dummy — replace with the real daughter
                parent_cell.children[i] = daughter
                daughter.parent = parent_cell
                replaced = True
                break

        if not replaced:
            logger.warning(
                "Parent '%s' already has 2 real children; cannot add '%s'",
                parent_cell.name, daughter.name,
            )

    # Mark parent as divided when it has 2 children
    if len(parent_cell.children) >= 2:
        parent_cell.end_fate = CellFate.DIVIDED
        parent_cell.end_time = time_1based - 1


def _process_root_cell(
    nuc: Nucleus,
    time_1based: int,
    nuc_index_1based: int,
    dummy_cells: dict[str, Cell],
    cells_by_hash: dict[str, Cell],
) -> Cell:
    """Process a root cell (no predecessor).

    If the cell's identity matches a dummy ancestor, replace the dummy
    with real data. Otherwise, create a new root cell.
    """
    name = nuc.effective_name or f"Nuc{time_1based}_{nuc_index_1based}"
    hash_key = _make_hash_key(time_1based, nuc_index_1based)
    nuc.hash_key = hash_key

    # Check if this matches a dummy ancestor
    if name in dummy_cells:
        cell = dummy_cells[name]
        cell.start_time = time_1based
        cell.end_time = time_1based
        cell.hash_key = hash_key
        cell.add_nucleus(time_1based, nuc)
        cells_by_hash[hash_key] = cell
        return cell

    # Create a new cell
    cell = Cell(
        name=name,
        start_time=time_1based,
        end_time=time_1based,
        hash_key=hash_key,
    )
    cell.add_nucleus(time_1based, nuc)
    cells_by_hash[hash_key] = cell

    # Try to find a parent among dummy ancestors
    for dummy_name, dummy_cell in dummy_cells.items():
        for child in dummy_cell.children:
            if child.name == name:
                # Already linked via dummy tree
                return cell

    return cell


def _apply_assigned_id_names(cells_by_hash: dict[str, Cell]) -> None:
    """Update Cell.name for cells that contain a nucleus with assigned_id.

    When a user renames a cell via assigned_id, the naming pipeline
    propagates the forced name to all continuation nuclei (identity +
    assigned_id).  However, the Cell object that spans those timepoints
    keeps its original .name from birth.  This function scans every Cell
    and, if any of its nuclei carries an assigned_id, updates the Cell's
    .name to match.

    To be deterministic in the face of legacy or externally-edited saves
    where different nuclei in the same chain might carry *different*
    assigned_ids, we pick the earliest (lowest-timepoint) value and log
    a warning if the chain is inconsistent.  With Part 9's cell-scoped
    RenameCell all nuclei in a chain share one assigned_id, so the
    warning should only fire on abnormal inputs.

    This must run *before* _build_name_lookup() so the name dict
    reflects the forced name.
    """
    seen: set[int] = set()  # avoid processing the same Cell twice
    for cell in cells_by_hash.values():
        cell_id = id(cell)
        if cell_id in seen:
            continue
        seen.add(cell_id)

        # Iterate in timepoint order, earliest first
        sorted_nuclei = sorted(cell.nuclei, key=lambda tn: tn[0])
        chosen: str | None = None
        mismatches: list[str] = []
        for _t, nuc in sorted_nuclei:
            if not nuc.assigned_id:
                continue
            if chosen is None:
                chosen = nuc.assigned_id
            elif nuc.assigned_id != chosen:
                mismatches.append(nuc.assigned_id)

        if chosen is not None:
            cell.name = chosen
            if mismatches:
                logger.warning(
                    "Cell '%s' has inconsistent assigned_ids across its chain: "
                    "chose earliest '%s', ignored %s",
                    cell.name, chosen, sorted(set(mismatches)),
                )


def _build_name_lookup(
    cells_by_hash: dict[str, Cell],
    dummy_cells: dict[str, Cell],
) -> dict[str, Cell]:
    """Build the cells_by_name lookup from all cells.

    Handles name collisions by suffixing (`Name_2`, `Name_3`, ...) so
    every cell is still reachable by a unique name.  The Part 9
    validate_rename_cell() prevents user-driven collisions, but this is
    a safety net for legacy saves, bulk imports, or non-RenameCell edits.
    """
    by_name: dict[str, Cell] = {}

    # Add dummy cells first
    for name, cell in dummy_cells.items():
        by_name[name] = cell

    # Add/override with real cells from the data
    polar_count = 0
    # Iterate in a deterministic order so suffix assignment is reproducible
    seen_cells: set[int] = set()
    for cell in cells_by_hash.values():
        if id(cell) in seen_cells:
            continue
        seen_cells.add(id(cell))

        name = cell.name
        if not name:
            continue

        # Handle duplicate polar body names (pre-existing rule)
        if "polar" in name.lower() and name in by_name:
            polar_count += 1
            name = f"polar{polar_count}"
            cell.name = name

        # General collision: same name already used by a *different* cell
        if name in by_name and by_name[name] is not cell:
            # Don't suffix when the existing entry is a dummy ancestor —
            # real cells should supersede dummy ancestors (this preserves
            # prior behavior where real P0/AB/... data replaces dummies).
            existing = by_name[name]
            existing_is_dummy = existing.start_time < 0
            if existing_is_dummy:
                by_name[name] = cell
                continue

            suffix = 2
            while f"{name}_{suffix}" in by_name:
                suffix += 1
            new_name = f"{name}_{suffix}"
            logger.warning(
                "Name collision: '%s' already used by a different cell; "
                "aliasing second cell as '%s'",
                name, new_name,
            )
            name = new_name
            cell.name = name

        by_name[name] = cell

    return by_name


def _adjust_dummy_timing(
    dummy_cells: dict[str, Cell],
    tree: LineageTree,
) -> None:
    """Adjust dummy ancestor timing based on actual observed data.

    Uses the earliest observed cells to back-calculate when earlier
    cells would have existed, based on known C. elegans division timing.
    """
    # Known approximate division intervals (in timepoints)
    # These are rough estimates based on typical C. elegans recordings
    DIVISION_INTERVAL = 15  # ~15 timepoints between divisions

    # Find the earliest real cell in the tree
    earliest_time = float("inf")
    for cell in tree.cells_by_hash.values():
        if cell.start_time > 0 and cell.start_time < earliest_time:
            earliest_time = cell.start_time

    if earliest_time == float("inf"):
        return

    earliest_time = int(earliest_time)

    # Set dummy cell times working backward from the earliest real cell
    # This provides a reasonable visual display
    for name, cell in dummy_cells.items():
        if cell.start_time >= 0:
            continue  # Already has real timing

        # Compute time based on depth in the dummy tree
        depth = cell.depth()
        estimated_start = max(1, earliest_time - (depth + 1) * DIVISION_INTERVAL)
        estimated_end = max(1, earliest_time - depth * DIVISION_INTERVAL - 1)

        cell.start_time = estimated_start
        cell.end_time = estimated_end
