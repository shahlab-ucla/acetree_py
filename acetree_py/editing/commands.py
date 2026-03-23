"""Edit commands — undoable operations on nuclei data.

Each command captures enough state to both execute and undo itself.
Commands operate on a nuclei_record (list[list[Nucleus]]) which is
the central mutable data structure.

All structural edits (add, remove, relink, kill) require the caller
to rebuild the lineage tree afterward. Commands only mutate the
nuclei_record; tree rebuilding is handled by EditHistory or the caller.

Ported from: org.rhwlab.nucedit.* (NucRelinkDialog, KillCellsDialog,
AddOneDialog, Lazarus, etc.)

Key improvement over Java: Every operation is fully undoable via the
Command pattern. Java had no undo support — edits directly mutated
shared state with no rollback capability.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.nucleus import NILLI, Nucleus

logger = logging.getLogger(__name__)

# Type alias for the nuclei record
NucleiRecord = list[list[Nucleus]]


class EditCommand(ABC):
    """Abstract base class for all edit commands.

    Each command must implement execute() and undo(), and provide
    a human-readable description for the edit log / UI display.
    """

    @abstractmethod
    def execute(self, nuclei_record: NucleiRecord) -> None:
        """Apply this edit to the nuclei record.

        Args:
            nuclei_record: The mutable nuclei data (list of timepoints).
        """
        ...

    @abstractmethod
    def undo(self, nuclei_record: NucleiRecord) -> None:
        """Reverse this edit.

        Args:
            nuclei_record: The mutable nuclei data.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this edit."""
        ...


@dataclass
class AddNucleus(EditCommand):
    """Add a new nucleus at a specific timepoint.

    The nucleus is appended to the end of the timepoint's list.
    Its index is set to len(nuclei_at_time) + 1 (1-based).
    """

    time: int  # 1-based timepoint
    x: int
    y: int
    z: float
    size: int = 20
    identity: str = ""
    predecessor: int = NILLI

    # Set after execute
    _added_index: int = 0

    def execute(self, nuclei_record: NucleiRecord) -> None:
        t_idx = self.time - 1
        # Extend record if needed
        while t_idx >= len(nuclei_record):
            nuclei_record.append([])

        nuclei_list = nuclei_record[t_idx]
        self._added_index = len(nuclei_list) + 1  # 1-based

        nuc = Nucleus(
            index=self._added_index,
            x=self.x,
            y=self.y,
            z=self.z,
            size=self.size,
            identity=self.identity,
            status=1,
            predecessor=self.predecessor,
        )
        nuclei_list.append(nuc)
        logger.info("Added nucleus at t=%d idx=%d pos=(%d,%d,%.1f)",
                     self.time, self._added_index, self.x, self.y, self.z)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        t_idx = self.time - 1
        if t_idx < len(nuclei_record) and nuclei_record[t_idx]:
            nuclei_record[t_idx].pop()
            logger.info("Undid add nucleus at t=%d", self.time)

    @property
    def description(self) -> str:
        name = self.identity or f"({self.x},{self.y},{self.z:.0f})"
        return f"Add nucleus {name} at t={self.time}"


@dataclass
class RemoveNucleus(EditCommand):
    """Remove a nucleus by setting its status to dead.

    Does NOT physically remove it from the list (which would break
    index-based links). Instead marks it as dead (status = -1) and
    clears its identity.

    This matches Java's kill behavior for a single nucleus.
    """

    time: int  # 1-based timepoint
    index: int  # 1-based nucleus index within timepoint

    # Saved state for undo
    _old_status: int = 0
    _old_identity: str = ""
    _old_assigned_id: str = ""

    def execute(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        self._old_status = nuc.status
        self._old_identity = nuc.identity
        self._old_assigned_id = nuc.assigned_id

        nuc.status = -1
        nuc.identity = ""
        nuc.assigned_id = ""
        logger.info("Removed nucleus at t=%d idx=%d (was %s)",
                     self.time, self.index, self._old_identity)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        nuc.status = self._old_status
        nuc.identity = self._old_identity
        nuc.assigned_id = self._old_assigned_id
        logger.info("Undid remove nucleus at t=%d idx=%d", self.time, self.index)

    @property
    def description(self) -> str:
        return f"Remove nucleus at t={self.time} idx={self.index}"


@dataclass
class MoveNucleus(EditCommand):
    """Move a nucleus to a new position and/or resize it.

    Any of x, y, z, size can be None to leave unchanged.
    """

    time: int  # 1-based timepoint
    index: int  # 1-based nucleus index
    new_x: int | None = None
    new_y: int | None = None
    new_z: float | None = None
    new_size: int | None = None

    # Saved state for undo
    _old_x: int = 0
    _old_y: int = 0
    _old_z: float = 0.0
    _old_size: int = 0

    def execute(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        self._old_x = nuc.x
        self._old_y = nuc.y
        self._old_z = nuc.z
        self._old_size = nuc.size

        if self.new_x is not None:
            nuc.x = self.new_x
        if self.new_y is not None:
            nuc.y = self.new_y
        if self.new_z is not None:
            nuc.z = self.new_z
        if self.new_size is not None:
            nuc.size = self.new_size
        logger.info("Moved nucleus at t=%d idx=%d to (%d,%d,%.1f) size=%d",
                     self.time, self.index, nuc.x, nuc.y, nuc.z, nuc.size)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        nuc.x = self._old_x
        nuc.y = self._old_y
        nuc.z = self._old_z
        nuc.size = self._old_size
        logger.info("Undid move nucleus at t=%d idx=%d", self.time, self.index)

    @property
    def description(self) -> str:
        parts = []
        if self.new_x is not None:
            parts.append(f"x={self.new_x}")
        if self.new_y is not None:
            parts.append(f"y={self.new_y}")
        if self.new_z is not None:
            parts.append(f"z={self.new_z:.1f}")
        if self.new_size is not None:
            parts.append(f"size={self.new_size}")
        return f"Move nucleus at t={self.time} idx={self.index}: {', '.join(parts)}"


@dataclass
class RenameCell(EditCommand):
    """Force a name on a nucleus (sets assigned_id).

    The assigned_id survives the naming pipeline — it's a manual override.
    """

    time: int  # 1-based timepoint
    index: int  # 1-based nucleus index
    new_name: str

    # Saved state for undo
    _old_identity: str = ""
    _old_assigned_id: str = ""

    def execute(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        self._old_identity = nuc.identity
        self._old_assigned_id = nuc.assigned_id

        nuc.assigned_id = self.new_name
        nuc.identity = self.new_name
        logger.info("Renamed nucleus at t=%d idx=%d to '%s' (was '%s')",
                     self.time, self.index, self.new_name, self._old_identity)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        nuc.identity = self._old_identity
        nuc.assigned_id = self._old_assigned_id
        logger.info("Undid rename at t=%d idx=%d", self.time, self.index)

    @property
    def description(self) -> str:
        return f"Rename nucleus at t={self.time} idx={self.index} to '{self.new_name}'"


@dataclass
class RelinkNucleus(EditCommand):
    """Change a nucleus's predecessor link.

    This is the core relink operation. It changes which parent cell
    a nucleus is connected to. Optionally also updates the old/new
    parent's successor links.
    """

    time: int  # 1-based timepoint of the nucleus being relinked
    index: int  # 1-based index of the nucleus being relinked
    new_predecessor: int  # New predecessor index (1-based, or NILLI)

    # Saved state for undo
    _old_predecessor: int = NILLI
    _old_parent_succ1: int = NILLI
    _old_parent_succ2: int = NILLI
    _old_parent_time: int = 0
    _old_parent_index: int = 0
    _new_parent_succ1: int = NILLI
    _new_parent_succ2: int = NILLI
    _new_parent_time: int = 0
    _new_parent_index: int = 0

    def execute(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        self._old_predecessor = nuc.predecessor

        # Disconnect from old parent's successor list
        if nuc.predecessor != NILLI and self.time >= 2:
            old_parent = _get_nucleus_safe(nuclei_record, self.time - 1, nuc.predecessor)
            if old_parent is not None:
                self._old_parent_time = self.time - 1
                self._old_parent_index = nuc.predecessor
                self._old_parent_succ1 = old_parent.successor1
                self._old_parent_succ2 = old_parent.successor2
                _remove_successor(old_parent, self.index)

        # Set new predecessor
        nuc.predecessor = self.new_predecessor

        # Connect to new parent's successor list
        if self.new_predecessor != NILLI and self.time >= 2:
            new_parent = _get_nucleus_safe(nuclei_record, self.time - 1, self.new_predecessor)
            if new_parent is not None:
                self._new_parent_time = self.time - 1
                self._new_parent_index = self.new_predecessor
                self._new_parent_succ1 = new_parent.successor1
                self._new_parent_succ2 = new_parent.successor2
                _add_successor(new_parent, self.index)

        logger.info("Relinked t=%d idx=%d: pred %d -> %d",
                     self.time, self.index, self._old_predecessor, self.new_predecessor)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        nuc.predecessor = self._old_predecessor

        # Restore old parent's successors
        if self._old_parent_index != 0:
            old_parent = _get_nucleus_safe(nuclei_record, self._old_parent_time, self._old_parent_index)
            if old_parent is not None:
                old_parent.successor1 = self._old_parent_succ1
                old_parent.successor2 = self._old_parent_succ2

        # Restore new parent's successors
        if self._new_parent_index != 0:
            new_parent = _get_nucleus_safe(nuclei_record, self._new_parent_time, self._new_parent_index)
            if new_parent is not None:
                new_parent.successor1 = self._new_parent_succ1
                new_parent.successor2 = self._new_parent_succ2

        logger.info("Undid relink at t=%d idx=%d", self.time, self.index)

    @property
    def description(self) -> str:
        return f"Relink nucleus at t={self.time} idx={self.index} to pred={self.new_predecessor}"


@dataclass
class KillCell(EditCommand):
    """Kill a named cell across a range of timepoints.

    Walks through timepoints from start_time to end_time (inclusive),
    finds nuclei with matching identity, and sets their status to dead.

    Matches Java KillCellsDialog behavior.
    """

    cell_name: str
    start_time: int  # 1-based
    end_time: int | None = None  # 1-based, None = all remaining

    # Saved state for undo (list of (time, index, old_status, old_identity, old_assigned_id))
    _killed: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._killed is None:
            self._killed = []

    def execute(self, nuclei_record: NucleiRecord) -> None:
        self._killed = []
        end = self.end_time if self.end_time is not None else len(nuclei_record)
        end = min(end, len(nuclei_record))

        for t_1based in range(self.start_time, end + 1):
            t_idx = t_1based - 1
            if t_idx < 0 or t_idx >= len(nuclei_record):
                continue
            for nuc in nuclei_record[t_idx]:
                if nuc.identity == self.cell_name and nuc.is_alive:
                    self._killed.append((
                        t_1based, nuc.index,
                        nuc.status, nuc.identity, nuc.assigned_id,
                    ))
                    nuc.status = -1
                    nuc.identity = ""
                    nuc.assigned_id = ""

        logger.info("Killed cell '%s': %d nuclei across t=%d-%d",
                     self.cell_name, len(self._killed), self.start_time, end)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        for t_1based, idx, old_status, old_identity, old_assigned_id in self._killed:
            nuc = _get_nucleus(nuclei_record, t_1based, idx)
            nuc.status = old_status
            nuc.identity = old_identity
            nuc.assigned_id = old_assigned_id
        logger.info("Undid kill cell '%s': restored %d nuclei",
                     self.cell_name, len(self._killed))

    @property
    def description(self) -> str:
        end = self.end_time or "end"
        return f"Kill cell '{self.cell_name}' from t={self.start_time} to t={end}"


@dataclass
class ResurrectCell(EditCommand):
    """Resurrect a dead nucleus (set status back to alive).

    This is the inverse of RemoveNucleus for a single nucleus.
    """

    time: int  # 1-based
    index: int  # 1-based
    identity: str = ""  # Name to assign upon resurrection

    # Saved state for undo
    _old_status: int = 0
    _old_identity: str = ""
    _old_assigned_id: str = ""

    def execute(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        self._old_status = nuc.status
        self._old_identity = nuc.identity
        self._old_assigned_id = nuc.assigned_id

        nuc.status = 1
        if self.identity:
            nuc.identity = self.identity
        logger.info("Resurrected nucleus at t=%d idx=%d as '%s'",
                     self.time, self.index, nuc.identity)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        nuc.status = self._old_status
        nuc.identity = self._old_identity
        nuc.assigned_id = self._old_assigned_id
        logger.info("Undid resurrect at t=%d idx=%d", self.time, self.index)

    @property
    def description(self) -> str:
        return f"Resurrect nucleus at t={self.time} idx={self.index}"


@dataclass
class RelinkWithInterpolation(EditCommand):
    """Relink two cells by creating interpolated nuclei between them.

    Given a start nucleus (at start_time) and an end nucleus (at end_time),
    creates linearly interpolated nuclei at each intermediate timepoint
    and chains them together via predecessor/successor links.

    This matches Java's NucRelinkDialog.createAndAddCells() behavior.
    """

    start_time: int  # 1-based
    start_index: int  # 1-based index at start_time
    end_time: int  # 1-based
    end_index: int  # 1-based index at end_time

    # Saved state for undo
    _added_nuclei: list = None  # type: ignore[assignment]
    _old_end_pred: int = NILLI
    _old_start_succ1: int = NILLI
    _old_start_succ2: int = NILLI

    def __post_init__(self) -> None:
        if self._added_nuclei is None:
            self._added_nuclei = []

    def execute(self, nuclei_record: NucleiRecord) -> None:
        self._added_nuclei = []

        start_nuc = _get_nucleus(nuclei_record, self.start_time, self.start_index)
        end_nuc = _get_nucleus(nuclei_record, self.end_time, self.end_index)

        # Save end nucleus's old predecessor
        self._old_end_pred = end_nuc.predecessor
        self._old_start_succ1 = start_nuc.successor1
        self._old_start_succ2 = start_nuc.successor2

        num_steps = self.end_time - self.start_time
        if num_steps <= 1:
            # Adjacent timepoints, just link directly
            end_nuc.predecessor = self.start_index
            _add_successor(start_nuc, self.end_index)
            return

        # Create interpolated nuclei for intermediate timepoints
        prev_index = self.start_index
        for step in range(1, num_steps):
            t_1based = self.start_time + step
            t_idx = t_1based - 1
            frac = step / num_steps

            # Linear interpolation
            ix = round(start_nuc.x + (end_nuc.x - start_nuc.x) * frac)
            iy = round(start_nuc.y + (end_nuc.y - start_nuc.y) * frac)
            iz = start_nuc.z + (end_nuc.z - start_nuc.z) * frac
            isize = round(start_nuc.size + (end_nuc.size - start_nuc.size) * frac)

            # Ensure timepoint exists
            while t_idx >= len(nuclei_record):
                nuclei_record.append([])

            new_index = len(nuclei_record[t_idx]) + 1
            new_nuc = Nucleus(
                index=new_index,
                x=ix,
                y=iy,
                z=iz,
                size=isize,
                status=1,
                predecessor=prev_index,
                identity=start_nuc.identity,
            )
            nuclei_record[t_idx].append(new_nuc)
            self._added_nuclei.append((t_1based, new_index))

            # Link previous to this
            if step == 1:
                _add_successor(start_nuc, new_index)
            else:
                prev_nuc = _get_nucleus(nuclei_record, t_1based - 1, prev_index)
                prev_nuc.successor1 = new_index

            prev_index = new_index

        # Link last interpolated to end nucleus
        end_nuc.predecessor = prev_index
        last_interp = _get_nucleus(nuclei_record, self.end_time - 1, prev_index)
        last_interp.successor1 = self.end_index

        logger.info("Relinked with %d interpolated nuclei from t=%d to t=%d",
                     len(self._added_nuclei), self.start_time, self.end_time)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        # Restore end nucleus predecessor
        end_nuc = _get_nucleus(nuclei_record, self.end_time, self.end_index)
        end_nuc.predecessor = self._old_end_pred

        # Restore start nucleus successors
        start_nuc = _get_nucleus(nuclei_record, self.start_time, self.start_index)
        start_nuc.successor1 = self._old_start_succ1
        start_nuc.successor2 = self._old_start_succ2

        # Remove interpolated nuclei (in reverse order)
        for t_1based, _ in reversed(self._added_nuclei):
            t_idx = t_1based - 1
            if t_idx < len(nuclei_record) and nuclei_record[t_idx]:
                nuclei_record[t_idx].pop()

        self._added_nuclei = []
        logger.info("Undid relink interpolation from t=%d to t=%d",
                     self.start_time, self.end_time)

    @property
    def description(self) -> str:
        return (f"Relink t={self.start_time} idx={self.start_index} "
                f"to t={self.end_time} idx={self.end_index} with interpolation")


# ── Helper functions ─────────────────────────────────────────────


def _get_nucleus(nuclei_record: NucleiRecord, time: int, index: int) -> Nucleus:
    """Get a nucleus by 1-based time and 1-based index. Raises on invalid."""
    t_idx = time - 1
    n_idx = index - 1
    if t_idx < 0 or t_idx >= len(nuclei_record):
        raise IndexError(f"Timepoint {time} out of range (1-{len(nuclei_record)})")
    nuclei = nuclei_record[t_idx]
    if n_idx < 0 or n_idx >= len(nuclei):
        raise IndexError(f"Nucleus index {index} out of range at t={time} (1-{len(nuclei)})")
    return nuclei[n_idx]


def _get_nucleus_safe(nuclei_record: NucleiRecord, time: int, index: int) -> Nucleus | None:
    """Get a nucleus, returning None if out of range."""
    try:
        return _get_nucleus(nuclei_record, time, index)
    except IndexError:
        return None


def _remove_successor(parent: Nucleus, child_index: int) -> None:
    """Remove a child index from a parent's successor fields."""
    if parent.successor1 == child_index:
        parent.successor1 = parent.successor2
        parent.successor2 = NILLI
    elif parent.successor2 == child_index:
        parent.successor2 = NILLI


def _add_successor(parent: Nucleus, child_index: int) -> None:
    """Add a child index to a parent's successor fields."""
    if parent.successor1 == NILLI:
        parent.successor1 = child_index
    elif parent.successor2 == NILLI:
        parent.successor2 = child_index
    else:
        logger.warning("Nucleus at idx=%d already has 2 successors; cannot add %d",
                        parent.index, child_index)
