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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.nucleus import NILLI, Nucleus, validate_storable_name

logger = logging.getLogger(__name__)

# Type alias for the nuclei record
NucleiRecord = list[list[Nucleus]]


class EditEffect(str, Enum):
    """A derived-data domain affected by an edit.

    The string values are intentionally stable: application code, plugins,
    and persisted diagnostics may compare them without importing this enum.
    ``EditCommand.structural`` remains the compatibility API for the existing
    nucleus editor; new routing code should prefer :attr:`EditCommand.effects`.
    """

    NUCLEI_TOPOLOGY = "nuclei_topology"
    NUCLEUS_GEOMETRY = "nucleus_geometry"
    ROI_GEOMETRY = "roi_geometry"
    ROI_ASSOCIATION = "roi_association"
    ROI_METADATA = "roi_metadata"
    CONFIG = "config"


# A descriptive alias used by a few integrations.  Keep one enum type so set
# equality and serialization stay straightforward.
EditEffectDomain = EditEffect
EditDomain = EditEffect


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

    @property
    def structural(self) -> bool:
        """Whether this edit changes lineage structure (links, identity, etc.).

        Non-structural edits (e.g. move/resize) skip the expensive
        naming + tree rebuild in _on_edit and just refresh the display.
        """
        return True

    @property
    def effects(self) -> frozenset[EditEffect]:
        """Domains whose derived state may need refreshing after this edit.

        Existing commands did not declare domains.  Their legacy
        ``structural`` flag therefore supplies a backwards-compatible
        default: structural commands affect nuclear topology, while
        non-structural commands affect only nucleus geometry.
        """

        if self.structural:
            return frozenset((EditEffect.NUCLEI_TOPOLOGY,))
        return frozenset((EditEffect.NUCLEUS_GEOMETRY,))

    @property
    def is_noop(self) -> bool:
        """Whether the most recent execution made no change.

        EditHistory uses this after ``execute`` so an accepted UI action that
        does not change data does not consume an undo slot, clear redo, mark
        the document dirty, or trigger an expensive rebuild.
        """
        return False

    def _prepare_execute(self) -> None:
        """Reset failed-execution rollback state before a composite runs us."""
        self._failed_execute_rollback_ready = False

    def _mark_rollback_ready(self) -> None:
        """Declare that ``undo`` has enough state to reverse partial work.

        Commands call this after capturing their undo snapshot and immediately
        before their first mutation.  This distinction matters when validation
        or lookup fails before mutation: calling the ordinary ``undo`` method
        in that case can apply uninitialised defaults to unrelated data.
        """
        self._failed_execute_rollback_ready = True

    def rollback_failed_execute(self, nuclei_record: NucleiRecord) -> None:
        """Undo partial work, if execution reached its first mutation."""
        if not getattr(self, "_failed_execute_rollback_ready", False):
            return
        try:
            self.undo(nuclei_record)
        finally:
            self._failed_execute_rollback_ready = False


@dataclass
class CompositeCommand(EditCommand):
    """Execute several edit commands as one atomic history operation.

    Children execute in the supplied order.  If a child fails, that child is
    given a chance to roll back any partial work and all previously completed
    children are undone in reverse order before the original exception is
    re-raised.  A normal undo also runs in reverse order.
    """

    commands: list[EditCommand]
    label: str = ""

    _executed: list[EditCommand] = field(default_factory=list, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        self._executed = []
        self._mark_rollback_ready()
        for command in self.commands:
            command._prepare_execute()
            try:
                command.execute(nuclei_record)
            except Exception:
                # A command may have captured state and mutated data before
                # failing.  Its undo is therefore part of best-effort atomic
                # rollback, but rollback failures must not hide the cause.
                try:
                    command.rollback_failed_execute(nuclei_record)
                except Exception:
                    logger.exception(
                        "Failed to roll back partially executed command: %s",
                        command.description,
                    )
                for completed in reversed(self._executed):
                    try:
                        completed.undo(nuclei_record)
                    except Exception:
                        logger.exception(
                            "Failed to roll back completed command: %s",
                            completed.description,
                        )
                self._executed = []
                raise
            self._executed.append(command)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        for command in reversed(self._executed):
            command.undo(nuclei_record)
        self._executed = []

    @property
    def description(self) -> str:
        if self.label:
            return self.label
        return "; ".join(command.description for command in self.commands)

    @property
    def structural(self) -> bool:
        return any(command.structural for command in self.commands)

    @property
    def effects(self) -> frozenset[EditEffect]:
        return frozenset(
            effect
            for command in self.commands
            for effect in command.effects
        )

    @property
    def is_noop(self) -> bool:
        return all(command.is_noop for command in self.commands)


@dataclass
class SetBodyAxes(EditCommand):
    """Install a validated manual body frame as one undoable edit."""

    manager: Any
    frame: Any

    _old_auxinfo: Any = field(default=None, init=False)
    _old_identity_assigner: Any = field(default=None, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        self._old_auxinfo = self.manager.auxinfo
        self._old_identity_assigner = self.manager.identity_assigner
        self._mark_rollback_ready()
        try:
            self.manager.set_manual_body_axes(self.frame)
        except Exception:
            self.manager.auxinfo = self._old_auxinfo
            self.manager.identity_assigner = self._old_identity_assigner
            raise

    def undo(self, nuclei_record: NucleiRecord) -> None:
        self.manager.auxinfo = self._old_auxinfo
        self.manager.identity_assigner = self._old_identity_assigner

    @property
    def description(self) -> str:
        reference_time = getattr(self.frame, "reference_time", None)
        suffix = f" at t={reference_time}" if reference_time is not None else ""
        return f"Set manual body axes{suffix}"


@dataclass
class AddNucleus(EditCommand):
    """Add a new nucleus at a specific timepoint.

    The nucleus is appended to the end of the timepoint's list.
    Its index is set to len(nuclei_at_time) + 1 (1-based).

    When ``assigned_id`` is set, it becomes a "forced name" that survives the
    naming pipeline (``IdentityAssigner._clear_all_names`` preserves it, and
    ``_propagate_assigned_ids`` extends it forward/backward through the
    predecessor chain).  This is how a newly-added nucleus inherits the
    name of the cell it's extending: the caller passes the parent's
    ``effective_name`` as ``assigned_id``.
    """

    time: int  # 1-based timepoint
    x: int
    y: int
    z: float
    size: int = 20
    identity: str = ""
    predecessor: int = NILLI
    assigned_id: str = ""

    # Set after execute
    _added_index: int = 0
    _original_record_len: int = 0
    _parent_time: int = 0
    _parent_index: int = 0
    _old_parent_succ1: int = NILLI
    _old_parent_succ2: int = NILLI
    _did_add: bool = field(default=False, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        from .validators import validate_add_nucleus

        errors = validate_add_nucleus(nuclei_record, self.time, self.predecessor)
        if errors:
            raise ValueError("; ".join(errors))
        for value in (self.identity, self.assigned_id):
            error = validate_storable_name(value)
            if error:
                raise ValueError(error)
        t_idx = self.time - 1
        self._original_record_len = len(nuclei_record)
        self._parent_time = 0
        self._parent_index = 0
        self._did_add = False
        self._mark_rollback_ready()
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
            assigned_id=self.assigned_id,
            status=1,
            predecessor=self.predecessor,
        )
        nuclei_list.append(nuc)
        self._did_add = True

        # Maintain the reciprocal link immediately.  Waiting for a global
        # successor rebuild leaves composite gestures and validation looking
        # at a transient one-way lineage.
        if self.predecessor != NILLI and self.time >= 2:
            parent = _get_nucleus_safe(
                nuclei_record, self.time - 1, self.predecessor
            )
            if parent is not None:
                self._parent_time = self.time - 1
                self._parent_index = self.predecessor
                self._old_parent_succ1 = parent.successor1
                self._old_parent_succ2 = parent.successor2
                _add_successor(parent, self._added_index)
        logger.info("Added nucleus at t=%d idx=%d pos=(%d,%d,%.1f) assigned_id=%r",
                     self.time, self._added_index, self.x, self.y, self.z,
                     self.assigned_id)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        t_idx = self.time - 1
        if self._parent_index:
            parent = _get_nucleus_safe(
                nuclei_record, self._parent_time, self._parent_index
            )
            if parent is not None:
                parent.successor1 = self._old_parent_succ1
                parent.successor2 = self._old_parent_succ2
        if self._did_add and t_idx < len(nuclei_record) and nuclei_record[t_idx]:
            nuclei_record[t_idx].pop()
            logger.info("Undid add nucleus at t=%d", self.time)
        while (
            len(nuclei_record) > self._original_record_len
            and not nuclei_record[-1]
        ):
            nuclei_record.pop()
        self._did_add = False

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

        self._mark_rollback_ready()
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

        self._mark_rollback_ready()
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
class SetCellNameState(EditCommand):
    """Set both naming fields across one continuation component.

    This lower-level operation is useful when a structural edit changes a
    historical continuation into a division and both the automatic identity
    and manual override state need to be corrected together.  It snapshots
    every nucleus so undo restores the exact heterogeneous pre-edit state.
    """

    time: int
    index: int
    identity: str
    assigned_id: str

    _touched: list[tuple[int, int, str, str]] = field(default_factory=list)
    _noop: bool = field(default=False, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        for value in (self.identity, self.assigned_id):
            error = validate_storable_name(value)
            if error:
                raise ValueError(error)
        chain = _walk_continuation_chain(
            nuclei_record, self.time - 1, self.index - 1
        )
        self._touched = []
        self._noop = True
        self._mark_rollback_ready()
        for t0, j0 in chain:
            nuc = nuclei_record[t0][j0]
            self._touched.append(
                (t0 + 1, j0 + 1, nuc.identity, nuc.assigned_id)
            )
            if (
                nuc.identity != self.identity
                or nuc.assigned_id != self.assigned_id
            ):
                self._noop = False
            nuc.identity = self.identity
            nuc.assigned_id = self.assigned_id

    def undo(self, nuclei_record: NucleiRecord) -> None:
        _restore_name_state(nuclei_record, self._touched)
        self._touched = []

    @property
    def description(self) -> str:
        return (
            f"Set cell name state at t={self.time} idx={self.index}: "
            f"identity='{self.identity}', assigned_id='{self.assigned_id}'"
        )

    @property
    def is_noop(self) -> bool:
        return self._noop


@dataclass
class ClearNameOverride(EditCommand):
    """Clear the manual name override across one continuation component."""

    time: int
    index: int

    _touched: list[tuple[int, int, str, str]] = field(default_factory=list)
    _noop: bool = field(default=False, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        chain = _walk_continuation_chain(
            nuclei_record, self.time - 1, self.index - 1
        )
        self._touched = []
        self._noop = True
        self._mark_rollback_ready()
        for t0, j0 in chain:
            nuc = nuclei_record[t0][j0]
            self._touched.append(
                (t0 + 1, j0 + 1, nuc.identity, nuc.assigned_id)
            )
            if nuc.assigned_id:
                self._noop = False
            nuc.assigned_id = ""

    def undo(self, nuclei_record: NucleiRecord) -> None:
        _restore_name_state(nuclei_record, self._touched)
        self._touched = []

    @property
    def description(self) -> str:
        return f"Use automatic name for cell at t={self.time} idx={self.index}"

    @property
    def is_noop(self) -> bool:
        return self._noop


@dataclass
class LockCellName(EditCommand):
    """Lock the selected cell's current effective name as a manual override.

    This command is deliberately separate from :class:`RenameCell`.  Accepting
    an unchanged, pre-filled Rename dialog remains a true no-op, while the
    explicit *Lock Current Name* UI action records the current automatic name
    in ``assigned_id`` across the cell's reciprocal continuation chain.

    Both naming fields are normalized to the locked name, matching the manual
    ownership state produced by Rename.  Undo restores the exact per-nucleus
    automatic and forced values that existed before the lock.
    """

    time: int
    index: int

    _locked_name: str = field(default="", init=False)
    _touched: list[tuple[int, int, str, str]] = field(default_factory=list)
    _noop: bool = field(default=False, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        anchor = _get_nucleus(nuclei_record, self.time, self.index)
        if not anchor.is_alive:
            raise ValueError("Cannot lock the name of a dead nucleus")

        # Keep the name captured by the first execution stable across Redo.
        if not self._locked_name:
            error = validate_storable_name(anchor.effective_name, allow_empty=False)
            if error:
                raise ValueError(error)
            self._locked_name = anchor.effective_name.strip()

        chain = _walk_continuation_chain(
            nuclei_record, self.time - 1, self.index - 1
        )
        if not chain:
            raise ValueError("Selected nucleus has no valid continuation to lock")

        self._touched = []
        self._noop = True
        self._mark_rollback_ready()
        for t0, j0 in chain:
            nuc = nuclei_record[t0][j0]
            self._touched.append(
                (t0 + 1, j0 + 1, nuc.identity, nuc.assigned_id)
            )
            if (
                nuc.identity != self._locked_name
                or nuc.assigned_id != self._locked_name
            ):
                self._noop = False
            nuc.identity = self._locked_name
            nuc.assigned_id = self._locked_name

        logger.info(
            "Locked cell name '%s': %d nuclei in continuation chain "
            "(t=%d idx=%d clicked)",
            self._locked_name,
            len(self._touched),
            self.time,
            self.index,
        )

    def undo(self, nuclei_record: NucleiRecord) -> None:
        _restore_name_state(nuclei_record, self._touched)
        logger.info(
            "Undid name lock on %d nuclei (clicked at t=%d idx=%d)",
            len(self._touched),
            self.time,
            self.index,
        )
        self._touched = []

    @property
    def description(self) -> str:
        name = self._locked_name or "current name"
        return (
            f"Lock cell name at t={self.time} idx={self.index} "
            f"as '{name}'"
        )

    @property
    def is_noop(self) -> bool:
        return self._noop


@dataclass
class RenameCell(EditCommand):
    """Force a name on a cell (sets assigned_id across continuation chain).

    A cell in AceTree spans from birth (division or first appearance) to its
    next division or disappearance — this is the *continuation chain* of a
    tracked nucleus.  Renaming a cell therefore means writing the forced
    name to every nucleus in that chain, not just the clicked one.  This
    keeps `assigned_id` consistent across the cell's lifetime regardless
    of which timepoint the user was viewing when they renamed it.

    The assigned_id survives the naming pipeline — it's a manual override.
    Daughters of the renamed cell will still be named automatically (they
    live in a separate continuation chain after division).
    """

    time: int  # 1-based timepoint where the rename was triggered
    index: int  # 1-based nucleus index at that timepoint
    new_name: str

    # Saved state for undo: (t_1based, index_1based, old_identity, old_assigned_id)
    _touched: list[tuple[int, int, str, str]] = field(default_factory=list)
    _noop: bool = field(default=False, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        error = validate_storable_name(self.new_name, allow_empty=False)
        if error:
            raise ValueError(error)
        self.new_name = self.new_name.strip()

        anchor = _get_nucleus(nuclei_record, self.time, self.index)
        self._touched = []
        if anchor.effective_name == self.new_name:
            self._noop = True
            logger.info(
                "Rename cell is a no-op at t=%d idx=%d: already '%s'",
                self.time, self.index, self.new_name,
            )
            return

        self._noop = False
        chain = _walk_continuation_chain(nuclei_record, self.time - 1, self.index - 1)
        self._mark_rollback_ready()
        for t0, j0 in chain:
            nuc = nuclei_record[t0][j0]
            self._touched.append((t0 + 1, j0 + 1, nuc.identity, nuc.assigned_id))
            nuc.assigned_id = self.new_name
            nuc.identity = self.new_name
        logger.info("Renamed cell to '%s': %d nuclei in continuation chain (t=%d idx=%d clicked)",
                     self.new_name, len(self._touched), self.time, self.index)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        for t_1based, idx_1based, old_identity, old_assigned_id in self._touched:
            nuc = _get_nucleus(nuclei_record, t_1based, idx_1based)
            nuc.identity = old_identity
            nuc.assigned_id = old_assigned_id
        logger.info("Undid rename of %d nuclei (clicked at t=%d idx=%d)",
                     len(self._touched), self.time, self.index)
        self._touched = []

    @property
    def description(self) -> str:
        return f"Rename cell at t={self.time} idx={self.index} to '{self.new_name}'"

    @property
    def is_noop(self) -> bool:
        return self._noop


@dataclass
class SwapCellNames(EditCommand):
    """Atomically swap the forced names of two cells.

    Each cell is identified by a (time, index) anchor — any nucleus in the
    cell's continuation chain works.  The swap writes the *current*
    effective name of cell B onto every nucleus in A's chain, and vice
    versa.  If either cell currently has no effective name, its chain is
    cleared (assigned_id + identity set to '') during the swap.

    This is the UI's answer to the name-collision case: rather than
    rejecting a rename to an already-used name and leaving the user
    stuck, they can opt to swap the two cells' names atomically.
    """

    time_a: int   # 1-based timepoint anchor for cell A
    index_a: int  # 1-based nucleus index at time_a
    time_b: int   # 1-based timepoint anchor for cell B
    index_b: int  # 1-based nucleus index at time_b

    # Saved state for undo
    _touched_a: list[tuple[int, int, str, str]] = field(default_factory=list)
    _touched_b: list[tuple[int, int, str, str]] = field(default_factory=list)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        # Read current effective names BEFORE mutating anything
        nuc_a = _get_nucleus(nuclei_record, self.time_a, self.index_a)
        nuc_b = _get_nucleus(nuclei_record, self.time_b, self.index_b)
        name_a = nuc_a.effective_name
        name_b = nuc_b.effective_name

        chain_a = _walk_continuation_chain(nuclei_record, self.time_a - 1, self.index_a - 1)
        chain_b = _walk_continuation_chain(nuclei_record, self.time_b - 1, self.index_b - 1)

        self._touched_a = []
        self._touched_b = []

        self._mark_rollback_ready()
        for t0, j0 in chain_a:
            nuc = nuclei_record[t0][j0]
            self._touched_a.append((t0 + 1, j0 + 1, nuc.identity, nuc.assigned_id))
            nuc.assigned_id = name_b
            nuc.identity = name_b

        for t0, j0 in chain_b:
            nuc = nuclei_record[t0][j0]
            self._touched_b.append((t0 + 1, j0 + 1, nuc.identity, nuc.assigned_id))
            nuc.assigned_id = name_a
            nuc.identity = name_a

        logger.info("Swapped cell names: '%s' <-> '%s' (%d + %d nuclei)",
                     name_a, name_b, len(self._touched_a), len(self._touched_b))

    def undo(self, nuclei_record: NucleiRecord) -> None:
        for t_1based, idx_1based, old_identity, old_assigned_id in self._touched_a:
            nuc = _get_nucleus(nuclei_record, t_1based, idx_1based)
            nuc.identity = old_identity
            nuc.assigned_id = old_assigned_id
        for t_1based, idx_1based, old_identity, old_assigned_id in self._touched_b:
            nuc = _get_nucleus(nuclei_record, t_1based, idx_1based)
            nuc.identity = old_identity
            nuc.assigned_id = old_assigned_id
        logger.info("Undid swap of %d + %d nuclei",
                     len(self._touched_a), len(self._touched_b))
        self._touched_a = []
        self._touched_b = []

    @property
    def description(self) -> str:
        return (f"Swap cell names: (t={self.time_a}, idx={self.index_a}) "
                f"<-> (t={self.time_b}, idx={self.index_b})")


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
    _noop: bool = field(default=False, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        from .validators import validate_relink

        errors = validate_relink(
            nuclei_record, self.time, self.index, self.new_predecessor
        )
        if errors:
            raise ValueError("; ".join(errors))
        nuc = _get_nucleus(nuclei_record, self.time, self.index)
        self._old_predecessor = nuc.predecessor
        self._old_parent_index = self._new_parent_index = 0
        self._noop = self.new_predecessor == nuc.predecessor
        if self._noop:
            return

        # Disconnect from old parent's successor list
        if nuc.predecessor != NILLI and self.time >= 2:
            old_parent = _get_nucleus_safe(nuclei_record, self.time - 1, nuc.predecessor)
            if old_parent is not None:
                self._old_parent_time = self.time - 1
                self._old_parent_index = nuc.predecessor
                self._old_parent_succ1 = old_parent.successor1
                self._old_parent_succ2 = old_parent.successor2
                self._mark_rollback_ready()
                _remove_successor(old_parent, self.index)

        # Set new predecessor
        self._mark_rollback_ready()
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
        if self._noop:
            return
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

    @property
    def is_noop(self) -> bool:
        return self._noop


@dataclass
class KillCell(EditCommand):
    """Kill one anchored cell continuation across a time range.

    The legacy constructor (name, start time, optional end time) remains
    valid.  The anchor is the first live nucleus with that *effective* name
    at ``start_time``; callers that know the selected nucleus can provide
    ``anchor_index`` to make the choice explicit.  Only that reciprocal
    continuation component is affected, even if duplicate names exist.
    """

    cell_name: str
    start_time: int  # 1-based
    end_time: int | None = None  # 1-based, None = all remaining
    anchor_index: int | None = None  # Optional exact index at start_time

    # Saved state for undo (list of (time, index, old_status, old_identity, old_assigned_id))
    _killed: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._killed is None:
            self._killed = []

    def execute(self, nuclei_record: NucleiRecord) -> None:
        self._killed = []
        end = self.end_time if self.end_time is not None else len(nuclei_record)
        end = min(end, len(nuclei_record))

        t0 = self.start_time - 1
        if t0 < 0 or t0 >= len(nuclei_record):
            return

        requested_name = self.cell_name.strip()
        anchor_j0: int | None = None
        if self.anchor_index is not None:
            candidate = _get_nucleus_safe(
                nuclei_record, self.start_time, self.anchor_index
            )
            if (
                candidate is not None
                and candidate.is_alive
                and candidate.effective_name == requested_name
            ):
                anchor_j0 = self.anchor_index - 1
        else:
            for j0, candidate in enumerate(nuclei_record[t0]):
                if candidate.is_alive and candidate.effective_name == requested_name:
                    anchor_j0 = j0
                    break

        if anchor_j0 is None:
            logger.info(
                "No live anchor for cell '%s' at t=%d",
                requested_name, self.start_time,
            )
            return

        chain = _walk_continuation_chain(nuclei_record, t0, anchor_j0)
        self._mark_rollback_ready()
        for chain_t0, chain_j0 in chain:
            t_1based = chain_t0 + 1
            if t_1based < self.start_time or t_1based > end:
                continue
            nuc = nuclei_record[chain_t0][chain_j0]
            self._killed.append((
                t_1based, chain_j0 + 1,
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

        error = validate_storable_name(self.identity)
        if error:
            raise ValueError(error)
        self.identity = self.identity.strip()

        self._mark_rollback_ready()
        nuc.status = 1
        if self.identity:
            nuc.identity = self.identity
            nuc.assigned_id = self.identity
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

    _command: CompositeCommand | None = field(default=None, init=False)

    def execute(self, nuclei_record: NucleiRecord) -> None:
        from .validators import validate_relink_interpolation

        errors = validate_relink_interpolation(
            nuclei_record, self.start_time, self.start_index,
            self.end_time, self.end_index,
        )
        if errors:
            raise ValueError("; ".join(errors))
        start_nuc = _get_nucleus(nuclei_record, self.start_time, self.start_index)
        end_nuc = _get_nucleus(nuclei_record, self.end_time, self.end_index)
        commands: list[EditCommand] = []
        previous_index = self.start_index
        num_steps = self.end_time - self.start_time
        for step in range(1, num_steps):
            time = self.start_time + step
            fraction = step / num_steps
            commands.append(AddNucleus(
                time=time,
                x=round(start_nuc.x + (end_nuc.x - start_nuc.x) * fraction),
                y=round(start_nuc.y + (end_nuc.y - start_nuc.y) * fraction),
                z=start_nuc.z + (end_nuc.z - start_nuc.z) * fraction,
                size=round(start_nuc.size + (end_nuc.size - start_nuc.size) * fraction),
                predecessor=previous_index,
                identity=start_nuc.identity,
            ))
            previous_index = len(nuclei_record[time - 1]) + 1
        commands.append(RelinkNucleus(
            time=self.end_time, index=self.end_index,
            new_predecessor=previous_index,
        ))
        # Reuse the same reciprocal-link and rollback rules as direct edits.
        # In particular, relinking the endpoint disconnects its previous parent.
        self._command = CompositeCommand(commands, label=self.description)
        self._mark_rollback_ready()
        self._command.execute(nuclei_record)

    def undo(self, nuclei_record: NucleiRecord) -> None:
        if self._command is not None:
            self._command.undo(nuclei_record)

    @property
    def is_noop(self) -> bool:
        return self._command is not None and self._command.is_noop

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
    if child_index in (parent.successor1, parent.successor2):
        return
    if parent.successor1 == NILLI:
        parent.successor1 = child_index
    elif parent.successor2 == NILLI:
        parent.successor2 = child_index
    else:
        logger.warning("Nucleus at idx=%d already has 2 successors; cannot add %d",
                        parent.index, child_index)


def _restore_name_state(
    nuclei_record: NucleiRecord,
    touched: list[tuple[int, int, str, str]],
) -> None:
    """Restore identity and assigned_id snapshots captured by name commands."""
    for time, index, identity, assigned_id in touched:
        nuc = _get_nucleus(nuclei_record, time, index)
        nuc.identity = identity
        nuc.assigned_id = assigned_id


def _walk_continuation_chain(
    nuclei_record: NucleiRecord,
    t0: int,
    j0: int,
) -> list[tuple[int, int]]:
    """Walk the continuation chain of a nucleus and return every (t_0based, j_0based) in it.

    A *continuation chain* is the sequence of nuclei that represent a single
    Cell — from birth (the division event that created it, or its first
    appearance) to its own next division or disappearance.  Concretely:

      - Forward: follow `successor1` as long as the current nucleus has
        exactly one successor (`successor2 == NILLI`).  Stop at divisions
        (both daughters belong to new cells) and at dead ends.
      - Backward: follow `predecessor` as long as the predecessor itself
        has exactly one successor (we're continuing a cell, not coming
        out of a division).  Stop when the predecessor has two successors
        — that predecessor is the parent cell, not part of our chain.

    The returned list is sorted by timepoint.  Invalid or dead anchors return
    an empty list.  Every followed edge must be reciprocal.
    """
    if t0 < 0 or t0 >= len(nuclei_record):
        return []
    if j0 < 0 or j0 >= len(nuclei_record[t0]):
        return []
    if not nuclei_record[t0][j0].is_alive:
        return []

    chain: list[tuple[int, int]] = [(t0, j0)]

    # Walk forward from (t0, j0)
    t, j = t0, j0
    while True:
        nuc = nuclei_record[t][j]
        # Stop if this nucleus divides (has two successors) or has no successor
        if nuc.successor1 == NILLI or nuc.successor2 != NILLI:
            break
        next_j_1based = nuc.successor1
        next_t = t + 1
        if next_t >= len(nuclei_record):
            break
        next_j = next_j_1based - 1
        if next_j < 0 or next_j >= len(nuclei_record[next_t]):
            break
        next_nuc = nuclei_record[next_t][next_j]
        if not next_nuc.is_alive or next_nuc.predecessor != j + 1:
            break
        chain.append((next_t, next_j))
        t, j = next_t, next_j

    # Walk backward from (t0, j0)
    t, j = t0, j0
    while True:
        nuc = nuclei_record[t][j]
        if nuc.predecessor == NILLI:
            break
        prev_t = t - 1
        if prev_t < 0:
            break
        prev_j = nuc.predecessor - 1
        if prev_j < 0 or prev_j >= len(nuclei_record[prev_t]):
            break
        prev_nuc = nuclei_record[prev_t][prev_j]
        # Stop if the predecessor is a dividing cell — it's the parent,
        # not part of this cell's chain.
        if (
            not prev_nuc.is_alive
            or prev_nuc.successor2 != NILLI
            or prev_nuc.successor1 != j + 1
        ):
            break
        chain.append((prev_t, prev_j))
        t, j = prev_t, prev_j

    chain.sort()
    return chain
