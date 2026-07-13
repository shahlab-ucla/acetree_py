"""Edit history — undo/redo stack for edit commands.

Manages a stack of executed commands, supporting unlimited undo and redo.
This is a major improvement over Java AceTree which had NO undo support.

Usage:
    history = EditHistory(nuclei_record)
    history.do(AddNucleus(time=5, x=100, y=200, z=10.0))
    history.undo()   # reverses the add
    history.redo()   # re-applies the add
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .commands import EditCommand, NucleiRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _HistoryEntry:
    """A command edge between two unique document states."""

    command: EditCommand
    before_state: int
    after_state: int


class EditHistory:
    """Manages undo/redo stacks for edit operations.

    All edits go through this class to ensure consistent undo/redo behavior.
    After each edit, the optional on_edit callback is called (e.g. to rebuild
    the lineage tree or refresh the GUI).

    Attributes:
        nuclei_record: The mutable nuclei data being edited.
        modified: True if any edits have been made since last save/reset.
    """

    def __init__(
        self,
        nuclei_record: NucleiRecord,
        on_edit: Callable[[], None] | None = None,
        max_history: int = 1000,
    ) -> None:
        """Initialize the edit history.

        Args:
            nuclei_record: The mutable nuclei data to edit.
            on_edit: Optional callback invoked after each do/undo/redo.
            max_history: Maximum number of commands to keep in the undo stack.
        """
        self.nuclei_record = nuclei_record
        self.on_edit = on_edit
        self.max_history = max_history
        self._undo_stack: list[_HistoryEntry] = []
        self._redo_stack: list[_HistoryEntry] = []
        self._current_state = 0
        self._saved_state = 0
        self._next_state = 1
        self.modified: bool = False
        self.last_command: EditCommand | None = None

    def do(self, command: EditCommand) -> None:
        """Execute a command and push it onto the undo stack.

        Clears the redo stack (future can no longer be re-done after a new edit).

        Args:
            command: The edit command to execute.
        """
        command.execute(self.nuclei_record)
        if command.is_noop:
            logger.info("No change: %s", command.description)
            return

        entry = _HistoryEntry(
            command=command,
            before_state=self._current_state,
            after_state=self._next_state,
        )
        self._next_state += 1
        self._current_state = entry.after_state
        self._undo_stack.append(entry)
        self._redo_stack.clear()
        self._sync_modified()

        # Enforce max history
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)

        logger.info("Executed: %s", command.description)
        self.last_command = command
        if self.on_edit:
            self.on_edit()

    def undo(self) -> EditCommand | None:
        """Undo the most recent command.

        Returns:
            The command that was undone, or None if nothing to undo.
        """
        if not self._undo_stack:
            logger.info("Nothing to undo")
            return None

        entry = self._undo_stack.pop()
        command = entry.command
        command.undo(self.nuclei_record)
        self._redo_stack.append(entry)
        self._current_state = entry.before_state
        self._sync_modified()

        logger.info("Undid: %s", command.description)
        self.last_command = command
        if self.on_edit:
            self.on_edit()
        return command

    def redo(self) -> EditCommand | None:
        """Redo the most recently undone command.

        Returns:
            The command that was re-done, or None if nothing to redo.
        """
        if not self._redo_stack:
            logger.info("Nothing to redo")
            return None

        entry = self._redo_stack.pop()
        command = entry.command
        command.execute(self.nuclei_record)
        self._undo_stack.append(entry)
        self._current_state = entry.after_state
        self._sync_modified()

        logger.info("Redid: %s", command.description)
        self.last_command = command
        if self.on_edit:
            self.on_edit()
        return command

    @property
    def can_undo(self) -> bool:
        """True if there are commands that can be undone."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """True if there are commands that can be re-done."""
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> str:
        """Description of the next command to undo, or empty string."""
        if self._undo_stack:
            return self._undo_stack[-1].command.description
        return ""

    @property
    def redo_description(self) -> str:
        """Description of the next command to redo, or empty string."""
        if self._redo_stack:
            return self._redo_stack[-1].command.description
        return ""

    @property
    def num_undoable(self) -> int:
        """Number of commands that can be undone."""
        return len(self._undo_stack)

    @property
    def num_redoable(self) -> int:
        """Number of commands that can be re-done."""
        return len(self._redo_stack)

    def clear(self) -> None:
        """Clear all history (undo and redo stacks)."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._sync_modified()
        logger.info("Edit history cleared")

    def mark_saved(self) -> None:
        """Mark the current state as saved (resets modified flag)."""
        self._saved_state = self._current_state
        self._sync_modified()

    def history_log(self) -> list[str]:
        """Get a list of all executed command descriptions (oldest first)."""
        return [entry.command.description for entry in self._undo_stack]

    def _sync_modified(self) -> None:
        """Keep the compatibility flag aligned with the current savepoint."""
        self.modified = self._current_state != self._saved_state
