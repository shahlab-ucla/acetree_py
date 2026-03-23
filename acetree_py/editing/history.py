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
from typing import Callable

from .commands import EditCommand, NucleiRecord

logger = logging.getLogger(__name__)


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
        self._undo_stack: list[EditCommand] = []
        self._redo_stack: list[EditCommand] = []
        self.modified: bool = False

    def do(self, command: EditCommand) -> None:
        """Execute a command and push it onto the undo stack.

        Clears the redo stack (future can no longer be re-done after a new edit).

        Args:
            command: The edit command to execute.
        """
        command.execute(self.nuclei_record)
        self._undo_stack.append(command)
        self._redo_stack.clear()
        self.modified = True

        # Enforce max history
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)

        logger.info("Executed: %s", command.description)
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

        command = self._undo_stack.pop()
        command.undo(self.nuclei_record)
        self._redo_stack.append(command)

        logger.info("Undid: %s", command.description)
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

        command = self._redo_stack.pop()
        command.execute(self.nuclei_record)
        self._undo_stack.append(command)
        self.modified = True

        logger.info("Redid: %s", command.description)
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
            return self._undo_stack[-1].description
        return ""

    @property
    def redo_description(self) -> str:
        """Description of the next command to redo, or empty string."""
        if self._redo_stack:
            return self._redo_stack[-1].description
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
        logger.info("Edit history cleared")

    def mark_saved(self) -> None:
        """Mark the current state as saved (resets modified flag)."""
        self.modified = False

    def history_log(self) -> list[str]:
        """Get a list of all executed command descriptions (oldest first)."""
        return [cmd.description for cmd in self._undo_stack]
