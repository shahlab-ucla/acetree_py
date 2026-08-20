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
from dataclasses import dataclass, replace
from typing import Callable

from .commands import EditCommand, NucleiRecord

logger = logging.getLogger(__name__)

_NameSnapshot = dict[tuple[int, int], tuple[str, str]]
_NameChange = tuple[
    int,
    int,
    str | None,
    str | None,
    str | None,
    str | None,
]


class PostCommitCallbackError(RuntimeError):
    """The edit committed, but the post-commit callback failed.

    This exception deliberately distinguishes an observer/UI refresh failure
    from a command execution failure.  By the time it is raised, the data and
    undo/redo stacks already represent the requested operation.
    """

    def __init__(self, operation: str, command: EditCommand) -> None:
        self.operation = operation
        self.command = command
        super().__init__(
            f"{operation.capitalize()} committed, but the post-commit "
            f"callback failed: {command.description}"
        )


@dataclass(frozen=True)
class _HistoryEntry:
    """A command edge between two unique document states."""

    command: EditCommand
    before_state: int
    after_state: int
    # Automatic naming is a post-command structural side effect.  Keep its
    # sparse before/after boundary with the command so Undo/Redo restores the
    # exact persisted document rather than depending on the immediately prior
    # display names used by a partial-movie naming fallback.
    name_changes: tuple[_NameChange, ...] = ()


@dataclass(frozen=True)
class _PendingNameCapture:
    """Context needed to finish a structural post-commit name boundary."""

    operation: str
    command: EditCommand
    before_state: int
    after_state: int
    operation_start: _NameSnapshot
    callback_start: _NameSnapshot


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
        # Monotonic event counter for open analysis/review sessions.  Unlike
        # ``revision``, this does not return to an older value after Undo, so a
        # draft that observed any intervening edit can remain safely stale.
        self._change_counter = 0
        self.modified: bool = False
        self.last_command: EditCommand | None = None
        self._pending_name_capture: _PendingNameCapture | None = None

    def do(self, command: EditCommand) -> None:
        """Execute a command and push it onto the undo stack.

        Clears the redo stack (future can no longer be re-done after a new edit).

        Args:
            command: The edit command to execute.
        """
        self._finalize_pending_name_capture()
        names_before = (
            _snapshot_name_state(self.nuclei_record)
            if command.structural and self.on_edit is not None
            else None
        )
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
        self._change_counter += 1
        self._undo_stack.append(entry)
        self._redo_stack.clear()
        self._sync_modified()

        # Enforce max history
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)

        logger.info("Executed: %s", command.description)
        self.last_command = command
        self._notify_with_name_capture("do", entry, names_before)

    def undo(self) -> EditCommand | None:
        """Undo the most recent command.

        Returns:
            The command that was undone, or None if nothing to undo.
        """
        if not self._undo_stack:
            logger.info("Nothing to undo")
            return None

        self._finalize_pending_name_capture()
        names_before_operation = (
            _snapshot_name_state(self.nuclei_record)
            if self._undo_stack[-1].command.structural and self.on_edit is not None
            else None
        )
        entry = self._undo_stack.pop()
        command = entry.command
        command.undo(self.nuclei_record)
        _restore_name_boundary(
            self.nuclei_record,
            entry.name_changes,
            use_after=False,
        )
        self._redo_stack.append(entry)
        self._current_state = entry.before_state
        self._change_counter += 1
        self._sync_modified()

        logger.info("Undid: %s", command.description)
        self.last_command = command
        self._notify_with_name_capture("undo", entry, names_before_operation)
        return command

    def redo(self) -> EditCommand | None:
        """Redo the most recently undone command.

        Returns:
            The command that was re-done, or None if nothing to redo.
        """
        if not self._redo_stack:
            logger.info("Nothing to redo")
            return None

        self._finalize_pending_name_capture()
        names_before_operation = (
            _snapshot_name_state(self.nuclei_record)
            if self._redo_stack[-1].command.structural and self.on_edit is not None
            else None
        )
        entry = self._redo_stack.pop()
        command = entry.command
        command.execute(self.nuclei_record)
        _restore_name_boundary(
            self.nuclei_record,
            entry.name_changes,
            use_after=True,
        )
        self._undo_stack.append(entry)
        self._current_state = entry.after_state
        self._change_counter += 1
        self._sync_modified()

        logger.info("Redid: %s", command.description)
        self.last_command = command
        self._notify_with_name_capture("redo", entry, names_before_operation)
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
    def next_undo_command(self) -> EditCommand | None:
        """Command currently at the Undo boundary, or ``None`` when empty."""

        return self._undo_stack[-1].command if self._undo_stack else None

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

    @property
    def revision(self) -> int:
        """Opaque token identifying the document state currently displayed.

        Long-running analysis workflows capture this value before reading the
        nuclei record and compare it again before applying their proposal.  A
        mismatch means that the user edited (or undid/redid) the dataset while
        analysis was running, so the preview must be recomputed instead of
        being committed against stale nucleus indices.

        The token is deliberately opaque: callers may compare it for equality
        but should not assume that it is monotonic across undo/redo.
        """
        return self._current_state

    @property
    def change_counter(self) -> int:
        """Monotonic count of successful do, undo, and redo operations.

        Long-lived previews use this in addition to :attr:`revision`: even if
        an Undo restores the same document-state token, the user has crossed
        an edit boundary and the visible analysis must be deliberately rerun.
        """

        return self._change_counter

    def clear(self) -> None:
        """Clear all history (undo and redo stacks)."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._pending_name_capture = None
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

    def _notify_post_commit(self, operation: str, command: EditCommand) -> None:
        """Run ``on_edit`` while preserving the command's committed status."""
        if not self.on_edit:
            return
        try:
            self.on_edit()
        except Exception as error:
            raise PostCommitCallbackError(operation, command) from error

    def retry_post_commit(self, error: PostCommitCallbackError) -> None:
        """Retry only a failed callback and finish its exact name boundary.

        The command is already committed.  Restoring the dense state from
        immediately before the first callback prevents a partial naming pass
        from contaminating the retry.  The final sparse boundary is then
        refreshed from the successful retry result.
        """
        pending = self._pending_name_capture
        if pending is None:
            # Non-structural callbacks do not run automatic naming and avoid
            # the full-record snapshot cost.  Preserve their established safe
            # retry behavior without replaying the committed command.
            if self.last_command is error.command:
                self._notify_post_commit(error.operation, error.command)
                return
            raise RuntimeError("No matching post-commit callback is pending")
        if (
            pending.operation != error.operation
            or pending.command is not error.command
        ):
            raise RuntimeError("No matching post-commit callback is pending")

        _restore_name_snapshot(self.nuclei_record, pending.callback_start)
        try:
            self._notify_post_commit(pending.operation, pending.command)
        except PostCommitCallbackError:
            # Leave committed data at the clean pre-callback boundary.  A
            # later retry can start from the same deterministic state.
            _restore_name_snapshot(self.nuclei_record, pending.callback_start)
            self._finalize_pending_name_capture(keep_pending=True)
            raise
        else:
            self._finalize_pending_name_capture()

    def _notify_with_name_capture(
        self,
        operation: str,
        entry: _HistoryEntry,
        operation_start: _NameSnapshot | None,
    ) -> None:
        """Run the callback while tracking its automatic naming side effects."""
        if operation_start is None:
            self._notify_post_commit(operation, entry.command)
            return

        callback_start = _snapshot_name_state(self.nuclei_record)
        self._pending_name_capture = _PendingNameCapture(
            operation=operation,
            command=entry.command,
            before_state=entry.before_state,
            after_state=entry.after_state,
            operation_start=operation_start,
            callback_start=callback_start,
        )
        try:
            self._notify_post_commit(operation, entry.command)
        except PostCommitCallbackError:
            _restore_name_snapshot(self.nuclei_record, callback_start)
            self._finalize_pending_name_capture(keep_pending=True)
            raise
        else:
            self._finalize_pending_name_capture()

    def _finalize_pending_name_capture(self, *, keep_pending: bool = False) -> None:
        """Publish the current sparse before/after boundary to its entry."""
        pending = self._pending_name_capture
        if pending is None:
            return

        current = _snapshot_name_state(self.nuclei_record)
        if pending.operation == "undo":
            changes = _name_state_changes(current, pending.operation_start)
        else:
            changes = _name_state_changes(pending.operation_start, current)

        for stack in (self._undo_stack, self._redo_stack):
            for position, candidate in enumerate(stack):
                if (
                    candidate.command is pending.command
                    and candidate.before_state == pending.before_state
                    and candidate.after_state == pending.after_state
                ):
                    stack[position] = replace(candidate, name_changes=changes)
                    break

        if not keep_pending:
            self._pending_name_capture = None


def _snapshot_name_state(
    nuclei_record: NucleiRecord,
) -> _NameSnapshot:
    """Capture names transiently while a structural callback runs."""
    return {
        (time, index): (nucleus.identity, nucleus.assigned_id)
        for time, nuclei in enumerate(nuclei_record)
        for index, nucleus in enumerate(nuclei)
    }


def _name_state_changes(
    before: _NameSnapshot,
    after: _NameSnapshot,
) -> tuple[_NameChange, ...]:
    """Return a compact boundary, including rows added by the command."""
    changes: list[_NameChange] = []
    for time, index in sorted(before.keys() | after.keys()):
        old_value = before.get((time, index))
        new_value = after.get((time, index))
        if old_value == new_value:
            continue
        old_identity, old_assigned_id = old_value or (None, None)
        new_identity, new_assigned_id = new_value or (None, None)
        changes.append((
            time,
            index,
            old_identity,
            old_assigned_id,
            new_identity,
            new_assigned_id,
        ))
    return tuple(changes)


def _restore_name_boundary(
    nuclei_record: NucleiRecord,
    changes: tuple[_NameChange, ...],
    *,
    use_after: bool,
) -> None:
    """Restore names before rerunning the post-commit naming callback."""
    for (
        time,
        index,
        old_identity,
        old_assigned_id,
        new_identity,
        new_assigned_id,
    ) in changes:
        if not (0 <= time < len(nuclei_record)):
            continue
        if not (0 <= index < len(nuclei_record[time])):
            continue
        nucleus = nuclei_record[time][index]
        if use_after:
            if new_identity is not None and new_assigned_id is not None:
                nucleus.identity = new_identity
                nucleus.assigned_id = new_assigned_id
        else:
            if old_identity is not None and old_assigned_id is not None:
                nucleus.identity = old_identity
                nucleus.assigned_id = old_assigned_id


def _restore_name_snapshot(
    nuclei_record: NucleiRecord,
    snapshot: _NameSnapshot,
) -> None:
    """Restore every still-present row represented by a dense snapshot."""
    for (time, index), (identity, assigned_id) in snapshot.items():
        if not (0 <= time < len(nuclei_record)):
            continue
        if not (0 <= index < len(nuclei_record[time])):
            continue
        nucleus = nuclei_record[time][index]
        nucleus.identity = identity
        nucleus.assigned_id = assigned_id
