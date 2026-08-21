"""Best-effort atomic installation of staged dataset artifacts.

Individual files can be replaced atomically, but a dataset save spans several
files.  :class:`DatasetTransaction` prepares a rollback sibling for every
destination, installs already-complete staged files in order, and restores the
previous generation if any installation fails.  Callers put their commit
marker (the XML config in AceTree) last.

Staging remains the responsibility of each format-specific writer.  Requiring
staged files to live beside their destinations keeps every ``os.replace`` on
one filesystem and prevents a partially-written payload from becoming public.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    """A complete private file waiting to replace *destination*."""

    staged_path: Path
    destination: Path
    precondition: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "staged_path", Path(self.staged_path))
        object.__setattr__(self, "destination", Path(self.destination))
        if self.precondition is not None and not callable(self.precondition):
            raise TypeError("artifact precondition must be callable")


@dataclass(slots=True)
class _CommitState:
    artifact: StagedArtifact
    backup: Path | None = None
    destination_moved: bool = False
    installed: bool = False
    rollback_failed: bool = False


def _unused_rollback_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".rollback",
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _flush_staged(path: Path) -> None:
    """Flush a completed staged file before it participates in a commit."""

    # Windows' CRT rejects ``fsync`` for a read-only descriptor.  Staged
    # artifacts are private writable files, so open read/write without
    # changing their contents.
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DatasetTransaction:
    """Install a collection of staged sibling files with rollback.

    Artifacts are committed in insertion order.  The caller should therefore
    add payloads first and the dataset's discovery/commit marker last.
    ``commit`` may be called only once.
    """

    def __init__(self, artifacts: Iterable[StagedArtifact] = ()) -> None:
        self._artifacts = list(artifacts)
        self._committed = False

    def add(
        self,
        staged_path: str | Path,
        destination: str | Path,
        *,
        precondition: Callable[[], None] | None = None,
    ) -> None:
        if self._committed:
            raise RuntimeError("Cannot add an artifact after commit")
        self._artifacts.append(
            StagedArtifact(Path(staged_path), Path(destination), precondition)
        )

    @property
    def artifacts(self) -> tuple[StagedArtifact, ...]:
        return tuple(self._artifacts)

    def _validate(self) -> None:
        seen: set[Path] = set()
        for artifact in self._artifacts:
            staged = artifact.staged_path.resolve(strict=False)
            destination = artifact.destination.resolve(strict=False)
            if not staged.is_file():
                raise FileNotFoundError(f"Staged artifact is missing: {staged}")
            if staged.parent != destination.parent:
                raise ValueError(
                    "Staged artifacts must be private siblings of their "
                    f"destinations: {staged} -> {destination}"
                )
            if destination in seen:
                raise ValueError(f"Duplicate transaction destination: {destination}")
            seen.add(destination)

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("Dataset transaction has already been committed")
        self._validate()
        self._committed = True

        states = [_CommitState(artifact) for artifact in self._artifacts]
        try:
            for state in states:
                artifact = state.artifact
                destination = artifact.destination
                destination.parent.mkdir(parents=True, exist_ok=True)
                if artifact.precondition is not None:
                    artifact.precondition()
                _flush_staged(artifact.staged_path)
                if destination.exists():
                    state.backup = _unused_rollback_path(destination)
                    # Keep the live destination in place until the staged
                    # file is ready to replace it.  Besides reducing the
                    # missing-file window, a failed ``os.replace`` therefore
                    # leaves the prior generation public without requiring a
                    # second replacement merely to undo that failed step.
                    shutil.copy2(destination, state.backup)
                    _flush_staged(state.backup)
                    state.destination_moved = True
                os.replace(artifact.staged_path, destination)
                state.installed = True
        except BaseException as error:
            rollback_errors: list[BaseException] = []
            for state in reversed(states):
                try:
                    destination = state.artifact.destination
                    if state.installed:
                        destination.unlink(missing_ok=True)
                    if (
                        state.installed
                        and state.destination_moved
                        and state.backup is not None
                        and state.backup.exists()
                    ):
                        os.replace(state.backup, destination)
                except BaseException as rollback_error:  # pragma: no cover - rare I/O failure
                    state.rollback_failed = True
                    rollback_errors.append(rollback_error)
                    logger.exception(
                        "Could not restore dataset artifact %s",
                        state.artifact.destination,
                    )
            if rollback_errors:
                # A failed restore may leave the sibling backup as the only
                # recoverable copy of the user's prior data.  Never delete it
                # merely to make cleanup look tidy.
                self._discard_private_files(
                    states, preserve_failed_backups=True
                )
                retained = tuple(
                    str(state.backup)
                    for state in states
                    if state.rollback_failed
                    and state.backup is not None
                    and state.backup.exists()
                )
                recovery_detail = (
                    f"; recovery backup(s) retained at {', '.join(retained)}"
                    if retained
                    else ""
                )
                raise RuntimeError(
                    "Dataset save failed and one or more previous artifacts "
                    f"could not be restored{recovery_detail}"
                ) from rollback_errors[0]
            self._discard_private_files(states)
            raise error

        self._discard_private_files(states)

    @staticmethod
    def _discard_private_files(
        states: Iterable[_CommitState], *, preserve_failed_backups: bool = False
    ) -> None:
        for state in states:
            private_paths = [state.artifact.staged_path]
            if not (preserve_failed_backups and state.rollback_failed):
                private_paths.append(state.backup)
            for private_path in private_paths:
                if private_path is None:
                    continue
                try:
                    private_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove private transaction file %s",
                        private_path,
                        exc_info=True,
                    )


def commit_staged_artifacts(artifacts: Iterable[StagedArtifact]) -> None:
    """Convenience wrapper for a one-shot :class:`DatasetTransaction`."""

    DatasetTransaction(artifacts).commit()
