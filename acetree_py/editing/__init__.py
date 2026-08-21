"""Editing system for nuclei data — command pattern with undo/redo.

Provides a set of undoable edit commands that modify the nuclei_record,
plus an EditHistory class that manages undo/redo stacks.

All edit operations go through EditHistory.do(command) to ensure
consistent undo/redo behavior.
"""

from .commands import EditCommand, EditDomain, EditEffect, EditEffectDomain
from .history import EditHistory

__all__ = [
    "EditCommand",
    "EditDomain",
    "EditEffect",
    "EditEffectDomain",
    "EditHistory",
]
