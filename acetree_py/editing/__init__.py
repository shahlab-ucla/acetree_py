"""Editing system for nuclei data — command pattern with undo/redo.

Provides a set of undoable edit commands that modify the nuclei_record,
plus an EditHistory class that manages undo/redo stacks.

All edit operations go through EditHistory.do(command) to ensure
consistent undo/redo behavior.
"""
