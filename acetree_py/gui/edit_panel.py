"""Edit panel — dock widget for cell/nucleus editing operations.

Provides a toolbar with buttons for each edit operation, dialogs for
parameter input, undo/redo controls, and a history log display.

All operations delegate to the existing editing commands (Phase 5)
via the EditHistory on AceTreeApp. Pre-edit validation is performed
using validators.py before executing any command.

Ported from: org.rhwlab.nucedit.* (AddOneDialog, NucRelinkDialog,
KillCellsDialog, Lazarus, NucEditDialog, SetEndTimeDialog)

Key improvement over Java: every operation is undoable, and validation
errors are shown before execution (Java dialogs had ad-hoc error handling).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


class EditPanel(QWidget):  # type: ignore[misc]
    """Dock widget providing editing tools for nuclei and cells.

    Layout:
        [Undo] [Redo]                  -- undo/redo buttons
        ─────────────────────
        [Add]  [Remove]  [Move]        -- single-nucleus operations
        [Rename]  [Kill]  [Resurrect]  -- cell-level operations
        [Relink]                       -- linking (auto-interpolates gaps)
        ─────────────────────
        Status: last edit description
        ─────────────────────
        Edit History (scrollable list)
    """

    def __init__(self, app: AceTreeApp, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Edit Tools")
        title.setFont(QFont("Sans Serif", 12, QFont.Bold))
        layout.addWidget(title)

        # ── File operations ──
        file_group = QGroupBox("File")
        file_layout = QHBoxLayout(file_group)

        self._btn_save = QPushButton("Save")
        self._btn_save.setToolTip("Save nuclei to ZIP (Ctrl+S)")
        self._btn_save.clicked.connect(self._on_save)

        self._btn_save_as = QPushButton("Save As…")
        self._btn_save_as.setToolTip("Save nuclei to a new ZIP file (Ctrl+Shift+S)")
        self._btn_save_as.clicked.connect(self._on_save_as)

        file_layout.addWidget(self._btn_save)
        file_layout.addWidget(self._btn_save_as)
        layout.addWidget(file_group)

        # ── Undo/Redo ──
        undo_redo_row = QHBoxLayout()

        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setToolTip("Undo last edit (Ctrl+Z)")
        self._btn_undo.clicked.connect(self._on_undo)

        self._btn_redo = QPushButton("Redo")
        self._btn_redo.setToolTip("Redo last undone edit (Ctrl+Y)")
        self._btn_redo.clicked.connect(self._on_redo)

        undo_redo_row.addWidget(self._btn_undo)
        undo_redo_row.addWidget(self._btn_redo)
        layout.addLayout(undo_redo_row)

        # ── Single-nucleus operations ──
        nuc_group = QGroupBox("Nucleus Operations")
        nuc_layout = QHBoxLayout(nuc_group)

        self._btn_add = QPushButton("Add")
        self._btn_add.setToolTip(
            "Click-to-add mode: left-click in the viewer to place a nucleus.\n"
            "If a cell is selected, new nucleus inherits its identity.\n"
            "Press Esc or click Add again to exit."
        )
        self._btn_add.setCheckable(True)
        self._btn_add.clicked.connect(self._on_add_toggle)

        self._btn_remove = QPushButton("Remove")
        self._btn_remove.setToolTip("Remove (kill) the selected nucleus")
        self._btn_remove.clicked.connect(self._on_remove_nucleus)

        nuc_layout.addWidget(self._btn_add)
        nuc_layout.addWidget(self._btn_remove)
        layout.addWidget(nuc_group)

        # ── Move / nudge controls ──
        move_group = QGroupBox("Move / Resize")
        move_layout = QVBoxLayout(move_group)

        self._build_move_controls(move_layout)
        layout.addWidget(move_group)

        # ── Cell-level operations ──
        cell_group = QGroupBox("Cell Operations")
        cell_layout = QHBoxLayout(cell_group)

        self._btn_rename = QPushButton("Rename")
        self._btn_rename.setToolTip("Force a name on the selected cell")
        self._btn_rename.clicked.connect(self._on_rename_cell)

        self._btn_kill = QPushButton("Kill")
        self._btn_kill.setToolTip("Kill the selected cell (mark all nuclei dead)")
        self._btn_kill.clicked.connect(self._on_kill_cell)

        self._btn_resurrect = QPushButton("Resurrect")
        self._btn_resurrect.setToolTip("Resurrect a dead nucleus")
        self._btn_resurrect.clicked.connect(self._on_resurrect)

        cell_layout.addWidget(self._btn_rename)
        cell_layout.addWidget(self._btn_kill)
        cell_layout.addWidget(self._btn_resurrect)
        layout.addWidget(cell_group)

        # ── Link operations ──
        link_group = QGroupBox("Link Operations")
        link_layout = QHBoxLayout(link_group)

        self._btn_relink = QPushButton("Relink")
        self._btn_relink.setToolTip(
            "Link two cells: select source, click Relink, then click target.\n"
            "Auto-interpolates if there is a time gap."
        )
        self._btn_relink.clicked.connect(self._on_relink)

        self._btn_track = QPushButton("Track")
        self._btn_track.setToolTip(
            "Click-to-place mode: select a parent cell first, then click Track.\n"
            "Right-click in the viewer to place nuclei along the track.\n"
            "With no cell selected, places a single root nucleus.\n"
            "Press Esc or click Track again to exit."
        )
        self._btn_track.setCheckable(True)
        self._btn_track.clicked.connect(self._on_track)

        link_layout.addWidget(self._btn_relink)
        link_layout.addWidget(self._btn_track)
        layout.addWidget(link_group)

        # ── Status ──
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # ── History log ──
        history_label = QLabel("Edit History")
        history_label.setFont(QFont("Sans Serif", 10, QFont.Bold))
        layout.addWidget(history_label)

        self._history_list = QListWidget()
        self._history_list.setMaximumHeight(150)
        layout.addWidget(self._history_list)

        layout.addStretch()

    def _build_move_controls(self, parent_layout: QVBoxLayout) -> None:
        """Build D-pad style move/resize controls."""
        # ── XY row (left / right / up / down) ──
        xy_grid = QHBoxLayout()

        # Left
        col_left = QVBoxLayout()
        btn_l1 = QPushButton("\u2190 1")
        btn_l5 = QPushButton("\u2190 5")
        btn_l1.setFixedWidth(48)
        btn_l5.setFixedWidth(48)
        btn_l1.clicked.connect(lambda: self._nudge(dx=-1))
        btn_l5.clicked.connect(lambda: self._nudge(dx=-5))
        col_left.addWidget(btn_l5)
        col_left.addWidget(btn_l1)

        # Up/Down center column
        col_center = QVBoxLayout()
        btn_u1 = QPushButton("\u2191 1")
        btn_u5 = QPushButton("\u2191 5")
        btn_d1 = QPushButton("\u2193 1")
        btn_d5 = QPushButton("\u2193 5")
        for b in (btn_u1, btn_u5, btn_d1, btn_d5):
            b.setFixedWidth(48)
        btn_u1.clicked.connect(lambda: self._nudge(dy=-1))
        btn_u5.clicked.connect(lambda: self._nudge(dy=-5))
        btn_d1.clicked.connect(lambda: self._nudge(dy=1))
        btn_d5.clicked.connect(lambda: self._nudge(dy=5))
        col_center.addWidget(btn_u5)
        col_center.addWidget(btn_u1)
        col_center.addWidget(btn_d1)
        col_center.addWidget(btn_d5)

        # Right
        col_right = QVBoxLayout()
        btn_r1 = QPushButton("\u2192 1")
        btn_r5 = QPushButton("\u2192 5")
        btn_r1.setFixedWidth(48)
        btn_r5.setFixedWidth(48)
        btn_r1.clicked.connect(lambda: self._nudge(dx=1))
        btn_r5.clicked.connect(lambda: self._nudge(dx=5))
        col_right.addWidget(btn_r5)
        col_right.addWidget(btn_r1)

        xy_grid.addLayout(col_left)
        xy_grid.addLayout(col_center)
        xy_grid.addLayout(col_right)
        parent_layout.addLayout(xy_grid)

        # ── Z and Size row ──
        zs_row = QHBoxLayout()

        z_label = QLabel("Z:")
        btn_zd1 = QPushButton("-1")
        btn_zd5 = QPushButton("-5")
        btn_zu1 = QPushButton("+1")
        btn_zu5 = QPushButton("+5")
        for b in (btn_zd1, btn_zd5, btn_zu1, btn_zu5):
            b.setFixedWidth(36)
        btn_zd5.clicked.connect(lambda: self._nudge(dz=-5.0))
        btn_zd1.clicked.connect(lambda: self._nudge(dz=-1.0))
        btn_zu1.clicked.connect(lambda: self._nudge(dz=1.0))
        btn_zu5.clicked.connect(lambda: self._nudge(dz=5.0))

        zs_row.addWidget(z_label)
        zs_row.addWidget(btn_zd5)
        zs_row.addWidget(btn_zd1)
        zs_row.addWidget(btn_zu1)
        zs_row.addWidget(btn_zu5)
        parent_layout.addLayout(zs_row)

        size_row = QHBoxLayout()
        s_label = QLabel("Size:")
        btn_sd1 = QPushButton("-1")
        btn_sd5 = QPushButton("-5")
        btn_su1 = QPushButton("+1")
        btn_su5 = QPushButton("+5")
        for b in (btn_sd1, btn_sd5, btn_su1, btn_su5):
            b.setFixedWidth(36)
        btn_sd5.clicked.connect(lambda: self._nudge(dsize=-5))
        btn_sd1.clicked.connect(lambda: self._nudge(dsize=-1))
        btn_su1.clicked.connect(lambda: self._nudge(dsize=1))
        btn_su5.clicked.connect(lambda: self._nudge(dsize=5))

        size_row.addWidget(s_label)
        size_row.addWidget(btn_sd5)
        size_row.addWidget(btn_sd1)
        size_row.addWidget(btn_su1)
        size_row.addWidget(btn_su5)
        parent_layout.addLayout(size_row)

    def refresh(self) -> None:
        """Update button states and history display."""
        history = self.app.edit_history

        # Update undo/redo button states
        self._btn_undo.setEnabled(history.can_undo)
        self._btn_redo.setEnabled(history.can_redo)

        undo_tip = f"Undo: {history.undo_description}" if history.can_undo else "Nothing to undo"
        redo_tip = f"Redo: {history.redo_description}" if history.can_redo else "Nothing to redo"
        self._btn_undo.setToolTip(undo_tip)
        self._btn_redo.setToolTip(redo_tip)

        # Update history list
        self._history_list.clear()
        for desc in history.history_log():
            self._history_list.addItem(desc)
        # Scroll to latest
        if self._history_list.count() > 0:
            self._history_list.scrollToBottom()

        # Sync toggle button states
        self._btn_track.setChecked(self.app._placement_mode)
        self._btn_add.setChecked(self.app._add_mode)

    # ── Save handlers ────────────────────────────────────────────

    def _on_save(self) -> None:
        path = self.app.save()
        if path is not None:
            self._status_label.setText(f"Saved to {path.name}")
        else:
            self._status_label.setText("Save cancelled or failed")

    def _on_save_as(self) -> None:
        path = self.app.save_as()
        if path is not None:
            self._status_label.setText(f"Saved to {path.name}")
        else:
            self._status_label.setText("Save cancelled")

    # ── Undo/Redo handlers ──────────────────────────────────────

    def _on_undo(self) -> None:
        cmd = self.app.edit_history.undo()
        if cmd:
            self._status_label.setText(f"Undid: {cmd.description}")
        else:
            self._status_label.setText("Nothing to undo")
        self.refresh()

    def _on_redo(self) -> None:
        cmd = self.app.edit_history.redo()
        if cmd:
            self._status_label.setText(f"Redid: {cmd.description}")
        else:
            self._status_label.setText("Nothing to redo")
        self.refresh()

    # ── Edit operation handlers ─────────────────────────────────

    def _get_selected_nucleus(self):
        """Find the currently selected nucleus, or None.

        Returns:
            Tuple of (nucleus, time, index) or None if no selection.
        """
        if not self.app.current_cell_name:
            return None

        cell = self.app.manager.get_cell(self.app.current_cell_name)
        if cell is None:
            return None

        nuc = cell.get_nucleus_at(self.app.current_time)
        if nuc is None:
            return None

        return nuc, self.app.current_time, nuc.index

    def _show_validation_errors(self, errors: list[str]) -> None:
        """Show validation errors in a message box."""
        QMessageBox.warning(
            self,
            "Validation Error",
            "\n".join(errors),
        )

    def _on_add_toggle(self, checked: bool) -> None:
        """Toggle click-to-add mode."""
        if checked:
            # Exit other modes first
            if self.app._placement_mode:
                self.app.exit_placement_mode()
                self._btn_track.setChecked(False)

            self.app.enter_add_mode()
            parent = self.app.current_cell_name
            if parent:
                self._status_label.setText(
                    f"ADD MODE: Left-click in viewer to place nucleus.\n"
                    f"Predecessor: {parent}\n"
                    f"Press Esc or click Add to exit."
                )
            else:
                self._status_label.setText(
                    "ADD MODE: Left-click in viewer to place a new root nucleus.\n"
                    "Press Esc or click Add to exit."
                )
        else:
            self.app.exit_add_mode()
            self._status_label.setText("Exited add mode")

    def _on_remove_nucleus(self) -> None:
        """Remove the currently selected nucleus."""
        sel = self._get_selected_nucleus()
        if sel is None:
            self._status_label.setText("No nucleus selected")
            return

        nuc, time, index = sel

        from ..editing.validators import validate_remove_nucleus

        errors = validate_remove_nucleus(
            self.app.edit_history.nuclei_record, time, index
        )
        if errors:
            self._show_validation_errors(errors)
            return

        name = nuc.effective_name or f"idx={index}"
        reply = QMessageBox.question(
            self,
            "Remove Nucleus",
            f"Remove nucleus '{name}' at t={time}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        from ..editing.commands import RemoveNucleus

        cmd = RemoveNucleus(time=time, index=index)
        self.app.edit_history.do(cmd)
        self._status_label.setText(f"Done: {cmd.description}")
        self.refresh()

    def _nudge(
        self, dx: int = 0, dy: int = 0, dz: float = 0.0, dsize: int = 0
    ) -> None:
        """Nudge the selected nucleus by the given deltas."""
        sel = self._get_selected_nucleus()
        if sel is None:
            self._status_label.setText("No nucleus selected")
            return

        nuc, time, index = sel
        new_x = nuc.x + dx
        new_y = nuc.y + dy
        new_z = max(0.0, nuc.z + dz)
        new_size = max(1, nuc.size + dsize)

        from ..editing.commands import MoveNucleus

        cmd = MoveNucleus(
            time=time,
            index=index,
            new_x=new_x,
            new_y=new_y,
            new_z=new_z,
            new_size=new_size,
        )
        self.app.edit_history.do(cmd)
        parts = []
        if dx:
            parts.append(f"x{dx:+d}")
        if dy:
            parts.append(f"y{dy:+d}")
        if dz:
            parts.append(f"z{dz:+.0f}")
        if dsize:
            parts.append(f"size{dsize:+d}")
        self._status_label.setText(f"Moved: {', '.join(parts)}")
        self.refresh()

    def _on_rename_cell(self) -> None:
        """Open the Rename Cell dialog."""
        sel = self._get_selected_nucleus()
        if sel is None:
            self._status_label.setText("No nucleus selected")
            return

        nuc, time, index = sel
        dialog = RenameCellDialog(nuc.effective_name, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            new_name = dialog.get_name()
            if not new_name:
                self._status_label.setText("Name cannot be empty")
                return

            from ..editing.commands import RenameCell

            cmd = RenameCell(time=time, index=index, new_name=new_name)
            self.app.edit_history.do(cmd)
            self._status_label.setText(f"Done: {cmd.description}")
            self.refresh()

    def _on_kill_cell(self) -> None:
        """Open the Kill Cell dialog."""
        if not self.app.current_cell_name:
            self._status_label.setText("No cell selected")
            return

        cell = self.app.manager.get_cell(self.app.current_cell_name)
        if cell is None:
            self._status_label.setText("Cell not found in lineage")
            return

        dialog = KillCellDialog(cell.name, cell.start_time, cell.end_time, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            from ..editing.validators import validate_kill_cell

            errors = validate_kill_cell(
                self.app.edit_history.nuclei_record,
                values["cell_name"],
                values["start_time"],
            )
            if errors:
                self._show_validation_errors(errors)
                return

            from ..editing.commands import KillCell

            cmd = KillCell(
                cell_name=values["cell_name"],
                start_time=values["start_time"],
                end_time=values["end_time"],
            )
            self.app.edit_history.do(cmd)
            self._status_label.setText(f"Done: {cmd.description}")
            self.refresh()

    def _on_resurrect(self) -> None:
        """Resurrect the currently selected (dead) nucleus."""
        sel = self._get_selected_nucleus()
        if sel is None:
            # Try to find a dead nucleus at the current position
            self._status_label.setText("No nucleus selected")
            return

        nuc, time, index = sel
        dialog = ResurrectDialog(nuc, time, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            identity = dialog.get_identity()
            from ..editing.commands import ResurrectCell

            cmd = ResurrectCell(time=time, index=index, identity=identity)
            self.app.edit_history.do(cmd)
            self._status_label.setText(f"Done: {cmd.description}")
            self.refresh()

    def _on_relink(self) -> None:
        """Start interactive relink: user picks a target cell in the viewer.

        Pick any two cells at different timepoints. The system determines
        which is earlier/later, and auto-interpolates if there is a gap.
        Works both forwards and backwards in time.
        """
        sel = self._get_selected_nucleus()
        if sel is None:
            self._status_label.setText("No nucleus selected")
            return

        nuc, time, index = sel
        self._relink_source = (nuc, time, index)
        self._status_label.setText(
            f"PICK MODE: Navigate to the target cell and click it.\n"
            f"Relinking: '{nuc.effective_name or f'idx={index}'}' at t={time}"
        )
        self._btn_relink.setEnabled(False)

        self.app.enter_relink_pick_mode(self._on_relink_target_picked)

    def _on_relink_target_picked(self, target_time: int, target_nuc) -> None:
        """Callback when the user picks a relink target in the viewer."""
        self._btn_relink.setEnabled(True)

        if not hasattr(self, "_relink_source") or self._relink_source is None:
            self._status_label.setText("Relink cancelled — no source")
            return

        src_nuc, src_time, src_index = self._relink_source
        self._relink_source = None

        target_index = target_nuc.index

        # Both indices are already 1-based (from nucleus.index)
        src_name = src_nuc.effective_name or f"idx={src_index}"
        tgt_name = target_nuc.effective_name or f"idx={target_index}"

        if src_time == target_time:
            QMessageBox.warning(
                self,
                "Invalid Target",
                "Source and target must be at different timepoints.",
            )
            self._status_label.setText("Relink cancelled — same timepoint")
            return

        # Sort so early_* is earlier in time, late_* is later.
        # The "late" cell's predecessor will point to the "early" cell.
        if src_time < target_time:
            early_time, early_index, early_name = src_time, src_index, src_name
            late_time, late_index, late_name = target_time, target_index, tgt_name
            early_nuc, late_nuc = src_nuc, target_nuc
        else:
            early_time, early_index, early_name = target_time, target_index, tgt_name
            late_time, late_index, late_name = src_time, src_index, src_name
            early_nuc, late_nuc = target_nuc, src_nuc

        time_gap = late_time - early_time

        if time_gap == 1:
            # Adjacent timepoints — simple relink
            reply = QMessageBox.question(
                self,
                "Confirm Relink",
                f"Link '{late_name}' at t={late_time}\n"
                f"to '{early_name}' at t={early_time} (predecessor)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                self._status_label.setText("Relink cancelled")
                return

            from ..editing.validators import validate_relink

            errors = validate_relink(
                self.app.edit_history.nuclei_record,
                late_time, late_index, early_index,
            )
            if errors:
                self._show_validation_errors(errors)
                return

            from ..editing.commands import RelinkNucleus

            cmd = RelinkNucleus(
                time=late_time, index=late_index, new_predecessor=early_index,
            )
            self.app.edit_history.do(cmd)
            self._status_label.setText(f"Done: {cmd.description}")
            self.refresh()
        else:
            # Time gap > 1 — auto-interpolate
            reply = QMessageBox.question(
                self,
                "Confirm Relink with Interpolation",
                f"Link '{late_name}' at t={late_time}\n"
                f"to '{early_name}' at t={early_time}\n\n"
                f"Time gap = {time_gap} frames.\n"
                f"Interpolated nuclei will be created for the gap.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                self._status_label.setText("Relink cancelled")
                return

            from ..editing.validators import validate_relink_interpolation

            errors = validate_relink_interpolation(
                self.app.edit_history.nuclei_record,
                early_time, early_index,
                late_time, late_index,
            )
            if errors:
                self._show_validation_errors(errors)
                return

            from ..editing.commands import RelinkWithInterpolation

            cmd = RelinkWithInterpolation(
                start_time=early_time,
                start_index=early_index,
                end_time=late_time,
                end_index=late_index,
            )
            self.app.edit_history.do(cmd)
            self._status_label.setText(f"Done: {cmd.description}")
            self.refresh()

    # ── Track mode ────────────────────────────────────────────────

    def _on_track(self, checked: bool) -> None:
        """Toggle click-to-place tracking mode."""
        if checked:
            parent_name = self.app.current_cell_name or None

            if parent_name:
                cell = self.app.manager.get_cell(parent_name)
                if cell is None:
                    self._status_label.setText(
                        f"Cell '{parent_name}' not in lineage tree — entering root mode"
                    )
                    parent_name = None

            if parent_name:
                self._status_label.setText(
                    f"TRACK MODE: Right-click in viewer to place nuclei.\n"
                    f"Tracking from: {parent_name}\n"
                    f"Navigate to the desired timepoint, then right-click.\n"
                    f"Press Esc or click Track to exit."
                )
            else:
                self._status_label.setText(
                    "TRACK MODE (root): Right-click to place a single root nucleus.\n"
                    "Press Esc or click Track to exit."
                )

            self.app.enter_placement_mode(
                parent_name=parent_name,
                default_size=20,
            )
        else:
            self.app.exit_placement_mode()
            self._status_label.setText("Exited tracking mode")


# ── Dialog classes ───────────────────────────────────────────────


class AddNucleusDialog(QDialog):
    """Dialog for adding a new nucleus.

    Pre-fills with current timepoint and z-plane from the app.
    """

    def __init__(self, current_time: int, current_plane: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Nucleus")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._time_spin = QSpinBox()
        self._time_spin.setRange(1, 9999)
        self._time_spin.setValue(current_time)
        form.addRow("Timepoint:", self._time_spin)

        self._x_spin = QSpinBox()
        self._x_spin.setRange(0, 2048)
        self._x_spin.setValue(256)
        form.addRow("X:", self._x_spin)

        self._y_spin = QSpinBox()
        self._y_spin.setRange(0, 2048)
        self._y_spin.setValue(256)
        form.addRow("Y:", self._y_spin)

        self._z_spin = QDoubleSpinBox()
        self._z_spin.setRange(0.0, 100.0)
        self._z_spin.setDecimals(1)
        self._z_spin.setValue(float(current_plane))
        form.addRow("Z:", self._z_spin)

        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 200)
        self._size_spin.setValue(20)
        form.addRow("Size:", self._size_spin)

        self._identity_edit = QLineEdit()
        self._identity_edit.setPlaceholderText("Optional cell name")
        form.addRow("Name:", self._identity_edit)

        self._pred_spin = QSpinBox()
        self._pred_spin.setRange(-1, 9999)
        self._pred_spin.setValue(-1)
        self._pred_spin.setSpecialValueText("None")
        form.addRow("Predecessor:", self._pred_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> dict:
        """Get the values entered in the dialog."""
        return {
            "time": self._time_spin.value(),
            "x": self._x_spin.value(),
            "y": self._y_spin.value(),
            "z": self._z_spin.value(),
            "size": self._size_spin.value(),
            "identity": self._identity_edit.text().strip(),
            "predecessor": self._pred_spin.value(),
        }


class MoveNucleusDialog(QDialog):
    """Dialog for moving/resizing a nucleus.

    Pre-fills with the nucleus's current position and size.
    """

    def __init__(self, nucleus, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Move Nucleus")
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)

        info = QLabel(
            f"Moving: {nucleus.effective_name or f'idx={nucleus.index}'}\n"
            f"Current: ({nucleus.x}, {nucleus.y}, {nucleus.z:.1f}) size={nucleus.size}"
        )
        layout.addWidget(info)

        form = QFormLayout()

        self._x_spin = QSpinBox()
        self._x_spin.setRange(0, 2048)
        self._x_spin.setValue(nucleus.x)
        form.addRow("New X:", self._x_spin)

        self._y_spin = QSpinBox()
        self._y_spin.setRange(0, 2048)
        self._y_spin.setValue(nucleus.y)
        form.addRow("New Y:", self._y_spin)

        self._z_spin = QDoubleSpinBox()
        self._z_spin.setRange(0.0, 100.0)
        self._z_spin.setDecimals(1)
        self._z_spin.setValue(nucleus.z)
        form.addRow("New Z:", self._z_spin)

        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 200)
        self._size_spin.setValue(nucleus.size)
        form.addRow("New Size:", self._size_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> dict:
        """Get the new position values."""
        return {
            "x": self._x_spin.value(),
            "y": self._y_spin.value(),
            "z": self._z_spin.value(),
            "size": self._size_spin.value(),
        }


class RenameCellDialog(QDialog):
    """Dialog for renaming a cell (setting assigned_id)."""

    def __init__(self, current_name: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rename Cell")
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)

        if current_name:
            layout.addWidget(QLabel(f"Current name: {current_name}"))

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Enter new cell name")
        if current_name:
            self._name_edit.setText(current_name)
        form.addRow("New name:", self._name_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_name(self) -> str:
        """Get the entered name."""
        return self._name_edit.text().strip()


class KillCellDialog(QDialog):
    """Dialog for killing a named cell across timepoints."""

    def __init__(
        self, cell_name: str, start_time: int, end_time: int, parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Kill Cell")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"Kill cell '{cell_name}'\n"
            f"This will mark all nuclei with this name as dead."
        ))

        form = QFormLayout()

        self._name_label = QLabel(cell_name)
        form.addRow("Cell:", self._name_label)
        self._cell_name = cell_name

        self._start_spin = QSpinBox()
        self._start_spin.setRange(1, 9999)
        self._start_spin.setValue(start_time)
        form.addRow("From time:", self._start_spin)

        self._end_spin = QSpinBox()
        self._end_spin.setRange(1, 9999)
        self._end_spin.setValue(end_time)
        form.addRow("To time:", self._end_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> dict:
        """Get the kill parameters."""
        return {
            "cell_name": self._cell_name,
            "start_time": self._start_spin.value(),
            "end_time": self._end_spin.value(),
        }


class ResurrectDialog(QDialog):
    """Dialog for resurrecting a dead nucleus."""

    def __init__(self, nucleus, time: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resurrect Nucleus")
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"Resurrect nucleus at t={time}, idx={nucleus.index}\n"
            f"Position: ({nucleus.x}, {nucleus.y}, {nucleus.z:.1f})"
        ))

        form = QFormLayout()
        self._identity_edit = QLineEdit()
        self._identity_edit.setPlaceholderText("Optional: assign a name")
        if nucleus.effective_name:
            self._identity_edit.setText(nucleus.effective_name)
        form.addRow("Name:", self._identity_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_identity(self) -> str:
        """Get the entered identity."""
        return self._identity_edit.text().strip()


class RelinkDialog(QDialog):
    """Dialog for changing a nucleus's predecessor link."""

    def __init__(self, nucleus, time: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Relink Nucleus")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"Relink nucleus '{nucleus.effective_name or f'idx={nucleus.index}'}' at t={time}\n"
            f"Current predecessor: {nucleus.predecessor}"
        ))

        form = QFormLayout()
        self._pred_spin = QSpinBox()
        self._pred_spin.setRange(-1, 9999)
        self._pred_spin.setValue(nucleus.predecessor)
        self._pred_spin.setSpecialValueText("None")
        form.addRow("New predecessor:", self._pred_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_predecessor(self) -> int:
        """Get the entered predecessor index."""
        return self._pred_spin.value()


class RelinkInterpolationDialog(QDialog):
    """Dialog for relink-with-interpolation between two nuclei."""

    def __init__(
        self, start_time: int, start_index: int, max_time: int, parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Relink with Interpolation")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Create interpolated nuclei between a start and end nucleus.\n"
            "Nuclei will be linearly interpolated in position and size."
        ))

        form = QFormLayout()

        self._start_time_spin = QSpinBox()
        self._start_time_spin.setRange(1, max_time)
        self._start_time_spin.setValue(start_time)
        form.addRow("Start time:", self._start_time_spin)

        self._start_index_spin = QSpinBox()
        self._start_index_spin.setRange(1, 9999)
        self._start_index_spin.setValue(start_index)
        form.addRow("Start index:", self._start_index_spin)

        self._end_time_spin = QSpinBox()
        self._end_time_spin.setRange(1, max_time)
        self._end_time_spin.setValue(min(start_time + 5, max_time))
        form.addRow("End time:", self._end_time_spin)

        self._end_index_spin = QSpinBox()
        self._end_index_spin.setRange(1, 9999)
        self._end_index_spin.setValue(1)
        form.addRow("End index:", self._end_index_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> dict:
        """Get the interpolation parameters."""
        return {
            "start_time": self._start_time_spin.value(),
            "start_index": self._start_index_spin.value(),
            "end_time": self._end_time_spin.value(),
            "end_index": self._end_index_spin.value(),
        }
