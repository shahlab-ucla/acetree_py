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
    from qtpy.QtCore import Qt, Signal
    from qtpy.QtGui import QColor, QFont
    from qtpy.QtWidgets import (
        QButtonGroup,
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QRadioButton,
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

        # ── Color mode toggle ──
        mode_group = QGroupBox("Color Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(2)

        radio_row = QHBoxLayout()
        self._radio_editing = QRadioButton("Editing")
        self._radio_editing.setToolTip(
            "Status-based colors: white=selected, purple=named, "
            "orange=unnamed, gray=none"
        )
        self._radio_editing.setChecked(True)
        self._radio_viz = QRadioButton("Visualization")
        self._radio_viz.setToolTip(
            "Rule-based coloring: choose a preset or define custom rules"
        )
        self._mode_btn_group = QButtonGroup(self)
        self._mode_btn_group.addButton(self._radio_editing, 0)
        self._mode_btn_group.addButton(self._radio_viz, 1)

        radio_row.addWidget(self._radio_editing)
        radio_row.addWidget(self._radio_viz)
        mode_layout.addLayout(radio_row)

        # Preset selector (only visible in visualization mode)
        preset_row = QHBoxLayout()
        preset_label = QLabel("Preset:")
        self._combo_preset = QComboBox()
        self._combo_preset.setToolTip("Select a visualization color preset")
        from .color_rules import PRESET_NAMES
        for key, label in PRESET_NAMES.items():
            self._combo_preset.addItem(label, userData=key)
        # Default to lineage depth
        self._combo_preset.setCurrentIndex(1)
        self._combo_preset.setEnabled(False)
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self._combo_preset, stretch=1)
        mode_layout.addLayout(preset_row)

        self._btn_edit_rules = QPushButton("Edit Rules\u2026")
        self._btn_edit_rules.setToolTip("Open the color rule editor")
        self._btn_edit_rules.setEnabled(False)
        self._btn_edit_rules.clicked.connect(self._open_color_rules_dialog)
        self._color_rules_dialog = None  # lazy-created
        mode_layout.addWidget(self._btn_edit_rules)

        self._mode_btn_group.idClicked.connect(self._on_mode_changed)
        self._combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        layout.addWidget(mode_group)

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

        self._btn_move = QPushButton("Move / Resize \u2197")
        self._btn_move.setToolTip(
            "Open the Move/Resize controls for nudging position, z, and size"
        )
        self._btn_move.clicked.connect(self._open_move_dialog)
        self._move_dialog = None  # lazy-created

        nuc_layout.addWidget(self._btn_add)
        nuc_layout.addWidget(self._btn_remove)
        nuc_layout.addWidget(self._btn_move)
        layout.addWidget(nuc_group)

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

        # ── Visualization tools ──
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.setSpacing(4)

        # Trail toggle + length spinner
        trail_row = QHBoxLayout()
        self._chk_trails = QPushButton("Trails")
        self._chk_trails.setCheckable(True)
        self._chk_trails.setToolTip(
            "Show ghost trail of selected cell's past positions"
        )
        self._chk_trails.clicked.connect(self._on_trail_toggle)

        trail_label = QLabel("Length:")
        self._spin_trail_len = QSpinBox()
        self._spin_trail_len.setRange(1, 100)
        self._spin_trail_len.setValue(10)
        self._spin_trail_len.setToolTip("Number of past timepoints to show")
        self._spin_trail_len.valueChanged.connect(self._on_trail_length_changed)

        trail_row.addWidget(self._chk_trails)
        trail_row.addWidget(trail_label)
        trail_row.addWidget(self._spin_trail_len)
        viz_layout.addLayout(trail_row)

        # Screenshot + export row
        export_row = QHBoxLayout()
        self._btn_screenshot = QPushButton("Screenshot")
        self._btn_screenshot.setToolTip("Save current view as PNG image")
        self._btn_screenshot.clicked.connect(self._on_screenshot)

        self._btn_record = QPushButton("Record...")
        self._btn_record.setToolTip(
            "Export a sequence of screenshots across a timepoint range"
        )
        self._btn_record.clicked.connect(self._on_record_sequence)

        export_row.addWidget(self._btn_screenshot)
        export_row.addWidget(self._btn_record)
        viz_layout.addLayout(export_row)

        layout.addWidget(viz_group)

        # ── History (opens in a popup window) ──
        self._btn_history = QPushButton("Edit History\u2026")
        self._btn_history.setToolTip("Show the full edit history in a popup")
        self._btn_history.clicked.connect(self._open_history_dialog)
        self._history_dialog = None  # lazy-created
        self._history_list = None  # created inside the dialog
        layout.addWidget(self._btn_history)

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

        # Update history popup if it's open
        if self._history_list is not None:
            self._history_list.clear()
            for desc in history.history_log():
                self._history_list.addItem(desc)
            if self._history_list.count() > 0:
                self._history_list.scrollToBottom()

        # Sync toggle button states
        self._btn_track.setChecked(self.app._placement_mode)
        self._btn_add.setChecked(self.app._add_mode)

        # Sync color mode radio
        if self.app._viz_mode:
            self._radio_viz.setChecked(True)
            self._combo_preset.setEnabled(True)
            self._btn_edit_rules.setEnabled(True)
        else:
            self._radio_editing.setChecked(True)
            self._combo_preset.setEnabled(False)
            self._btn_edit_rules.setEnabled(False)

        # Sync trail button
        vi = self.app._viewer_integration
        if vi is not None:
            self._chk_trails.setChecked(vi.trails_visible)

    # ── Color mode handlers ──────────────────────────────────────

    def _on_mode_changed(self, btn_id: int) -> None:
        """Handle editing / visualization radio toggle."""
        viz = btn_id == 1
        self._combo_preset.setEnabled(viz)
        self._btn_edit_rules.setEnabled(viz)
        if viz:
            # Apply current preset
            self._on_preset_changed(self._combo_preset.currentIndex())
        self.app.set_viz_mode(viz)

    def _on_preset_changed(self, index: int) -> None:
        """Handle preset combo box change."""
        from .color_rules import PRESET_EDITING

        key = self._combo_preset.itemData(index)
        if key is None or key == PRESET_EDITING:
            return
        self.app.color_engine.load_preset(key)
        if self.app._viz_mode:
            self.app.set_viz_mode(True)  # re-render with new preset

    def _open_color_rules_dialog(self) -> None:
        """Open (or raise) the Color Rules editor popup."""
        if self._color_rules_dialog is None:
            self._color_rules_dialog = ColorRulesDialog(
                self.app, parent=self
            )
            self._color_rules_dialog.rules_changed.connect(self._on_rules_applied)
        # Sync dialog with current engine rules
        self._color_rules_dialog.load_rules(self.app.color_engine.rules)
        self._color_rules_dialog.show()
        self._color_rules_dialog.raise_()

    def _on_rules_applied(self) -> None:
        """Called when the Color Rules dialog applies new rules."""
        if self.app._viz_mode:
            self.app.set_viz_mode(True)  # re-render

    # ── Trail + export handlers ────────────────────────────────────

    def _on_trail_toggle(self, checked: bool) -> None:
        vi = self.app._viewer_integration
        if vi is not None:
            vi.toggle_trails(checked)

    def _on_trail_length_changed(self, value: int) -> None:
        vi = self.app._viewer_integration
        if vi is not None:
            vi.set_trail_length(value)

    def _on_screenshot(self) -> None:
        path = self.app.screenshot()
        if path is not None:
            self._status_label.setText(f"Screenshot: {path.name}")
        else:
            self._status_label.setText("Screenshot cancelled")

    def _on_record_sequence(self) -> None:
        """Open a dialog to record a sequence of screenshots."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Record Sequence")
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        form = QFormLayout(dlg)

        spin_start = QSpinBox()
        spin_start.setRange(1, 9999)
        spin_start.setValue(self.app.current_time)
        form.addRow("Start time:", spin_start)

        spin_end = QSpinBox()
        spin_end.setRange(1, 9999)
        spin_end.setValue(min(self.app.current_time + 20,
                              len(self.app.manager.nuclei_record)))
        form.addRow("End time:", spin_end)

        spin_step = QSpinBox()
        spin_step.setRange(1, 100)
        spin_step.setValue(1)
        form.addRow("Step:", spin_step)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec_() != QDialog.Accepted:
            self._status_label.setText("Recording cancelled")
            return

        t_start = spin_start.value()
        t_end = spin_end.value()
        step = spin_step.value()

        from qtpy.QtWidgets import QFileDialog
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select output directory"
        )
        if not out_dir:
            self._status_label.setText("Recording cancelled")
            return

        count = self.app.record_sequence(t_start, t_end, step, out_dir)
        self._status_label.setText(
            f"Recorded {count} frames to {out_dir}"
        )

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

    # ── Popup dialogs (Move/Resize and History) ────────────────

    def _open_move_dialog(self) -> None:
        """Open (or focus) the non-modal Move/Resize dialog."""
        if self._move_dialog is not None:
            self._move_dialog.raise_()
            self._move_dialog.activateWindow()
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Move / Resize")
        dlg.setWindowFlags(
            dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )
        layout = QVBoxLayout(dlg)
        self._build_move_controls(layout)
        dlg.finished.connect(self._on_move_dialog_closed)
        dlg.setFixedSize(dlg.sizeHint())
        self._move_dialog = dlg
        dlg.show()

    def _on_move_dialog_closed(self) -> None:
        self._move_dialog = None

    def _open_history_dialog(self) -> None:
        """Open (or focus) the non-modal Edit History dialog."""
        if self._history_dialog is not None:
            self._history_dialog.raise_()
            self._history_dialog.activateWindow()
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit History")
        dlg.setWindowFlags(
            dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )
        dlg.setMinimumSize(300, 250)
        layout = QVBoxLayout(dlg)

        self._history_list = QListWidget()
        layout.addWidget(self._history_list)

        # Populate with current history
        for desc in self.app.edit_history.history_log():
            self._history_list.addItem(desc)
        if self._history_list.count() > 0:
            self._history_list.scrollToBottom()

        dlg.finished.connect(self._on_history_dialog_closed)
        self._history_dialog = dlg
        dlg.show()

    def _on_history_dialog_closed(self) -> None:
        self._history_list = None
        self._history_dialog = None

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


# ── Color Rules Dialog ──────────────────────────────────────────────


class ColorRulesDialog(QDialog):  # type: ignore[misc]
    """Popup dialog for viewing and editing visualization color rules.

    Provides a list of rules with enable/disable checkboxes, and buttons
    to add, edit, delete, reorder, and clear rules.  Changes are pushed
    to the :class:`ColorRuleEngine` on the app when *Apply* is clicked.
    """

    rules_changed = Signal()

    def __init__(self, app: AceTreeApp, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self.setWindowTitle("Color Rules")
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(420)
        self._rules: list = []  # list[ColorRule] — local working copy

        layout = QVBoxLayout(self)

        # Rule list
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemDoubleClicked.connect(self._on_edit_rule)
        layout.addWidget(self._list)

        # "All other cells" default color
        default_row = QHBoxLayout()
        default_row.addWidget(QLabel("All other cells:"))
        self._btn_default_color = QPushButton()
        self._btn_default_color.setToolTip(
            "Color for cells that don't match any rule above"
        )
        self._btn_default_color.clicked.connect(self._pick_default_color)
        self._default_color: tuple = (1.0, 1.0, 1.0, 0.3)
        self._update_default_color_button()
        default_row.addWidget(self._btn_default_color, stretch=1)
        layout.addLayout(default_row)

        # Buttons row
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add")
        self._btn_add.clicked.connect(self._on_add_rule)
        self._btn_edit = QPushButton("Edit")
        self._btn_edit.clicked.connect(
            lambda: self._on_edit_rule(self._list.currentItem())
        )
        self._btn_delete = QPushButton("Delete")
        self._btn_delete.clicked.connect(self._on_delete_rule)
        self._btn_up = QPushButton("\u25b2")
        self._btn_up.setToolTip("Move rule up (higher priority)")
        self._btn_up.setFixedWidth(30)
        self._btn_up.clicked.connect(lambda: self._move_rule(-1))
        self._btn_down = QPushButton("\u25bc")
        self._btn_down.setToolTip("Move rule down (lower priority)")
        self._btn_down.setFixedWidth(30)
        self._btn_down.clicked.connect(lambda: self._move_rule(1))
        self._btn_clear = QPushButton("Clear All")
        self._btn_clear.clicked.connect(self._on_clear)

        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_edit)
        btn_row.addWidget(self._btn_delete)
        btn_row.addWidget(self._btn_up)
        btn_row.addWidget(self._btn_down)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

        # Apply / Close
        bottom = QHBoxLayout()
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setToolTip("Push rules to the viewer")
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        bottom.addStretch()
        bottom.addWidget(self._btn_apply)
        bottom.addWidget(self._btn_close)
        layout.addLayout(bottom)

    def load_rules(self, rules: list) -> None:
        """Populate the dialog from a list of ColorRule objects."""
        from .color_rules import ColorRule

        # Deep copy so edits don't affect the engine until Apply
        self._rules = [
            ColorRule(
                name=r.name,
                criterion=r.criterion,
                pattern=r.pattern,
                color_mode=r.color_mode,
                color=r.color,
                colormap=r.colormap,
                vmin=r.vmin,
                vmax=r.vmax,
                priority=r.priority,
                enabled=r.enabled,
            )
            for r in rules
        ]
        # Sync default color from engine
        self._default_color = self.app.color_engine.default_color
        self._update_default_color_button()
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        """Refresh the QListWidget from self._rules."""
        self._list.clear()
        for rule in self._rules:
            item = QListWidgetItem()
            item.setFlags(
                item.flags() | Qt.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.Checked if rule.enabled else Qt.Unchecked
            )
            item.setText(self._rule_label(rule))
            self._list.addItem(item)

    @staticmethod
    def _rule_label(rule) -> str:
        """Build a human-readable label for a rule."""
        from .color_rules import ColorMode

        label = rule.name or rule.criterion.value
        if rule.pattern:
            label += f"  [{rule.pattern}]"
        if rule.color_mode == ColorMode.SOLID:
            r, g, b, a = rule.color
            label += f"  \u25a0 ({r:.1f},{g:.1f},{b:.1f},{a:.1f})"
        else:
            label += f"  cmap={rule.colormap}"
        return label

    def _sync_enabled_from_list(self) -> None:
        """Sync rule.enabled from checkbox states."""
        for i, rule in enumerate(self._rules):
            item = self._list.item(i)
            if item is not None:
                rule.enabled = item.checkState() == Qt.Checked

    def _on_add_rule(self) -> None:
        from .color_rules import ColorRule

        dlg = _RuleEditorDialog(ColorRule(), parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._sync_enabled_from_list()
            self._rules.append(dlg.get_rule())
            self._rebuild_list()

    def _on_edit_rule(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        row = self._list.row(item)
        if row < 0 or row >= len(self._rules):
            return
        self._sync_enabled_from_list()
        dlg = _RuleEditorDialog(self._rules[row], parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._rules[row] = dlg.get_rule()
            self._rebuild_list()

    def _on_delete_rule(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._rules):
            return
        self._sync_enabled_from_list()
        del self._rules[row]
        self._rebuild_list()

    def _move_rule(self, direction: int) -> None:
        """Move the selected rule up (-1) or down (+1)."""
        row = self._list.currentRow()
        new_row = row + direction
        if row < 0 or new_row < 0 or new_row >= len(self._rules):
            return
        self._sync_enabled_from_list()
        self._rules[row], self._rules[new_row] = (
            self._rules[new_row],
            self._rules[row],
        )
        self._rebuild_list()
        self._list.setCurrentRow(new_row)

    def _on_clear(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear All Rules",
            "Remove all color rules?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._rules.clear()
            self._rebuild_list()

    def _update_default_color_button(self) -> None:
        r, g, b, a = self._default_color
        ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
        self._btn_default_color.setStyleSheet(
            f"background-color: rgba({ri},{gi},{bi},{int(a * 255)}); "
            f"min-height: 20px;"
        )
        self._btn_default_color.setText(
            f"({r:.2f}, {g:.2f}, {b:.2f}, {a:.2f})"
        )

    def _pick_default_color(self) -> None:
        r, g, b, a = self._default_color
        initial = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
        color = QColorDialog.getColor(
            initial, self, "Default Color (All Other Cells)",
            QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self._default_color = (
                color.redF(), color.greenF(), color.blueF(), color.alphaF(),
            )
            self._update_default_color_button()

    def _on_apply(self) -> None:
        """Push the working rule list to the engine."""
        self._sync_enabled_from_list()
        # Assign descending priorities based on list order
        for i, rule in enumerate(self._rules):
            rule.priority = len(self._rules) - i
        self.app.color_engine.set_rules(list(self._rules))
        self.app.color_engine.default_color = self._default_color
        self.rules_changed.emit()


class _RuleEditorDialog(QDialog):  # type: ignore[misc]
    """Sub-dialog for creating or editing a single ColorRule."""

    def __init__(self, rule, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from .color_rules import ColorMode, ColorRule, RuleCriterion

        self._ColorMode = ColorMode
        self._ColorRule = ColorRule
        self._RuleCriterion = RuleCriterion

        self.setWindowTitle("Edit Color Rule")
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Name
        self._name_edit = QLineEdit(rule.name)
        form.addRow("Name:", self._name_edit)

        # Criterion + info button
        criterion_row = QHBoxLayout()
        self._criterion_combo = QComboBox()
        for c in RuleCriterion:
            self._criterion_combo.addItem(c.value, userData=c)
        idx = [c for c in RuleCriterion].index(rule.criterion)
        self._criterion_combo.setCurrentIndex(idx)
        criterion_row.addWidget(self._criterion_combo, stretch=1)

        btn_info = QPushButton("?")
        btn_info.setFixedWidth(24)
        btn_info.setToolTip("Explain match modes")
        btn_info.clicked.connect(self._show_match_help)
        criterion_row.addWidget(btn_info)
        form.addRow("Match:", criterion_row)

        # Pattern
        self._pattern_edit = QLineEdit(rule.pattern)
        self._pattern_edit.setPlaceholderText(
            "e.g. AB*, 2-5, divided, ^MS[ap]$"
        )
        form.addRow("Pattern:", self._pattern_edit)

        # Color mode
        self._color_mode_combo = QComboBox()
        for m in ColorMode:
            self._color_mode_combo.addItem(m.value, userData=m)
        cm_idx = [m for m in ColorMode].index(rule.color_mode)
        self._color_mode_combo.setCurrentIndex(cm_idx)
        self._color_mode_combo.currentIndexChanged.connect(
            self._on_color_mode_changed
        )
        form.addRow("Color mode:", self._color_mode_combo)

        # Solid color picker
        self._color_btn = QPushButton()
        self._current_color = rule.color  # (r, g, b, a)
        self._update_color_button()
        self._color_btn.clicked.connect(self._pick_color)
        form.addRow("Color:", self._color_btn)

        # Colormap settings
        self._cmap_edit = QLineEdit(rule.colormap)
        self._cmap_edit.setPlaceholderText("viridis, inferno, plasma, ...")
        form.addRow("Colormap:", self._cmap_edit)

        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(-1e6, 1e6)
        self._vmin_spin.setValue(rule.vmin)
        form.addRow("Min value:", self._vmin_spin)

        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(-1e6, 1e6)
        self._vmax_spin.setValue(rule.vmax)
        form.addRow("Max value:", self._vmax_spin)

        layout.addLayout(form)

        # Show/hide fields based on color mode
        self._on_color_mode_changed(cm_idx)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _show_match_help(self) -> None:
        """Show a help dialog explaining the match modes."""
        QMessageBox.information(
            self,
            "Match Modes",
            "<b>all</b> — Matches every cell. Use this for a catch-all "
            "rule or to color everything with one colormap.<br><br>"
            "<b>name_exact</b> — Matches one specific cell by name.<br>"
            "Pattern: <code>ABala</code><br><br>"
            "<b>name_pattern</b> — Matches cell names using wildcards "
            "(like file globbing).<br>"
            "Pattern: <code>AB*</code> (all AB-lineage cells), "
            "<code>MS?</code> (MSa, MSp, etc.)<br><br>"
            "<b>name_regex</b> — Matches cell names using a regular "
            "expression for advanced patterns.<br>"
            "Pattern: <code>^AB[ap]$</code> (only ABa and ABp), "
            "<code>.*ala.*</code> (any name containing 'ala')<br><br>"
            "<b>lineage_depth</b> — Matches cells by how many divisions "
            "from the founder cell P0. Depth 0 = P0 itself.<br>"
            "Pattern: <code>2-4</code> (depths 2 through 4), "
            "<code>3</code> (depth 3 only)<br><br>"
            "<b>fate</b> — Matches cells by their end fate.<br>"
            "Pattern: <code>divided</code>, <code>alive</code>, or "
            "<code>died</code><br><br>"
            "<b>expression</b> — Matches cells by expression level "
            "(rweight / fluorescence intensity).<br>"
            "Pattern: <code>500-2000</code> (values between 500 and 2000)",
        )

    def _on_color_mode_changed(self, index: int) -> None:
        """Show/hide fields based on color mode."""
        is_solid = index == 0  # SOLID
        self._color_btn.setVisible(is_solid)
        self._cmap_edit.setVisible(not is_solid)
        self._vmin_spin.setVisible(not is_solid)
        self._vmax_spin.setVisible(not is_solid)

    def _update_color_button(self) -> None:
        r, g, b, a = self._current_color
        ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
        self._color_btn.setStyleSheet(
            f"background-color: rgb({ri},{gi},{bi}); "
            f"min-width: 60px; min-height: 20px;"
        )
        self._color_btn.setText(f"({r:.2f}, {g:.2f}, {b:.2f}, {a:.2f})")

    def _pick_color(self) -> None:
        r, g, b, a = self._current_color
        initial = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
        color = QColorDialog.getColor(
            initial, self, "Pick Rule Color",
            QColorDialog.ShowAlphaChannel,
        )
        if color.isValid():
            self._current_color = (
                color.redF(),
                color.greenF(),
                color.blueF(),
                color.alphaF(),
            )
            self._update_color_button()

    def get_rule(self):
        """Build a ColorRule from the dialog's current values."""
        criterion = self._criterion_combo.currentData()
        color_mode = self._color_mode_combo.currentData()
        return self._ColorRule(
            name=self._name_edit.text().strip(),
            criterion=criterion,
            pattern=self._pattern_edit.text().strip(),
            color_mode=color_mode,
            color=self._current_color,
            colormap=self._cmap_edit.text().strip() or "viridis",
            vmin=self._vmin_spin.value(),
            vmax=self._vmax_spin.value(),
            priority=0,
            enabled=True,
        )
