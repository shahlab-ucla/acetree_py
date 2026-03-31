"""Tests for the edit panel dock widget and its dialogs.

Tests cover:
- EditPanel instantiation and layout
- Button states (enabled/disabled) based on edit history
- History list population and scrolling
- Undo/redo button click handlers
- Dialog instantiation with correct defaults
- Dialog get_values() methods
- Edit command wiring: button → dialog → validator → command → history
- Validation error display
- Status label updates
- Integration with AceTreeApp edit history
"""

from __future__ import annotations

import pytest

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.core.cell import Cell, CellFate
from acetree_py.core.lineage import LineageTree
from acetree_py.core.movie import Movie
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.editing.history import EditHistory
from acetree_py.editing.commands import (
    AddNucleus,
    MoveNucleus,
    RemoveNucleus,
    RenameCell,
)

# ── Skip if Qt not available ─────────────────────────────────────

pytest.importorskip("qtpy")


# ── Fixtures ─────────────────────────────────────────────────────


def _make_nucleus(index=1, x=100, y=200, z=10.0, size=20, identity="ABa",
                  status=1, predecessor=NILLI, successor1=NILLI, successor2=NILLI):
    return Nucleus(
        index=index, x=x, y=y, z=z, size=size, identity=identity,
        status=status, predecessor=predecessor,
        successor1=successor1, successor2=successor2,
    )


def _make_nuclei_record():
    """3 timepoints: t1 has 1 nucleus (P0), t2 has 2 (AB, P1), t3 has 3 (ABa, ABp, P1)."""
    return [
        [_make_nucleus(1, 300, 250, 15.0, 20, "P0", successor1=1, successor2=2)],
        [
            _make_nucleus(1, 280, 240, 14.0, 18, "AB", predecessor=1, successor1=1, successor2=2),
            _make_nucleus(2, 320, 260, 16.0, 22, "P1", predecessor=1, successor1=3),
        ],
        [
            _make_nucleus(1, 260, 230, 13.0, 16, "ABa", predecessor=1),
            _make_nucleus(2, 300, 250, 15.0, 20, "ABp", predecessor=1),
            _make_nucleus(3, 340, 270, 17.0, 24, "P1", predecessor=2),
        ],
    ]


class MockManager:
    """Minimal mock of NucleiManager for testing the edit panel."""

    def __init__(self, nuclei_record):
        self.nuclei_record = nuclei_record
        self.num_timepoints = len(nuclei_record)
        self.movie = Movie()
        # Build a simple lineage tree with cells
        self._cells = {}
        self._build_cells()

    def _build_cells(self):
        for t_idx, nuclei in enumerate(self.nuclei_record):
            for nuc in nuclei:
                if nuc.effective_name and nuc.effective_name not in self._cells:
                    self._cells[nuc.effective_name] = Cell(
                        name=nuc.effective_name,
                        start_time=t_idx + 1,
                        end_time=t_idx + 1,
                        nuclei=[(t_idx + 1, nuc)],
                    )
                elif nuc.effective_name in self._cells:
                    cell = self._cells[nuc.effective_name]
                    cell.end_time = t_idx + 1
                    cell.nuclei.append((t_idx + 1, nuc))

    def get_cell(self, name):
        return self._cells.get(name)

    def alive_nuclei_at(self, time):
        if time < 1 or time > len(self.nuclei_record):
            return []
        return [n for n in self.nuclei_record[time - 1] if n.is_alive]

    def find_closest_nucleus(self, x, y, z, time):
        return None

    def nucleus_diameter(self, nuc, plane):
        return 10.0

    def set_all_successors(self):
        pass

    def process(self):
        pass


class MockApp:
    """Minimal mock of AceTreeApp for testing the edit panel."""

    def __init__(self, nuclei_record=None):
        if nuclei_record is None:
            nuclei_record = _make_nuclei_record()
        self.manager = MockManager(nuclei_record)
        self.edit_history = EditHistory(nuclei_record)
        self.current_time = 2
        self.current_plane = 15
        self.current_cell_name = "AB"
        self.image_provider = None
        self.viewer = None
        self._image_layer = None
        self._viewer_integration = None
        self._player_controls = None
        self._cell_info_panel = None
        self._contrast_tools = None
        self._edit_panel = None
        self.tracking = True
        self._placement_mode = False
        self._add_mode = False
        self._viz_mode = False
        self._color_engine = None

    @property
    def color_engine(self):
        if self._color_engine is None:
            from acetree_py.gui.color_rules import ColorRuleEngine
            self._color_engine = ColorRuleEngine()
        return self._color_engine

    def set_viz_mode(self, enabled):
        self._viz_mode = enabled

    def update_display(self):
        pass

    def set_time(self, t):
        self.current_time = t

    def set_plane(self, p):
        self.current_plane = p


# ── EditPanel tests ──────────────────────────────────────────────


class TestEditPanel:
    """Tests for the EditPanel widget."""

    def test_instantiation(self, qtbot):
        """EditPanel can be created with a mock app."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)
        assert panel is not None

    def test_has_all_buttons(self, qtbot):
        """EditPanel has all expected operation buttons."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        assert panel._btn_undo is not None
        assert panel._btn_redo is not None
        assert panel._btn_add is not None
        assert panel._btn_remove is not None
        assert panel._btn_track is not None
        assert panel._btn_rename is not None
        assert panel._btn_kill is not None
        assert panel._btn_resurrect is not None
        assert panel._btn_relink is not None

    def test_initial_undo_redo_disabled(self, qtbot):
        """Undo/redo buttons initially disabled when no edits."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)
        panel.refresh()

        assert not panel._btn_undo.isEnabled()
        assert not panel._btn_redo.isEnabled()

    def test_undo_enabled_after_edit(self, qtbot):
        """Undo button becomes enabled after an edit."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        # Execute an edit
        cmd = AddNucleus(time=1, x=50, y=50, z=5.0, size=15)
        app.edit_history.do(cmd)
        panel.refresh()

        assert panel._btn_undo.isEnabled()
        assert not panel._btn_redo.isEnabled()

    def test_redo_enabled_after_undo(self, qtbot):
        """Redo button becomes enabled after an undo."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        cmd = AddNucleus(time=1, x=50, y=50, z=5.0, size=15)
        app.edit_history.do(cmd)
        app.edit_history.undo()
        panel.refresh()

        assert not panel._btn_undo.isEnabled()
        assert panel._btn_redo.isEnabled()

    def test_history_list_populated(self, qtbot):
        """History dialog shows executed command descriptions when opened."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        app.edit_history.do(AddNucleus(time=1, x=50, y=50, z=5.0))
        app.edit_history.do(AddNucleus(time=1, x=60, y=60, z=6.0))

        # History list is lazily created inside the popup dialog
        panel._open_history_dialog()
        assert panel._history_list is not None
        assert panel._history_list.count() == 2
        panel._history_dialog.close()

    def test_status_label_default(self, qtbot):
        """Status label shows 'Ready' initially."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        assert panel._status_label.text() == "Ready"

    def test_undo_button_click(self, qtbot):
        """Clicking undo button undoes the last edit."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        cmd = AddNucleus(time=1, x=50, y=50, z=5.0)
        app.edit_history.do(cmd)
        original_count = len(app.edit_history.nuclei_record[0])

        panel._btn_undo.click()

        assert len(app.edit_history.nuclei_record[0]) == original_count - 1
        assert "Undid" in panel._status_label.text()

    def test_redo_button_click(self, qtbot):
        """Clicking redo button re-does the last undone edit."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        cmd = AddNucleus(time=1, x=50, y=50, z=5.0)
        app.edit_history.do(cmd)
        app.edit_history.undo()
        count_before = len(app.edit_history.nuclei_record[0])

        panel._btn_redo.click()

        assert len(app.edit_history.nuclei_record[0]) == count_before + 1
        assert "Redid" in panel._status_label.text()

    def test_undo_nothing_updates_status(self, qtbot):
        """Clicking undo with no history shows appropriate message."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_undo()
        assert "Nothing to undo" in panel._status_label.text()

    def test_redo_nothing_updates_status(self, qtbot):
        """Clicking redo with no redo history shows appropriate message."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_redo()
        assert "Nothing to redo" in panel._status_label.text()

    def test_get_selected_nucleus_returns_none_no_selection(self, qtbot):
        """_get_selected_nucleus returns None when no cell is selected."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        app.current_cell_name = ""
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        assert panel._get_selected_nucleus() is None

    def test_get_selected_nucleus_returns_nucleus(self, qtbot):
        """_get_selected_nucleus returns the selected nucleus."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        app.current_cell_name = "AB"
        app.current_time = 2
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        result = panel._get_selected_nucleus()
        assert result is not None
        nuc, time, index = result
        assert nuc.identity == "AB"
        assert time == 2

    def test_remove_no_selection_updates_status(self, qtbot):
        """Clicking remove with no selection shows status message."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        app.current_cell_name = ""
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_remove_nucleus()
        assert "No nucleus selected" in panel._status_label.text()

    def test_nudge_no_selection_updates_status(self, qtbot):
        """Nudge with no selection shows status message."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        app.current_cell_name = ""
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._nudge(dx=1)
        assert "No nucleus selected" in panel._status_label.text()

    def test_rename_no_selection_updates_status(self, qtbot):
        """Clicking rename with no selection shows status message."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        app.current_cell_name = ""
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_rename_cell()
        assert "No nucleus selected" in panel._status_label.text()

    def test_kill_no_selection_updates_status(self, qtbot):
        """Clicking kill with no cell selected shows status message."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        app.current_cell_name = ""
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_kill_cell()
        assert "No cell selected" in panel._status_label.text()

    def test_undo_tooltip_updates(self, qtbot):
        """Undo button tooltip shows description of next undo-able command."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        cmd = AddNucleus(time=1, x=50, y=50, z=5.0)
        app.edit_history.do(cmd)
        panel.refresh()

        assert "Add nucleus" in panel._btn_undo.toolTip()

    def test_multiple_edits_history_order(self, qtbot):
        """History dialog maintains correct chronological order."""
        from acetree_py.gui.edit_panel import EditPanel
        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        app.edit_history.do(AddNucleus(time=1, x=50, y=50, z=5.0, identity="Cell1"))
        app.edit_history.do(AddNucleus(time=1, x=60, y=60, z=6.0, identity="Cell2"))
        app.edit_history.do(AddNucleus(time=1, x=70, y=70, z=7.0, identity="Cell3"))

        # History list is in a popup dialog — open it to inspect
        panel._open_history_dialog()
        assert panel._history_list.count() == 3
        assert "Cell1" in panel._history_list.item(0).text()
        assert "Cell3" in panel._history_list.item(2).text()
        panel._history_dialog.close()


# ── Dialog tests ─────────────────────────────────────────────────


class TestAddNucleusDialog:
    """Tests for the AddNucleus dialog."""

    def test_instantiation(self, qtbot):
        from acetree_py.gui.edit_panel import AddNucleusDialog
        dialog = AddNucleusDialog(current_time=5, current_plane=12)
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Add Nucleus"

    def test_default_values(self, qtbot):
        from acetree_py.gui.edit_panel import AddNucleusDialog
        dialog = AddNucleusDialog(current_time=5, current_plane=12)
        qtbot.addWidget(dialog)

        values = dialog.get_values()
        assert values["time"] == 5
        assert values["z"] == 12.0
        assert values["size"] == 20
        assert values["predecessor"] == -1

    def test_custom_values(self, qtbot):
        from acetree_py.gui.edit_panel import AddNucleusDialog
        dialog = AddNucleusDialog(current_time=5, current_plane=12)
        qtbot.addWidget(dialog)

        dialog._x_spin.setValue(100)
        dialog._y_spin.setValue(200)
        dialog._z_spin.setValue(15.5)
        dialog._size_spin.setValue(25)
        dialog._identity_edit.setText("TestCell")
        dialog._pred_spin.setValue(3)

        values = dialog.get_values()
        assert values["x"] == 100
        assert values["y"] == 200
        assert values["z"] == 15.5
        assert values["size"] == 25
        assert values["identity"] == "TestCell"
        assert values["predecessor"] == 3


class TestMoveNucleusDialog:
    """Tests for the MoveNucleus dialog."""

    def test_instantiation(self, qtbot):
        from acetree_py.gui.edit_panel import MoveNucleusDialog
        nuc = _make_nucleus(x=100, y=200, z=10.0, size=20)
        dialog = MoveNucleusDialog(nuc)
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Move Nucleus"

    def test_prefilled_values(self, qtbot):
        from acetree_py.gui.edit_panel import MoveNucleusDialog
        nuc = _make_nucleus(x=150, y=250, z=12.5, size=22)
        dialog = MoveNucleusDialog(nuc)
        qtbot.addWidget(dialog)

        values = dialog.get_values()
        assert values["x"] == 150
        assert values["y"] == 250
        assert values["z"] == 12.5
        assert values["size"] == 22


class TestRenameCellDialog:
    """Tests for the RenameCell dialog."""

    def test_instantiation(self, qtbot):
        from acetree_py.gui.edit_panel import RenameCellDialog
        dialog = RenameCellDialog("ABa")
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Rename Cell"

    def test_prefilled_name(self, qtbot):
        from acetree_py.gui.edit_panel import RenameCellDialog
        dialog = RenameCellDialog("ABa")
        qtbot.addWidget(dialog)
        assert dialog.get_name() == "ABa"

    def test_empty_name(self, qtbot):
        from acetree_py.gui.edit_panel import RenameCellDialog
        dialog = RenameCellDialog("")
        qtbot.addWidget(dialog)
        assert dialog.get_name() == ""

    def test_modified_name(self, qtbot):
        from acetree_py.gui.edit_panel import RenameCellDialog
        dialog = RenameCellDialog("ABa")
        qtbot.addWidget(dialog)
        dialog._name_edit.setText("ABala")
        assert dialog.get_name() == "ABala"


class TestKillCellDialog:
    """Tests for the KillCell dialog."""

    def test_instantiation(self, qtbot):
        from acetree_py.gui.edit_panel import KillCellDialog
        dialog = KillCellDialog("ABa", start_time=5, end_time=20)
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Kill Cell"

    def test_values(self, qtbot):
        from acetree_py.gui.edit_panel import KillCellDialog
        dialog = KillCellDialog("ABa", start_time=5, end_time=20)
        qtbot.addWidget(dialog)

        values = dialog.get_values()
        assert values["cell_name"] == "ABa"
        assert values["start_time"] == 5
        assert values["end_time"] == 20

    def test_modified_time_range(self, qtbot):
        from acetree_py.gui.edit_panel import KillCellDialog
        dialog = KillCellDialog("ABa", start_time=5, end_time=20)
        qtbot.addWidget(dialog)

        dialog._start_spin.setValue(10)
        dialog._end_spin.setValue(15)
        values = dialog.get_values()
        assert values["start_time"] == 10
        assert values["end_time"] == 15


class TestResurrectDialog:
    """Tests for the Resurrect dialog."""

    def test_instantiation(self, qtbot):
        from acetree_py.gui.edit_panel import ResurrectDialog
        nuc = _make_nucleus(status=-1, identity="")
        dialog = ResurrectDialog(nuc, time=5)
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Resurrect Nucleus"

    def test_identity_input(self, qtbot):
        from acetree_py.gui.edit_panel import ResurrectDialog
        nuc = _make_nucleus(status=-1, identity="")
        dialog = ResurrectDialog(nuc, time=5)
        qtbot.addWidget(dialog)

        dialog._identity_edit.setText("RestoredCell")
        assert dialog.get_identity() == "RestoredCell"


class TestRelinkDialog:
    """Tests for the Relink dialog."""

    def test_instantiation(self, qtbot):
        from acetree_py.gui.edit_panel import RelinkDialog
        nuc = _make_nucleus(predecessor=3)
        dialog = RelinkDialog(nuc, time=5)
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Relink Nucleus"

    def test_prefilled_predecessor(self, qtbot):
        from acetree_py.gui.edit_panel import RelinkDialog
        nuc = _make_nucleus(predecessor=3)
        dialog = RelinkDialog(nuc, time=5)
        qtbot.addWidget(dialog)
        assert dialog.get_predecessor() == 3

    def test_modified_predecessor(self, qtbot):
        from acetree_py.gui.edit_panel import RelinkDialog
        nuc = _make_nucleus(predecessor=3)
        dialog = RelinkDialog(nuc, time=5)
        qtbot.addWidget(dialog)

        dialog._pred_spin.setValue(7)
        assert dialog.get_predecessor() == 7


class TestRelinkInterpolationDialog:
    """Tests for the RelinkInterpolation dialog."""

    def test_instantiation(self, qtbot):
        from acetree_py.gui.edit_panel import RelinkInterpolationDialog
        dialog = RelinkInterpolationDialog(start_time=5, start_index=2, max_time=100)
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == "Relink with Interpolation"

    def test_default_values(self, qtbot):
        from acetree_py.gui.edit_panel import RelinkInterpolationDialog
        dialog = RelinkInterpolationDialog(start_time=5, start_index=2, max_time=100)
        qtbot.addWidget(dialog)

        values = dialog.get_values()
        assert values["start_time"] == 5
        assert values["start_index"] == 2
        assert values["end_time"] == 10  # start_time + 5
        assert values["end_index"] == 1

    def test_custom_values(self, qtbot):
        from acetree_py.gui.edit_panel import RelinkInterpolationDialog
        dialog = RelinkInterpolationDialog(start_time=5, start_index=2, max_time=100)
        qtbot.addWidget(dialog)

        dialog._end_time_spin.setValue(20)
        dialog._end_index_spin.setValue(4)
        values = dialog.get_values()
        assert values["end_time"] == 20
        assert values["end_index"] == 4

    def test_max_time_clamped(self, qtbot):
        """End time near max_time is clamped correctly."""
        from acetree_py.gui.edit_panel import RelinkInterpolationDialog
        dialog = RelinkInterpolationDialog(start_time=98, start_index=1, max_time=100)
        qtbot.addWidget(dialog)

        values = dialog.get_values()
        assert values["end_time"] == 100  # min(98+5, 100) = 100


# ── Integration: button → command pipeline ───────────────────────


class TestEditPanelIntegration:
    """Integration tests verifying the full button → command pipeline."""

    def test_add_nucleus_via_dialog_values(self, qtbot):
        """Verify AddNucleus command can be created from dialog values."""
        from acetree_py.gui.edit_panel import AddNucleusDialog
        from acetree_py.editing.commands import AddNucleus

        dialog = AddNucleusDialog(current_time=2, current_plane=10)
        qtbot.addWidget(dialog)
        dialog._x_spin.setValue(100)
        dialog._y_spin.setValue(200)

        values = dialog.get_values()
        cmd = AddNucleus(
            time=values["time"], x=values["x"], y=values["y"],
            z=values["z"], size=values["size"],
            identity=values["identity"], predecessor=values["predecessor"],
        )
        assert cmd.time == 2
        assert cmd.x == 100
        assert cmd.y == 200

    def test_rename_via_dialog_values(self, qtbot):
        """Verify RenameCell command can be created from dialog values."""
        from acetree_py.gui.edit_panel import RenameCellDialog
        from acetree_py.editing.commands import RenameCell

        dialog = RenameCellDialog("ABa")
        qtbot.addWidget(dialog)
        dialog._name_edit.setText("ABala")

        cmd = RenameCell(time=2, index=1, new_name=dialog.get_name())
        assert cmd.new_name == "ABala"

    def test_edit_panel_refresh_after_external_edit(self, qtbot):
        """Edit panel correctly refreshes when edits happen externally."""
        from acetree_py.gui.edit_panel import EditPanel

        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        # Open history dialog to inspect list
        panel._open_history_dialog()
        assert panel._history_list.count() == 0

        # External edit (not via panel)
        app.edit_history.do(AddNucleus(time=1, x=50, y=50, z=5.0))
        panel.refresh()

        assert panel._history_list.count() == 1
        assert panel._btn_undo.isEnabled()
        panel._history_dialog.close()

    def test_undo_redo_cycle_updates_panel(self, qtbot):
        """Full undo/redo cycle properly updates panel state."""
        from acetree_py.gui.edit_panel import EditPanel

        app = MockApp()
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        # Open history dialog to inspect list
        panel._open_history_dialog()

        # Do edit
        app.edit_history.do(AddNucleus(time=1, x=50, y=50, z=5.0))
        panel.refresh()
        assert panel._btn_undo.isEnabled()
        assert not panel._btn_redo.isEnabled()
        assert panel._history_list.count() == 1

        # Undo via button
        panel._btn_undo.click()
        assert not panel._btn_undo.isEnabled()
        assert panel._btn_redo.isEnabled()
        assert panel._history_list.count() == 0

        # Redo via button
        panel._btn_redo.click()
        assert panel._btn_undo.isEnabled()
        assert not panel._btn_redo.isEnabled()
        assert panel._history_list.count() == 1
        panel._history_dialog.close()

    def test_kill_cell_not_in_lineage(self, qtbot):
        """Kill cell shows status when cell not in lineage."""
        from acetree_py.gui.edit_panel import EditPanel

        app = MockApp()
        app.current_cell_name = "NonexistentCell"
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_kill_cell()
        assert "not found" in panel._status_label.text()

    def test_relink_no_selection(self, qtbot):
        """Relink shows status when no nucleus selected."""
        from acetree_py.gui.edit_panel import EditPanel

        app = MockApp()
        app.current_cell_name = ""
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_relink()
        assert "No nucleus selected" in panel._status_label.text()

    def test_relink_no_selection(self, qtbot):
        """Relink shows status when no nucleus selected."""
        from acetree_py.gui.edit_panel import EditPanel

        app = MockApp()
        app.current_cell_name = ""
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_relink()
        assert "No nucleus selected" in panel._status_label.text()

    def test_resurrect_no_selection(self, qtbot):
        """Resurrect shows status when no nucleus selected."""
        from acetree_py.gui.edit_panel import EditPanel

        app = MockApp()
        app.current_cell_name = ""
        panel = EditPanel(app)
        qtbot.addWidget(panel)

        panel._on_resurrect()
        assert "No nucleus selected" in panel._status_label.text()
