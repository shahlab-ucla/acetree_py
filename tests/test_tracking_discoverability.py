"""User-facing contracts for finding every supported tracking workflow.

These tests intentionally exercise the labels and entry points a user sees,
not just the underlying tracking registry.  They protect against a release
that contains working trackers but accidentally hides them below a clipped
dock, omits them from the creation wizard, or exposes only internal plugin
identifiers.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QDialog, QMainWindow, QScrollArea

from acetree_py.__main__ import app as cli_app
from acetree_py.gui.app import AceTreeApp
from acetree_py.gui.auto_tracking_dialog import AutoTrackForwardDialog
from acetree_py.gui.dataset_dialog import DatasetCreationDialog
from acetree_py.gui.edit_panel import EditPanel
from acetree_py.gui.global_tracking_dialog import GlobalTrackingDialog
from acetree_py.tracking.registry import build_default_registry


def _combo_items(combo) -> dict[str, str]:
    """Return stable workflow IDs keyed by their displayed labels."""

    return {
        str(combo.itemData(index)): combo.itemText(index)
        for index in range(combo.count())
    }


def test_creation_wizard_tracking_page_is_scrollable_and_names_all_modes(qtbot):
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    dialog._stack.setCurrentWidget(dialog._page4)
    dialog.show()

    assert isinstance(dialog._page4, QScrollArea)
    assert dialog._page4.widgetResizable()
    assert dialog._page4.widget() is not None
    assert not dialog._radio_tracking_manual.isHidden()
    assert not dialog._radio_tracking_auto.isHidden()
    assert "Manual annotation" in dialog._radio_tracking_manual.text()
    assert "whole-movie tracking draft" in dialog._radio_tracking_auto.text()

    workflows = _combo_items(dialog._tracking_workflow_combo)
    assert set(workflows) == {"modern_starrynite", "log_lap", "dog_lap"}
    assert "Modern StarryNite" in workflows["modern_starrynite"]
    assert "LoG" in workflows["log_lap"] and "LAP" in workflows["log_lap"]
    assert "DoG" in workflows["dog_lap"] and "LAP" in workflows["dog_lap"]

    guidance = dialog._tracking_explanation_label.text()
    assert "Tracking menu" in guidance
    assert "Manual Track" in guidance
    assert "Track Selected Cell" in guidance
    assert "Track Whole Movie" in guidance


def test_creation_wizard_cancel_uses_qt6_dialog_code(monkeypatch, qtbot):
    """Closing the native Qt 6 wizard must return cleanly, not dereference dlg.Accepted."""

    del qtbot  # Its fixture guarantees that QApplication already exists.

    class _RejectedCreationDialog:
        def exec_(self):
            return QDialog.Rejected

    monkeypatch.setattr(
        "acetree_py.gui.dataset_dialog.DatasetCreationDialog",
        _RejectedCreationDialog,
    )

    assert AceTreeApp.from_dialog() is None


def test_edit_panel_keeps_all_tracking_buttons_in_a_scrollable_visible_group(qtbot):
    # Building the panel connects callbacks but does not need a loaded dataset.
    panel = EditPanel(SimpleNamespace())
    qtbot.addWidget(panel)
    panel.resize(520, 360)
    panel.show()

    assert isinstance(panel._scroll_area, QScrollArea)
    assert panel._scroll_area.widgetResizable()
    assert panel._scroll_area.widget() is not None

    expected = {
        panel._btn_track: "Manual Track",
        panel._btn_auto_track: "Track Selected Cell",
        panel._btn_global_track: "Track Whole Movie",
    }
    for button, label in expected.items():
        assert label in button.text()
        assert not button.isHidden()
        assert button.isVisibleTo(panel)
        assert button.toolTip().strip()


class _ButtonSpy:
    def __init__(self, calls: Counter[str], name: str) -> None:
        self._calls = calls
        self._name = name

    def click(self) -> None:
        self._calls[self._name] += 1


def test_tracking_menu_has_plain_language_callable_entry_points(qtbot):
    calls: Counter[str] = Counter()
    window = QMainWindow()
    qtbot.addWidget(window)

    edit_panel = SimpleNamespace(
        _btn_track=_ButtonSpy(calls, "manual"),
        _btn_relink=_ButtonSpy(calls, "relink"),
        _on_auto_track_forward=lambda: calls.update(["selected_forward"]),
        _on_global_track=lambda: calls.update(["whole_movie"]),
    )
    acetree = AceTreeApp.__new__(AceTreeApp)
    acetree.viewer = SimpleNamespace(
        window=SimpleNamespace(_qt_window=window),
    )
    acetree._edit_panel = edit_panel
    acetree._show_edit_tracking_panel = lambda: calls.update(["show_panel"])

    acetree._add_tracking_menu_actions()

    actions = acetree._tracking_menu_actions
    assert set(actions) == {
        "manual",
        "selected_forward",
        "whole_movie",
        "relink",
        "show_panel",
    }
    expected_labels = {
        "manual": "Manual Track",
        "selected_forward": "Track Selected Cell Forward",
        "whole_movie": "Track Whole Movie",
        "relink": "Relink Selected Cells",
        "show_panel": "Show Tracking",
    }
    for action_id, label in expected_labels.items():
        action = actions[action_id]
        assert label in action.text()
        assert callable(action.trigger)
        assert action.isEnabled()
        assert action.statusTip().strip()
        action.trigger()

    assert calls == Counter({name: 1 for name in expected_labels})
    assert acetree._tracking_menu.title().replace("&", "") == "Tracking"


def test_global_and_forward_workflow_selectors_expose_the_right_boundaries(qtbot):
    registry = build_default_registry(discover_plugins=False)
    global_dialog = GlobalTrackingDialog(1, 5, registry=registry)
    forward_dialog = AutoTrackForwardDialog(1, 5)
    qtbot.addWidget(global_dialog)
    qtbot.addWidget(forward_dialog)
    global_dialog.resize(760, 520)
    forward_dialog.resize(760, 520)
    global_dialog.show()
    forward_dialog.show()

    global_workflows = _combo_items(global_dialog._workflow_combo)
    forward_workflows = _combo_items(forward_dialog._workflow_combo)

    common = {"modern_starrynite", "log_lap", "dog_lap", "custom"}
    assert set(forward_workflows) == common
    assert set(global_workflows) == common | {"legacy_starrynite_exact"}
    assert "Legacy StarryNite exact replay" in global_workflows[
        "legacy_starrynite_exact"
    ]
    assert "advanced" in global_workflows["legacy_starrynite_exact"].lower()
    assert "legacy_starrynite_exact" not in forward_workflows

    for workflows in (global_workflows, forward_workflows):
        assert "Modern StarryNite" in workflows["modern_starrynite"]
        assert "LoG" in workflows["log_lap"] and "LAP" in workflows["log_lap"]
        assert "DoG" in workflows["dog_lap"] and "LAP" in workflows["dog_lap"]

    # Primary actions are pinned below the scrollable forms, so users never
    # have to discover them by scrolling even at each dialog's minimum size.
    for button in (
        forward_dialog._reset_button,
        forward_dialog._quick_preview_button,
        forward_dialog._preview_button,
    ):
        assert button.isVisibleTo(forward_dialog)
    for button in (
        global_dialog._reset_button,
        global_dialog._detector_preview_button,
        global_dialog._preview_button,
    ):
        assert button.isVisibleTo(global_dialog)


def test_cli_identifies_alpha_build_and_documents_starrynite_presets():
    runner = CliRunner()

    version = runner.invoke(cli_app, ["--version"])
    assert version.exit_code == 0
    assert "AceTree-Py 0.2.0" in version.output
    assert "alpha v2" in version.output

    create_help = runner.invoke(cli_app, ["create", "--help"])
    assert create_help.exit_code == 0
    assert "--tracking" in create_help.output
    assert "starrynite" in create_help.output.lower()
    assert "--starrynite-preset" in create_help.output
    assert "dispim_singleview" in create_help.output
