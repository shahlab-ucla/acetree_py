"""Qt workflow tests for the modeless Expression Plot window."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    from qtpy.QtWidgets import QMainWindow

    from acetree_py.gui.expression_plot_window import ExpressionPlotWindow

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _GUI_AVAILABLE, reason="Qt/Matplotlib GUI unavailable")

from acetree_py.core.lineage import build_lineage_tree
from acetree_py.analysis.measure_runner import run_measure
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import Nucleus
from acetree_py.editing.commands import MoveNucleus
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.image_provider import NumpyProvider


def _manager(*, complete: bool = True) -> NucleiManager:
    manager = NucleiManager()
    rweights = (100, 200, 300) if complete else (100, 0, 0)
    manager.nuclei_record = [
        [
            Nucleus(
                index=1,
                x=5,
                y=5,
                z=1.0,
                size=6,
                status=1,
                identity="A",
                predecessor=-1 if time == 1 else 1,
                successor1=-1 if time == 3 else 1,
                rweight=rweights[time - 1],
                rwraw=rweights[time - 1],
            )
        ]
        for time in range(1, 4)
    ]
    manager.lineage_tree = build_lineage_tree(
        manager.nuclei_record,
        starting_index=0,
        ending_index=3,
        create_dummy_ancestors=False,
    )
    return manager


def _app(manager: NucleiManager):
    app = SimpleNamespace(
        manager=manager,
        current_cell_name="A",
        image_provider=object(),
        _expression_plot_windows=[],
        _on_measure=lambda: None,
    )
    return app


def _manager_with_neighbor() -> NucleiManager:
    manager = _manager()
    for time, nuclei in enumerate(manager.nuclei_record, start=1):
        nuclei.append(
            Nucleus(
                index=2,
                x=11,
                y=5,
                z=1.0,
                size=6,
                status=1,
                identity="B",
                predecessor=-1 if time == 1 else 2,
                successor1=-1 if time == 3 else 2,
            )
        )
    manager.lineage_tree = build_lineage_tree(
        manager.nuclei_record,
        starting_index=0,
        ending_index=3,
        create_dummy_ancestors=False,
    )
    return manager


def test_active_cell_is_preselected_and_plot_is_exportable(qtbot, tmp_path: Path):
    app = _app(_manager())
    window = ExpressionPlotWindow(app, window_number=2)
    app._expression_plot_windows.append(window)
    qtbot.addWidget(window)

    assert "Expression Plot 2" in window.windowTitle()
    assert [cell.name for cell in window.selected_cells()] == ["A"]
    assert window._plot_data is not None
    assert window._plot_data.series[0].y_values == (100.0, 200.0, 300.0)
    assert window._btn_export_csv.isEnabled()

    csv_path = window.export_csv(tmp_path / "values")
    svg_path = window.export_svg(tmp_path / "plot")
    assert csv_path.suffix == ".csv"
    assert "absolute_time" in csv_path.read_text(encoding="utf-8")
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "Expression by cell" in svg_text


def test_gaussian_smoothing_is_tunable_and_exported_from_main_plot(
    qtbot,
    tmp_path: Path,
):
    window = ExpressionPlotWindow(_app(_manager()))
    qtbot.addWidget(window)
    raw = window._plot_data.series[0].y_values

    window._smooth_sigma.setValue(1.5)
    window._smooth_check.setChecked(True)

    assert window._smooth_sigma.isEnabled()
    assert window._plot_data.smoothing_sigma == pytest.approx(1.5)
    assert window._plot_data.series[0].source_y_values == raw
    assert window._plot_data.series[0].y_values != raw
    csv_path = window.export_csv(tmp_path / "smoothed.csv")
    text = csv_path.read_text(encoding="utf-8")
    assert "raw_value" in text
    assert "smoothing_sigma" in text


def test_incomplete_data_prompts_for_measure_and_prevents_export(qtbot):
    window = ExpressionPlotWindow(_app(_manager(complete=False)))
    qtbot.addWidget(window)

    assert window._measure_banner.isVisible() or not window.isVisible()
    assert window._last_measure_issue is not None
    assert "INCOMPLETE" in window._last_measure_issue
    assert not window._btn_export_csv.isEnabled()
    assert "Run Measure" in window._btn_measure.text()
    with pytest.raises(RuntimeError, match="incomplete or stale"):
        window.export_csv("ignored.csv")


def test_edit_revision_retains_snapshot_but_disables_export(qtbot):
    manager = _manager()
    window = ExpressionPlotWindow(_app(manager))
    qtbot.addWidget(window)
    before = window._plot_data
    assert before is not None

    manager.mark_data_edited()
    window.on_document_edited()

    assert window._plot_data is before
    assert window._last_measure_issue is not None
    assert window._last_measure_issue.startswith("STALE:")
    assert not window._btn_export_svg.isEnabled()
    assert "Stale plot snapshot" in window._status.text()
    assert window._stale_artist is not None
    assert not window._toolbar._save_action.isEnabled()


def test_export_rechecks_revision_without_window_notification(qtbot, tmp_path: Path):
    manager = _manager()
    window = ExpressionPlotWindow(_app(manager))
    qtbot.addWidget(window)
    assert window._btn_export_csv.isEnabled()

    manager.mark_data_edited()

    with pytest.raises(RuntimeError, match="incomplete or stale"):
        window.export_csv(tmp_path / "must_not_exist.csv")
    assert not (tmp_path / "must_not_exist.csv").exists()
    assert not window._toolbar._save_action.isEnabled()


def test_export_rechecks_direct_geometry_mutation(qtbot, tmp_path: Path):
    manager = _manager()
    window = ExpressionPlotWindow(_app(manager))
    qtbot.addWidget(window)
    manager.nuclei_record[0][0].x += 1  # bypass edit history deliberately

    with pytest.raises(RuntimeError, match="incomplete or stale"):
        window.export_svg(tmp_path / "must_not_exist.svg")
    assert not (tmp_path / "must_not_exist.svg").exists()


def test_blot_export_rechecks_unselected_neighbor_geometry(qtbot, tmp_path: Path):
    manager = _manager_with_neighbor()
    provider = NumpyProvider(
        np.full((3, 1, 3, 16, 16), 50, dtype=np.uint16)
    )
    run_measure(manager, provider, tmp_path / "measure", 0, correction_method="blot")
    window = ExpressionPlotWindow(_app(manager))
    qtbot.addWidget(window)
    assert [cell.name for cell in window.selected_cells()] == ["A"]
    assert window._btn_export_csv.isEnabled()

    manager.nuclei_record[0][1].x += 1  # B changed outside edit history

    with pytest.raises(RuntimeError, match="incomplete or stale"):
        window.export_csv(tmp_path / "must_not_exist.csv")
    assert not (tmp_path / "must_not_exist.csv").exists()
    assert window._last_measure_issue is not None
    assert "neighbor masking changed" in window._last_measure_issue


def test_loaded_legacy_freshness_is_prompted_but_complete_data_can_export(qtbot):
    manager = _manager()
    manager.expression_measurement_freshness_known = False
    window = ExpressionPlotWindow(_app(manager))
    qtbot.addWidget(window)

    assert "does not record whether Measure ran" in window._measure_message.text()
    assert window._btn_export_csv.isEnabled()


def test_partial_current_measure_blocks_numbered_and_legacy_at_views(
    qtbot,
    tmp_path: Path,
):
    manager = _manager()
    # Only t=1 exists. Measure emits explicit missing samples for t=2/3 and
    # clears their persisted legacy fields so stale values cannot look current.
    provider = NumpyProvider(
        np.full((1, 1, 3, 16, 16), 50, dtype=np.uint16)
    )
    run_measure(manager, provider, tmp_path, 0, correction_method="none")
    window = ExpressionPlotWindow(_app(manager))
    qtbot.addWidget(window)

    assert window._last_measure_issue is not None
    assert "1/3" in window._last_measure_issue
    legacy_index = window._channel_combo.findData("rweight")
    window._channel_combo.setCurrentIndex(legacy_index)
    window.refresh_plot()
    assert window._last_measure_issue is not None
    assert "1/3" in window._last_measure_issue
    assert not window._btn_export_csv.isEnabled()


def test_close_unregisters_only_that_window(qtbot):
    app = _app(_manager())
    first = ExpressionPlotWindow(app, window_number=1)
    second = ExpressionPlotWindow(app, window_number=2)
    app._expression_plot_windows[:] = [first, second]
    qtbot.addWidget(first)
    qtbot.addWidget(second)

    first.close()
    assert app._expression_plot_windows == [second]


def test_window_menu_action_creates_multiple_distinct_windows(qtbot):
    manager = _manager()
    app = AceTreeApp(manager)
    app.current_cell_name = "A"
    qt_window = QMainWindow()
    qtbot.addWidget(qt_window)
    qt_window.menuBar().addMenu("&Window")
    fake_napari_window = SimpleNamespace(
        _qt_window=qt_window,
        _dock_widgets={},
    )
    app.viewer = SimpleNamespace(window=fake_napari_window)

    app._add_panel_menu_actions()
    action = app._panel_menu_actions["new_expression_plot"]
    assert "Expression Plot" in action.text()
    action.trigger()
    action.trigger()

    assert len(app._expression_plot_windows) == 2
    assert app._expression_plot_windows[0] is not app._expression_plot_windows[1]
    assert (
        app._expression_plot_windows[0].windowTitle()
        != app._expression_plot_windows[1].windowTitle()
    )
    for window in tuple(app._expression_plot_windows):
        window.close()


def test_app_edit_refresh_retry_is_revision_idempotent():
    manager = _manager()
    app = AceTreeApp(manager)
    notifications: list[str] = []
    app._expression_plot_windows.append(
        SimpleNamespace(
            on_document_edited=lambda **_kwargs: notifications.append("edited")
        )
    )
    app.update_display = lambda: None

    app._on_edit()
    # A post-commit display retry carries the same EditHistory token and must
    # not look like a second dataset mutation.
    app._on_edit()

    assert manager.data_revision == 1
    assert notifications == ["edited", "edited"]


def test_real_edit_undo_redo_each_advance_measurement_revision():
    manager = _manager()
    app = AceTreeApp(manager)
    app.update_display = lambda: None

    app.edit_history.do(MoveNucleus(time=1, index=1, new_x=9))
    assert manager.data_revision == 1
    app.edit_history.undo()
    assert manager.data_revision == 2
    app.edit_history.redo()
    assert manager.data_revision == 3


def test_broken_expression_observer_does_not_skip_other_windows_or_main_redraw():
    manager = _manager()
    app = AceTreeApp(manager)
    calls: list[str] = []

    def fail(**_kwargs):
        raise ValueError("plot failed")

    app._expression_plot_windows = [
        SimpleNamespace(on_document_edited=fail),
        SimpleNamespace(
            on_document_edited=lambda **_kwargs: calls.append("second")
        ),
    ]
    app.update_display = lambda: calls.append("main")

    app._on_edit()

    assert calls == ["second", "main"]
