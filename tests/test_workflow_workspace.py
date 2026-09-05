"""One real napari workflow verifies compact layout and retained entry points."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")

napari = pytest.importorskip("napari")
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QApplication, QDockWidget

from acetree_py.core.roi_manager import RoiManager
from acetree_py.core.subcellular_roi import (
    ObjectClass,
    Polygon2D,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
)
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.image_provider import NumpyProvider
from tests.test_gui_widgets import _build_test_manager


def _workspace_app():
    y, x = np.mgrid[:256, :256]
    plane = (12000 * np.exp(-((x - 150) ** 2 + (y - 150) ** 2) / 400)).astype("uint16")
    provider = NumpyProvider(np.broadcast_to(plane, (5, 2, 30, 256, 256)))
    app = AceTreeApp(_build_test_manager(), image_provider=provider)
    app.current_plane = 15
    object_class = ObjectClass("Membrane", (1.0, 0.7, 0.2, 1.0), next_instance_index=2)
    track = RoiObjectTrack(
        class_id=object_class.class_id, instance_index=1,
        frames={1: RoiFrameRecord(
            1, Presence.SEGMENTED, ReviewState.DRAFT,
            Polygon2D(15, ((135, 135), (165, 135), (165, 165), (135, 165))),
        )},
    )
    app.roi_manager = RoiManager(SubcellularRoiDocument(
        object_classes=(object_class,), objects=(track,),
    ))
    return app, track


def _reachable(scroll, widget):
    scroll.ensureWidgetVisible(widget)
    QApplication.processEvents()
    top_left = widget.mapTo(scroll.viewport(), QPoint(0, 0))
    return (
        top_left.x() >= 0 and top_left.y() >= 0
        and top_left.x() + widget.width() <= scroll.viewport().width()
        and top_left.y() + widget.height() <= scroll.viewport().height()
    )


def test_compact_workspace_preserves_selection_actions_menus_and_windows(qtbot, monkeypatch, tmp_path):
    factory = napari.Viewer

    def hidden_viewer(*args, **kwargs):
        kwargs["show"] = False
        viewer = factory(*args, **kwargs)
        viewer.window._qt_window.setAttribute(Qt.WA_DontShowOnScreen, True)
        return viewer

    monkeypatch.setattr(napari, "Viewer", hidden_viewer)
    app, track = _workspace_app()
    app.launch()
    window = app.viewer.window._qt_window
    workspace = app._workspace
    objects = app._subcellular_objects_panel
    try:
        workspace.show_tab("Objects")
        window.resize(1280, 720)
        window.show()
        QApplication.processEvents()
        objects.select_object(track.object_id)
        # A successful save can expose a long authoritative sidecar path.
        # This metadata must not change the available working area.
        app.roi_manager._sidecar_path = Path(
            "C:/research/" + "long_dataset_identifier_" * 15 + ".subcellular-rois.json"
        )
        app.update_display()
        QApplication.processEvents()
        assert (window.width(), window.height()) == (1280, 720)
        assert objects._btn_measure.isVisibleTo(window)
        assert objects._btn_measure.isEnabled()
        assert objects._btn_plot.isVisibleTo(window)
        assert "Membrane #1" in workspace._context.text()
        assert _reachable(objects._scroll_area, objects._btn_delete)
        objects._scroll_area.verticalScrollBar().setValue(0)
        docks = {dock.windowTitle(): dock for dock in window.findChildren(QDockWidget)}
        assert not docks["layer controls"].isVisible()
        assert not docks["layer list"].isVisible()
        # Native layer tools can still be recovered through their existing toggle.
        docks["layer list"].toggleViewAction().trigger()
        assert docks["layer list"].isVisible()
        docks["layer list"].hide()

        objects._btn_review.click()
        assert app.roi_manager.get_object(track.object_id).frames[1].review_state is ReviewState.REVIEWED
        assert workspace._save_state.text() == "Unsaved changes"
        app._edit_panel._btn_undo.click()
        assert app.roi_manager.get_object(track.object_id).frames[1].review_state is ReviewState.DRAFT

        docks["Workflow"].hide()
        app._tracking_menu_actions["show_panel"].trigger()
        assert docks["Workflow"].isVisible()
        assert workspace._tabs.tabText(workspace._tabs.currentIndex()) == "Tracking"
        assert _reachable(workspace._tracking_scroll, app._edit_panel._btn_apply_axes)
        app._edit_panel._btn_track.click()
        assert app._placement_mode and "Manual track" in workspace._context.text()
        app._exit_all_modes()
        assert not app._placement_mode
        app._show_subcellular_objects_panel()
        assert workspace._tabs.tabText(workspace._tabs.currentIndex()) == "Objects"

        workspace.show_tab("Nuclei")
        assert _reachable(app._edit_panel._scroll_area, app._edit_panel._btn_record)
        app._edit_panel._scroll_area.verticalScrollBar().setValue(0)
        assert workspace._measure_nuclei.isVisibleTo(window)
        workspace._plot_expression.click()
        assert len(app._expression_plot_windows) == 1
        expression = app._expression_plot_windows[0]
        assert expression.isVisible()
        workspace.show_tab("Tracking")
        assert expression.isVisible()
        expression.close()
        app._edit_panel._btn_history.click()
        assert app._edit_panel._history_dialog.isVisible()
        app._edit_panel._history_dialog.close()
        workspace.show_tab("Objects")
        QApplication.processEvents()
        assert (window.width(), window.height()) == (1280, 720)
        window.grab().save(str(tmp_path / "workspace-objects.png"))
    finally:
        app.viewer.close()
