"""Compact napari workspace assembly; editing behavior stays in its panels."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def scroll_panel(widget: QWidget, name: str) -> QScrollArea:
    """Let controls keep their natural height without enlarging the window."""
    scroll = QScrollArea()
    scroll.setAccessibleName(name)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(widget)
    return scroll


def hide_default_layer_docks(viewer) -> None:
    """Use napari's dock handles, with a title fallback across versions."""
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    for name in ("dockLayerList", "dockLayerControls"):
        dock = getattr(qt_viewer, name, None)
        if dock is not None:
            dock.hide()
    for dock in viewer.window._qt_window.findChildren(QDockWidget):
        title = dock.windowTitle().replace("&", "").strip().lower()
        if title in {"layer list", "layer controls"}:
            dock.hide()


class WorkflowWorkspace(QWidget):
    """Own shared document controls and the three scientific workflows."""

    TAB_NAMES = ("Nuclei", "Objects", "Tracking")

    def __init__(self, app, edit_panel, objects_panel) -> None:
        super().__init__()
        self.app = app
        self._dock = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        document_row = QHBoxLayout()
        for button in edit_panel._document_actions:
            document_row.addWidget(button)
        edit_panel._document_group.hide()
        edit_panel._scroll_area.widget().layout().removeItem(edit_panel._undo_redo_row)
        layout.addLayout(document_row)

        self._context = QLabel()
        self._context.setWordWrap(True)
        self._context.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._context.setAccessibleName("Current editing target")
        layout.addWidget(self._context)
        self._tabs = QTabWidget()
        self._tabs.setAccessibleName("Scientific workflows")
        self._tabs.setDocumentMode(True)
        edit_panel._heading.hide()
        self._tabs.addTab(edit_panel, "Nuclei")
        objects_panel._context_label.hide()
        objects_panel._mode_label.hide()
        self._tabs.addTab(objects_panel, "Objects")
        tracking = QWidget()
        tracking_layout = QVBoxLayout(tracking)
        tracking_layout.setContentsMargins(8, 8, 8, 8)
        for group in edit_panel._tracking_groups:
            tracking_layout.addWidget(group)
        tracking_layout.addStretch(1)
        self._tracking_scroll = scroll_panel(tracking, "Tracking tools")
        self._tabs.addTab(self._tracking_scroll, "Tracking")
        nuclei_layout = edit_panel._scroll_area.widget().layout()
        nuclei_layout.insertWidget(0, edit_panel._nucleus_group)
        nuclei_layout.insertWidget(1, edit_panel._cell_group)
        analysis = QWidget()
        analysis_row = QHBoxLayout(analysis)
        analysis_row.setContentsMargins(0, 0, 0, 0)
        self._measure_nuclei = QPushButton("Measure nuclei...")
        self._measure_nuclei.setToolTip("Measure nuclear expression from an image channel")
        self._measure_nuclei.clicked.connect(app._on_measure)
        self._plot_expression = QPushButton("Plot expression...")
        self._plot_expression.setToolTip("Open an independent expression plot")
        self._plot_expression.clicked.connect(app.open_expression_plot_window)
        analysis_row.addWidget(self._measure_nuclei)
        analysis_row.addWidget(self._plot_expression)
        nuclei_layout.insertWidget(1, analysis)
        layout.addWidget(self._tabs, 1)
        # Existing status updates and history popup retain their owners/handlers.
        status_row = QHBoxLayout()
        self._save_state = QLabel()
        self._save_state.setAccessibleName("Document save state")
        status_row.addWidget(self._save_state)
        status_row.addWidget(edit_panel._status_label, 1)
        edit_panel._btn_history.setText("History...")
        status_row.addWidget(edit_panel._btn_history)
        layout.addLayout(status_row)
        for button in edit_panel._document_actions:
            button.clicked.connect(self.refresh)
        self._tabs.currentChanged.connect(self.refresh)
        objects_panel.objectSelected.connect(self.refresh)
        objects_panel.modeChanged.connect(self.refresh)
        self.refresh()

    def attach(self, viewer) -> None:
        self._dock = viewer.window.add_dock_widget(
            self, name="Workflow", area="right",
        )

    def show_tab(self, name: str) -> None:
        self._tabs.setCurrentIndex(self.TAB_NAMES.index(name))
        if self._dock is not None:
            self._dock.show()
            self._dock.raise_()
        self.show()
        self.refresh()

    def refresh(self, *_args) -> None:
        app = self.app
        dirty = (
            bool(app.edit_history.modified)
            or bool(getattr(app.manager, "_config_dirty", False))
            or bool(getattr(app.roi_manager, "is_dirty", False))
            or bool(getattr(app, "_nuclear_measurement_unsaved", False))
        )
        save_target = app._default_save_path
        self._save_state.setText(
            "Unsaved changes" if dirty else "Saved" if save_target else "Not saved"
        )
        cell = getattr(app, "current_cell_name", "") or "None"
        mode = "Inspect"
        if getattr(app, "_add_mode", False):
            mode = "Add nucleus (Esc to finish)"
        elif getattr(app, "_placement_mode", False):
            mode = "Manual track (Esc to finish)"
        elif getattr(app, "_relink_pick_mode", False):
            mode = "Pick relink target (Esc to cancel)"
        else:
            roi_mode = getattr(getattr(app, "_subcellular_objects_panel", None), "mode", None)
            value = getattr(roi_mode, "value", "inspect")
            if value != "inspect":
                mode = value.replace("_", " ").capitalize() + " (Esc to cancel)"
        panel = getattr(app, "_subcellular_objects_panel", None)
        row = getattr(panel, "_rows", {}).get(getattr(panel, "current_object_id", None))
        target = f" | Object: {row.label}" if row is not None else ""
        self._context.setText(
            f"Cell: {cell} | t={app.current_time}  z={app.current_plane}{target}\n{mode}"
        )


class BrowseChannelsWorkspace(QSplitter):
    """One adjustable browse surface with independently scrollable channels."""

    def __init__(self, lineage_list, contrast_tools) -> None:
        super().__init__(Qt.Vertical)
        self.setAccessibleName("Browse cells and image channels")
        self.setChildrenCollapsible(False)
        self.addWidget(lineage_list)
        self._channels_scroll = scroll_panel(contrast_tools, "Image channels and contrast")
        self.addWidget(self._channels_scroll)
        self.setSizes((330, 230))
        self.setStretchFactor(0, 2)
        self.setStretchFactor(1, 1)


def arrange_workspace(viewer, browse, workflow, player, lineage) -> None:
    """Give the image canvas useful room and set initial, adjustable dock sizes."""
    window = viewer.window._qt_window
    window.setCorner(Qt.TopLeftCorner, Qt.TopDockWidgetArea)
    window.setCorner(Qt.TopRightCorner, Qt.TopDockWidgetArea)
    window.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
    window.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
    docks = {dock.widget(): dock for dock in window.findChildren(QDockWidget)}
    window.resizeDocks([docks[browse], docks[workflow]], [240, 360], Qt.Horizontal)
    window.resizeDocks([docks[player]], [74], Qt.Vertical)
    window.resizeDocks([docks[lineage]], [150], Qt.Vertical)
