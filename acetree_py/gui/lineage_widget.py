"""Lineage tree widget — QGraphicsView-based Sulston tree renderer.

Renders the lineage tree layout as an interactive QGraphicsView that can
be used as a napari dock widget. Supports:
- Vertical branches with expression-colored segments
- Horizontal connectors at division points
- Cell name labels (rotated for leaves)
- Click to select cell (syncs with image viewer)
- Zoom via mouse wheel
- Export to PNG/SVG

Ported from: org.rhwlab.tree.TreePanel + SulstonTree
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import QPointF, QRectF, Qt, Signal
    from qtpy.QtGui import QColor, QFont, QPainter, QPen, QTransform
    from qtpy.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QGraphicsItem,
        QGraphicsLineItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]

from .lineage_layout import (
    LayoutNode,
    LayoutParams,
    compute_layout,
    compute_tree_bounds,
    expression_to_color,
)

# Default colors
COLOR_BRANCH = QColor(100, 100, 255) if _QT_AVAILABLE else None  # Default branch color
COLOR_SELECTED = QColor(255, 255, 0) if _QT_AVAILABLE else None  # Selected cell highlight
COLOR_DIVISION = QColor(200, 200, 200) if _QT_AVAILABLE else None  # Horizontal connector
COLOR_LABEL = QColor(255, 255, 255) if _QT_AVAILABLE else None  # Label text
COLOR_BACKGROUND = QColor(0, 0, 0) if _QT_AVAILABLE else None  # Scene background
BRANCH_WIDTH = 3.0
SELECTED_WIDTH = 5.0
LABEL_FONT_SIZE = 8


if _QT_AVAILABLE:
    class _ClickableGraphicsView(QGraphicsView):
        """QGraphicsView subclass that handles single-click cell selection.

        ScrollHandDrag mode consumes mousePress for panning, so we detect
        clicks on mouseRelease instead: if the mouse didn't move significantly
        between press and release, treat it as a click.
        """

        def __init__(self, scene, lineage_widget: LineageWidget, parent=None) -> None:
            super().__init__(scene, parent)
            self._lineage_widget = lineage_widget
            self._press_pos = None
            self._press_button = None

        def mousePressEvent(self, event) -> None:
            self._press_pos = event.pos()
            self._press_button = event.button()
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event) -> None:
            super().mouseReleaseEvent(event)
            if self._press_pos is not None:
                delta = event.pos() - self._press_pos
                # Only treat as click if mouse didn't move much (< 5px)
                if (delta.x() ** 2 + delta.y() ** 2) < 25:
                    self._lineage_widget._handle_tree_click(
                        self.mapToScene(event.pos()),
                        self._press_button,
                    )
            self._press_pos = None
            self._press_button = None


class LineageWidget(QWidget):  # type: ignore[misc]
    """Interactive Sulston lineage tree widget.

    Can be added as a napari dock widget. Shows the full lineage tree
    with expression-colored branches and supports click-to-select.

    Supports configurable root cell, time range, expression range, and
    matplotlib colormap for multi-panel usage.

    Layout:
        [Toolbar: zoom controls, settings, export]
        [QGraphicsView showing the tree]
    """

    def __init__(
        self,
        app: AceTreeApp,
        parent=None,
        *,
        root_cell_name: str | None = None,
        time_start: int | None = None,
        time_end: int | None = None,
        expr_min: float = -500.0,
        expr_max: float = 5000.0,
        cmap_name: str | None = None,
    ) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.app = app

        # Panel configuration
        self.root_cell_name: str | None = root_cell_name
        self.time_start: int | None = time_start
        self.time_end: int | None = time_end
        self._expr_min = expr_min
        self._expr_max = expr_max
        self.cmap_name: str | None = cmap_name  # None = legacy green-to-red

        # Layout cache
        self._layout: dict[str, LayoutNode] | None = None
        self._layout_params = LayoutParams()

        # Incremental selection tracking
        self._prev_selected_name: str = ""
        self._cell_items: dict[str, list] = {}  # cell_name -> [QGraphicsItems]
        self._time_indicator_item = None

        self._build_ui()
        self.rebuild_tree()

    def _build_ui(self) -> None:
        """Build the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar
        toolbar = QHBoxLayout()

        self._btn_zoom_in = QPushButton("+")
        self._btn_zoom_in.setFixedWidth(28)
        self._btn_zoom_in.setToolTip("Zoom in")
        self._btn_zoom_in.clicked.connect(lambda: self._zoom(1.25))

        self._btn_zoom_out = QPushButton("−")
        self._btn_zoom_out.setFixedWidth(28)
        self._btn_zoom_out.setToolTip("Zoom out")
        self._btn_zoom_out.clicked.connect(lambda: self._zoom(0.8))

        self._btn_fit = QPushButton("Fit")
        self._btn_fit.setToolTip("Fit tree to view")
        self._btn_fit.clicked.connect(self._fit_to_view)

        self._btn_export = QPushButton("Export")
        self._btn_export.setToolTip("Export tree as image")
        self._btn_export.clicked.connect(self._export)

        self._btn_settings = QPushButton("Settings")
        self._btn_settings.setToolTip("Configure this lineage panel")
        self._btn_settings.clicked.connect(self._open_settings)

        toolbar.addWidget(QLabel("Zoom:"))
        toolbar.addWidget(self._btn_zoom_in)
        toolbar.addWidget(self._btn_zoom_out)
        toolbar.addWidget(self._btn_fit)
        toolbar.addStretch()
        toolbar.addWidget(self._btn_settings)
        toolbar.addWidget(self._btn_export)

        layout.addLayout(toolbar)

        # Graphics view
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(COLOR_BACKGROUND)

        self._view = _ClickableGraphicsView(self._scene, self)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        layout.addWidget(self._view)

    def rebuild_tree(self) -> None:
        """Recompute layout and redraw the entire tree.

        Uses the configured root cell if set, otherwise picks the best root:
        1. P0 (the canonical C. elegans founder cell) if it exists
        2. Otherwise the root with the largest subtree
        """
        tree = self.app.manager.lineage_tree
        if tree is None:
            self._scene.clear()
            return

        # Use configured root or auto-detect
        if self.root_cell_name:
            display_root = tree.get_cell(self.root_cell_name)
            if display_root is None:
                # Configured root no longer exists — fall back
                display_root = self._find_best_root(tree)
        else:
            display_root = self._find_best_root(tree)

        if display_root is None:
            self._scene.clear()
            return

        # Apply time range configuration
        self._layout_params.root_time = self.time_start
        self._layout_params.late_time = self.time_end

        self._layout = compute_layout(display_root, self._layout_params)
        self._redraw()

    def _find_best_root(self, tree):
        """Find the best root cell to display in the graphical tree.

        Prefers P0 (canonical founder), then the largest named subtree,
        then the overall largest subtree.
        """
        # Try P0 first (canonical C. elegans lineage root)
        p0 = tree.get_cell("P0")
        if p0 is not None:
            return p0

        # Find all roots, pick the one with the largest subtree
        best_root = None
        best_size = 0
        for cell in tree.cells_by_name.values():
            if cell.parent is not None:
                continue
            size = sum(1 for _ in cell.iter_subtree_preorder())
            if size > best_size:
                best_size = size
                best_root = cell

        return best_root

    def refresh_selection(self) -> None:
        """Update the selection highlight incrementally.

        Instead of clearing and redrawing all 600+ cells, only redraws
        the previously-selected and newly-selected cells, plus the
        time indicator line.
        """
        if self._layout is None or not self._cell_items:
            # No cached items — need a full redraw (first time or after rebuild)
            self._redraw()
            return

        new_name = self.app.current_cell_name
        old_name = self._prev_selected_name

        if new_name == old_name:
            # Selection unchanged — only update time indicator
            self._update_time_indicator()
            return

        # Redraw only the two affected cells
        if old_name and old_name in self._layout:
            self._redraw_single_cell(old_name, is_selected=False)
        if new_name and new_name in self._layout:
            self._redraw_single_cell(new_name, is_selected=True)

        self._prev_selected_name = new_name
        self._update_time_indicator()

    def _redraw(self) -> None:
        """Full redraw of the tree from the cached layout."""
        if self._layout is None:
            return

        self._scene.clear()
        self._cell_items.clear()
        self._time_indicator_item = None

        for name, node in self._layout.items():
            self._draw_cell(node)

        self._prev_selected_name = self.app.current_cell_name
        self._update_time_indicator()

    def _update_time_indicator(self) -> None:
        """Update the time indicator line position without full redraw."""
        # Remove old indicator
        if self._time_indicator_item is not None:
            self._scene.removeItem(self._time_indicator_item)
            self._time_indicator_item = None

        if self.app.current_time > 0 and self._layout:
            root_time = self._layout_params.root_time
            if root_time is None and self.app.manager.lineage_tree and self.app.manager.lineage_tree.root:
                root_time = self.app.manager.lineage_tree.root.start_time
            if root_time is not None:
                y = self._layout_params.top_margin + \
                    (self.app.current_time - root_time) * self._layout_params.y_scale
                bounds = compute_tree_bounds(self._layout)
                pen = QPen(QColor(255, 255, 0, 80), 1, Qt.DashLine)
                self._time_indicator_item = self._scene.addLine(
                    bounds[0] - 20, y, bounds[2] + 20, y, pen,
                )

    def _redraw_single_cell(self, cell_name: str, is_selected: bool) -> None:
        """Redraw a single cell's graphics items (for selection changes)."""
        # Remove old items for this cell
        for item in self._cell_items.get(cell_name, []):
            self._scene.removeItem(item)
        self._cell_items[cell_name] = []

        node = self._layout.get(cell_name)
        if node is None:
            return

        self._draw_cell_with_state(node, is_selected)

    def _draw_cell(self, node: LayoutNode) -> None:
        """Draw a single cell's branch, label, and connectors."""
        is_selected = (node.cell_name == self.app.current_cell_name)
        self._draw_cell_with_state(node, is_selected)

    def _draw_cell_with_state(self, node: LayoutNode, is_selected: bool) -> None:
        """Draw a single cell, tracking its items for incremental updates."""
        items = []

        # Vertical branch line (with expression coloring)
        if node.y_end > node.y_start:
            items.extend(self._draw_branch(node, is_selected))

        # Division connector (horizontal line to children)
        for child_name, child_x in node.children_x:
            child_node = self._layout.get(child_name)
            if child_node:
                pen = QPen(COLOR_DIVISION, 1.5)
                item = self._scene.addLine(
                    node.x, child_node.y_start,
                    child_x, child_node.y_start,
                    pen,
                )
                items.append(item)

        # Cell name label
        items.append(self._draw_label(node, is_selected))

        # Birth dot
        dot_size = 3
        color = COLOR_SELECTED if is_selected else QColor(150, 150, 255)
        item = self._scene.addEllipse(
            node.x - dot_size / 2, node.y_start - dot_size / 2,
            dot_size, dot_size,
            QPen(Qt.NoPen), color,
        )
        items.append(item)

        self._cell_items[node.cell_name] = items

    def _draw_branch(self, node: LayoutNode, is_selected: bool) -> list:
        """Draw a vertical branch with expression coloring. Returns items."""
        width = SELECTED_WIDTH if is_selected else BRANCH_WIDTH
        items = []

        if not node.expression_values or all(v == 0 for v in node.expression_values):
            # Uniform color
            color = COLOR_SELECTED if is_selected else COLOR_BRANCH
            pen = QPen(color, width)
            items.append(self._scene.addLine(node.x, node.y_start, node.x, node.y_end, pen))
            return items

        # Per-timepoint expression coloring
        n_segments = len(node.expression_values)
        if n_segments == 0:
            return items

        seg_height = (node.y_end - node.y_start) / max(1, n_segments)

        for i, val in enumerate(node.expression_values):
            y0 = node.y_start + i * seg_height
            y1 = y0 + seg_height

            if is_selected:
                color = COLOR_SELECTED
            else:
                r, g, b = expression_to_color(
                    val, self._expr_min, self._expr_max, self.cmap_name,
                )
                color = QColor(int(r * 255), int(g * 255), int(b * 255))

            pen = QPen(color, width)
            items.append(self._scene.addLine(node.x, y0, node.x, y1, pen))
        return items

    def _draw_label(self, node: LayoutNode, is_selected: bool):
        """Draw a cell name label. Returns the text item."""
        font = QFont("Sans Serif", LABEL_FONT_SIZE)
        color = COLOR_SELECTED if is_selected else COLOR_LABEL

        text_item = QGraphicsSimpleTextItem(node.cell_name)
        text_item.setFont(font)
        text_item.setBrush(color)

        if node.is_leaf:
            # Rotate leaf labels 90 degrees, place below branch tip
            text_item.setRotation(90)
            text_item.setPos(node.x + 2, node.y_end + 4)
        else:
            # Angled labels for internal nodes
            text_item.setRotation(-20)
            text_item.setPos(node.x + 4, node.y_start - 12)

        # Make text clickable
        text_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        text_item.setData(0, node.cell_name)  # Store name for click lookup

        self._scene.addItem(text_item)
        return text_item

    def _zoom(self, factor: float) -> None:
        """Zoom the view by a factor."""
        self._view.scale(factor, factor)

    def _fit_to_view(self) -> None:
        """Fit the entire tree into the view."""
        self._view.fitInView(
            self._scene.sceneRect(),
            Qt.KeepAspectRatio,
        )

    def _export(self) -> None:
        """Export the tree to a PNG file."""
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Tree",
            "lineage_tree.png",
            "PNG (*.png);;SVG (*.svg)",
        )
        if not path:
            return

        if path.endswith(".svg"):
            self._export_svg(path)
        else:
            self._export_png(path)

    def _export_png(self, path: str) -> None:
        """Export tree as PNG."""
        from qtpy.QtGui import QImage

        rect = self._scene.sceneRect()
        image = QImage(
            int(rect.width()) + 40,
            int(rect.height()) + 40,
            QImage.Format_ARGB32,
        )
        image.fill(COLOR_BACKGROUND)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        self._scene.render(painter)
        painter.end()

        image.save(path)
        logger.info("Exported tree to %s", path)

    def _export_svg(self, path: str) -> None:
        """Export tree as SVG."""
        try:
            from qtpy.QtSvg import QSvgGenerator
        except ImportError:
            logger.warning("SVG export requires QtSvg; falling back to PNG")
            self._export_png(path.replace(".svg", ".png"))
            return

        rect = self._scene.sceneRect()
        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(rect.size().toSize())
        generator.setViewBox(rect)

        painter = QPainter(generator)
        painter.setRenderHint(QPainter.Antialiasing)
        self._scene.render(painter)
        painter.end()

        logger.info("Exported tree SVG to %s", path)

    def _open_settings(self) -> None:
        """Open the panel configuration dialog."""
        dlg = LineagePanelConfigDialog(self, parent=self)
        if dlg.exec_():
            config = dlg.get_config()
            self.root_cell_name = config["root_cell_name"]
            self.time_start = config["time_start"]
            self.time_end = config["time_end"]
            self._expr_min = config["expr_min"]
            self._expr_max = config["expr_max"]
            self.cmap_name = config["cmap_name"]
            self.rebuild_tree()

    # ── Mouse interaction ─────────────────────────────────────────

    def _handle_tree_click(self, scene_pos, button) -> None:
        """Handle a single click on the tree for cell selection.

        Called by the custom _ClickableGraphicsView on mouseRelease when the
        mouse didn't move (i.e. a genuine click, not a drag/pan).

        Left-click:  Select cell, jump to the timepoint at the y-position
                     clicked (matches Java AceTree left-click behavior).
        Right-click: Select cell, jump to the LAST timepoint of the cell's
                     lifetime (matches Java AceTree right-click behavior).
        """
        # Find nearest cell
        cell_name = self._find_cell_at(scene_pos.x(), scene_pos.y())
        if cell_name:
            cell = self.app.manager.get_cell(cell_name)

            if button == Qt.RightButton:
                # Right-click: jump to end of cell's lifetime
                if cell is not None:
                    self.app.select_cell(cell_name, cell.end_time)
                else:
                    self.app.select_cell(cell_name)
            else:
                # Left-click: jump to timepoint at y-position
                root_time = self._layout_params.root_time
                if root_time is None and self.app.manager.lineage_tree and self.app.manager.lineage_tree.root:
                    root_time = self.app.manager.lineage_tree.root.start_time
                if root_time is not None and self._layout_params.y_scale > 0:
                    time = root_time + int(
                        (scene_pos.y() - self._layout_params.top_margin)
                        / self._layout_params.y_scale
                    )
                    time = max(1, min(time, self.app.manager.num_timepoints))
                    # Clamp to cell's actual lifetime
                    if cell is not None:
                        time = max(cell.start_time, min(time, cell.end_time))
                else:
                    time = None

                self.app.select_cell(cell_name, time)

            self.refresh_selection()

    def _find_cell_at(self, x: float, y: float) -> str | None:
        """Find the cell closest to a point in the tree layout."""
        if self._layout is None:
            return None

        best_name = None
        best_dist = float("inf")
        x_tolerance = self._layout_params.x_scale

        for name, node in self._layout.items():
            # Check if y is within the cell's time range
            if y < node.y_start or y > node.y_end:
                continue

            dx = abs(x - node.x)
            if dx < x_tolerance and dx < best_dist:
                best_dist = dx
                best_name = name

        return best_name

    def wheelEvent(self, event) -> None:
        """Zoom on mouse wheel."""
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom(1.15)
        elif delta < 0:
            self._zoom(0.87)

    def panel_title(self) -> str:
        """Human-readable title for this panel."""
        root = self.root_cell_name or "Full"
        return f"Lineage: {root}"


# ── Available matplotlib colormaps for the dialog ─────────────────

AVAILABLE_CMAPS = [
    "(legacy green-to-red)",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "hot",
    "cool",
    "coolwarm",
    "Spectral",
    "RdYlGn",
    "RdYlBu",
    "RdBu",
    "PiYG",
    "PRGn",
    "turbo",
    "jet",
    "Greys",
    "Reds",
    "Greens",
    "Blues",
    "YlOrRd",
    "YlGnBu",
]


class LineagePanelConfigDialog(QDialog):  # type: ignore[misc]
    """Dialog for configuring a lineage tree panel.

    Allows setting:
    - Root cell name (combobox with all named cells + auto-detect)
    - Time range (start / end spinboxes)
    - Expression color range (min / max)
    - Colormap (matplotlib cmap dropdown)
    """

    def __init__(self, widget: LineageWidget, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lineage Panel Settings")
        self._widget = widget

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Root cell selector
        self._root_combo = QComboBox()
        self._root_combo.setEditable(True)
        self._root_combo.addItem("(auto-detect)")
        tree = widget.app.manager.lineage_tree
        if tree:
            names = sorted(tree.cells_by_name.keys())
            self._root_combo.addItems(names)
        if widget.root_cell_name:
            idx = self._root_combo.findText(widget.root_cell_name)
            if idx >= 0:
                self._root_combo.setCurrentIndex(idx)
            else:
                self._root_combo.setEditText(widget.root_cell_name)
        form.addRow("Root cell:", self._root_combo)

        # Time range
        max_time = widget.app.manager.num_timepoints
        self._time_start = QSpinBox()
        self._time_start.setRange(0, max_time)
        self._time_start.setSpecialValueText("(auto)")
        self._time_start.setValue(widget.time_start if widget.time_start is not None else 0)
        form.addRow("Time start:", self._time_start)

        self._time_end = QSpinBox()
        self._time_end.setRange(0, max_time)
        self._time_end.setSpecialValueText("(auto)")
        self._time_end.setValue(widget.time_end if widget.time_end is not None else 0)
        form.addRow("Time end:", self._time_end)

        # Expression range
        self._expr_min = QDoubleSpinBox()
        self._expr_min.setRange(-100000, 100000)
        self._expr_min.setDecimals(1)
        self._expr_min.setValue(widget._expr_min)
        form.addRow("Expr min:", self._expr_min)

        self._expr_max = QDoubleSpinBox()
        self._expr_max.setRange(-100000, 100000)
        self._expr_max.setDecimals(1)
        self._expr_max.setValue(widget._expr_max)
        form.addRow("Expr max:", self._expr_max)

        # Colormap selector
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(AVAILABLE_CMAPS)
        if widget.cmap_name is None:
            self._cmap_combo.setCurrentIndex(0)  # legacy
        else:
            idx = self._cmap_combo.findText(widget.cmap_name)
            if idx >= 0:
                self._cmap_combo.setCurrentIndex(idx)
        form.addRow("Colormap:", self._cmap_combo)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> dict:
        """Return the configuration as a dictionary."""
        root_text = self._root_combo.currentText().strip()
        root_cell_name = None if root_text in ("(auto-detect)", "") else root_text

        time_start = self._time_start.value()
        time_end = self._time_end.value()

        cmap_text = self._cmap_combo.currentText()
        cmap_name = None if cmap_text == "(legacy green-to-red)" else cmap_text

        return {
            "root_cell_name": root_cell_name,
            "time_start": time_start if time_start > 0 else None,
            "time_end": time_end if time_end > 0 else None,
            "expr_min": self._expr_min.value(),
            "expr_max": self._expr_max.value(),
            "cmap_name": cmap_name,
        }
