"""Modeless, multi-instance expression plotting window.

Each window owns an independent cell group, time alignment, styling, and
immutable plot-data snapshot.  The same snapshot drives both Matplotlib and
CSV export so exported values cannot drift from what the user saw.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.cell import Cell
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import (
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]

from ..analysis.expression_measurements import (
    expression_cells_fingerprint,
    legacy_expression_coverage,
)
from ..analysis.expression_plot import (
    DEFAULT_EXPRESSION_CHANNELS,
    ExpressionChannel,
    ExpressionPlotData,
    ExpressionPlotService,
    ExpressionSeriesStyle,
    TimeAxisMode,
    export_expression_plot_csv,
)


_DEFAULT_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_LINE_STYLES = (
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dash-dot", "-."),
    ("Dotted", ":"),
)

_MARKERS = (
    ("None", ""),
    ("Circle", "o"),
    ("Square", "s"),
    ("Triangle", "^"),
    ("Diamond", "D"),
    ("Plus", "+"),
    ("Cross", "x"),
)


if _GUI_AVAILABLE:
    class _ExpressionNavigationToolbar(NavigationToolbar):
        """Matplotlib navigation whose Save action honors data validity."""

        def __init__(self, canvas, parent) -> None:
            super().__init__(canvas, parent)
            self._save_action = next(
                (
                    action
                    for action in self.actions()
                    if "save the figure" in action.toolTip().lower()
                    or action.text().replace("&", "").strip().lower() == "save"
                ),
                None,
            )

        def set_save_enabled(self, enabled: bool) -> None:
            if self._save_action is not None:
                self._save_action.setEnabled(enabled)

        def save_figure(self, *_args) -> None:
            owner = self.parent()
            try:
                owner._exportable_snapshot()
            except (AttributeError, RuntimeError) as error:
                QMessageBox.warning(
                    owner,
                    "Cannot export expression plot",
                    str(error),
                )
                return
            owner._choose_svg_path()


class ExpressionPlotWindow(QWidget):  # type: ignore[misc]
    """Independent expression plot editor and export surface."""

    def __init__(
        self,
        app: AceTreeApp,
        *,
        window_number: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        if not _GUI_AVAILABLE:
            raise ImportError("Expression Plot requires 'acetree-py[gui]'")
        super().__init__(parent)
        self.app = app
        self.window_number = int(window_number)
        self.setWindowFlags(Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(f"AceTree — Expression Plot {self.window_number}")
        self.resize(1180, 760)

        self._cells_by_key: dict[str, Cell] = {}
        self._series_styles: dict[str, ExpressionSeriesStyle] = {}
        self._service = ExpressionPlotService()
        self._plot_data: ExpressionPlotData | None = None
        self._channel_initialized = False
        self._building_style_table = False
        self._last_measure_issue: str | None = None
        self._plot_source_revision: int | None = None
        self._plot_source_fingerprint: str | None = None
        self._stale_artist = None

        self._build_ui()
        self.refresh_cells(preserve_selection=False)
        self.refresh_channels()
        self.refresh_plot()

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(5)

        intro = QLabel(
            "1. Select one or more cells   2. Choose an expression channel and "
            "time axis   3. Customize, then export CSV or SVG"
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self._measure_banner = QWidget()
        banner_layout = QHBoxLayout(self._measure_banner)
        banner_layout.setContentsMargins(8, 5, 8, 5)
        self._measure_message = QLabel()
        self._measure_message.setWordWrap(True)
        self._btn_measure = QPushButton("Run Measure…")
        self._btn_measure.setToolTip(
            "Measure every image channel against the current nuclei, then refresh this plot"
        )
        self._btn_measure.clicked.connect(self._run_measure)
        banner_layout.addWidget(self._measure_message, 1)
        banner_layout.addWidget(self._btn_measure)
        self._measure_banner.setStyleSheet(
            "QWidget { background: #5a4300; border: 1px solid #b58a00; }"
            "QLabel { color: #fff2bd; border: none; }"
        )
        outer.addWidget(self._measure_banner)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_plot_area())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 800])
        outer.addWidget(splitter, 1)

        self._status = QLabel()
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    def _build_controls(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(3, 3, 3, 3)

        cell_group = QGroupBox("1. Cells to plot")
        cell_layout = QVBoxLayout(cell_group)
        self._cell_filter = QLineEdit()
        self._cell_filter.setPlaceholderText("Filter cells by name…")
        self._cell_filter.setClearButtonEnabled(True)
        self._cell_filter.textChanged.connect(self._filter_cells)
        cell_layout.addWidget(self._cell_filter)

        self._cell_list = QListWidget()
        self._cell_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._cell_list.setMinimumHeight(170)
        self._cell_list.itemSelectionChanged.connect(self._on_cell_selection_changed)
        cell_layout.addWidget(self._cell_list)

        cell_buttons = QHBoxLayout()
        self._btn_current = QPushButton("Current cell")
        self._btn_current.setToolTip("Select the active cell from the main AceTree viewer")
        self._btn_current.clicked.connect(self.select_current_cell)
        self._btn_filtered = QPushButton("All filtered")
        self._btn_filtered.setToolTip("Select every cell visible under the current filter")
        self._btn_filtered.clicked.connect(self.select_all_filtered)
        self._btn_subtree = QPushButton("+ Descendants")
        self._btn_subtree.setToolTip("Add all descendants of the selected cells")
        self._btn_subtree.clicked.connect(self.select_descendants)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self._cell_list.clearSelection)
        for button in (
            self._btn_current,
            self._btn_filtered,
            self._btn_subtree,
            self._btn_clear,
        ):
            cell_buttons.addWidget(button)
        cell_layout.addLayout(cell_buttons)
        layout.addWidget(cell_group)

        data_group = QGroupBox("2. Data and time")
        data_form = QFormLayout(data_group)
        self._channel_combo = QComboBox()
        self._channel_combo.setToolTip(
            "Run Measure to make every image channel available here; "
            "legacy files retain only one AT value"
        )
        self._channel_combo.currentIndexChanged.connect(self.refresh_plot)
        data_form.addRow("Y-axis channel", self._channel_combo)
        self._time_combo = QComboBox()
        self._time_combo.addItem("Absolute timepoint", TimeAxisMode.ABSOLUTE.value)
        self._time_combo.addItem("Relative to cell birth", TimeAxisMode.RELATIVE.value)
        self._time_combo.addItem("Normalized lifetime (0–1)", TimeAxisMode.NORMALIZED.value)
        self._time_combo.currentIndexChanged.connect(self.refresh_plot)
        data_form.addRow("Time axis", self._time_combo)
        layout.addWidget(data_group)

        style_group = QGroupBox("3. Series labels and colors")
        style_layout = QVBoxLayout(style_group)
        self._style_table = QTableWidget(0, 3)
        self._style_table.setHorizontalHeaderLabels(["Cell", "Legend label", "Color"])
        self._style_table.verticalHeader().setVisible(False)
        self._style_table.horizontalHeader().setStretchLastSection(True)
        self._style_table.setMinimumHeight(130)
        self._style_table.cellChanged.connect(self._on_style_changed)
        self._style_table.cellDoubleClicked.connect(self._choose_series_color)
        style_layout.addWidget(self._style_table)
        hint = QLabel("Double-click a Color cell to choose a custom series color.")
        hint.setWordWrap(True)
        style_layout.addWidget(hint)
        layout.addWidget(style_group)

        appearance = QGroupBox("4. Plot appearance")
        form = QFormLayout(appearance)
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("Expression by cell")
        self._x_label_edit = QLineEdit()
        self._x_label_edit.setPlaceholderText("Automatic for selected time mode")
        self._y_label_edit = QLineEdit()
        self._y_label_edit.setPlaceholderText("Automatic from selected channel")
        for edit in (self._title_edit, self._x_label_edit, self._y_label_edit):
            edit.editingFinished.connect(self.refresh_plot)
        form.addRow("Title", self._title_edit)
        form.addRow("X label", self._x_label_edit)
        form.addRow("Y label", self._y_label_edit)

        self._line_style_combo = QComboBox()
        for label, value in _LINE_STYLES:
            self._line_style_combo.addItem(label, value)
        self._line_style_combo.currentIndexChanged.connect(self.refresh_plot)
        form.addRow("Line style", self._line_style_combo)

        self._marker_combo = QComboBox()
        for label, value in _MARKERS:
            self._marker_combo.addItem(label, value)
        self._marker_combo.setCurrentIndex(1)
        self._marker_combo.currentIndexChanged.connect(self.refresh_plot)
        form.addRow("Marker", self._marker_combo)

        self._line_width = _double_spin(0.1, 10.0, 1.5, 0.1)
        self._marker_size = _double_spin(0.0, 30.0, 4.0, 0.5)
        self._opacity = _double_spin(0.05, 1.0, 1.0, 0.05)
        self._font_size = _double_spin(6.0, 36.0, 10.0, 1.0)
        self._title_size = _double_spin(6.0, 48.0, 13.0, 1.0)
        for spin in (
            self._line_width,
            self._marker_size,
            self._opacity,
            self._font_size,
            self._title_size,
        ):
            spin.valueChanged.connect(self.refresh_plot)
        form.addRow("Line width", self._line_width)
        form.addRow("Marker size", self._marker_size)
        form.addRow("Opacity", self._opacity)
        form.addRow("Label/tick font", self._font_size)
        form.addRow("Title font", self._title_size)

        self._y_scale = QComboBox()
        self._y_scale.addItem("Linear", "linear")
        self._y_scale.addItem("Logarithmic", "log")
        self._y_scale.currentIndexChanged.connect(self.refresh_plot)
        form.addRow("Y scale", self._y_scale)

        self._legend_check = QCheckBox("Show legend")
        self._legend_check.setChecked(True)
        self._legend_check.toggled.connect(self.refresh_plot)
        form.addRow("Legend", self._legend_check)
        self._legend_location = QComboBox()
        for label, value in (
            ("Best", "best"),
            ("Upper right", "upper right"),
            ("Upper left", "upper left"),
            ("Lower right", "lower right"),
            ("Lower left", "lower left"),
            ("Outside right", "outside"),
        ):
            self._legend_location.addItem(label, value)
        self._legend_location.currentIndexChanged.connect(self.refresh_plot)
        form.addRow("Legend position", self._legend_location)
        self._legend_title = QLineEdit()
        self._legend_title.setPlaceholderText("Optional legend title")
        self._legend_title.editingFinished.connect(self.refresh_plot)
        form.addRow("Legend title", self._legend_title)
        self._legend_columns = QSpinBox()
        self._legend_columns.setRange(1, 12)
        self._legend_columns.setValue(1)
        self._legend_columns.valueChanged.connect(self.refresh_plot)
        form.addRow("Legend columns", self._legend_columns)

        self._grid_check = QCheckBox("Show grid")
        self._grid_check.setChecked(True)
        self._grid_check.toggled.connect(self.refresh_plot)
        form.addRow("Grid", self._grid_check)

        self._auto_x = QCheckBox("Automatic")
        self._auto_x.setChecked(True)
        self._auto_x.toggled.connect(self._on_auto_x_changed)
        form.addRow("X limits", self._auto_x)
        x_limits = QHBoxLayout()
        self._x_min = _double_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self._x_max = _double_spin(-1.0e12, 1.0e12, 1.0, 1.0)
        self._x_min.setEnabled(False)
        self._x_max.setEnabled(False)
        self._x_min.valueChanged.connect(self.refresh_plot)
        self._x_max.valueChanged.connect(self.refresh_plot)
        x_limits.addWidget(QLabel("Min"))
        x_limits.addWidget(self._x_min)
        x_limits.addWidget(QLabel("Max"))
        x_limits.addWidget(self._x_max)
        form.addRow("", x_limits)

        self._auto_y = QCheckBox("Automatic")
        self._auto_y.setChecked(True)
        self._auto_y.toggled.connect(self._on_auto_y_changed)
        form.addRow("Y limits", self._auto_y)
        y_limits = QHBoxLayout()
        self._y_min = _double_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self._y_max = _double_spin(-1.0e12, 1.0e12, 5000.0, 1.0)
        self._y_min.setEnabled(False)
        self._y_max.setEnabled(False)
        self._y_min.valueChanged.connect(self.refresh_plot)
        self._y_max.valueChanged.connect(self.refresh_plot)
        y_limits.addWidget(QLabel("Min"))
        y_limits.addWidget(self._y_min)
        y_limits.addWidget(QLabel("Max"))
        y_limits.addWidget(self._y_max)
        form.addRow("", y_limits)

        backgrounds = QHBoxLayout()
        self._figure_background = "#ffffff"
        self._axes_background = "#ffffff"
        self._text_color = "#202020"
        self._btn_figure_background = QPushButton("Figure…")
        self._btn_figure_background.clicked.connect(self._choose_figure_background)
        self._btn_background = QPushButton("Axes…")
        self._btn_background.clicked.connect(self._choose_axes_background)
        self._btn_text_color = QPushButton("Axis/text color…")
        self._btn_text_color.clicked.connect(self._choose_text_color)
        backgrounds.addWidget(self._btn_figure_background)
        backgrounds.addWidget(self._btn_background)
        backgrounds.addWidget(self._btn_text_color)
        form.addRow("Colors", backgrounds)
        layout.addWidget(appearance)

        refresh = QPushButton("Refresh Plot")
        refresh.setToolTip("Rebuild the plot from current cells, channel, and styles")
        refresh.clicked.connect(self.refresh_plot)
        layout.addWidget(refresh)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setMinimumWidth(340)
        return scroll

    def _build_plot_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(constrained_layout=True)
        self._axes = self._figure.add_subplot(111)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = _ExpressionNavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        exports = QHBoxLayout()
        exports.addStretch(1)
        self._btn_export_csv = QPushButton("Save plotted data as CSV…")
        self._btn_export_csv.clicked.connect(self._choose_csv_path)
        self._btn_export_svg = QPushButton("Export plot as SVG…")
        self._btn_export_svg.clicked.connect(self._choose_svg_path)
        exports.addWidget(self._btn_export_csv)
        exports.addWidget(self._btn_export_svg)
        layout.addLayout(exports)
        return area

    # -- Cells and channels ---------------------------------------------

    def refresh_cells(self, *, preserve_selection: bool = True) -> None:
        selected = set(self.selected_cell_keys()) if preserve_selection else set()
        tree = self.app.manager.lineage_tree
        cells = [] if tree is None else [cell for cell in tree.all_cells() if cell.nuclei]
        cells.sort(key=lambda cell: (cell.name.casefold(), cell.start_time, cell.end_time))
        self._cells_by_key = {_cell_key(cell): cell for cell in cells}

        self._cell_list.blockSignals(True)
        self._cell_list.clear()
        for key, cell in self._cells_by_key.items():
            item = QListWidgetItem(f"{cell.name}   (t{cell.start_time}–{cell.end_time})")
            item.setData(Qt.UserRole, key)
            item.setToolTip(
                f"{cell.name}; birth t={cell.start_time}, end t={cell.end_time}, "
                f"{len(cell.nuclei)} observed timepoint(s)"
            )
            self._cell_list.addItem(item)
            if key in selected:
                item.setSelected(True)

        if not self._cell_list.selectedItems():
            self._select_initial_cell()
        self._cell_list.blockSignals(False)
        self._filter_cells(self._cell_filter.text())
        self._rebuild_style_table()

    def refresh_channels(self) -> None:
        previous = self._channel_combo.currentData()
        channels = list(DEFAULT_EXPRESSION_CHANNELS)
        # Make the compatibility limitation explicit.  A reloaded legacy
        # dataset cannot tell us which physical image channel supplied rweight.
        if channels:
            legacy = channels[0]
            channels[0] = ExpressionChannel(
                key=legacy.key,
                label="Legacy AT expression (stored rweight; channel unknown)",
                reader=legacy.reader,
                unit=legacy.unit,
            )
        measured = getattr(self.app.manager, "expression_measurements", None)
        if measured is not None:
            channels.extend(measured.expression_channels(self.app.manager))
        self._service = ExpressionPlotService(channels)

        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for channel in channels:
            self._channel_combo.addItem(channel.label, channel.key)

        desired = previous
        if not self._channel_initialized and measured is not None:
            desired = f"measured_channel_{measured.at_channel + 1}"
        index = self._channel_combo.findData(desired)
        if index < 0:
            index = 0
        self._channel_combo.setCurrentIndex(index)
        self._channel_combo.blockSignals(False)
        self._channel_initialized = True

    def selected_cell_keys(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(Qt.UserRole))
            for item in self._cell_list.selectedItems()
        )

    def selected_cells(self) -> tuple[Cell, ...]:
        return tuple(
            self._cells_by_key[key]
            for key in self.selected_cell_keys()
            if key in self._cells_by_key
        )

    def select_cells(self, names_or_keys: list[str] | tuple[str, ...]) -> None:
        """Select cells by stable key or display name (useful for scripting/tests)."""

        wanted = set(names_or_keys)
        self._cell_list.blockSignals(True)
        self._cell_list.clearSelection()
        for row in range(self._cell_list.count()):
            item = self._cell_list.item(row)
            key = str(item.data(Qt.UserRole))
            cell = self._cells_by_key.get(key)
            if key in wanted or (cell is not None and cell.name in wanted):
                item.setSelected(True)
        self._cell_list.blockSignals(False)
        self._on_cell_selection_changed()

    def select_current_cell(self) -> None:
        key = self._active_cell_key()
        if key is not None:
            self.select_cells([key])

    def select_all_filtered(self) -> None:
        self._cell_list.blockSignals(True)
        for row in range(self._cell_list.count()):
            item = self._cell_list.item(row)
            if not item.isHidden():
                item.setSelected(True)
        self._cell_list.blockSignals(False)
        self._on_cell_selection_changed()

    def select_descendants(self) -> None:
        wanted = set(self.selected_cell_keys())
        for cell in self.selected_cells():
            wanted.update(_cell_key(child) for child in cell.iter_descendants())
        self.select_cells(tuple(wanted))

    def _select_initial_cell(self) -> None:
        active_key = self._active_cell_key()
        if active_key is None:
            return
        for row in range(self._cell_list.count()):
            item = self._cell_list.item(row)
            if str(item.data(Qt.UserRole)) == active_key:
                item.setSelected(True)
                break

    def _active_cell_key(self) -> str | None:
        anchor = getattr(self.app, "selection_anchor", None)
        if (
            isinstance(anchor, tuple)
            and len(anchor) == 2
            and 1 <= int(anchor[0]) <= len(self.app.manager.nuclei_record)
        ):
            time, index = int(anchor[0]), int(anchor[1])
            nuclei = self.app.manager.nuclei_record[time - 1]
            nucleus = next((item for item in nuclei if item.index == index), None)
            if nucleus is not None:
                for key, cell in self._cells_by_key.items():
                    if cell.get_nucleus_at(time) is nucleus:
                        return key
        active_name = str(getattr(self.app, "current_cell_name", "") or "")
        for key, cell in self._cells_by_key.items():
            if cell.name == active_name:
                return key
        return None

    def _filter_cells(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self._cell_list.count()):
            item = self._cell_list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().casefold())

    def _on_cell_selection_changed(self) -> None:
        self._ensure_default_styles()
        self._rebuild_style_table()
        self.refresh_plot()

    # -- Per-series style table -----------------------------------------

    def _ensure_default_styles(self) -> None:
        for key in self.selected_cell_keys():
            if key in self._series_styles:
                continue
            cell = self._cells_by_key.get(key)
            color = _DEFAULT_COLORS[len(self._series_styles) % len(_DEFAULT_COLORS)]
            self._series_styles[key] = ExpressionSeriesStyle(
                label=cell.name if cell is not None else key,
                color=color,
            )

    def _rebuild_style_table(self) -> None:
        self._ensure_default_styles()
        self._building_style_table = True
        self._style_table.blockSignals(True)
        keys = self.selected_cell_keys()
        self._style_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            cell = self._cells_by_key[key]
            style = self._series_styles[key]
            name_item = QTableWidgetItem(cell.name)
            name_item.setData(Qt.UserRole, key)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            label_item = QTableWidgetItem(style.label or cell.name)
            color_item = QTableWidgetItem(
                style.color or _DEFAULT_COLORS[row % len(_DEFAULT_COLORS)]
            )
            color_item.setBackground(QColor(color_item.text()))
            self._style_table.setItem(row, 0, name_item)
            self._style_table.setItem(row, 1, label_item)
            self._style_table.setItem(row, 2, color_item)
        self._style_table.resizeColumnsToContents()
        self._style_table.blockSignals(False)
        self._building_style_table = False

    def _on_style_changed(self, row: int, column: int) -> None:
        if self._building_style_table or column not in (1, 2):
            return
        key_item = self._style_table.item(row, 0)
        if key_item is None:
            return
        key = str(key_item.data(Qt.UserRole))
        current = self._series_styles.get(key, ExpressionSeriesStyle())
        label_item = self._style_table.item(row, 1)
        color_item = self._style_table.item(row, 2)
        label = label_item.text().strip() if label_item else current.label
        color = color_item.text().strip() if color_item else current.color
        parsed = QColor(color or "")
        if not parsed.isValid():
            color = current.color
            if color_item is not None:
                color_item.setText(color or "")
        elif color_item is not None:
            color = parsed.name(QColor.HexRgb)
            color_item.setText(color)
            color_item.setBackground(parsed)
        self._series_styles[key] = ExpressionSeriesStyle(
            label=label or self._cells_by_key[key].name,
            color=color,
        )
        self.refresh_plot()

    def _choose_series_color(self, row: int, column: int) -> None:
        if column != 2:
            return
        item = self._style_table.item(row, column)
        initial = QColor(item.text() if item is not None else "#1f77b4")
        color = QColorDialog.getColor(initial, self, "Choose series color")
        if color.isValid() and item is not None:
            item.setText(color.name(QColor.HexRgb))

    # -- Plotting and validation ----------------------------------------

    def refresh_plot(self, *_args) -> None:
        cells = self.selected_cells()
        issue = self._measure_issue(cells)
        notice = issue or self._freshness_notice(cells)
        self._last_measure_issue = issue
        self._update_measure_prompt(notice)

        if not cells:
            self._plot_data = None
            self._plot_source_revision = None
            self._plot_source_fingerprint = None
            self._draw_empty("Select one or more cells to generate an expression plot.")
            self._set_export_enabled(False)
            self._status.setText("No cells selected.")
            return
        if self._channel_combo.count() == 0:
            self._plot_data = None
            self._plot_source_revision = None
            self._plot_source_fingerprint = None
            self._draw_empty("No expression channels are available.")
            self._set_export_enabled(False)
            return

        # Once an edit makes a rendered snapshot stale, retain it visibly but
        # never replace/export it as though it matched the new document.
        if (
            issue is not None
            and issue.startswith("STALE:")
            and self._plot_data is not None
        ):
            self._show_stale_overlay()
            self._set_export_enabled(False)
            self._status.setText(
                "Stale plot snapshot retained for reference. Run Measure "
                "before refreshing or exporting."
            )
            return

        channel_key = str(self._channel_combo.currentData())
        mode = TimeAxisMode(str(self._time_combo.currentData()))
        try:
            data = self._service.build(
                cells,
                channel_key,
                mode,
                styles=self._series_styles,
            )
        except Exception as error:  # noqa: BLE001 - surface plugin/reader errors in-window
            logger.exception("Could not build expression plot")
            self._plot_data = None
            self._plot_source_revision = None
            self._plot_source_fingerprint = None
            self._draw_empty(f"Could not build plot: {error}")
            self._set_export_enabled(False)
            return

        self._plot_data = data
        self._plot_source_revision = int(
            getattr(self.app.manager, "data_revision", 0)
        )
        self._plot_source_fingerprint = expression_cells_fingerprint(cells)
        self._draw_data(data)
        export_ok = data.has_data and issue is None
        self._set_export_enabled(export_ok)
        measured_points = sum(
            1 for series in data.series for value in series.y_values if value is not None
        )
        suffix = f" Warning: {notice}" if notice else ""
        self._status.setText(
            f"{len(data.series)} cell series; {measured_points} plotted sample(s).{suffix}"
        )

    def _draw_data(self, data: ExpressionPlotData) -> None:
        axes = self._axes
        axes.clear()
        self._stale_artist = None
        axes.set_facecolor(self._axes_background)
        self._figure.patch.set_facecolor(self._figure_background)

        for series in data.series:
            xs, ys = series.plot_xy
            axes.plot(
                xs,
                ys,
                label=series.label,
                color=series.color,
                linestyle=str(self._line_style_combo.currentData()),
                marker=str(self._marker_combo.currentData()),
                linewidth=self._line_width.value(),
                markersize=self._marker_size.value(),
                alpha=self._opacity.value(),
            )

        title = self._title_edit.text().strip() or "Expression by cell"
        x_label = self._x_label_edit.text().strip() or data.x_label
        y_label = self._y_label_edit.text().strip() or data.y_label
        axes.set_title(title, color=self._text_color, fontsize=self._title_size.value())
        axes.set_xlabel(x_label, color=self._text_color, fontsize=self._font_size.value())
        axes.set_ylabel(y_label, color=self._text_color, fontsize=self._font_size.value())
        axes.set_yscale(str(self._y_scale.currentData()))
        axes.grid(self._grid_check.isChecked(), alpha=0.25)
        axes.tick_params(colors=self._text_color, labelsize=self._font_size.value())
        for spine in axes.spines.values():
            spine.set_color(self._text_color)
        if not self._auto_x.isChecked() and self._x_min.value() < self._x_max.value():
            axes.set_xlim(self._x_min.value(), self._x_max.value())
        if not self._auto_y.isChecked() and self._y_min.value() < self._y_max.value():
            axes.set_ylim(self._y_min.value(), self._y_max.value())
        if self._legend_check.isChecked() and data.series:
            location = str(self._legend_location.currentData())
            legend_options = {
                "title": self._legend_title.text().strip() or None,
                "ncols": self._legend_columns.value(),
                "fontsize": self._font_size.value(),
            }
            if location == "outside":
                legend = axes.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    **legend_options,
                )
            else:
                legend = axes.legend(loc=location, **legend_options)
            if legend is not None:
                legend.get_frame().set_facecolor(self._axes_background)
                for text in legend.get_texts():
                    text.set_color(self._text_color)
                if legend.get_title() is not None:
                    legend.get_title().set_color(self._text_color)
        # Draw synchronously. QtAgg's draw_idle queues a callback that can run
        # after a modeless window has been closed and its C++ canvas deleted.
        self._canvas.draw()

    def _draw_empty(self, message: str) -> None:
        self._axes.clear()
        self._stale_artist = None
        self._axes.set_facecolor(self._axes_background)
        self._figure.patch.set_facecolor(self._figure_background)
        self._axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self._axes.transAxes,
            color=self._text_color,
            wrap=True,
        )
        self._axes.set_axis_off()
        self._canvas.draw()

    def _show_stale_overlay(self) -> None:
        if self._stale_artist is not None:
            return
        self._stale_artist = self._axes.text(
            0.5,
            0.98,
            "STALE SNAPSHOT — RUN MEASURE BEFORE EXPORT",
            ha="center",
            va="top",
            transform=self._axes.transAxes,
            color="#8b0000",
            fontsize=max(10.0, self._font_size.value()),
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#ffd6a5",
                "edgecolor": "#8b0000",
                "alpha": 0.95,
            },
            zorder=1000,
        )
        self._canvas.draw()

    def _measure_issue(self, cells: tuple[Cell, ...]) -> str | None:
        if not cells:
            return None
        manager = self.app.manager
        measured = getattr(manager, "expression_measurements", None)
        revision = int(getattr(manager, "data_revision", 0))
        if revision > 0 and (measured is None or not measured.is_current(manager)):
            return (
                "STALE: the dataset was edited after the last available measurement. "
                "Run Measure again so expression and nuclei geometry match."
            )
        if measured is not None and not measured.is_current(manager):
            return (
                "STALE: these measurements belong to an earlier dataset revision. "
                "Run Measure again before export."
            )
        if measured is not None and not measured.dependencies_current(manager):
            detail = (
                "a nucleus that contributes to neighbor masking changed"
                if measured.correction_method == "blot"
                else "the image calibration changed"
            )
            return (
                f"STALE: {detail} after Measure. Run Measure again before "
                "refreshing or exporting expression data."
            )

        key = str(self._channel_combo.currentData() or "")
        if measured is not None and measured.is_current(manager):
            metric_by_key = {
                "rweight": "value",
                "rwraw": "raw",
                "red_global": "global",
                "red_blot": "blot",
            }
            if key in ("red_local", "red_cross"):
                return (
                    "INCOMPLETE: the current Python Measure run does not produce "
                    "local or cross-talk correction fields. Choose the numbered "
                    "measured channel, raw/global/blot data, or load compatible "
                    "legacy corrections."
                )
            if key in metric_by_key:
                valid, expected = measured.coverage(
                    manager,
                    cells,
                    measured.at_channel,
                    metric=metric_by_key[key],
                )
                if valid < expected:
                    suffix = (
                        " Run Measure with Blot correction selected."
                        if key == "red_blot"
                        else " Run Measure to fill the missing samples."
                    )
                    return (
                        f"INCOMPLETE: the current AT-channel measurement has "
                        f"{valid}/{expected} required samples.{suffix}"
                    )
                return None
        if key.startswith("measured_channel_") and measured is not None:
            try:
                image_channel = int(key.rsplit("_", 1)[1]) - 1
                metric = {
                    "global": "global",
                    "blot": "blot",
                    "local": "global",
                    "cross": "global",
                }.get(measured.correction_method, "value")
                valid, expected = measured.coverage(
                    manager,
                    cells,
                    image_channel,
                    metric=metric,
                )
            except (KeyError, ValueError):
                valid, expected = 0, sum(len(cell.nuclei) for cell in cells)
            if valid < expected:
                return (
                    f"INCOMPLETE: channel data exist for {valid}/{expected} selected "
                    "nuclei. Run Measure to fill missing samples."
                )
            return None

        populated, expected = legacy_expression_coverage(cells, key)
        if populated < expected:
            return (
                f"INCOMPLETE: stored legacy expression appears populated for "
                f"{populated}/{expected} selected nuclei. Run Measure to avoid "
                "treating missing values as biological zeros."
            )
        return None

    def _update_measure_prompt(self, issue: str | None) -> None:
        self._measure_banner.setVisible(issue is not None)
        if issue is None:
            return
        self._measure_message.setText(
            issue.replace("STALE: ", "")
            .replace("INCOMPLETE: ", "")
            .replace("UNVERIFIED: ", "")
        )
        can_measure = getattr(self.app, "image_provider", None) is not None
        self._btn_measure.setEnabled(can_measure)
        if not can_measure:
            self._btn_measure.setToolTip("Load the corresponding image dataset before measuring")

    def _freshness_notice(self, cells: tuple[Cell, ...]) -> str | None:
        manager = self.app.manager
        if (
            cells
            and getattr(manager, "expression_measurements", None) is None
            and not getattr(manager, "expression_measurement_freshness_known", True)
        ):
            return (
                "UNVERIFIED: this legacy dataset stores expression values but "
                "does not record whether Measure ran after its last saved edit. "
                "Remeasure before quantitative comparison when images are available."
            )
        return None

    def _run_measure(self) -> None:
        callback = getattr(self.app, "_on_measure", None)
        if callback is None or getattr(self.app, "image_provider", None) is None:
            QMessageBox.information(
                self,
                "Measure expression",
                "Load the image data for this dataset before running Measure.",
            )
            return
        callback()
        # _on_measure notifies all windows after success. Refresh here as well
        # so cancel/failure leaves the warning state accurate.
        self.refresh_channels()
        self.refresh_plot()

    # -- Appearance helpers and export ----------------------------------

    def _on_auto_y_changed(self, checked: bool) -> None:
        self._y_min.setEnabled(not checked)
        self._y_max.setEnabled(not checked)
        self.refresh_plot()

    def _on_auto_x_changed(self, checked: bool) -> None:
        self._x_min.setEnabled(not checked)
        self._x_max.setEnabled(not checked)
        self.refresh_plot()

    def _choose_figure_background(self) -> None:
        color = QColorDialog.getColor(
            QColor(self._figure_background), self, "Figure background"
        )
        if color.isValid():
            self._figure_background = color.name(QColor.HexRgb)
            self.refresh_plot()

    def _choose_axes_background(self) -> None:
        color = QColorDialog.getColor(QColor(self._axes_background), self, "Plot background")
        if color.isValid():
            self._axes_background = color.name(QColor.HexRgb)
            self.refresh_plot()

    def _choose_text_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._text_color), self, "Axis and text color")
        if color.isValid():
            self._text_color = color.name(QColor.HexRgb)
            self.refresh_plot()

    def _set_export_enabled(self, enabled: bool) -> None:
        self._btn_export_csv.setEnabled(enabled)
        self._btn_export_svg.setEnabled(enabled)
        self._toolbar.set_save_enabled(enabled)

    def _choose_csv_path(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save plotted expression data",
            "expression_plot.csv",
            "CSV files (*.csv)",
        )
        if path:
            try:
                self.export_csv(path)
            except Exception as error:  # noqa: BLE001 - surface filesystem errors
                logger.exception("Could not export expression CSV")
                QMessageBox.warning(self, "Cannot export expression data", str(error))

    def _choose_svg_path(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export expression plot",
            "expression_plot.svg",
            "SVG files (*.svg)",
        )
        if path:
            try:
                self.export_svg(path)
            except Exception as error:  # noqa: BLE001 - surface filesystem errors
                logger.exception("Could not export expression SVG")
                QMessageBox.warning(self, "Cannot export expression plot", str(error))

    def export_csv(self, path: str | Path) -> Path:
        data = self._exportable_snapshot()
        destination = _with_suffix(path, ".csv")
        export_expression_plot_csv(data, destination)
        logger.info("Exported expression plot data to %s", destination)
        return destination

    def export_svg(self, path: str | Path) -> Path:
        self._exportable_snapshot()
        destination = _with_suffix(path, ".svg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(
            destination,
            format="svg",
            bbox_inches="tight",
            facecolor=self._figure.get_facecolor(),
        )
        logger.info("Exported expression plot SVG to %s", destination)
        return destination

    def _exportable_snapshot(self) -> ExpressionPlotData:
        cells = self.selected_cells()
        issue = self._measure_issue(cells)
        current_revision = int(getattr(self.app.manager, "data_revision", 0))
        source_changed = (
            self._plot_source_revision is None
            or current_revision != self._plot_source_revision
            or self._plot_source_fingerprint is None
            or expression_cells_fingerprint(cells) != self._plot_source_fingerprint
        )
        if source_changed and issue is None:
            issue = (
                "STALE: the selected cell data changed after this plot was "
                "rendered. Refresh the plot or run Measure before export."
            )
        self._last_measure_issue = issue
        self._update_measure_prompt(issue or self._freshness_notice(cells))
        if issue is not None:
            self._set_export_enabled(False)
            raise RuntimeError(
                "Expression data are incomplete or stale. Run Measure against the "
                "current dataset before exporting."
            )
        if self._plot_data is None or not self._plot_data.has_data:
            raise RuntimeError("There is no plotted expression data to export.")
        return self._plot_data

    # -- Host notifications/lifecycle -----------------------------------

    def on_document_edited(self, *, structural: bool = True) -> None:
        """Refresh cell identity and visibly invalidate old measurements."""

        if structural:
            self.refresh_cells(preserve_selection=True)
            self.refresh_channels()
        self.refresh_plot()

    def on_measurements_updated(self) -> None:
        """Expose freshly measured channels and clear stale warnings."""

        self.refresh_channels()
        self.refresh_plot()

    def closeEvent(self, event) -> None:
        windows = getattr(self.app, "_expression_plot_windows", None)
        if windows is not None:
            try:
                windows.remove(self)
            except ValueError:
                pass
        super().closeEvent(event)


def _double_spin(
    minimum: float,
    maximum: float,
    value: float,
    step: float,
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(3)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


def _cell_key(cell: Cell) -> str:
    if cell.hash_key:
        return str(cell.hash_key)
    if cell.nuclei:
        time, nucleus = cell.nuclei[0]
        return f"birth:{int(time)}:{int(nucleus.index)}"
    return f"cell:{cell.name}:{cell.start_time}:{cell.end_time}"


def _with_suffix(path: str | Path, suffix: str) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != suffix:
        destination = destination.with_suffix(suffix)
    return destination


__all__ = ["ExpressionPlotWindow"]
