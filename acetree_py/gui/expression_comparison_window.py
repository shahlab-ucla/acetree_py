"""Modeless cross-dataset expression comparison window.

The application owns an :class:`ExpressionDatasetRepository` and may share it
between any number of these windows.  A window owns only presentation state:
which repository datasets are visible, their labels and colours, the exact
cell/channel request, and the immutable comparison snapshot sent to both
Matplotlib and the tidy CSV exporter.

Repository calls are deliberately kept behind ``Prepare included datasets``.
This makes potentially expensive image measurement explicit, exposes cache
reuse, and prevents a plot refresh from unexpectedly reading an embryo movie.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import (
        QApplication,
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QProgressDialog,
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

from ..analysis.expression_comparison import (
    BandStatistic,
    CenterStatistic,
    ComparisonSpec,
    DatasetAcquisitionStatus,
    DatasetExpressionTrace,
    DatasetProvenance,
    ExpressionComparisonData,
    ExpressionComparisonService,
    ExpressionDataset,
    GridDomain,
    GridSpec,
    SmoothingSpec,
    SummarySpec,
    TraceAvailability,
    export_expression_comparison_tidy_csv,
)
from ..analysis.expression_comparison_result import (
    APPEARANCE_INCLUDED_DATASET_IDS,
    EXPRESSION_COMPARISON_RESULT_SUFFIX,
    ExpressionComparisonResult,
    ExpressionComparisonSourceMode,
    build_expression_comparison_data,
    capture_expression_comparison_result,
    revise_expression_comparison_result,
    save_expression_comparison_result,
)
from ..analysis.expression_dataset_repository import (
    CanonicalCellAmbiguousError,
    CanonicalCellNotFoundError,
    DatasetBusyError,
    DatasetSourceChangedError,
    ExpressionDatasetRepository,
    ExpressionDatasetStatus,
    ExpressionChannelUnavailableError,
    ExpressionDataIncompleteError,
    ExpressionTraceSource,
    NativeExpressionTrace,
)
from ..analysis.expression_plot import DEFAULT_EXPRESSION_CHANNELS, TimeAxisMode
from ..core.nucleus import RED_CORRECTIONS
from .expression_plot_window import _ExpressionNavigationToolbar


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

_CORRECTION_LABELS = {
    "none": "None (raw intensity)",
    "global": "Global annulus",
    "local": "Local (global fallback)",
    "blot": "Blot / neighbour-masked annulus",
    "cross": "Cross-talk (global fallback)",
}

_MEAN_BAND_OPTIONS = (
    ("None", BandStatistic.NONE.value),
    ("Sample SD", BandStatistic.SAMPLE_SD.value),
    ("SEM", BandStatistic.SEM.value),
    ("95% Student-t CI", BandStatistic.STUDENT_T_95.value),
)

_MEDIAN_BAND_OPTIONS = (
    ("None", BandStatistic.NONE.value),
    ("IQR", BandStatistic.IQR.value),
    ("Scaled MAD", BandStatistic.SCALED_MAD.value),
)


@dataclass(slots=True)
class _DatasetViewState:
    """Per-window presentation state for one live or frozen dataset."""

    dataset_id: str
    source_uri: str
    included: bool
    label: str
    group_id: str
    color: str
    path: Path | None = None
    repository_status: ExpressionDatasetStatus | None = None
    frozen_dataset: ExpressionDataset | None = None
    trace: NativeExpressionTrace | None = None
    request_key: tuple[object, ...] | None = None
    resolution: str = "unprepared"  # unprepared, ready, acquisition_status, error
    availability: TraceAvailability | None = None
    message: str = "Needs preparation"


class _WindowDataMode(str, Enum):
    LIVE = "live"
    FROZEN = "frozen"


class ExpressionComparisonWindow(QWidget):  # type: ignore[misc]
    """Independent editor and renderer for one multi-dataset comparison."""

    COL_USE = 0
    COL_LABEL = 1
    COL_GROUP = 2
    COL_COLOR = 3
    COL_STATUS = 4
    COL_CACHE = 5
    COL_XML = 6

    def __init__(
        self,
        app: AceTreeApp,
        repository: ExpressionDatasetRepository | None = None,
        window_number: int = 1,
        parent: QWidget | None = None,
        *,
        result: ExpressionComparisonResult | None = None,
        result_path: str | None = None,
    ) -> None:
        if not _GUI_AVAILABLE:
            raise ImportError("Expression Comparison requires 'acetree-py[gui]'")
        super().__init__(parent)
        self.app = app
        if result is None and repository is None:
            raise ValueError("a live comparison requires an expression repository")
        self.repository = repository
        self._data_mode = (
            _WindowDataMode.FROZEN if result is not None else _WindowDataMode.LIVE
        )
        self._portable_result = result
        self._result_path = result_path
        self.window_number = int(window_number)
        self.setWindowFlags(Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setAcceptDrops(True)
        self._update_window_title()
        self.resize(1320, 820)

        self._datasets: dict[str, _DatasetViewState] = {}
        self._service = ExpressionComparisonService()
        self._plot_data: ExpressionComparisonData | None = None
        self._building_table = False
        self._updating_controls = False
        self._preparing = False
        self._close_when_ready = False
        self._active_progress: QProgressDialog | None = None

        self._build_ui()
        if self._data_mode is _WindowDataMode.FROZEN:
            assert result is not None
            self._load_frozen_result(result)
            self._configure_frozen_ui(result)
        else:
            self._sync_source_controls()
            self._on_plot_option_changed()
            self._on_center_changed()
            self._on_show_traces_toggled(True)
            self._add_session_datasets()
            self._add_current_dataset_if_available()
            self._refresh_cell_selector()
        self._refresh_plot()

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(7, 7, 7, 7)
        outer.setSpacing(6)

        self._intro_label = QLabel(
            "Compare one exact cell across independent AceTree XML datasets. "
            "Saved legacy expression is previewable but has unknown channel, "
            "correction, and freshness provenance; image recomputation is cached "
            "in the shared session repository. Comparison reads saved XML/ZIP "
            "snapshots, so save any main-window edits that should be included."
        )
        self._intro_label.setWordWrap(True)
        outer.addWidget(self._intro_label)

        dataset_group = QGroupBox("1. Datasets")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_buttons = QHBoxLayout()
        self._btn_add = QPushButton("Add XMLs…")
        self._btn_add.setToolTip("Add one or more AceTree XML configuration files")
        self._btn_add.clicked.connect(self._choose_dataset_paths)
        self._btn_remove = QPushButton("Remove from this plot")
        self._btn_remove.setToolTip(
            "Remove selected rows from this window; shared measurement caches remain loaded"
        )
        self._btn_remove.clicked.connect(self.remove_selected_datasets)
        self._btn_reload = QPushButton("Reload selected")
        self._btn_reload.setToolTip(
            "Reload selected XML datasets after their source files change; "
            "shared recomputation caches for them are cleared"
        )
        self._btn_reload.clicked.connect(self.reload_selected_datasets)
        self._btn_prepare = QPushButton("Prepare included datasets")
        self._btn_prepare.setToolTip(
            "Extract saved values or, on the first image recompute, read each movie "
            "once and cache every channel and correction for every checked row"
        )
        self._btn_prepare.clicked.connect(self.prepare_included_datasets)
        dataset_buttons.addWidget(self._btn_add)
        dataset_buttons.addWidget(self._btn_remove)
        dataset_buttons.addWidget(self._btn_reload)
        dataset_buttons.addStretch(1)
        dataset_buttons.addWidget(self._btn_prepare)
        dataset_layout.addLayout(dataset_buttons)

        self._dataset_table = QTableWidget(0, 7)
        self._dataset_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Dataset label",
                "Condition / group",
                "Trace color",
                "Data status",
                "Session cache",
                "XML",
            ]
        )
        self._dataset_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._dataset_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._dataset_table.verticalHeader().setVisible(False)
        header = self._dataset_table.horizontalHeader()
        header.setSectionResizeMode(self.COL_USE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_LABEL, QHeaderView.Interactive)
        header.setSectionResizeMode(self.COL_GROUP, QHeaderView.Interactive)
        header.setSectionResizeMode(self.COL_COLOR, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_STATUS, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_CACHE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_XML, QHeaderView.Interactive)
        self._dataset_table.setColumnWidth(self.COL_LABEL, 150)
        self._dataset_table.setColumnWidth(self.COL_GROUP, 135)
        self._dataset_table.setColumnWidth(self.COL_XML, 260)
        self._dataset_table.setMinimumHeight(145)
        self._dataset_table.itemChanged.connect(self._on_dataset_item_changed)
        self._dataset_table.cellDoubleClicked.connect(self._on_dataset_cell_double_clicked)
        dataset_layout.addWidget(self._dataset_table)
        outer.addWidget(dataset_group)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_plot_area())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 900])
        outer.addWidget(splitter, 1)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        outer.addWidget(self._status_label)

    def _build_controls(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(3, 3, 3, 3)

        data_group = QGroupBox("2. Exact cell and expression source")
        data_form = QFormLayout(data_group)
        self._data_form = data_form
        self._cell_combo = QComboBox()
        self._cell_combo.setEditable(True)
        self._cell_combo.setInsertPolicy(QComboBox.NoInsert)
        self._cell_combo.setToolTip(
            "Type or choose one exact, case-sensitive canonical cell name"
        )
        self._cell_combo.currentTextChanged.connect(self._on_trace_request_changed)
        data_form.addRow("Exact cell", self._cell_combo)
        self._cell_availability = QLabel("No datasets loaded")
        self._cell_availability.setWordWrap(True)
        data_form.addRow("Availability", self._cell_availability)

        self._source_combo = QComboBox()
        self._source_combo.addItem("Saved legacy expression", "saved")
        self._source_combo.addItem("Recompute from image channel", "recomputed")
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        data_form.addRow("Expression source", self._source_combo)

        self._saved_channel_combo = QComboBox()
        for channel in DEFAULT_EXPRESSION_CHANNELS:
            self._saved_channel_combo.addItem(channel.label, channel.key)
        self._saved_channel_combo.currentIndexChanged.connect(
            self._on_trace_request_changed
        )
        data_form.addRow("Saved field", self._saved_channel_combo)

        self._image_channel = QSpinBox()
        self._image_channel.setRange(1, 1)
        self._image_channel.setValue(1)
        self._image_channel.setToolTip(
            "The same one-based physical image-channel number is used for every "
            "included dataset; verify that channel ordering is comparable"
        )
        self._image_channel.valueChanged.connect(self._on_trace_request_changed)
        data_form.addRow("Image channel", self._image_channel)

        self._correction_combo = QComboBox()
        for method in RED_CORRECTIONS:
            self._correction_combo.addItem(
                _CORRECTION_LABELS.get(method, method), method
            )
        global_index = self._correction_combo.findData("global")
        if global_index >= 0:
            self._correction_combo.setCurrentIndex(global_index)
        self._correction_combo.currentIndexChanged.connect(
            self._on_trace_request_changed
        )
        data_form.addRow("Correction", self._correction_combo)

        self._legacy_ack = QCheckBox(
            "I understand that legacy values do not identify their physical "
            "channel or correction and may predate the latest saved edits."
        )
        self._legacy_ack.toggled.connect(self._refresh_plot)
        data_form.addRow("Legacy provenance", self._legacy_ack)
        layout.addWidget(data_group)

        alignment_group = QGroupBox("3. Alignment and statistics")
        alignment_form = QFormLayout(alignment_group)
        self._time_combo = QComboBox()
        self._time_combo.addItem("Absolute timepoint", TimeAxisMode.ABSOLUTE.value)
        self._time_combo.addItem("Relative to cell birth", TimeAxisMode.RELATIVE.value)
        self._time_combo.addItem(
            "Normalized lifetime (0–1)", TimeAxisMode.NORMALIZED.value
        )
        self._time_combo.currentIndexChanged.connect(self._on_plot_option_changed)
        alignment_form.addRow("Time axis", self._time_combo)

        self._grid_domain = QComboBox()
        self._grid_domain.addItem("Union of lifetimes", GridDomain.UNION.value)
        self._grid_domain.addItem(
            "Shared intersection only", GridDomain.INTERSECTION.value
        )
        self._grid_domain.currentIndexChanged.connect(self._on_plot_option_changed)
        alignment_form.addRow("Grid domain", self._grid_domain)

        self._grid_step = _double_spin(0.001, 1.0e9, 1.0, 1.0, decimals=3)
        self._grid_step.valueChanged.connect(self._on_plot_option_changed)
        alignment_form.addRow("Grid step", self._grid_step)
        self._normalized_points = QSpinBox()
        self._normalized_points.setRange(2, 10001)
        self._normalized_points.setValue(101)
        self._normalized_points.valueChanged.connect(self._on_plot_option_changed)
        alignment_form.addRow("Normalized points", self._normalized_points)

        self._show_traces = QCheckBox("Show individual dataset traces")
        self._show_traces.setChecked(True)
        self._show_traces.toggled.connect(self._on_show_traces_toggled)
        alignment_form.addRow("Traces", self._show_traces)
        self._trace_opacity = _double_spin(0.05, 1.0, 0.35, 0.05)
        self._trace_opacity.valueChanged.connect(self._refresh_plot)
        alignment_form.addRow("Trace opacity", self._trace_opacity)

        self._center_combo = QComboBox()
        self._center_combo.addItem("None", CenterStatistic.NONE.value)
        self._center_combo.addItem("Mean", CenterStatistic.MEAN.value)
        self._center_combo.addItem("Median", CenterStatistic.MEDIAN.value)
        self._center_combo.setCurrentIndex(1)
        self._center_combo.currentIndexChanged.connect(self._on_center_changed)
        alignment_form.addRow("Center line", self._center_combo)

        self._band_combo = QComboBox()
        self._populate_band_options()
        self._band_combo.currentIndexChanged.connect(self._on_band_changed)
        alignment_form.addRow("Error band", self._band_combo)

        self._band_opacity = _double_spin(0.0, 1.0, 0.22, 0.05)
        self._band_opacity.valueChanged.connect(self._refresh_plot)
        alignment_form.addRow("Band opacity", self._band_opacity)
        self._smoothing_check = QCheckBox("Apply Gaussian smoothing")
        self._smoothing_check.toggled.connect(self._on_smoothing_toggled)
        self._smoothing_sigma = _double_spin(0.01, 1.0e6, 1.0, 0.25)
        self._smoothing_sigma.setEnabled(False)
        self._smoothing_sigma.setToolTip(
            "Gaussian sigma in displayed-axis units; smoothing never crosses gaps"
        )
        self._smoothing_sigma.valueChanged.connect(self._on_plot_option_changed)
        alignment_form.addRow("Smoothing", self._smoothing_check)
        alignment_form.addRow("Gaussian sigma", self._smoothing_sigma)
        layout.addWidget(alignment_group)

        appearance_group = QGroupBox("4. Plot appearance")
        appearance_form = QFormLayout(appearance_group)
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("Expression comparison")
        self._x_label_edit = QLineEdit()
        self._x_label_edit.setPlaceholderText("Automatic from time mode")
        self._y_label_edit = QLineEdit()
        self._y_label_edit.setPlaceholderText("Automatic from expression channel")
        for edit in (self._title_edit, self._x_label_edit, self._y_label_edit):
            edit.editingFinished.connect(self._refresh_plot)
        appearance_form.addRow("Title", self._title_edit)
        appearance_form.addRow("X label", self._x_label_edit)
        appearance_form.addRow("Y label", self._y_label_edit)

        self._trace_line_style = QComboBox()
        self._center_line_style = QComboBox()
        for label, value in _LINE_STYLES:
            self._trace_line_style.addItem(label, value)
            self._center_line_style.addItem(label, value)
        self._trace_line_style.currentIndexChanged.connect(self._refresh_plot)
        self._center_line_style.currentIndexChanged.connect(self._refresh_plot)
        appearance_form.addRow("Trace line", self._trace_line_style)
        appearance_form.addRow("Center line", self._center_line_style)

        self._marker_combo = QComboBox()
        for label, value in _MARKERS:
            self._marker_combo.addItem(label, value)
        self._marker_combo.currentIndexChanged.connect(self._refresh_plot)
        appearance_form.addRow("Trace marker", self._marker_combo)
        self._trace_width = _double_spin(0.1, 12.0, 1.0, 0.1)
        self._center_width = _double_spin(0.1, 16.0, 2.5, 0.1)
        self._marker_size = _double_spin(0.0, 30.0, 3.0, 0.5)
        self._font_size = _double_spin(6.0, 36.0, 10.0, 1.0)
        self._title_size = _double_spin(6.0, 48.0, 13.0, 1.0)
        for spin in (
            self._trace_width,
            self._center_width,
            self._marker_size,
            self._font_size,
            self._title_size,
        ):
            spin.valueChanged.connect(self._refresh_plot)
        appearance_form.addRow("Trace width", self._trace_width)
        appearance_form.addRow("Center width", self._center_width)
        appearance_form.addRow("Marker size", self._marker_size)
        appearance_form.addRow("Label/tick font", self._font_size)
        appearance_form.addRow("Title font", self._title_size)

        self._legend_check = QCheckBox("Show legend")
        self._legend_check.setChecked(True)
        self._legend_check.toggled.connect(self._refresh_plot)
        self._grid_check = QCheckBox("Show grid")
        self._grid_check.setChecked(True)
        self._grid_check.toggled.connect(self._refresh_plot)
        appearance_form.addRow("Legend", self._legend_check)
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
        self._legend_location.currentIndexChanged.connect(self._refresh_plot)
        appearance_form.addRow("Legend position", self._legend_location)
        self._legend_title = QLineEdit()
        self._legend_title.setPlaceholderText("Optional legend title")
        self._legend_title.editingFinished.connect(self._refresh_plot)
        appearance_form.addRow("Legend title", self._legend_title)
        self._legend_columns = QSpinBox()
        self._legend_columns.setRange(1, 12)
        self._legend_columns.setValue(1)
        self._legend_columns.valueChanged.connect(self._refresh_plot)
        appearance_form.addRow("Legend columns", self._legend_columns)
        appearance_form.addRow("Grid", self._grid_check)
        self._y_scale = QComboBox()
        self._y_scale.addItem("Linear", "linear")
        self._y_scale.addItem("Logarithmic", "log")
        self._y_scale.currentIndexChanged.connect(self._refresh_plot)
        appearance_form.addRow("Y scale", self._y_scale)

        self._auto_x = QCheckBox("Automatic")
        self._auto_x.setChecked(True)
        self._auto_x.toggled.connect(self._on_auto_x_changed)
        appearance_form.addRow("X limits", self._auto_x)
        x_limits = QHBoxLayout()
        self._x_min = _double_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self._x_max = _double_spin(-1.0e12, 1.0e12, 1.0, 1.0)
        self._x_min.setEnabled(False)
        self._x_max.setEnabled(False)
        self._x_min.valueChanged.connect(self._refresh_plot)
        self._x_max.valueChanged.connect(self._refresh_plot)
        x_limits.addWidget(QLabel("Min"))
        x_limits.addWidget(self._x_min)
        x_limits.addWidget(QLabel("Max"))
        x_limits.addWidget(self._x_max)
        appearance_form.addRow("", x_limits)

        self._auto_y = QCheckBox("Automatic")
        self._auto_y.setChecked(True)
        self._auto_y.toggled.connect(self._on_auto_y_changed)
        appearance_form.addRow("Y limits", self._auto_y)
        y_limits = QHBoxLayout()
        self._y_min = _double_spin(-1.0e12, 1.0e12, 0.0, 1.0)
        self._y_max = _double_spin(-1.0e12, 1.0e12, 5000.0, 1.0)
        self._y_min.setEnabled(False)
        self._y_max.setEnabled(False)
        self._y_min.valueChanged.connect(self._refresh_plot)
        self._y_max.valueChanged.connect(self._refresh_plot)
        y_limits.addWidget(QLabel("Min"))
        y_limits.addWidget(self._y_min)
        y_limits.addWidget(QLabel("Max"))
        y_limits.addWidget(self._y_max)
        appearance_form.addRow("", y_limits)

        self._figure_background = "#ffffff"
        self._axes_background = "#ffffff"
        self._text_color = "#202020"
        colors = QHBoxLayout()
        self._btn_figure_background = QPushButton("Figure…")
        self._btn_figure_background.clicked.connect(self._choose_figure_background)
        self._btn_background = QPushButton("Axes…")
        self._btn_background.clicked.connect(self._choose_axes_background)
        self._btn_text_color = QPushButton("Axis/text…")
        self._btn_text_color.clicked.connect(self._choose_text_color)
        colors.addWidget(self._btn_figure_background)
        colors.addWidget(self._btn_background)
        colors.addWidget(self._btn_text_color)
        appearance_form.addRow("Colors", colors)
        layout.addWidget(appearance_group)

        refresh = QPushButton("Refresh plot")
        refresh.clicked.connect(self._refresh_plot)
        layout.addWidget(refresh)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setMinimumWidth(370)
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
        self._btn_open_result = QPushButton("Open frozen result…")
        self._btn_open_result.setToolTip(
            "Open an .aceexpr capture in a new, source-independent comparison window"
        )
        self._btn_open_result.clicked.connect(self._open_frozen_result)
        self._btn_save_result = QPushButton("Save portable result…")
        self._btn_save_result.setToolTip(
            "Save native traces, statuses, provenance, settings, and plot appearance"
        )
        self._btn_save_result.clicked.connect(self._choose_result_path)
        self._btn_export_csv = QPushButton("Save exact comparison CSV…")
        self._btn_export_svg = QPushButton("Export plot as SVG…")
        self._btn_export_csv.clicked.connect(self._choose_csv_path)
        self._btn_export_svg.clicked.connect(self._choose_svg_path)
        exports.addWidget(self._btn_open_result)
        exports.addWidget(self._btn_save_result)
        exports.addStretch(1)
        exports.addWidget(self._btn_export_csv)
        exports.addWidget(self._btn_export_svg)
        layout.addLayout(exports)
        return area

    def _update_window_title(self) -> None:
        if self._data_mode is _WindowDataMode.FROZEN:
            name = _display_filename(self._result_path) or "unsaved result"
            self.setWindowTitle(
                f"AceTree — Frozen Expression Result: {name} "
                f"({self.window_number})"
            )
            return
        self.setWindowTitle(f"AceTree — Expression Comparison {self.window_number}")

    def _load_frozen_result(self, result: ExpressionComparisonResult) -> None:
        """Materialise a portable result without touching repository/source paths."""

        if len(result.spec.cell_names) != 1:
            raise ValueError(
                "Expression Comparison currently opens portable results containing "
                "exactly one canonical cell."
            )
        appearance = result.appearance
        included_value = appearance.get(APPEARANCE_INCLUDED_DATASET_IDS)
        included_ids = (
            {str(value) for value in included_value}
            if isinstance(included_value, (tuple, list))
            else {dataset.provenance.dataset_id for dataset in result.datasets}
        )
        overrides = appearance.get("dataset_overrides")
        if not isinstance(overrides, Mapping):
            overrides = {}

        for index, dataset in enumerate(result.datasets):
            provenance = dataset.provenance
            override = overrides.get(provenance.dataset_id, {})
            if not isinstance(override, Mapping):
                override = {}
            trace = dataset.traces[0] if dataset.traces else None
            color = trace.color if trace is not None and trace.color else None
            candidate_color = override.get("color", color)
            if not _valid_color(candidate_color):
                candidate_color = _DEFAULT_COLORS[index % len(_DEFAULT_COLORS)]
            label = _nonblank_string(override.get("label"), provenance.label)
            group = _nonblank_string(override.get("group_id"), provenance.group_id)
            included = provenance.dataset_id in included_ids
            if type(override.get("included")) is bool:
                included = bool(override["included"])
            if dataset.traces:
                resolution = "ready"
                availability = None
                message = "Frozen native values — source files are not required"
            elif dataset.acquisition_statuses:
                resolution = "acquisition_status"
                availability = dataset.acquisition_statuses[0].availability
                message = dataset.acquisition_statuses[0].message
            else:
                resolution = "error"
                availability = None
                message = "Frozen result contains neither values nor an acquisition status"
            self._datasets[provenance.dataset_id] = _DatasetViewState(
                dataset_id=provenance.dataset_id,
                source_uri=provenance.source_uri,
                included=included,
                label=label,
                group_id=group,
                color=str(candidate_color),
                frozen_dataset=dataset,
                request_key=("frozen", result.result_id),
                resolution=resolution,
                availability=availability,
                message=message,
            )

        self._apply_result_controls(result)
        self._rebuild_dataset_table()

    def _apply_result_controls(self, result: ExpressionComparisonResult) -> None:
        self._updating_controls = True
        try:
            spec = result.spec
            self._cell_combo.clear()
            self._cell_combo.addItem(spec.cell_names[0])
            self._cell_combo.setCurrentIndex(0)
            source_mode = result.source_mode.value
            if self._source_combo.findData(source_mode) < 0:
                self._source_combo.addItem("Mixed captured sources", source_mode)
            _select_combo_data(self._source_combo, source_mode)
            metadata = result.acquisition_metadata
            saved_channel = metadata.get("saved_channel_key")
            if isinstance(saved_channel, str):
                _select_combo_data(self._saved_channel_combo, saved_channel)
            image_channel = metadata.get("image_channel_one_based")
            if type(image_channel) is int and image_channel >= 1:
                self._image_channel.setMaximum(
                    max(self._image_channel.maximum(), image_channel)
                )
                self._image_channel.setValue(image_channel)
            correction = metadata.get("correction_method")
            if isinstance(correction, str):
                _select_combo_data(self._correction_combo, correction)
            _select_combo_data(self._time_combo, spec.time_mode.value)
            _select_combo_data(self._grid_domain, spec.grid.domain.value)
            if spec.grid.step is not None:
                self._grid_step.setValue(spec.grid.step)
                if not math.isclose(
                    self._grid_step.value(),
                    spec.grid.step,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Grid step {spec.grid.step!r} is outside the range or "
                        "precision supported by this AceTree UI."
                    )
            elif spec.time_mode is not TimeAxisMode.NORMALIZED:
                raise ValueError(
                    "Automatic grid step is not editable in this AceTree UI; "
                    "save the result with an explicit step."
                )
            self._normalized_points.setValue(spec.grid.normalized_points)
            if self._normalized_points.value() != spec.grid.normalized_points:
                raise ValueError(
                    f"Normalized point count {spec.grid.normalized_points!r} is "
                    "outside the range supported by this AceTree UI."
                )
            self._smoothing_check.setChecked(spec.smoothing.sigma > 0)
            if spec.smoothing.sigma > 0:
                self._smoothing_sigma.setValue(spec.smoothing.sigma)
                if not math.isclose(
                    self._smoothing_sigma.value(),
                    spec.smoothing.sigma,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Smoothing sigma {spec.smoothing.sigma!r} is outside the "
                        "range or precision supported by this AceTree UI."
                    )
            _select_combo_data(self._center_combo, spec.summary.center.value)
            self._populate_band_options(previous=spec.summary.band.value)
            _select_combo_data(self._band_combo, spec.summary.band.value)
            self._apply_appearance(result.appearance)
            self._on_plot_option_changed()
            self._on_center_changed()
            self._on_show_traces_toggled(self._show_traces.isChecked())
            self._on_smoothing_toggled(self._smoothing_check.isChecked())
            self._on_auto_x_changed(self._auto_x.isChecked())
            self._on_auto_y_changed(self._auto_y.isChecked())
        finally:
            self._updating_controls = False

    def _apply_appearance(self, appearance: Mapping[str, Any]) -> None:
        """Apply known appearance keys defensively; unknown/future keys are ignored."""

        bool_controls = {
            "show_traces": self._show_traces,
            "show_legend": self._legend_check,
            "show_grid": self._grid_check,
            "auto_x": self._auto_x,
            "auto_y": self._auto_y,
        }
        for key, control in bool_controls.items():
            value = appearance.get(key)
            if type(value) is bool:
                control.setChecked(value)
        text_controls = {
            "title": self._title_edit,
            "x_label": self._x_label_edit,
            "y_label": self._y_label_edit,
            "legend_title": self._legend_title,
        }
        for key, control in text_controls.items():
            value = appearance.get(key)
            if isinstance(value, str):
                control.setText(value)
        combo_controls = {
            "trace_line_style": self._trace_line_style,
            "center_line_style": self._center_line_style,
            "marker": self._marker_combo,
            "legend_location": self._legend_location,
            "y_scale": self._y_scale,
        }
        for key, control in combo_controls.items():
            value = appearance.get(key)
            if isinstance(value, str):
                _select_combo_data(control, value)
        numeric_controls = {
            "trace_opacity": self._trace_opacity,
            "band_opacity": self._band_opacity,
            "trace_width": self._trace_width,
            "center_width": self._center_width,
            "marker_size": self._marker_size,
            "font_size": self._font_size,
            "title_size": self._title_size,
            "legend_columns": self._legend_columns,
            "x_min": self._x_min,
            "x_max": self._x_max,
            "y_min": self._y_min,
            "y_max": self._y_max,
        }
        for key, control in numeric_controls.items():
            value = appearance.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    control.setValue(value)
                except (TypeError, ValueError, OverflowError):
                    pass
        for key, attribute in (
            ("figure_background", "_figure_background"),
            ("axes_background", "_axes_background"),
            ("text_color", "_text_color"),
        ):
            value = appearance.get(key)
            if _valid_color(value):
                setattr(self, attribute, str(value))

    def _configure_frozen_ui(self, result: ExpressionComparisonResult) -> None:
        self._intro_label.setText(
            "FROZEN RESULT — all native expression values and acquisition statuses "
            "are embedded in this portable capture. Original XML/image paths below "
            "are provenance only and are never opened or validated."
        )
        self._intro_label.setStyleSheet(
            "QLabel { background: #e8f2ff; border: 1px solid #4d86b8; "
            "padding: 6px; font-weight: bold; }"
        )
        self._dataset_table.setHorizontalHeaderItem(
            self.COL_CACHE, QTableWidgetItem("Frozen capture")
        )
        self._dataset_table.setHorizontalHeaderItem(
            self.COL_XML, QTableWidgetItem("Original source (provenance only)")
        )
        for widget in (
            self._btn_add,
            self._btn_remove,
            self._btn_reload,
            self._btn_prepare,
        ):
            widget.setVisible(False)
            widget.setEnabled(False)
        for widget in (
            self._cell_combo,
            self._source_combo,
            self._saved_channel_combo,
            self._image_channel,
            self._correction_combo,
            self._legacy_ack,
        ):
            widget.setEnabled(False)
        mode = result.source_mode
        row_visibility = {
            self._saved_channel_combo: mode is ExpressionComparisonSourceMode.SAVED,
            self._image_channel: mode is ExpressionComparisonSourceMode.RECOMPUTED,
            self._correction_combo: mode is ExpressionComparisonSourceMode.RECOMPUTED,
        }
        for widget, visible in row_visibility.items():
            widget.setVisible(visible)
            label = self._data_form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        self._legacy_ack.setVisible(False)
        legacy = "yes" if result.legacy_acknowledged else "no"
        self._cell_availability.setText(
            f"Capture {result.result_id[:8]} · captured {result.captured_at} · "
            f"source mode {result.source_mode.value} · producer {result.producer_version} · "
            f"legacy acknowledgement recorded: {legacy}"
        )
        self._update_window_title()

    def _open_frozen_result(self) -> None:
        opener = getattr(self.app, "open_expression_comparison_result_window", None)
        if callable(opener):
            opener()
            return
        QMessageBox.warning(
            self,
            "Cannot open result",
            "This application instance does not expose the portable-result opener.",
        )

    # -- Dataset membership ---------------------------------------------

    def _add_session_datasets(self) -> None:
        """Prepopulate later windows from the shared application repository."""

        if self._data_mode is _WindowDataMode.FROZEN:
            return
        assert self.repository is not None
        try:
            session_statuses = getattr(self.repository, "session_statuses", None)
            statuses = (
                session_statuses()
                if callable(session_statuses)
                else self.repository.statuses()
            )
        except Exception:  # noqa: BLE001 - users can still add/reload explicitly
            logger.exception("Could not list cached expression datasets")
            return
        for cached_status in statuses:
            stale_message = ""
            try:
                status = self.repository.status(cached_status.config_path)
            except DatasetSourceChangedError as error:
                status = cached_status
                stale_message = (
                    f"Source changed: {error}. Select this row and click "
                    "Reload selected."
                )
            except Exception as error:  # noqa: BLE001 - retain a recoverable row
                status = cached_status
                stale_message = f"Cannot validate source: {error}"
            self._insert_dataset_status(status, stale_message=stale_message)
        if statuses:
            self._rebuild_dataset_table()

    def _add_current_dataset_if_available(self) -> None:
        if self._data_mode is _WindowDataMode.FROZEN:
            return
        manager = getattr(self.app, "manager", None)
        config = getattr(manager, "config", None)
        path = getattr(config, "config_file", None)
        if path is None:
            return
        candidate = Path(path)
        if candidate.suffix.lower() == ".xml" and candidate.is_file():
            self.add_dataset_paths([candidate], show_errors=False)
            state = self._datasets.get(_path_key(candidate))
            if state is not None and self._current_dataset_has_unsaved_edits(state):
                state.trace = None
                state.request_key = None
                state.resolution = "error"
                state.message = (
                    "Main viewer has unsaved edits. Save it, then select this "
                    "row and click Reload selected before preparing."
                )
                self._rebuild_dataset_table()

    def _choose_dataset_paths(self) -> None:
        if self._data_mode is _WindowDataMode.FROZEN:
            return
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Add AceTree expression datasets",
            "",
            "AceTree XML datasets (*.xml)",
        )
        if paths:
            self.add_dataset_paths(paths)

    def add_dataset_paths(
        self,
        paths: Iterable[str | Path],
        *,
        show_errors: bool = True,
    ) -> int:
        """Add and locally deduplicate XMLs; repository ownership is shared."""

        if self._data_mode is _WindowDataMode.FROZEN:
            return 0
        assert self.repository is not None
        added = 0
        errors: list[str] = []
        for raw_path in paths:
            stale_message = ""
            try:
                status = self.repository.load_dataset(raw_path)
            except DatasetSourceChangedError as error:
                try:
                    status = self.repository.session_status(raw_path)
                except Exception:
                    logger.exception(
                        "Could not expose stale expression dataset %s", raw_path
                    )
                    errors.append(f"{raw_path}: {error}")
                    continue
                stale_message = (
                    f"Source changed: {error}. Select this row and click "
                    "Reload selected."
                )
            except Exception as error:  # noqa: BLE001 - one bad XML must not abort a batch
                logger.exception("Could not add expression dataset %s", raw_path)
                errors.append(f"{raw_path}: {error}")
                continue
            added += int(
                self._insert_dataset_status(status, stale_message=stale_message)
            )

        if added:
            self._rebuild_dataset_table()
            self._refresh_cell_selector()
            self._refresh_image_channel_range()
            self._refresh_plot()
        elif paths:
            self._status_label.setText("No new datasets were added (duplicates are ignored).")
        if errors and show_errors:
            QMessageBox.warning(
                self,
                "Some datasets could not be added",
                "\n\n".join(errors),
            )
        elif errors:
            self._status_label.setText(errors[0])
        return added

    def _insert_dataset_status(
        self,
        status: ExpressionDatasetStatus,
        *,
        stale_message: str = "",
    ) -> bool:
        """Insert one shared status, retaining stale entries for reload UX."""

        key = _path_key(status.config_path)
        existing = self._datasets.get(key)
        if existing is not None:
            existing.repository_status = status
            if stale_message:
                existing.trace = None
                existing.request_key = None
                existing.resolution = "error"
                existing.availability = None
                existing.message = stale_message
            return False
        state = _DatasetViewState(
            dataset_id=_dataset_id(status.config_path),
            source_uri=str(status.config_path),
            included=True,
            label=self._unique_dataset_label(status.config_path.stem),
            group_id="all",
            color=_DEFAULT_COLORS[len(self._datasets) % len(_DEFAULT_COLORS)],
            path=status.config_path,
            repository_status=status,
        )
        if stale_message:
            state.resolution = "error"
            state.message = stale_message
        self._datasets[key] = state
        return True

    def remove_selected_datasets(self) -> None:
        rows = sorted(
            {index.row() for index in self._dataset_table.selectionModel().selectedRows()},
            reverse=True,
        )
        if not rows:
            return
        keys = list(self._datasets)
        for row in rows:
            if 0 <= row < len(keys):
                # Deliberately do not call repository.remove_dataset(): other
                # comparison windows and their measurement caches own it too.
                self._datasets.pop(keys[row], None)
        self._rebuild_dataset_table()
        self._refresh_cell_selector()
        self._refresh_image_channel_range()
        self._refresh_plot()

    def reload_selected_datasets(self) -> None:
        """Reload selected shared entries and invalidate this window's traces."""

        if self._data_mode is _WindowDataMode.FROZEN:
            self._status_label.setText(
                "Frozen results are source-independent and cannot be reloaded."
            )
            return
        assert self.repository is not None
        rows = sorted(
            {index.row() for index in self._dataset_table.selectionModel().selectedRows()}
        )
        if not rows:
            self._status_label.setText(
                "Select one or more dataset rows to reload their XML and source files."
            )
            return
        keys = list(self._datasets)
        errors: list[str] = []
        for row in rows:
            if not 0 <= row < len(keys):
                continue
            state = self._datasets[keys[row]]
            assert state.path is not None
            try:
                state.repository_status = self.repository.reload_dataset(state.path)
            except Exception as error:  # noqa: BLE001 - report each selected source
                logger.exception("Could not reload expression dataset %s", state.path)
                state.resolution = "error"
                state.availability = None
                state.message = f"Reload failed: {error}"
                errors.append(f"{state.label}: {error}")
                continue
            state.trace = None
            state.request_key = None
            state.resolution = "unprepared"
            state.availability = None
            state.message = "Reloaded; prepare this dataset again"
        self._rebuild_dataset_table()
        self._refresh_cell_selector()
        self._refresh_image_channel_range()
        self._refresh_plot()
        if errors:
            QMessageBox.warning(
                self,
                "Some datasets could not be reloaded",
                "\n\n".join(errors),
            )

    def _unique_dataset_label(self, base: str) -> str:
        used = {state.label for state in self._datasets.values()}
        if base not in used:
            return base
        suffix = 2
        while f"{base} ({suffix})" in used:
            suffix += 1
        return f"{base} ({suffix})"

    def _rebuild_dataset_table(self) -> None:
        self._building_table = True
        self._dataset_table.blockSignals(True)
        self._dataset_table.setRowCount(len(self._datasets))
        for row, (key, state) in enumerate(self._datasets.items()):
            use_item = QTableWidgetItem()
            use_item.setFlags(
                (use_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable
            )
            use_item.setCheckState(Qt.Checked if state.included else Qt.Unchecked)
            use_item.setData(Qt.UserRole, key)
            self._dataset_table.setItem(row, self.COL_USE, use_item)

            label_item = QTableWidgetItem(state.label)
            label_item.setData(Qt.UserRole, key)
            self._dataset_table.setItem(row, self.COL_LABEL, label_item)

            group_item = QTableWidgetItem(state.group_id)
            group_item.setData(Qt.UserRole, key)
            group_item.setToolTip(
                "Datasets with the same condition/group are summarized together. "
                "The summary uses the first available row's trace color for that group; "
                "individual traces keep their own colors."
            )
            self._dataset_table.setItem(row, self.COL_GROUP, group_item)

            color_item = QTableWidgetItem(state.color)
            color_item.setData(Qt.UserRole, key)
            color_item.setBackground(QColor(state.color))
            color_item.setToolTip("Double-click to choose a dataset trace color")
            self._dataset_table.setItem(row, self.COL_COLOR, color_item)

            status_item = QTableWidgetItem(state.message)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
            status_item.setData(Qt.UserRole, key)
            if state.repository_status is not None:
                status_item.setToolTip(
                    f"Source fingerprint: {state.repository_status.source_fingerprint}\n"
                    f"Session snapshot: {state.repository_status.snapshot_token}"
                )
            elif state.frozen_dataset is not None:
                provenance = state.frozen_dataset.provenance
                status_item.setToolTip(
                    "Embedded portable capture\n"
                    f"Source fingerprint: {provenance.source_fingerprint or 'not recorded'}\n"
                    f"Source revision: {provenance.source_revision!s}"
                )
            self._dataset_table.setItem(row, self.COL_STATUS, status_item)

            cache_item = QTableWidgetItem(
                _cache_label(state.repository_status)
                if state.repository_status is not None
                else "Embedded; offline-ready"
            )
            cache_item.setFlags(cache_item.flags() & ~Qt.ItemIsEditable)
            cache_item.setData(Qt.UserRole, key)
            self._dataset_table.setItem(row, self.COL_CACHE, cache_item)

            path_item = QTableWidgetItem(state.source_uri)
            path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
            path_item.setData(Qt.UserRole, key)
            path_item.setToolTip(
                state.source_uri
                + (
                    "\nProvenance only; this path is never opened in frozen mode."
                    if self._data_mode is _WindowDataMode.FROZEN
                    else ""
                )
            )
            self._dataset_table.setItem(row, self.COL_XML, path_item)
        self._dataset_table.blockSignals(False)
        self._building_table = False

    def _on_dataset_item_changed(self, item: QTableWidgetItem) -> None:
        if self._building_table:
            return
        key = str(item.data(Qt.UserRole) or "")
        state = self._datasets.get(key)
        if state is None:
            return
        if item.column() == self.COL_USE:
            state.included = item.checkState() == Qt.Checked
            self._refresh_cell_selector()
            self._refresh_image_channel_range()
        elif item.column() == self.COL_LABEL:
            label = item.text().strip()
            if not label:
                label = state.label or state.dataset_id
                item.setText(label)
            state.label = label
        elif item.column() == self.COL_GROUP:
            group_id = item.text().strip()
            if not group_id:
                group_id = "all"
                item.setText(group_id)
            state.group_id = group_id
        elif item.column() == self.COL_COLOR:
            color = QColor(item.text().strip())
            if not color.isValid():
                item.setText(state.color)
                return
            state.color = color.name(QColor.HexRgb)
            item.setText(state.color)
            item.setBackground(color)
        self._refresh_plot()

    def _on_dataset_cell_double_clicked(self, row: int, column: int) -> None:
        if column != self.COL_COLOR:
            return
        item = self._dataset_table.item(row, column)
        if item is None:
            return
        color = QColorDialog.getColor(QColor(item.text()), self, "Dataset trace color")
        if color.isValid():
            item.setText(color.name(QColor.HexRgb))

    # -- Cell/channel request -------------------------------------------

    def _refresh_cell_selector(self) -> None:
        if self._data_mode is _WindowDataMode.FROZEN:
            return
        previous = self._cell_combo.currentText().strip()
        states = self._included_states() or tuple(self._datasets.values())
        availability: dict[str, int] = {}
        for state in states:
            assert state.repository_status is not None
            for name in state.repository_status.cell_names:
                availability[name] = availability.get(name, 0) + 1

        self._cell_combo.blockSignals(True)
        self._cell_combo.clear()
        for name in sorted(availability, key=str.casefold):
            self._cell_combo.addItem(name)
        preferred = previous
        active = str(getattr(self.app, "current_cell_name", "") or "")
        if not preferred or preferred not in availability:
            preferred = active if active in availability else ""
        if not preferred and availability:
            common = [name for name, count in availability.items() if count == len(states)]
            preferred = sorted(common or list(availability), key=str.casefold)[0]
        self._cell_combo.setEditText(preferred)
        self._cell_combo.blockSignals(False)
        self._update_cell_availability()

    def _update_cell_availability(self) -> None:
        if self._data_mode is _WindowDataMode.FROZEN:
            return
        cell = self._cell_combo.currentText().strip()
        states = self._included_states()
        available = 0
        for state in states:
            assert state.repository_status is not None
            if cell in state.repository_status.cell_names:
                available += 1
        if not states:
            text = "Check at least one loaded dataset."
        elif not cell:
            text = "Enter one exact canonical cell name."
        else:
            text = f"{cell}: available in {available}/{len(states)} included dataset(s)."
            if available < len(states):
                text += " Missing datasets remain explicit in the exported status rows."
        self._cell_availability.setText(text)

    def _refresh_image_channel_range(self) -> None:
        if self._data_mode is _WindowDataMode.FROZEN:
            return
        assert self.repository is not None
        # Saved legacy fields never require opening an image provider.  Keep
        # provider discovery behind the user's explicit recompute choice.
        if self._source_mode() != "recomputed":
            return
        channel_counts: list[int] = []
        for state in self._included_states():
            assert state.path is not None
            try:
                channel_counts.append(
                    max(1, int(self.repository.image_channel_count(state.path)))
                )
            except Exception:
                continue
        # One shared physical channel is requested for every included
        # replicate, so expose only their common range.
        self._image_channel.setMaximum(min(channel_counts, default=1))

    def _on_source_changed(self, *_args) -> None:
        if self._data_mode is _WindowDataMode.FROZEN:
            return
        self._sync_source_controls()
        if self._source_mode() == "saved":
            self._legacy_ack.setChecked(False)
        else:
            self._refresh_image_channel_range()
        self._invalidate_local_traces()

    def _sync_source_controls(self) -> None:
        """Keep source-specific controls readable without changing data state."""

        if self._data_mode is _WindowDataMode.FROZEN:
            return
        saved = self._source_mode() == "saved"
        self._saved_channel_combo.setEnabled(saved)
        self._legacy_ack.setEnabled(saved)
        self._legacy_ack.setVisible(saved)
        self._image_channel.setEnabled(not saved)
        self._correction_combo.setEnabled(not saved)
        self._btn_prepare.setText(
            "Prepare included datasets" if saved else "Prepare / recompute included datasets"
        )

    def _on_trace_request_changed(self, *_args) -> None:
        if self._updating_controls or self._data_mode is _WindowDataMode.FROZEN:
            return
        self._update_cell_availability()
        if self._source_mode() == "saved":
            self._legacy_ack.setChecked(False)
        self._invalidate_local_traces()

    def _invalidate_local_traces(self) -> None:
        if self._data_mode is _WindowDataMode.FROZEN:
            return
        for state in self._datasets.values():
            state.trace = None
            state.request_key = None
            state.resolution = "unprepared"
            state.availability = None
            state.message = "Needs preparation"
        self._rebuild_dataset_table()
        self._refresh_plot()

    def _source_mode(self) -> str:
        if self._data_mode is _WindowDataMode.FROZEN:
            assert self._portable_result is not None
            return self._portable_result.source_mode.value
        return str(self._source_combo.currentData() or "saved")

    def _request_key(self) -> tuple[object, ...]:
        if self._data_mode is _WindowDataMode.FROZEN:
            assert self._portable_result is not None
            return ("frozen", self._portable_result.result_id)
        cell = self._cell_combo.currentText().strip()
        if self._source_mode() == "saved":
            return ("saved", cell, str(self._saved_channel_combo.currentData()))
        return (
            "recomputed",
            cell,
            self._image_channel.value() - 1,
            str(self._correction_combo.currentData()),
        )

    # -- Preparation -----------------------------------------------------

    def prepare_included_datasets(self) -> None:
        """Resolve every included trace, measuring images when requested."""

        if self._data_mode is _WindowDataMode.FROZEN:
            self._status_label.setText(
                "Frozen results already contain their native values and statuses."
            )
            return
        assert self.repository is not None
        if self._preparing:
            return
        states = self._included_states()
        cell = self._cell_combo.currentText().strip()
        if not states:
            QMessageBox.information(
                self, "Expression comparison", "Check at least one dataset first."
            )
            return
        if not cell:
            QMessageBox.information(
                self, "Expression comparison", "Enter one exact canonical cell name."
            )
            return
        unsaved = [
            state for state in states if self._current_dataset_has_unsaved_edits(state)
        ]
        if unsaved:
            for state in unsaved:
                state.trace = None
                state.request_key = None
                state.resolution = "error"
                state.availability = None
                state.message = (
                    "Main viewer has unsaved edits. Save it, then select this "
                    "row and click Reload selected before preparing."
                )
            self._rebuild_dataset_table()
            self._refresh_plot()
            QMessageBox.warning(
                self,
                "Save current dataset first",
                "The active AceTree dataset has unsaved edits, while comparison "
                "reads its XML/ZIP snapshot from disk. Save the main dataset, "
                "then use Reload selected before preparing this comparison.",
            )
            return

        progress = QProgressDialog(
            "Preparing expression datasets…",
            "Cancel",
            0,
            1000,
            self,
        )
        progress.setWindowTitle("Prepare expression comparison")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._active_progress = progress
        self._preparing = True
        self._set_preparing_controls(True)
        request_key = self._request_key()
        total = len(states)
        cancelled = False

        try:
            for dataset_index, state in enumerate(states):
                if progress.wasCanceled():
                    cancelled = True
                    break
                progress.setLabelText(f"Preparing {state.label}…")
                progress.setValue(int(1000 * dataset_index / total))
                QApplication.processEvents()

                if state.request_key == request_key and state.resolution in (
                    "ready",
                    "acquisition_status",
                ):
                    try:
                        current_status = self.repository.status(state.path)
                    except Exception:
                        current_status = None
                    expected_fingerprint = (
                        state.trace.dataset_snapshot_token
                        if state.trace is not None
                        else state.repository_status.snapshot_token
                    )
                    if (
                        current_status is not None
                        and current_status.snapshot_token == expected_fingerprint
                        and not (
                            state.availability is TraceAvailability.INCOMPLETE_DATA
                            and self._source_mode() == "recomputed"
                        )
                    ):
                        state.repository_status = current_status
                        continue
                    if current_status is not None:
                        state.repository_status = current_status
                    state.trace = None
                    state.request_key = None
                    state.resolution = "unprepared"
                    state.availability = None
                    state.message = "Source was reloaded; preparing again"

                completed_steps = 0

                def progress_cb(
                    channel_index: int,
                    num_channels: int,
                    timepoint: int,
                    num_timepoints: int,
                ) -> bool:
                    nonlocal completed_steps
                    local_total = max(1, int(num_channels) * int(num_timepoints))
                    completed_steps += 1
                    fraction = (
                        dataset_index + min(1.0, completed_steps / local_total)
                    ) / total
                    progress.setValue(min(999, int(1000 * fraction)))
                    progress.setLabelText(
                        f"{state.label}: reading movie once for all channels and "
                        f"corrections ({completed_steps}/{local_total})"
                    )
                    QApplication.processEvents()
                    return not progress.wasCanceled()

                try:
                    # Another already-open comparison may have populated the
                    # shared cache since this row was last refreshed.
                    status_reader = getattr(
                        self.repository,
                        "session_status",
                        self.repository.status,
                    )
                    state.repository_status = status_reader(state.path)
                    before_cache = set(state.repository_status.cached_corrections)
                    if self._source_mode() == "saved":
                        trace = self.repository.extract_saved_trace(
                            state.path,
                            cell,
                            str(self._saved_channel_combo.currentData()),
                        )
                    else:
                        trace = self.repository.extract_recomputed_trace(
                            state.path,
                            cell,
                            self._image_channel.value() - 1,
                            str(self._correction_combo.currentData()),
                            progress_cb=progress_cb,
                        )
                    if progress.wasCanceled():
                        cancelled = True
                        state.trace = None
                        state.request_key = None
                        state.resolution = "unprepared"
                        state.availability = None
                        state.message = "Preparation canceled"
                        break
                    state.trace = trace
                    state.request_key = request_key
                    state.resolution = "ready"
                    state.availability = None
                    if trace.provenance.source is ExpressionTraceSource.SAVED_LEGACY:
                        state.message = "Saved legacy values (provenance unverified)"
                    else:
                        method = trace.provenance.correction_method or "unknown"
                        cache_note = (
                            "reused session cache"
                            if method in before_cache
                            else "measured now"
                        )
                        state.message = (
                            f"Recomputed channel {(trace.provenance.image_channel or 0) + 1}, "
                            f"{method} ({cache_note})"
                        )
                except CanonicalCellNotFoundError:
                    state.trace = None
                    state.request_key = request_key
                    state.resolution = "acquisition_status"
                    state.availability = TraceAvailability.MISSING_CELL
                    state.message = f"Cell {cell!r} is absent from this dataset."
                except CanonicalCellAmbiguousError as error:
                    state.trace = None
                    state.request_key = request_key
                    state.resolution = "acquisition_status"
                    state.availability = TraceAvailability.AMBIGUOUS
                    state.message = (
                        f"Cell {cell!r} is ambiguous in this dataset: {error}. "
                        "Correct the duplicate canonical names in the source dataset."
                    )
                except ExpressionChannelUnavailableError as error:
                    state.trace = None
                    state.request_key = request_key
                    state.resolution = "acquisition_status"
                    state.availability = TraceAvailability.MISSING_CHANNEL
                    if self._source_mode() == "saved":
                        state.message = (
                            f"Saved field is unavailable: {error}. Choose "
                            "'Recompute from image channel' or another saved field."
                        )
                    else:
                        state.message = (
                            f"Requested image channel is unavailable: {error}. "
                            "Choose a channel present in every included dataset."
                        )
                except ExpressionDataIncompleteError as error:
                    state.trace = None
                    state.request_key = request_key
                    state.resolution = "acquisition_status"
                    state.availability = TraceAvailability.INCOMPLETE_DATA
                    if self._source_mode() == "saved":
                        state.message = (
                            f"Saved expression values are incomplete: {error}. "
                            "Choose 'Recompute from image channel' to fill this trace."
                        )
                    else:
                        state.message = (
                            f"Recomputed expression values are incomplete: {error}. "
                            "Verify the image source and selected channel, then prepare "
                            "again; known-incomplete cache data will not be reused."
                        )
                except DatasetBusyError as error:
                    state.trace = None
                    state.request_key = None
                    state.resolution = "unprepared"
                    state.availability = None
                    state.message = f"Dataset busy: {error}"
                except Exception as error:  # noqa: BLE001 - keep other datasets usable
                    logger.exception("Could not prepare expression dataset %s", state.path)
                    if progress.wasCanceled():
                        cancelled = True
                        state.trace = None
                        state.request_key = None
                        state.resolution = "unprepared"
                        state.availability = None
                        state.message = "Preparation canceled"
                        break
                    state.trace = None
                    state.request_key = request_key
                    state.resolution = "error"
                    state.availability = None
                    state.message = f"Error: {error}"

                try:
                    status_reader = getattr(
                        self.repository,
                        "session_status",
                        self.repository.status,
                    )
                    state.repository_status = status_reader(state.path)
                except Exception as error:  # noqa: BLE001 - source revalidation is fail closed
                    state.trace = None
                    state.resolution = "error"
                    state.availability = None
                    state.message = f"Source changed: {error}"

                progress.setValue(int(1000 * (dataset_index + 1) / total))
                self._rebuild_dataset_table()
                QApplication.processEvents()
        finally:
            progress.setValue(1000 if not cancelled else progress.value())
            progress.close()
            self._active_progress = None
            self._preparing = False
            self._set_preparing_controls(False)
            self._rebuild_dataset_table()
            self._refresh_plot()
            if self._close_when_ready:
                self._close_when_ready = False
                self.close()

    def _set_preparing_controls(self, preparing: bool) -> None:
        for widget in (
            self._btn_add,
            self._btn_remove,
            self._btn_reload,
            self._btn_prepare,
            self._dataset_table,
            self._cell_combo,
            self._source_combo,
            self._saved_channel_combo,
            self._image_channel,
            self._correction_combo,
        ):
            widget.setEnabled(not preparing)
        if not preparing:
            self._sync_source_controls()

    # -- Plot model and rendering ---------------------------------------

    def _on_plot_option_changed(self, *_args) -> None:
        normalized = str(self._time_combo.currentData()) == TimeAxisMode.NORMALIZED.value
        self._normalized_points.setEnabled(normalized)
        self._grid_step.setEnabled(not normalized)
        self._refresh_plot()

    def _on_center_changed(self, *_args) -> None:
        previous = str(self._band_combo.currentData() or BandStatistic.NONE.value)
        self._populate_band_options(previous=previous)
        has_center = str(self._center_combo.currentData()) != CenterStatistic.NONE.value
        self._band_combo.setEnabled(has_center)
        self._band_opacity.setEnabled(
            has_center
            and str(self._band_combo.currentData()) != BandStatistic.NONE.value
        )
        self._refresh_plot()

    def _populate_band_options(self, *, previous: str | None = None) -> None:
        center = str(self._center_combo.currentData() or CenterStatistic.NONE.value)
        if center == CenterStatistic.MEAN.value:
            options = _MEAN_BAND_OPTIONS
        elif center == CenterStatistic.MEDIAN.value:
            options = _MEDIAN_BAND_OPTIONS
        else:
            options = (("None", BandStatistic.NONE.value),)
        previous = previous or str(
            self._band_combo.currentData() or BandStatistic.NONE.value
        )
        self._band_combo.blockSignals(True)
        self._band_combo.clear()
        for label, value in options:
            self._band_combo.addItem(label, value)
        index = self._band_combo.findData(previous)
        self._band_combo.setCurrentIndex(max(0, index))
        self._band_combo.blockSignals(False)

    def _on_band_changed(self, *_args) -> None:
        self._band_opacity.setEnabled(
            str(self._band_combo.currentData()) != BandStatistic.NONE.value
        )
        self._refresh_plot()

    def _on_show_traces_toggled(self, visible: bool) -> None:
        for widget in (
            self._trace_opacity,
            self._trace_line_style,
            self._trace_width,
            self._marker_combo,
            self._marker_size,
        ):
            widget.setEnabled(bool(visible))
        self._refresh_plot()

    def _on_smoothing_toggled(self, enabled: bool) -> None:
        self._smoothing_sigma.setEnabled(bool(enabled))
        self._refresh_plot()

    def _on_auto_x_changed(self, automatic: bool) -> None:
        self._x_min.setEnabled(not automatic)
        self._x_max.setEnabled(not automatic)
        self._refresh_plot()

    def _on_auto_y_changed(self, automatic: bool) -> None:
        self._y_min.setEnabled(not automatic)
        self._y_max.setEnabled(not automatic)
        self._refresh_plot()

    def _choose_figure_background(self) -> None:
        color = QColorDialog.getColor(
            QColor(self._figure_background),
            self,
            "Figure background",
        )
        if color.isValid():
            self._figure_background = color.name(QColor.HexRgb)
            self._refresh_plot()

    def _choose_axes_background(self) -> None:
        color = QColorDialog.getColor(
            QColor(self._axes_background),
            self,
            "Plot background",
        )
        if color.isValid():
            self._axes_background = color.name(QColor.HexRgb)
            self._refresh_plot()

    def _choose_text_color(self) -> None:
        color = QColorDialog.getColor(
            QColor(self._text_color),
            self,
            "Axis and text color",
        )
        if color.isValid():
            self._text_color = color.name(QColor.HexRgb)
            self._refresh_plot()

    def _refresh_plot(self, *_args) -> None:
        if not hasattr(self, "_axes") or self._updating_controls:
            return
        states = self._included_states()
        request_key = self._request_key()
        unresolved = [
            state
            for state in states
            if state.request_key != request_key
            or state.resolution not in ("ready", "acquisition_status")
        ]
        if not states:
            self._plot_data = None
            message = (
                "Check one or more datasets embedded in this frozen result."
                if self._data_mode is _WindowDataMode.FROZEN
                else "Add and check one or more AceTree XML datasets."
            )
            self._draw_empty(message)
            self._set_export_enabled(False)
            self._status_label.setText("No datasets included in this comparison.")
            return
        if not self._cell_combo.currentText().strip():
            self._plot_data = None
            self._draw_empty("Choose one exact canonical cell name.")
            self._set_export_enabled(False)
            return
        if unresolved:
            self._plot_data = None
            self._draw_empty(
                "Prepare included datasets before plotting.\n"
                "Rows with errors can be corrected, retried, or unchecked."
            )
            self._set_export_enabled(False)
            self._status_label.setText(
                f"{len(unresolved)}/{len(states)} included dataset(s) need preparation."
            )
            return

        try:
            datasets = tuple(self._expression_dataset(state) for state in states)
            spec = self._current_spec(states)
            if self._data_mode is _WindowDataMode.FROZEN:
                assert self._portable_result is not None
                all_datasets = tuple(
                    self._expression_dataset(state)
                    for state in self._datasets.values()
                )
                transient = revise_expression_comparison_result(
                    self._portable_result,
                    datasets=all_datasets,
                    spec=spec,
                    appearance=self._current_appearance(),
                )
                data = build_expression_comparison_data(
                    transient,
                    included_dataset_ids=[state.dataset_id for state in states],
                )
            else:
                data = self._service.build(
                    datasets,
                    cell_names=spec.cell_names,
                    channel_key=spec.channel_key,
                    channel_label=spec.channel_label,
                    channel_unit=spec.channel_unit,
                    time_mode=spec.time_mode,
                    grid=spec.grid,
                    smoothing=spec.smoothing,
                    summary=spec.summary,
                    channel_bindings=spec.channel_bindings,
                    cell_aliases=spec.cell_aliases,
                )
        except Exception as error:  # noqa: BLE001 - show model validation in-window
            logger.exception("Could not build expression comparison")
            self._plot_data = None
            self._draw_empty(f"Could not build comparison: {error}")
            self._set_export_enabled(False)
            self._status_label.setText(str(error))
            return

        self._plot_data = data
        has_visible_artists = self._draw_data(data)
        legacy_blocked = self._legacy_export_unacknowledged(states)
        self._set_export_enabled(
            data.has_data and has_visible_artists and not legacy_blocked,
            csv_enabled=bool(data.statuses) and not legacy_blocked,
        )
        missing = sum(1 for status in data.statuses if status.availability.value != "available")
        message = (
            f"{len(data.aligned_traces)} available dataset trace(s); "
            f"{missing} explicit unavailable/acquisition status record(s)."
        )
        if legacy_blocked:
            if self._data_mode is _WindowDataMode.FROZEN:
                message += (
                    " Export is disabled because this capture contains legacy "
                    "numeric values without a recorded provenance acknowledgement."
                )
            else:
                message += " Check the legacy provenance acknowledgement to enable export."
        if data.has_data and not has_visible_artists:
            message += " Enable individual traces or a center line to export an SVG."
        if data.warnings:
            message += f" {len(data.warnings)} model warning(s)."
        self._status_label.setText(message)

    def _current_spec(
        self, states: tuple[_DatasetViewState, ...]
    ) -> ComparisonSpec:
        center = CenterStatistic(str(self._center_combo.currentData()))
        band = BandStatistic(str(self._band_combo.currentData()))
        if center is CenterStatistic.NONE:
            band = BandStatistic.NONE
        time_mode = TimeAxisMode(str(self._time_combo.currentData()))
        if self._data_mode is _WindowDataMode.FROZEN:
            assert self._portable_result is not None
            base = self._portable_result.spec
            channel_key = base.channel_key
            channel_label = base.channel_label
            channel_unit = base.channel_unit
            cell_names = base.cell_names
            channel_bindings = base.channel_bindings
            cell_aliases = base.cell_aliases
            grid_start = base.grid.start
            grid_end = base.grid.end
            max_points = base.grid.max_points
            truncate = base.smoothing.truncate
        else:
            trace = next((state.trace for state in states if state.trace is not None), None)
            if trace is not None:
                channel_key = trace.channel_key
                channel_label = trace.channel_label
                channel_unit = trace.channel_unit
            elif self._source_mode() == "saved":
                channel_key = str(self._saved_channel_combo.currentData())
                selected = next(
                    channel for channel in DEFAULT_EXPRESSION_CHANNELS
                    if channel.key == channel_key
                )
                channel_label = selected.label
                channel_unit = selected.unit
            else:
                channel_key = f"measured_channel_{self._image_channel.value()}"
                channel_label = f"Channel {self._image_channel.value()}"
                channel_unit = "scaled mean intensity"
            cell_names = (self._cell_combo.currentText().strip(),)
            channel_bindings = ()
            cell_aliases = ()
            grid_start = None
            grid_end = None
            max_points = 1_000_000
            truncate = 4.0
        return ComparisonSpec(
            cell_names=cell_names,
            channel_key=channel_key,
            channel_label=channel_label,
            channel_unit=channel_unit,
            time_mode=time_mode,
            grid=GridSpec(
                domain=GridDomain(str(self._grid_domain.currentData())),
                step=None if time_mode is TimeAxisMode.NORMALIZED else self._grid_step.value(),
                normalized_points=self._normalized_points.value(),
                start=grid_start,
                end=grid_end,
                max_points=max_points,
            ),
            smoothing=SmoothingSpec(
                sigma=(
                    self._smoothing_sigma.value()
                    if self._smoothing_check.isChecked()
                    else 0.0
                ),
                truncate=truncate,
            ),
            summary=SummarySpec(center=center, band=band),
            channel_bindings=channel_bindings,
            cell_aliases=cell_aliases,
        )

    def _expression_dataset(self, state: _DatasetViewState) -> ExpressionDataset:
        if self._data_mode is _WindowDataMode.FROZEN:
            if state.frozen_dataset is None:
                raise RuntimeError(f"Frozen dataset {state.dataset_id!r} is unavailable")
            dataset = state.frozen_dataset
            return replace(
                dataset,
                provenance=replace(
                    dataset.provenance,
                    label=state.label,
                    group_id=state.group_id,
                ),
                traces=tuple(
                    replace(trace, series_label=state.label, color=state.color)
                    for trace in dataset.traces
                ),
            )
        assert state.repository_status is not None
        assert state.path is not None
        traces: tuple[DatasetExpressionTrace, ...]
        acquisition_statuses: tuple[DatasetAcquisitionStatus, ...] = ()
        if state.trace is None:
            traces = ()
            fingerprint = state.repository_status.source_fingerprint
            snapshot_token = state.repository_status.snapshot_token
            generation = state.repository_status.generation
            manifest_token = state.repository_status.image_manifest_token
            if state.availability is not None:
                requested_channel = (
                    str(self._saved_channel_combo.currentData())
                    if self._source_mode() == "saved"
                    else f"measured_channel_{self._image_channel.value()}"
                )
                acquisition_statuses = (
                    DatasetAcquisitionStatus(
                        cell_name=self._cell_combo.currentText().strip(),
                        channel_key=requested_channel,
                        availability=state.availability,
                        message=state.message,
                    ),
                )
        else:
            native = state.trace
            fingerprint = native.dataset_fingerprint
            snapshot_token = native.dataset_snapshot_token
            generation = native.dataset_generation
            manifest_token = native.image_manifest_token
            traces = (
                DatasetExpressionTrace(
                    cell_name=native.cell_name,
                    channel_key=native.channel_key,
                    channel_label=native.channel_label,
                    channel_unit=native.channel_unit,
                    absolute_times=tuple(float(value) for value in native.timepoints),
                    values=tuple(float(value) for value in native.values),
                    birth_time=float(native.start_time),
                    end_time=float(native.end_time),
                    series_label=state.label,
                    color=state.color,
                ),
            )
        metadata = (
            ("resolution", state.resolution),
            ("window_source_mode", self._source_mode()),
            ("session_snapshot_token", snapshot_token),
            ("image_manifest_token", manifest_token or ""),
        )
        if state.trace is not None:
            provenance = state.trace.provenance
            metadata += (
                ("trace_source", provenance.source.value),
                ("trace_freshness", provenance.freshness.value),
                (
                    "image_channel",
                    ""
                    if provenance.image_channel is None
                    else str(provenance.image_channel + 1),
                ),
                ("correction_method", provenance.correction_method or "unknown"),
                ("channel_verified", str(provenance.channel_verified).lower()),
                ("correction_verified", str(provenance.correction_verified).lower()),
            )
        if self._source_mode() == "saved" and self._legacy_ack.isChecked():
            metadata += (("legacy_acknowledgement", "true"),)
        return ExpressionDataset(
            provenance=DatasetProvenance(
                dataset_id=state.dataset_id,
                label=state.label,
                group_id=state.group_id,
                source_uri=str(state.path),
                source_fingerprint=fingerprint,
                source_revision=generation,
                metadata=metadata,
            ),
            traces=traces,
            acquisition_statuses=acquisition_statuses,
        )

    def _draw_data(self, data: ExpressionComparisonData) -> bool:
        axes = self._axes
        axes.clear()
        axes.set_facecolor(self._axes_background)
        self._figure.patch.set_facecolor(self._figure_background)
        artist_count = 0
        if self._show_traces.isChecked():
            for trace in data.aligned_traces:
                if not any(value is not None for value in trace.display_values):
                    continue
                axes.plot(
                    trace.grid_x,
                    _nan_values(trace.display_values),
                    label=trace.series_label,
                    color=trace.color,
                    alpha=self._trace_opacity.value(),
                    linestyle=str(self._trace_line_style.currentData()),
                    linewidth=self._trace_width.value(),
                    marker=str(self._marker_combo.currentData()),
                    markersize=self._marker_size.value(),
                )
                artist_count += 1

        center_name = str(self._center_combo.currentText())
        band_name = (
            str(self._band_combo.currentText())
            if str(self._band_combo.currentData()) != BandStatistic.NONE.value
            else ""
        )
        summary_colors = _summary_colors(data.aligned_traces)
        for summary in data.summaries:
            color = summary_colors.get((summary.group_id, summary.cell_name), "#111111")
            center = _nan_values(summary.center)
            if any(value is not None for value in summary.center):
                axes.plot(
                    summary.grid_x,
                    center,
                    label=(
                        f"{summary.group_id}: {center_name}"
                        f"{' + ' + band_name + ' band' if band_name else ''} "
                        f"(n≤{summary.n_available}/{summary.n_selected})"
                    ),
                    color=color,
                    linestyle=str(self._center_line_style.currentData()),
                    linewidth=self._center_width.value(),
                )
                artist_count += 1
            lower = _nan_values(summary.lower)
            upper = _nan_values(summary.upper)
            if any(value is not None for value in summary.lower) and any(
                value is not None for value in summary.upper
            ):
                axes.fill_between(
                    summary.grid_x,
                    lower,
                    upper,
                    color=color,
                    alpha=self._band_opacity.value(),
                    linewidth=0,
                )
                artist_count += 1

        if artist_count == 0:
            self._draw_empty(
                "No visible plot layer. Enable individual traces or choose a "
                "mean/median center line."
            )
            return False

        title = self._title_edit.text().strip() or "Expression comparison"
        x_label = self._x_label_edit.text().strip() or _time_axis_label(data.spec.time_mode)
        y_label = self._y_label_edit.text().strip() or data.spec.channel_label
        if data.spec.channel_unit:
            y_label = (
                self._y_label_edit.text().strip()
                or f"{data.spec.channel_label} ({data.spec.channel_unit})"
            )
        axes.set_title(
            title,
            color=self._text_color,
            fontsize=self._title_size.value(),
        )
        axes.set_xlabel(
            x_label,
            color=self._text_color,
            fontsize=self._font_size.value(),
        )
        axes.set_ylabel(
            y_label,
            color=self._text_color,
            fontsize=self._font_size.value(),
        )
        axes.set_yscale(str(self._y_scale.currentData()))
        axes.grid(self._grid_check.isChecked(), alpha=0.25)
        axes.tick_params(colors=self._text_color, labelsize=self._font_size.value())
        for spine in axes.spines.values():
            spine.set_color(self._text_color)
        if not self._auto_x.isChecked() and self._x_min.value() < self._x_max.value():
            axes.set_xlim(self._x_min.value(), self._x_max.value())
        if not self._auto_y.isChecked() and self._y_min.value() < self._y_max.value():
            axes.set_ylim(self._y_min.value(), self._y_max.value())
        if self._legend_check.isChecked():
            location = str(self._legend_location.currentData())
            options = {
                "title": self._legend_title.text().strip() or None,
                "ncols": self._legend_columns.value(),
                "fontsize": self._font_size.value(),
            }
            if location == "outside":
                legend = axes.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    **options,
                )
            else:
                legend = axes.legend(loc=location, **options)
            if legend is not None:
                legend.get_frame().set_facecolor(self._axes_background)
                for text in legend.get_texts():
                    text.set_color(self._text_color)
                if legend.get_title() is not None:
                    legend.get_title().set_color(self._text_color)
        self._canvas.draw()
        return True

    def _draw_empty(self, message: str) -> None:
        self._axes.clear()
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

    # -- Export ----------------------------------------------------------

    def _current_appearance(
        self,
        states: Iterable[_DatasetViewState] | None = None,
    ) -> dict[str, Any]:
        selected = tuple(self._datasets.values() if states is None else states)
        current = {
            APPEARANCE_INCLUDED_DATASET_IDS: [
                state.dataset_id for state in selected if state.included
            ],
            "dataset_overrides": {
                state.dataset_id: {
                    "included": state.included,
                    "label": state.label,
                    "group_id": state.group_id,
                    "color": state.color,
                }
                for state in selected
            },
            "show_traces": self._show_traces.isChecked(),
            "trace_opacity": self._trace_opacity.value(),
            "band_opacity": self._band_opacity.value(),
            "title": self._title_edit.text(),
            "x_label": self._x_label_edit.text(),
            "y_label": self._y_label_edit.text(),
            "trace_line_style": str(self._trace_line_style.currentData()),
            "center_line_style": str(self._center_line_style.currentData()),
            "marker": str(self._marker_combo.currentData()),
            "trace_width": self._trace_width.value(),
            "center_width": self._center_width.value(),
            "marker_size": self._marker_size.value(),
            "font_size": self._font_size.value(),
            "title_size": self._title_size.value(),
            "show_legend": self._legend_check.isChecked(),
            "legend_location": str(self._legend_location.currentData()),
            "legend_title": self._legend_title.text(),
            "legend_columns": self._legend_columns.value(),
            "show_grid": self._grid_check.isChecked(),
            "y_scale": str(self._y_scale.currentData()),
            "auto_x": self._auto_x.isChecked(),
            "x_min": self._x_min.value(),
            "x_max": self._x_max.value(),
            "auto_y": self._auto_y.isChecked(),
            "y_min": self._y_min.value(),
            "y_max": self._y_max.value(),
            "figure_background": self._figure_background,
            "axes_background": self._axes_background,
            "text_color": self._text_color,
        }
        if (
            self._data_mode is _WindowDataMode.FROZEN
            and self._portable_result is not None
        ):
            preserved = dict(self._portable_result.appearance)
            previous_overrides = preserved.get("dataset_overrides")
            current_overrides = current["dataset_overrides"]
            if isinstance(previous_overrides, Mapping):
                merged_overrides: dict[str, Any] = {}
                for dataset_id, values in current_overrides.items():
                    previous = previous_overrides.get(dataset_id)
                    merged = dict(previous) if isinstance(previous, Mapping) else {}
                    merged.update(values)
                    merged_overrides[dataset_id] = merged
                current["dataset_overrides"] = merged_overrides
            preserved.update(current)
            return preserved
        return current

    def _legacy_export_unacknowledged(
        self, states: tuple[_DatasetViewState, ...]
    ) -> bool:
        if self._data_mode is _WindowDataMode.FROZEN:
            result = self._portable_result
            if result is None or result.legacy_acknowledged:
                return False
            for state in states:
                dataset = state.frozen_dataset
                if dataset is None or not dataset.traces:
                    continue
                if result.source_mode is ExpressionComparisonSourceMode.SAVED:
                    return True
                if result.source_mode is ExpressionComparisonSourceMode.MIXED:
                    metadata = dict(dataset.provenance.metadata)
                    source = metadata.get("trace_source")
                    if source != ExpressionTraceSource.RECOMPUTED.value:
                        # Known legacy or absent/future provenance fails closed.
                        return True
            return False
        return any(
            state.trace is not None
            and state.trace.provenance.source is ExpressionTraceSource.SAVED_LEGACY
            for state in states
        ) and not self._legacy_ack.isChecked()

    def _set_export_enabled(
        self, enabled: bool, *, csv_enabled: bool | None = None
    ) -> None:
        csv_state = enabled if csv_enabled is None else csv_enabled
        self._btn_export_csv.setEnabled(csv_state)
        self._btn_export_svg.setEnabled(enabled)
        self._btn_save_result.setEnabled(csv_state)
        self._toolbar.set_save_enabled(enabled)

    def _exportable_snapshot(
        self, *, allow_status_only: bool = False
    ) -> ExpressionComparisonData:
        """Revalidate every included source before returning the rendered data."""

        states = self._included_states()
        request_key = self._request_key()
        if not states:
            raise RuntimeError("No datasets are included in this comparison.")
        if self._data_mode is _WindowDataMode.FROZEN:
            if self._legacy_export_unacknowledged(states):
                self._set_export_enabled(False)
                raise RuntimeError(
                    "This frozen result contains legacy numeric values without a "
                    "recorded provenance acknowledgement. Reopen the source comparison, "
                    "acknowledge its legacy provenance, and save a new portable result."
                )
            unresolved = [
                state
                for state in states
                if state.request_key != request_key
                or state.resolution not in ("ready", "acquisition_status")
            ]
            if unresolved:
                self._set_export_enabled(False)
                raise RuntimeError("The frozen result contains an unresolved dataset.")
            if self._plot_data is None or (
                not allow_status_only and not self._plot_data.has_data
            ):
                raise RuntimeError("There is no frozen expression comparison to export.")
            if allow_status_only and not self._plot_data.statuses:
                raise RuntimeError("There are no frozen dataset statuses to export.")
            return self._plot_data
        assert self.repository is not None
        for state in states:
            assert state.path is not None
            assert state.repository_status is not None
            if self._current_dataset_has_unsaved_edits(state):
                state.trace = None
                state.request_key = None
                state.resolution = "error"
                state.availability = None
                state.message = (
                    "Main viewer has unsaved edits. Save it, then use Reload "
                    "selected before exporting."
                )
                self._rebuild_dataset_table()
                self._set_export_enabled(False)
                raise RuntimeError(state.message)
            if state.request_key != request_key or state.resolution not in (
                "ready",
                "acquisition_status",
            ):
                self._set_export_enabled(False)
                raise RuntimeError(
                    f"Dataset {state.label!r} is unresolved. Prepare it or "
                    "uncheck it before export."
                )
            try:
                current = self.repository.status(state.path)
            except Exception as error:
                state.trace = None
                state.resolution = "error"
                state.message = f"Source changed: {error}"
                self._rebuild_dataset_table()
                self._set_export_enabled(False)
                raise RuntimeError(
                    f"Dataset {state.label!r} changed after preparation; "
                    "reload and prepare it again."
                ) from error
            expected = (
                state.trace.dataset_snapshot_token
                if state.trace is not None
                else state.repository_status.snapshot_token
            )
            if current.snapshot_token != expected:
                state.trace = None
                state.request_key = None
                state.resolution = "unprepared"
                state.availability = None
                state.repository_status = current
                state.message = (
                    "The shared dataset was reloaded after this plot was prepared. "
                    "Prepare this row again before export."
                )
                self._rebuild_dataset_table()
                self._set_export_enabled(False)
                raise RuntimeError(
                    f"Dataset {state.label!r} no longer matches the rendered snapshot; "
                    "prepare it again."
                )
            state.repository_status = current

        if self._legacy_export_unacknowledged(states):
            self._set_export_enabled(False)
            raise RuntimeError(
                "Saved legacy expression has unknown channel, correction, and freshness "
                "provenance. Check the acknowledgement or recompute from images before export."
            )
        if self._plot_data is None or (
            not allow_status_only and not self._plot_data.has_data
        ):
            raise RuntimeError("There is no prepared expression comparison to export.")
        if allow_status_only and not self._plot_data.statuses:
            raise RuntimeError("There are no prepared dataset statuses to export.")
        return self._plot_data

    def _portable_result_for_save(self) -> ExpressionComparisonResult:
        """Create a portable revision after the same fail-closed export checks."""

        data = self._exportable_snapshot(allow_status_only=True)
        if self._data_mode is _WindowDataMode.FROZEN:
            assert self._portable_result is not None
            datasets = tuple(
                self._expression_dataset(state) for state in self._datasets.values()
            )
            return revise_expression_comparison_result(
                self._portable_result,
                datasets=datasets,
                spec=data.spec,
                appearance=self._current_appearance(),
            )

        assert self.repository is not None
        request_key = self._request_key()
        capture_states: list[_DatasetViewState] = []
        for state in self._datasets.values():
            if state.request_key != request_key or state.resolution not in (
                "ready",
                "acquisition_status",
            ):
                # An unchecked, never-prepared row is not part of this capture.
                if state.included:
                    raise RuntimeError(
                        f"Dataset {state.label!r} is unresolved; prepare it first."
                    )
                continue
            if not state.included:
                assert state.path is not None
                assert state.repository_status is not None
                try:
                    current = self.repository.status(state.path)
                except Exception as error:
                    raise RuntimeError(
                        f"Excluded dataset {state.label!r} changed after preparation; "
                        "reload it or remove it before saving the portable result."
                    ) from error
                expected = (
                    state.trace.dataset_snapshot_token
                    if state.trace is not None
                    else state.repository_status.snapshot_token
                )
                if current.snapshot_token != expected:
                    raise RuntimeError(
                        f"Excluded dataset {state.label!r} no longer matches its "
                        "prepared snapshot."
                    )
                state.repository_status = current
            capture_states.append(state)
        if not capture_states:
            raise RuntimeError("No prepared datasets are available to capture.")
        datasets = tuple(self._expression_dataset(state) for state in capture_states)
        source_mode = ExpressionComparisonSourceMode(self._source_mode())
        acquisition_metadata: dict[str, Any] = {
            "cell_name": self._cell_combo.currentText().strip(),
            "source_mode": self._source_mode(),
        }
        if source_mode is ExpressionComparisonSourceMode.SAVED:
            acquisition_metadata["saved_channel_key"] = str(
                self._saved_channel_combo.currentData()
            )
        else:
            acquisition_metadata.update(
                {
                    "image_channel_one_based": self._image_channel.value(),
                    "correction_method": str(self._correction_combo.currentData()),
                }
            )
        captured = capture_expression_comparison_result(
            datasets,
            data.spec,
            source_mode=source_mode,
            acquisition_metadata=acquisition_metadata,
            legacy_acknowledged=self._legacy_ack.isChecked(),
            appearance=self._current_appearance(capture_states),
        )
        if (
            self._portable_result is not None
            and self._portable_result.source_mode is source_mode
        ):
            try:
                return revise_expression_comparison_result(
                    self._portable_result,
                    datasets=datasets,
                    spec=data.spec,
                    appearance=self._current_appearance(capture_states),
                )
            except ValueError:
                # Changed native values or request identity start a new
                # capture; presentation-only resaves remain linked revisions.
                pass
        return captured

    def _choose_result_path(self) -> None:
        default_name = _display_filename(self._result_path) or (
            "expression_comparison" + EXPRESSION_COMPARISON_RESULT_SUFFIX
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save portable expression result",
            default_name,
            "AceTree expression results (*.aceexpr)",
        )
        if not path:
            return
        try:
            self.save_portable_result(path)
        except Exception as error:  # noqa: BLE001 - validation and filesystem failures
            logger.exception("Could not save portable expression result")
            QMessageBox.warning(self, "Cannot save portable result", str(error))

    def save_portable_result(self, path: str | Path) -> Path:
        result = self._portable_result_for_save()
        saved = save_expression_comparison_result(path, result)
        self._portable_result = saved.result
        self._result_path = str(saved.path)
        if self._data_mode is _WindowDataMode.FROZEN:
            by_id = {
                dataset.provenance.dataset_id: dataset
                for dataset in saved.result.datasets
            }
            for state in self._datasets.values():
                state.frozen_dataset = by_id[state.dataset_id]
                state.request_key = ("frozen", saved.result.result_id)
            self._configure_frozen_ui(saved.result)
            self._refresh_plot()
        self._status_label.setText(f"Saved portable result: {saved.path}")
        return saved.path

    def _choose_csv_path(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save exact expression comparison",
            "expression_comparison.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            self.export_csv(path)
        except Exception as error:  # noqa: BLE001 - filesystem and revalidation errors
            logger.exception("Could not export expression comparison CSV")
            QMessageBox.warning(self, "Cannot export comparison", str(error))

    def _choose_svg_path(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export expression comparison",
            "expression_comparison.svg",
            "SVG files (*.svg)",
        )
        if not path:
            return
        try:
            self.export_svg(path)
        except Exception as error:  # noqa: BLE001 - filesystem and revalidation errors
            logger.exception("Could not export expression comparison SVG")
            QMessageBox.warning(self, "Cannot export comparison", str(error))

    def export_csv(self, path: str | Path) -> Path:
        data = self._exportable_snapshot(allow_status_only=True)
        destination = _with_suffix(path, ".csv")
        export_expression_comparison_tidy_csv(data, destination)
        return destination

    def export_svg(self, path: str | Path) -> Path:
        self._exportable_snapshot()
        destination = _with_suffix(path, ".svg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(destination, format="svg", bbox_inches="tight")
        return destination

    # -- Lifecycle and drag/drop ----------------------------------------

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls() if event.mimeData() is not None else ()
        accepted_suffixes = {EXPRESSION_COMPARISON_RESULT_SUFFIX}
        if self._data_mode is _WindowDataMode.LIVE:
            accepted_suffixes.add(".xml")
        if any(
            Path(url.toLocalFile()).suffix.lower() in accepted_suffixes
            for url in urls
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls() if event.mimeData() is not None else ()
        result_paths = [
            url.toLocalFile()
            for url in urls
            if Path(url.toLocalFile()).suffix.lower()
            == EXPRESSION_COMPARISON_RESULT_SUFFIX
        ]
        opener = getattr(self.app, "open_expression_comparison_result_window", None)
        for result_path in result_paths:
            if callable(opener):
                opener(result_path)
        xml_paths = [
            Path(url.toLocalFile())
            for url in urls
            if Path(url.toLocalFile()).suffix.lower() == ".xml"
        ]
        if xml_paths and self._data_mode is _WindowDataMode.LIVE:
            self.add_dataset_paths(xml_paths)
        if result_paths or (xml_paths and self._data_mode is _WindowDataMode.LIVE):
            event.acceptProposedAction()

    def closeEvent(self, event) -> None:
        if self._preparing:
            self._close_when_ready = True
            if self._active_progress is not None:
                self._active_progress.cancel()
            event.ignore()
            return
        windows = getattr(self.app, "_expression_comparison_windows", None)
        if windows is not None:
            try:
                windows.remove(self)
            except ValueError:
                pass
        super().closeEvent(event)

    # -- Small helpers ---------------------------------------------------

    def _included_states(self) -> tuple[_DatasetViewState, ...]:
        return tuple(state for state in self._datasets.values() if state.included)

    def _current_dataset_has_unsaved_edits(
        self,
        state: _DatasetViewState,
    ) -> bool:
        if self._data_mode is _WindowDataMode.FROZEN or state.path is None:
            return False
        manager = getattr(self.app, "manager", None)
        config = getattr(manager, "config", None)
        current_path = getattr(config, "config_file", None)
        if current_path is None:
            return False
        if _path_key(Path(current_path)) != _path_key(state.path):
            return False
        history = getattr(self.app, "edit_history", None)
        return bool(
            getattr(history, "modified", False)
            or getattr(manager, "_config_dirty", False)
        )


def _path_key(path: Path) -> str:
    return str(path.resolve(strict=False)).casefold()


def _dataset_id(path: Path) -> str:
    return hashlib.sha256(_path_key(path).encode("utf-8")).hexdigest()[:20]


def _cache_label(status: ExpressionDatasetStatus) -> str:
    if not status.cached_corrections:
        return "No recompute cache"
    return "Cached: " + ", ".join(status.cached_corrections)


def _double_spin(
    minimum: float,
    maximum: float,
    value: float,
    step: float,
    *,
    decimals: int = 3,
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


def _nan_values(values: Iterable[float | None]) -> tuple[float, ...]:
    return tuple(math.nan if value is None else float(value) for value in values)


def _summary_colors(traces) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    for trace in traces:
        key = (trace.group_id, trace.cell_name)
        if key not in output and trace.color:
            output[key] = trace.color
    return output


def _time_axis_label(mode: TimeAxisMode) -> str:
    if mode is TimeAxisMode.RELATIVE:
        return "Time since birth (timepoints)"
    if mode is TimeAxisMode.NORMALIZED:
        return "Normalized lifetime"
    return "Timepoint"


def _with_suffix(path: str | Path, suffix: str) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != suffix:
        destination = destination.with_suffix(suffix)
    return destination


def _display_filename(path: str | None) -> str:
    """Return a display-only basename without filesystem/path resolution."""

    if not path:
        return ""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _nonblank_string(value: object, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _valid_color(value: object) -> bool:
    return isinstance(value, str) and QColor(value).isValid()


def _select_combo_data(combo: QComboBox, value: object) -> bool:
    index = combo.findData(value)
    if index < 0:
        return False
    combo.setCurrentIndex(index)
    return True


__all__ = ["ExpressionComparisonWindow"]
