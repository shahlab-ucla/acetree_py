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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

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
    """Per-window state for one repository-owned dataset."""

    path: Path
    repository_status: ExpressionDatasetStatus
    included: bool
    label: str
    group_id: str
    color: str
    trace: NativeExpressionTrace | None = None
    request_key: tuple[object, ...] | None = None
    resolution: str = "unprepared"  # unprepared, ready, acquisition_status, error
    availability: TraceAvailability | None = None
    message: str = "Needs preparation"


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
        repository: ExpressionDatasetRepository,
        window_number: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        if not _GUI_AVAILABLE:
            raise ImportError("Expression Comparison requires 'acetree-py[gui]'")
        super().__init__(parent)
        self.app = app
        self.repository = repository
        self.window_number = int(window_number)
        self.setWindowFlags(Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setAcceptDrops(True)
        self.setWindowTitle(
            f"AceTree — Expression Comparison {self.window_number}"
        )
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

        intro = QLabel(
            "Compare one exact cell across independent AceTree XML datasets. "
            "Saved legacy expression is previewable but has unknown channel, "
            "correction, and freshness provenance; image recomputation is cached "
            "in the shared session repository. Comparison reads saved XML/ZIP "
            "snapshots, so save any main-window edits that should be included."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

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
            "Extract saved values or recompute image measurements for every checked row"
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
        exports.addStretch(1)
        self._btn_export_csv = QPushButton("Save exact comparison CSV…")
        self._btn_export_svg = QPushButton("Export plot as SVG…")
        self._btn_export_csv.clicked.connect(self._choose_csv_path)
        self._btn_export_svg.clicked.connect(self._choose_svg_path)
        exports.addWidget(self._btn_export_csv)
        exports.addWidget(self._btn_export_svg)
        layout.addLayout(exports)
        return area

    # -- Dataset membership ---------------------------------------------

    def _add_session_datasets(self) -> None:
        """Prepopulate later windows from the shared application repository."""

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
            path=status.config_path,
            repository_status=status,
            included=True,
            label=self._unique_dataset_label(status.config_path.stem),
            group_id="all",
            color=_DEFAULT_COLORS[len(self._datasets) % len(_DEFAULT_COLORS)],
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
            status_item.setToolTip(
                f"Source fingerprint: {state.repository_status.source_fingerprint}\n"
                f"Session snapshot: {state.repository_status.snapshot_token}"
            )
            self._dataset_table.setItem(row, self.COL_STATUS, status_item)

            cache_item = QTableWidgetItem(_cache_label(state.repository_status))
            cache_item.setFlags(cache_item.flags() & ~Qt.ItemIsEditable)
            cache_item.setData(Qt.UserRole, key)
            self._dataset_table.setItem(row, self.COL_CACHE, cache_item)

            path_item = QTableWidgetItem(str(state.path))
            path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
            path_item.setData(Qt.UserRole, key)
            path_item.setToolTip(str(state.path))
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
                label = state.path.stem
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
        previous = self._cell_combo.currentText().strip()
        states = self._included_states() or tuple(self._datasets.values())
        availability: dict[str, int] = {}
        for state in states:
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
        cell = self._cell_combo.currentText().strip()
        states = self._included_states()
        available = 0
        for state in states:
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
        # Saved legacy fields never require opening an image provider.  Keep
        # provider discovery behind the user's explicit recompute choice.
        if self._source_mode() != "recomputed":
            return
        channel_counts: list[int] = []
        for state in self._included_states():
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
        self._sync_source_controls()
        if self._source_mode() == "saved":
            self._legacy_ack.setChecked(False)
        else:
            self._refresh_image_channel_range()
        self._invalidate_local_traces()

    def _sync_source_controls(self) -> None:
        """Keep source-specific controls readable without changing data state."""

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
        if self._updating_controls:
            return
        self._update_cell_availability()
        if self._source_mode() == "saved":
            self._legacy_ack.setChecked(False)
        self._invalidate_local_traces()

    def _invalidate_local_traces(self) -> None:
        for state in self._datasets.values():
            state.trace = None
            state.request_key = None
            state.resolution = "unprepared"
            state.availability = None
            state.message = "Needs preparation"
        self._rebuild_dataset_table()
        self._refresh_plot()

    def _source_mode(self) -> str:
        return str(self._source_combo.currentData() or "saved")

    def _request_key(self) -> tuple[object, ...]:
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

                def progress_cb(
                    channel_index: int,
                    num_channels: int,
                    timepoint: int,
                    num_timepoints: int,
                ) -> bool:
                    local_total = max(1, int(num_channels) * int(num_timepoints))
                    local_done = int(channel_index) * int(num_timepoints) + int(timepoint)
                    fraction = (dataset_index + min(1.0, local_done / local_total)) / total
                    progress.setValue(min(999, int(1000 * fraction)))
                    progress.setLabelText(
                        f"{state.label}: channel {channel_index + 1}/{num_channels}, "
                        f"timepoint {timepoint}/{num_timepoints}"
                    )
                    QApplication.processEvents()
                    return not progress.wasCanceled()

                try:
                    # Another already-open comparison may have populated the
                    # shared cache since this row was last refreshed.
                    state.repository_status = self.repository.status(state.path)
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
                    state.repository_status = self.repository.status(state.path)
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
        if not hasattr(self, "_axes"):
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
            self._draw_empty("Add and check one or more AceTree XML datasets.")
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

        datasets = tuple(self._expression_dataset(state) for state in states)
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
        try:
            center = CenterStatistic(str(self._center_combo.currentData()))
            band = BandStatistic(str(self._band_combo.currentData()))
            if center is CenterStatistic.NONE:
                band = BandStatistic.NONE
            data = self._service.build(
                datasets,
                cell_names=[self._cell_combo.currentText().strip()],
                channel_key=channel_key,
                channel_label=channel_label,
                channel_unit=channel_unit,
                time_mode=TimeAxisMode(str(self._time_combo.currentData())),
                grid=GridSpec(
                    domain=GridDomain(str(self._grid_domain.currentData())),
                    step=(
                        None
                        if str(self._time_combo.currentData())
                        == TimeAxisMode.NORMALIZED.value
                        else self._grid_step.value()
                    ),
                    normalized_points=self._normalized_points.value(),
                ),
                smoothing=SmoothingSpec(
                    sigma=(
                        self._smoothing_sigma.value()
                        if self._smoothing_check.isChecked()
                        else 0.0
                    )
                ),
                summary=SummarySpec(center=center, band=band),
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
            message += " Check the legacy provenance acknowledgement to enable export."
        if data.has_data and not has_visible_artists:
            message += " Enable individual traces or a center line to export an SVG."
        if data.warnings:
            message += f" {len(data.warnings)} model warning(s)."
        self._status_label.setText(message)

    def _expression_dataset(self, state: _DatasetViewState) -> ExpressionDataset:
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
                dataset_id=_dataset_id(state.path),
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

    def _legacy_export_unacknowledged(
        self, states: tuple[_DatasetViewState, ...]
    ) -> bool:
        return any(
            state.trace is not None
            and state.trace.provenance.source is ExpressionTraceSource.SAVED_LEGACY
            for state in states
        ) and not self._legacy_ack.isChecked()

    def _set_export_enabled(
        self, enabled: bool, *, csv_enabled: bool | None = None
    ) -> None:
        self._btn_export_csv.setEnabled(enabled if csv_enabled is None else csv_enabled)
        self._btn_export_svg.setEnabled(enabled)
        self._toolbar.set_save_enabled(enabled)

    def _exportable_snapshot(
        self, *, allow_status_only: bool = False
    ) -> ExpressionComparisonData:
        """Revalidate every included source before returning the rendered data."""

        states = self._included_states()
        request_key = self._request_key()
        if not states:
            raise RuntimeError("No datasets are included in this comparison.")
        for state in states:
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
        if any(Path(url.toLocalFile()).suffix.lower() == ".xml" for url in urls):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if Path(url.toLocalFile()).suffix.lower() == ".xml"
        ]
        if paths:
            self.add_dataset_paths(paths)
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


__all__ = ["ExpressionComparisonWindow"]
