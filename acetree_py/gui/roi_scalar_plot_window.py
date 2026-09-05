"""Scalar time-series plot surface for measured subcellular ROI tracks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..analysis.expression_plot import (
    ExpressionPlotData,
    ExpressionSeriesStyle,
    TemporalSeriesService,
    TimeAxisMode,
    export_scalar_series_csv,
)
from ..analysis.roi_measurements import (
    RoiMeasurementSnapshot,
    RoiScalarSeriesChannel,
    roi_temporal_subjects,
)

ROI_SCALAR_METRIC_LABELS: dict[str, str] = {
    "intensity.sum": "Integrated intensity",
    "intensity.mean": "Mean intensity",
    "intensity.median": "Median intensity",
    "intensity.sum_per_length_um": "Integrated intensity per length",
    "intensity.sum_per_area_um2": "Integrated intensity per area",
    "intensity.sum_per_volume_um3": "Integrated intensity per volume",
    "intensity.sum_per_surface_area_um2": "Integrated intensity per surface area",
    "geometry.length_um": "Geometry length",
    "geometry.area_um2": "Geometry area",
    "geometry.volume_um3": "Geometry volume",
    "geometry.surface_area_um2": "Geometry surface area",
}


class RoiScalarPlotController:
    """Headless controller shared by the ROI plot widget and CSV export."""

    def __init__(
        self,
        roi_manager: Any,
        snapshot: RoiMeasurementSnapshot,
        *,
        image_provider: Any | None = None,
        object_ids: Iterable[Any] | None = None,
        image_channel: int | None = None,
        metric_key: str = "intensity.mean",
        time_mode: TimeAxisMode | str = TimeAxisMode.ABSOLUTE,
        smoothing_sigma: float = 0.0,
    ) -> None:
        self.roi_manager = roi_manager
        self.image_provider = image_provider
        self.snapshot = snapshot
        self.object_ids = self._normalize_object_ids(object_ids)
        channels = self.available_channels
        self.image_channel = (
            channels[0] if image_channel is None and channels else int(image_channel or 0)
        )
        self.metric_key = str(metric_key)
        self.time_mode = TimeAxisMode(time_mode)
        self.smoothing_sigma = float(smoothing_sigma)
        self._plot_data: ExpressionPlotData | None = None

    @property
    def available_channels(self) -> tuple[int, ...]:
        return tuple(sorted({int(value) for value in self.snapshot.channels}))

    @property
    def available_metric_keys(self) -> tuple[str, ...]:
        available: set[str] = set()
        for sample in self.snapshot.samples.values():
            if sample.image_channel == self.image_channel:
                available.update(sample.metrics)
        ordered = [key for key in ROI_SCALAR_METRIC_LABELS if key in available]
        ordered.extend(sorted(available.difference(ordered)))
        return tuple(ordered)

    @property
    def selection_metadata(self) -> dict[str, object]:
        return {
            "object_ids": self.object_ids,
            "source_image_channel": self.image_channel + 1,
            "metric_key": self.metric_key,
            "algorithm_version": self.snapshot.algorithm_version,
            "document_id": self.snapshot.source_document_id,
            "roi_revision": self.snapshot.source_roi_revision,
        }

    @property
    def plot_data(self) -> ExpressionPlotData | None:
        return self._plot_data

    def configure(
        self,
        *,
        object_ids: Iterable[Any] | None = None,
        image_channel: int | None = None,
        metric_key: str | None = None,
        time_mode: TimeAxisMode | str | None = None,
        smoothing_sigma: float | None = None,
    ) -> None:
        if object_ids is not None:
            self.object_ids = self._normalize_object_ids(object_ids)
        if image_channel is not None:
            self.image_channel = int(image_channel)
        if metric_key is not None:
            self.metric_key = str(metric_key)
        if time_mode is not None:
            self.time_mode = TimeAxisMode(time_mode)
        if smoothing_sigma is not None:
            value = float(smoothing_sigma)
            if value < 0:
                raise ValueError("smoothing_sigma cannot be negative")
            self.smoothing_sigma = value

    def update_snapshot(self, snapshot: RoiMeasurementSnapshot) -> None:
        """Install one fully published snapshot and preserve valid selections."""

        self.snapshot = snapshot
        if self.available_channels and self.image_channel not in self.available_channels:
            self.image_channel = self.available_channels[0]
        available_metrics = self.available_metric_keys
        if available_metrics and self.metric_key not in available_metrics:
            self.metric_key = available_metrics[0]
        self.object_ids = self._normalize_object_ids(self.object_ids)
        self._plot_data = None

    def build(self) -> ExpressionPlotData:
        tracks = tuple(
            track
            for track in getattr(self.roi_manager, "objects", ())
            if str(track.object_id) in set(self.object_ids)
        )
        subjects = roi_temporal_subjects(
            tracks,
            object_classes=getattr(self.roi_manager, "classes", ()),
        )
        metric_label = ROI_SCALAR_METRIC_LABELS.get(
            self.metric_key,
            self.metric_key,
        )
        unit = self._metric_unit()
        channel = RoiScalarSeriesChannel(
            snapshot=self.snapshot,
            image_channel=self.image_channel,
            metric_key=self.metric_key,
            label=metric_label,
            unit=unit,
            source=self.roi_manager,
            image_provider=self.image_provider,
        ).as_scalar_series_channel()
        styles = self._styles(tracks)
        self._plot_data = TemporalSeriesService().build(
            subjects,
            channel,
            self.time_mode,
            styles=styles,
            smoothing_sigma=self.smoothing_sigma,
            series_kind="subcellular_objects",
        )
        return self._plot_data

    def stale_samples(self, data: ExpressionPlotData | None = None) -> int:
        current = data or self._plot_data or self.build()
        return sum(
            reason == "stale"
            for series in current.series
            for reason in (series.missing_reasons or ())
        )

    def export_csv(self, path: str | Path) -> Path:
        data = self.build()
        if self.stale_samples(data):
            raise RuntimeError(
                "ROI measurements are stale; remeasure before exporting"
            )
        destination = Path(path)
        export_scalar_series_csv(data, destination)
        return destination

    def _normalize_object_ids(
        self,
        values: Iterable[Any] | None,
    ) -> tuple[str, ...]:
        measured = {key[0] for key in self.snapshot.samples}
        known = {
            str(track.object_id)
            for track in getattr(self.roi_manager, "objects", ())
        }
        if values is None:
            values = measured
        return tuple(
            sorted({str(value) for value in values if str(value) in known})
        )

    def _metric_unit(self) -> str:
        for (object_id, _timepoint, channel), sample in self.snapshot.samples.items():
            if object_id not in self.object_ids or channel != self.image_channel:
                continue
            metric = sample.metrics.get(self.metric_key)
            if metric is not None and metric.unit:
                return metric.unit
        return ""

    def _styles(self, tracks: Iterable[Any]) -> dict[str, ExpressionSeriesStyle]:
        classes = {
            str(item.class_id): item for item in getattr(self.roi_manager, "classes", ())
        }
        result: dict[str, ExpressionSeriesStyle] = {}
        for track in tracks:
            object_class = classes.get(str(track.class_id))
            color = None
            if object_class is not None:
                rgba = tuple(float(value) for value in object_class.color_rgba)
                color = "#{:02x}{:02x}{:02x}".format(
                    *(round(max(0.0, min(1.0, value)) * 255) for value in rgba[:3])
                )
            result[str(track.object_id)] = ExpressionSeriesStyle(color=color)
        return result


try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
    from matplotlib.backends.backend_qtagg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMessageBox,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


if _GUI_AVAILABLE:
    class _RoiNavigationToolbar(NavigationToolbar):
        """Matplotlib navigation whose Save action honors ROI freshness."""

        def __init__(self, canvas, parent) -> None:
            self._roi_plot_owner = parent
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
            self._roi_plot_owner._choose_svg()


class RoiScalarPlotWindow(QWidget):  # type: ignore[misc]
    """Modeless multi-object scalar time-series plot and exact CSV export."""

    def __init__(
        self,
        roi_manager: Any,
        snapshot: RoiMeasurementSnapshot,
        *,
        image_provider: Any | None = None,
        object_ids: Iterable[Any] | None = None,
        image_channel: int | None = None,
        metric_key: str = "intensity.mean",
        title: str = "Subcellular Object Measurements",
        parent=None,
    ) -> None:
        if not _GUI_AVAILABLE:
            raise ImportError("ROI scalar plotting requires 'acetree-py[gui]'")
        super().__init__(parent)
        self.app = None
        self.controller = RoiScalarPlotController(
            roi_manager,
            snapshot,
            image_provider=image_provider,
            object_ids=object_ids,
            image_channel=image_channel,
            metric_key=metric_key,
        )
        self.setWindowFlags(Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(title)
        self.resize(1000, 650)
        self._building_controls = False
        self._plot_data: ExpressionPlotData | None = None
        self._build_ui()
        self._refresh_controls()
        self.refresh_plot()

    @classmethod
    def from_app(
        cls,
        app: Any,
        *,
        object_ids: Iterable[Any] | None = None,
        image_channel: int | None = None,
        metric_key: str = "intensity.mean",
        parent=None,
    ) -> RoiScalarPlotWindow:
        snapshot = app.roi_measurement_engine.latest_snapshot
        if snapshot is None:
            raise RuntimeError("Measure subcellular objects before plotting a track")
        if object_ids is None:
            selected = getattr(app, "current_roi_object_id", None)
            object_ids = None if selected is None else (selected,)
        window = cls(
            app.roi_manager,
            snapshot,
            image_provider=getattr(app, "image_provider", None),
            object_ids=object_ids,
            image_channel=image_channel,
            metric_key=metric_key,
            parent=parent,
        )
        window.app = app
        return window

    @property
    def selected_object_ids(self) -> tuple[str, ...]:
        return self.controller.object_ids

    @property
    def image_channel(self) -> int:
        return self.controller.image_channel

    @property
    def metric_key(self) -> str:
        return self.controller.metric_key

    @property
    def plot_data(self) -> ExpressionPlotData | None:
        return self._plot_data

    @property
    def selection_metadata(self) -> dict[str, object]:
        """Return the selected UUIDs and scalar measurement provenance."""

        return self.controller.selection_metadata

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        form = QFormLayout()
        self._object_list = QListWidget()
        self._object_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._object_list.setAccessibleName("Subcellular objects to plot")
        form.addRow("Objects", self._object_list)
        self._channel_combo = QComboBox()
        self._channel_combo.setAccessibleName("ROI image channel")
        form.addRow("Image channel", self._channel_combo)
        self._metric_combo = QComboBox()
        self._metric_combo.setAccessibleName("ROI scalar metric")
        form.addRow("Metric", self._metric_combo)
        self._time_combo = QComboBox()
        self._time_combo.addItem("Absolute timepoint", TimeAxisMode.ABSOLUTE.value)
        self._time_combo.addItem(
            "Since first segmentation", TimeAxisMode.RELATIVE.value
        )
        self._time_combo.addItem("Normalized track", TimeAxisMode.NORMALIZED.value)
        self._time_combo.setAccessibleName("ROI plot time axis")
        form.addRow("Time axis", self._time_combo)
        self._smoothing = QDoubleSpinBox()
        self._smoothing.setRange(0.0, 20.0)
        self._smoothing.setDecimals(2)
        self._smoothing.setSingleStep(0.25)
        self._smoothing.setSpecialValueText("Off")
        self._smoothing.setAccessibleName("ROI time-series smoothing")
        form.addRow("Smoothing σ", self._smoothing)
        controls_layout.addLayout(form)
        controls_layout.addStretch(1)
        export_row = QHBoxLayout()
        self._export_button = QPushButton("Export CSV…")
        self._export_button.setAccessibleName("Export ROI scalar time series")
        export_row.addWidget(self._export_button)
        self._export_svg_button = QPushButton("Export SVG…")
        self._export_svg_button.setAccessibleName("Export ROI scalar plot")
        export_row.addWidget(self._export_svg_button)
        controls_layout.addLayout(export_row)
        splitter.addWidget(controls)

        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = _RoiNavigationToolbar(self._canvas, self)
        self._axes = self._figure.add_subplot(111)
        plot_layout.addWidget(self._toolbar)
        plot_layout.addWidget(self._canvas, 1)
        splitter.addWidget(plot_host)
        splitter.setSizes((280, 720))
        outer.addWidget(splitter, 1)
        status_row = QHBoxLayout()
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setAccessibleName("ROI scalar plot status")
        status_row.addWidget(self._status, 1)
        outer.addLayout(status_row)

        self._object_list.itemSelectionChanged.connect(self.refresh_plot)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self._metric_combo.currentIndexChanged.connect(self.refresh_plot)
        self._time_combo.currentIndexChanged.connect(self.refresh_plot)
        self._smoothing.valueChanged.connect(self.refresh_plot)
        self._export_button.clicked.connect(self._choose_export)
        self._export_svg_button.clicked.connect(self._choose_svg)

    def _refresh_controls(self) -> None:
        self._building_controls = True
        try:
            selected = set(self.controller.object_ids)
            self._object_list.clear()
            classes = {
                str(item.class_id): item
                for item in getattr(self.controller.roi_manager, "classes", ())
            }
            for track in getattr(self.controller.roi_manager, "objects", ()):
                object_class = classes.get(str(track.class_id))
                class_name = "Object" if object_class is None else object_class.name
                self._object_list.addItem(f"{class_name} #{track.instance_index}")
                item = self._object_list.item(self._object_list.count() - 1)
                item.setData(Qt.UserRole, str(track.object_id))
                item.setSelected(str(track.object_id) in selected)
            self._channel_combo.clear()
            for channel in self.controller.available_channels:
                self._channel_combo.addItem(f"Channel {channel + 1}", channel)
            channel_index = self._channel_combo.findData(self.controller.image_channel)
            if channel_index >= 0:
                self._channel_combo.setCurrentIndex(channel_index)
            self._refresh_metric_combo()
        finally:
            self._building_controls = False

    def _refresh_metric_combo(self) -> None:
        selected_metric = self.controller.metric_key
        self._metric_combo.clear()
        for key in self.controller.available_metric_keys:
            self._metric_combo.addItem(ROI_SCALAR_METRIC_LABELS.get(key, key), key)
        metric_index = self._metric_combo.findData(selected_metric)
        if metric_index >= 0:
            self._metric_combo.setCurrentIndex(metric_index)

    def _on_channel_changed(self, *_args) -> None:
        if self._building_controls:
            return
        channel = self._channel_combo.currentData()
        if channel is not None:
            self.controller.configure(image_channel=int(channel))
        self._building_controls = True
        try:
            self._refresh_metric_combo()
        finally:
            self._building_controls = False
        self.refresh_plot()

    def refresh_plot(self, *_args) -> ExpressionPlotData | None:
        if self._building_controls:
            return
        object_ids = tuple(
            str(item.data(Qt.UserRole)) for item in self._object_list.selectedItems()
        )
        channel = self._channel_combo.currentData()
        metric = self._metric_combo.currentData()
        time_mode = self._time_combo.currentData()
        self.controller.configure(
            object_ids=object_ids,
            image_channel=None if channel is None else int(channel),
            metric_key=None if metric is None else str(metric),
            time_mode=str(time_mode or TimeAxisMode.ABSOLUTE.value),
            smoothing_sigma=float(self._smoothing.value()),
        )
        data = self.controller.build()
        self._plot_data = data
        self._axes.clear()
        missing = 0
        for series in data.series:
            x_values, y_values = series.plot_xy
            missing += sum(value is None for value in series.y_values)
            self._axes.plot(
                x_values,
                y_values,
                label=series.label,
                color=series.color,
                marker="o",
            )
        self._axes.set_xlabel(data.x_label)
        self._axes.set_ylabel(data.y_label)
        self._axes.grid(True, alpha=0.25)
        if data.series:
            self._axes.legend(loc="best")
        stale = self.controller.stale_samples(data)
        export_enabled = stale == 0 and bool(data.series)
        self._export_button.setEnabled(export_enabled)
        self._export_svg_button.setEnabled(export_enabled)
        self._toolbar.set_save_enabled(export_enabled)
        detail = f"{len(data.series)} object track(s); {missing} missing sample(s)"
        if stale:
            detail += f"; {stale} stale — remeasure before export"
        self._status.setText(detail)
        self._canvas.draw_idle()
        return data

    def update_snapshot(self, snapshot: RoiMeasurementSnapshot) -> None:
        self.controller.update_snapshot(snapshot)
        self._refresh_controls()
        self.refresh_plot()

    def on_document_edited(self) -> None:
        self._refresh_controls()
        self.refresh_plot()

    def on_measurements_updated(
        self,
        snapshot: RoiMeasurementSnapshot | None = None,
    ) -> None:
        if snapshot is None and self.app is not None:
            snapshot = self.app.roi_measurement_engine.latest_snapshot
        if snapshot is None:
            raise RuntimeError("No ROI measurement snapshot is available")
        self.update_snapshot(snapshot)

    def export_csv(self, path: str | Path) -> Path:
        return self.controller.export_csv(path)

    def export_svg(self, path: str | Path) -> Path:
        """Export the current fresh plot through the same guarded UI path."""

        data = self.refresh_plot()
        if data is None:
            raise RuntimeError("Plot controls are updating; try exporting again")
        self._assert_exportable(data)
        destination = _with_suffix(path, ".svg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(
            destination,
            format="svg",
            bbox_inches="tight",
            facecolor=self._figure.get_facecolor(),
        )
        return destination

    def _assert_exportable(self, data: ExpressionPlotData | None = None) -> ExpressionPlotData:
        if data is None:
            data = self.controller.build()
        if not data.series:
            raise RuntimeError("Select at least one ROI object before exporting")
        if self.controller.stale_samples(data):
            raise RuntimeError("ROI measurements are stale; remeasure before exporting")
        return data

    def _choose_export(self) -> None:
        chosen, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export ROI scalar time series",
            "subcellular-object-measurements.csv",
            "CSV files (*.csv)",
        )
        if chosen:
            try:
                self.export_csv(chosen)
            except RuntimeError as error:
                QMessageBox.warning(self, "Cannot export ROI data", str(error))

    def _choose_svg(self) -> None:
        chosen, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export ROI scalar plot",
            "subcellular-object-measurements.svg",
            "SVG files (*.svg)",
        )
        if chosen:
            try:
                self.export_svg(chosen)
            except RuntimeError as error:
                QMessageBox.warning(self, "Cannot export ROI plot", str(error))


def _with_suffix(path: str | Path, suffix: str) -> Path:
    destination = Path(path)
    if destination.suffix.lower() == suffix:
        return destination
    return destination.with_suffix(suffix)


__all__ = [
    "ROI_SCALAR_METRIC_LABELS",
    "RoiScalarPlotController",
    "RoiScalarPlotWindow",
]
