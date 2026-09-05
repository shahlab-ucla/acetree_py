"""Dedicated vector-profile plot and CSV export for thick-line ROIs."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RoiProfileSeries:
    label: str
    profile: Any
    object_id: str = ""
    timepoint: int = 0
    image_channel: int = 0
    color: str | None = None


def profile_values(profile: Any, reducer: str) -> tuple[float | None, ...]:
    if reducer not in {"mean", "median", "sum"}:
        raise ValueError(f"Unsupported profile reducer: {reducer}")
    values = getattr(profile, reducer, None)
    if values is None:
        return tuple(None for _ in getattr(profile, "distance_um", ()))
    return tuple(None if value is None else float(value) for value in values)


def export_roi_profiles_csv(
    path: str | Path,
    series: Iterable[RoiProfileSeries],
    *,
    reducer: str = "mean",
) -> Path:
    """Export the exact plotted profile rows, retaining gaps and reasons."""

    output = Path(path)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "object_id",
            "label",
            "timepoint",
            "image_channel",
            "distance_um",
            "reducer",
            "value",
            "sample_count",
            "missing_reason",
        ))
        for item in series:
            profile = item.profile
            values = profile_values(profile, reducer)
            distances = tuple(getattr(profile, "distance_um", ()))
            counts = tuple(getattr(profile, "sample_count", (0,) * len(distances)))
            reasons = tuple(getattr(profile, "missing_reasons", (None,) * len(distances)))
            for distance, value, count, reason in zip(
                distances,
                values,
                counts,
                reasons,
                strict=True,
            ):
                writer.writerow((
                    item.object_id,
                    item.label,
                    item.timepoint,
                    item.image_channel + 1,
                    f"{float(distance):.12g}",
                    reducer,
                    "" if value is None else f"{float(value):.12g}",
                    int(count),
                    reason or "",
                ))
    return output


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
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


if _GUI_AVAILABLE:
    class _ProfileNavigationToolbar(NavigationToolbar):
        """Route toolbar Save through the same fresh-snapshot export as SVG."""

        def __init__(self, canvas, parent) -> None:
            self._owner = parent
            super().__init__(canvas, parent)
            self._save_action = next(
                (
                    action for action in self.actions()
                    if "save the figure" in action.toolTip().lower()
                    or action.text().replace("&", "").strip().lower() == "save"
                ),
                None,
            )

        def set_save_enabled(self, enabled: bool) -> None:
            if self._save_action is not None:
                self._save_action.setEnabled(enabled)

        def save_figure(self, *_args) -> None:
            self._owner._choose_svg()


class RoiProfileWindow(QWidget):  # type: ignore[misc]
    """Modeless single-time or multi-time line-profile overlay."""

    def __init__(
        self,
        app_or_profiles: Any = None,
        profiles: Iterable[RoiProfileSeries] | None = None,
        *,
        snapshot: Any = None,
        object_ids: Iterable[Any] | None = None,
        title: str = "Subcellular ROI Profiles",
        parent=None,
    ) -> None:
        if not _GUI_AVAILABLE:
            raise ImportError("ROI profile plotting requires 'acetree-py[gui]'")
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(title)
        self.resize(850, 560)
        if profiles is None and not hasattr(app_or_profiles, "roi_manager"):
            self.app = None
            profiles = () if app_or_profiles is None else app_or_profiles
        else:
            self.app = app_or_profiles
        self._bound_manager = getattr(self.app, "roi_manager", None)
        self._bound_provider = getattr(self.app, "image_provider", None)
        engine = getattr(self.app, "roi_measurement_engine", None)
        self._snapshot = snapshot if snapshot is not None else getattr(engine, "latest_snapshot", None)
        provided = None if profiles is None else tuple(profiles)
        self._object_ids = tuple(
            str(item) for item in object_ids
        ) if object_ids is not None else tuple(dict.fromkeys(
            str(item.object_id) for item in (provided or ())
        ))
        self._profiles = (
            self._profiles_from_snapshot(self._snapshot) if provided is None else provided
        )
        self._valid_series = 0

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Width reducer"))
        self._reducer_combo = QComboBox()
        self._reducer_combo.addItem("Mean", userData="mean")
        self._reducer_combo.addItem("Median", userData="median")
        self._reducer_combo.addItem("Sum", userData="sum")
        self._reducer_combo.setAccessibleName("Spatial profile width reducer")
        controls.addWidget(self._reducer_combo)
        controls.addStretch(1)
        self._export_button = QPushButton("Export CSV…")
        self._export_button.setAccessibleName("Export spatial profiles as CSV")
        self._export_button.setToolTip("Export the exact plotted profile values and gaps")
        controls.addWidget(self._export_button)
        self._export_svg_button = QPushButton("Export SVG…")
        self._export_svg_button.setAccessibleName("Export spatial profiles as SVG")
        self._export_svg_button.clicked.connect(self._choose_svg)
        controls.addWidget(self._export_svg_button)
        layout.addLayout(controls)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setAccessibleName("Spatial profile status")
        layout.addWidget(self._status_label)

        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = _ProfileNavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        self._axes = self._figure.add_subplot(111)

        self._reducer_combo.currentIndexChanged.connect(self.refresh_plot)
        self._export_button.clicked.connect(self._choose_export)
        self.refresh_plot()

    @classmethod
    def from_app(
        cls,
        app: Any,
        *,
        object_ids: Iterable[Any] | None = None,
        parent=None,
    ) -> RoiProfileWindow:
        snapshot = app.roi_measurement_engine.latest_snapshot
        if snapshot is None:
            raise ValueError("Measure this object with spatial profiles first")
        if object_ids is None:
            selected = getattr(app, "current_roi_object_id", None)
            object_ids = (
                (selected,) if selected is not None else
                tuple(track.object_id for track in app.roi_manager.objects)
            )
        object_ids = tuple(str(item) for item in object_ids)
        if not any(
            sample.object_id in object_ids and sample.profile is not None
            for sample in snapshot.samples.values()
        ):
            raise ValueError("No line profiles are available; remeasure with Profiles enabled")
        return cls(app, snapshot=snapshot, object_ids=object_ids, parent=parent)

    def _profiles_from_snapshot(self, snapshot: Any) -> tuple[RoiProfileSeries, ...]:
        if snapshot is None:
            return ()
        selected = set(self._object_ids)
        return tuple(
            RoiProfileSeries(
                label=f"t={sample.timepoint}, channel {sample.image_channel + 1}",
                profile=sample.profile,
                object_id=sample.object_id,
                timepoint=sample.timepoint,
                image_channel=sample.image_channel,
            )
            for sample in snapshot.samples.values()
            if sample.object_id in selected and sample.profile is not None
        )

    @property
    def profiles(self) -> tuple[RoiProfileSeries, ...]:
        return self._profiles

    def set_profiles(self, profiles: Iterable[RoiProfileSeries]) -> None:
        self._profiles = tuple(profiles)
        self._object_ids = tuple(dict.fromkeys(str(item.object_id) for item in self._profiles))
        self.refresh_plot()

    def refresh_plot(self) -> None:
        reducer = str(self._reducer_combo.currentData() or "mean")
        self._axes.clear()
        valid_series = 0
        for item in self._profiles:
            distances = tuple(float(value) for value in item.profile.distance_um)
            values = profile_values(item.profile, reducer)
            plotted = [float("nan") if value is None else value for value in values]
            if any(value is not None for value in values):
                valid_series += 1
            kwargs = {"label": item.label}
            if item.color:
                kwargs["color"] = item.color
            self._axes.plot(distances, plotted, **kwargs)
        self._axes.set_xlabel("Distance from first vertex (µm)")
        self._axes.set_ylabel(f"Raw intensity ({reducer})")
        self._axes.grid(True, alpha=0.25)
        if self._profiles:
            self._axes.legend(loc="best")
        self._valid_series = valid_series
        stale = self._update_export_state()
        if stale:
            self._axes.text(
                0.5, 0.98, "Stale — remeasure before export",
                transform=self._axes.transAxes, ha="center", va="top",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
            )
        self._canvas.draw_idle()

    def _stale_profiles(self) -> int:
        if self.app is None:
            return 0
        if (
            self._snapshot is None or self._bound_provider is None
            or getattr(self.app, "roi_manager", None) is not self._bound_manager
            or getattr(self.app, "image_provider", None) is not self._bound_provider
        ):
            return len(self._profiles)
        context = self._snapshot.prepare_read(
            self.app.roi_manager,
            image_provider=getattr(self.app, "image_provider", None),
        )
        stale = 0
        for item in self._profiles:
            sample = self._snapshot.sample(item.object_id, item.timepoint, item.image_channel)
            if (
                sample is None or sample.profile is not item.profile
                or not context.sample_is_current(item.object_id, item.timepoint)
            ):
                stale += 1
        return stale

    def _update_export_state(self) -> int:
        stale = self._stale_profiles()
        enabled = bool(self._profiles) and not stale
        self._export_button.setEnabled(enabled)
        self._export_svg_button.setEnabled(enabled)
        self._toolbar.set_save_enabled(enabled)
        detail = f"{len(self._profiles)} profile(s); {self._valid_series} contain finite values"
        if stale:
            detail += f"; {stale} stale — remeasure before export"
        self._status_label.setText(detail)
        return stale

    def _assert_exportable(self) -> None:
        if not self._profiles:
            raise RuntimeError("There are no spatial profiles to export")
        if self._update_export_state():
            raise RuntimeError("ROI profiles are stale; remeasure before exporting")

    def on_document_edited(self) -> None:
        self.refresh_plot()

    def on_measurements_updated(self, snapshot: Any = None) -> None:
        if snapshot is None and self.app is not None:
            snapshot = self.app.roi_measurement_engine.latest_snapshot
        profiles = self._profiles_from_snapshot(snapshot)
        # A scalar-only run does not replace the profile data being inspected.
        # The old snapshot remains a reference and its freshness is rechecked.
        if profiles:
            self._snapshot = snapshot
            self._profiles = profiles
        self.refresh_plot()

    def export_csv(self, path: str | Path) -> Path:
        self._assert_exportable()
        reducer = str(self._reducer_combo.currentData() or "mean")
        return export_roi_profiles_csv(path, self._profiles, reducer=reducer)

    def export_svg(self, path: str | Path) -> Path:
        self._assert_exportable()
        destination = Path(path)
        if destination.suffix.lower() != ".svg":
            destination = destination.with_suffix(".svg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(destination, format="svg", bbox_inches="tight")
        return destination

    def _choose_svg(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export ROI profiles", "roi-profiles.svg", "SVG files (*.svg)",
        )
        if chosen:
            try:
                self.export_svg(chosen)
            except (OSError, RuntimeError) as error:
                QMessageBox.warning(self, "Cannot export ROI profiles", str(error))

    def _choose_export(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Export ROI profiles",
            "roi-profiles.csv",
            "CSV files (*.csv)",
        )
        if chosen:
            try:
                self.export_csv(chosen)
            except (OSError, RuntimeError) as error:
                QMessageBox.warning(self, "Cannot export ROI profiles", str(error))


__all__ = [
    "RoiProfileSeries",
    "RoiProfileWindow",
    "export_roi_profiles_csv",
    "profile_values",
]
