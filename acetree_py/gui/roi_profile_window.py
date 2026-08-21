"""Dedicated vector-profile plot and CSV export for thick-line ROIs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


class RoiProfileWindow(QWidget):  # type: ignore[misc]
    """Modeless single-time or multi-time line-profile overlay."""

    def __init__(
        self,
        app_or_profiles: Any = None,
        profiles: Iterable[RoiProfileSeries] | None = None,
        *,
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
            profiles = () if profiles is None else profiles
        self._profiles = tuple(profiles)

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
        layout.addLayout(controls)

        self._status_label = QLabel()
        self._status_label.setAccessibleName("Spatial profile status")
        layout.addWidget(self._status_label)

        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        self._axes = self._figure.add_subplot(111)

        self._reducer_combo.currentIndexChanged.connect(self.refresh_plot)
        self._export_button.clicked.connect(self._choose_export)
        self.refresh_plot()

    @property
    def profiles(self) -> tuple[RoiProfileSeries, ...]:
        return self._profiles

    def set_profiles(self, profiles: Iterable[RoiProfileSeries]) -> None:
        self._profiles = tuple(profiles)
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
        self._status_label.setText(
            f"{len(self._profiles)} profile(s); {valid_series} contain finite values"
        )
        self._canvas.draw_idle()

    def export_csv(self, path: str | Path) -> Path:
        if not self._profiles:
            raise RuntimeError("There are no spatial profiles to export")
        reducer = str(self._reducer_combo.currentData() or "mean")
        return export_roi_profiles_csv(path, self._profiles, reducer=reducer)

    def _choose_export(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Export ROI profiles",
            "roi-profiles.csv",
            "CSV files (*.csv)",
        )
        if chosen:
            self.export_csv(chosen)


__all__ = [
    "RoiProfileSeries",
    "RoiProfileWindow",
    "export_roi_profiles_csv",
    "profile_values",
]
