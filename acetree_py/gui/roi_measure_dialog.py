"""Configuration dialog for atomic subcellular-object measurement runs."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .app import AceTreeApp


ROI_METRIC_CHOICES = (
    ("Integrated intensity", "intensity.sum"),
    ("Mean intensity", "intensity.mean"),
    ("Median intensity", "intensity.median"),
    ("Integrated intensity per length", "intensity.sum_per_length_um"),
    ("Integrated intensity per area", "intensity.sum_per_area_um2"),
    ("Integrated intensity per volume", "intensity.sum_per_volume_um3"),
    ("Integrated intensity per surface area", "intensity.sum_per_surface_area_um2"),
    ("Geometry length", "geometry.length_um"),
    ("Geometry area", "geometry.area_um2"),
    ("Geometry volume", "geometry.volume_um3"),
    ("Geometry surface area", "geometry.surface_area_um2"),
)


try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QLabel,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QDialog = object  # type: ignore[misc,assignment]


class RoiMeasureDialog(QDialog):  # type: ignore[misc]
    """Pick ROI scope, channels, scalar metrics, and optional profile outputs."""

    def __init__(self, app: AceTreeApp | Any, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")
        super().__init__(parent)
        self.app = app
        self.manager = getattr(app, "roi_manager", None)
        self.setWindowTitle("Measure Subcellular Objects")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        introduction = QLabel(
            "Measure raw image intensities for curated subcellular objects. "
            "Results publish only after the complete run succeeds."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        content = QWidget()
        options = QVBoxLayout(content)
        options.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.setMinimumHeight(220)
        layout.addWidget(scroll, 1)

        basic = QGroupBox("Basic")
        form = QFormLayout(basic)
        self._scope_combo = QComboBox()
        for label, value in (
            ("All objects", "all"),
            ("Selected object", "selected"),
            ("Current class", "class"),
        ):
            self._scope_combo.addItem(label, userData=value)
        self._scope_combo.setAccessibleName("Subcellular object measurement scope")
        form.addRow("Scope", self._scope_combo)

        self._time_combo = QComboBox()
        self._time_combo.addItem("All annotated timepoints", userData="all")
        self._time_combo.addItem("Current timepoint", userData="current")
        self._time_combo.setAccessibleName("ROI measurement time scope")
        form.addRow("Time", self._time_combo)

        channel_box = QGroupBox("Image channels")
        channel_layout = QVBoxLayout(channel_box)
        count = 1
        provider = getattr(app, "image_provider", None)
        try:
            count = max(1, int(getattr(provider, "num_channels", 1)))
        except (TypeError, ValueError):
            count = 1
        self._channel_checks: list[Any] = []
        for channel in range(count):
            checkbox = QCheckBox(f"Channel {channel + 1}")
            checkbox.setChecked(True)
            checkbox.setAccessibleName(f"Measure image channel {channel + 1}")
            channel_layout.addWidget(checkbox)
            self._channel_checks.append(checkbox)
        form.addRow(channel_box)

        metric_box = QGroupBox("Scalar outputs")
        metric_layout = QVBoxLayout(metric_box)
        self._metric_checks: dict[str, Any] = {}
        for label, key in ROI_METRIC_CHOICES:
            checkbox = QCheckBox(label)
            checkbox.setChecked(key in {
                "intensity.sum",
                "intensity.mean",
                "intensity.median",
            })
            checkbox.setAccessibleName(f"Include {label.lower()}")
            metric_layout.addWidget(checkbox)
            self._metric_checks[key] = checkbox
        form.addRow(metric_box)
        options.addWidget(basic)

        self._advanced_toggle = QCheckBox("Show advanced options")
        self._advanced_toggle.setAccessibleName("Show advanced ROI measurement options")
        options.addWidget(self._advanced_toggle)

        self._advanced_group = QGroupBox("Advanced")
        advanced = QFormLayout(self._advanced_group)
        self._profiles_check = QCheckBox("Spatial profiles for thick lines")
        self._profiles_check.setAccessibleName("Include thick-line spatial profiles")
        advanced.addRow(self._profiles_check)
        self._profile_step = QDoubleSpinBox()
        self._profile_step.setDecimals(4)
        self._profile_step.setRange(0.0, 100000.0)
        self._profile_step.setSpecialValueText("One XY pixel")
        self._profile_step.setSuffix(" µm")
        self._profile_step.setAccessibleName("Line profile sampling step")
        advanced.addRow("Profile step", self._profile_step)
        self._distributions_check = QCheckBox("Histograms and quantiles")
        self._distributions_check.setAccessibleName("Include ROI intensity distributions")
        advanced.addRow(self._distributions_check)
        self._histogram_bins = QSpinBox()
        self._histogram_bins.setRange(2, 4096)
        self._histogram_bins.setValue(32)
        self._histogram_bins.setAccessibleName("Histogram bin count")
        advanced.addRow("Histogram bins", self._histogram_bins)
        self._advanced_group.setVisible(False)
        options.addWidget(self._advanced_group)
        options.addStretch(1)
        self._advanced_toggle.toggled.connect(self._advanced_group.setVisible)

        self._validation_label = QLabel()
        self._validation_label.setWordWrap(True)
        self._validation_label.setAccessibleName("ROI measurement validation")
        self._validation_label.hide()
        layout.addWidget(self._validation_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        screen = self.screen()
        height = 650 if screen is None else min(650, screen.availableGeometry().height() - 80)
        self.resize(560, max(360, height))

    def _accept_if_valid(self) -> None:
        try:
            self.build_request()
        except (ValueError, RuntimeError) as error:
            self._validation_label.setText(str(error))
            self._validation_label.show()
            return
        self._validation_label.hide()
        self.accept()

    def get_values(self) -> dict[str, Any]:
        return {
            "scope": str(self._scope_combo.currentData()),
            "time_scope": str(self._time_combo.currentData()),
            "channels": tuple(
                index
                for index, checkbox in enumerate(self._channel_checks)
                if checkbox.isChecked()
            ),
            "metric_keys": tuple(
                key for key, checkbox in self._metric_checks.items() if checkbox.isChecked()
            ),
            "include_profiles": self._profiles_check.isChecked(),
            "profile_step_um": (
                float(self._profile_step.value()) if self._profile_step.value() > 0 else None
            ),
            "include_distributions": self._distributions_check.isChecked(),
            "histogram_bins": int(self._histogram_bins.value()),
        }

    def build_request(self) -> Any:
        """Build the headless immutable request consumed by the engine."""

        from ..analysis.roi_measurements import RoiMeasurementRequest

        if self.manager is None:
            raise RuntimeError("No ROI manager is available")
        values = self.get_values()
        scope = values.pop("scope")
        time_scope = values.pop("time_scope")
        if not values["channels"]:
            raise ValueError("Select at least one image channel.")
        if not values["metric_keys"]:
            raise ValueError("Select at least one scalar output.")
        objects = tuple(getattr(self.manager, "objects", ()))
        if not objects:
            raise ValueError("Create a subcellular object before measuring.")
        object_ids = None
        if scope == "selected":
            selected = getattr(self.app, "current_roi_object_id", None)
            if selected is None or not any(item.object_id == selected for item in objects):
                raise ValueError("Select a subcellular object first")
            object_ids = (str(selected),)
        elif scope == "class":
            panel = getattr(self.app, "_subcellular_objects_panel", None)
            class_id = (
                panel.selected_class_id if panel is not None
                else getattr(self.app, "current_roi_class_id", None)
            )
            if class_id is None:
                raise ValueError("Select a subcellular object class first")
            object_ids = tuple(
                str(item.object_id) for item in objects if item.class_id == class_id
            )
            if not object_ids:
                raise ValueError("The selected class has no objects to measure.")
        timepoints = (
            (int(getattr(self.app, "current_time", 1)),)
            if time_scope == "current"
            else None
        )
        return RoiMeasurementRequest(
            # Preserve manager-owned freshness and coordinate-mismatch state;
            # the headless engine accepts either a manager or a bare document.
            document=self.manager,
            object_ids=object_ids,
            timepoints=timepoints,
            **values,
        )


SubcellularMeasureDialog = RoiMeasureDialog

__all__ = ["ROI_METRIC_CHOICES", "RoiMeasureDialog", "SubcellularMeasureDialog"]
