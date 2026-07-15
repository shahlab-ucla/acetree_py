"""Modeless, non-destructive workbench for reviewing a global tracking draft.

The dialog owns presentation state only.  It never calls the tracking pipeline
or mutates a :class:`~acetree_py.core.nuclei_manager.NucleiManager` directly.
The host may either inject ``analysis_starter`` / ``accept_callback`` callables
or connect the public signals and finish the operation through the matching
public methods::

    dialog.analysisRequested.connect(start_worker)
    # Worker callbacks, delivered on the Qt thread:
    dialog.update_analysis_progress(done, total, message, run_id)
    dialog.finish_analysis(result, revision, run_id, document_token)
    dialog.fail_analysis(error, run_id)

    dialog.acceptRequested.connect(apply_proposal)
    dialog.accept_succeeded()  # or dialog.accept_failed(error)

This split lets ``AceTreeApp`` choose the appropriate worker implementation
without weakening the pre-commit review boundary.
"""

from __future__ import annotations

import html
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from statistics import fmean
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .tracking_preview import ExpandedTrackingPreview, expand_tracking_preview

if TYPE_CHECKING:
    from ..tracking.api import (
        Calibration,
        ComponentSpec,
        Detection,
        TrackingRequest,
        TrackingResult,
    )
    from ..tracking.registry import TrackingRegistry
    from .viewer_integration import ViewerIntegration

logger = logging.getLogger(__name__)


AnalysisStarter = Callable[["TrackingRequest", int, "GlobalTrackingDialog"], Any]
DetectorPreviewStarter = Callable[
    ["ComponentSpec", int, int, "GlobalTrackingDialog"],
    Any,
]
AcceptCallback = Callable[["TrackingResult", Any], Any]
RevisionGetter = Callable[[], Any]
FrameNavigator = Callable[[int], None]
CurrentFrameGetter = Callable[[], int]
DatasetEmptyGetter = Callable[[], bool]


class GlobalTrackingDialog(QDialog):
    """Configure, inspect, and explicitly accept one global tracking proposal."""

    analysisRequested = Signal(object, int)
    detectorPreviewRequested = Signal(object, int, int)
    cancelRequested = Signal(int)
    acceptRequested = Signal(object, object)
    draftAccepted = Signal(int)
    draftDiscarded = Signal()

    CONFIGURING = "configuring"
    RUNNING = "running"
    CANCELING = "canceling"
    READY = "ready"
    OUTDATED = "outdated"
    STALE = "stale"
    EMPTY = "empty"
    FAILED = "failed"
    ACCEPTING = "accepting"
    DETECTOR_READY = "detector_ready"

    def __init__(
        self,
        start_time: int,
        end_time: int,
        num_channels: int = 1,
        parent=None,
        *,
        initial_request: TrackingRequest | None = None,
        registry: TrackingRegistry | None = None,
        viewer_integration: ViewerIntegration | None = None,
        calibration: Calibration | None = None,
        analysis_starter: AnalysisStarter | None = None,
        detector_preview_starter: DetectorPreviewStarter | None = None,
        accept_callback: AcceptCallback | None = None,
        revision_getter: RevisionGetter | None = None,
        navigate_to_frame: FrameNavigator | None = None,
        current_frame_getter: CurrentFrameGetter | None = None,
        dataset_empty_getter: DatasetEmptyGetter | None = None,
    ) -> None:
        super().__init__(parent)
        if start_time < 1 or end_time < start_time:
            raise ValueError("Global tracking needs a valid 1-based time range")

        from ..tracking.registry import get_default_registry

        self._range_start = int(start_time)
        self._range_end = int(end_time)
        self._num_channels = max(1, int(num_channels))
        self._registry = registry or get_default_registry()
        self._viewer_integration = viewer_integration
        self._calibration = calibration
        self._analysis_starter = analysis_starter
        self._detector_preview_starter = detector_preview_starter
        self._accept_callback = accept_callback
        self._revision_getter = revision_getter
        self._navigate_callback = navigate_to_frame
        self._current_frame_getter = current_frame_getter
        self._dataset_empty_getter = dataset_empty_getter

        self._proposal: TrackingResult | None = None
        self._expanded_preview: ExpandedTrackingPreview | None = None
        self._expected_revision: Any = None
        self._proposal_document_token: Any = None
        self._analysis_start_token: Any = None
        self._detector_preview: ExpandedTrackingPreview | None = None
        self._detector_preview_frame: int | None = None
        self._detector_preview_document_token: Any = None
        self._state = self.CONFIGURING
        self._state_before_run = self.CONFIGURING
        self._run_serial = 0
        self._active_run_id: int | None = None
        self._active_run_kind: str | None = None
        self._active_detector_frame: int | None = None
        self._active_detector_spec: ComponentSpec | None = None
        self._detector_frame_changed_during_run = False
        self._cancel_handle: Callable[[], Any] | None = None
        self._accepted = False
        self._discard_emitted = False
        self._cleaned_up = False
        self._navigating_review = False
        self._solo_channel_visibility: list[tuple[object, bool]] | None = None
        self._original_view = self._capture_view_state()

        self.setWindowTitle("Review Initial Tracking Draft")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setMinimumSize(760, 520)
        self.resize(1080, 720)
        # Worker callbacks may still be queued while a cooperative cancel is
        # completing. Parent ownership provides deterministic, safe cleanup.
        self.setAccessibleName("Global tracking draft workbench")

        self._build_ui()
        self._apply_initial_request(initial_request)
        self._connect_parameter_signals()
        self._apply_accessibility_descriptions()
        self._update_detector_button_text()
        self._validate_settings()
        self._set_state(
            self.CONFIGURING,
            "Choose settings, then build a preview. Nothing is added until you accept the draft.",
        )

    @property
    def state(self) -> str:
        """Current workflow state, exposed for host integration and tests."""

        return self._state

    @property
    def proposal(self) -> TrackingResult | None:
        """The currently reviewed immutable proposal, if one exists."""

        return self._proposal

    @property
    def active_run_id(self) -> int | None:
        """Opaque identifier hosts should echo with worker callbacks."""

        return self._active_run_id

    @property
    def active_run_kind(self) -> str | None:
        """Whether the active worker is a detector test or a full draft."""

        return self._active_run_kind

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        title = QLabel(
            "<span style='font-size:16px'><b>Review initial tracking draft</b></span>"
            "<br><span style='color:#9aa0a6'>Analysis is read-only. Inspect every "
            "frame before explicitly accepting the draft.</span>"
        )
        title.setWordWrap(True)
        title.setAccessibleName("Global tracking review introduction")
        outer.addWidget(title)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        outer.addLayout(columns, stretch=1)

        configure_group = QGroupBox("1. Configure")
        configure_group.setMinimumWidth(300)
        configure_layout = QVBoxLayout(configure_group)
        self._settings_widget = QWidget()
        form = QFormLayout(self._settings_widget)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._start_spin = QSpinBox()
        self._start_spin.setRange(self._range_start, self._range_end)
        self._start_spin.setValue(self._range_start)
        self._start_spin.setToolTip("First timepoint included in detection")
        self._start_spin.setAccessibleName("First tracking timepoint")
        form.addRow("Start at:", self._start_spin)

        self._end_spin = QSpinBox()
        self._end_spin.setRange(self._range_start, self._range_end)
        self._end_spin.setValue(self._range_end)
        self._end_spin.setToolTip("Last timepoint included in detection")
        self._end_spin.setAccessibleName("Last tracking timepoint")
        form.addRow("End at:", self._end_spin)

        self._detector_combo = QComboBox()
        for descriptor in self._registry.detector_descriptors():
            self._detector_combo.addItem(descriptor.display_name, descriptor.plugin_id)
        self._detector_combo.setToolTip("Method used to find nucleus-like bright blobs")
        self._detector_combo.setAccessibleName("Nucleus detector")
        form.addRow("Detector:", self._detector_combo)

        self._tracker_combo = QComboBox()
        for descriptor in self._registry.tracker_descriptors():
            self._tracker_combo.addItem(descriptor.display_name, descriptor.plugin_id)
        self._tracker_combo.setToolTip("Method used to connect detections over time")
        self._tracker_combo.setAccessibleName("Detection linker")
        form.addRow("Tracker:", self._tracker_combo)

        self._channel_spin = QSpinBox()
        self._channel_spin.setRange(1, self._num_channels)
        self._channel_spin.setValue(1)
        self._channel_spin.setToolTip(
            f"Image channel containing the nuclear signal (1–{self._num_channels})"
        )
        self._channel_spin.setAccessibleName("Nuclear image channel")
        form.addRow("Image channel:", self._channel_spin)

        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.05, 100.0)
        self._radius_spin.setDecimals(2)
        self._radius_spin.setValue(4.0)
        self._radius_spin.setSuffix(" µm")
        self._radius_spin.setToolTip("Approximate physical radius of a nucleus")
        self._radius_spin.setAccessibleName("Expected nucleus radius")
        form.addRow("Nucleus radius:", self._radius_spin)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1_000_000.0)
        self._threshold_spin.setDecimals(4)
        self._threshold_spin.setValue(5.0)
        self._threshold_spin.setToolTip(
            "Minimum detector response. Higher values keep fewer, stronger candidates."
        )
        self._threshold_spin.setAccessibleName("Detection threshold")
        form.addRow("Detection threshold:", self._threshold_spin)

        self._distance_spin = QDoubleSpinBox()
        self._distance_spin.setRange(0.05, 1_000.0)
        self._distance_spin.setDecimals(2)
        self._distance_spin.setValue(8.0)
        self._distance_spin.setSuffix(" µm")
        self._distance_spin.setToolTip("Largest plausible movement between linked frames")
        self._distance_spin.setAccessibleName("Maximum movement per frame")
        form.addRow("Maximum movement:", self._distance_spin)

        self._gap_spin = QSpinBox()
        self._gap_spin.setRange(0, 20)
        self._gap_spin.setValue(1)
        self._gap_spin.setToolTip("How many missing frames may be bridged by interpolation")
        self._gap_spin.setAccessibleName("Missing frames allowed")
        form.addRow("Missing frames:", self._gap_spin)
        configure_layout.addWidget(self._settings_widget)

        self._advanced_toggle = QCheckBox("Show advanced detection options")
        self._advanced_toggle.toggled.connect(self._set_advanced_visible)
        configure_layout.addWidget(self._advanced_toggle)

        self._advanced_widget = QWidget()
        advanced_layout = QVBoxLayout(self._advanced_widget)
        advanced_layout.setContentsMargins(18, 0, 0, 0)
        self._subpixel_check = QCheckBox("Refine positions below one pixel")
        self._subpixel_check.setChecked(True)
        self._median_check = QCheckBox("Apply a 3×3×3 median filter")
        self._median_check.setChecked(False)
        advanced_layout.addWidget(self._subpixel_check)
        advanced_layout.addWidget(self._median_check)
        self._advanced_widget.setVisible(False)
        configure_layout.addWidget(self._advanced_widget)

        self._settings_error = QLabel()
        self._settings_error.setWordWrap(True)
        self._settings_error.setAccessibleName("Tracking settings problem")
        self._settings_error.setStyleSheet("QLabel { color: #d9a441; }")
        self._settings_error.hide()
        configure_layout.addWidget(self._settings_error)

        configure_actions = QHBoxLayout()
        self._reset_button = QPushButton("Restore Defaults")
        self._reset_button.clicked.connect(self._restore_defaults)
        self._reset_button.setToolTip("Restore recommended starting values")
        self._detector_preview_button = QPushButton("&Test Current Frame")
        self._detector_preview_button.clicked.connect(self.start_detector_preview)
        self._detector_preview_button.setToolTip(
            "Run only the detector on the current full 3D stack. No links or "
            "accept-capable draft are created."
        )
        self._preview_button = QPushButton("Build &Full Draft")
        self._preview_button.setDefault(True)
        self._preview_button.clicked.connect(self.start_analysis)
        self._preview_button.setToolTip(
            "Detect every requested frame and link positions into a reviewable draft"
        )
        configure_actions.addWidget(self._reset_button)
        configure_actions.addStretch()
        configure_actions.addWidget(self._detector_preview_button)
        configure_actions.addWidget(self._preview_button)
        configure_layout.addLayout(configure_actions)

        self._detector_status_label = QLabel(
            "Fast tuning: test the detector on the image currently shown before "
            "building the full draft. Detector tests cannot be accepted."
        )
        self._detector_status_label.setWordWrap(True)
        self._detector_status_label.setAccessibleName("Current-frame detector test status")
        self._detector_status_label.setStyleSheet("QLabel { color: #b8bec7; }")
        configure_layout.addWidget(self._detector_status_label)
        columns.addWidget(configure_group, stretch=0)

        review_group = QGroupBox("2. Review")
        review_layout = QVBoxLayout(review_group)
        self._banner = QLabel()
        self._banner.setWordWrap(True)
        self._banner.setMinimumHeight(52)
        self._banner.setTextFormat(Qt.PlainText)
        self._banner.setAccessibleName("Global tracking draft status")
        review_layout.addWidget(self._banner)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setAccessibleName("Global tracking analysis progress")
        self._progress_bar.hide()
        review_layout.addWidget(self._progress_bar)

        self._summary_label = QLabel("No preview has been built yet.")
        self._summary_label.setWordWrap(True)
        self._summary_label.setAccessibleName("Global tracking draft summary")
        review_layout.addWidget(self._summary_label)

        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setAccessibleName("Global tracking warnings")
        self._warning_label.hide()
        review_layout.addWidget(self._warning_label)

        self._legend_label = QLabel(
            "Legend: ○ proposed detection; ◇ interpolated gap; □/× diagnostic "
            "candidate; ━ movement path; ⊕ predicted search region. White marks "
            "the table selection; purple rings are a detector-only current-frame "
            "test; amber throughout means the draft must be rebuilt."
        )
        self._legend_label.setWordWrap(True)
        self._legend_label.setAccessibleName("Tracking preview legend")
        self._legend_label.setStyleSheet("QLabel { color: #b8bec7; }")
        review_layout.addWidget(self._legend_label)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            [
                "Time",
                "Detections",
                "Draft positions",
                "Links in",
                "Track starts",
                "Mean quality",
                "Review",
            ]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.currentCellChanged.connect(self._on_table_current_cell_changed)
        self._table.cellClicked.connect(self._on_table_row_clicked)
        self._table.setAccessibleName("Per-frame global tracking review")
        self._table.setAccessibleDescription(
            "One row per requested frame, including frames with no detections"
        )
        review_layout.addWidget(self._table, stretch=1)

        navigation = QHBoxLayout()
        self._previous_button = QPushButton("◀ Previous Frame")
        self._previous_button.setEnabled(False)
        self._previous_button.clicked.connect(lambda: self._step_review_row(-1))
        self._next_button = QPushButton("Next Frame ▶")
        self._next_button.setEnabled(False)
        self._next_button.clicked.connect(lambda: self._step_review_row(1))
        self._overlay_check = QCheckBox("Show draft on image")
        self._overlay_check.setChecked(True)
        self._overlay_check.toggled.connect(self._set_overlay_visible)
        if self._viewer_integration is None or self._calibration is None:
            self._overlay_check.setEnabled(False)
            self._overlay_check.setToolTip(
                "The host has not attached a viewer and dataset calibration"
            )
        navigation.addWidget(self._previous_button)
        navigation.addWidget(self._next_button)
        navigation.addStretch()
        navigation.addWidget(self._overlay_check)
        review_layout.addLayout(navigation)

        self._solo_channel_check = QCheckBox("Solo detection channel while reviewing")
        self._solo_channel_check.setToolTip(
            "Temporarily hide other image channels; their visibility is restored on close"
        )
        self._solo_channel_check.setAccessibleDescription(
            self._solo_channel_check.toolTip()
        )
        self._solo_channel_check.toggled.connect(self._set_detection_channel_solo)
        if self._viewer_integration is None:
            self._solo_channel_check.setEnabled(False)
        review_layout.addWidget(self._solo_channel_check)
        columns.addWidget(review_group, stretch=1)

        footer = QHBoxLayout()
        self._cancel_run_button = QPushButton("Cancel Analysis")
        self._cancel_run_button.clicked.connect(self.cancel_analysis)
        self._cancel_run_button.hide()
        self._discard_button = QPushButton("&Discard Draft")
        self._discard_button.clicked.connect(self._discard_draft)
        self._discard_button.setToolTip("Close without changing the dataset")
        self._accept_button = QPushButton("&Accept Draft")
        self._accept_button.setEnabled(False)
        self._accept_button.clicked.connect(self._accept_draft)
        self._accept_button.setToolTip("Commit the visible draft as one undoable edit")
        footer.addWidget(self._cancel_run_button)
        footer.addStretch()
        footer.addWidget(self._discard_button)
        footer.addWidget(self._accept_button)
        outer.addLayout(footer)

    def _connect_parameter_signals(self) -> None:
        for widget in (
            self._channel_spin,
            self._radius_spin,
            self._threshold_spin,
        ):
            widget.valueChanged.connect(self._detector_parameters_changed)
        for widget in (
            self._start_spin,
            self._end_spin,
            self._distance_spin,
            self._gap_spin,
        ):
            widget.valueChanged.connect(self._tracking_parameters_changed)
        self._detector_combo.currentIndexChanged.connect(
            self._detector_parameters_changed
        )
        self._tracker_combo.currentIndexChanged.connect(
            self._tracking_parameters_changed
        )
        self._subpixel_check.toggled.connect(self._detector_parameters_changed)
        self._median_check.toggled.connect(self._detector_parameters_changed)
        self._channel_spin.valueChanged.connect(self._refresh_solo_detection_channel)

    def _apply_accessibility_descriptions(self) -> None:
        """Expose explanatory tooltips through assistive-technology APIs."""

        for widget in (
            self._start_spin,
            self._end_spin,
            self._detector_combo,
            self._tracker_combo,
            self._channel_spin,
            self._radius_spin,
            self._threshold_spin,
            self._distance_spin,
            self._gap_spin,
            self._detector_preview_button,
            self._preview_button,
            self._overlay_check,
            self._solo_channel_check,
            self._discard_button,
            self._accept_button,
        ):
            if widget.toolTip():
                widget.setAccessibleDescription(widget.toolTip())

    def _apply_initial_request(self, request: TrackingRequest | None) -> None:
        if request is None:
            return
        if request.scope.kind != "global":
            raise ValueError("GlobalTrackingDialog requires a global TrackingRequest")
        self._select_combo_value(self._detector_combo, request.detector.plugin_id)
        self._select_combo_value(self._tracker_combo, request.tracker.plugin_id)
        self._start_spin.setValue(request.scope.start_frame)
        self._end_spin.setValue(request.scope.end_frame)
        detector = request.detector.settings
        tracker = request.tracker.settings
        if "TARGET_CHANNEL" in detector:
            self._channel_spin.setValue(int(detector["TARGET_CHANNEL"]))
        if "RADIUS" in detector:
            self._radius_spin.setValue(float(detector["RADIUS"]))
        if "THRESHOLD" in detector:
            self._threshold_spin.setValue(float(detector["THRESHOLD"]))
        if "DO_SUBPIXEL_LOCALIZATION" in detector:
            self._subpixel_check.setChecked(bool(detector["DO_SUBPIXEL_LOCALIZATION"]))
        if "DO_MEDIAN_FILTERING" in detector:
            self._median_check.setChecked(bool(detector["DO_MEDIAN_FILTERING"]))
        if "LINKING_MAX_DISTANCE" in tracker:
            self._distance_spin.setValue(float(tracker["LINKING_MAX_DISTANCE"]))
        if "MAX_FRAME_GAP" in tracker:
            self._gap_spin.setValue(max(0, int(tracker["MAX_FRAME_GAP"]) - 1))

    @staticmethod
    def _select_combo_value(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def export_settings(self) -> dict[str, Any]:
        """Return the current common settings for reopening or rerunning."""

        return {
            "start_time": self._start_spin.value(),
            "end_time": self._end_spin.value(),
            "detector_id": self._detector_combo.currentData(),
            "tracker_id": self._tracker_combo.currentData(),
            "channel": self._channel_spin.value(),
            "radius_um": self._radius_spin.value(),
            "threshold": self._threshold_spin.value(),
            "max_distance_um": self._distance_spin.value(),
            "missing_frames": self._gap_spin.value(),
            "subpixel": self._subpixel_check.isChecked(),
            "median_filter": self._median_check.isChecked(),
            "show_overlay": self._overlay_check.isChecked(),
        }

    def get_detector_spec(self) -> ComponentSpec:
        """Build detector settings without requiring any tracker configuration."""

        error = self._detector_validation_error()
        if error:
            raise ValueError(error)

        from ..tracking.api import ComponentSpec

        detector_id = str(self._detector_combo.currentData())
        detector_settings = self._registry.default_settings(detector_id)
        detector_common = {
            "TARGET_CHANNEL": self._channel_spin.value(),
            "RADIUS": self._radius_spin.value(),
            "THRESHOLD": self._threshold_spin.value(),
            "DO_SUBPIXEL_LOCALIZATION": self._subpixel_check.isChecked(),
            "DO_MEDIAN_FILTERING": self._median_check.isChecked(),
        }
        detector_schema = self._registry.get_descriptor(detector_id).settings_schema
        detector_settings.update(
            (key, value)
            for key, value in detector_common.items()
            if key in detector_schema
        )
        return ComponentSpec(detector_id, detector_settings)

    def get_request(self) -> TrackingRequest:
        """Build the immutable global request represented by the form."""

        error = self._settings_validation_error()
        if error:
            raise ValueError(error)

        from ..tracking.api import ComponentSpec, TrackingRequest, TrackingScope

        detector_spec = self.get_detector_spec()
        tracker_id = str(self._tracker_combo.currentData())
        tracker_settings = self._registry.default_settings(tracker_id)
        gap_frames = self._gap_spin.value()
        max_distance = self._distance_spin.value()
        tracker_common = {
            "LINKING_MAX_DISTANCE": max_distance,
            "ALLOW_GAP_CLOSING": gap_frames > 0,
            "GAP_CLOSING_MAX_DISTANCE": max_distance,
            "MAX_FRAME_GAP": gap_frames + 1 if gap_frames > 0 else 1,
            "ALLOW_TRACK_SPLITTING": False,
            "ALLOW_TRACK_MERGING": False,
        }
        tracker_schema = self._registry.get_descriptor(tracker_id).settings_schema
        tracker_settings.update(
            (key, value) for key, value in tracker_common.items() if key in tracker_schema
        )
        return TrackingRequest(
            detector=detector_spec,
            tracker=ComponentSpec(tracker_id, tracker_settings),
            scope=TrackingScope(
                "global",
                self._start_spin.value(),
                self._end_spin.value(),
            ),
        )

    def start_detector_preview(self) -> int | None:
        """Run only the configured detector on the viewer's current 3D frame."""

        if self._active_run_id is not None or self._state in {
            self.READY,
            self.ACCEPTING,
        }:
            return None
        try:
            detector = self.get_detector_spec()
            frame = self._read_current_frame()
        except Exception as exc:
            self._show_detector_failure(
                "Check the detector settings and current frame, then try again.",
                exc,
            )
            return None

        self._clear_detector_preview()
        self._state_before_run = self._state
        self._run_serial += 1
        run_id = self._run_serial
        self._active_run_id = run_id
        self._active_run_kind = "detector"
        self._active_detector_frame = frame
        self._active_detector_spec = detector
        self._detector_frame_changed_during_run = False
        self._analysis_start_token = self._read_document_token()
        self._cancel_handle = None
        self._set_running(True)
        self._set_state(
            self.RUNNING,
            f"Testing the detector on the complete 3D stack at t={frame}. "
            "No linking is running and nothing can be accepted.",
        )
        self._set_detector_status(
            f"Testing detector at t={frame}… This reads one 3D frame only."
        )
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Loading current frame…")

        try:
            if self._detector_preview_starter is not None:
                handle = self._detector_preview_starter(
                    detector,
                    frame,
                    run_id,
                    self,
                )
                if callable(handle):
                    self._cancel_handle = handle
                elif callable(getattr(handle, "cancel", None)):
                    self._cancel_handle = handle.cancel
                elif isinstance(handle, tuple) and len(handle) >= 1:
                    self.finish_detector_preview(handle[0], frame, run_id)
            else:
                self.detectorPreviewRequested.emit(detector, frame, run_id)
        except Exception as exc:
            self.fail_detector_preview(exc, run_id)
            return None
        return run_id

    def start_analysis(self) -> int | None:
        """Enter the running state and hand the request to the host worker."""

        if self._active_run_id is not None or self._state == self.ACCEPTING:
            return None
        try:
            request = self.get_request()
        except Exception as exc:
            self._show_failure("Check the tracking settings and try again.", exc)
            return None

        self._clear_detector_preview()
        self._state_before_run = self._state
        self._run_serial += 1
        run_id = self._run_serial
        self._active_run_id = run_id
        self._active_run_kind = "full"
        self._active_detector_frame = None
        self._active_detector_spec = None
        self._detector_frame_changed_during_run = False
        self._analysis_start_token = self._read_document_token()
        self._cancel_handle = None
        self._set_running(True)
        self._set_state(
            self.RUNNING,
            f"Analyzing every frame from t={request.scope.start_frame} to "
            f"t={request.scope.end_frame}…",
        )
        self._set_detector_status(
            "Building the full draft. The transient detector test has been cleared."
        )
        self._warning_label.hide()
        total = request.scope.end_frame - request.scope.start_frame + 1
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Starting…")

        try:
            if self._analysis_starter is not None:
                handle = self._analysis_starter(request, run_id, self)
                if callable(handle):
                    self._cancel_handle = handle
                elif callable(getattr(handle, "cancel", None)):
                    self._cancel_handle = handle.cancel
                elif isinstance(handle, tuple) and len(handle) >= 2:
                    # Convenient headless/test fallback; production hosts should
                    # normally return a worker or cancellation handle.
                    self.finish_analysis(handle[0], handle[1], run_id)
            else:
                self.analysisRequested.emit(request, run_id)
        except Exception as exc:
            self.fail_analysis(exc, run_id)
            return None
        return run_id

    def update_analysis_progress(
        self,
        done: int,
        total: int,
        message: str,
        run_id: int | None = None,
    ) -> bool:
        """Update progress from a host worker; return whether the run is active."""

        if not self._callback_matches(run_id):
            return False
        self._progress_bar.setRange(0, max(1, int(total)))
        self._progress_bar.setValue(max(0, min(int(done), max(1, int(total)))))
        self._progress_bar.setFormat(f"{message}  %p%")
        return self._state == self.RUNNING

    def finish_analysis(
        self,
        proposal: TrackingResult,
        expected_revision: Any = None,
        run_id: int | None = None,
        document_token: Any = None,
    ) -> bool:
        """Install a worker result as an uncommitted preview on the Qt thread."""

        if not self._callback_matches(run_id) or self._active_run_kind != "full":
            return False
        if self._state == self.CANCELING:
            self.analysis_cancelled(run_id)
            return False
        if proposal.request.scope.kind != "global":
            self.fail_analysis(
                ValueError("The analysis worker returned a non-global proposal"),
                run_id,
            )
            return False

        self._active_run_id = None
        self._active_run_kind = None
        self._cancel_handle = None
        self._set_running(False)
        self._proposal = proposal
        self._expanded_preview = expand_tracking_preview(proposal)
        self._expected_revision = expected_revision
        # Read the live token on delivery. The worker's snapshot token can be
        # older than the document by the time this queued callback reaches Qt.
        live_token = self._read_document_token()
        current_token = live_token if live_token is not None else document_token
        self._proposal_document_token = current_token
        record_changed_during_run = not self._dataset_is_empty()
        changed_during_run = (
            self._analysis_start_token is not None
            and current_token is not None
            and current_token != self._analysis_start_token
        )
        self._populate_review()
        self._show_viewer_preview(
            stale=changed_during_run or record_changed_during_run
        )
        self._preview_button.setText("Update &Full Draft")
        self._set_detector_status(
            "Full draft ready. Change detector settings to enable another "
            "current-frame detector test."
        )

        count = self._expanded_preview.proposed_count
        if count <= 0:
            self._set_state(
                self.EMPTY,
                "No positions were found. Adjust the detector settings and update the preview.",
                warning=True,
            )
            self._accept_button.setEnabled(False)
        elif record_changed_during_run:
            self._set_state(
                self.STALE,
                "Curated positions were added while analysis was running. The draft "
                "cannot be accepted; Undo those edits or use Auto Forward.",
                warning=True,
            )
            self._accept_button.setEnabled(False)
        elif changed_during_run:
            self._set_state(
                self.STALE,
                "The dataset changed while analysis was running. Update Preview before accepting.",
                warning=True,
            )
            self._accept_button.setEnabled(False)
        else:
            self._set_state(
                self.READY,
                "Draft ready. Inspect the frame table and image overlay before accepting.",
                success=True,
            )
            self._accept_button.setText(f"&Accept {count} Positions")
            self._accept_button.setEnabled(True)
        self._validate_settings()
        return True

    @Slot(object)
    def finish_worker_result(self, payload: object) -> None:
        """Install the completion tuple emitted by ``TrackingAnalysisWorker``."""

        try:
            proposal, revision, run_id, document_token = payload  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            self.fail_analysis(exc)
            return
        self.finish_analysis(
            proposal,
            revision,
            int(run_id),
            document_token,
        )

    def finish_detector_preview(
        self,
        detections: tuple[Detection, ...],
        frame: int,
        run_id: int | None = None,
        document_token: Any = None,
    ) -> bool:
        """Install a detector-only result if its captured context is still current."""

        if not self._callback_matches(run_id) or self._active_run_kind != "detector":
            return False
        if self._state == self.CANCELING:
            self.analysis_cancelled(run_id)
            return False
        frame = int(frame)
        try:
            detections = tuple(detections)
            if any(detection.frame != frame for detection in detections):
                raise ValueError("Detector preview returned a position for another frame")
        except Exception as exc:
            return self.fail_detector_preview(exc, run_id)

        live_token = self._read_document_token()
        current_token = live_token if live_token is not None else document_token
        token_changed = (
            self._analysis_start_token is not None
            and current_token is not None
            and current_token != self._analysis_start_token
        )
        try:
            current_frame = self._read_current_frame()
        except Exception:
            current_frame = None
        frame_changed = (
            self._detector_frame_changed_during_run
            or self._active_detector_frame != frame
            or current_frame != frame
        )
        try:
            detector_changed = self.get_detector_spec() != self._active_detector_spec
        except Exception:
            detector_changed = True

        self._active_run_id = None
        self._active_run_kind = None
        self._active_detector_frame = None
        self._active_detector_spec = None
        self._cancel_handle = None
        self._set_running(False)
        if token_changed or frame_changed or detector_changed:
            reason = (
                "The dataset changed while the detector was running."
                if token_changed
                else (
                    f"The viewer moved away from t={frame} while the detector was running."
                    if frame_changed
                    else "The detector settings changed while the test was running."
                )
            )
            self._clear_detector_preview()
            self._restore_after_detector_run(
                f"{reason} The late result was ignored; test the current frame again.",
                warning=True,
            )
            return False

        from .tracking_preview import expand_detector_preview

        self._detector_preview = expand_detector_preview(detections)
        self._detector_preview_frame = frame
        self._detector_preview_document_token = current_token
        qualities = [float(detection.quality) for detection in detections]
        quality_text = f"; mean quality {fmean(qualities):.3g}" if qualities else ""
        status = (
            f"t={frame}: {len(detections)} detector candidate(s){quality_text}. "
            "Purple rings are detector-only and cannot be accepted or linked."
        )
        self._set_detector_status(status, color="#caa7ff")
        self._show_detector_viewer_preview(detections)
        if self._proposal is None:
            if self._dataset_is_empty():
                self._set_state(
                    self.DETECTOR_READY,
                    f"Detector test ready for t={frame}. Tune the detector or build the "
                    "full draft to compute links.",
                    success=True,
                )
            else:
                self._set_state(
                    self.STALE,
                    f"Detector test ready for t={frame}, but curated positions are now "
                    "present. Whole-dataset tracking is disabled; use Auto Forward or "
                    "Undo those edits.",
                    warning=True,
                )
        else:
            self._restore_after_detector_run(
                f"Detector test ready for t={frame}. The full draft remains "
                "out of date until it is rebuilt."
            )
        self._validate_settings()
        return True

    @Slot(object)
    def finish_detector_preview_worker_result(self, payload: object) -> None:
        """Install a detector completion tuple emitted by the shared worker."""

        try:
            detections, frame, run_id, document_token = payload  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            self.fail_detector_preview(exc)
            return
        self.finish_detector_preview(
            tuple(detections),
            int(frame),
            int(run_id),
            document_token,
        )

    def fail_detector_preview(
        self,
        error: Exception | str,
        run_id: int | None = None,
    ) -> bool:
        """Finish a failed detector test without disturbing a full draft."""

        if not self._callback_matches(run_id) or self._active_run_kind != "detector":
            return False
        if self._state == self.CANCELING:
            return self.analysis_cancelled(run_id)
        self._active_run_id = None
        self._active_run_kind = None
        self._active_detector_frame = None
        self._active_detector_spec = None
        self._cancel_handle = None
        self._set_running(False)
        exc = error if isinstance(error, Exception) else RuntimeError(str(error))
        logger.warning("Current-frame detector preview failed: %s", exc)
        self._clear_detector_preview()
        self._show_detector_failure(
            "Detector test failed; no overlay or draft was retained.",
            exc,
        )
        return True

    def fail_analysis(
        self,
        error: Exception | str,
        run_id: int | None = None,
    ) -> bool:
        """Finish a failed host run without changing or accepting the dataset."""

        if not self._callback_matches(run_id) or self._active_run_kind != "full":
            return False
        if self._state == self.CANCELING:
            return self.analysis_cancelled(run_id)
        self._active_run_id = None
        self._active_run_kind = None
        self._cancel_handle = None
        self._set_running(False)
        exc = error if isinstance(error, Exception) else RuntimeError(str(error))
        logger.warning("Global tracking preview failed: %s", exc)
        self._show_failure(
            "Global tracking could not build a draft. No changes were made.",
            exc,
        )
        if self._proposal is not None:
            self._show_viewer_preview(stale=True)
        return True

    def analysis_cancelled(self, run_id: int | None = None) -> bool:
        """Finish a canceled worker run and keep settings available for retry."""

        if not self._callback_matches(run_id):
            return False
        run_kind = self._active_run_kind
        self._active_run_id = None
        self._active_run_kind = None
        self._active_detector_frame = None
        self._active_detector_spec = None
        self._cancel_handle = None
        self._set_running(False)
        if run_kind == "detector":
            self._clear_detector_preview()
            self._set_detector_status(
                "Detector test canceled. No partial overlay was retained."
            )
            self._restore_after_detector_run(
                "Detector test canceled. No changes were made."
            )
            return True
        if self._proposal is None:
            self._set_state(
                self.CONFIGURING,
                "Analysis canceled. No changes were made; adjust settings or try again.",
            )
        else:
            self._set_state(
                self.OUTDATED,
                "Analysis canceled. The visible preview is from the previous settings and "
                "cannot be accepted until it is updated.",
                warning=True,
            )
            self._show_viewer_preview(stale=True)
        self._validate_settings()
        return True

    def cancel_analysis(self) -> None:
        """Request cooperative cancellation of the active host worker."""

        if self._active_run_id is None or self._state not in {self.RUNNING, self.CANCELING}:
            return
        if self._state == self.CANCELING:
            return
        self._state = self.CANCELING
        self._cancel_run_button.setEnabled(False)
        self._progress_bar.setFormat("Canceling after the current frame…")
        if self._cancel_handle is not None:
            try:
                self._cancel_handle()
            except Exception:
                logger.debug("Global tracking cancellation handle failed", exc_info=True)
        self.cancelRequested.emit(self._active_run_id)

    def sync_document_revision(self) -> bool:
        """Invalidate a proposal when the host document token has changed."""

        current = self._read_document_token()
        if (
            self._detector_preview is not None
            and current is not None
            and self._detector_preview_document_token is not None
            and current != self._detector_preview_document_token
        ):
            self._clear_detector_preview()
            self._set_detector_status(
                "The dataset changed; the current-frame detector test was cleared."
            )
        if not self._dataset_is_empty():
            self._accept_button.setEnabled(False)
            if self._proposal is not None:
                self.mark_stale(
                    "The dataset now contains curated positions. Whole-dataset "
                    "tracking cannot be accepted; Undo those edits or use Auto Forward."
                )
            elif self._state not in {self.RUNNING, self.CANCELING, self.ACCEPTING}:
                self._set_state(
                    self.STALE,
                    "The dataset now contains curated positions. Whole-dataset "
                    "tracking is disabled; Undo those edits or use Auto Forward.",
                    warning=True,
                )
            self._validate_settings()
            return False
        if self._proposal is None:
            if self._state == self.STALE:
                if self._detector_preview is not None:
                    self._set_state(
                        self.DETECTOR_READY,
                        "The nuclei record is empty again. The detector test remains "
                        "visible; tune it or build the full draft.",
                        success=True,
                    )
                else:
                    self._set_state(
                        self.CONFIGURING,
                        "The nuclei record is empty again. Test a representative frame "
                        "or build the full draft.",
                    )
            elif self._state == self.DETECTOR_READY and self._detector_preview is None:
                self._set_state(
                    self.CONFIGURING,
                    "The detector test was cleared. Test the current frame again or "
                    "build the full draft.",
                )
            self._validate_settings()
            return True
        if self._state in {
            self.RUNNING,
            self.CANCELING,
            self.ACCEPTING,
            self.STALE,
        }:
            self._validate_settings()
            return self._state != self.STALE
        if (
            current is not None
            and self._proposal_document_token is not None
            and current != self._proposal_document_token
        ):
            self.mark_stale(
                "The dataset changed while this draft was open. Update Preview before accepting."
            )
            return False
        return True

    def mark_stale(self, message: str | None = None) -> None:
        """Public fallback for hosts that cannot provide a revision getter."""

        if self._proposal is None:
            return
        self._accept_button.setEnabled(False)
        self._set_state(
            self.STALE,
            message
            or "The dataset changed while this draft was open. Update Preview before accepting.",
            warning=True,
        )
        self._show_viewer_preview(stale=True)

    def _accept_draft(self) -> None:
        if self._proposal is None or self._expanded_preview is None or self._state != self.READY:
            return
        if not self.sync_document_revision():
            return
        self._state = self.ACCEPTING
        self._accept_button.setEnabled(False)
        self._preview_button.setEnabled(False)
        self._discard_button.setEnabled(False)
        self._set_state(
            self.ACCEPTING,
            "Applying the reviewed draft as one undoable edit…",
        )
        if self._accept_callback is not None:
            try:
                self._accept_callback(self._proposal, self._expected_revision)
            except Exception as exc:
                self.accept_failed(exc)
                return
            self.accept_succeeded()
        else:
            self.acceptRequested.emit(self._proposal, self._expected_revision)

    def accept_succeeded(self) -> None:
        """Close after the host confirms that the proposal was committed."""

        if self._proposal is None or self._accepted:
            return
        count = self._expanded_preview.proposed_count if self._expanded_preview else 0
        self._accepted = True
        self.draftAccepted.emit(count)
        self.accept()

    def accept_failed(self, error: Exception | str) -> None:
        """Return to review after a host-side commit failure."""

        exc = error if isinstance(error, Exception) else RuntimeError(str(error))
        self._state = self.READY
        self._preview_button.setEnabled(True)
        self._discard_button.setEnabled(True)
        self._accept_button.setEnabled(True)
        self._set_state(
            self.READY,
            "The draft could not be applied. It remains uncommitted and available for review.",
            warning=True,
        )
        self._warning_label.setText(
            f"<b>Details</b><br>{html.escape(type(exc).__name__)}: "
            f"{html.escape(str(exc) or 'Unknown error')}"
        )
        self._warning_label.show()

    def _populate_review(self) -> None:
        assert self._proposal is not None
        assert self._expanded_preview is not None
        proposal = self._proposal
        preview = self._expanded_preview
        detections = [
            detection
            for detection in proposal.detections
            if detection.detection_id not in proposal.existing_anchors
        ]
        target_ids = {edge.target_id for edge in proposal.edges}
        roots = [detection for detection in detections if detection.detection_id not in target_ids]
        original_gaps = sum(edge.kind == "gap" for edge in proposal.edges)
        self._summary_label.setText(
            f"<b>{len(detections)} detected spots</b> · "
            f"{preview.proposed_count} positions to add after interpolation · "
            f"{len(roots)} tracks · {len(preview.links)} adjacent link segments · "
            f"{original_gaps} bridged gaps ({preview.interpolated_count} interpolated positions)"
        )

        detections_by_frame: dict[int, list[Any]] = defaultdict(list)
        for detection in detections:
            detections_by_frame[detection.frame].append(detection)
        spots_by_frame: dict[int, list[Any]] = defaultdict(list)
        for spot in preview.spots:
            if spot.kind != "seed":
                spots_by_frame[spot.frame].append(spot)
        links_by_frame: dict[int, list[Any]] = defaultdict(list)
        preview_by_id = preview.by_id
        for link in preview.links:
            links_by_frame[preview_by_id[link.target_id].frame].append(link)
        roots_by_frame: dict[int, int] = defaultdict(int)
        for root in roots:
            roots_by_frame[root.frame] += 1

        start = proposal.request.scope.start_frame
        end = proposal.request.scope.end_frame
        frames = list(range(start, end + 1))
        self._table.setRowCount(len(frames))
        warning_frames = _warning_frames(tuple(proposal.warnings))
        for row, frame in enumerate(frames):
            frame_detections = detections_by_frame.get(frame, [])
            frame_spots = spots_by_frame.get(frame, [])
            interpolated = sum(spot.kind == "interpolated" for spot in frame_spots)
            qualities = [float(detection.quality) for detection in frame_detections]
            quality = f"{fmean(qualities):.3g}" if qualities else "—"
            notes: list[str] = []
            if not frame_spots:
                notes.append("No draft positions")
            elif interpolated:
                notes.append(f"{interpolated} interpolated gap")
            if roots_by_frame.get(frame, 0) and frame > start:
                notes.append(f"{roots_by_frame[frame]} new track start")
            if frame in warning_frames:
                notes.append("Analysis warning")
            review = "; ".join(notes) if notes else "Inspect"
            values = (
                str(frame),
                str(len(frame_detections)),
                str(len(frame_spots)),
                str(len(links_by_frame.get(frame, []))),
                str(roots_by_frame.get(frame, 0)),
                quality,
                review,
            )
            first_preview_id = frame_spots[0].preview_id if frame_spots else None
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, frame)
                item.setData(Qt.UserRole + 1, first_preview_id)
                if "No draft positions" in review or frame in warning_frames:
                    item.setForeground(QColor("#d9a441"))
                self._table.setItem(row, column, item)

        warnings = tuple(proposal.warnings)
        if warnings:
            items = "<br>".join(f"• {html.escape(item)}" for item in warnings)
            self._warning_label.setText(
                "<b>Analysis warnings</b><br>"
                f"{items}<br><span style='color:#9aa0a6'>Warnings do not change the "
                "dataset; inspect the affected frames before accepting.</span>"
            )
        else:
            self._warning_label.setText(
                "The requested range completed. Automated completion is not a quality "
                "guarantee; inspect frames with no positions or new track starts."
            )
        self._warning_label.show()
        if frames:
            self._table.selectRow(0)
        self._update_review_navigation_buttons()

    def _detector_parameters_changed(self, *_args) -> None:
        self._parameters_changed(detector_changed=True)

    def _tracking_parameters_changed(self, *_args) -> None:
        self._parameters_changed(detector_changed=False)

    def _parameters_changed(self, *, detector_changed: bool) -> None:
        self._validate_settings()
        if self._state in {self.RUNNING, self.CANCELING, self.ACCEPTING}:
            return
        cleared_detector = detector_changed and self._detector_preview is not None
        if cleared_detector:
            self._clear_detector_preview()
            self._set_detector_status(
                "Detector settings changed. Test the current frame again to refresh "
                "the purple rings."
            )
        if self._proposal is None:
            if cleared_detector:
                self._set_state(
                    self.CONFIGURING,
                    "Detector settings changed. Test the current frame again, or "
                    "build the full draft.",
                )
            return
        self._accept_button.setEnabled(False)
        self._set_state(
            self.OUTDATED,
            "Settings changed. The visible preview uses the previous settings; "
            "update the full draft before accepting.",
            warning=True,
        )
        self._show_viewer_preview(stale=True)
        self._validate_settings()

    def _detector_validation_error(self) -> str:
        if self._detector_combo.count() == 0:
            return "No compatible detector is installed."
        if not 1 <= self._channel_spin.value() <= self._num_channels:
            return f"Choose an image channel between 1 and {self._num_channels}."
        return ""

    def _settings_validation_error(self) -> str:
        detector_error = self._detector_validation_error()
        if detector_error:
            return detector_error
        if self._tracker_combo.count() == 0:
            return "No compatible tracker is installed."
        if self._start_spin.value() > self._end_spin.value():
            return "The start time must not be later than the end time."
        if not self._dataset_is_empty():
            return (
                "Whole-dataset tracking requires an empty nuclei record. Undo "
                "curation edits or use Auto Forward for a selected cell."
            )
        return ""

    def _validate_settings(self) -> bool:
        error = self._settings_validation_error()
        self._settings_error.setText(error)
        self._settings_error.setVisible(bool(error))
        can_build = not error and self._state not in {
            self.RUNNING,
            self.CANCELING,
            self.ACCEPTING,
        }
        self._preview_button.setEnabled(bool(can_build))
        detector_error = self._detector_validation_error()
        can_test = not detector_error and self._state not in {
            self.RUNNING,
            self.CANCELING,
            self.ACCEPTING,
            self.READY,
        }
        self._detector_preview_button.setEnabled(bool(can_test))
        return not bool(error)

    def _set_running(self, running: bool) -> None:
        self._settings_widget.setEnabled(not running)
        self._advanced_toggle.setEnabled(not running)
        self._advanced_widget.setEnabled(not running)
        self._reset_button.setEnabled(not running)
        self._preview_button.setEnabled(
            not running and not self._settings_validation_error()
        )
        self._detector_preview_button.setEnabled(
            not running
            and not self._detector_validation_error()
            and self._state != self.READY
        )
        self._accept_button.setEnabled(False if running else self._accept_button.isEnabled())
        self._cancel_run_button.setVisible(running)
        self._cancel_run_button.setEnabled(running)
        self._progress_bar.setVisible(running)

    def _show_failure(self, message: str, exc: Exception) -> None:
        self._accept_button.setEnabled(False)
        self._set_state(self.FAILED, message, warning=True)
        self._warning_label.setText(
            f"<b>Details</b><br>{html.escape(type(exc).__name__)}: "
            f"{html.escape(str(exc) or 'Unknown error')}"
        )
        self._warning_label.show()

    def _show_detector_failure(self, message: str, exc: Exception) -> None:
        detail = f"{type(exc).__name__}: {str(exc) or 'Unknown error'}"
        text = f"{message} {detail}"
        self._set_detector_status(text, color="#e06c75")
        self._restore_after_detector_run(message, warning=True)

    def _set_detector_status(
        self,
        message: str,
        *,
        color: str = "#b8bec7",
    ) -> None:
        """Present detector-test status consistently to sighted and AT users."""

        self._detector_status_label.setText(message)
        self._detector_status_label.setAccessibleDescription(message)
        self._detector_status_label.setStyleSheet(f"QLabel {{ color: {color}; }}")

    def _restore_after_detector_run(
        self,
        message: str,
        *,
        warning: bool = False,
    ) -> None:
        """Restore full-draft state after a detector-only worker finishes."""

        if self._proposal is None:
            state = self.STALE if not self._dataset_is_empty() else self.CONFIGURING
            self._accept_button.setEnabled(False)
            self._set_state(state, message, warning=warning or state == self.STALE)
            self._validate_settings()
            return

        state = self._state_before_run
        if state not in {self.READY, self.OUTDATED, self.STALE, self.EMPTY, self.FAILED}:
            state = self.OUTDATED
        ready = state == self.READY and self._dataset_is_empty()
        self._accept_button.setEnabled(ready)
        self._set_state(
            self.READY if ready else state,
            message,
            success=ready and not warning,
            warning=warning or not ready,
        )
        self._show_viewer_preview(stale=not ready)
        self._validate_settings()

    def _dataset_is_empty(self) -> bool:
        if self._dataset_empty_getter is None:
            return True
        try:
            return bool(self._dataset_empty_getter())
        except Exception:
            logger.debug("Could not verify the whole-dataset empty-record guard", exc_info=True)
            return False

    def _set_state(
        self,
        state: str,
        message: str,
        *,
        success: bool = False,
        warning: bool = False,
    ) -> None:
        self._state = state
        if success:
            background, border = "#173d2b", "#42b883"
        elif warning:
            background, border = "#453716", "#d9a441"
        elif state == self.FAILED:
            background, border = "#4a2024", "#e06c75"
        else:
            background, border = "#252a32", "#667085"
        self._banner.setStyleSheet(
            f"QLabel {{ background: {background}; border-left: 4px solid {border}; "
            "color: #f1f3f4; padding: 8px; border-radius: 3px; }"
        )
        self._banner.setText(message)
        self._banner.setAccessibleDescription(message)
        self._banner.setToolTip(message)

    def _restore_defaults(self) -> None:
        self._start_spin.setValue(self._range_start)
        self._end_spin.setValue(self._range_end)
        self._channel_spin.setValue(1)
        self._radius_spin.setValue(4.0)
        self._threshold_spin.setValue(5.0)
        self._distance_spin.setValue(8.0)
        self._gap_spin.setValue(1)
        self._subpixel_check.setChecked(True)
        self._median_check.setChecked(False)

    def _set_advanced_visible(self, visible: bool) -> None:
        self._advanced_widget.setVisible(visible)

    def _on_table_current_cell_changed(
        self,
        row: int,
        _column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if not self._navigating_review:
            self._navigate_to_row(row)

    def _on_table_row_clicked(self, row: int, _column: int) -> None:
        self._navigate_to_row(row)

    def _step_review_row(self, step: int) -> None:
        if self._table.rowCount() == 0:
            return
        row = self._table.currentRow()
        if row < 0:
            row = 0 if step > 0 else self._table.rowCount() - 1
        else:
            row = max(0, min(self._table.rowCount() - 1, row + step))
        self._navigate_to_row(row)

    def _navigate_to_row(self, row: int) -> None:
        if not (0 <= row < self._table.rowCount()):
            return
        item = self._table.item(row, 0)
        frame = int(item.data(Qt.UserRole))
        preview_id = item.data(Qt.UserRole + 1)
        self._navigating_review = True
        try:
            self._table.selectRow(row)
            self._navigate_to_frame(frame)
        finally:
            self._navigating_review = False
        if self._viewer_integration is not None:
            self._viewer_integration.highlight_tracking_preview(
                None if preview_id is None else str(preview_id)
            )
        self._update_review_navigation_buttons()

    def _update_review_navigation_buttons(self) -> None:
        row = self._table.currentRow()
        count = self._table.rowCount()
        self._previous_button.setEnabled(count > 0 and row > 0)
        self._next_button.setEnabled(count > 0 and 0 <= row < count - 1)

    def sync_viewer_position(self, frame: int | None = None) -> None:
        """Follow main-view time navigation without feeding it back again."""

        if frame is None:
            try:
                frame = self._read_current_frame()
            except Exception:
                frame = None
        if frame is None:
            return
        frame = int(frame)
        self._update_detector_button_text(frame)
        if (
            self._active_run_kind == "detector"
            and self._active_detector_frame is not None
            and frame != self._active_detector_frame
        ):
            self._detector_frame_changed_during_run = True
        if (
            self._detector_preview is not None
            and self._detector_preview_frame is not None
            and frame != self._detector_preview_frame
        ):
            old_frame = self._detector_preview_frame
            self._clear_detector_preview()
            self._set_detector_status(
                f"Viewer moved from t={old_frame} to t={frame}; the detector-only "
                "overlay was cleared. Test this frame when ready."
            )
            if self._proposal is None and self._state == self.DETECTOR_READY:
                self._set_state(
                    self.CONFIGURING,
                    "Choose a representative frame and test the detector, or build "
                    "the full draft.",
                )
            self._validate_settings()
        if self._navigating_review or self._table.rowCount() == 0:
            return
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None or int(item.data(Qt.UserRole)) != int(frame):
                continue
            self._navigating_review = True
            try:
                self._table.selectRow(row)
            finally:
                self._navigating_review = False
            preview_id = item.data(Qt.UserRole + 1)
            if self._viewer_integration is not None:
                self._viewer_integration.highlight_tracking_preview(
                    None if preview_id is None else str(preview_id)
                )
            self._update_review_navigation_buttons()
            return

    def _read_current_frame(self) -> int:
        if self._current_frame_getter is not None:
            frame = int(self._current_frame_getter())
        else:
            app = getattr(self._viewer_integration, "app", None)
            value = getattr(app, "current_time", None)
            if value is None:
                raise ValueError("The current viewer timepoint is unavailable")
            frame = int(value)
        if frame < self._range_start or frame > self._range_end:
            raise ValueError(
                f"Current frame t={frame} is outside the available range "
                f"{self._range_start}–{self._range_end}"
            )
        return frame

    def _update_detector_button_text(self, frame: int | None = None) -> None:
        if frame is None:
            try:
                frame = self._read_current_frame()
            except Exception:
                frame = None
        if frame is None:
            self._detector_preview_button.setText("&Test Current Frame")
        else:
            self._detector_preview_button.setText(f"&Test Detector at t={int(frame)}")

    def _navigate_to_frame(self, frame: int) -> None:
        if self._navigate_callback is not None:
            self._navigate_callback(frame)
            return
        app = getattr(self._viewer_integration, "app", None)
        setter = getattr(app, "set_time", None)
        if callable(setter):
            setter(frame)

    def _show_viewer_preview(self, *, stale: bool) -> None:
        if (
            self._viewer_integration is None
            or self._calibration is None
            or self._proposal is None
        ):
            return
        self._viewer_integration.show_tracking_preview(
            self._proposal,
            self._calibration,
            visible=self._overlay_check.isChecked(),
            stale=stale,
        )

    def _show_detector_viewer_preview(
        self,
        detections: tuple[Detection, ...],
    ) -> None:
        if self._viewer_integration is None or self._calibration is None:
            return
        shower = getattr(self._viewer_integration, "show_detector_preview", None)
        if callable(shower):
            shower(
                detections,
                self._calibration,
                visible=self._overlay_check.isChecked(),
            )

    def _clear_detector_preview(self) -> None:
        self._detector_preview = None
        self._detector_preview_frame = None
        self._detector_preview_document_token = None
        if self._viewer_integration is None:
            return
        clearer = getattr(self._viewer_integration, "clear_detector_preview", None)
        if callable(clearer):
            clearer()

    def _set_overlay_visible(self, visible: bool) -> None:
        if self._viewer_integration is not None:
            self._viewer_integration.set_tracking_preview_visible(visible)
            setter = getattr(
                self._viewer_integration,
                "set_detector_preview_visible",
                None,
            )
            if callable(setter):
                setter(visible)

    def _set_detection_channel_solo(self, enabled: bool) -> None:
        app = getattr(self._viewer_integration, "app", None)
        if app is None:
            return
        if enabled:
            if self._solo_channel_visibility is None:
                capture = getattr(
                    self._viewer_integration,
                    "capture_image_channel_visibility",
                    None,
                )
                if callable(capture):
                    self._solo_channel_visibility = capture()
                else:
                    self._solo_channel_visibility = [
                        (layer, bool(getattr(layer, "visible", True)))
                        for layer in getattr(app, "_image_layers", ())
                    ]
            selected = self._channel_spin.value() - 1
            solo = getattr(
                self._viewer_integration,
                "set_detection_channel_solo",
                None,
            )
            if callable(solo):
                solo(selected)
            else:
                for index, layer in enumerate(getattr(app, "_image_layers", ())):
                    layer.visible = index == selected
        else:
            self._restore_channel_visibility()

    def _refresh_solo_detection_channel(self, _value: int) -> None:
        if not self._solo_channel_check.isChecked():
            return
        self._set_detection_channel_solo(True)

    def _restore_channel_visibility(self) -> None:
        app = getattr(self._viewer_integration, "app", None)
        if app is None or self._solo_channel_visibility is None:
            return
        restore = getattr(
            self._viewer_integration,
            "restore_image_channel_visibility",
            None,
        )
        if callable(restore):
            restore(self._solo_channel_visibility)
        else:
            for layer, visible in self._solo_channel_visibility:
                try:
                    layer.visible = visible
                except RuntimeError:
                    pass
        self._solo_channel_visibility = None

    def _clear_viewer_preview(self) -> None:
        if self._viewer_integration is not None:
            self._viewer_integration.clear_tracking_preview()

    def _read_document_token(self) -> Any:
        if self._revision_getter is None:
            return None
        try:
            return self._revision_getter()
        except Exception:
            logger.debug("Could not read tracking document token", exc_info=True)
            return None

    def _callback_matches(self, run_id: int | None) -> bool:
        if self._active_run_id is None:
            return False
        return run_id is None or int(run_id) == self._active_run_id

    def _capture_view_state(self) -> dict[str, Any]:
        app = getattr(self._viewer_integration, "app", None)
        if app is None:
            return {}
        return {
            "time": getattr(app, "current_time", None),
            "plane": getattr(app, "current_plane", None),
            "cell_name": getattr(app, "current_cell_name", None),
            "selection_anchor": getattr(app, "selection_anchor", None),
            "tracking": getattr(app, "tracking", None),
        }

    def _restore_view_state(self) -> None:
        app = getattr(self._viewer_integration, "app", None)
        if app is None or not self._original_view:
            return
        try:
            for attribute, key in (
                ("current_time", "time"),
                ("current_plane", "plane"),
                ("current_cell_name", "cell_name"),
                ("selection_anchor", "selection_anchor"),
                ("tracking", "tracking"),
            ):
                if key in self._original_view:
                    setattr(app, attribute, self._original_view[key])
            updater = getattr(app, "update_display", None)
            if callable(updater):
                updater()
        except (AttributeError, RuntimeError):
            logger.debug("Could not restore the pre-review viewer state")

    def _request_close_cancellation(self) -> None:
        if self._active_run_id is None:
            return
        run_id = self._active_run_id
        if self._cancel_handle is not None:
            try:
                self._cancel_handle()
            except Exception:
                logger.debug("Global tracking close cancellation failed", exc_info=True)
        self.cancelRequested.emit(run_id)
        # Invalidate the run immediately; late worker callbacks are ignored.
        self._run_serial += 1
        self._active_run_id = None
        self._active_run_kind = None
        self._active_detector_frame = None
        self._active_detector_spec = None

    def _discard_draft(self) -> None:
        self._request_close_cancellation()
        self.reject()

    def _cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._request_close_cancellation()
        self._restore_channel_visibility()
        self._clear_detector_preview()
        self._clear_viewer_preview()
        if not self._accepted:
            self._restore_view_state()

    def done(self, result: int) -> None:
        if result == QDialog.Rejected and not self._discard_emitted:
            self._discard_emitted = True
            self.draftDiscarded.emit()
        self._cleanup()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._state == self.ACCEPTING:
            event.ignore()
            self._set_state(
                self.ACCEPTING,
                "Finishing the accepted edit; the window will close when it is safe.",
            )
            return
        event.ignore()
        self.reject()


def _warning_frames(warnings: tuple[str, ...]) -> set[int]:
    """Extract conventional ``t=N`` references for per-frame review flags."""

    import re

    frames: set[int] = set()
    for warning in warnings:
        frames.update(int(value) for value in re.findall(r"\bt=(\d+)\b", warning))
    return frames
