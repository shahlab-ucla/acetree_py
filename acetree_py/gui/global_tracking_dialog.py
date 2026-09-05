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
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QSettings, Qt, Signal, Slot
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
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

_STARRYNITE_DETECTOR_ID = "acetree.starrynite_detector"
_STARRYNITE_NATIVE_TRACKER_ID = "acetree.starrynite_division"
_STARRYNITE_EXACT_TRACKER_ID = "acetree.starrynite_legacy_exact"
_STARRYNITE_EXACT_BACKEND = "legacy_exact_refinement"
_STARRYNITE_TRACKER_IDS = frozenset(
    {_STARRYNITE_NATIVE_TRACKER_ID, _STARRYNITE_EXACT_TRACKER_ID}
)


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
    _RECENT_PARAMETERS_KEY = "tracking/starrynite/recent_parameter_file"
    _NEUTRAL_CLASSIFIER_KEY_PREFIX = (
        "tracking/starrynite/neutral_classifier_by_model"
    )

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
        exact_scope_error: str | None = None,
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
        self._exact_scope_error = (
            "" if exact_scope_error is None else str(exact_scope_error).strip()
        )

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
        self._starrynite_detector_settings: dict[str, Any] = {}
        self._starrynite_tracker_settings: dict[str, Any] = {}
        self._starrynite_parameter_path: Path | None = None
        self._starrynite_profile = None
        self._starrynite_neutral_classifier_path: Path | None = None
        self._starrynite_compatibility_report = None
        self._starrynite_session_note_html = ""
        self._starrynite_classifier_note_html = ""
        self._workflow_change_in_progress = False
        self._solo_channel_visibility: list[tuple[object, bool]] | None = None
        self._original_view = self._capture_view_state()

        self.setWindowTitle("Track Whole Movie — Review Draft")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setMinimumSize(760, 520)
        self.resize(1080, 720)
        # Worker callbacks may still be queued while a cooperative cancel is
        # completing. Parent ownership provides deterministic, safe cleanup.
        self.setAccessibleName("Global tracking draft workbench")

        self._build_ui()
        self._apply_initial_request(initial_request)
        if initial_request is None:
            self._apply_tracking_workflow("modern_starrynite")
        else:
            self._sync_tracking_workflow_from_components()
        self._refresh_recent_parameter_button()
        self._sync_division_capability(use_default=initial_request is None)
        self._update_starrynite_behavior_visibility()
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
            "<span style='font-size:16px'><b>Track the whole movie</b></span>"
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

        from ..tracking.workflows import GLOBAL_TRACKING_WORKFLOWS

        self._workflow_combo = QComboBox()
        for workflow in GLOBAL_TRACKING_WORKFLOWS:
            self._workflow_combo.addItem(workflow.display_name, workflow.workflow_id)
        self._workflow_combo.setToolTip(
            "Choose a complete detector and tracker workflow. Modern StarryNite "
            "is the recommended default."
        )
        self._workflow_combo.setAccessibleName("Tracking workflow")
        form.addRow("Tracking method:", self._workflow_combo)

        self._workflow_description = QLabel()
        self._workflow_description.setWordWrap(True)
        self._workflow_description.setAccessibleName("Tracking workflow explanation")
        form.addRow("", self._workflow_description)

        from ..tracking.starrynite import bundled_parameter_presets

        self._starrynite_preset_combo = QComboBox()
        for preset in bundled_parameter_presets():
            self._starrynite_preset_combo.addItem(preset.display_name, preset.preset_id)
            index = self._starrynite_preset_combo.count() - 1
            self._starrynite_preset_combo.setItemData(index, preset.description, Qt.ToolTipRole)
        self._starrynite_preset_combo.setToolTip(
            "Install-ready parameter files distributed with StarryNite. No model "
            "conversion or classifier-file selection is needed."
        )
        self._starrynite_preset_combo.setAccessibleName("Bundled StarryNite preset")
        form.addRow("Imaging preset:", self._starrynite_preset_combo)
        self._starrynite_preset_label = form.labelForField(
            self._starrynite_preset_combo
        )

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
        self._detector_label = form.labelForField(self._detector_combo)
        self._detector_combo.hide()
        self._detector_label.hide()

        self._tracker_combo = QComboBox()
        for descriptor in self._registry.tracker_descriptors():
            self._tracker_combo.addItem(descriptor.display_name, descriptor.plugin_id)
        self._tracker_combo.setToolTip("Method used to connect detections over time")
        self._tracker_combo.setAccessibleName("Detection linker")
        form.addRow("Tracker:", self._tracker_combo)
        self._tracker_label = form.labelForField(self._tracker_combo)
        self._tracker_combo.hide()
        self._tracker_label.hide()

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

        self._division_check = QCheckBox("Propose two-daughter divisions")
        self._division_check.setChecked(False)
        self._division_check.setToolTip(
            "Available for division-aware trackers. Every proposed split remains "
            "uncommitted until the full draft is accepted."
        )
        form.addRow("Division proposals:", self._division_check)
        configure_layout.addWidget(self._settings_widget)

        self._starrynite_file_button = QPushButton(
            "Start from StarryNite parameters…"
        )
        self._starrynite_file_button.setToolTip(
            "Load a legacy StarryNite parameter file as editable whole-movie "
            "tracking defaults"
        )
        self._starrynite_file_button.clicked.connect(
            self._choose_starrynite_parameter_file
        )
        configure_layout.addWidget(self._starrynite_file_button)

        self._starrynite_recent_button = QPushButton()
        self._starrynite_recent_button.setToolTip(
            "Reload the most recently used StarryNite parameter file"
        )
        self._starrynite_recent_button.clicked.connect(
            self._load_recent_starrynite_parameter_file
        )
        self._starrynite_recent_button.hide()
        configure_layout.addWidget(self._starrynite_recent_button)

        self._starrynite_save_button = QPushButton("Save tuned parameter copy…")
        self._starrynite_save_button.setToolTip(
            "Save compatible radius, intensity-threshold, and missing-frame edits "
            "without rewriting the legacy source"
        )
        self._starrynite_save_button.clicked.connect(
            self._choose_starrynite_parameter_destination
        )
        self._starrynite_save_button.setEnabled(False)
        configure_layout.addWidget(self._starrynite_save_button)

        self._starrynite_file_label = QLabel()
        self._starrynite_file_label.setWordWrap(True)
        self._starrynite_file_label.setAccessibleName(
            "Loaded StarryNite parameter file"
        )
        self._starrynite_file_label.hide()
        configure_layout.addWidget(self._starrynite_file_label)

        starrynite_compatibility_actions = QHBoxLayout()
        self._starrynite_neutral_button = QPushButton(
            "Use another legacy model..."
        )
        self._starrynite_neutral_button.setToolTip(
            "Advanced: select an AceTree-compatible model exported from another "
            "legacy StarryNite MAT file. Bundled presets already include their models."
        )
        self._starrynite_neutral_button.clicked.connect(
            self._choose_starrynite_neutral_classifier
        )
        self._starrynite_neutral_button.setEnabled(False)
        starrynite_compatibility_actions.addWidget(self._starrynite_neutral_button)
        self._starrynite_report_button = QPushButton("Compatibility details…")
        self._starrynite_report_button.setToolTip(
            "Show which StarryNite behavior is runnable and why"
        )
        self._starrynite_report_button.clicked.connect(
            self._show_starrynite_compatibility_report
        )
        self._starrynite_report_button.setEnabled(False)
        starrynite_compatibility_actions.addWidget(self._starrynite_report_button)
        configure_layout.addLayout(starrynite_compatibility_actions)

        self._starrynite_behavior_label = QLabel(
            "Compatibility note: legacy parameter values configure the native "
            "StarryNite detector and tracker. Referenced MATLAB classifier models "
            "are retained and hashed for provenance only; this whole-movie "
            "workbench currently uses the native geometry scorer. Saving writes "
            "only radius, intensity threshold, and missing frames to a copy."
        )
        self._starrynite_behavior_label.setWordWrap(True)
        self._starrynite_behavior_label.setAccessibleName(
            "StarryNite legacy model behavior"
        )
        self._starrynite_behavior_label.setStyleSheet(
            "QLabel { color: #d9a441; }"
        )
        self._starrynite_behavior_label.hide()
        configure_layout.addWidget(self._starrynite_behavior_label)

        self._advanced_toggle = QCheckBox("Show advanced and custom settings")
        self._advanced_toggle.toggled.connect(self._set_advanced_visible)
        configure_layout.addWidget(self._advanced_toggle)

        self._advanced_widget = QWidget()
        advanced_layout = QVBoxLayout(self._advanced_widget)
        advanced_layout.setContentsMargins(18, 0, 0, 0)
        self._subpixel_check = QCheckBox("Refine positions below one pixel")
        self._subpixel_check.setChecked(True)
        self._subpixel_check.setToolTip(
            "Optional native refinement. Leave off to preserve legacy StarryNite "
            "ray-recentered positions."
        )
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

        self._detector_status_label = QLabel(
            "Fast tuning: test the detector on the image currently shown before "
            "building the full draft. Detector tests cannot be accepted."
        )
        self._detector_status_label.setWordWrap(True)
        self._detector_status_label.setAccessibleName("Current-frame detector test status")
        self._detector_status_label.setStyleSheet("QLabel { color: #b8bec7; }")
        configure_layout.addWidget(self._detector_status_label)
        configure_scroll = QScrollArea()
        configure_scroll.setWidgetResizable(True)
        configure_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        configure_scroll.setMinimumWidth(350)
        configure_scroll.setAccessibleName("Whole-movie tracking settings")
        configure_scroll.setWidget(configure_group)
        self._configure_scroll = configure_scroll
        configure_column = QWidget()
        configure_column_layout = QVBoxLayout(configure_column)
        configure_column_layout.setContentsMargins(0, 0, 0, 0)
        configure_column_layout.setSpacing(6)
        configure_column_layout.addWidget(configure_scroll, stretch=1)
        configure_column_layout.addLayout(configure_actions)
        self._configure_column = configure_column
        columns.addWidget(configure_column, stretch=0)

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
            "candidate; ━ movement path; a forked path marks a proposed "
            "two-daughter division; ⊕ predicted search region. White marks the "
            "table selection; purple rings are a detector-only current-frame test; "
            "amber throughout means the draft must be rebuilt."
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
        self._workflow_combo.currentIndexChanged.connect(
            self._tracking_workflow_changed
        )
        self._starrynite_preset_combo.currentIndexChanged.connect(
            self._bundled_starrynite_preset_changed
        )
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
        self._detector_combo.currentIndexChanged.connect(self._detector_changed)
        self._tracker_combo.currentIndexChanged.connect(self._tracker_changed)
        self._subpixel_check.toggled.connect(self._detector_parameters_changed)
        self._median_check.toggled.connect(self._detector_parameters_changed)
        self._division_check.toggled.connect(self._tracking_parameters_changed)
        self._channel_spin.valueChanged.connect(self._refresh_solo_detection_channel)

    def _apply_accessibility_descriptions(self) -> None:
        """Expose explanatory tooltips through assistive-technology APIs."""

        for widget in (
            self._start_spin,
            self._end_spin,
            self._workflow_combo,
            self._starrynite_preset_combo,
            self._detector_combo,
            self._tracker_combo,
            self._channel_spin,
            self._radius_spin,
            self._threshold_spin,
            self._distance_spin,
            self._gap_spin,
            self._starrynite_file_button,
            self._starrynite_recent_button,
            self._starrynite_save_button,
            self._starrynite_neutral_button,
            self._starrynite_report_button,
            self._detector_preview_button,
            self._preview_button,
            self._overlay_check,
            self._solo_channel_check,
            self._discard_button,
            self._accept_button,
        ):
            if widget.toolTip():
                widget.setAccessibleDescription(widget.toolTip())

    def _tracking_workflow_changed(self, *_args) -> None:
        if self._workflow_change_in_progress:
            return
        workflow_id = self._workflow_combo.currentData()
        if workflow_id is not None:
            self._apply_tracking_workflow(str(workflow_id))

    def _apply_tracking_workflow(self, workflow_id: str) -> None:
        from ..tracking.workflows import CUSTOM_COMPONENTS, tracking_workflow

        workflow = tracking_workflow(workflow_id)
        self._workflow_change_in_progress = True
        try:
            if workflow is not CUSTOM_COMPONENTS:
                if workflow.uses_bundled_starrynite:
                    self._load_selected_bundled_starrynite_preset()
                self._select_combo_value(self._detector_combo, workflow.detector_id)
                self._select_combo_value(self._tracker_combo, workflow.tracker_id)
                if workflow.workflow_id == "legacy_starrynite_exact":
                    self._start_spin.setValue(self._range_start)
                    self._end_spin.setValue(self._range_end)
            self._workflow_description.setText(workflow.description)
        finally:
            self._workflow_change_in_progress = False
        self._update_workflow_visibility()
        self._sync_division_capability(use_default=True)
        self._update_starrynite_behavior_visibility()
        self._parameters_changed(detector_changed=True)

    def _load_selected_bundled_starrynite_preset(self) -> None:
        from ..tracking.starrynite import (
            DEFAULT_BUNDLED_PRESET_ID,
            bundled_parameter_preset,
        )

        preset_id = self._starrynite_preset_combo.currentData()
        if not preset_id or str(preset_id).startswith("__custom__"):
            preset_id = DEFAULT_BUNDLED_PRESET_ID
            index = self._starrynite_preset_combo.findData(preset_id)
            if index >= 0:
                self._starrynite_preset_combo.setCurrentIndex(index)
        preset = bundled_parameter_preset(str(preset_id))
        target = preset.parameter_file.resolve(strict=False)
        current = self._starrynite_parameter_path
        if current is None or current.resolve(strict=False) != target:
            self.load_starrynite_parameter_file(str(target))

    def _bundled_starrynite_preset_changed(self, *_args) -> None:
        if self._workflow_change_in_progress:
            return
        workflow_id = self._workflow_combo.currentData()
        if workflow_id not in {"modern_starrynite", "legacy_starrynite_exact"}:
            return
        self._workflow_change_in_progress = True
        try:
            self._load_selected_bundled_starrynite_preset()
            from ..tracking.workflows import tracking_workflow

            workflow = tracking_workflow(str(workflow_id))
            self._select_combo_value(self._detector_combo, workflow.detector_id)
            self._select_combo_value(self._tracker_combo, workflow.tracker_id)
        finally:
            self._workflow_change_in_progress = False
        self._update_workflow_visibility()
        self._parameters_changed(detector_changed=True)

    def _sync_tracking_workflow_from_components(self) -> None:
        from ..tracking.workflows import workflow_for_components

        workflow = workflow_for_components(
            self._detector_combo.currentData(),
            self._tracker_combo.currentData(),
        )
        self._workflow_change_in_progress = True
        try:
            index = self._workflow_combo.findData(workflow.workflow_id)
            if index >= 0:
                self._workflow_combo.setCurrentIndex(index)
            self._workflow_description.setText(workflow.description)
        finally:
            self._workflow_change_in_progress = False
        self._update_workflow_visibility()

    def _sync_bundled_preset_for_path(self, path: Path) -> None:
        from ..tracking.starrynite import bundled_parameter_presets

        resolved = path.resolve(strict=False)
        for preset in bundled_parameter_presets():
            if preset.parameter_file.resolve(strict=False) == resolved:
                index = self._starrynite_preset_combo.findData(preset.preset_id)
                if index >= 0:
                    self._workflow_change_in_progress = True
                    try:
                        self._starrynite_preset_combo.setCurrentIndex(index)
                    finally:
                        self._workflow_change_in_progress = False
                return
        custom_id = f"__custom__:{resolved}"
        index = self._starrynite_preset_combo.findData(custom_id)
        if index < 0:
            self._starrynite_preset_combo.addItem(f"Custom: {resolved.name}", custom_id)
            index = self._starrynite_preset_combo.count() - 1
        self._workflow_change_in_progress = True
        try:
            self._starrynite_preset_combo.setCurrentIndex(index)
        finally:
            self._workflow_change_in_progress = False

    def _update_workflow_visibility(self) -> None:
        workflow_id = self._workflow_combo.currentData()
        uses_starrynite = workflow_id in {
            "modern_starrynite",
            "legacy_starrynite_exact",
        }
        self._starrynite_preset_combo.setVisible(uses_starrynite)
        self._starrynite_preset_label.setVisible(uses_starrynite)
        self._starrynite_report_button.setVisible(uses_starrynite)
        self._starrynite_file_label.setVisible(
            uses_starrynite and self._starrynite_profile is not None
        )
        advanced = self._advanced_toggle.isChecked()
        for widget in (
            self._detector_combo,
            self._detector_label,
            self._tracker_combo,
            self._tracker_label,
        ):
            widget.setVisible(advanced)
        for widget in (
            self._starrynite_file_button,
            self._starrynite_save_button,
            self._starrynite_neutral_button,
        ):
            widget.setVisible(advanced and uses_starrynite)
        self._refresh_recent_parameter_button()

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
        requested_mode = tracker.get("STARRYNITE_COMPATIBILITY_MODE")
        if (
            request.tracker.plugin_id == _STARRYNITE_NATIVE_TRACKER_ID
            and requested_mode not in (None, "", "native_fast")
        ):
            raise ValueError(
                "Global tracking cannot restore StarryNite compatibility backend "
                f"{requested_mode!r}; this workbench supports only the explicit "
                "native_fast backend"
            )
        if (
            request.tracker.plugin_id == _STARRYNITE_EXACT_TRACKER_ID
            and requested_mode != _STARRYNITE_EXACT_BACKEND
        ):
            raise ValueError(
                "The exact StarryNite tracker request must explicitly select "
                f"{_STARRYNITE_EXACT_BACKEND!r}; no compatibility fallback is used"
            )
        parameter_path = detector.get("STARRYNITE_PARAMETER_FILE") or tracker.get(
            "STARRYNITE_PARAMETER_FILE"
        )
        uses_starrynite = (
            request.detector.plugin_id == _STARRYNITE_DETECTOR_ID
            or request.tracker.plugin_id in _STARRYNITE_TRACKER_IDS
        )
        loaded_fresh_profile = False
        if parameter_path and uses_starrynite:
            try:
                self.load_starrynite_parameter_file(
                    str(parameter_path),
                    neutral_classifier_path=(
                        tracker.get("STARRYNITE_NEUTRAL_CLASSIFIER_FILE") or None
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "Could not restore the StarryNite parameter source from the "
                    "initial request",
                    exc_info=True,
                )
                self._starrynite_file_label.setText(
                    "The previous StarryNite parameter source is unavailable or "
                    f"unreadable: {html.escape(str(parameter_path))}. The embedded "
                    "request values remain available, but save-copy is disabled."
                )
                self._starrynite_file_label.setToolTip(str(exc))
                self._starrynite_file_label.show()
            else:
                loaded_fresh_profile = True
                from ..tracking.starrynite.presets import tuning_identity_changed

                if tuning_identity_changed(detector, tracker, self._starrynite_profile):
                    self._starrynite_session_note_html = (
                        "<b>Source/model changed:</b> staged preset metadata was "
                        "refreshed; visible tuning edits were kept."
                    )
        if request.detector.plugin_id == _STARRYNITE_DETECTOR_ID:
            if loaded_fresh_profile:
                profile_controlled = {
                    "SIGMA",
                    "INTENSITY_THRESHOLD",
                    "BOUNDARY_PERCENT",
                    "LARGE_RAY_THRESHOLD",
                    "SMALL_RAY_THRESHOLD",
                    "NNDIST_MERGE",
                    "AR_MERGE",
                    "RADIUS",
                }
                for key, value in detector.items():
                    if key.startswith("STARRYNITE_") or key in profile_controlled:
                        continue
                    self._starrynite_detector_settings.setdefault(key, value)
            else:
                self._starrynite_detector_settings.update(detector)
        if request.tracker.plugin_id in _STARRYNITE_TRACKER_IDS:
            if loaded_fresh_profile:
                profile_controlled = {
                    "CANDIDATE_CUTOFF",
                    "SAFE_FACTOR",
                    "NN_NUMBER",
                    "FORWARD_NN_NUMBER",
                    "MAX_FRAME_GAP",
                    "ALLOW_GAP_CLOSING",
                }
                for key, value in tracker.items():
                    if key.startswith("STARRYNITE_") or key in profile_controlled:
                        continue
                    self._starrynite_tracker_settings.setdefault(key, value)
            else:
                self._starrynite_tracker_settings.update(tracker)
            if request.tracker.plugin_id == _STARRYNITE_EXACT_TRACKER_ID:
                for key in (
                    "STARRYNITE_FORCE_MODE",
                    "STARRYNITE_FORCE_END_FRAME",
                    "STARRYNITE_RECORD_ANSWERS",
                    "STARRYNITE_USE_STATIC_DIAMETER",
                ):
                    if key in tracker:
                        self._starrynite_tracker_settings[key] = tracker[key]
        # Loading a source intentionally selects the paired StarryNite
        # components for an interactive user.  An immutable initial request,
        # however, may deliberately combine a StarryNite detector with another
        # tracker, so restore its exact component choices after reading source
        # metadata.
        self._select_combo_value(self._detector_combo, request.detector.plugin_id)
        self._select_combo_value(self._tracker_combo, request.tracker.plugin_id)
        if "TARGET_CHANNEL" in detector:
            self._channel_spin.setValue(int(detector["TARGET_CHANNEL"]))
        if "RADIUS" in detector:
            self._radius_spin.setValue(float(detector["RADIUS"]))
        if (
            request.detector.plugin_id == _STARRYNITE_DETECTOR_ID
            and "INTENSITY_THRESHOLD" in detector
        ):
            self._threshold_spin.setValue(float(detector["INTENSITY_THRESHOLD"]))
        elif "THRESHOLD" in detector:
            self._threshold_spin.setValue(float(detector["THRESHOLD"]))
        if "DO_SUBPIXEL_LOCALIZATION" in detector:
            self._subpixel_check.setChecked(bool(detector["DO_SUBPIXEL_LOCALIZATION"]))
        if "DO_MEDIAN_FILTERING" in detector:
            self._median_check.setChecked(bool(detector["DO_MEDIAN_FILTERING"]))
        if "LINKING_MAX_DISTANCE" in tracker:
            self._distance_spin.setValue(float(tracker["LINKING_MAX_DISTANCE"]))
        if "MAX_FRAME_GAP" in tracker:
            self._gap_spin.setValue(max(0, int(tracker["MAX_FRAME_GAP"]) - 1))
        if "ALLOW_TRACK_SPLITTING" in tracker:
            self._division_check.setChecked(bool(tracker["ALLOW_TRACK_SPLITTING"]))
        self._render_starrynite_file_summary()

    @staticmethod
    def _select_combo_value(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _settings_store() -> QSettings:
        return QSettings("AceTree", "AceTreePy")

    def _starrynite_components_active(self) -> bool:
        return (
            self._detector_combo.currentData() == _STARRYNITE_DETECTOR_ID
            or self._tracker_combo.currentData() in _STARRYNITE_TRACKER_IDS
        )

    def _exact_starrynite_selected(self) -> bool:
        return self._tracker_combo.currentData() == _STARRYNITE_EXACT_TRACKER_ID

    def _starrynite_runtime_capabilities(self) -> tuple[str, ...]:
        try:
            descriptor = self._registry.get_descriptor(_STARRYNITE_EXACT_TRACKER_ID)
        except KeyError:
            return ()
        return (
            (_STARRYNITE_EXACT_BACKEND,)
            if _STARRYNITE_EXACT_BACKEND in descriptor.capabilities
            else ()
        )

    def _build_starrynite_report(self, profile, neutral_classifier_path=None):
        from ..tracking.starrynite import build_compatibility_report

        return build_compatibility_report(
            profile,
            neutral_classifier_path=neutral_classifier_path,
            runtime_capabilities=self._starrynite_runtime_capabilities(),
        )

    def _render_starrynite_file_summary(self) -> None:
        profile = self._starrynite_profile
        if profile is None:
            return
        source = profile.parameters.source_path or self._starrynite_parameter_path
        source_name = "in-memory parameters" if source is None else source.name
        tracker_id = self._tracker_combo.currentData()
        tracker_is_native = tracker_id == _STARRYNITE_NATIVE_TRACKER_ID
        tracker_is_exact = tracker_id == _STARRYNITE_EXACT_TRACKER_ID
        detector_is_starrynite = (
            self._detector_combo.currentData() == _STARRYNITE_DETECTOR_ID
        )
        if tracker_is_exact:
            workflow = (
                "Tune basic values only through a saved parameter copy, then build "
                "the full draft. A current-frame exact detector test is valid only "
                "at t=1 because later frames depend on earlier detector state."
            )
        else:
            workflow = (
                "Tune the basic values, test a frame, then build the full draft."
            )
        parts = [
            f"Loaded {html.escape(source_name)} (stage {profile.stage_index + 1}, "
            f"{profile.cell_count} starting cells). {workflow}"
        ]
        if profile.model_path is not None:
            if tracker_is_exact:
                parts.append(
                    f"Model {html.escape(profile.model_path.name)} supplies the "
                    "numeric legacy geometry state and is used by this exact draft."
                )
            elif tracker_is_native:
                parts.append(
                    f"Model {html.escape(profile.model_path.name)} is provenance-only; "
                    "native geometry scoring remains active."
                )
            elif detector_is_starrynite:
                parts.append(
                    f"Model {html.escape(profile.model_path.name)} is reporting-only; "
                    "the selected non-StarryNite tracker does not consume it."
                )
            else:
                parts.append(
                    "<b>Inactive preset:</b> The selected detector and tracker are not "
                    "StarryNite, so this loaded model and its parameter overrides are "
                    "not used by the draft."
                )
        report = self._starrynite_compatibility_report
        neutral_path = self._starrynite_neutral_classifier_path
        if (
            neutral_path is not None
            and report is not None
            and report.neutral_classifier_source_bound
        ):
            if tracker_is_exact:
                parts.append(
                    f"Tracking model {html.escape(neutral_path.name)} is "
                    "source-bound and will classify tentative divisions in this "
                    "exact whole-movie draft."
                )
            elif tracker_is_native:
                parts.append(
                    f"Tracking model {html.escape(neutral_path.name)} was validated "
                    "as source-bound for reporting only; native geometry scoring "
                    "remains active."
                )
            else:
                parts.append(
                    f"Tracking model {html.escape(neutral_path.name)} is "
                    "source-bound to the loaded preset for reporting only; the "
                    "selected tracker does not use it."
                )
        if tracker_is_exact and report is not None:
            readiness = report.backend(_STARRYNITE_EXACT_BACKEND)
            if not readiness.runnable:
                blocker_codes = {issue.code for issue in readiness.blockers}
                if "neutral_classifier_not_selected" in blocker_codes:
                    next_step = (
                        "Choose a bundled preset with a ready tracking model, or "
                        "use the advanced legacy-model exporter."
                    )
                elif blocker_codes & {
                    "parameter_source_changed",
                    "tracking_model_changed",
                    "neutral_classifier_unbound",
                }:
                    next_step = (
                        "Reload the parameter file, then attach a freshly "
                        "source-bound tracking model."
                    )
                elif blocker_codes & {
                    "legacy_exact_detector_parameter_missing",
                    "legacy_exact_parameter_missing",
                    "legacy_downsampling_missing",
                }:
                    next_step = (
                        "Add the required value to a parameter-file copy and reload "
                        "that copy."
                    )
                else:
                    next_step = (
                        "Open Compatibility details, resolve the first blocker, "
                        "then reload the source."
                    )
                first_blocker = html.escape(readiness.blockers[0].message)
                parts.append(
                    f"<b>Exact mode is blocked.</b> Next step: {next_step} "
                    f"First blocker: {first_blocker}"
                )
        if self._starrynite_classifier_note_html:
            parts.append(self._starrynite_classifier_note_html)
        if self._starrynite_session_note_html:
            parts.append(self._starrynite_session_note_html)
        if not self._starrynite_components_active() and profile.model_path is None:
            parts.append(
                "<b>Inactive preset:</b> The selected detector and tracker are not "
                "StarryNite, so the loaded parameter overrides are not used by this "
                "draft."
            )
        presentation_warnings = [
            *profile.warnings,
            *self._starrynite_calibration_warnings(profile),
        ]
        if presentation_warnings:
            parts.append(
                f"{len(presentation_warnings)} parameter/calibration note(s); hover "
                "for details."
            )
        details = [
            *(
                ()
                if self._starrynite_compatibility_report is None
                else (self._starrynite_compatibility_report.format_text(),)
            ),
            *presentation_warnings,
        ]
        self._starrynite_file_label.setText(" ".join(parts))
        self._starrynite_file_label.setToolTip("\n".join(details))
        self._starrynite_file_label.setAccessibleDescription("\n".join(details))
        self._starrynite_file_label.show()

    def _refresh_starrynite_compatibility(self) -> None:
        profile = self._starrynite_profile
        if profile is None:
            return
        neutral_path = self._starrynite_neutral_classifier_path
        report = self._build_starrynite_report(
            profile,
            neutral_path,
        )
        if neutral_path is not None and not report.neutral_classifier_source_bound:
            self._forget_neutral_classifier(profile)
            self._starrynite_neutral_classifier_path = None
            self._starrynite_classifier_note_html = (
                "<b>Classifier binding changed:</b> "
                f"{html.escape(neutral_path.name)} is no longer attached. Reload the "
                "parameter file and validate a fresh export."
            )
            report = self._build_starrynite_report(profile)
        elif neutral_path is not None:
            self._starrynite_classifier_note_html = ""
        self._starrynite_compatibility_report = report
        self._render_starrynite_file_summary()

    def recent_starrynite_parameter_file(self) -> Path | None:
        """Return the shared most-recent usable StarryNite parameter file."""

        try:
            value = self._settings_store().value(self._RECENT_PARAMETERS_KEY, "")
        except (OSError, RuntimeError):
            return None
        if value is None or not str(value).strip():
            return None
        candidate = Path(str(value)).expanduser()
        return candidate.resolve(strict=False) if candidate.is_file() else None

    def _remember_starrynite_parameter_file(self, path: Path) -> None:
        from ..tracking.starrynite import is_bundled_parameter_file

        if is_bundled_parameter_file(path):
            # Loading the recommended default must not erase the user's most
            # recently selected or tuned custom parameter file.
            return
        try:
            settings = self._settings_store()
            settings.setValue(
                self._RECENT_PARAMETERS_KEY,
                str(path.resolve(strict=False)),
            )
            settings.sync()
        except (OSError, RuntimeError):
            logger.debug("Could not persist recent StarryNite parameters", exc_info=True)

    def _neutral_classifier_settings_key(self, profile=None) -> str | None:
        active_profile = self._starrynite_profile if profile is None else profile
        parameters = getattr(active_profile, "parameters", None)
        references = tuple(getattr(parameters, "model_references", ()))
        load_references = tuple(
            reference
            for reference in references
            if getattr(reference, "source_kind", None) == "load"
        )
        active_references = load_references or references
        if len(active_references) != 1:
            return None
        model_hash = getattr(active_profile, "model_sha256", None)
        model_path = getattr(active_profile, "model_path", None)
        if not model_hash or model_path is None or not Path(model_path).is_file():
            return None
        return f"{self._NEUTRAL_CLASSIFIER_KEY_PREFIX}/{model_hash}"

    def _remembered_neutral_classifier(self, profile) -> Path | None:
        key = self._neutral_classifier_settings_key(profile)
        if key is None:
            return None
        try:
            settings = self._settings_store()
            value = settings.value(key, "")
            if value is None or not str(value).strip():
                return None
            candidate = Path(str(value)).expanduser().resolve(strict=False)
            return candidate
        except (OSError, RuntimeError):
            logger.debug("Could not restore classifier association", exc_info=True)
        return None

    def _remember_neutral_classifier(self, profile, path: Path) -> None:
        key = self._neutral_classifier_settings_key(profile)
        if key is None:
            return
        try:
            settings = self._settings_store()
            settings.setValue(key, str(path.resolve(strict=False)))
            settings.sync()
        except (OSError, RuntimeError):
            logger.debug("Could not persist classifier association", exc_info=True)

    def _forget_neutral_classifier(self, profile) -> None:
        key = self._neutral_classifier_settings_key(profile)
        if key is None:
            return
        try:
            settings = self._settings_store()
            settings.remove(key)
            settings.sync()
        except (OSError, RuntimeError):
            logger.debug("Could not clear classifier association", exc_info=True)

    def _refresh_recent_parameter_button(self) -> None:
        recent = self.recent_starrynite_parameter_file()
        current = (
            None
            if self._starrynite_parameter_path is None
            else self._starrynite_parameter_path.resolve(strict=False)
        )
        visible = recent is not None and recent != current
        if recent is not None:
            self._starrynite_recent_button.setText(f"Use recent: {recent.name}")
            self._starrynite_recent_button.setToolTip(
                f"Reload the most recently used parameter file:\n{recent}"
            )
        self._starrynite_recent_button.setVisible(visible)

    def _load_recent_starrynite_parameter_file(self) -> None:
        recent = self.recent_starrynite_parameter_file()
        if recent is None:
            self._refresh_recent_parameter_button()
            return
        try:
            self.load_starrynite_parameter_file(str(recent))
        except Exception as exc:
            logger.exception("Could not reload recent StarryNite parameters")
            QMessageBox.warning(
                self,
                "Could Not Read Parameters",
                f"The recent StarryNite parameter file could not be read.\n\n{exc}",
            )

    def _choose_starrynite_parameter_file(self) -> None:
        initial = self._starrynite_parameter_path or (
            self.recent_starrynite_parameter_file()
        )
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose a StarryNite parameter file",
            "" if initial is None else str(initial),
            "StarryNite parameters (*.txt *.m);;All files (*)",
        )
        if not path:
            return
        try:
            self.load_starrynite_parameter_file(path)
        except Exception as exc:
            logger.exception("Could not load StarryNite parameter preset")
            QMessageBox.warning(
                self,
                "Could Not Read Parameters",
                f"That StarryNite parameter file could not be read.\n\n{exc}",
            )

    def load_starrynite_parameter_file(
        self,
        path: str,
        *,
        neutral_classifier_path: str | Path | None = None,
    ) -> None:
        """Apply a legacy parameter file as editable whole-movie defaults."""

        from ..tracking.starrynite import (
            bundled_classifier_for_profile,
            load_tuning_profile,
        )

        profile = load_tuning_profile(
            path,
            fallback_radius_um=self._radius_spin.value(),
        )
        self._starrynite_session_note_html = ""
        self._starrynite_classifier_note_html = ""
        self._starrynite_detector_settings = dict(profile.detector_settings)
        self._starrynite_tracker_settings = dict(profile.tracker_settings)
        source = profile.parameters.source_path or Path(path)
        self._starrynite_parameter_path = source.resolve(strict=False)
        self._starrynite_profile = profile
        self._sync_bundled_preset_for_path(self._starrynite_parameter_path)
        restored_from_settings = neutral_classifier_path is None
        requested_neutral = (
            bundled_classifier_for_profile(profile)
            or self._remembered_neutral_classifier(profile)
            if restored_from_settings
            else Path(neutral_classifier_path).expanduser().resolve(strict=False)
        )
        self._starrynite_compatibility_report = self._build_starrynite_report(
            profile,
            requested_neutral,
        )
        if (
            requested_neutral is not None
            and self._starrynite_compatibility_report.neutral_classifier_source_bound
        ):
            self._starrynite_neutral_classifier_path = requested_neutral
            self._remember_neutral_classifier(profile, requested_neutral)
        else:
            self._starrynite_neutral_classifier_path = None
            if requested_neutral is not None:
                if restored_from_settings:
                    self._forget_neutral_classifier(profile)
                self._starrynite_classifier_note_html = (
                    f"<b>Classifier not restored:</b> "
                    f"{html.escape(requested_neutral.name)} could not be proven "
                    "source-bound. See Compatibility details."
                )
        self._starrynite_save_button.setEnabled(True)
        self._starrynite_neutral_button.setEnabled(True)
        self._starrynite_report_button.setEnabled(True)
        self._remember_starrynite_parameter_file(self._starrynite_parameter_path)
        self._refresh_recent_parameter_button()

        self._select_combo_value(
            self._detector_combo,
            "acetree.starrynite_detector",
        )
        self._select_combo_value(
            self._tracker_combo,
            "acetree.starrynite_division",
        )
        self._sync_division_capability(use_default=True)
        self._subpixel_check.setChecked(
            bool(profile.detector_settings.get("DO_SUBPIXEL_LOCALIZATION", False))
        )
        self._radius_spin.setValue(
            float(profile.detector_settings.get("RADIUS", self._radius_spin.value()))
        )
        self._threshold_spin.setValue(
            float(
                profile.detector_settings.get(
                    "INTENSITY_THRESHOLD",
                    self._threshold_spin.value(),
                )
            )
        )
        max_frame_gap = int(profile.tracker_settings.get("MAX_FRAME_GAP", 2))
        self._gap_spin.setValue(max(0, max_frame_gap - 1))
        self._render_starrynite_file_summary()
        self._update_starrynite_behavior_visibility()
        if not self._workflow_change_in_progress:
            self._sync_tracking_workflow_from_components()
        self._parameters_changed(detector_changed=True)

    def _starrynite_calibration_warnings(self, profile) -> tuple[str, ...]:
        calibration = self._calibration
        if calibration is None:
            return ()
        from ..tracking.starrynite.presets import tuning_calibration_warnings

        return tuning_calibration_warnings(
            profile, xy_um=calibration.xy_um, z_um=calibration.z_um
        )

    def _choose_starrynite_neutral_classifier(self) -> None:
        profile = self._starrynite_profile
        if profile is None:
            return
        initial = self._starrynite_neutral_classifier_path
        if initial is None:
            initial = profile.model_path or self._starrynite_parameter_path
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose an AceTree-compatible StarryNite model",
            "" if initial is None else str(initial),
            "AceTree tracking models (*.atpy-model *.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.attach_starrynite_neutral_classifier(path)
        except Exception as exc:
            logger.exception("Could not validate neutral StarryNite classifier")
            QMessageBox.warning(
                self,
                "Tracking Model Not Usable",
                f"That tracking model could not be used.\n\n{exc}",
            )

    def attach_starrynite_neutral_classifier(self, path: str | Path) -> None:
        """Validate and remember a source-bound neutral classifier export."""

        profile = self._starrynite_profile
        if profile is None:
            raise ValueError("Load a StarryNite parameter file first")
        candidate = Path(path).expanduser().resolve(strict=False)
        report = self._build_starrynite_report(
            profile,
            candidate,
        )
        if not report.neutral_classifier_source_bound:
            relevant_codes = {
                "neutral_classifier_invalid",
                "neutral_classifier_unbound",
                "tracking_model_ambiguous",
                "tracking_model_changed",
                "tracking_model_identity_unavailable",
                "tracking_model_missing",
                "tracking_model_not_referenced",
            }
            issue = next(
                (
                    blocker
                    for backend in report.backends.values()
                    for blocker in backend.blockers
                    if blocker.code in relevant_codes
                ),
                None,
            )
            error_message = (
                issue.message
                if issue is not None
                else "The tracking model could not be proven source-bound."
            )
            previous_path = self._starrynite_neutral_classifier_path
            if previous_path is not None:
                previous_report = self._build_starrynite_report(
                    profile,
                    previous_path,
                )
                if previous_report.neutral_classifier_source_bound:
                    self._starrynite_compatibility_report = previous_report
                    self._starrynite_classifier_note_html = ""
                    self._render_starrynite_file_summary()
                    raise ValueError(error_message)
                self._forget_neutral_classifier(profile)
            self._starrynite_neutral_classifier_path = None
            self._starrynite_compatibility_report = self._build_starrynite_report(
                profile
            )
            if previous_path is None:
                self._starrynite_classifier_note_html = (
                    "<b>Classifier not attached:</b> "
                    f"{html.escape(candidate.name)} was not usable. See Compatibility "
                    "details."
                )
            else:
                self._starrynite_classifier_note_html = (
                    "<b>Classifier binding changed:</b> the previous export is no "
                    "longer attached, and the selected replacement was not usable. "
                    "Reload the parameter file and validate a fresh export."
                )
            self._starrynite_report_button.setEnabled(True)
            self._render_starrynite_file_summary()
            raise ValueError(error_message)
        self._starrynite_neutral_classifier_path = candidate
        self._starrynite_compatibility_report = report
        self._starrynite_classifier_note_html = ""
        self._remember_neutral_classifier(profile, candidate)
        self._starrynite_report_button.setEnabled(True)
        self._render_starrynite_file_summary()
        self._validate_settings()

    def _show_starrynite_compatibility_report(self) -> None:
        self._refresh_starrynite_compatibility()
        report = self._starrynite_compatibility_report
        if report is None:
            return
        message = QMessageBox(self)
        message.setWindowTitle("StarryNite Compatibility")
        message.setIcon(QMessageBox.Information)
        if self._tracker_combo.currentData() == _STARRYNITE_EXACT_TRACKER_ID:
            summary = (
                "The selected tracker executes the source-bound legacy geometry, "
                "classifier, repair, and deletion stages. Any listed blocker "
                "prevents the draft; no native fallback is used."
            )
        elif self._tracker_combo.currentData() == _STARRYNITE_NATIVE_TRACKER_ID:
            summary = (
                "The selected StarryNite tracker runs native geometry tracking. "
                "Details also report whether the inputs are valid for a separate "
                "exact-refinement runtime; validating an export does not change this "
                "draft."
            )
        elif self._detector_combo.currentData() == _STARRYNITE_DETECTOR_ID:
            summary = (
                "The selected StarryNite detector is paired with a different tracker. "
                "Classifier compatibility is reporting-only and does not change this "
                "draft."
            )
        else:
            summary = (
                "The loaded StarryNite preset is inactive because the selected "
                "detector and tracker are not StarryNite. These details describe the "
                "loaded preset only."
            )
        message.setText(summary)
        message.setDetailedText(report.format_text())
        message.exec()

    def _choose_starrynite_parameter_destination(self) -> None:
        source = self._starrynite_parameter_path
        if source is None:
            return
        suffix = source.suffix if source.suffix else ".txt"
        default = source.with_name(f"{source.stem}-tuned{suffix}")
        destination, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save tuned StarryNite parameters",
            str(default),
            "StarryNite parameters (*.txt *.m);;All files (*)",
        )
        if not destination:
            return
        try:
            warnings = self.save_starrynite_parameter_file(destination)
        except Exception as exc:
            logger.exception("Could not save tuned StarryNite parameters")
            QMessageBox.warning(
                self,
                "Could Not Save Parameters",
                f"The tuned parameter copy could not be saved.\n\n{exc}",
            )
            return
        message = "The tuned copy was saved and is now the active parameter file."
        if warnings:
            message += "\n\n" + "\n".join(warnings)
        QMessageBox.information(self, "Parameter Copy Saved", message)

    def save_starrynite_parameter_file(self, path: str) -> tuple[str, ...]:
        """Save compatible edits and make the new copy the active source."""

        from ..tracking.starrynite import (
            build_tuning_save_plan,
            load_tuning_profile,
            write_parameter_file,
        )

        source = self._starrynite_parameter_path
        if source is None:
            raise ValueError("Load a StarryNite parameter file before saving a copy")
        profile = self._starrynite_profile
        if profile is None:
            profile = load_tuning_profile(
                source,
                fallback_radius_um=self._radius_spin.value(),
            )
        plan = build_tuning_save_plan(
            profile,
            radius_um=self._radius_spin.value(),
            intensity_threshold=self._threshold_spin.value(),
            max_frame_gap=(
                self._gap_spin.value() + 1
                if self._gap_spin.value() > 0
                else 1
            ),
        )
        destination = Path(path).expanduser()
        write_parameter_file(
            profile.parameters,
            destination,
            overrides=plan.overrides,
        )
        warnings = list(plan.warnings)
        if source.parent.resolve(strict=False) != destination.parent.resolve(
            strict=False
        ):
            warnings.append(
                "This copy is in a different folder. Check any relative model paths "
                "before using it in MATLAB."
            )
        # Reloading the saved copy refreshes all source-controlled hashes and
        # staged values.  Preserve the rest of the user's workbench choices;
        # saving a parameter copy must not silently switch components or
        # re-enable native-only refinements.
        session_choices = {
            "detector": self._detector_combo.currentData(),
            "tracker": self._tracker_combo.currentData(),
            "start": self._start_spin.value(),
            "end": self._end_spin.value(),
            "channel": self._channel_spin.value(),
            "distance": self._distance_spin.value(),
            "divisions": self._division_check.isChecked(),
            "subpixel": self._subpixel_check.isChecked(),
            "median": self._median_check.isChecked(),
        }
        self.load_starrynite_parameter_file(
            str(destination),
            neutral_classifier_path=self._starrynite_neutral_classifier_path,
        )
        self._select_combo_value(self._detector_combo, session_choices["detector"])
        self._select_combo_value(self._tracker_combo, session_choices["tracker"])
        self._start_spin.setValue(int(session_choices["start"]))
        self._end_spin.setValue(int(session_choices["end"]))
        self._channel_spin.setValue(int(session_choices["channel"]))
        self._distance_spin.setValue(float(session_choices["distance"]))
        if self._tracker_combo.currentData() != _STARRYNITE_EXACT_TRACKER_ID:
            self._division_check.setChecked(bool(session_choices["divisions"]))
        self._subpixel_check.setChecked(bool(session_choices["subpixel"]))
        self._median_check.setChecked(bool(session_choices["median"]))
        self._starrynite_session_note_html = (
            "Compatible edits were saved to this copy."
        )
        self._render_starrynite_file_summary()
        return tuple(warnings)

    def _update_starrynite_behavior_visibility(self) -> None:
        detector_is_starrynite = (
            self._detector_combo.currentData() == _STARRYNITE_DETECTOR_ID
        )
        tracker_id = self._tracker_combo.currentData()
        tracker_is_native = tracker_id == _STARRYNITE_NATIVE_TRACKER_ID
        tracker_is_exact = tracker_id == _STARRYNITE_EXACT_TRACKER_ID
        using_starrynite = detector_is_starrynite or tracker_id in _STARRYNITE_TRACKER_IDS
        if detector_is_starrynite and tracker_is_exact:
            text = (
                "Exact mode uses the source-bound detector distribution, staged "
                "legacy geometry, and source-bound tracking model. Save any tuning "
                "edits to a parameter copy before building; unsupported options "
                "stop with an actionable compatibility message."
            )
        elif detector_is_starrynite and tracker_is_native:
            text = (
                "Compatibility note: legacy parameter values configure the native "
                "StarryNite detector and tracker. Referenced MATLAB classifier models "
                "are retained and hashed for provenance only; this whole-movie "
                "workbench currently uses the native geometry scorer. Saving writes "
                "only radius, intensity threshold, and missing frames to a copy."
            )
        elif tracker_is_exact:
            text = (
                "Exact StarryNite tracking requires the StarryNite detector. Select "
                "it and load a source-bound parameter file before building."
            )
        elif tracker_is_native:
            text = (
                "Compatibility note: the selected StarryNite tracker uses native "
                "geometry scoring with the separately selected detector. Referenced "
                "MATLAB classifier models remain provenance-only."
            )
        else:
            text = (
                "Compatibility note: the selected StarryNite detector uses compatible "
                "parameter values, while the separately selected tracker does not use "
                "the referenced MATLAB classifier model."
            )
        self._starrynite_behavior_label.setText(text)
        self._starrynite_behavior_label.setVisible(using_starrynite)
        if tracker_is_exact:
            self._starrynite_neutral_button.setText("Use another legacy model...")
            self._starrynite_neutral_button.setToolTip(
                "Advanced: replace the bundled source-bound model used by exact "
                "whole-movie tracking"
            )
        else:
            self._starrynite_neutral_button.setText("Use another legacy model...")
            self._starrynite_neutral_button.setToolTip(
                "Advanced: validate another source-bound legacy model. The native "
                "tracker remains unchanged; exact whole-movie tracking executes it."
            )

    def export_settings(self) -> dict[str, Any]:
        """Return the current common settings for reopening or rerunning."""

        self._refresh_starrynite_compatibility()
        return {
            "workflow_id": self._workflow_combo.currentData(),
            "bundled_starrynite_preset_id": self._starrynite_preset_combo.currentData(),
            "start_time": self._start_spin.value(),
            "end_time": self._end_spin.value(),
            "detector_id": self._detector_combo.currentData(),
            "tracker_id": self._tracker_combo.currentData(),
            "channel": self._channel_spin.value(),
            "radius_um": self._radius_spin.value(),
            "threshold": self._threshold_spin.value(),
            "max_distance_um": self._distance_spin.value(),
            "missing_frames": self._gap_spin.value(),
            "allow_divisions": self._division_check.isChecked(),
            "subpixel": self._subpixel_check.isChecked(),
            "median_filter": self._median_check.isChecked(),
            "show_overlay": self._overlay_check.isChecked(),
            "starrynite_parameter_file": (
                ""
                if self._starrynite_parameter_path is None
                else str(self._starrynite_parameter_path)
            ),
            "starrynite_neutral_classifier_file": (
                ""
                if self._starrynite_neutral_classifier_path is None
                else str(self._starrynite_neutral_classifier_path)
            ),
            "starrynite_compatibility_backend": (
                _STARRYNITE_EXACT_BACKEND
                if self._tracker_combo.currentData() == _STARRYNITE_EXACT_TRACKER_ID
                else (
                    "native_fast"
                    if self._tracker_combo.currentData()
                    == _STARRYNITE_NATIVE_TRACKER_ID
                    else None
                )
            ),
        }

    def get_detector_spec(self) -> ComponentSpec:
        """Build detector settings without requiring any tracker configuration."""

        self._refresh_starrynite_compatibility()
        error = self._detector_validation_error()
        if error:
            raise ValueError(error)

        from ..tracking.settings import build_detector_spec

        detector_id = str(self._detector_combo.currentData())
        return build_detector_spec(
            self._registry,
            detector_id,
            channel=self._channel_spin.value(),
            radius_um=self._radius_spin.value(),
            threshold=self._threshold_spin.value(),
            subpixel=self._subpixel_check.isChecked(),
            median_filter=self._median_check.isChecked(),
            source_settings=(
                self._starrynite_detector_settings
                if detector_id == _STARRYNITE_DETECTOR_ID else None
            ),
            exact_starrynite=self._exact_starrynite_selected(),
        )

    def get_request(self) -> TrackingRequest:
        """Build the immutable global request represented by the form."""

        self._refresh_starrynite_compatibility()
        error = self._settings_validation_error()
        if error:
            raise ValueError(error)

        from ..tracking.api import TrackingRequest, TrackingScope
        from ..tracking.settings import build_tracker_spec

        detector_spec = self.get_detector_spec()
        tracker_id = str(self._tracker_combo.currentData())
        tracker_settings: dict[str, Any] = {}
        if tracker_id == _STARRYNITE_NATIVE_TRACKER_ID:
            tracker_settings.update(self._starrynite_tracker_settings)
        elif tracker_id == _STARRYNITE_EXACT_TRACKER_ID:
            from ..tracking.starrynite import sha256_file

            profile = self._starrynite_profile
            parameter_path = self._starrynite_parameter_path
            classifier_path = self._starrynite_neutral_classifier_path
            assert profile is not None
            assert parameter_path is not None
            assert profile.model_path is not None
            assert classifier_path is not None
            tracker_settings.update(
                {
                    "STARRYNITE_COMPATIBILITY_MODE": _STARRYNITE_EXACT_BACKEND,
                    "STARRYNITE_PARAMETER_FILE": str(parameter_path),
                    "STARRYNITE_PARAMETER_SHA256": profile.parameter_sha256,
                    "STARRYNITE_MODEL_FILE": str(profile.model_path),
                    "STARRYNITE_MODEL_SHA256": profile.model_sha256,
                    "STARRYNITE_NEUTRAL_CLASSIFIER_FILE": str(classifier_path),
                    "STARRYNITE_NEUTRAL_CLASSIFIER_SHA256": sha256_file(
                        classifier_path
                    ),
                }
            )
            for key in (
                "STARRYNITE_FORCE_MODE",
                "STARRYNITE_FORCE_END_FRAME",
                "STARRYNITE_RECORD_ANSWERS",
                "STARRYNITE_USE_STATIC_DIAMETER",
            ):
                if key in self._starrynite_tracker_settings:
                    tracker_settings[key] = self._starrynite_tracker_settings[key]
        tracker = build_tracker_spec(
            self._registry,
            tracker_id,
            max_distance_um=self._distance_spin.value(),
            missing_frames=self._gap_spin.value(),
            allow_splitting=self._division_check.isChecked(),
            source_settings=tracker_settings,
        )
        return TrackingRequest(
            detector=detector_spec,
            tracker=tracker,
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
        preview_error = self._detector_preview_validation_error(frame)
        if preview_error:
            self._set_detector_status(preview_error, color="#e06c75")
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
                "cannot be accepted; Undo those edits or use Track Selected Cell.",
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
                    "present. Whole-movie tracking is disabled; use Track Selected Cell or "
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
                    "tracking cannot be accepted; Undo those edits or use Track Selected Cell."
                )
            elif self._state not in {self.RUNNING, self.CANCELING, self.ACCEPTING}:
                self._set_state(
                    self.STALE,
                    "The dataset now contains curated positions. Whole-dataset "
                    "tracking is disabled; Undo those edits or use Track Selected Cell.",
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
        division_source_ids = {
            edge.source_id for edge in proposal.edges if edge.kind == "split"
        }
        self._summary_label.setText(
            f"<b>{len(detections)} detected spots</b> · "
            f"{preview.proposed_count} positions to add after interpolation · "
            f"{len(roots)} tracks · {len(preview.links)} adjacent link segments · "
            f"{original_gaps} bridged gaps ({preview.interpolated_count} interpolated positions) · "
            f"{len(division_source_ids)} proposed division"
            f"{'s' if len(division_source_ids) != 1 else ''}"
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
        detections_by_id = {
            detection.detection_id: detection for detection in proposal.detections
        }
        divisions_by_frame: dict[int, int] = defaultdict(int)
        for source_id in division_source_ids:
            source = detections_by_id.get(source_id)
            if source is not None:
                divisions_by_frame[source.frame] += 1

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
            if divisions_by_frame.get(frame, 0):
                count = divisions_by_frame[frame]
                notes.append(
                    f"{count} proposed division{'s' if count != 1 else ''}"
                )
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

    def _detector_changed(self, *_args) -> None:
        self._update_starrynite_behavior_visibility()
        detector_id = self._detector_combo.currentData()
        if detector_id is not None:
            try:
                default = self._registry.default_settings(str(detector_id)).get(
                    "DO_SUBPIXEL_LOCALIZATION",
                    True,
                )
                self._subpixel_check.setChecked(bool(default))
            except (KeyError, ValueError):
                pass
        self._render_starrynite_file_summary()
        if not self._workflow_change_in_progress:
            self._sync_tracking_workflow_from_components()
        self._detector_parameters_changed()

    def _tracking_parameters_changed(self, *_args) -> None:
        self._parameters_changed(detector_changed=False)

    def _tracker_changed(self, *_args) -> None:
        if self._exact_starrynite_selected():
            # The exact runtime consumes detector history and geometry state
            # from the complete source movie.  Selecting it interactively
            # should therefore select its required detector and a runnable
            # scope immediately instead of leaving native-tracker choices in
            # place.
            self._select_combo_value(
                self._detector_combo,
                _STARRYNITE_DETECTOR_ID,
            )
            self._start_spin.setValue(self._range_start)
            self._end_spin.setValue(self._range_end)
        self._update_starrynite_behavior_visibility()
        self._sync_division_capability(use_default=True)
        self._render_starrynite_file_summary()
        if not self._workflow_change_in_progress:
            self._sync_tracking_workflow_from_components()
        self._tracking_parameters_changed()

    def _sync_division_capability(self, *, use_default: bool) -> None:
        tracker_id = self._tracker_combo.currentData()
        if tracker_id == _STARRYNITE_EXACT_TRACKER_ID:
            self._division_check.setChecked(True)
            self._division_check.setEnabled(False)
            self._division_check.setToolTip(
                "Exact StarryNite replay intrinsically evaluates two-daughter "
                "division hypotheses."
            )
            return
        capable = False
        default = False
        if tracker_id is not None:
            try:
                descriptor = self._registry.get_descriptor(str(tracker_id))
                capable = "splitting" in descriptor.capabilities
                default = bool(
                    self._registry.default_settings(str(tracker_id)).get(
                        "ALLOW_TRACK_SPLITTING", False
                    )
                )
            except (KeyError, ValueError):
                capable = False
        self._division_check.setEnabled(capable)
        if not capable:
            self._division_check.setChecked(False)
        elif use_default:
            self._division_check.setChecked(default)
        self._division_check.setToolTip(
            "Propose two-daughter lineage branches for review."
            if capable
            else "The selected tracker does not support divisions."
        )

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

    def _detector_preview_validation_error(self, frame: int | None = None) -> str:
        detector_error = self._detector_validation_error()
        if detector_error:
            return detector_error
        if not self._exact_starrynite_selected():
            return ""
        if self._detector_combo.currentData() != _STARRYNITE_DETECTOR_ID:
            return (
                "Exact current-frame testing requires the StarryNite detector. "
                "Select it before testing the source-bound detector."
            )
        profile = self._starrynite_profile
        if profile is None or self._starrynite_parameter_path is None:
            return (
                "Load a StarryNite parameter file before testing the exact "
                "detector."
            )
        distribution = profile.detector_settings.get(
            "STARRYNITE_DISTRIBUTION_FILE"
        )
        if not distribution or not Path(str(distribution)).is_file():
            return (
                "The loaded parameter file must select an existing detector "
                "distribution MAT file before exact detector testing."
            )
        calibration_warnings = self._starrynite_calibration_warnings(profile)
        if calibration_warnings:
            return (
                "Exact detector testing requires matching calibration. "
                + calibration_warnings[0]
            )
        expected_radius = profile.detector_settings.get("RADIUS")
        if expected_radius is not None and abs(
            self._radius_spin.value() - float(expected_radius)
        ) > 1e-9:
            return (
                "Save the changed nucleus radius to a parameter copy and reload "
                "it before exact detector testing."
            )
        expected_threshold = profile.detector_settings.get("INTENSITY_THRESHOLD")
        if expected_threshold is not None and abs(
            self._threshold_spin.value() - float(expected_threshold)
        ) > 1e-9:
            return (
                "Save the changed detection threshold to a parameter copy and "
                "reload it before exact detector testing."
            )
        if self._subpixel_check.isChecked() or self._median_check.isChecked():
            return (
                "Turn off native subpixel localization and median filtering "
                "before exact detector testing."
            )
        if frame is None:
            try:
                frame = self._read_current_frame()
            except (TypeError, ValueError):
                return (
                    "Exact current-frame testing requires the viewer at t=1. "
                    "Build the full draft to warm the sequential detector across "
                    "all requested frames."
                )
        if int(frame) != 1:
            return (
                "Exact current-frame testing is available only at t=1. Later "
                "frames depend on diameter and cell-count state from prior frames; "
                "build the full draft to warm that sequential state."
            )
        return ""

    def _set_detector_preview_tooltip(self, error: str) -> None:
        if error:
            tooltip = error
        elif self._exact_starrynite_selected():
            tooltip = (
                "Run the source-bound detector on t=1. For any later frame, build "
                "the full draft so all prior sequential detector state is warmed."
            )
        else:
            tooltip = (
                "Run only the detector on the current full 3D stack. No links or "
                "accept-capable draft are created."
            )
        self._detector_preview_button.setToolTip(tooltip)
        self._detector_preview_button.setAccessibleDescription(tooltip)

    def _settings_validation_error(self) -> str:
        detector_error = self._detector_validation_error()
        if detector_error:
            return detector_error
        if self._tracker_combo.count() == 0:
            return "No compatible tracker is installed."
        if self._start_spin.value() > self._end_spin.value():
            return "The start time must not be later than the end time."
        exact_error = self._exact_settings_validation_error()
        if exact_error:
            return exact_error
        if not self._dataset_is_empty():
            return (
                "Whole-dataset tracking requires an empty nuclei record. Undo "
                "curation edits or use Track Selected Cell for a selected lineage."
            )
        return ""

    def _exact_settings_validation_error(self) -> str:
        if not self._exact_starrynite_selected():
            return ""
        if self._detector_combo.currentData() != _STARRYNITE_DETECTOR_ID:
            return "Exact StarryNite tracking requires the StarryNite detector."
        if (
            self._start_spin.value() != self._range_start
            or self._end_spin.value() != self._range_end
        ):
            return (
                "Exact StarryNite tracking must cover the complete movie "
                f"(t={self._range_start}–{self._range_end}) to preserve sequential "
                "detector state, initialization, and MATLAB event order."
            )
        if self._exact_scope_error:
            return self._exact_scope_error
        profile = self._starrynite_profile
        if profile is None or self._starrynite_parameter_path is None:
            return "Load a StarryNite parameter file before selecting exact tracking."
        if self._starrynite_neutral_classifier_path is None:
            return (
                "No source-bound tracking model is available. Choose a bundled "
                "preset or use the advanced legacy-model exporter."
            )
        distribution = profile.detector_settings.get(
            "STARRYNITE_DISTRIBUTION_FILE"
        )
        if not distribution or not Path(str(distribution)).is_file():
            return (
                "The parameter file must select an existing detector distribution "
                "MAT file for exact tracking."
            )
        report = self._starrynite_compatibility_report
        if report is None:
            return "Compatibility has not been validated for this parameter file."
        try:
            readiness = report.backend(_STARRYNITE_EXACT_BACKEND)
        except KeyError:
            return "The installed registry does not expose the exact runtime."
        if not readiness.runnable:
            blocker = readiness.blockers[0]
            return f"Exact StarryNite tracking is blocked: {blocker.message}"
        calibration_warnings = self._starrynite_calibration_warnings(profile)
        if calibration_warnings:
            return "Exact tracking requires matching calibration. " + calibration_warnings[0]
        expected_radius = profile.detector_settings.get("RADIUS")
        if expected_radius is not None and abs(
            self._radius_spin.value() - float(expected_radius)
        ) > 1e-9:
            return (
                "The nucleus radius differs from the loaded parameter source. Save "
                "the tuned parameter copy and reload it before exact tracking."
            )
        expected_threshold = profile.detector_settings.get("INTENSITY_THRESHOLD")
        if expected_threshold is not None and abs(
            self._threshold_spin.value() - float(expected_threshold)
        ) > 1e-9:
            return (
                "The detection threshold differs from the loaded parameter source. "
                "Save the tuned parameter copy and reload it before exact tracking."
            )
        if self._subpixel_check.isChecked() or self._median_check.isChecked():
            return (
                "Subpixel localization and median filtering are native-only "
                "overrides; turn both off for exact tracking."
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
        detector_preview_error = self._detector_preview_validation_error()
        can_test = not detector_preview_error and self._state not in {
            self.RUNNING,
            self.CANCELING,
            self.ACCEPTING,
            self.READY,
        }
        self._detector_preview_button.setEnabled(bool(can_test))
        self._set_detector_preview_tooltip(detector_preview_error)
        return not bool(error)

    def _set_running(self, running: bool) -> None:
        self._settings_widget.setEnabled(not running)
        self._advanced_toggle.setEnabled(not running)
        self._advanced_widget.setEnabled(not running)
        self._starrynite_file_button.setEnabled(not running)
        self._starrynite_recent_button.setEnabled(not running)
        self._starrynite_save_button.setEnabled(
            not running and self._starrynite_profile is not None
        )
        self._starrynite_neutral_button.setEnabled(
            not running and self._starrynite_profile is not None
        )
        self._starrynite_report_button.setEnabled(
            not running and self._starrynite_compatibility_report is not None
        )
        self._reset_button.setEnabled(not running)
        self._preview_button.setEnabled(
            not running and not self._settings_validation_error()
        )
        self._detector_preview_button.setEnabled(
            not running
            and not self._detector_preview_validation_error()
            and self._state != self.READY
        )
        self._set_detector_preview_tooltip(
            self._detector_preview_validation_error() if not running else ""
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
        self._starrynite_detector_settings = {}
        self._starrynite_tracker_settings = {}
        self._starrynite_parameter_path = None
        self._starrynite_profile = None
        self._starrynite_neutral_classifier_path = None
        self._starrynite_compatibility_report = None
        self._starrynite_session_note_html = ""
        self._starrynite_classifier_note_html = ""
        self._starrynite_save_button.setEnabled(False)
        self._starrynite_neutral_button.setEnabled(False)
        self._starrynite_report_button.setEnabled(False)
        self._starrynite_file_label.clear()
        self._starrynite_file_label.setToolTip("")
        self._starrynite_file_label.setAccessibleDescription("")
        self._starrynite_file_label.hide()
        self._refresh_recent_parameter_button()
        self._start_spin.setValue(self._range_start)
        self._end_spin.setValue(self._range_end)
        self._channel_spin.setValue(1)
        self._radius_spin.setValue(4.0)
        self._threshold_spin.setValue(5.0)
        self._distance_spin.setValue(8.0)
        self._gap_spin.setValue(1)
        detector_id = self._detector_combo.currentData()
        default_subpixel = True
        if detector_id is not None:
            try:
                default_subpixel = bool(
                    self._registry.default_settings(str(detector_id)).get(
                        "DO_SUBPIXEL_LOCALIZATION",
                        True,
                    )
                )
            except (KeyError, ValueError):
                pass
        self._subpixel_check.setChecked(default_subpixel)
        self._median_check.setChecked(False)
        self._sync_division_capability(use_default=True)
        self._update_starrynite_behavior_visibility()

    def _set_advanced_visible(self, visible: bool) -> None:
        self._advanced_widget.setVisible(visible)
        self._update_workflow_visibility()
        self._refresh_recent_parameter_button()

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
        self._validate_settings()
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
