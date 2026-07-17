"""Modeless, iterative review workbench for selected-cell Auto Forward.

The dialog deliberately keeps analysis separate from acceptance.  Users can
adjust parameters, build a non-destructive draft, inspect it while navigating
the main image viewer, and rerun as often as needed before one explicit commit.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping

from qtpy.QtCore import QSettings, QThread, QTimer, Qt, Signal
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .tracking_preview import ExpandedTrackingPreview, PreviewSpot, expand_tracking_preview

if TYPE_CHECKING:
    from ..tracking.api import TrackingResult
    from .app import AceTreeApp

logger = logging.getLogger(__name__)


class AutoTrackForwardDialog(QDialog):
    """Configure, preview, refine, and accept one selected-cell continuation."""

    draftApplied = Signal(int)

    CONFIGURING = "configuring"
    RUNNING = "running"
    READY = "ready"
    OUTDATED = "outdated"
    STALE = "stale"
    EMPTY = "empty"
    FAILED = "failed"
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
        app: AceTreeApp | None = None,
        seed_anchor: tuple[int, int] | None = None,
        seed_label: str = "Selected cell",
        seed_radius_um: float = 4.0,
        initial_settings: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.app = app
        self._seed_anchor = seed_anchor
        self._start_time = start_time
        self._end_time = end_time
        self._seed_label = seed_label
        self._seed_radius_um = seed_radius_um
        self._proposal: TrackingResult | None = None
        self._expanded_preview: ExpandedTrackingPreview | None = None
        self._proposal_revision: int | None = None
        self._proposal_change_counter: int | None = None
        self._cancel_requested = False
        self._accepting = False
        self._accepted = False
        self._cleaned_up = False
        self._state = self.CONFIGURING
        self._stop_frame: int | None = None
        self._analysis_thread: QThread | None = None
        self._analysis_worker = None
        self._pending_analysis_outcome: tuple[str, object] | None = None
        self._cancel_event: Event | None = None
        self._close_after_run = False
        self._deferred_result = QDialog.Rejected
        self._rerun_after_thread = False
        self._run_started_at = 0.0
        self._generated_settings: dict[str, Any] | None = None
        self._generated_html = ""
        self._generated_duration = 0.0
        self._navigating_review = False
        self._starrynite_detector_settings: dict[str, Any] = {}
        self._starrynite_tracker_settings: dict[str, Any] = {}
        self._starrynite_parameter_path: Path | None = None
        self._starrynite_profile = None
        self._starrynite_neutral_classifier_path: Path | None = None
        self._starrynite_compatibility_report = None
        self._starrynite_session_note_html = ""
        self._starrynite_classifier_note_html = ""
        self._solo_channel_visibility: list[tuple[object, bool]] | None = None
        self._review_timer = QTimer(self)
        self._review_timer.setInterval(350)
        self._review_timer.timeout.connect(self._advance_review_playback)
        self._original_view = self._capture_view_state()

        self.setWindowTitle(f"Auto Forward — {seed_label}")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setMinimumSize(760, 520)
        self.resize(980, 690)
        # Keep the Python/Qt object alive through queued worker teardown. The
        # parent owns it and the Edit panel releases its reference on finish.

        self._build_ui(max(1, num_channels))
        self._apply_initial_settings(initial_settings or {})
        self._refresh_recent_parameter_button()
        self._connect_parameter_signals()
        self._apply_accessibility_descriptions()
        self._pause_host_playback()
        self._set_state(
            self.CONFIGURING,
            "Choose settings, then build a preview. The dataset will not change.",
        )

    @property
    def state(self) -> str:
        """Current review-workflow state, exposed for lightweight GUI tests."""

        return self._state

    @property
    def proposal(self) -> TrackingResult | None:
        return self._proposal

    def _build_ui(self, num_channels: int) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        title = QLabel(
            "<span style='font-size:16px'><b>Track "
            f"{html.escape(self._seed_label)} forward</b></span>"
            f"<br><span style='color:#9aa0a6'>Starting at t={self._start_time}. "
            "Existing nuclei and manual names are protected.</span>"
        )
        title.setWordWrap(True)
        outer.addWidget(title)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        outer.addLayout(columns, stretch=1)

        # ── Configure column ──────────────────────────────────────────
        configure_group = QGroupBox("1. Configure")
        configure_group.setMinimumWidth(300)
        configure_layout = QVBoxLayout(configure_group)
        self._settings_widget = QWidget()
        form = QFormLayout(self._settings_widget)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._end_spin = QSpinBox()
        self._end_spin.setRange(self._start_time + 1, max(self._start_time + 1, self._end_time))
        self._end_spin.setValue(max(self._start_time + 1, self._end_time))
        self._end_spin.setToolTip("Last timepoint Auto Forward should attempt")
        form.addRow("Track through:", self._end_spin)

        from ..tracking.registry import get_default_registry

        registry = get_default_registry()
        self._registry = registry
        self._detector_combo = QComboBox()
        for descriptor in registry.detector_descriptors():
            self._detector_combo.addItem(descriptor.display_name, descriptor.plugin_id)
        self._detector_combo.setToolTip("Method used to find nucleus-like bright blobs")
        form.addRow("Detector:", self._detector_combo)

        self._tracker_combo = QComboBox()
        for descriptor in registry.tracker_descriptors():
            if "global_only" in descriptor.capabilities:
                continue
            self._tracker_combo.addItem(descriptor.display_name, descriptor.plugin_id)
        self._tracker_combo.setToolTip("Method used to connect detections over time")
        form.addRow("Tracker:", self._tracker_combo)

        self._channel_spin = QSpinBox()
        self._channel_spin.setRange(1, num_channels)
        self._channel_spin.setValue(1)
        self._channel_spin.setToolTip("Image channel containing the nuclear signal")
        form.addRow("Image channel:", self._channel_spin)

        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.05, 100.0)
        self._radius_spin.setDecimals(2)
        self._radius_spin.setValue(max(0.05, self._seed_radius_um))
        self._radius_spin.setSuffix(" µm")
        self._radius_spin.setToolTip("Approximate physical radius of the selected nucleus")
        form.addRow("Nucleus radius:", self._radius_spin)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1_000_000.0)
        self._threshold_spin.setDecimals(4)
        self._threshold_spin.setValue(5.0)
        self._threshold_spin.setToolTip(
            "Minimum detector response. Higher values keep fewer, stronger candidates."
        )
        form.addRow("Detection threshold:", self._threshold_spin)

        self._roi_spin = QDoubleSpinBox()
        self._roi_spin.setRange(0.1, 1_000.0)
        self._roi_spin.setDecimals(2)
        self._roi_spin.setValue(max(12.0, self._seed_radius_um * 3.0))
        self._roi_spin.setSuffix(" µm")
        self._roi_spin.setToolTip("Radius searched around the predicted cell position")
        form.addRow("Search area:", self._roi_spin)

        self._distance_spin = QDoubleSpinBox()
        self._distance_spin.setRange(0.05, 1_000.0)
        self._distance_spin.setDecimals(2)
        self._distance_spin.setValue(max(8.0, self._seed_radius_um * 2.0))
        self._distance_spin.setSuffix(" µm")
        self._distance_spin.setToolTip("Largest plausible movement between linked frames")
        form.addRow("Maximum movement:", self._distance_spin)

        self._gap_spin = QSpinBox()
        self._gap_spin.setRange(0, 20)
        self._gap_spin.setValue(1)
        self._gap_spin.setToolTip("How many missing frames may be bridged by interpolation")
        form.addRow("Missing frames:", self._gap_spin)

        self._ambiguity_spin = QDoubleSpinBox()
        self._ambiguity_spin.setRange(1.01, 10.0)
        self._ambiguity_spin.setDecimals(2)
        self._ambiguity_spin.setSingleStep(0.05)
        self._ambiguity_spin.setValue(1.20)
        self._ambiguity_spin.setToolTip(
            "Higher values stop more cautiously when two candidates are similarly likely"
        )
        form.addRow("Caution for close choices:", self._ambiguity_spin)

        self._branch_policy_combo = QComboBox()
        self._branch_policy_combo.addItem(
            "Stop and review likely divisions",
            "stop",
        )
        self._branch_policy_combo.addItem(
            "Follow the best daughter only",
            "follow_best",
        )
        self._branch_policy_combo.addItem(
            "Follow both daughters",
            "follow_both",
        )
        self._follow_both_index = self._branch_policy_combo.findData("follow_both")
        self._branch_policy_combo.setToolTip(
            "Stop for review at a likely division, or include both daughter branches "
            "when the selected tracker supports splitting"
        )
        form.addRow("Division behavior:", self._branch_policy_combo)
        configure_layout.addWidget(self._settings_widget)

        self._starrynite_file_button = QPushButton("Start from StarryNite parameters...")
        self._starrynite_file_button.setToolTip(
            "Load a legacy StarryNite parameter file as a safe, tunable starting point"
        )
        self._starrynite_file_button.clicked.connect(self._choose_starrynite_parameter_file)
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
        self._starrynite_save_button = QPushButton("Save tuned parameter copy...")
        self._starrynite_save_button.setToolTip(
            "Save the loaded source with compatible threshold, radius, and gap "
            "changes appended losslessly"
        )
        self._starrynite_save_button.clicked.connect(
            self._choose_starrynite_parameter_destination
        )
        self._starrynite_save_button.setEnabled(False)
        configure_layout.addWidget(self._starrynite_save_button)
        self._starrynite_save_explanation = QLabel(
            "A saved copy updates only expected radius, intensity threshold, and "
            "missing frames. Search area, movement, and caution remain Python-only "
            "review settings."
        )
        self._starrynite_save_explanation.setWordWrap(True)
        self._starrynite_save_explanation.setAccessibleName(
            "StarryNite parameter save scope"
        )
        configure_layout.addWidget(self._starrynite_save_explanation)
        self._starrynite_file_label = QLabel()
        self._starrynite_file_label.setWordWrap(True)
        self._starrynite_file_label.setAccessibleName("Loaded StarryNite parameter file")
        self._starrynite_file_label.hide()
        configure_layout.addWidget(self._starrynite_file_label)
        starrynite_compatibility_actions = QHBoxLayout()
        self._starrynite_neutral_button = QPushButton(
            "Validate classifier export (report only)…"
        )
        self._starrynite_neutral_button.setToolTip(
            "Validate a numeric JSON classifier export against the MAT model "
            "referenced by the loaded parameter file. This does not change the "
            "tracker used by this draft."
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

        self._advanced_toggle = QCheckBox("Show advanced detection options")
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

        configure_actions = QHBoxLayout()
        self._reset_button = QPushButton("Restore defaults")
        self._reset_button.setToolTip("Restore recommended starting values")
        self._reset_button.clicked.connect(self._restore_defaults)
        self._preview_button = QPushButton("&Build Preview")
        self._preview_button.setDefault(True)
        self._preview_button.setToolTip("Analyze images without changing the dataset")
        self._preview_button.clicked.connect(self._run_preview)
        configure_actions.addWidget(self._reset_button)
        configure_actions.addStretch()
        configure_actions.addWidget(self._preview_button)
        configure_layout.addLayout(configure_actions)
        columns.addWidget(configure_group, stretch=0)

        # ── Review column ─────────────────────────────────────────────
        review_group = QGroupBox("2. Review")
        review_layout = QVBoxLayout(review_group)
        self._banner = QLabel()
        self._banner.setWordWrap(True)
        self._banner.setMinimumHeight(52)
        self._banner.setTextFormat(Qt.PlainText)
        self._banner.setAccessibleName("Tracking draft status")
        review_layout.addWidget(self._banner)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()
        review_layout.addWidget(self._progress_bar)

        self._summary_label = QLabel("No preview has been built yet.")
        self._summary_label.setWordWrap(True)
        self._summary_label.setAccessibleName("Tracking draft summary")
        review_layout.addWidget(self._summary_label)

        self._generated_label = QLabel()
        self._generated_label.setWordWrap(True)
        self._generated_label.setAccessibleName("Settings used to generate the visible draft")
        self._generated_label.hide()
        review_layout.addWidget(self._generated_label)

        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setAccessibleName("Tracking stop explanation")
        self._warning_label.hide()
        review_layout.addWidget(self._warning_label)

        self._legend_label = QLabel(
            "○ Proposed position   ◇ Interpolated gap   □/× Review candidate only   "
            "━ Movement path   ⊕ Predicted search region\n"
            "A forked path marks a proposed division with both daughters in the draft.\n"
            "Amber outline means the draft must be rebuilt before acceptance."
        )
        self._legend_label.setWordWrap(True)
        self._legend_label.setAccessibleName("Tracking preview legend")
        self._legend_label.setToolTip(
            "Symbols and text duplicate the overlay colors so review does not depend on color."
        )
        review_layout.addWidget(self._legend_label)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Time", "X (µm)", "Y (µm)", "Z (µm)", "Quality", "Draft status"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.currentCellChanged.connect(self._on_table_current_cell_changed)
        self._table.setAccessibleName("Proposed tracking positions")
        review_layout.addWidget(self._table, stretch=1)

        navigation = QHBoxLayout()
        self._previous_button = QPushButton("◀ Previous")
        self._previous_button.setEnabled(False)
        self._previous_button.clicked.connect(lambda: self._step_review_row(-1))
        self._next_button = QPushButton("Next ▶")
        self._next_button.setEnabled(False)
        self._next_button.clicked.connect(lambda: self._step_review_row(1))
        self._play_button = QPushButton("▶ Play Draft")
        self._play_button.setCheckable(True)
        self._play_button.setEnabled(False)
        self._play_button.setToolTip("Play only the time range represented by this draft")
        self._play_button.clicked.connect(self._toggle_review_playback)
        self._stop_button = QPushButton("Go to stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._go_to_stop)
        self._overlay_check = QCheckBox("Show draft on image")
        self._overlay_check.setChecked(True)
        self._overlay_check.toggled.connect(self._set_overlay_visible)
        self._center_check = QCheckBox("Center selected position")
        self._center_check.setChecked(True)
        self._center_check.setToolTip(
            "Center the 2D or 3D camera when a review row is selected"
        )
        navigation.addWidget(self._previous_button)
        navigation.addWidget(self._next_button)
        navigation.addWidget(self._play_button)
        navigation.addWidget(self._stop_button)
        navigation.addStretch()
        navigation.addWidget(self._overlay_check)
        review_layout.addLayout(navigation)

        review_options = QHBoxLayout()
        self._solo_channel_check = QCheckBox("Solo detection channel while reviewing")
        self._solo_channel_check.setToolTip(
            "Temporarily hide other image channels; their visibility is restored on close"
        )
        self._solo_channel_check.toggled.connect(self._set_detection_channel_solo)
        review_options.addWidget(self._center_check)
        review_options.addWidget(self._solo_channel_check)
        review_options.addStretch()
        review_layout.addLayout(review_options)
        columns.addWidget(review_group, stretch=1)

        footer = QHBoxLayout()
        self._cancel_run_button = QPushButton("Cancel analysis")
        self._cancel_run_button.clicked.connect(self._cancel_run)
        self._cancel_run_button.hide()
        self._discard_button = QPushButton("&Discard Draft")
        self._discard_button.setToolTip("Close without changing the dataset")
        self._discard_button.clicked.connect(self.reject)
        self._accept_button = QPushButton("&Accept Draft")
        self._accept_button.setEnabled(False)
        self._accept_button.setToolTip("Apply the visible draft as one undoable edit")
        self._accept_button.clicked.connect(self._accept_draft)
        footer.addWidget(self._cancel_run_button)
        footer.addStretch()
        footer.addWidget(self._discard_button)
        footer.addWidget(self._accept_button)
        outer.addLayout(footer)

    def _connect_parameter_signals(self) -> None:
        for widget in (
            self._end_spin,
            self._channel_spin,
            self._radius_spin,
            self._threshold_spin,
            self._roi_spin,
            self._distance_spin,
            self._gap_spin,
            self._ambiguity_spin,
        ):
            widget.valueChanged.connect(self._parameters_changed)
        self._detector_combo.currentIndexChanged.connect(self._detector_changed)
        self._tracker_combo.currentIndexChanged.connect(self._tracker_changed)
        self._branch_policy_combo.currentIndexChanged.connect(
            self._branch_policy_changed
        )
        self._subpixel_check.toggled.connect(self._parameters_changed)
        self._median_check.toggled.connect(self._parameters_changed)
        self._channel_spin.valueChanged.connect(self._refresh_solo_detection_channel)

    def _apply_accessibility_descriptions(self) -> None:
        """Expose every explanatory tooltip to assistive technology."""

        for widget in (
            self._end_spin,
            self._detector_combo,
            self._tracker_combo,
            self._channel_spin,
            self._radius_spin,
            self._threshold_spin,
            self._roi_spin,
            self._distance_spin,
            self._gap_spin,
            self._ambiguity_spin,
            self._branch_policy_combo,
            self._starrynite_file_button,
            self._starrynite_recent_button,
            self._starrynite_save_button,
            self._starrynite_neutral_button,
            self._starrynite_report_button,
            self._preview_button,
            self._play_button,
            self._overlay_check,
            self._center_check,
            self._solo_channel_check,
            self._discard_button,
            self._accept_button,
        ):
            description = widget.toolTip()
            if description:
                widget.setAccessibleDescription(description)

    def _apply_initial_settings(self, settings: Mapping[str, Any]) -> None:
        detector_preset = settings.get("starrynite_detector_settings", {})
        tracker_preset = settings.get("starrynite_tracker_settings", {})
        saved_detector = (
            dict(detector_preset) if isinstance(detector_preset, Mapping) else {}
        )
        saved_tracker = (
            dict(tracker_preset) if isinstance(tracker_preset, Mapping) else {}
        )
        requested_modes = (
            settings.get("starrynite_compatibility_backend"),
            saved_tracker.get("STARRYNITE_COMPATIBILITY_MODE"),
        )
        for requested_mode in requested_modes:
            if requested_mode in (None, "", "native_fast"):
                continue
            raise ValueError(
                "Auto Forward cannot restore StarryNite compatibility backend "
                f"{requested_mode!r}; this workbench supports only the explicit "
                "native_fast backend"
            )
        if isinstance(detector_preset, Mapping):
            self._starrynite_detector_settings = saved_detector
        if isinstance(tracker_preset, Mapping):
            self._starrynite_tracker_settings = saved_tracker

        parameter_path = (
            settings.get("starrynite_parameter_file")
            or saved_detector.get("STARRYNITE_PARAMETER_FILE")
            or saved_tracker.get("STARRYNITE_PARAMETER_FILE")
        )
        neutral_path = settings.get("starrynite_neutral_classifier_file")
        if parameter_path:
            try:
                self.load_starrynite_parameter_file(
                    str(parameter_path),
                    cell_count=self._alive_cell_count_at_start(),
                    neutral_classifier_path=(neutral_path or None),
                )
            except Exception as exc:
                logger.warning(
                    "Could not restore the StarryNite parameter session",
                    exc_info=True,
                )
                self._starrynite_detector_settings = saved_detector
                self._starrynite_tracker_settings = saved_tracker
                self._starrynite_parameter_path = Path(str(parameter_path))
                self._starrynite_profile = None
                self._starrynite_neutral_classifier_path = None
                self._starrynite_compatibility_report = None
                self._starrynite_file_label.setText(
                    "The previous StarryNite parameter file is unavailable or "
                    f"unreadable: {html.escape(str(parameter_path))}. Stored tuning "
                    "values remain active for native tracking, but compatibility "
                    "details and save-copy are disabled."
                )
                self._starrynite_file_label.setToolTip(str(exc))
                self._starrynite_file_label.show()
                self._starrynite_save_button.setEnabled(False)
                self._starrynite_neutral_button.setEnabled(False)
                self._starrynite_report_button.setEnabled(False)
            else:
                fresh_detector = self._starrynite_detector_settings
                fresh_tracker = self._starrynite_tracker_settings
                identity_keys = (
                    (saved_detector, fresh_detector, "STARRYNITE_PARAMETER_SHA256"),
                    (saved_detector, fresh_detector, "STARRYNITE_STAGE_INDEX"),
                    (saved_detector, fresh_detector, "STARRYNITE_CELL_COUNT"),
                    (saved_tracker, fresh_tracker, "STARRYNITE_MODEL_SHA256"),
                    (saved_tracker, fresh_tracker, "STARRYNITE_MODEL_FILE"),
                )
                rebased = any(
                    old.get(key) not in (None, "")
                    and old.get(key) != fresh.get(key)
                    for old, fresh, key in identity_keys
                )
                restored_notes = ["<b>Restored parameter session.</b>"]
                if rebased:
                    restored_notes.append(
                        "<b>Source/model changed:</b> staged preset metadata was "
                        "refreshed; your visible tuning edits were kept."
                    )
                self._starrynite_session_note_html = "<br>".join(restored_notes)

        # Reapply only user-facing choices after refreshing source-controlled
        # stage, hash, and model metadata from disk.
        self._select_combo_value(self._detector_combo, settings.get("detector_id"))
        self._select_combo_value(self._tracker_combo, settings.get("tracker_id"))
        values = (
            (self._end_spin, "end_time"),
            (self._channel_spin, "channel"),
            (self._radius_spin, "radius_um"),
            (self._threshold_spin, "threshold"),
            (self._roi_spin, "roi_radius_um"),
            (self._distance_spin, "max_distance_um"),
            (self._gap_spin, "missing_frames"),
            (self._ambiguity_spin, "ambiguity_ratio"),
        )
        for widget, key in values:
            if key in settings:
                widget.setValue(settings[key])
        if "subpixel" in settings:
            self._subpixel_check.setChecked(bool(settings["subpixel"]))
        if "median_filter" in settings:
            self._median_check.setChecked(bool(settings["median_filter"]))
        if "show_overlay" in settings:
            self._overlay_check.setChecked(bool(settings["show_overlay"]))
        branch_policy = str(settings.get("branch_policy", "stop"))
        branch_index = self._branch_policy_combo.findData(branch_policy)
        if branch_index >= 0:
            self._branch_policy_combo.setCurrentIndex(branch_index)
        self._update_division_behavior_availability()
        self._render_starrynite_file_summary()

    @staticmethod
    def _select_combo_value(combo: QComboBox, value: Any) -> None:
        if value is None:
            return
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _tracker_changed(self, *_args) -> None:
        self._update_division_behavior_availability()
        self._render_starrynite_file_summary()
        self._parameters_changed()

    def _detector_changed(self, *_args) -> None:
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
        self._parameters_changed()

    def _branch_policy_changed(self, *_args) -> None:
        self._update_division_behavior_availability()
        self._parameters_changed()

    def _tracker_capabilities(self) -> set[str]:
        tracker_id = self._tracker_combo.currentData()
        if tracker_id is None:
            return set()
        try:
            descriptor = self._registry.get_descriptor(str(tracker_id))
        except KeyError:
            return set()
        return {
            str(capability).strip().lower()
            for capability in descriptor.capabilities
        }

    def _tracker_supports_splitting(self) -> bool:
        return "splitting" in self._tracker_capabilities()

    def _update_division_behavior_availability(self) -> None:
        supports_splitting = self._tracker_supports_splitting()
        model = self._branch_policy_combo.model()
        item_getter = getattr(model, "item", None)
        item = (
            item_getter(self._follow_both_index)
            if callable(item_getter) and self._follow_both_index >= 0
            else None
        )
        if item is not None:
            item.setEnabled(supports_splitting)
            item.setToolTip(
                "Include both daughter branches in the reviewed draft."
                if supports_splitting
                else "The selected tracker does not advertise splitting support."
            )
        if (
            not supports_splitting
            and self._branch_policy_combo.currentData() == "follow_both"
        ):
            stop_index = self._branch_policy_combo.findData("stop")
            if stop_index >= 0:
                self._branch_policy_combo.setCurrentIndex(stop_index)
        follows_both = self._branch_policy_combo.currentData() == "follow_both"
        supports_frontier_gaps = "frontier_tracking" in self._tracker_capabilities()
        if follows_both and not supports_frontier_gaps:
            self._gap_spin.setValue(0)
        self._gap_spin.setEnabled(not follows_both or supports_frontier_gaps)
        self._gap_spin.setToolTip(
            "How many missing frames may be bridged by interpolation"
            if not follows_both or supports_frontier_gaps
            else "This division tracker cannot combine daughter branches with gap closing"
        )

    def export_settings(self, *, revalidate: bool = True) -> dict[str, Any]:
        """Return user choices so reopening Auto Forward preserves refinements."""

        if revalidate:
            self._refresh_starrynite_compatibility()
        return {
            "detector_id": self._detector_combo.currentData(),
            "tracker_id": self._tracker_combo.currentData(),
            "end_time": self._end_spin.value(),
            "channel": self._channel_spin.value(),
            "radius_um": self._radius_spin.value(),
            "threshold": self._threshold_spin.value(),
            "roi_radius_um": self._roi_spin.value(),
            "max_distance_um": self._distance_spin.value(),
            "missing_frames": self._gap_spin.value(),
            "ambiguity_ratio": self._ambiguity_spin.value(),
            "branch_policy": str(self._branch_policy_combo.currentData()),
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
                "native_fast"
                if self._tracker_combo.currentData()
                == "acetree.starrynite_division"
                else None
            ),
            "starrynite_detector_settings": dict(self._starrynite_detector_settings),
            "starrynite_tracker_settings": dict(self._starrynite_tracker_settings),
        }

    def get_request(self, seed_anchor: tuple[int, int] | None = None):
        """Build the immutable selected-forward request represented by the form."""

        self._refresh_starrynite_compatibility()
        from ..tracking.api import ComponentSpec, TrackingRequest, TrackingScope
        anchor = seed_anchor or self._seed_anchor
        if anchor is None:
            raise ValueError("Auto Forward needs a selected seed nucleus")
        registry = self._registry
        detector_id = str(self._detector_combo.currentData())
        tracker_id = str(self._tracker_combo.currentData())
        if not detector_id or detector_id == "None":
            raise ValueError("No compatible detector is installed")
        if not tracker_id or tracker_id == "None":
            raise ValueError("No compatible tracker is installed")
        branch_policy = str(self._branch_policy_combo.currentData())
        if branch_policy == "follow_both" and not self._tracker_supports_splitting():
            raise ValueError(
                "The selected tracker cannot follow both daughters at a division"
            )
        if (
            branch_policy == "follow_both"
            and self._gap_spin.value() > 0
            and "frontier_tracking" not in self._tracker_capabilities()
        ):
            raise ValueError(
                "The selected division tracker cannot close gaps across daughter branches"
            )

        detector_settings = registry.default_settings(detector_id)
        tracker_settings = registry.default_settings(tracker_id)
        if detector_id == "acetree.starrynite_detector":
            detector_settings.update(self._starrynite_detector_settings)
        if tracker_id == "acetree.starrynite_division":
            tracker_settings.update(self._starrynite_tracker_settings)
        detector_common = {
            "TARGET_CHANNEL": self._channel_spin.value(),
            "RADIUS": self._radius_spin.value(),
            "THRESHOLD": self._threshold_spin.value(),
            "DO_SUBPIXEL_LOCALIZATION": self._subpixel_check.isChecked(),
            "DO_MEDIAN_FILTERING": self._median_check.isChecked(),
        }
        if detector_id == "acetree.starrynite_detector":
            detector_common["THRESHOLD"] = 0.0
            detector_common["INTENSITY_THRESHOLD"] = self._threshold_spin.value()
        gap_frames = self._gap_spin.value()
        max_distance = self._distance_spin.value()
        tracker_common = {
            "LINKING_MAX_DISTANCE": max_distance,
            "ALLOW_GAP_CLOSING": gap_frames > 0,
            "GAP_CLOSING_MAX_DISTANCE": max_distance,
            "MAX_FRAME_GAP": gap_frames + 1 if gap_frames > 0 else 1,
            "ALLOW_TRACK_SPLITTING": branch_policy == "follow_both",
            "ALLOW_TRACK_MERGING": False,
        }
        detector_schema = registry.get_descriptor(detector_id).settings_schema
        tracker_schema = registry.get_descriptor(tracker_id).settings_schema
        detector_settings.update(
            (key, value) for key, value in detector_common.items() if key in detector_schema
        )
        tracker_settings.update(
            (key, value) for key, value in tracker_common.items() if key in tracker_schema
        )
        return TrackingRequest(
            detector=ComponentSpec(plugin_id=detector_id, settings=detector_settings),
            tracker=ComponentSpec(plugin_id=tracker_id, settings=tracker_settings),
            scope=TrackingScope(
                kind="selected_forward",
                start_frame=self._start_time,
                end_frame=self._end_spin.value(),
                seed_anchors=(anchor,),
                roi_radius_um=self._roi_spin.value(),
                ambiguity_ratio=self._ambiguity_spin.value(),
                branch_policy=branch_policy,
            ),
        )

    @staticmethod
    def _settings_store() -> QSettings:
        return QSettings("AceTree", "AceTreePy")

    def _starrynite_components_active(self) -> bool:
        return (
            self._detector_combo.currentData() == "acetree.starrynite_detector"
            or self._tracker_combo.currentData() == "acetree.starrynite_division"
        )

    def _render_starrynite_file_summary(self) -> None:
        profile = self._starrynite_profile
        if profile is None:
            return
        source = profile.parameters.source_path or self._starrynite_parameter_path
        source_name = "in-memory parameters" if source is None else source.name
        tracker_is_starrynite = (
            self._tracker_combo.currentData() == "acetree.starrynite_division"
        )
        detector_is_starrynite = (
            self._detector_combo.currentData() == "acetree.starrynite_detector"
        )
        parts = [
            f"Loaded {html.escape(source_name)} (stage {profile.stage_index + 1}, "
            f"{profile.cell_count} cells). Tune the values above, then preview."
        ]
        if profile.model_path is not None:
            if tracker_is_starrynite:
                parts.append(
                    "<b>Model compatibility:</b> The legacy tracking model is "
                    "preserved for provenance but is not applied by this workbench; "
                    "the draft uses the native geometry scorer."
                )
            elif detector_is_starrynite:
                parts.append(
                    "<b>Model compatibility:</b> The loaded model is reporting-only; "
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
            if tracker_is_starrynite:
                parts.append(
                    f"Classifier export {html.escape(neutral_path.name)} was validated "
                    "as source-bound for reporting only; native geometry scoring "
                    "remains active."
                )
            else:
                parts.append(
                    f"Classifier export {html.escape(neutral_path.name)} is "
                    "source-bound to the loaded preset for reporting only; the "
                    "selected tracker does not use it."
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
        summary = "<br>".join(parts)
        details = [
            *(() if report is None else (report.format_text(),)),
            *presentation_warnings,
        ]
        self._starrynite_file_label.setText(summary)
        self._starrynite_file_label.setToolTip("\n".join(details))
        self._starrynite_file_label.setAccessibleDescription("\n".join(details))
        self._starrynite_file_label.show()

    def _refresh_starrynite_compatibility(self) -> None:
        profile = self._starrynite_profile
        if profile is None:
            return
        from ..tracking.starrynite import build_compatibility_report

        neutral_path = self._starrynite_neutral_classifier_path
        report = build_compatibility_report(
            profile,
            neutral_classifier_path=neutral_path,
        )
        if neutral_path is not None and not report.neutral_classifier_source_bound:
            self._forget_neutral_classifier(profile)
            self._starrynite_neutral_classifier_path = None
            self._starrynite_classifier_note_html = (
                "<b>Classifier binding changed:</b> "
                f"{html.escape(neutral_path.name)} is no longer attached. Reload the "
                "parameter file and validate a fresh export."
            )
            report = build_compatibility_report(profile)
        elif neutral_path is not None:
            self._starrynite_classifier_note_html = ""
        self._starrynite_compatibility_report = report
        self._render_starrynite_file_summary()

    def recent_starrynite_parameter_file(self) -> Path | None:
        """Return the last usable parameter file selected in any workbench."""

        try:
            value = self._settings_store().value(self._RECENT_PARAMETERS_KEY, "")
        except (OSError, RuntimeError):
            return None
        if value is None or not str(value).strip():
            return None
        candidate = Path(str(value)).expanduser()
        return candidate.resolve(strict=False) if candidate.is_file() else None

    def _remember_starrynite_parameter_file(self, path: Path) -> None:
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
            self._show_failure("Could not read the recent parameter file.", exc)

    def _choose_starrynite_parameter_file(self) -> None:
        initial = self._starrynite_parameter_path or self.recent_starrynite_parameter_file()
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
            self._show_failure("Could not read that StarryNite parameter file.", exc)

    def load_starrynite_parameter_file(
        self,
        path: str,
        *,
        cell_count: int | None = None,
        neutral_classifier_path: str | Path | None = None,
    ) -> None:
        """Apply a legacy parameter file as editable sparse-tracking defaults."""

        from ..tracking.starrynite import (
            build_compatibility_report,
            load_tuning_profile,
        )

        profile = load_tuning_profile(
            path,
            cell_count=(
                self._alive_cell_count_at_start()
                if cell_count is None
                else cell_count
            ),
            fallback_radius_um=self._radius_spin.value(),
        )
        self._starrynite_session_note_html = ""
        self._starrynite_classifier_note_html = ""
        self._starrynite_detector_settings = dict(profile.detector_settings)
        self._starrynite_tracker_settings = dict(profile.tracker_settings)
        source = profile.parameters.source_path or Path(path)
        self._starrynite_parameter_path = source.resolve(strict=False)
        self._starrynite_profile = profile
        restored_from_settings = neutral_classifier_path is None
        requested_neutral = (
            self._remembered_neutral_classifier(profile)
            if restored_from_settings
            else Path(neutral_classifier_path).expanduser().resolve(strict=False)
        )
        self._starrynite_compatibility_report = build_compatibility_report(
            profile,
            neutral_classifier_path=requested_neutral,
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
        self._select_combo_value(self._detector_combo, "acetree.starrynite_detector")
        self._select_combo_value(self._tracker_combo, "acetree.starrynite_division")
        self._subpixel_check.setChecked(
            bool(profile.detector_settings.get("DO_SUBPIXEL_LOCALIZATION", False))
        )

        radius = float(profile.detector_settings.get("RADIUS", self._radius_spin.value()))
        threshold = float(
            profile.detector_settings.get(
                "INTENSITY_THRESHOLD",
                self._threshold_spin.value(),
            )
        )
        self._radius_spin.setValue(radius)
        self._threshold_spin.setValue(threshold)
        self._roi_spin.setValue(max(self._roi_spin.minimum(), radius * 3.0))
        self._distance_spin.setValue(max(self._distance_spin.minimum(), radius * 2.0))
        max_frame_gap = int(profile.tracker_settings.get("MAX_FRAME_GAP", 2))
        self._gap_spin.setValue(max(0, max_frame_gap - 1))
        follow_both = self._branch_policy_combo.findData("follow_both")
        if follow_both >= 0:
            self._branch_policy_combo.setCurrentIndex(follow_both)
        self._render_starrynite_file_summary()
        self._parameters_changed()

    def _choose_starrynite_neutral_classifier(self) -> None:
        profile = self._starrynite_profile
        if profile is None:
            return
        initial = self._starrynite_neutral_classifier_path
        if initial is None:
            initial = profile.model_path or self._starrynite_parameter_path
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose a StarryNite classifier export",
            "" if initial is None else str(initial),
            "Classifier export JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.attach_starrynite_neutral_classifier(path)
        except Exception as exc:
            logger.exception("Could not validate neutral StarryNite classifier")
            self._show_failure("That classifier export could not be used.", exc)

    def attach_starrynite_neutral_classifier(self, path: str | Path) -> None:
        """Validate and retain a neutral export without claiming run support."""

        from ..tracking.starrynite import build_compatibility_report

        profile = self._starrynite_profile
        if profile is None:
            raise ValueError("Load a StarryNite parameter file first")
        candidate = Path(path).expanduser().resolve(strict=False)
        report = build_compatibility_report(
            profile,
            neutral_classifier_path=candidate,
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
                else "The classifier export could not be proven source-bound."
            )
            previous_path = self._starrynite_neutral_classifier_path
            if previous_path is not None:
                previous_report = build_compatibility_report(
                    profile,
                    neutral_classifier_path=previous_path,
                )
                if previous_report.neutral_classifier_source_bound:
                    self._starrynite_compatibility_report = previous_report
                    self._starrynite_classifier_note_html = ""
                    self._render_starrynite_file_summary()
                    raise ValueError(error_message)
                self._forget_neutral_classifier(profile)
            self._starrynite_neutral_classifier_path = None
            self._starrynite_compatibility_report = build_compatibility_report(profile)
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

    def _show_starrynite_compatibility_report(self) -> None:
        self._refresh_starrynite_compatibility()
        report = self._starrynite_compatibility_report
        if report is None:
            return
        message = QMessageBox(self)
        message.setWindowTitle("StarryNite Compatibility")
        message.setIcon(QMessageBox.Information)
        if self._tracker_combo.currentData() == "acetree.starrynite_division":
            summary = (
                "The selected StarryNite tracker runs native geometry tracking. "
                "Details also report whether the inputs are valid for a separate "
                "exact-refinement runtime; validating an export does not change this "
                "draft."
            )
        elif self._detector_combo.currentData() == "acetree.starrynite_detector":
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
            self._show_failure("Could not save the tuned parameter file.", exc)
            return
        message = "The tuned copy was saved and is now the active parameter file."
        if warnings:
            message += "\n\n" + "\n".join(warnings)
        QMessageBox.information(
            self,
            "Parameter Copy Saved",
            message,
        )

    def save_starrynite_parameter_file(self, path: str) -> tuple[str, ...]:
        """Save compatible basic edits and make the new copy the active preset."""

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
                cell_count=self._alive_cell_count_at_start(),
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
        if source.parent.resolve(strict=False) != destination.parent.resolve(strict=False):
            warnings.append(
                "This copy is in a different folder. Check any relative model paths "
                "before using it in MATLAB."
            )
        # Activating the saved copy refreshes source-controlled hashes and
        # staged values.  Keep every Python-only review choice unchanged;
        # users should not lose a tuned ROI, movement limit, caution level, or
        # division policy merely because they saved the three compatible
        # legacy edits.
        session_choices = {
            "detector": self._detector_combo.currentData(),
            "tracker": self._tracker_combo.currentData(),
            "end": self._end_spin.value(),
            "channel": self._channel_spin.value(),
            "roi": self._roi_spin.value(),
            "distance": self._distance_spin.value(),
            "ambiguity": self._ambiguity_spin.value(),
            "branch_policy": self._branch_policy_combo.currentData(),
            "subpixel": self._subpixel_check.isChecked(),
            "median": self._median_check.isChecked(),
        }
        self.load_starrynite_parameter_file(
            str(destination),
            neutral_classifier_path=self._starrynite_neutral_classifier_path,
        )
        self._select_combo_value(self._detector_combo, session_choices["detector"])
        self._select_combo_value(self._tracker_combo, session_choices["tracker"])
        self._end_spin.setValue(int(session_choices["end"]))
        self._channel_spin.setValue(int(session_choices["channel"]))
        self._roi_spin.setValue(float(session_choices["roi"]))
        self._distance_spin.setValue(float(session_choices["distance"]))
        self._ambiguity_spin.setValue(float(session_choices["ambiguity"]))
        branch_index = self._branch_policy_combo.findData(
            session_choices["branch_policy"]
        )
        if branch_index >= 0:
            self._branch_policy_combo.setCurrentIndex(branch_index)
        self._subpixel_check.setChecked(bool(session_choices["subpixel"]))
        self._median_check.setChecked(bool(session_choices["median"]))
        self._starrynite_session_note_html = (
            "Compatible edits were saved to this copy."
        )
        self._render_starrynite_file_summary()
        return tuple(warnings)

    def _alive_cell_count_at_start(self) -> int | None:
        manager = getattr(self.app, "manager", None)
        record = getattr(manager, "nuclei_record", None)
        if record is None or not (1 <= self._start_time <= len(record)):
            return None
        return sum(
            1
            for nucleus in record[self._start_time - 1]
            if bool(getattr(nucleus, "is_alive", False))
        )

    def _starrynite_calibration_warnings(self, profile) -> tuple[str, ...]:
        manager = getattr(self.app, "manager", None)
        config = getattr(manager, "config", None)
        if config is None:
            return ()
        warnings: list[str] = []
        pairs = (
            ("xyres", profile.xy_um, getattr(config, "xy_res", None), "pixel"),
            ("zres", profile.z_um, getattr(config, "z_res", None), "plane"),
        )
        for name, parameter_value, dataset_value, unit in pairs:
            if parameter_value is None or dataset_value is None:
                continue
            parameter_number = float(parameter_value)
            dataset_number = float(dataset_value)
            tolerance = max(
                1e-9,
                1e-6 * max(abs(parameter_number), abs(dataset_number)),
            )
            if abs(parameter_number - dataset_number) <= tolerance:
                continue
            warnings.append(
                f"Parameter {name} is {parameter_number:g} µm/{unit}, but this "
                f"dataset uses {dataset_number:g}; physical sizes come from the "
                "parameter file while image sampling follows the dataset calibration."
            )
        return tuple(warnings)

    def _run_preview(self) -> None:
        if self._analysis_thread is not None:
            # A worker can have delivered its result while its QThread is
            # still draining the final ``finished`` event.  Preserve a fast
            # user click on Update Preview and start it as soon as teardown
            # completes instead of silently ignoring the click.
            if self._state != self.RUNNING:
                self._rerun_after_thread = True
                self._preview_button.setEnabled(False)
            return
        if self.app is None or self._seed_anchor is None:
            self._set_state(self.FAILED, "Auto Forward is not connected to an open dataset.")
            return
        try:
            request = self.get_request()
        except Exception as exc:
            self._show_failure("Check the tracking settings and try again.", exc)
            return

        try:
            prepare = getattr(self.app, "prepare_tracking_analysis", None)
            analyze_prepared = getattr(self.app, "analyze_prepared_tracking", None)
            if callable(prepare) and callable(analyze_prepared):
                snapshot = prepare(request)

                def analysis(progress, cancelled):
                    proposal = analyze_prepared(
                        snapshot,
                        progress=progress,
                        cancelled=cancelled,
                    )
                    return proposal, snapshot.revision, snapshot.change_counter

            else:

                def analysis(progress, cancelled):
                    proposal, revision = self.app.analyze_tracking_request(
                        request,
                        progress=progress,
                        cancelled=cancelled,
                    )
                    counter = getattr(self.app.edit_history, "change_counter", revision)
                    return proposal, revision, counter

        except Exception as exc:
            self._show_failure("Could not prepare a tracking snapshot.", exc)
            return

        self._cancel_requested = False
        self._cancel_event = Event()
        self._pending_analysis_outcome = None
        self._run_started_at = perf_counter()
        self._set_running(True)
        self._set_state(self.RUNNING, f"Analyzing {self._seed_label}…")
        self._warning_label.hide()
        self._progress_bar.setRange(0, max(1, request.scope.end_frame - self._start_time))
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Starting…")

        from .tracking_worker import TrackingAnalysisWorker

        thread = QThread(self)
        worker = TrackingAnalysisWorker(analysis, self._cancel_event)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_analysis_progress)
        worker.succeeded.connect(self._on_analysis_succeeded)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._defer_analysis_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._analysis_thread = thread
        self._analysis_worker = worker
        thread.start()

    def _on_analysis_progress(self, done: int, total: int, message: str) -> None:
        self._progress_bar.setRange(0, max(1, total))
        self._progress_bar.setValue(done)
        self._progress_bar.setFormat(f"{message}  %p%")
        self._set_state(self.RUNNING, message)

    def _on_analysis_succeeded(self, payload: object) -> None:
        # A worker result is not user-visible completion until QThread has
        # emitted ``finished``.  Publishing READY/EMPTY here leaves a narrow
        # interval in which callers can close or rerun while Qt still owns a
        # live thread; under a long GUI suite that can destroy a draining
        # QThread and abort the process.  Apply the result from
        # ``_on_analysis_thread_finished`` instead.
        self._pending_analysis_outcome = ("succeeded", payload)

    def _apply_analysis_succeeded(self, payload: object) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            self._finish_cancelled_analysis()
            return
        try:
            proposal, revision, change_counter = payload  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            self._apply_analysis_failed(exc)
            return

        self._set_running(False)
        self._proposal = proposal
        self._expanded_preview = expand_tracking_preview(proposal)
        self._proposal_revision = int(revision)
        self._proposal_change_counter = int(change_counter)
        self._generated_settings = self.export_settings()
        self._generated_duration = max(0.0, perf_counter() - self._run_started_at)
        self._populate_review()
        self._show_viewer_preview(stale=False)
        self._preview_button.setText("&Update Preview")

        count = self._expanded_preview.proposed_count
        split_count = self._expanded_preview.split_count
        if count <= 0:
            self._set_state(
                self.EMPTY,
                "No continuation was found. Try a lower threshold, a larger search area, "
                "or Manual Track.",
            )
            self._accept_button.setEnabled(False)
        elif proposal.warnings:
            division_note = (
                f" It includes {split_count} proposed division"
                f"{'s' if split_count != 1 else ''}."
                if split_count
                else ""
            )
            self._set_state(
                self.READY,
                "A partial draft is ready. Inspect the stopping point before accepting."
                + division_note,
                warning=True,
            )
            self._accept_button.setEnabled(True)
        elif split_count:
            self._set_state(
                self.READY,
                f"Draft includes {split_count} proposed division"
                f"{'s' if split_count != 1 else ''}. Review both daughters before "
                "accepting.",
                success=True,
            )
            self._accept_button.setEnabled(True)
        else:
            self._set_state(
                self.READY,
                "Draft ready through "
                f"t={max(spot.frame for spot in self._expanded_preview.spots)}.",
                success=True,
            )
            self._accept_button.setEnabled(True)

        self._accept_button.setText(f"&Accept {count} Position{'s' if count != 1 else ''}")
        self._play_button.setEnabled(bool(self._expanded_preview.spots))

    def _on_analysis_failed(self, exc: object) -> None:
        self._pending_analysis_outcome = ("failed", exc)

    def _apply_analysis_failed(self, exc: object) -> None:
        from ..tracking.pipeline import TrackingCancelled

        self._set_running(False)
        if isinstance(exc, TrackingCancelled) or (
            self._cancel_event is not None and self._cancel_event.is_set()
        ):
            self._finish_cancelled_analysis()
            return
        error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
        logger.error(
            "Selected-forward tracking failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        self._show_failure(
            "Auto Forward could not build a draft. No changes were made.",
            error,
        )
        if self._proposal is not None:
            self._show_viewer_preview(stale=True)

    def _finish_cancelled_analysis(self) -> None:
        self._set_running(False)
        if self._proposal is None:
            self._set_state(
                self.CONFIGURING,
                "Tracking canceled. No changes were made; adjust settings or try again.",
            )
        else:
            self._set_state(
                self.OUTDATED,
                "Update canceled. The previous draft remains visible but cannot be accepted "
                "until it is rebuilt.",
                warning=True,
            )
            self._show_viewer_preview(stale=True)
        self._accept_button.setEnabled(False)

    def _defer_analysis_thread_finished(self) -> None:
        """Keep the QThread wrapper alive until its ``finished`` signal returns."""

        # Clearing ``self._analysis_thread`` from inside QThread.finished can
        # destroy the last Python wrapper while Qt is still dispatching that
        # signal, which intermittently aborts long GUI test sessions and can
        # affect rapid real-world cancel/rerun workflows.  Finish the dialog
        # state transition on the next GUI turn instead.
        QTimer.singleShot(0, self._on_analysis_thread_finished)

    def _on_analysis_thread_finished(self) -> None:
        outcome = self._pending_analysis_outcome
        self._pending_analysis_outcome = None
        self._analysis_thread = None
        self._analysis_worker = None
        if self._close_after_run:
            self._close_after_run = False
            result = self._deferred_result
            self._cancel_event = None
            self._cleanup()
            QDialog.done(self, result)
            return

        if self._cancel_event is not None and self._cancel_event.is_set():
            self._finish_cancelled_analysis()
        elif outcome is None:
            self._apply_analysis_failed(
                RuntimeError("Tracking worker stopped without reporting a result")
            )
        elif outcome[0] == "succeeded":
            try:
                self._apply_analysis_succeeded(outcome[1])
            except Exception as exc:
                # Compatibility/source revalidation is intentionally allowed
                # to fail closed.  Convert any such finish-time exception into
                # the ordinary non-mutating failure state instead of letting an
                # exception escape a Qt slot.
                self._apply_analysis_failed(exc)
        else:
            self._apply_analysis_failed(outcome[1])
        self._cancel_event = None

        if self._rerun_after_thread:
            self._rerun_after_thread = False
            QTimer.singleShot(0, self._run_preview)

    def _populate_review(self) -> None:
        assert self._proposal is not None
        assert self._expanded_preview is not None
        preview = self._expanded_preview
        proposed = [spot for spot in preview.spots if spot.kind != "seed"]
        candidates = tuple(getattr(preview, "candidates", ()))
        review_spots = tuple(getattr(preview, "review_spots", (*preview.spots, *candidates)))
        links = len(preview.links)
        gaps = preview.interpolated_count
        splits = preview.split_count
        if proposed:
            actual_start = min(spot.frame for spot in proposed)
            actual_end = max(spot.frame for spot in proposed)
            span = f"t={actual_start}–{actual_end}"
        else:
            span = f"after t={self._start_time}"
        division_summary = (
            f" · {splits} proposed division{'s' if splits != 1 else ''}"
            if splits
            else ""
        )
        self._summary_label.setText(
            f"<b>{len(proposed)} proposed position(s)</b> · {span} · "
            f"{links} link segment(s) · {gaps} interpolated gap position(s) · "
            f"{len(candidates)} review-only candidate(s){division_summary}"
        )

        request = self._proposal.request
        detector = request.detector.plugin_id
        tracker = request.tracker.plugin_id
        channel = request.detector.settings.get("TARGET_CHANNEL", "?")
        self._generated_html = (
            "<b>Generated with</b> "
            f"{html.escape(detector)} + {html.escape(tracker)} · channel {channel} · "
            f"t={request.scope.start_frame}–{request.scope.end_frame} · "
            f"document revision {self._proposal_revision} · {self._generated_duration:.2f} s"
        )
        self._generated_label.setText(self._generated_html)
        self._generated_label.show()

        self._table.setRowCount(len(review_spots))
        split_targets = {
            link.target_id for link in preview.links if link.kind == "split"
        }
        for row, spot in enumerate(review_spots):
            status = {
                "seed": "Existing seed",
                "detection": (
                    "Proposed daughter"
                    if spot.preview_id in split_targets
                    else "Proposed"
                ),
                "interpolated": "Interpolated gap",
                "candidate": "Review candidate — not accepted",
            }[spot.kind]
            if spot.kind == "seed":
                quality = "—"
            elif spot.kind == "interpolated":
                quality = "estimated"
            else:
                quality = f"{spot.quality:.3g}"
            values = (
                str(spot.frame),
                f"{spot.x_um:.2f}",
                f"{spot.y_um:.2f}",
                f"{spot.z_um:.2f}",
                quality,
                status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, spot.preview_id)
                if spot.kind == "interpolated":
                    item.setForeground(QColor("#d9a441"))
                elif spot.kind == "candidate":
                    item.setForeground(QColor("#e879f9"))
                elif spot.kind == "seed":
                    item.setForeground(QColor("#9aa0a6"))
                self._table.setItem(row, column, item)

        warnings = tuple(self._proposal.warnings)
        outcome = getattr(self._proposal, "outcome", None)
        self._stop_frame = (
            int(outcome.frame)
            if outcome is not None and getattr(outcome, "code", "completed") != "completed"
            else _first_warning_frame(warnings)
        )
        self._stop_button.setEnabled(self._stop_frame is not None)
        if self._stop_frame is not None:
            self._stop_button.setText(f"Go to stop (t={self._stop_frame})")
        else:
            self._stop_button.setText("Go to stop")
        if outcome is not None and getattr(outcome, "code", "completed") != "completed":
            warning_text = html.escape(_friendly_outcome(outcome))
            guidance = _outcome_guidance(outcome)
            self._warning_label.setText(
                f"<b>Why the run stopped</b><br>{warning_text}"
                f"<br><span style='color:#9aa0a6'>{html.escape(guidance)}</span>"
            )
            self._warning_label.show()
        elif warnings:
            warning_text = "\n".join(
                f"• {html.escape(_friendly_warning(item))}" for item in warnings
            )
            guidance = _warning_guidance(warnings)
            self._warning_label.setText(
                f"<b>Why the run stopped</b><br>{warning_text.replace(chr(10), '<br>')}"
                f"<br><span style='color:#9aa0a6'>{html.escape(guidance)}</span>"
            )
            self._warning_label.show()
        else:
            self._warning_label.setText(
                f"Reached the requested end time, t={self._end_spin.value()}."
            )
            self._warning_label.show()
        if review_spots:
            self._table.selectRow(0)
        self._update_review_navigation_buttons()

    def _parameters_changed(self, *_args) -> None:
        if self._state == self.RUNNING or self._proposal is None:
            return
        self._stop_review_playback()
        self._accept_button.setEnabled(False)
        self._set_state(
            self.OUTDATED,
            "Settings changed. The visible preview uses the previous settings; "
            "click Update Preview before accepting.",
            warning=True,
        )
        self._show_viewer_preview(stale=True)
        changed = self._changed_setting_labels()
        if changed:
            self._generated_label.setText(
                self._generated_html
                + "<br><b>Changed since this draft:</b> "
                + html.escape(", ".join(changed))
            )

    def sync_document_revision(self) -> None:
        """Invalidate a visible draft after any edit, undo, or redo event."""

        if (
            self.app is None
            or self._proposal is None
            or self._accepting
            or self._state in {self.RUNNING, self.STALE}
        ):
            return
        current_revision = self.app.edit_history.revision
        current_counter = getattr(self.app.edit_history, "change_counter", current_revision)
        if (
            current_revision != self._proposal_revision
            or current_counter != self._proposal_change_counter
        ):
            self._accept_button.setEnabled(False)
            self._set_state(
                self.STALE,
                "The dataset changed while this draft was open. Update Preview before accepting.",
                warning=True,
            )
            self._show_viewer_preview(stale=True)

    def _accept_draft(self) -> None:
        if (
            self.app is None
            or self._proposal is None
            or self._expanded_preview is None
            or self._state != self.READY
        ):
            return
        self.sync_document_revision()
        if self._state != self.READY:
            return
        self._accepting = True
        try:
            mapping = self.app.accept_tracking_proposal(
                self._proposal,
                expected_revision=int(self._proposal_revision),
            )
        except Exception as exc:
            logger.exception("Could not accept selected-forward tracking proposal")
            self._accepting = False
            self._show_failure(
                "The draft could not be applied. It remains uncommitted.",
                exc,
            )
            return

        count = self._expanded_preview.proposed_count
        self._accepted = True
        self._navigate_to_accepted_endpoint(mapping)
        self.draftApplied.emit(count)
        self._accepting = False
        self.accept()

    def _navigate_to_accepted_endpoint(self, mapping: Mapping[str, tuple[int, int]]) -> None:
        if self.app is None or self._proposal is None:
            return
        endpoint = _unique_new_terminal(self._proposal)
        if endpoint is None:
            return
        location = mapping.get(endpoint.detection_id)
        if location is None:
            return
        frame, index = location
        try:
            nucleus = self.app.manager.nuclei_record[frame - 1][index - 1]
            self.app.current_time = frame
            config = self.app.manager.config
            if config is not None:
                self.app.current_plane = max(
                    1,
                    round(endpoint.z_um / config.z_res + config.plane_start),
                )
            self.app._set_selection_from_nucleus(frame, nucleus)
            self.app.tracking = True
            self.app.update_display()
        except (AttributeError, IndexError):
            logger.debug("Could not navigate to accepted Auto Forward endpoint")

    def _show_failure(self, message: str, exc: Exception) -> None:
        self._accept_button.setEnabled(False)
        self._set_state(self.FAILED, message, warning=True)
        self._warning_label.setText(
            f"<b>Details</b><br>{html.escape(type(exc).__name__)}: "
            f"{html.escape(str(exc) or 'Unknown error')}"
        )
        self._warning_label.show()

    def _set_running(self, running: bool) -> None:
        if running:
            self._stop_review_playback()
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
        self._preview_button.setEnabled(not running)
        self._reset_button.setEnabled(not running)
        self._discard_button.setEnabled(not running)
        self._accept_button.setEnabled(False if running else self._accept_button.isEnabled())
        self._table.setEnabled(not running)
        self._previous_button.setEnabled(False if running else self._table.currentRow() > 0)
        self._next_button.setEnabled(
            False
            if running
            else 0 <= self._table.currentRow() < self._table.rowCount() - 1
        )
        self._play_button.setEnabled(not running and self._table.rowCount() > 1)
        self._cancel_run_button.setVisible(running)
        self._cancel_run_button.setEnabled(running)
        self._progress_bar.setVisible(running)

    def _cancel_run(self) -> None:
        self._cancel_requested = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_run_button.setEnabled(False)
        self._progress_bar.setFormat("Canceling after the current frame…")
        self._set_state(
            self.RUNNING,
            "Cancel requested. Finishing the current detector step safely…",
            warning=True,
        )

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
        self._end_spin.setValue(max(self._start_time + 1, self._end_time))
        self._channel_spin.setValue(1)
        self._radius_spin.setValue(max(0.05, self._seed_radius_um))
        self._threshold_spin.setValue(5.0)
        self._roi_spin.setValue(max(12.0, self._seed_radius_um * 3.0))
        self._distance_spin.setValue(max(8.0, self._seed_radius_um * 2.0))
        self._gap_spin.setValue(1)
        self._ambiguity_spin.setValue(1.20)
        stop_index = self._branch_policy_combo.findData("stop")
        if stop_index >= 0:
            self._branch_policy_combo.setCurrentIndex(stop_index)
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

    def _set_advanced_visible(self, visible: bool) -> None:
        self._advanced_widget.setVisible(visible)

    def _on_table_current_cell_changed(
        self,
        row: int,
        _column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if not self._navigating_review and row >= 0:
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
        if self._expanded_preview is None or not (0 <= row < self._table.rowCount()):
            return
        item = self._table.item(row, 0)
        if item is None:
            return
        preview_id = str(item.data(Qt.UserRole))
        by_id = dict(self._expanded_preview.by_id)
        by_id.update(
            {
                spot.preview_id: spot
                for spot in getattr(self._expanded_preview, "candidates", ())
            }
        )
        spot = by_id.get(preview_id)
        if spot is None:
            return
        self._navigating_review = True
        try:
            self._table.selectRow(row)
        finally:
            self._navigating_review = False
        self._navigate_to_spot(spot)
        viewer_integration = getattr(self.app, "_viewer_integration", None)
        if viewer_integration is not None:
            viewer_integration.highlight_tracking_preview(preview_id)
            if self._center_check.isChecked():
                center = getattr(viewer_integration, "center_tracking_preview", None)
                if callable(center):
                    center(preview_id)
        self._update_review_navigation_buttons()

    def _navigate_to_spot(self, spot: PreviewSpot) -> None:
        if self.app is None:
            return
        self.app.set_time(spot.frame)
        config = self.app.manager.config
        if config is not None and not bool(getattr(self.app, "_3d_mode", False)):
            plane = round(spot.z_um / config.z_res + config.plane_start)
            self.app.set_plane(plane)

    def _go_to_stop(self) -> None:
        if self.app is None or self._stop_frame is None:
            return
        preview = self._expanded_preview
        if preview is not None:
            review_spots = tuple(
                getattr(
                    preview,
                    "review_spots",
                    (*preview.spots, *getattr(preview, "candidates", ())),
                )
            )
            candidate = next(
                (
                    spot
                    for spot in review_spots
                    if spot.frame == self._stop_frame and spot.kind == "candidate"
                ),
                None,
            )
            if candidate is not None:
                self._navigate_to_spot(candidate)
                viewer_integration = getattr(self.app, "_viewer_integration", None)
                if viewer_integration is not None:
                    viewer_integration.highlight_tracking_preview(candidate.preview_id)
                    center = getattr(viewer_integration, "center_tracking_preview", None)
                    if callable(center):
                        center(candidate.preview_id)
                return
        self.app.set_time(self._stop_frame)
        outcome = getattr(self._proposal, "outcome", None)
        predicted = getattr(outcome, "predicted_position_um", None)
        if predicted is not None:
            self._center_physical_position(*predicted)

    def sync_viewer_position(self) -> None:
        """Keep keyboard/playback time navigation aligned with the review table."""

        if (
            self.app is None
            or self._expanded_preview is None
            or self._navigating_review
            or self._state == self.RUNNING
        ):
            return
        current_row = self._table.currentRow()
        if current_row >= 0:
            item = self._table.item(current_row, 0)
            if item is not None and item.text() == str(self.app.current_time):
                return
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.text() == str(self.app.current_time):
                self._navigating_review = True
                try:
                    self._table.selectRow(row)
                finally:
                    self._navigating_review = False
                preview_id = str(item.data(Qt.UserRole))
                viewer_integration = getattr(self.app, "_viewer_integration", None)
                if viewer_integration is not None:
                    viewer_integration.highlight_tracking_preview(preview_id)
                self._update_review_navigation_buttons()
                return

    def _update_review_navigation_buttons(self) -> None:
        row = self._table.currentRow()
        count = self._table.rowCount()
        self._previous_button.setEnabled(count > 0 and row > 0)
        self._next_button.setEnabled(count > 0 and 0 <= row < count - 1)
        self._play_button.setEnabled(count > 1 and self._state != self.RUNNING)

    def _toggle_review_playback(self, playing: bool) -> None:
        if playing and self._table.rowCount() > 1:
            if self._table.currentRow() >= self._table.rowCount() - 1:
                self._navigate_to_row(0)
            self._play_button.setText("⏸ Pause Draft")
            self._review_timer.start()
        else:
            self._stop_review_playback()

    def _advance_review_playback(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            row = 0
        elif row >= self._table.rowCount() - 1:
            self._stop_review_playback()
            return
        self._navigate_to_row(row + 1)

    def _stop_review_playback(self) -> None:
        self._review_timer.stop()
        self._play_button.blockSignals(True)
        self._play_button.setChecked(False)
        self._play_button.blockSignals(False)
        self._play_button.setText("▶ Play Draft")

    def _show_viewer_preview(self, *, stale: bool) -> None:
        viewer_integration = getattr(self.app, "_viewer_integration", None)
        if viewer_integration is None or self._proposal is None:
            return
        config = self.app.manager.config
        if config is None:
            return
        from ..tracking.api import Calibration

        viewer_integration.show_tracking_preview(
            self._proposal,
            Calibration(config.xy_res, config.z_res, config.plane_start),
            visible=self._overlay_check.isChecked(),
            stale=stale,
        )

    def _set_overlay_visible(self, visible: bool) -> None:
        viewer_integration = getattr(self.app, "_viewer_integration", None)
        if viewer_integration is not None:
            viewer_integration.set_tracking_preview_visible(visible)

    def _set_detection_channel_solo(self, enabled: bool) -> None:
        if self.app is None:
            return
        integration = getattr(self.app, "_viewer_integration", None)
        if enabled:
            if self._solo_channel_visibility is None:
                capture = getattr(integration, "capture_image_channel_visibility", None)
                if callable(capture):
                    self._solo_channel_visibility = capture()
                else:
                    self._solo_channel_visibility = [
                        (layer, bool(getattr(layer, "visible", True)))
                        for layer in getattr(self.app, "_image_layers", ())
                    ]
            selected = self._channel_spin.value() - 1
            solo = getattr(integration, "set_detection_channel_solo", None)
            if callable(solo):
                solo(selected)
            else:
                for index, layer in enumerate(getattr(self.app, "_image_layers", ())):
                    layer.visible = index == selected
        else:
            self._restore_channel_visibility()

    def _refresh_solo_detection_channel(self, _value: int) -> None:
        if self._solo_channel_check.isChecked():
            self._set_detection_channel_solo(True)

    def _restore_channel_visibility(self) -> None:
        if self.app is None or self._solo_channel_visibility is None:
            return
        integration = getattr(self.app, "_viewer_integration", None)
        restore = getattr(integration, "restore_image_channel_visibility", None)
        if callable(restore):
            restore(self._solo_channel_visibility)
        else:
            for layer, visible in self._solo_channel_visibility:
                try:
                    layer.visible = visible
                except RuntimeError:
                    pass
        self._solo_channel_visibility = None

    def _pause_host_playback(self) -> None:
        player = getattr(self.app, "_player_controls", None)
        if player is not None and bool(getattr(player, "_playing", False)):
            stop = getattr(player, "_stop_play", None)
            if callable(stop):
                stop()

    def _center_physical_position(self, x_um: float, y_um: float, z_um: float) -> None:
        if self.app is None:
            return
        viewer_integration = getattr(self.app, "_viewer_integration", None)
        center = getattr(viewer_integration, "center_tracking_position", None)
        if callable(center):
            center(x_um, y_um, z_um)
            return
        config = self.app.manager.config
        camera = getattr(getattr(self.app, "viewer", None), "camera", None)
        if config is None or camera is None:
            return
        x_px = x_um / config.xy_res
        y_px = y_um / config.xy_res
        if bool(getattr(self.app, "_3d_mode", False)):
            camera.center = (z_um / config.z_res, y_px, x_px)
        else:
            camera.center = (y_px, x_px)

    def _changed_setting_labels(self) -> list[str]:
        if self._generated_settings is None:
            return []
        labels = {
            "detector_id": "detector",
            "tracker_id": "tracker",
            "end_time": "end time",
            "channel": "channel",
            "radius_um": "cell size",
            "threshold": "detection threshold",
            "roi_radius_um": "search area",
            "max_distance_um": "maximum movement",
            "missing_frames": "missing frames",
            "ambiguity_ratio": "caution",
            "branch_policy": "division behavior",
            "subpixel": "subpixel refinement",
            "median_filter": "median filter",
            "starrynite_parameter_file": "StarryNite parameter file",
            "starrynite_detector_settings": "StarryNite detector preset",
            "starrynite_tracker_settings": "StarryNite tracker preset",
        }
        current = self.export_settings(revalidate=False)
        return [
            label
            for key, label in labels.items()
            if current.get(key) != self._generated_settings.get(key)
        ]

    def _clear_viewer_preview(self) -> None:
        viewer_integration = getattr(self.app, "_viewer_integration", None)
        if viewer_integration is not None:
            viewer_integration.clear_tracking_preview()

    def _capture_view_state(self) -> dict[str, Any]:
        if self.app is None:
            return {}
        state: dict[str, Any] = {
            "time": self.app.current_time,
            "plane": self.app.current_plane,
            "cell_name": self.app.current_cell_name,
            "selection_anchor": self.app.selection_anchor,
            "tracking": self.app.tracking,
            "3d_mode": bool(getattr(self.app, "_3d_mode", False)),
            "change_counter": getattr(self.app.edit_history, "change_counter", 0),
        }
        camera = getattr(getattr(self.app, "viewer", None), "camera", None)
        if camera is not None:
            state["camera_center"] = tuple(camera.center)
            state["camera_zoom"] = camera.zoom
            if hasattr(camera, "angles"):
                state["camera_angles"] = tuple(camera.angles)
            if hasattr(camera, "perspective"):
                state["camera_perspective"] = camera.perspective
        return state

    def _restore_view_state(self) -> None:
        if self.app is None or not self._original_view:
            return
        try:
            desired_3d = bool(self._original_view.get("3d_mode", False))
            if bool(getattr(self.app, "_3d_mode", False)) != desired_3d:
                set_mode = getattr(self.app, "set_3d_mode", None)
                if callable(set_mode):
                    set_mode(desired_3d)
                else:
                    self.app.toggle_3d()
            self.app.current_time = self._original_view["time"]
            self.app.current_plane = self._original_view["plane"]
            anchor = self._original_view.get("selection_anchor")
            nucleus = None
            if anchor is not None:
                time, index = anchor
                record = self.app.manager.nuclei_record
                if 1 <= time <= len(record) and 1 <= index <= len(record[time - 1]):
                    candidate = record[time - 1][index - 1]
                    if candidate.is_alive:
                        nucleus = candidate
            if nucleus is not None and callable(
                getattr(self.app, "_set_selection_from_nucleus", None)
            ):
                self.app._set_selection_from_nucleus(anchor[0], nucleus)
                self.app.tracking = self._original_view["tracking"]
            elif nucleus is not None:
                self.app.selection_anchor = anchor
                self.app.current_cell_name = (
                    nucleus.effective_name or self._original_view["cell_name"]
                )
                self.app.tracking = self._original_view["tracking"]
            else:
                self.app.current_cell_name = ""
                self.app.selection_anchor = None
                self.app.tracking = False
            self.app.update_display()
            camera = getattr(getattr(self.app, "viewer", None), "camera", None)
            if camera is not None:
                if "camera_center" in self._original_view:
                    camera.center = self._original_view["camera_center"]
                if "camera_zoom" in self._original_view:
                    camera.zoom = self._original_view["camera_zoom"]
                if "camera_angles" in self._original_view and hasattr(camera, "angles"):
                    camera.angles = self._original_view["camera_angles"]
                if "camera_perspective" in self._original_view and hasattr(
                    camera, "perspective"
                ):
                    camera.perspective = self._original_view["camera_perspective"]
        except (AttributeError, RuntimeError):
            logger.debug("Could not fully restore the pre-preview viewer state")

    def _cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._stop_review_playback()
        self._restore_channel_visibility()
        self._clear_viewer_preview()
        if not self._accepted:
            self._restore_view_state()

    def done(self, result: int) -> None:
        if self._analysis_thread is not None:
            self._close_after_run = True
            self._deferred_result = result
            if self._state == self.RUNNING:
                self._cancel_run()
            return
        self._cleanup()
        super().done(result)

    def reject(self) -> None:
        self.done(QDialog.Rejected)

    def closeEvent(self, event) -> None:
        if self._analysis_thread is not None:
            self._close_after_run = True
            self._deferred_result = QDialog.Rejected
            if self._state == self.RUNNING:
                self._cancel_run()
            event.ignore()
            return
        super().closeEvent(event)


def _first_warning_frame(warnings: tuple[str, ...]) -> int | None:
    for warning in warnings:
        match = re.search(r"\bt=(\d+)\b", warning)
        if match:
            return int(match.group(1))
    return None


def _unique_new_terminal(proposal):
    """Return one unambiguous new graph endpoint, or ``None`` for branches."""

    proposed = tuple(
        detection
        for detection in proposal.detections
        if detection.detection_id not in proposal.existing_anchors
    )
    outgoing = {edge.source_id for edge in proposal.edges}
    terminals = tuple(
        detection
        for detection in proposed
        if detection.detection_id not in outgoing
    )
    return terminals[0] if len(terminals) == 1 else None


def _friendly_warning(warning: str) -> str:
    text = warning.strip().rstrip(".")
    text = re.sub(r"^Stopped at t=(\d+):", r"Stopped before t=\1:", text)
    text = text.replace(
        "two candidates had similar assignment costs",
        "two possible nuclei were similarly likely",
    )
    text = text.replace(
        "no unique candidate passed the distance gate",
        "no suitable nucleus was found within the movement limit",
    )
    text = text.replace(
        "a candidate overlaps an existing curated nucleus",
        "the draft reached an existing curated nucleus",
    )
    text = text.replace(
        "two candidates form a probable division; Simple LAP does not create daughter branches",
        "this looks like a division",
    )
    return text + "."


def _warning_guidance(warnings: tuple[str, ...]) -> str:
    joined = " ".join(warnings).lower()
    if "division" in joined:
        return "Accept the continuation, then use Manual Track for each daughter."
    if "existing curated" in joined:
        return "The existing annotation was protected; inspect that frame before continuing."
    if "similar assignment" in joined:
        return "Try a smaller search area, a shorter movement limit, or a higher threshold."
    if "distance gate" in joined or "no unique candidate" in joined:
        return "Try a lower threshold, a larger search area, or a larger movement limit."
    return "Inspect the stopping frame, then adjust the settings or continue manually."


def _friendly_outcome(outcome: object) -> str:
    code = str(getattr(outcome, "code", "stopped"))
    frame = int(getattr(outcome, "frame", 0))
    candidates = len(getattr(outcome, "candidates", ()))
    messages = {
        "ambiguity": (
            f"Stopped before t={frame}: {max(2, candidates)} possible nuclei were "
            "similarly likely. Their square/cross markers are review-only and will not "
            "be accepted."
        ),
        "division": (
            f"Stopped before t={frame}: two candidates look like a division. "
            "They are shown as review-only square/cross markers."
        ),
        "conflict": (
            f"Stopped before t={frame}: the search reached an existing curated nucleus, "
            "which was protected from replacement."
        ),
        "lost": (
            f"Stopped before t={frame}: no suitable nucleus was found inside the "
            "movement limit. The predicted search location remains visible."
        ),
    }
    return messages.get(
        code,
        f"Stopped before t={frame}. Inspect the marked search area before continuing.",
    )


def _outcome_guidance(outcome: object) -> str:
    code = str(getattr(outcome, "code", "stopped"))
    if code == "division":
        return "Accept the reliable prefix, then use Manual Track for each daughter."
    if code == "conflict":
        return "Inspect the curated annotation; it will not be changed by this draft."
    if code == "ambiguity":
        return "Compare both candidates, then reduce the search area or increase caution."
    if code == "lost":
        return "Try a lower detection threshold, a larger search area, or more movement."
    return "Inspect the stop frame, adjust settings, or continue manually."
