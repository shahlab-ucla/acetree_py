"""Modeless, iterative review workbench for selected-cell Auto Forward.

The dialog deliberately keeps analysis separate from acceptance.  Users can
adjust parameters, build a non-destructive draft, inspect it while navigating
the main image viewer, and rerun as often as needed before one explicit commit.
"""

from __future__ import annotations

import html
import logging
import re
from threading import Event
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping

from qtpy.QtCore import QThread, QTimer, Qt, Signal
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
        self._cancel_event: Event | None = None
        self._close_after_run = False
        self._deferred_result = QDialog.Rejected
        self._rerun_after_thread = False
        self._run_started_at = 0.0
        self._generated_settings: dict[str, Any] | None = None
        self._generated_html = ""
        self._generated_duration = 0.0
        self._navigating_review = False
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
        self._detector_combo = QComboBox()
        for descriptor in registry.detector_descriptors():
            self._detector_combo.addItem(descriptor.display_name, descriptor.plugin_id)
        self._detector_combo.setToolTip("Method used to find nucleus-like bright blobs")
        form.addRow("Detector:", self._detector_combo)

        self._tracker_combo = QComboBox()
        for descriptor in registry.tracker_descriptors():
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
        self._detector_combo.currentIndexChanged.connect(self._parameters_changed)
        self._tracker_combo.currentIndexChanged.connect(self._parameters_changed)
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

    @staticmethod
    def _select_combo_value(combo: QComboBox, value: Any) -> None:
        if value is None:
            return
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def export_settings(self) -> dict[str, Any]:
        """Return user choices so reopening Auto Forward preserves refinements."""

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
            "subpixel": self._subpixel_check.isChecked(),
            "median_filter": self._median_check.isChecked(),
            "show_overlay": self._overlay_check.isChecked(),
        }

    def get_request(self, seed_anchor: tuple[int, int] | None = None):
        """Build the immutable selected-forward request represented by the form."""

        from ..tracking.api import ComponentSpec, TrackingRequest, TrackingScope
        from ..tracking.registry import get_default_registry

        anchor = seed_anchor or self._seed_anchor
        if anchor is None:
            raise ValueError("Auto Forward needs a selected seed nucleus")
        registry = get_default_registry()
        detector_id = str(self._detector_combo.currentData())
        tracker_id = str(self._tracker_combo.currentData())
        if not detector_id or detector_id == "None":
            raise ValueError("No compatible detector is installed")
        if not tracker_id or tracker_id == "None":
            raise ValueError("No compatible tracker is installed")

        detector_settings = registry.default_settings(detector_id)
        tracker_settings = registry.default_settings(tracker_id)
        detector_common = {
            "TARGET_CHANNEL": self._channel_spin.value(),
            "RADIUS": self._radius_spin.value(),
            "THRESHOLD": self._threshold_spin.value(),
            "DO_SUBPIXEL_LOCALIZATION": self._subpixel_check.isChecked(),
            "DO_MEDIAN_FILTERING": self._median_check.isChecked(),
        }
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
            ),
        )

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
        thread.finished.connect(self._on_analysis_thread_finished)
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
        if self._cancel_event is not None and self._cancel_event.is_set():
            self._finish_cancelled_analysis()
            return
        try:
            proposal, revision, change_counter = payload  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            self._on_analysis_failed(exc)
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
        if count <= 0:
            self._set_state(
                self.EMPTY,
                "No continuation was found. Try a lower threshold, a larger search area, "
                "or Manual Track.",
            )
            self._accept_button.setEnabled(False)
        elif proposal.warnings:
            self._set_state(
                self.READY,
                "A partial draft is ready. Inspect the stopping point before accepting.",
                warning=True,
            )
            self._accept_button.setEnabled(True)
        else:
            self._set_state(
                self.READY,
                f"Draft ready through t={max(spot.frame for spot in self._expanded_preview.spots)}.",
                success=True,
            )
            self._accept_button.setEnabled(True)

        self._accept_button.setText(f"&Accept {count} Position{'s' if count != 1 else ''}")
        self._play_button.setEnabled(bool(self._expanded_preview.spots))

    def _on_analysis_failed(self, exc: object) -> None:
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

    def _on_analysis_thread_finished(self) -> None:
        self._analysis_thread = None
        self._analysis_worker = None
        self._cancel_event = None
        if self._close_after_run:
            self._close_after_run = False
            result = self._deferred_result
            self._cleanup()
            QDialog.done(self, result)
        elif self._rerun_after_thread:
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
        if proposed:
            actual_start = min(spot.frame for spot in proposed)
            actual_end = max(spot.frame for spot in proposed)
            span = f"t={actual_start}–{actual_end}"
        else:
            span = f"after t={self._start_time}"
        self._summary_label.setText(
            f"<b>{len(proposed)} proposed position(s)</b> · {span} · "
            f"{links} link segment(s) · {gaps} interpolated gap position(s) · "
            f"{len(candidates)} review-only candidate(s)"
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
        for row, spot in enumerate(review_spots):
            status = {
                "seed": "Existing seed",
                "detection": "Proposed",
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
        proposed = [
            detection
            for detection in self._proposal.detections
            if detection.detection_id not in self._proposal.existing_anchors
        ]
        if not proposed:
            return
        endpoint = max(proposed, key=lambda detection: (detection.frame, detection.detection_id))
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
        self._end_spin.setValue(max(self._start_time + 1, self._end_time))
        self._channel_spin.setValue(1)
        self._radius_spin.setValue(max(0.05, self._seed_radius_um))
        self._threshold_spin.setValue(5.0)
        self._roi_spin.setValue(max(12.0, self._seed_radius_um * 3.0))
        self._distance_spin.setValue(max(8.0, self._seed_radius_um * 2.0))
        self._gap_spin.setValue(1)
        self._ambiguity_spin.setValue(1.20)
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
            "subpixel": "subpixel refinement",
            "median_filter": "median filter",
        }
        current = self.export_settings()
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
