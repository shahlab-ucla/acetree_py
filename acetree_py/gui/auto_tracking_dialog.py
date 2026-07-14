"""Modeless, iterative review workbench for selected-cell Auto Forward.

The dialog deliberately keeps analysis separate from acceptance.  Users can
adjust parameters, build a non-destructive draft, inspect it while navigating
the main image viewer, and rerun as often as needed before one explicit commit.
"""

from __future__ import annotations

import html
import logging
import re
from typing import TYPE_CHECKING, Any, Mapping

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QApplication,
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
        self._original_view = self._capture_view_state()

        self.setWindowTitle(f"Auto Forward — {seed_label}")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setMinimumSize(860, 610)
        self.resize(980, 690)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self._build_ui(max(1, num_channels))
        self._apply_initial_settings(initial_settings or {})
        self._connect_parameter_signals()
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
        configure_group.setMinimumWidth(330)
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
        self._preview_button = QPushButton("Build Preview")
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

        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setAccessibleName("Tracking stop explanation")
        self._warning_label.hide()
        review_layout.addWidget(self._warning_label)

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
        self._table.cellClicked.connect(self._on_table_row_clicked)
        self._table.setAccessibleName("Proposed tracking positions")
        review_layout.addWidget(self._table, stretch=1)

        navigation = QHBoxLayout()
        self._previous_button = QPushButton("◀ Previous")
        self._previous_button.clicked.connect(lambda: self._step_review_row(-1))
        self._next_button = QPushButton("Next ▶")
        self._next_button.clicked.connect(lambda: self._step_review_row(1))
        self._stop_button = QPushButton("Go to stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._go_to_stop)
        self._overlay_check = QCheckBox("Show draft on image")
        self._overlay_check.setChecked(True)
        self._overlay_check.toggled.connect(self._set_overlay_visible)
        navigation.addWidget(self._previous_button)
        navigation.addWidget(self._next_button)
        navigation.addWidget(self._stop_button)
        navigation.addStretch()
        navigation.addWidget(self._overlay_check)
        review_layout.addLayout(navigation)
        columns.addWidget(review_group, stretch=1)

        footer = QHBoxLayout()
        self._cancel_run_button = QPushButton("Cancel analysis")
        self._cancel_run_button.clicked.connect(self._cancel_run)
        self._cancel_run_button.hide()
        self._discard_button = QPushButton("Discard Draft")
        self._discard_button.setToolTip("Close without changing the dataset")
        self._discard_button.clicked.connect(self.reject)
        self._accept_button = QPushButton("Accept Draft")
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
        if self.app is None or self._seed_anchor is None:
            self._set_state(self.FAILED, "Auto Forward is not connected to an open dataset.")
            return
        try:
            request = self.get_request()
        except Exception as exc:
            self._show_failure("Check the tracking settings and try again.", exc)
            return

        self._cancel_requested = False
        self._set_running(True)
        self._set_state(self.RUNNING, f"Analyzing {self._seed_label}…")
        self._warning_label.hide()
        self._progress_bar.setRange(0, max(1, request.scope.end_frame - self._start_time))
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Starting…")

        def progress(done: int, total: int, message: str) -> None:
            self._progress_bar.setRange(0, max(1, total))
            self._progress_bar.setValue(done)
            self._progress_bar.setFormat(f"{message}  %p%")
            QApplication.processEvents()

        try:
            proposal, revision = self.app.analyze_tracking_request(
                request,
                progress=progress,
                cancelled=lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                from ..tracking.pipeline import TrackingCancelled

                raise TrackingCancelled("Tracking analysis was cancelled")
        except Exception as exc:
            from ..tracking.pipeline import TrackingCancelled

            if isinstance(exc, TrackingCancelled):
                self._proposal = None
                self._expanded_preview = None
                self._clear_viewer_preview()
                self._set_state(
                    self.CONFIGURING,
                    "Tracking canceled. No changes were made; adjust settings or try again.",
                )
            else:
                logger.exception("Selected-forward tracking failed")
                self._show_failure(
                    "Auto Forward could not build a draft. No changes were made.",
                    exc,
                )
            return
        finally:
            self._set_running(False)

        self._proposal = proposal
        self._expanded_preview = expand_tracking_preview(proposal)
        self._proposal_revision = revision
        self._proposal_change_counter = getattr(self.app.edit_history, "change_counter", revision)
        self._populate_review()
        self._show_viewer_preview(stale=False)
        self._preview_button.setText("Update Preview")

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

    def _populate_review(self) -> None:
        assert self._proposal is not None
        assert self._expanded_preview is not None
        preview = self._expanded_preview
        proposed = [spot for spot in preview.spots if spot.kind != "seed"]
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
            f"{links} link segment(s) · {gaps} interpolated gap position(s)"
        )

        self._table.setRowCount(len(preview.spots))
        for row, spot in enumerate(preview.spots):
            status = {
                "seed": "Existing seed",
                "detection": "Proposed",
                "interpolated": "Interpolated gap",
            }[spot.kind]
            values = (
                str(spot.frame),
                f"{spot.x_um:.2f}",
                f"{spot.y_um:.2f}",
                f"{spot.z_um:.2f}",
                f"{spot.quality:.3g}",
                status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, spot.preview_id)
                if spot.kind == "interpolated":
                    item.setForeground(QColor("#d9a441"))
                elif spot.kind == "seed":
                    item.setForeground(QColor("#9aa0a6"))
                self._table.setItem(row, column, item)

        warnings = tuple(self._proposal.warnings)
        self._stop_frame = _first_warning_frame(warnings)
        self._stop_button.setEnabled(self._stop_frame is not None)
        if self._stop_frame is not None:
            self._stop_button.setText(f"Go to stop (t={self._stop_frame})")
        else:
            self._stop_button.setText("Go to stop")
        if warnings:
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
        self._previous_button.setEnabled(bool(preview.spots))
        self._next_button.setEnabled(bool(preview.spots))

    def _parameters_changed(self, *_args) -> None:
        if self._state == self.RUNNING or self._proposal is None:
            return
        self._accept_button.setEnabled(False)
        self._set_state(
            self.OUTDATED,
            "Settings changed. The visible preview uses the previous settings; "
            "click Update Preview before accepting.",
            warning=True,
        )
        self._show_viewer_preview(stale=True)

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
        self._settings_widget.setEnabled(not running)
        self._advanced_toggle.setEnabled(not running)
        self._advanced_widget.setEnabled(not running)
        self._preview_button.setEnabled(not running)
        self._reset_button.setEnabled(not running)
        self._discard_button.setEnabled(not running)
        self._accept_button.setEnabled(False if running else self._accept_button.isEnabled())
        self._cancel_run_button.setVisible(running)
        self._cancel_run_button.setEnabled(running)
        self._progress_bar.setVisible(running)

    def _cancel_run(self) -> None:
        self._cancel_requested = True
        self._cancel_run_button.setEnabled(False)
        self._progress_bar.setFormat("Canceling after the current frame…")

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
        if self._expanded_preview is None or not (0 <= row < self._table.rowCount()):
            return
        item = self._table.item(row, 0)
        preview_id = str(item.data(Qt.UserRole))
        by_id = self._expanded_preview.by_id
        spot = by_id.get(preview_id)
        if spot is None:
            return
        self._table.selectRow(row)
        self._navigate_to_spot(spot)
        viewer_integration = getattr(self.app, "_viewer_integration", None)
        if viewer_integration is not None:
            viewer_integration.highlight_tracking_preview(preview_id)

    def _navigate_to_spot(self, spot: PreviewSpot) -> None:
        if self.app is None:
            return
        self.app.set_time(spot.frame)
        config = self.app.manager.config
        if config is not None:
            plane = round(spot.z_um / config.z_res + config.plane_start)
            self.app.set_plane(plane)

    def _go_to_stop(self) -> None:
        if self.app is not None and self._stop_frame is not None:
            self.app.set_time(self._stop_frame)

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
        }
        camera = getattr(getattr(self.app, "viewer", None), "camera", None)
        if camera is not None:
            state["camera_center"] = tuple(camera.center)
            state["camera_zoom"] = camera.zoom
        return state

    def _restore_view_state(self) -> None:
        if self.app is None or not self._original_view:
            return
        try:
            self.app.current_cell_name = self._original_view["cell_name"]
            self.app.selection_anchor = self._original_view["selection_anchor"]
            self.app.current_time = self._original_view["time"]
            self.app.current_plane = self._original_view["plane"]
            self.app.tracking = self._original_view["tracking"]
            camera = getattr(getattr(self.app, "viewer", None), "camera", None)
            if camera is not None:
                if "camera_center" in self._original_view:
                    camera.center = self._original_view["camera_center"]
                if "camera_zoom" in self._original_view:
                    camera.zoom = self._original_view["camera_zoom"]
            self.app.update_display()
        except (AttributeError, RuntimeError):
            logger.debug("Could not fully restore the pre-preview viewer state")

    def _cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._clear_viewer_preview()
        if not self._accepted:
            self._restore_view_state()

    def done(self, result: int) -> None:
        self._cleanup()
        super().done(result)


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
