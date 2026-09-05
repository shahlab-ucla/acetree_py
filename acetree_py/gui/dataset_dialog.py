"""Dataset creation dialog — create a new AceTree dataset from raw images.

Provides a wizard-style dialog that walks the user through:
1. Selecting an image directory
2. Configuring image format (single channel, side-by-side, separate dirs, multichannel stack)
3. Setting voxel sizes and reviewing auto-detected parameters
4. Choosing manual annotation or an initial automated tracking draft
5. Choosing an output directory for the nuclei ZIP and config XML
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..io.config import AceTreeConfig, NamingMethod

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QDialog = object  # type: ignore[misc,assignment]

class DatasetCreationDialog(QDialog):  # type: ignore[misc]
    """Multi-page wizard for creating a new AceTree dataset from images."""

    def __init__(self, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")
        super().__init__(parent)
        self.setWindowTitle("Create New Dataset")
        self.setMinimumWidth(550)
        self.setMinimumHeight(450)

        self._detected: dict = {}  # auto-detection results

        layout = QVBoxLayout(self)

        # Stacked pages
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._page1 = self._build_page1_directory()
        self._page2 = self._build_page2_format()
        self._page3 = self._build_page3_parameters()
        self._page4 = self._build_page4_tracking()
        self._page5 = self._build_page5_output()

        self._stack.addWidget(self._page1)
        self._stack.addWidget(self._page2)
        self._stack.addWidget(self._page3)
        self._stack.addWidget(self._page4)
        self._stack.addWidget(self._page5)
        self._stack.currentChanged.connect(self._update_nav_buttons)

        # Navigation buttons
        nav = QHBoxLayout()
        self._btn_back = QPushButton("Back")
        self._btn_back.clicked.connect(self._go_back)
        self._btn_next = QPushButton("Next")
        self._btn_next.clicked.connect(self._go_next)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)

        nav.addWidget(self._btn_back)
        nav.addStretch()
        nav.addWidget(self._btn_cancel)
        nav.addWidget(self._btn_next)
        layout.addLayout(nav)

        self._connect_tracking_channel_controls()
        self._sync_tracking_channel_range()
        self._update_nav_buttons()

    # ── Page 1: Image directory ───────────────────────────────────

    def _build_page1_directory(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>Step 1: Select Image Directory</b>"))
        layout.addWidget(QLabel("Choose the directory containing your TIFF image files."))

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Path to image directory...")
        self._dir_edit.setReadOnly(True)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_directory)
        dir_row.addWidget(self._dir_edit)
        dir_row.addWidget(btn_browse)
        layout.addLayout(dir_row)

        self._detect_label = QTextEdit()
        self._detect_label.setReadOnly(True)
        self._detect_label.setMaximumHeight(200)
        layout.addWidget(QLabel("Auto-detection results:"))
        layout.addWidget(self._detect_label)
        self._image_validation_label = QLabel()
        self._image_validation_label.setWordWrap(True)
        self._image_validation_label.setAccessibleName("Image directory problem")
        self._image_validation_label.setStyleSheet("QLabel { color: #a85f00; }")
        layout.addWidget(self._image_validation_label)
        layout.addStretch()
        return page

    def _browse_directory(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if d:
            self._dir_edit.setText(d)
            self._run_auto_detect(Path(d))

    def _run_auto_detect(self, directory: Path) -> None:
        """Probe the directory to guess format, timepoints, planes."""
        self._detected = _auto_detect_format(directory)
        d = self._detected

        lines = []
        lines.append(f"Directory: {directory}")
        lines.append(f"TIFF files found: {d.get('num_files', 0)}")
        if d.get("pattern"):
            lines.append(f"Naming pattern: {d['pattern']}")
        if d.get("prefix"):
            lines.append(f"Prefix: {d['prefix']}")
        lines.append(f"Timepoints detected: {d.get('num_timepoints', '?')}")
        lines.append(f"Z-planes per stack: {d.get('num_planes', '?')}")
        if d.get("image_shape"):
            h, w = d["image_shape"]
            lines.append(f"Image size: {w} x {h}")
            if w > h * 1.8:
                lines.append("  (wide image — may be side-by-side dual channel)")
        if d.get("per_plane"):
            lines.append("Format: per-plane TIFFs (filename contains -p)")
        else:
            lines.append("Format: multi-page TIFF stacks")
        if d.get("error"):
            lines.append(f"Warning: {d['error']}")

        self._detect_label.setPlainText("\n".join(lines))

        # Pre-fill downstream pages
        if d.get("num_timepoints"):
            self._timepoints_spin.setValue(d["num_timepoints"])
        if d.get("num_planes"):
            # Respect an already-selected multichannel layout when the user
            # goes back and chooses a different image directory.  Writing the
            # raw TIFF page count directly would save Z*C as the Z count.
            self._recompute_planes()
        self._refresh_tracking_validation()

    # ── Page 2: Image format ──────────────────────────────────────

    def _build_page2_format(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>Step 2: Image Format</b>"))

        fmt_group = QGroupBox("Channel Layout")
        fmt_layout = QVBoxLayout(fmt_group)

        self._radio_single = QRadioButton("Single channel")
        self._radio_single.setChecked(True)
        self._radio_split = QRadioButton("Side-by-side dual channel (left/right halves)")
        self._radio_separate = QRadioButton("Separate directory per channel")
        self._radio_multistack = QRadioButton("Multichannel TIFF stack")

        fmt_layout.addWidget(self._radio_single)
        fmt_layout.addWidget(self._radio_split)
        fmt_layout.addWidget(self._radio_separate)
        fmt_layout.addWidget(self._radio_multistack)

        layout.addWidget(fmt_group)

        # Separate dirs config (shown when radio_separate selected)
        self._sep_group = QGroupBox("Channel Directories")
        sep_layout = QFormLayout(self._sep_group)
        self._ch2_dir_edit = QLineEdit()
        self._ch2_dir_edit.setPlaceholderText("Path to second channel directory...")
        btn_ch2 = QPushButton("Browse...")
        btn_ch2.clicked.connect(self._browse_ch2_dir)
        ch2_row = QHBoxLayout()
        ch2_row.addWidget(self._ch2_dir_edit)
        ch2_row.addWidget(btn_ch2)
        sep_layout.addRow("Channel 2:", ch2_row)
        self._sep_group.setVisible(False)
        layout.addWidget(self._sep_group)

        # Multichannel stack ordering
        self._stack_group = QGroupBox("Stack Ordering")
        stack_layout = QFormLayout(self._stack_group)
        self._n_channels_spin = QSpinBox()
        self._n_channels_spin.setRange(2, 8)
        self._n_channels_spin.setValue(2)
        stack_layout.addRow("Number of channels:", self._n_channels_spin)
        self._ordering_combo = QComboBox()
        self._ordering_combo.addItems([
            "Interleaved (Z1C1, Z1C2, Z2C1, ...)",
            "Planar (all Z for C1, then all Z for C2)",
        ])
        stack_layout.addRow("Page order:", self._ordering_combo)
        self._stack_group.setVisible(False)
        layout.addWidget(self._stack_group)

        # Flip checkbox
        self._flip_check = QCheckBox("Flip left/right (mirror horizontally)")
        layout.addWidget(self._flip_check)

        self._layout_validation_label = QLabel()
        self._layout_validation_label.setWordWrap(True)
        self._layout_validation_label.setAccessibleName("Image layout problem")
        self._layout_validation_label.setStyleSheet("QLabel { color: #a85f00; }")
        layout.addWidget(self._layout_validation_label)

        # Toggle visibility of sub-groups
        self._radio_separate.toggled.connect(self._sep_group.setVisible)
        self._radio_multistack.toggled.connect(self._stack_group.setVisible)

        # When the multistack config changes, recompute the z-plane count
        # from the probed page count and the user-selected channel count.
        self._radio_multistack.toggled.connect(self._recompute_planes)
        self._n_channels_spin.valueChanged.connect(self._recompute_planes)

        layout.addStretch()
        return page

    def _recompute_planes(self) -> None:
        """Adjust the z-plane spinbox based on multistack channel count.

        Auto-detection stores the raw TIFF page count in ``self._detected``;
        when interleaved multichannel is selected, the true Z count is
        ``pages / num_channels``.
        """
        d = self._detected
        raw_pages = d.get("num_planes")
        if raw_pages is None:
            return
        if self._radio_multistack.isChecked():
            n_ch = self._n_channels_spin.value()
            if n_ch > 1:
                planes = max(1, raw_pages // n_ch)
            else:
                planes = raw_pages
        else:
            planes = raw_pages
        # Avoid recursion if the user has already edited the value away
        self._planes_spin.setValue(planes)

    def _browse_ch2_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Channel 2 Directory")
        if d:
            self._ch2_dir_edit.setText(d)

    def _connect_tracking_channel_controls(self) -> None:
        """Keep the tracking channel choices synchronized with image layout."""

        for radio in (
            self._radio_single,
            self._radio_split,
            self._radio_separate,
            self._radio_multistack,
        ):
            radio.toggled.connect(self._sync_tracking_channel_range)
        self._n_channels_spin.valueChanged.connect(self._sync_tracking_channel_range)
        self._ch2_dir_edit.textChanged.connect(self._refresh_tracking_validation)
        self._dir_edit.textChanged.connect(self._refresh_tracking_validation)
        self._output_edit.textChanged.connect(self._refresh_tracking_validation)
        self._dataset_name_edit.textChanged.connect(self._refresh_tracking_validation)
        self._radio_tracking_auto.toggled.connect(self._refresh_tracking_validation)
        self._tracking_workflow_combo.currentIndexChanged.connect(
            self._tracking_workflow_changed
        )
        self._tracking_starrynite_preset_combo.currentIndexChanged.connect(
            self._tracking_workflow_changed
        )
        self._tracking_channel_spin.valueChanged.connect(
            self._refresh_tracking_validation
        )
        self._tracking_detector_combo.currentIndexChanged.connect(
            self._refresh_tracking_validation
        )
        self._tracking_tracker_combo.currentIndexChanged.connect(
            self._tracking_tracker_changed
        )
        self._tracking_division_check.toggled.connect(
            self._refresh_tracking_validation
        )

    def _available_tracking_channels(self) -> int:
        if self._radio_multistack.isChecked():
            return self._n_channels_spin.value()
        if self._radio_split.isChecked() or self._radio_separate.isChecked():
            return 2
        return 1

    def _sync_tracking_channel_range(self, *_args) -> None:
        """Clamp the detector channel immediately after a layout change."""

        available = self._available_tracking_channels()
        self._tracking_channel_spin.setRange(1, available)
        self._tracking_channel_spin.setToolTip(
            f"Available channels for the selected image layout: 1–{available}"
        )
        self._tracking_channel_spin.setAccessibleDescription(
            f"The selected image layout provides {available} channel(s)"
        )
        self._refresh_tracking_validation()

    def _tracking_tracker_changed(self, *_args) -> None:
        self._sync_tracking_division_capability(use_default=True)
        if hasattr(self, "_output_edit"):
            self._refresh_tracking_validation()

    def _tracking_workflow_changed(self, *_args) -> None:
        """Apply one understandable workflow to the hidden component choices."""

        from ..tracking.workflows import tracking_workflow

        workflow_id = self._tracking_workflow_combo.currentData()
        if workflow_id is None:
            return
        workflow = tracking_workflow(str(workflow_id))
        detector_index = self._tracking_detector_combo.findData(workflow.detector_id)
        tracker_index = self._tracking_tracker_combo.findData(workflow.tracker_id)
        if detector_index >= 0:
            self._tracking_detector_combo.setCurrentIndex(detector_index)
        if tracker_index >= 0:
            self._tracking_tracker_combo.setCurrentIndex(tracker_index)
        self._tracking_workflow_description.setText(workflow.description)
        uses_starrynite = workflow.workflow_id == "modern_starrynite"
        self._tracking_starrynite_preset_combo.setVisible(uses_starrynite)
        self._tracking_starrynite_preset_label.setVisible(uses_starrynite)
        if uses_starrynite:
            try:
                from ..tracking.starrynite import (
                    bundled_parameter_preset,
                    load_tuning_profile,
                )

                preset = bundled_parameter_preset(
                    str(self._tracking_starrynite_preset_combo.currentData())
                )
                profile = load_tuning_profile(preset.parameter_file)
                self._tracking_radius_spin.setValue(
                    float(profile.detector_settings.get("RADIUS", 4.0))
                )
                self._tracking_threshold_spin.setValue(
                    float(profile.detector_settings.get("INTENSITY_THRESHOLD", 5.0))
                )
            except (KeyError, OSError, ValueError):
                pass
        self._sync_tracking_division_capability(use_default=True)
        if hasattr(self, "_output_edit"):
            self._refresh_tracking_validation()

    def _sync_tracking_division_capability(
        self,
        *,
        use_default: bool = True,
    ) -> None:
        """Match the division choice to the selected tracker's contract."""

        from ..tracking.registry import get_default_registry

        tracker_id = self._tracking_tracker_combo.currentData()
        capable = False
        default = False
        display_name = self._tracking_tracker_combo.currentText() or "Selected tracker"
        if tracker_id is not None:
            try:
                registry = get_default_registry()
                descriptor = registry.get_descriptor(str(tracker_id))
                schema = descriptor.settings_schema
                capabilities = {
                    str(capability).strip().lower()
                    for capability in descriptor.capabilities
                }
                capable = (
                    "splitting" in capabilities
                    and "ALLOW_TRACK_SPLITTING" in schema
                )
                default = bool(
                    registry.default_settings(str(tracker_id)).get(
                        "ALLOW_TRACK_SPLITTING",
                        False,
                    )
                )
            except (KeyError, ValueError):
                capable = False

        self._tracking_division_check.setEnabled(capable)
        if not capable:
            self._tracking_division_check.setChecked(False)
        elif use_default:
            self._tracking_division_check.setChecked(default)
        self._tracking_division_check.setToolTip(
            "Include proposed two-daughter branches in the uncommitted draft. "
            "Every division must still be reviewed before acceptance."
            if capable
            else "The selected tracker does not support two-daughter divisions."
        )
        if capable:
            self._tracking_capability_label.setText(
                f"{display_name} can propose two-daughter divisions. Keep the "
                "division option on to include them in the uncommitted review draft; "
                "turn it off for continuation-only tracking. Merges are never enabled."
            )
        else:
            self._tracking_capability_label.setText(
                f"{display_name} links continuations and short gaps but does not "
                "propose divisions or merges. Choose a division-aware tracker to "
                "include reviewed two-daughter branches in the draft."
            )

    def _tracking_validation_error(self) -> str:
        layout_error = self._image_layout_validation_error()
        if layout_error:
            return layout_error
        if not self._radio_tracking_auto.isChecked():
            return ""
        if self._tracking_detector_combo.count() == 0:
            return "No compatible detector is installed; choose Manual annotation."
        if self._tracking_tracker_combo.count() == 0:
            return "No compatible tracker is installed; choose Manual annotation."
        if (
            self._tracking_division_check.isChecked()
            and not self._tracking_division_check.isEnabled()
        ):
            return "The selected tracker cannot propose divisions; turn divisions off."

        available = self._available_tracking_channels()
        channel = self._tracking_channel_spin.value()
        if not 1 <= channel <= available:
            return (
                f"Detection channel {channel} is unavailable for this layout; "
                f"choose a channel from 1 to {available}."
            )
        return ""

    def _image_source_validation_error(self) -> str:
        """Return a blocking problem with the primary image directory."""

        text = self._dir_edit.text().strip()
        if not text:
            return "Choose an image directory containing TIFF files."
        directory = Path(text)
        try:
            if not directory.is_dir():
                return "The selected image directory does not exist."
        except OSError as exc:
            return f"The selected image directory cannot be read: {exc}"
        if not _tiff_files(directory):
            return "The selected image directory contains no TIFF files."
        error = self._detected.get("error")
        if error:
            return f"The image source could not be validated: {error}"
        return ""

    def _image_layout_validation_error(self) -> str:
        """Validate the selected channel layout independent of tracker mode."""

        if self._radio_separate.isChecked():
            text = self._ch2_dir_edit.text().strip()
            if not text:
                return "Choose the Channel 2 directory for the separate-channel layout."
            directory = Path(text)
            try:
                if not directory.is_dir():
                    return "The Channel 2 directory does not exist."
            except OSError as exc:
                return f"The Channel 2 directory cannot be read: {exc}"
            if not _tiff_files(directory):
                return "The Channel 2 directory contains no TIFF files."

        if self._radio_multistack.isChecked():
            raw_pages = self._detected.get("num_planes")
            channels = self._n_channels_spin.value()
            if raw_pages and raw_pages % channels:
                return (
                    f"The detected stack has {raw_pages} pages, which cannot be "
                    f"divided evenly across {channels} channels."
                )
        return ""

    def _output_validation_error(self) -> str:
        """Return a blocking output-path or dataset-name problem."""

        text = self._output_edit.text().strip()
        if not text:
            return "Choose an output directory for the dataset files."
        output = Path(text)
        try:
            if output.exists() and not output.is_dir():
                return "The output location is a file, not a directory."
            ancestor = output
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            if not ancestor.is_dir():
                return "The output directory has no usable parent directory."
        except OSError as exc:
            return f"The output directory is not usable: {exc}"

        name = self._dataset_name_edit.text().strip() or "dataset"
        if name in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', name):
            return "Use a dataset name without path separators or reserved filename characters."
        if name.endswith((" ", ".")):
            return "The dataset name must not end with a space or period."
        return ""

    def _refresh_tracking_validation(self, *_args) -> None:
        error = self._tracking_validation_error()
        self._tracking_validation_label.setText(error)
        self._tracking_validation_label.setVisible(bool(error))
        image_error = self._image_source_validation_error()
        self._image_validation_label.setText(image_error)
        self._image_validation_label.setVisible(bool(image_error))
        layout_error = self._image_layout_validation_error()
        self._layout_validation_label.setText(layout_error)
        self._layout_validation_label.setVisible(bool(layout_error))
        output_error = self._output_validation_error()
        self._output_validation_label.setText(output_error)
        self._output_validation_label.setVisible(bool(output_error))
        if hasattr(self, "_btn_next"):
            self._update_nav_buttons()

    # ── Page 3: Parameters ────────────────────────────────────────

    def _build_page3_parameters(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>Step 3: Dataset Parameters</b>"))

        form = QFormLayout()

        self._xy_res_spin = QDoubleSpinBox()
        self._xy_res_spin.setRange(0.001, 100.0)
        self._xy_res_spin.setDecimals(4)
        self._xy_res_spin.setValue(0.09)
        self._xy_res_spin.setSuffix(" \u00b5m")
        form.addRow("XY resolution:", self._xy_res_spin)

        self._z_res_spin = QDoubleSpinBox()
        self._z_res_spin.setRange(0.001, 100.0)
        self._z_res_spin.setDecimals(4)
        self._z_res_spin.setValue(1.0)
        self._z_res_spin.setSuffix(" \u00b5m")
        form.addRow("Z resolution:", self._z_res_spin)

        self._timepoints_spin = QSpinBox()
        self._timepoints_spin.setRange(1, 99999)
        self._timepoints_spin.setValue(100)
        form.addRow("Number of timepoints:", self._timepoints_spin)

        self._planes_spin = QSpinBox()
        self._planes_spin.setRange(1, 999)
        self._planes_spin.setValue(30)
        form.addRow("Number of z-planes:", self._planes_spin)

        layout.addLayout(form)
        layout.addStretch()
        return page

    # ── Page 4: Output ────────────────────────────────────────────

    def _build_page4_tracking(self) -> QWidget:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page.setAccessibleName("Initial tracking setup")
        content = QWidget()
        layout = QVBoxLayout(content)
        page.setWidget(content)
        layout.addWidget(QLabel("<b>Step 4: Initial Tracking</b>"))
        self._tracking_explanation_label = QLabel(
            "Choose manual annotation or prepare a reviewed whole-movie draft. After "
            "this dataset or an existing XML dataset opens, use the Tracking menu or "
            "the Edit & Tracking panel for Manual Track, Track Selected Cell, and Track "
            "Whole Movie. Automated results are never accepted automatically."
        )
        self._tracking_explanation_label.setWordWrap(True)
        self._tracking_explanation_label.setAccessibleName(
            "Initial tracking review explanation"
        )
        layout.addWidget(self._tracking_explanation_label)

        mode_group = QGroupBox("Starting workflow")
        mode_layout = QVBoxLayout(mode_group)
        self._radio_tracking_manual = QRadioButton(
            "Manual annotation + optional selected-cell forward tracking"
        )
        self._radio_tracking_manual.setChecked(True)
        self._radio_tracking_auto = QRadioButton(
            "Open a reviewed whole-movie tracking draft"
        )
        mode_layout.addWidget(self._radio_tracking_manual)
        mode_layout.addWidget(self._radio_tracking_auto)
        layout.addWidget(mode_group)

        self._tracking_settings_group = QGroupBox("Draft settings")
        settings = QFormLayout(self._tracking_settings_group)

        from ..tracking.registry import get_default_registry
        from ..tracking.workflows import INITIAL_TRACKING_WORKFLOWS

        registry = get_default_registry()
        self._tracking_workflow_combo = QComboBox()
        for workflow in INITIAL_TRACKING_WORKFLOWS:
            self._tracking_workflow_combo.addItem(
                workflow.display_name, workflow.workflow_id
            )
        self._tracking_workflow_combo.setToolTip(
            "Modern StarryNite is division-aware; LoG/DoG with LAP provide simpler "
            "general-purpose whole-movie alternatives."
        )
        settings.addRow("Tracking method:", self._tracking_workflow_combo)

        self._tracking_workflow_description = QLabel()
        self._tracking_workflow_description.setWordWrap(True)
        settings.addRow("", self._tracking_workflow_description)

        from ..tracking.starrynite import bundled_parameter_presets

        self._tracking_starrynite_preset_combo = QComboBox()
        for preset in bundled_parameter_presets():
            self._tracking_starrynite_preset_combo.addItem(
                preset.display_name, preset.preset_id
            )
            index = self._tracking_starrynite_preset_combo.count() - 1
            self._tracking_starrynite_preset_combo.setItemData(
                index, preset.description, Qt.ToolTipRole
            )
        settings.addRow("Imaging preset:", self._tracking_starrynite_preset_combo)
        self._tracking_starrynite_preset_label = settings.labelForField(
            self._tracking_starrynite_preset_combo
        )

        self._tracking_detector_combo = QComboBox()
        for descriptor in registry.detector_descriptors():
            self._tracking_detector_combo.addItem(
                descriptor.display_name,
                descriptor.plugin_id,
            )
        settings.addRow("Detector:", self._tracking_detector_combo)
        self._tracking_detector_combo.hide()
        settings.labelForField(self._tracking_detector_combo).hide()

        self._tracking_tracker_combo = QComboBox()
        for descriptor in registry.tracker_descriptors():
            if "global_only" in descriptor.capabilities:
                continue
            self._tracking_tracker_combo.addItem(
                descriptor.display_name,
                descriptor.plugin_id,
            )
        settings.addRow("Tracker:", self._tracking_tracker_combo)
        self._tracking_tracker_combo.hide()
        settings.labelForField(self._tracking_tracker_combo).hide()

        self._tracking_channel_spin = QSpinBox()
        self._tracking_channel_spin.setRange(1, 8)
        self._tracking_channel_spin.setValue(1)
        self._tracking_channel_spin.setToolTip(
            "One-based image channel, matching TrackMate"
        )
        settings.addRow("Detection channel:", self._tracking_channel_spin)

        self._tracking_radius_spin = QDoubleSpinBox()
        self._tracking_radius_spin.setRange(0.05, 100.0)
        self._tracking_radius_spin.setDecimals(2)
        self._tracking_radius_spin.setValue(4.0)
        self._tracking_radius_spin.setSuffix(" µm")
        self._tracking_radius_spin.setToolTip(
            "Approximate nucleus radius in physical units"
        )
        settings.addRow("Expected radius:", self._tracking_radius_spin)

        self._tracking_threshold_spin = QDoubleSpinBox()
        self._tracking_threshold_spin.setRange(0.0, 1_000_000.0)
        self._tracking_threshold_spin.setDecimals(4)
        self._tracking_threshold_spin.setValue(5.0)
        self._tracking_threshold_spin.setToolTip(
            "Minimum scale-space response (in image-intensity units)"
        )
        settings.addRow("Quality threshold:", self._tracking_threshold_spin)

        self._tracking_link_distance_spin = QDoubleSpinBox()
        self._tracking_link_distance_spin.setRange(0.05, 1_000.0)
        self._tracking_link_distance_spin.setDecimals(2)
        self._tracking_link_distance_spin.setValue(8.0)
        self._tracking_link_distance_spin.setSuffix(" µm")
        settings.addRow("Maximum displacement:", self._tracking_link_distance_spin)

        self._tracking_gap_spin = QSpinBox()
        self._tracking_gap_spin.setRange(0, 20)
        self._tracking_gap_spin.setValue(1)
        self._tracking_gap_spin.setToolTip(
            "Maximum number of missing frames bridged by a draft link"
        )
        settings.addRow("Missing frames allowed:", self._tracking_gap_spin)

        self._tracking_division_check = QCheckBox(
            "Propose two-daughter divisions for review"
        )
        self._tracking_division_check.setChecked(False)
        self._tracking_division_check.setAccessibleName(
            "Propose divisions in the initial tracking draft"
        )
        settings.addRow("Division handling:", self._tracking_division_check)

        self._tracking_settings_group.setEnabled(False)
        self._radio_tracking_auto.toggled.connect(
            self._tracking_settings_group.setEnabled
        )
        layout.addWidget(self._tracking_settings_group)

        self._tracking_capability_label = QLabel(
            "Modern StarryNite can propose reviewed two-daughter divisions. LoG + LAP "
            "and DoG + LAP are simpler one-to-one alternatives. Advanced custom and "
            "source-bound legacy exact replay are available in Track Whole Movie after "
            "the empty dataset opens."
        )
        self._tracking_capability_label.setWordWrap(True)
        self._tracking_capability_label.setAccessibleName(
            "Selected tracker capabilities"
        )
        layout.addWidget(self._tracking_capability_label)

        self._tracking_validation_label = QLabel()
        self._tracking_validation_label.setWordWrap(True)
        self._tracking_validation_label.setAccessibleName(
            "Initial tracking settings problem"
        )
        self._tracking_validation_label.setStyleSheet("QLabel { color: #a85f00; }")
        self._tracking_validation_label.hide()
        layout.addWidget(self._tracking_validation_label)
        layout.addStretch()
        self._tracking_workflow_changed()
        return page

    def _build_page5_output(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>Step 5: Output Location</b>"))
        layout.addWidget(
            QLabel("Choose where to save the dataset files (nuclei ZIP + config XML).")
        )

        dir_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Output directory...")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_output)
        dir_row.addWidget(self._output_edit)
        dir_row.addWidget(btn_browse)
        layout.addLayout(dir_row)

        form = QFormLayout()
        self._dataset_name_edit = QLineEdit()
        self._dataset_name_edit.setText("dataset")
        self._dataset_name_edit.setPlaceholderText("Dataset name (used for filenames)")
        form.addRow("Dataset name:", self._dataset_name_edit)
        layout.addLayout(form)

        # Summary
        self._summary_label = QTextEdit()
        self._summary_label.setReadOnly(True)
        self._summary_label.setMaximumHeight(150)
        layout.addWidget(QLabel("Summary:"))
        layout.addWidget(self._summary_label)

        self._output_validation_label = QLabel()
        self._output_validation_label.setWordWrap(True)
        self._output_validation_label.setAccessibleName("Dataset output problem")
        self._output_validation_label.setStyleSheet("QLabel { color: #a85f00; }")
        layout.addWidget(self._output_validation_label)

        layout.addStretch()
        return page

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_edit.setText(d)

    # ── Navigation ────────────────────────────────────────────────

    def _go_back(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
        self._update_nav_buttons()

    def _go_next(self) -> None:
        idx = self._stack.currentIndex()
        page = self._stack.widget(idx)
        error = self._page_validation_error(page)
        if error:
            self._show_page_validation_error(page, error)
            self._update_nav_buttons()
            return
        if idx < self._stack.count() - 1:
            self._stack.setCurrentIndex(idx + 1)
            if idx + 1 == self._stack.count() - 1:
                self._update_summary()
        else:
            # Last page — "Create" pressed
            if not self._confirm_overwrite():
                return
            self.accept()
        self._update_nav_buttons()

    def _update_nav_buttons(self, *_args) -> None:
        idx = self._stack.currentIndex()
        self._btn_back.setEnabled(idx > 0)
        is_last = idx == self._stack.count() - 1
        self._btn_next.setText("Create" if is_last else "Next")
        self._btn_next.setEnabled(not bool(self._page_validation_error(self._stack.widget(idx))))

    def _page_validation_error(self, page: QWidget) -> str:
        if page is self._page1:
            return self._image_source_validation_error()
        if page is self._page2:
            return self._image_layout_validation_error()
        if page is self._page4:
            return self._tracking_validation_error()
        if page is self._page5:
            return self._output_validation_error()
        return ""

    def _show_page_validation_error(self, page: QWidget, error: str) -> None:
        if page is self._page1:
            label = self._image_validation_label
        elif page is self._page2:
            label = self._layout_validation_label
        elif page is self._page4:
            label = self._tracking_validation_label
        else:
            label = self._output_validation_label
        label.setText(error)
        label.show()

    def _confirm_overwrite(self) -> bool:
        """Require an explicit opt-in before replacing either dataset file."""

        output = Path(self._output_edit.text().strip())
        name = self.get_dataset_name()
        existing = [
            path for path in (output / f"{name}.zip", output / f"{name}.xml")
            if path.exists()
        ]
        if not existing:
            return True
        files = "\n".join(f"• {path.name}" for path in existing)
        reply = QMessageBox.question(
            self,
            "Replace Existing Dataset?",
            "Creating this dataset will replace the following existing file(s):\n\n"
            f"{files}\n\nThis cannot be undone. Replace them?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _update_summary(self) -> None:
        lines = [
            f"Image directory: {self._dir_edit.text()}",
            f"Format: {self._format_description()}",
            f"Flip: {'Yes' if self._flip_check.isChecked() else 'No'}",
            f"XY res: {self._xy_res_spin.value()} \u00b5m",
            f"Z res: {self._z_res_spin.value()} \u00b5m",
            f"Timepoints: {self._timepoints_spin.value()}",
            f"Z-planes: {self._planes_spin.value()}",
            f"Initial tracking: {self._tracking_description()}",
            f"Output: {self._output_edit.text()}",
            f"Dataset name: {self._dataset_name_edit.text()}",
        ]
        self._summary_label.setPlainText("\n".join(lines))

    def _format_description(self) -> str:
        if self._radio_split.isChecked():
            return "Side-by-side dual channel"
        elif self._radio_separate.isChecked():
            return "Separate directory per channel"
        elif self._radio_multistack.isChecked():
            n_ch = self._n_channels_spin.value()
            ordering = "CZ" if self._ordering_combo.currentIndex() == 0 else "ZC"
            return f"Interleaved multichannel TIFF stack ({n_ch} channels, order={ordering})"
        return "Single channel"

    def _tracking_description(self) -> str:
        if not self._radio_tracking_auto.isChecked():
            return "Manual annotation with optional Track Selected Cell"
        return (
            "Uncommitted review: "
            f"{self._tracking_workflow_combo.currentText()} draft "
            f"(radius={self._tracking_radius_spin.value():g} µm, "
            f"max displacement={self._tracking_link_distance_spin.value():g} µm, "
            "divisions="
            f"{'on' if self._tracking_division_check.isChecked() else 'off'})"
        )

    # ── Results ───────────────────────────────────────────────────

    def get_config(self) -> AceTreeConfig:
        """Build an AceTreeConfig from the dialog's current values."""
        validation_error = (
            self._image_source_validation_error()
            or self._image_layout_validation_error()
        )
        if validation_error:
            raise ValueError(validation_error)
        d = self._detected
        image_dir = Path(self._dir_edit.text())

        # Determine split/flip
        split = 1 if self._radio_split.isChecked() else 0
        flip = 1 if self._flip_check.isChecked() else 0

        # Build image_file path from detected pattern
        prefix = d.get("prefix", "")
        first_file = d.get("first_file")
        if first_file:
            image_file = image_dir / first_file
        elif prefix:
            image_file = image_dir / f"{prefix}1.tif"
        else:
            # Fallback: use first tif in directory
            tifs = _tiff_files(image_dir)
            image_file = tifs[0] if tifs else image_dir / "image_t001.tif"

        # Multi-channel config
        image_channels: dict[int, Path] = {}
        num_channels = 1
        stack_interleaved = False
        stack_channel_order = "CZ"
        if self._radio_separate.isChecked():
            num_channels = 2
            image_channels[1] = image_file
            ch2_dir = Path(self._ch2_dir_edit.text())
            # Layout validation above guarantees that a real Channel 2 TIFF is
            # present.  Never degrade a requested two-channel dataset to one
            # channel and silently clamp the detector to Channel 1 later.
            image_channels[2] = _tiff_files(ch2_dir)[0]
        elif self._radio_multistack.isChecked():
            num_channels = self._n_channels_spin.value()
            stack_interleaved = True
            stack_channel_order = (
                "CZ" if self._ordering_combo.currentIndex() == 0 else "ZC"
            )
            # Interleaved multichannel always reads channels from the pages;
            # split/flip would halve the image again.
            split = 0

        config = AceTreeConfig(
            image_file=image_file,
            image_channels=image_channels,
            num_channels=num_channels,
            xy_res=self._xy_res_spin.value(),
            z_res=self._z_res_spin.value(),
            plane_end=self._planes_spin.value(),
            starting_index=1,
            ending_index=self._timepoints_spin.value(),
            split=split,
            flip=flip,
            use_zip=0,
            use_stack=0,
            naming_method=NamingMethod.NEWCANONICAL,
            stack_interleaved=stack_interleaved,
            stack_channel_order=stack_channel_order,
        )

        # Derive tif_directory and tif_prefix
        config.tif_directory = image_dir
        if prefix:
            config.tif_prefix = prefix

        return config

    def get_output_directory(self) -> Path:
        validation_error = self._output_validation_error()
        if validation_error:
            raise ValueError(validation_error)
        return Path(self._output_edit.text().strip())

    def get_dataset_name(self) -> str:
        return self._dataset_name_edit.text().strip() or "dataset"

    def get_num_timepoints(self) -> int:
        return self._timepoints_spin.value()

    def get_tracking_request(self):
        """Return an initial global tracking request, or ``None`` for manual mode."""
        if not self._radio_tracking_auto.isChecked():
            return None

        validation_error = self._tracking_validation_error()
        if validation_error:
            raise ValueError(validation_error)

        from ..tracking.api import TrackingRequest, TrackingScope
        from ..tracking.registry import get_default_registry
        from ..tracking.settings import build_detector_spec, build_tracker_spec

        registry = get_default_registry()
        detector_id = str(self._tracking_detector_combo.currentData())
        tracker_id = str(self._tracking_tracker_combo.currentData())
        profile = None
        if self._tracking_workflow_combo.currentData() == "modern_starrynite":
            from ..tracking.starrynite import bundled_parameter_preset, load_tuning_profile

            preset = bundled_parameter_preset(
                str(self._tracking_starrynite_preset_combo.currentData())
            )
            profile = load_tuning_profile(
                preset.parameter_file,
                fallback_radius_um=self._tracking_radius_spin.value(),
            )
        return TrackingRequest(
            detector=build_detector_spec(
                registry,
                detector_id,
                channel=self._tracking_channel_spin.value(),
                radius_um=self._tracking_radius_spin.value(),
                threshold=self._tracking_threshold_spin.value(),
                source_settings=None if profile is None else profile.detector_settings,
            ),
            tracker=build_tracker_spec(
                registry,
                tracker_id,
                max_distance_um=self._tracking_link_distance_spin.value(),
                missing_frames=self._tracking_gap_spin.value(),
                allow_splitting=self._tracking_division_check.isChecked(),
                source_settings=None if profile is None else profile.tracker_settings,
            ),
            scope=TrackingScope(
                kind="global",
                start_frame=1,
                end_frame=self._timepoints_spin.value(),
            ),
        )


# ── Auto-detection helpers ────────────────────────────────────────


def _tiff_files(directory: Path) -> list[Path]:
    """Return TIFF files with case-insensitive suffix handling."""

    try:
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        )
    except OSError:
        return []


def _auto_detect_format(directory: Path) -> dict:
    """Probe a directory to guess image format, timepoints, and planes.

    Returns a dict with keys:
        num_files, pattern, prefix, per_plane, num_timepoints,
        num_planes, image_shape, first_file, error
    """
    result: dict = {"num_files": 0, "error": None}

    tifs = _tiff_files(directory)
    result["num_files"] = len(tifs)

    if not tifs:
        result["error"] = "No TIFF files found in directory"
        return result

    result["first_file"] = tifs[0].name

    # Check for per-plane pattern (-p in filename)
    per_plane_files = [f for f in tifs if re.search(r'-p\d+', f.stem, re.IGNORECASE)]
    result["per_plane"] = len(per_plane_files) > len(tifs) * 0.5

    # Try to extract timepoint numbers — scan end-to-beginning so a
    # stray earlier 't<digits>' in the prefix doesn't fool the detector.
    from acetree_py.io.image_provider import _parse_time_from_name

    timepoints = set()
    prefix_candidates = []
    for f in tifs:
        info = _parse_time_from_name(f.name)
        if info is None:
            continue
        timepoints.add(info.time)
        # Everything before the digit-bearing token (incl. trailing 't'
        # when present) is the dataset prefix candidate.
        prefix_end = info.prefix_end + 1 if info.has_t else info.prefix_end
        prefix_candidates.append(f.name[:prefix_end])

    if timepoints:
        result["num_timepoints"] = max(timepoints) - min(timepoints) + 1
        result["pattern"] = f"t{{NNN}} (range: {min(timepoints)}-{max(timepoints)})"
    else:
        # Fallback: count files as timepoints
        result["num_timepoints"] = len(tifs)
        result["pattern"] = "sequential files"

    if prefix_candidates:
        # Most common prefix
        from collections import Counter
        result["prefix"] = Counter(prefix_candidates).most_common(1)[0][0]

    # Probe first file for shape and planes
    try:
        import tifffile
        with tifffile.TiffFile(str(tifs[0])) as tif:
            n_pages = len(tif.pages)
            if n_pages > 0:
                page = tif.pages[0]
                result["image_shape"] = (page.shape[-2], page.shape[-1])
            if result.get("per_plane"):
                # Count planes by counting -p variants for the first timepoint
                first_t = min(timepoints) if timepoints else 1
                plane_count = sum(
                    1 for f in tifs
                    if re.search(rf't0*{first_t}\b', f.stem, re.IGNORECASE)
                    and re.search(r'-p\d+', f.stem, re.IGNORECASE)
                )
                result["num_planes"] = max(1, plane_count)
            else:
                result["num_planes"] = n_pages
    except Exception as e:
        result["error"] = f"Could not read TIFF: {e}"
        result["num_planes"] = 30  # fallback

    return result
