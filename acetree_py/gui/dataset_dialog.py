"""Dataset creation dialog — create a new AceTree dataset from raw images.

Provides a wizard-style dialog that walks the user through:
1. Selecting an image directory
2. Configuring image format (single channel, side-by-side, separate dirs, multichannel stack)
3. Setting voxel sizes and reviewing auto-detected parameters
4. Choosing an output directory for the nuclei ZIP and config XML
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

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
        QPushButton,
        QRadioButton,
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

from ..io.config import AceTreeConfig, NamingMethod


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
        self._page4 = self._build_page4_output()

        self._stack.addWidget(self._page1)
        self._stack.addWidget(self._page2)
        self._stack.addWidget(self._page3)
        self._stack.addWidget(self._page4)

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
            self._planes_spin.setValue(d["num_planes"])

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

    def _build_page4_output(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<b>Step 4: Output Location</b>"))
        layout.addWidget(QLabel("Choose where to save the dataset files (nuclei ZIP + config XML)."))

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
        if idx < self._stack.count() - 1:
            self._stack.setCurrentIndex(idx + 1)
            if idx + 1 == self._stack.count() - 1:
                self._update_summary()
        else:
            # Last page — "Create" pressed
            self.accept()
        self._update_nav_buttons()

    def _update_nav_buttons(self) -> None:
        idx = self._stack.currentIndex()
        self._btn_back.setEnabled(idx > 0)
        is_last = idx == self._stack.count() - 1
        self._btn_next.setText("Create" if is_last else "Next")

    def _update_summary(self) -> None:
        d = self._detected
        lines = [
            f"Image directory: {self._dir_edit.text()}",
            f"Format: {self._format_description()}",
            f"Flip: {'Yes' if self._flip_check.isChecked() else 'No'}",
            f"XY res: {self._xy_res_spin.value()} \u00b5m",
            f"Z res: {self._z_res_spin.value()} \u00b5m",
            f"Timepoints: {self._timepoints_spin.value()}",
            f"Z-planes: {self._planes_spin.value()}",
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

    # ── Results ───────────────────────────────────────────────────

    def get_config(self) -> AceTreeConfig:
        """Build an AceTreeConfig from the dialog's current values."""
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
            tifs = sorted(image_dir.glob("*.tif")) + sorted(image_dir.glob("*.tiff"))
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
            if ch2_dir.exists():
                ch2_tifs = sorted(ch2_dir.glob("*.tif")) + sorted(ch2_dir.glob("*.tiff"))
                if ch2_tifs:
                    image_channels[2] = ch2_tifs[0]
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
        return Path(self._output_edit.text())

    def get_dataset_name(self) -> str:
        return self._dataset_name_edit.text().strip() or "dataset"

    def get_num_timepoints(self) -> int:
        return self._timepoints_spin.value()


# ── Auto-detection helpers ────────────────────────────────────────


def _auto_detect_format(directory: Path) -> dict:
    """Probe a directory to guess image format, timepoints, and planes.

    Returns a dict with keys:
        num_files, pattern, prefix, per_plane, num_timepoints,
        num_planes, image_shape, first_file, error
    """
    result: dict = {"num_files": 0, "error": None}

    tifs = sorted(directory.glob("*.tif")) + sorted(directory.glob("*.tiff"))
    # Deduplicate (in case .tif and .tiff overlap)
    seen = set()
    unique_tifs = []
    for t in tifs:
        if t.name not in seen:
            seen.add(t.name)
            unique_tifs.append(t)
    tifs = unique_tifs
    result["num_files"] = len(tifs)

    if not tifs:
        result["error"] = "No TIFF files found in directory"
        return result

    result["first_file"] = tifs[0].name

    # Check for per-plane pattern (-p in filename)
    per_plane_files = [f for f in tifs if re.search(r'-p\d+', f.stem, re.IGNORECASE)]
    result["per_plane"] = len(per_plane_files) > len(tifs) * 0.5

    # Try to extract timepoint numbers
    time_pattern = re.compile(r't(\d+)', re.IGNORECASE)
    timepoints = set()
    prefix_candidates = []
    for f in tifs:
        m = time_pattern.search(f.stem)
        if m:
            timepoints.add(int(m.group(1)))
            # Extract prefix (everything before 't' + digits)
            pm = re.match(r'^(.+?t)\d+', f.stem, re.IGNORECASE)
            if pm:
                prefix_candidates.append(pm.group(1))

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
