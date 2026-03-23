"""Contrast tools — per-channel brightness/contrast adjustment.

Provides sliders for adjusting the display range (min/max) for each
image channel. Changes are applied in real-time via napari's built-in
contrast limits feature.

Ported from: org.rhwlab.image.ImageContrastTool
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


class ContrastTools(QWidget):  # type: ignore[misc]
    """Widget for adjusting image contrast and brightness.

    Provides min/max sliders for the display range. When the image layer
    exists, adjusting the sliders updates napari's contrast_limits in real time.

    Layout:
        Min: [slider] [spinbox]
        Max: [slider] [spinbox]
        [Auto] [Reset]
    """

    def __init__(self, app: AceTreeApp, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.app = app

        # Detect bit depth from image data
        self._max_val = 65535  # Default to 16-bit
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Contrast Adjustment")
        layout.addWidget(title)

        form = QFormLayout()

        # Min slider + spinbox
        min_row = QHBoxLayout()
        self._min_slider = QSlider(Qt.Horizontal)
        self._min_slider.setRange(0, self._max_val)
        self._min_slider.setValue(0)
        self._min_slider.valueChanged.connect(self._on_min_changed)

        self._min_spin = QSpinBox()
        self._min_spin.setRange(0, self._max_val)
        self._min_spin.setValue(0)
        self._min_spin.valueChanged.connect(self._min_slider.setValue)

        min_row.addWidget(self._min_slider)
        min_row.addWidget(self._min_spin)
        form.addRow("Min:", min_row)

        # Max slider + spinbox
        max_row = QHBoxLayout()
        self._max_slider = QSlider(Qt.Horizontal)
        self._max_slider.setRange(0, self._max_val)
        self._max_slider.setValue(self._max_val)
        self._max_slider.valueChanged.connect(self._on_max_changed)

        self._max_spin = QSpinBox()
        self._max_spin.setRange(0, self._max_val)
        self._max_spin.setValue(self._max_val)
        self._max_spin.valueChanged.connect(self._max_slider.setValue)

        max_row.addWidget(self._max_slider)
        max_row.addWidget(self._max_spin)
        form.addRow("Max:", max_row)

        layout.addLayout(form)

        # Buttons
        btn_row = QHBoxLayout()

        self._btn_auto = QPushButton("Auto")
        self._btn_auto.setToolTip("Auto-adjust contrast from image data")
        self._btn_auto.clicked.connect(self._auto_contrast)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setToolTip("Reset to full range")
        self._btn_reset.clicked.connect(self._reset_contrast)

        btn_row.addWidget(self._btn_auto)
        btn_row.addWidget(self._btn_reset)
        layout.addLayout(btn_row)

        layout.addStretch()

    def _on_min_changed(self, value: int) -> None:
        """Handle min slider change."""
        self._min_spin.blockSignals(True)
        self._min_spin.setValue(value)
        self._min_spin.blockSignals(False)
        self._apply_contrast()

    def _on_max_changed(self, value: int) -> None:
        """Handle max slider change."""
        self._max_spin.blockSignals(True)
        self._max_spin.setValue(value)
        self._max_spin.blockSignals(False)
        self._apply_contrast()

    def _apply_contrast(self) -> None:
        """Apply current min/max to the napari image layer."""
        if self.app._image_layer is None:
            return

        lo = self._min_slider.value()
        hi = self._max_slider.value()
        if hi <= lo:
            hi = lo + 1

        try:
            self.app._image_layer.contrast_limits = (lo, hi)
        except Exception as e:
            logger.debug("Could not set contrast limits: %s", e)

    def _auto_contrast(self) -> None:
        """Auto-adjust contrast from the current image data."""
        if self.app._image_layer is None:
            return

        import numpy as np

        data = self.app._image_layer.data
        if data is None or data.size == 0:
            return

        # Use 1st/99th percentile for robust auto-contrast
        lo = int(np.percentile(data, 1))
        hi = int(np.percentile(data, 99))
        if hi <= lo:
            hi = lo + 1

        self._min_slider.blockSignals(True)
        self._max_slider.blockSignals(True)
        self._min_slider.setValue(lo)
        self._max_slider.setValue(hi)
        self._min_slider.blockSignals(False)
        self._max_slider.blockSignals(False)

        self._min_spin.setValue(lo)
        self._max_spin.setValue(hi)
        self._apply_contrast()

    def _reset_contrast(self) -> None:
        """Reset contrast to full range."""
        self._min_slider.setValue(0)
        self._max_slider.setValue(self._max_val)
