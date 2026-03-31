"""Contrast tools — per-channel brightness/contrast adjustment.

Provides sliders for adjusting the display range (min/max) for each
image channel, with visibility toggles.  Changes are applied in
real-time via napari's built-in contrast limits feature.

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
        QCheckBox,
        QGroupBox,
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


class _ChannelControls:
    """Slider/spinbox pair for a single channel."""

    def __init__(self, max_val: int = 65535) -> None:
        self.max_val = max_val
        self.group = QGroupBox()
        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(True)
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, max_val)
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, max_val)
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, max_val)
        self.max_slider.setValue(max_val)
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, max_val)
        self.max_spin.setValue(max_val)


class ContrastTools(QWidget):  # type: ignore[misc]
    """Widget for adjusting image contrast and brightness.

    Dynamically creates one control group per channel when channels
    are detected.  Each channel has:
      - Visibility checkbox
      - Min/Max sliders with spinboxes
      - Auto / Reset buttons

    For single-channel data, shows a simplified layout without the
    visibility checkbox.
    """

    def __init__(self, app: AceTreeApp, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.app = app
        self._max_val = 65535
        self._channel_ctrls: list[_ChannelControls] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        title = QLabel("Contrast")
        self._layout.addWidget(title)

        self._channels_container = QVBoxLayout()
        self._layout.addLayout(self._channels_container)

        # Global buttons
        btn_row = QHBoxLayout()
        btn_auto = QPushButton("Auto All")
        btn_auto.setToolTip("Auto-adjust contrast for all channels")
        btn_auto.clicked.connect(self._auto_all)
        btn_reset = QPushButton("Reset All")
        btn_reset.setToolTip("Reset all channels to full range")
        btn_reset.clicked.connect(self._reset_all)
        btn_row.addWidget(btn_auto)
        btn_row.addWidget(btn_reset)
        self._layout.addLayout(btn_row)

        self._layout.addStretch()

    def refresh(self) -> None:
        """Rebuild channel controls if the number of layers changed."""
        n_layers = len(self.app._image_layers)
        if n_layers != len(self._channel_ctrls):
            self._rebuild_channels(n_layers)

    def _rebuild_channels(self, n_channels: int) -> None:
        """Tear down and rebuild per-channel controls."""
        # Remove old widgets
        for ctrl in self._channel_ctrls:
            ctrl.group.setParent(None)
            ctrl.group.deleteLater()
        self._channel_ctrls.clear()

        for ch in range(n_channels):
            ctrl = _ChannelControls(self._max_val)
            label = f"Ch{ch + 1}" if n_channels > 1 else "Image"
            ctrl.group.setTitle(label)
            group_layout = QVBoxLayout(ctrl.group)
            group_layout.setSpacing(2)
            group_layout.setContentsMargins(4, 4, 4, 4)

            if n_channels > 1:
                vis_row = QHBoxLayout()
                ctrl.chk_visible.setText("Visible")
                ctrl.chk_visible.setChecked(True)
                ch_idx = ch  # capture for closure
                ctrl.chk_visible.toggled.connect(
                    lambda checked, c=ch_idx: self._on_visible_toggled(c, checked)
                )
                vis_row.addWidget(ctrl.chk_visible)
                vis_row.addStretch()
                group_layout.addLayout(vis_row)

            # Min row
            min_row = QHBoxLayout()
            min_row.addWidget(QLabel("Min:"))
            min_row.addWidget(ctrl.min_slider, stretch=1)
            min_row.addWidget(ctrl.min_spin)
            group_layout.addLayout(min_row)

            # Max row
            max_row = QHBoxLayout()
            max_row.addWidget(QLabel("Max:"))
            max_row.addWidget(ctrl.max_slider, stretch=1)
            max_row.addWidget(ctrl.max_spin)
            group_layout.addLayout(max_row)

            # Per-channel auto/reset
            btn_row = QHBoxLayout()
            btn_auto = QPushButton("Auto")
            btn_auto.clicked.connect(
                lambda _, c=ch: self._auto_channel(c)
            )
            btn_reset = QPushButton("Reset")
            btn_reset.clicked.connect(
                lambda _, c=ch: self._reset_channel(c)
            )
            btn_row.addWidget(btn_auto)
            btn_row.addWidget(btn_reset)
            group_layout.addLayout(btn_row)

            # Wire signals
            ch_idx = ch
            ctrl.min_slider.valueChanged.connect(
                lambda v, c=ch_idx: self._on_min_changed(c, v)
            )
            ctrl.max_slider.valueChanged.connect(
                lambda v, c=ch_idx: self._on_max_changed(c, v)
            )
            ctrl.min_spin.valueChanged.connect(
                lambda v, c=ch_idx: self._channel_ctrls[c].min_slider.setValue(v)
            )
            ctrl.max_spin.valueChanged.connect(
                lambda v, c=ch_idx: self._channel_ctrls[c].max_slider.setValue(v)
            )

            self._channels_container.addWidget(ctrl.group)
            self._channel_ctrls.append(ctrl)

    def _on_visible_toggled(self, ch: int, visible: bool) -> None:
        if ch < len(self.app._image_layers):
            self.app._image_layers[ch].visible = visible

    def _on_min_changed(self, ch: int, value: int) -> None:
        ctrl = self._channel_ctrls[ch]
        ctrl.min_spin.blockSignals(True)
        ctrl.min_spin.setValue(value)
        ctrl.min_spin.blockSignals(False)
        self._apply_contrast(ch)

    def _on_max_changed(self, ch: int, value: int) -> None:
        ctrl = self._channel_ctrls[ch]
        ctrl.max_spin.blockSignals(True)
        ctrl.max_spin.setValue(value)
        ctrl.max_spin.blockSignals(False)
        self._apply_contrast(ch)

    def _apply_contrast(self, ch: int) -> None:
        if ch >= len(self.app._image_layers):
            return
        ctrl = self._channel_ctrls[ch]
        lo = ctrl.min_slider.value()
        hi = ctrl.max_slider.value()
        if hi <= lo:
            hi = lo + 1
        try:
            self.app._image_layers[ch].contrast_limits = (lo, hi)
        except Exception as e:
            logger.debug("Could not set contrast limits ch%d: %s", ch, e)

    def _auto_channel(self, ch: int) -> None:
        if ch >= len(self.app._image_layers):
            return
        import numpy as np

        data = self.app._image_layers[ch].data
        if data is None or data.size == 0:
            return
        lo = int(np.percentile(data, 1))
        hi = int(np.percentile(data, 99))
        if hi <= lo:
            hi = lo + 1
        ctrl = self._channel_ctrls[ch]
        ctrl.min_slider.blockSignals(True)
        ctrl.max_slider.blockSignals(True)
        ctrl.min_slider.setValue(lo)
        ctrl.max_slider.setValue(hi)
        ctrl.min_slider.blockSignals(False)
        ctrl.max_slider.blockSignals(False)
        ctrl.min_spin.setValue(lo)
        ctrl.max_spin.setValue(hi)
        self._apply_contrast(ch)

    def _reset_channel(self, ch: int) -> None:
        if ch >= len(self._channel_ctrls):
            return
        ctrl = self._channel_ctrls[ch]
        ctrl.min_slider.setValue(0)
        ctrl.max_slider.setValue(self._max_val)

    def _auto_all(self) -> None:
        for ch in range(len(self._channel_ctrls)):
            self._auto_channel(ch)

    def _reset_all(self) -> None:
        for ch in range(len(self._channel_ctrls)):
            self._reset_channel(ch)
