"""Contrast tools — per-channel brightness/contrast adjustment.

Provides sliders for adjusting the display range (min/max) for each
image channel, with visibility toggles.  Changes are applied in
real-time via napari's built-in contrast limits feature.

Ported from: org.rhwlab.image.ImageContrastTool
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .app import AceTreeApp
    from .viewer_3d_window import Viewer3DWindow

try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QDoubleSpinBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


def _nonzero_range(lo: float, hi: float) -> tuple[float, float]:
    if hi > lo:
        return lo, hi
    padding = abs(lo) * 1e-6 if lo else 0.5
    return lo - padding, hi + padding


def _display_range(data) -> tuple[float, float]:
    """Integer dtype range, or finite floating-point data range."""
    array = np.asarray(data)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return float(info.min), float(info.max)
    finite = np.isfinite(array)
    if not finite.any():
        return 0.0, 1.0
    return _nonzero_range(
        float(np.min(array, where=finite, initial=np.inf)),
        float(np.max(array, where=finite, initial=-np.inf)),
    )


class _ChannelControls:
    """Keep exact display values separate from integer slider positions."""

    def __init__(self, layer) -> None:
        self.layer = layer
        self._data = None
        self.group = QGroupBox()
        self.chk_visible = QCheckBox()
        self.min_slider = QSlider(Qt.Horizontal)
        self.max_slider = QSlider(Qt.Horizontal)
        self.min_spin = QDoubleSpinBox()
        self.max_spin = QDoubleSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setFixedWidth(110)
            spin.setKeyboardTracking(False)
        self.refresh()
        self.min_slider.valueChanged.connect(lambda v: self.edit(0, self.from_slider(v)))
        self.max_slider.valueChanged.connect(lambda v: self.edit(1, self.from_slider(v)))
        self.min_spin.valueChanged.connect(lambda v: self.edit(0, v))
        self.max_spin.valueChanged.connect(lambda v: self.edit(1, v))
        self.chk_visible.toggled.connect(lambda v: setattr(self.layer, "visible", v))
        self._events = getattr(layer, "events", None)
        if self._events is not None:
            self._events.contrast_limits.connect(self.refresh)
            self._events.visible.connect(self.refresh)

    def dispose(self):
        if self._events is not None:
            self._events.contrast_limits.disconnect(self.refresh)
            self._events.visible.disconnect(self.refresh)

    def refresh(self, event=None):
        if self._data is not self.layer.data:
            self._data = self.layer.data
            self.full_range = _display_range(self._data)
        limits = tuple(float(v) for v in self.layer.contrast_limits)
        if not all(math.isfinite(v) for v in limits) or limits[1] <= limits[0]:
            limits = self.full_range
        self.bounds = (min(self.full_range[0], limits[0]), max(self.full_range[1], limits[1]))
        span = self.bounds[1] - self.bounds[0]
        integer = np.issubdtype(np.asarray(self._data).dtype, np.integer)
        self.direct_slider = integer and self.bounds[0] >= -2147483647 and self.bounds[1] <= 2147483647
        decimals = 0 if integer else min(323, max(7, int(-math.floor(math.log10(span))) + 7))
        for index, (slider, spin) in enumerate(((self.min_slider, self.min_spin), (self.max_slider, self.max_spin))):
            slider.blockSignals(True)
            spin.blockSignals(True)
            if self.direct_slider:
                slider.setRange(int(self.bounds[0]), int(self.bounds[1]))
            else:
                slider.setRange(0, 65535)
            spin.setDecimals(decimals)
            spin.setRange(*self.bounds)
            spin.setSingleStep(1.0 if integer else span / 1000)
            spin.setValue(limits[index])
            slider.setValue(self.to_slider(limits[index]))
            spin.blockSignals(False)
            slider.blockSignals(False)
        self.chk_visible.blockSignals(True)
        self.chk_visible.setChecked(self.layer.visible)
        self.chk_visible.blockSignals(False)

    def from_slider(self, value):
        if self.direct_slider:
            return float(value)
        return self.bounds[0] + value / 65535 * (self.bounds[1] - self.bounds[0])

    def to_slider(self, value):
        if self.direct_slider:
            return round(value)
        return round((value - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) * 65535)

    def edit(self, index, value):
        limits = [self.min_spin.value(), self.max_spin.value()]
        limits[index] = value
        if limits[1] <= limits[0]:
            step = max(self.min_spin.singleStep(), abs(value) * 1e-12)
            limits[1 - index] = value + step if index == 0 else value - step
        self.set_limits(limits)

    def set_limits(self, limits):
        self.layer.contrast_limits = tuple(limits)
        self.refresh()

    def auto(self):
        data = np.asarray(self.layer.data)
        finite = data[np.isfinite(data)]
        if finite.size:
            lo, hi = np.percentile(finite.astype(np.float64, copy=False), [1, 99])
            self.set_limits(_nonzero_range(float(lo), float(hi)))

    def reset(self):
        self.refresh()
        self.set_limits(self.full_range)


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

    def __init__(self, app: AceTreeApp | Viewer3DWindow, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.app = app
        self._channel_ctrls: list[_ChannelControls] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        title = QLabel("Channels and contrast")
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
        if (n_layers != len(self._channel_ctrls)
                or any(ctrl.layer is not layer for ctrl, layer in zip(self._channel_ctrls, self.app._image_layers))):
            self._rebuild_channels(n_layers)
        for ctrl in self._channel_ctrls:
            ctrl.refresh()

    def _rebuild_channels(self, n_channels: int) -> None:
        """Tear down and rebuild per-channel controls."""
        # Remove old widgets
        for ctrl in self._channel_ctrls:
            ctrl.dispose()
            ctrl.group.setParent(None)
            ctrl.group.deleteLater()
        self._channel_ctrls.clear()

        for ch in range(n_channels):
            ctrl = _ChannelControls(self.app._image_layers[ch])
            for name, spin, slider in (("minimum", ctrl.min_spin, ctrl.min_slider),
                                       ("maximum", ctrl.max_spin, ctrl.max_slider)):
                spin.setAccessibleName(f"Channel {ch + 1} contrast {name}")
                slider.setAccessibleName(f"Channel {ch + 1} contrast {name} slider")
            label = f"Ch{ch + 1}" if n_channels > 1 else "Image"
            ctrl.group.setTitle(label)
            group_layout = QVBoxLayout(ctrl.group)
            group_layout.setSpacing(2)
            group_layout.setContentsMargins(4, 4, 4, 4)

            if n_channels > 1:
                vis_row = QHBoxLayout()
                ctrl.chk_visible.setText("Visible")
                vis_row.addWidget(ctrl.chk_visible)
                vis_row.addStretch()
                group_layout.addLayout(vis_row)

            # Min row
            min_row = QHBoxLayout()
            min_label = QLabel("&Min:")
            min_label.setBuddy(ctrl.min_spin)
            min_row.addWidget(min_label)
            min_row.addWidget(ctrl.min_slider, stretch=1)
            min_row.addWidget(ctrl.min_spin)
            group_layout.addLayout(min_row)

            # Max row
            max_row = QHBoxLayout()
            max_label = QLabel("Ma&x:")
            max_label.setBuddy(ctrl.max_spin)
            max_row.addWidget(max_label)
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

            self._channels_container.addWidget(ctrl.group)
            self._channel_ctrls.append(ctrl)

    def _auto_channel(self, ch: int) -> None:
        if ch < len(self._channel_ctrls):
            self._channel_ctrls[ch].auto()

    def _reset_channel(self, ch: int) -> None:
        if ch < len(self._channel_ctrls):
            self._channel_ctrls[ch].reset()

    def _auto_all(self) -> None:
        for ch in range(len(self._channel_ctrls)):
            self._auto_channel(ch)

    def _reset_all(self) -> None:
        for ch in range(len(self._channel_ctrls)):
            self._reset_channel(ch)
