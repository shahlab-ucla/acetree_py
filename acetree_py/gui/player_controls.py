"""Player controls — time and plane navigation widget.

A Qt widget that provides buttons and sliders for navigating through
timepoints and z-planes. Also shows current position information.

Ported from: org.rhwlab.acetree.PlayerControl (PlayerControl.java)

Key differences from Java:
- No separate play/pause threading — napari's animation framework handles this
- Cleaner separation: this widget only emits navigation requests to AceTreeApp
- Additional slider for continuous scrubbing through time
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import QTimer, Qt
    from qtpy.QtWidgets import (
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
    # Create stub for type checking
    QWidget = object  # type: ignore[misc,assignment]


class PlayerControls(QWidget):  # type: ignore[misc]
    """Navigation widget for time and z-plane control.

    Layout:
        Row 1: |< < ▐▐ > >|  [time spinner]  t=042/200
        Row 2: [time slider ==================]
        Row 3: ▲ ▼  [plane spinner]  plane=15/30  [tracking checkbox]
    """

    # Playback interval in milliseconds
    PLAY_INTERVAL_MS = 200

    def __init__(self, app: AceTreeApp, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required for PlayerControls: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.app = app
        self._playing = False
        self._play_direction = 1  # 1 = forward, -1 = backward

        self._timer = QTimer()
        self._timer.timeout.connect(self._on_play_tick)

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Row 1: Time navigation buttons ──
        time_row = QHBoxLayout()

        self._btn_start = QPushButton("⏮")
        self._btn_start.setToolTip("Go to first timepoint")
        self._btn_start.setFixedWidth(32)
        self._btn_start.clicked.connect(lambda: self.app.set_time(1))

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setToolTip("Previous timepoint")
        self._btn_prev.setFixedWidth(32)
        self._btn_prev.clicked.connect(self.app.prev_time)

        self._btn_play_back = QPushButton("◀◀")
        self._btn_play_back.setToolTip("Play backward")
        self._btn_play_back.setFixedWidth(40)
        self._btn_play_back.clicked.connect(lambda: self._toggle_play(-1))

        self._btn_pause = QPushButton("⏸")
        self._btn_pause.setToolTip("Pause")
        self._btn_pause.setFixedWidth(32)
        self._btn_pause.clicked.connect(self._stop_play)

        self._btn_play_fwd = QPushButton("▶▶")
        self._btn_play_fwd.setToolTip("Play forward")
        self._btn_play_fwd.setFixedWidth(40)
        self._btn_play_fwd.clicked.connect(lambda: self._toggle_play(1))

        self._btn_next = QPushButton("▶")
        self._btn_next.setToolTip("Next timepoint")
        self._btn_next.setFixedWidth(32)
        self._btn_next.clicked.connect(self.app.next_time)

        self._btn_end = QPushButton("⏭")
        self._btn_end.setToolTip("Go to last timepoint")
        self._btn_end.setFixedWidth(32)
        self._btn_end.clicked.connect(
            lambda: self.app.set_time(self.app.manager.num_timepoints)
        )

        self._time_spin = QSpinBox()
        self._time_spin.setRange(1, max(1, self.app.manager.num_timepoints))
        self._time_spin.setValue(self.app.current_time)
        self._time_spin.setPrefix("t=")
        self._time_spin.valueChanged.connect(self.app.set_time)

        self._time_label = QLabel()

        for w in [self._btn_start, self._btn_prev, self._btn_play_back,
                   self._btn_pause, self._btn_play_fwd, self._btn_next,
                   self._btn_end]:
            time_row.addWidget(w)
        time_row.addWidget(self._time_spin)
        time_row.addWidget(self._time_label)
        time_row.addStretch()

        layout.addLayout(time_row)

        # ── Row 2: Time slider ──
        self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setRange(1, max(1, self.app.manager.num_timepoints))
        self._time_slider.setValue(self.app.current_time)
        self._time_slider.valueChanged.connect(self.app.set_time)
        layout.addWidget(self._time_slider)

        # ── Row 3: Plane navigation ──
        plane_row = QHBoxLayout()

        self._btn_plane_up = QPushButton("▲")
        self._btn_plane_up.setToolTip("Next z-plane (up)")
        self._btn_plane_up.setFixedWidth(32)
        self._btn_plane_up.clicked.connect(self.app.next_plane)

        self._btn_plane_down = QPushButton("▼")
        self._btn_plane_down.setToolTip("Previous z-plane (down)")
        self._btn_plane_down.setFixedWidth(32)
        self._btn_plane_down.clicked.connect(self.app.prev_plane)

        max_planes = self.app.image_provider.num_planes if self.app.image_provider else 30
        self._plane_spin = QSpinBox()
        self._plane_spin.setRange(1, max(1, max_planes))
        self._plane_spin.setValue(self.app.current_plane)
        self._plane_spin.setPrefix("z=")
        self._plane_spin.valueChanged.connect(self.app.set_plane)

        self._plane_label = QLabel()

        plane_row.addWidget(self._btn_plane_up)
        plane_row.addWidget(self._btn_plane_down)
        plane_row.addWidget(self._plane_spin)
        plane_row.addWidget(self._plane_label)
        plane_row.addStretch()

        # Label visibility controls
        self._btn_toggle_labels = QPushButton("Labels: ON")
        self._btn_toggle_labels.setToolTip("Toggle cell name labels on/off")
        self._btn_toggle_labels.setFixedWidth(80)
        self._btn_toggle_labels.clicked.connect(self._on_toggle_labels)

        self._btn_clear_labels = QPushButton("Clear Labels")
        self._btn_clear_labels.setToolTip("Remove all shown cell name labels")
        self._btn_clear_labels.setFixedWidth(90)
        self._btn_clear_labels.clicked.connect(self._on_clear_labels)

        plane_row.addWidget(self._btn_toggle_labels)
        plane_row.addWidget(self._btn_clear_labels)

        self._btn_deselect = QPushButton("Deselect")
        self._btn_deselect.setToolTip("Clear cell selection (key: Escape)")
        self._btn_deselect.setFixedWidth(65)
        self._btn_deselect.clicked.connect(self._on_deselect)
        plane_row.addWidget(self._btn_deselect)

        self._btn_3d = QPushButton("3D")
        self._btn_3d.setToolTip("Toggle 3D volume view (key: 3)")
        self._btn_3d.setFixedWidth(40)
        self._btn_3d.setCheckable(True)
        self._btn_3d.clicked.connect(self._on_toggle_3d)
        plane_row.addWidget(self._btn_3d)

        self._btn_3d_window = QPushButton("3D Window")
        self._btn_3d_window.setToolTip(
            "Open a detached 3D viewer window (visualization mode, "
            "synced to the main viewer's timepoint)"
        )
        self._btn_3d_window.setFixedWidth(80)
        self._btn_3d_window.clicked.connect(self._on_open_3d_window)
        plane_row.addWidget(self._btn_3d_window)

        layout.addLayout(plane_row)

    def refresh(self) -> None:
        """Update the widget to reflect the current app state."""
        # Block signals while updating to prevent recursive set_time calls
        self._time_spin.blockSignals(True)
        self._time_slider.blockSignals(True)
        self._plane_spin.blockSignals(True)

        self._time_spin.setValue(self.app.current_time)
        self._time_slider.setValue(self.app.current_time)
        self._plane_spin.setValue(self.app.current_plane)

        max_t = self.app.manager.num_timepoints
        self._time_label.setText(f"/ {max_t}")

        max_p = self.app.image_provider.num_planes if self.app.image_provider else 30
        self._plane_label.setText(f"/ {max_p}")

        self._btn_3d.setChecked(self.app._3d_mode)

        self._time_spin.blockSignals(False)
        self._time_slider.blockSignals(False)
        self._plane_spin.blockSignals(False)

    def _toggle_play(self, direction: int) -> None:
        """Start or toggle playback direction."""
        if self._playing and self._play_direction == direction:
            self._stop_play()
        else:
            self._play_direction = direction
            self._playing = True
            self._timer.start(self.PLAY_INTERVAL_MS)

    def _stop_play(self) -> None:
        """Stop playback."""
        self._playing = False
        self._timer.stop()

    def _on_play_tick(self) -> None:
        """Advance one frame during playback."""
        new_time = self.app.current_time + self._play_direction
        if new_time < 1 or new_time > self.app.manager.num_timepoints:
            self._stop_play()
            return
        self.app.set_time(new_time)

    def _on_toggle_labels(self) -> None:
        """Toggle global label visibility."""
        vi = self.app._viewer_integration
        if vi is not None:
            vi.toggle_labels_global()
            self._btn_toggle_labels.setText(
                "Labels: ON" if vi.labels_visible else "Labels: OFF"
            )

    def _on_clear_labels(self) -> None:
        """Clear all individually shown labels."""
        vi = self.app._viewer_integration
        if vi is not None:
            vi.clear_labels()

    def _on_deselect(self) -> None:
        """Clear the current cell selection."""
        self.app.current_cell_name = ""
        self.app.tracking = False
        self.app.update_display()

    def _on_open_3d_window(self) -> None:
        """Open a detached 3D viewer window."""
        self.app.open_3d_window()

    def _on_toggle_3d(self) -> None:
        """Toggle 3D volume view."""
        self.app.toggle_3d()
        self._btn_3d.setChecked(self.app._3d_mode)
