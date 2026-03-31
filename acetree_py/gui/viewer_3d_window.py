"""Detached 3D Viewer window — visualization-focused secondary viewer.

Opens a separate napari viewer that always displays in 3D mode with
visualization (rule-engine) coloring.  The timepoint is synced from
the main AceTree viewer so the 3D window follows along as you navigate
or edit in the main 2D view.

Features:
    - Own color-mode selector (preset dropdown)
    - Per-channel contrast sliders with visibility toggles
    - Left-click to toggle cell labels, label on/off + clear buttons
    - Time navigation with sync toggle
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
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


# Default colormaps for multi-channel 3D display
_CHANNEL_COLORMAPS = ["green", "magenta", "cyan", "yellow", "red", "blue"]


class Viewer3DWindow(QWidget):  # type: ignore[misc]
    """A detached 3D viewer window synced to the main AceTree app.

    Always operates in visualization mode (color-rule-engine coloring)
    and 3D display.  Does not support editing — it is a read-only
    visualization companion to the main viewer.
    """

    def __init__(self, app: AceTreeApp, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self.setWindowTitle("AceTree \u2014 3D Viewer")
        self.setWindowFlags(Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(900, 700)

        self._viewer = None  # napari.Viewer (created on show)
        self._image_layers: list = []  # one per channel
        self._points_layer = None
        self._trail_points_layer = None
        self._last_time: int = -1
        self._shown_labels: set[str] = set()
        self._labels_visible: bool = True

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # ── Top toolbar row ──
        toolbar = QHBoxLayout()

        # Time controls
        toolbar.addWidget(QLabel("Time:"))
        self._time_spin = QSpinBox()
        self._time_spin.setRange(1, self.app.manager.num_timepoints)
        self._time_spin.setValue(self.app.current_time)
        self._time_spin.valueChanged.connect(self._on_time_spin)
        toolbar.addWidget(self._time_spin)

        self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setRange(1, self.app.manager.num_timepoints)
        self._time_slider.setValue(self.app.current_time)
        self._time_slider.valueChanged.connect(self._on_time_slider)
        toolbar.addWidget(self._time_slider, stretch=1)

        self._chk_sync = QPushButton("Sync")
        self._chk_sync.setCheckable(True)
        self._chk_sync.setChecked(True)
        self._chk_sync.setToolTip(
            "When enabled, this window follows the main viewer's timepoint"
        )
        self._chk_sync.setFixedWidth(50)
        toolbar.addWidget(self._chk_sync)

        # Label controls
        self._btn_labels = QPushButton("Labels: ON")
        self._btn_labels.setFixedWidth(80)
        self._btn_labels.setToolTip("Toggle label visibility on/off")
        self._btn_labels.clicked.connect(self._on_toggle_labels)
        toolbar.addWidget(self._btn_labels)

        self._btn_clear_labels = QPushButton("Clear Labels")
        self._btn_clear_labels.setFixedWidth(85)
        self._btn_clear_labels.setToolTip("Remove all shown cell labels")
        self._btn_clear_labels.clicked.connect(self._on_clear_labels)
        toolbar.addWidget(self._btn_clear_labels)

        layout.addLayout(toolbar)

        # ── Middle: napari canvas + side panel ──
        middle = QHBoxLayout()

        # napari viewer container (takes most space)
        self._viewer_container = QVBoxLayout()
        middle.addLayout(self._viewer_container, stretch=1)

        # Side panel: color mode + contrast/channel controls
        side = QVBoxLayout()
        side.setSpacing(4)

        # Color preset selector
        preset_group = QGroupBox("Color Preset")
        preset_layout = QVBoxLayout(preset_group)
        preset_layout.setContentsMargins(4, 4, 4, 4)
        self._combo_preset = QComboBox()
        from .color_rules import PRESET_NAMES, PRESET_EDITING
        for key, label in PRESET_NAMES.items():
            if key != PRESET_EDITING:  # exclude editing mode
                self._combo_preset.addItem(label, userData=key)
        self._combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._combo_preset)
        side.addWidget(preset_group)

        # Channel/contrast controls (built dynamically)
        self._contrast_group = QGroupBox("Channels / Contrast")
        self._contrast_layout = QVBoxLayout(self._contrast_group)
        self._contrast_layout.setContentsMargins(4, 4, 4, 4)
        self._contrast_layout.setSpacing(2)
        self._channel_widgets: list[dict] = []
        side.addWidget(self._contrast_group)

        # Auto/Reset all
        btn_row = QHBoxLayout()
        btn_auto = QPushButton("Auto All")
        btn_auto.clicked.connect(self._auto_all_contrast)
        btn_reset = QPushButton("Reset All")
        btn_reset.clicked.connect(self._reset_all_contrast)
        btn_row.addWidget(btn_auto)
        btn_row.addWidget(btn_reset)
        side.addLayout(btn_row)

        side.addStretch()

        side_widget = QWidget()
        side_widget.setLayout(side)
        side_widget.setFixedWidth(220)
        middle.addWidget(side_widget)

        layout.addLayout(middle, stretch=1)

    def show(self) -> None:
        """Show the window and create the napari viewer inside it."""
        super().show()
        if self._viewer is None:
            self._create_viewer()
        self._last_time = -1  # force refresh
        self.refresh()

    def _create_viewer(self) -> None:
        """Create the embedded napari viewer in 3D mode."""
        try:
            import napari
        except ImportError:
            logger.error("napari is required for the 3D viewer window")
            return

        self._viewer = napari.Viewer(show=False, title="3D Viewer")

        # Embed the napari Qt window
        qt_widget = self._viewer.window._qt_window
        self._viewer_container.addWidget(qt_widget)

        # Hide napari's dock widgets
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for dw in list(self._viewer.window._dock_widgets.values()):
                dw.setVisible(False)

        # Load image stacks (all channels)
        self._load_stacks()

        # Build channel controls
        self._rebuild_channel_controls()

        # Switch to 3D
        self._viewer.dims.ndisplay = 3

    def _load_stacks(self) -> None:
        """Load full z-stacks for all channels."""
        if self._viewer is None or self.app.image_provider is None:
            return

        n_ch = self.app.image_provider.num_channels
        z_scale = self.app.manager.z_pix_res

        for ch in range(n_ch):
            try:
                stack = self.app.image_provider.get_stack(
                    self.app.current_time, channel=ch
                )
            except (FileNotFoundError, IndexError) as e:
                logger.warning("3D window: could not load ch%d: %s", ch, e)
                continue

            if ch < len(self._image_layers):
                self._image_layers[ch].data = stack
                self._image_layers[ch].scale = (z_scale, 1.0, 1.0)
            else:
                cmap = (
                    "gray"
                    if n_ch == 1
                    else _CHANNEL_COLORMAPS[ch % len(_CHANNEL_COLORMAPS)]
                )
                layer = self._viewer.add_image(
                    stack,
                    name=f"Ch{ch + 1}" if n_ch > 1 else "Image",
                    scale=(z_scale, 1.0, 1.0),
                    colormap=cmap,
                    blending="additive" if n_ch > 1 else "translucent",
                    opacity=0.6,
                )
                # Copy contrast from main viewer if available
                if ch < len(self.app._image_layers):
                    try:
                        layer.contrast_limits = (
                            self.app._image_layers[ch].contrast_limits
                        )
                    except Exception:
                        pass
                self._image_layers.append(layer)

    def _rebuild_channel_controls(self) -> None:
        """Build per-channel contrast/visibility controls."""
        # Clear existing
        for w in self._channel_widgets:
            w["group"].setParent(None)
            w["group"].deleteLater()
        self._channel_widgets.clear()

        n_ch = len(self._image_layers)
        max_val = 65535

        for ch in range(n_ch):
            grp = QGroupBox(f"Ch{ch + 1}" if n_ch > 1 else "Image")
            grp_layout = QVBoxLayout(grp)
            grp_layout.setSpacing(2)
            grp_layout.setContentsMargins(2, 2, 2, 2)

            widgets: dict = {"group": grp}

            if n_ch > 1:
                chk = QCheckBox("Visible")
                chk.setChecked(True)
                ch_idx = ch
                chk.toggled.connect(
                    lambda vis, c=ch_idx: self._on_ch_visible(c, vis)
                )
                grp_layout.addWidget(chk)
                widgets["chk"] = chk

            # Min
            min_row = QHBoxLayout()
            min_row.addWidget(QLabel("Min:"))
            min_sl = QSlider(Qt.Horizontal)
            min_sl.setRange(0, max_val)
            min_sp = QSpinBox()
            min_sp.setRange(0, max_val)
            min_sl.valueChanged.connect(
                lambda v, c=ch: self._on_contrast_min(c, v)
            )
            min_sp.valueChanged.connect(min_sl.setValue)
            min_row.addWidget(min_sl, stretch=1)
            min_row.addWidget(min_sp)
            grp_layout.addLayout(min_row)
            widgets["min_sl"] = min_sl
            widgets["min_sp"] = min_sp

            # Max
            max_row = QHBoxLayout()
            max_row.addWidget(QLabel("Max:"))
            max_sl = QSlider(Qt.Horizontal)
            max_sl.setRange(0, max_val)
            max_sl.setValue(max_val)
            max_sp = QSpinBox()
            max_sp.setRange(0, max_val)
            max_sp.setValue(max_val)
            max_sl.valueChanged.connect(
                lambda v, c=ch: self._on_contrast_max(c, v)
            )
            max_sp.valueChanged.connect(max_sl.setValue)
            max_row.addWidget(max_sl, stretch=1)
            max_row.addWidget(max_sp)
            grp_layout.addLayout(max_row)
            widgets["max_sl"] = max_sl
            widgets["max_sp"] = max_sp

            self._contrast_layout.addWidget(grp)
            self._channel_widgets.append(widgets)

    # ── Contrast handlers ──

    def _on_ch_visible(self, ch: int, visible: bool) -> None:
        if ch < len(self._image_layers):
            self._image_layers[ch].visible = visible

    def _on_contrast_min(self, ch: int, value: int) -> None:
        w = self._channel_widgets[ch]
        w["min_sp"].blockSignals(True)
        w["min_sp"].setValue(value)
        w["min_sp"].blockSignals(False)
        self._apply_contrast(ch)

    def _on_contrast_max(self, ch: int, value: int) -> None:
        w = self._channel_widgets[ch]
        w["max_sp"].blockSignals(True)
        w["max_sp"].setValue(value)
        w["max_sp"].blockSignals(False)
        self._apply_contrast(ch)

    def _apply_contrast(self, ch: int) -> None:
        if ch >= len(self._image_layers):
            return
        w = self._channel_widgets[ch]
        lo = w["min_sl"].value()
        hi = w["max_sl"].value()
        if hi <= lo:
            hi = lo + 1
        try:
            self._image_layers[ch].contrast_limits = (lo, hi)
        except Exception:
            pass

    def _auto_all_contrast(self) -> None:
        for ch in range(len(self._image_layers)):
            data = self._image_layers[ch].data
            if data is None or data.size == 0:
                continue
            lo = int(np.percentile(data, 1))
            hi = int(np.percentile(data, 99))
            if hi <= lo:
                hi = lo + 1
            w = self._channel_widgets[ch]
            w["min_sl"].blockSignals(True)
            w["max_sl"].blockSignals(True)
            w["min_sl"].setValue(lo)
            w["max_sl"].setValue(hi)
            w["min_sl"].blockSignals(False)
            w["max_sl"].blockSignals(False)
            w["min_sp"].setValue(lo)
            w["max_sp"].setValue(hi)
            self._apply_contrast(ch)

    def _reset_all_contrast(self) -> None:
        for ch in range(len(self._channel_widgets)):
            w = self._channel_widgets[ch]
            w["min_sl"].setValue(0)
            w["max_sl"].setValue(65535)

    # ── Color preset ──

    def _on_preset_changed(self, index: int) -> None:
        key = self._combo_preset.itemData(index)
        if key is None:
            return
        self.app.color_engine.load_preset(key)
        self._last_time = -1  # force redraw
        self.refresh()
        # Also update main viewer if it's in viz mode
        if self.app._viz_mode:
            self.app.update_display()

    # ── Labels ──

    def _on_toggle_labels(self) -> None:
        self._labels_visible = not self._labels_visible
        self._btn_labels.setText(
            "Labels: ON" if self._labels_visible else "Labels: OFF"
        )
        self._update_label_display()

    def _on_clear_labels(self) -> None:
        self._shown_labels.clear()
        self._update_label_display()

    def _update_label_display(self) -> None:
        """Refresh the text display on the points layer."""
        if self._points_layer is None:
            return
        features = self._points_layer.features
        if "full_name" not in features:
            return
        display_names = []
        for name in features["full_name"]:
            if self._labels_visible and name in self._shown_labels:
                display_names.append(name)
            else:
                display_names.append("")
        self._points_layer.features = {
            "name": display_names,
            "full_name": list(features["full_name"]),
        }

    # ── Refresh / update ──

    def refresh(self) -> None:
        """Update the 3D viewer to match the current app state."""
        if self._viewer is None:
            return

        cur_time = self.app.current_time
        if cur_time == self._last_time:
            return
        self._last_time = cur_time

        # Update time controls
        self._time_spin.blockSignals(True)
        self._time_slider.blockSignals(True)
        self._time_spin.setValue(cur_time)
        self._time_slider.setValue(cur_time)
        self._time_spin.blockSignals(False)
        self._time_slider.blockSignals(False)

        self._load_stacks()
        self._update_points()

    def _update_points(self) -> None:
        """Create/update 3D Points layer with visualization-mode colors."""
        if self._viewer is None:
            return

        nuclei = self.app.manager.alive_nuclei_at(self.app.current_time)
        z_scale = self.app.manager.z_pix_res

        coords = []
        sizes = []
        names_list = []

        for nuc in nuclei:
            coords.append([nuc.z, nuc.y, nuc.x])
            sizes.append(nuc.size)
            names_list.append(nuc.effective_name or f"Nuc{nuc.index}")

        # Always use visualization-mode coloring
        colors = [
            list(c)
            for c in self.app.color_engine.colors_for_frame(
                nuclei,
                self.app.manager,
                self.app.current_time,
                selected_name=self.app.current_cell_name,
            )
        ]

        if not coords:
            if self._points_layer is not None:
                self._points_layer.data = np.empty((0, 3))
            return

        coords_arr = np.array(coords)
        sizes_arr = np.array(sizes)
        colors_arr = np.array(colors)

        # Build display names (only show labels for toggled cells)
        display_names = []
        for name in names_list:
            if self._labels_visible and name in self._shown_labels:
                display_names.append(name)
            else:
                display_names.append("")

        if self._points_layer is None:
            self._points_layer = self._viewer.add_points(
                coords_arr,
                size=sizes_arr,
                face_color=colors_arr,
                border_color="transparent",
                name="Nuclei 3D",
                scale=(z_scale, 1.0, 1.0),
                opacity=0.8,
            )
            self._points_layer.features = {
                "name": display_names,
                "full_name": names_list,
            }
            self._points_layer.text = {
                "string": "{name}",
                "color": "white",
                "size": 10,
            }
            # Click callback for label toggling
            self._points_layer.mouse_drag_callbacks.append(self._on_click)
        else:
            self._points_layer.data = coords_arr
            self._points_layer.size = sizes_arr
            self._points_layer.face_color = colors_arr
            self._points_layer.features = {
                "name": display_names,
                "full_name": names_list,
            }

        self._update_trail()

    def _on_click(self, layer, event):
        """Handle click on 3D Points — left-click toggles cell label."""
        if event.button != 1:  # left click only
            yield
            return

        # Find clicked point index
        idx = layer.get_value(event.position, world=True)
        if idx is None:
            yield
            return

        features = layer.features
        if "full_name" not in features or idx >= len(features["full_name"]):
            yield
            return

        name = features["full_name"][idx]
        if name in self._shown_labels:
            self._shown_labels.discard(name)
        else:
            self._shown_labels.add(name)

        self._update_label_display()
        yield

    def _update_trail(self) -> None:
        """Update 3D ghost trail for the selected cell."""
        vi = self.app._viewer_integration
        if self._viewer is None or vi is None or not vi.trails_visible:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        cell_name = self.app.current_cell_name
        if not cell_name:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        cell = self.app.manager.get_cell(cell_name)
        if cell is None:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        trail_len = vi.trail_length
        start = max(cell.start_time, self.app.current_time - trail_len)

        coords = []
        sizes = []
        colors = []

        for t in range(start, self.app.current_time):
            nuc = cell.get_nucleus_at(t)
            if nuc is None:
                continue
            age = self.app.current_time - t
            alpha = max(0.15, 0.6 * (1.0 - age / (trail_len + 1)))
            coords.append([nuc.z, nuc.y, nuc.x])
            sizes.append(nuc.size * 0.6)
            colors.append([0.3, 0.8, 1.0, alpha])

        z_scale = self.app.manager.z_pix_res

        if not coords:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        coords_arr = np.array(coords)
        sizes_arr = np.array(sizes)
        colors_arr = np.array(colors)

        if self._trail_points_layer is None:
            self._trail_points_layer = self._viewer.add_points(
                coords_arr,
                size=sizes_arr,
                face_color=colors_arr,
                border_color="transparent",
                name="Trail 3D",
                scale=(z_scale, 1.0, 1.0),
                opacity=0.5,
            )
        else:
            self._trail_points_layer.data = coords_arr
            self._trail_points_layer.size = sizes_arr
            self._trail_points_layer.face_color = colors_arr

    # ── Time controls ──

    def _on_time_spin(self, value: int) -> None:
        self._time_slider.blockSignals(True)
        self._time_slider.setValue(value)
        self._time_slider.blockSignals(False)
        if self._chk_sync.isChecked():
            self.app.set_time(value)
        else:
            self._last_time = -1
            self.refresh()

    def _on_time_slider(self, value: int) -> None:
        self._time_spin.blockSignals(True)
        self._time_spin.setValue(value)
        self._time_spin.blockSignals(False)
        if self._chk_sync.isChecked():
            self.app.set_time(value)
        else:
            self._last_time = -1
            self.refresh()

    # ── Cleanup ──

    def closeEvent(self, event) -> None:
        """Clean up the napari viewer when the window is closed."""
        if hasattr(self.app, '_3d_windows'):
            try:
                self.app._3d_windows.remove(self)
            except ValueError:
                pass

        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

        super().closeEvent(event)
