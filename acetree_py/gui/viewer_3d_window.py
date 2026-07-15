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

from .marker_layers import (
    configure_curated_points_layer,
    passed_drag_threshold,
    point_anchor,
    pointer_position,
    replace_points_layer,
)

if TYPE_CHECKING:
    from ..tracking.api import Calibration
    from .app import AceTreeApp
    from .tracking_preview import ExpandedTrackingPreview

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import QTimer, Qt
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


def _document_change_counter(app) -> int | None:
    """Return the monotonic edit token when the host app exposes one."""

    history = getattr(app, "edit_history", None)
    value = getattr(history, "change_counter", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
        self._last_change_counter: int | None = _document_change_counter(app)
        self._local_time: int = app.current_time
        self._shown_labels: set[str] = set()
        self._labels_visible: bool = True
        self._tracking_preview_points_layer = None
        self._tracking_preview_paths_layer = None
        self._tracking_preview: ExpandedTrackingPreview | None = None
        self._tracking_preview_calibration: Calibration | None = None
        self._tracking_preview_visible: bool = False
        self._tracking_preview_stale: bool = False
        self._tracking_preview_highlight: str | None = None
        self._detector_preview_points_layer = None
        self._detector_preview: ExpandedTrackingPreview | None = None
        self._detector_preview_calibration: Calibration | None = None
        self._detector_preview_visible: bool = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # ── Top toolbar row ──
        toolbar = QHBoxLayout()

        # Time controls
        time_label = QLabel("&Time:")
        self._time_spin = QSpinBox()
        time_label.setBuddy(self._time_spin)
        self._time_spin.setAccessibleName("Detached 3D timepoint")
        self._time_spin.setRange(1, self.app.manager.num_timepoints)
        self._time_spin.setValue(self.app.current_time)
        self._time_spin.valueChanged.connect(self._on_time_spin)
        toolbar.addWidget(time_label)
        toolbar.addWidget(self._time_spin)

        self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setAccessibleName("Detached 3D timepoint slider")
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
        self._chk_sync.setMinimumWidth(50)
        self._chk_sync.toggled.connect(self._on_sync_toggled)
        toolbar.addWidget(self._chk_sync)

        # Label controls
        self._btn_labels = QPushButton("Labels: ON")
        self._btn_labels.setMinimumWidth(80)
        self._btn_labels.setToolTip("Toggle label visibility on/off")
        self._btn_labels.clicked.connect(self._on_toggle_labels)
        toolbar.addWidget(self._btn_labels)

        self._btn_clear_labels = QPushButton("Clear Labels")
        self._btn_clear_labels.setMinimumWidth(85)
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
        side_widget.setMinimumWidth(220)
        middle.addWidget(side_widget)

        layout.addLayout(middle, stretch=1)

    def show(self) -> None:
        """Show the window and create the napari viewer inside it."""
        super().show()
        created = self._viewer is None
        if created:
            self._create_viewer()
        self._last_time = -1  # refresh images and curated points once
        self.refresh()
        if created:
            self._rebuild_channel_controls()
        viewer_integration = getattr(self.app, "_viewer_integration", None)
        if viewer_integration is not None:
            viewer_integration.sync_tracking_preview_window(self)

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

        # Switch to 3D
        self._viewer.dims.ndisplay = 3

    def _load_stacks(self, timepoint: int | None = None) -> None:
        """Load full z-stacks for all channels."""
        if self._viewer is None or self.app.image_provider is None:
            return
        if timepoint is None:
            timepoint = self.view_time

        n_ch = self.app.image_provider.num_channels
        z_scale = self.app.manager.z_pix_res

        for ch in range(n_ch):
            try:
                stack = self.app.image_provider.get_stack(
                    timepoint, channel=ch
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
            min_label = QLabel("&Min:")
            min_sl = QSlider(Qt.Horizontal)
            min_sl.setAccessibleName(f"Channel {ch + 1} contrast minimum slider")
            min_sl.setRange(0, max_val)
            min_sp = QSpinBox()
            min_sp.setAccessibleName(f"Channel {ch + 1} contrast minimum")
            min_label.setBuddy(min_sp)
            min_sp.setRange(0, max_val)
            min_sl.valueChanged.connect(
                lambda v, c=ch: self._on_contrast_min(c, v)
            )
            min_sp.valueChanged.connect(min_sl.setValue)
            min_row.addWidget(min_label)
            min_row.addWidget(min_sl, stretch=1)
            min_row.addWidget(min_sp)
            grp_layout.addLayout(min_row)
            widgets["min_sl"] = min_sl
            widgets["min_sp"] = min_sp

            # Max
            max_row = QHBoxLayout()
            max_label = QLabel("Ma&x:")
            max_sl = QSlider(Qt.Horizontal)
            max_sl.setAccessibleName(f"Channel {ch + 1} contrast maximum slider")
            max_sl.setRange(0, max_val)
            max_sl.setValue(max_val)
            max_sp = QSpinBox()
            max_sp.setAccessibleName(f"Channel {ch + 1} contrast maximum")
            max_label.setBuddy(max_sp)
            max_sp.setRange(0, max_val)
            max_sp.setValue(max_val)
            max_sl.valueChanged.connect(
                lambda v, c=ch: self._on_contrast_max(c, v)
            )
            max_sp.valueChanged.connect(max_sl.setValue)
            max_row.addWidget(max_label)
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

    @property
    def view_time(self) -> int:
        """Time displayed here, independent from the main app when unsynced."""

        try:
            if self._chk_sync.isChecked():
                return int(self.app.current_time)
        except Exception:
            pass
        return int(self._local_time)

    def refresh(self, *, force: bool = False) -> None:
        """Update this viewer, honoring its local time when sync is disabled."""
        if self._viewer is None:
            return

        cur_time = self.view_time
        change_counter = _document_change_counter(self.app)
        if (
            cur_time == self._last_time
            and change_counter == getattr(self, "_last_change_counter", None)
            and not force
        ):
            return
        # Update time controls
        self._time_spin.blockSignals(True)
        self._time_slider.blockSignals(True)
        self._time_spin.setValue(cur_time)
        self._time_slider.setValue(cur_time)
        self._time_spin.blockSignals(False)
        self._time_slider.blockSignals(False)

        self._load_stacks(cur_time)
        self._update_points(cur_time)
        self._update_tracking_preview()
        detector_updater = getattr(self, "_update_detector_preview", None)
        if callable(detector_updater):
            detector_updater()

        # Cache only a fully rendered refresh. If any stack, marker, or
        # preview update fails, leave the old tokens intact so the same frame
        # is retried instead of being mistaken for successfully current.
        self._last_time = cur_time
        self._last_change_counter = change_counter

    def _update_points(self, timepoint: int | None = None) -> None:
        """Create/update 3D Points layer with visualization-mode colors."""
        if self._viewer is None:
            return
        if timepoint is None:
            timepoint = self.view_time

        nuclei = self.app.manager.alive_nuclei_at(timepoint)
        z_scale = self.app.manager.z_pix_res
        selection_resolver = getattr(self.app, "get_selected_nucleus", None)
        resolved_selection = (
            selection_resolver(timepoint)
            if callable(selection_resolver)
            else None
        )
        selected_nucleus = (
            resolved_selection[0]
            if resolved_selection is not None
            and resolved_selection[1] == timepoint
            else None
        )

        coords = []
        sizes = []
        names_list = []

        for nuc in nuclei:
            coords.append([self._stack_z_from_plane(nuc.z), nuc.y, nuc.x])
            sizes.append(nuc.size)
            names_list.append(nuc.effective_name or f"Nuc{nuc.index}")

        # Always use visualization-mode coloring
        colors = [
            list(c)
            for c in self.app.color_engine.colors_for_frame(
                nuclei,
                self.app.manager,
                timepoint,
                selected_name="",
            )
        ]
        selected_color = list(
            getattr(
                self.app.color_engine,
                "selected_color",
                (1.0, 1.0, 1.0, 1.0),
            )
        )
        for i, nuc in enumerate(nuclei):
            if nuc is selected_nucleus:
                colors[i] = selected_color

        if not coords:
            if self._points_layer is not None:
                try:
                    completed = replace_points_layer(
                        self._points_layer,
                        data=np.empty((0, 3)),
                        size=np.empty(0),
                        face_color=np.empty((0, 4)),
                        features={
                            "name": [],
                            "full_name": [],
                            "acetree_time": [],
                            "acetree_index": [],
                        },
                    )
                    if not completed:
                        raise RuntimeError(
                            "Centroid marker redraw failed; the previous "
                            "complete marker set was restored"
                        )
                finally:
                    self._make_curated_points_read_only(self._points_layer)
            self._update_trail(timepoint)
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
        point_features = {
            "name": display_names,
            "full_name": names_list,
            "acetree_time": [timepoint] * len(nuclei),
            "acetree_index": [nuc.index for nuc in nuclei],
        }

        if self._points_layer is None:
            # Create a single layer, then enforce read-only behavior. A broad
            # TypeError retry could otherwise duplicate a partially created
            # layer when the error was unrelated to ``editable`` support.
            layer = self._viewer.add_points(
                coords_arr,
                size=sizes_arr,
                face_color=colors_arr,
                border_color="transparent",
                name="Nuclei 3D",
                scale=(z_scale, 1.0, 1.0),
                opacity=0.8,
                features=point_features,
            )
            self._points_layer = layer
            configure_curated_points_layer(
                layer,
                callback=self._on_click,
                lock=self._make_curated_points_read_only,
            )
        else:
            configure_curated_points_layer(
                self._points_layer,
                callback=self._on_click,
                lock=self._make_curated_points_read_only,
            )
            try:
                completed = replace_points_layer(
                    self._points_layer,
                    data=coords_arr,
                    size=sizes_arr,
                    face_color=colors_arr,
                    features=point_features,
                )
                if not completed:
                    raise RuntimeError(
                        "Centroid marker redraw failed; the previous "
                        "complete marker set was restored"
                    )
            finally:
                self._make_curated_points_read_only(self._points_layer)

        self._make_curated_points_read_only(self._points_layer)

        self._update_trail(timepoint)

    @staticmethod
    def _make_curated_points_read_only(layer) -> None:
        """Lock curated points while retaining the label-click callback."""

        try:
            layer.editable = False
        except Exception:
            pass
        try:
            layer.selected_data = set()
        except Exception:
            pass
        try:
            layer.mode = "pan_zoom"
        except Exception:
            pass

    def _stack_z_from_plane(self, plane: float) -> float:
        """Translate an absolute AceTree plane to this stack's local Z index."""

        converter = getattr(self.app, "stack_z_from_plane", None)
        if callable(converter):
            return float(converter(plane))
        config = getattr(self.app.manager, "config", None)
        plane_start = getattr(config, "plane_start", 1)
        return float(plane) - float(plane_start)

    def _on_click(self, layer, event):
        """Handle click on 3D Points — left-click toggles cell label."""
        if event.type != "mouse_press":
            return

        button = event.button
        idx = layer.get_value(event.position, world=True)
        anchor = None
        if idx is not None and isinstance(idx, (int, np.integer)):
            anchor = point_anchor(layer, int(idx))
            if anchor is None:
                timepoint = self.view_time
                nuclei = self.app.manager.alive_nuclei_at(timepoint)
                if 0 <= idx < len(nuclei):
                    anchor = (timepoint, nuclei[int(idx)].index)

        timepoint = self.view_time
        change_counter = _document_change_counter(self.app)
        press_pointer = pointer_position(event)
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = dragged or passed_drag_threshold(event, press_pointer)
            yield
        dragged = dragged or passed_drag_threshold(event, press_pointer)
        if dragged or button != 1 or anchor is None:
            return

        QTimer.singleShot(
            0,
            lambda: self._apply_deferred_label_click(
                layer=layer,
                anchor=anchor,
                timepoint=timepoint,
                change_counter=change_counter,
            ),
        )

    def _apply_deferred_label_click(
        self,
        *,
        layer,
        anchor: tuple[int, int],
        timepoint: int,
        change_counter: int | None,
    ) -> None:
        """Toggle one stable label after the 3D camera drag has ended."""

        if (
            self._points_layer is not layer
            or self.view_time != timepoint
            or anchor[0] != timepoint
            or _document_change_counter(self.app) != change_counter
        ):
            return
        resolver = getattr(self.app, "_nucleus_at_anchor", None)
        nuc = resolver(anchor) if callable(resolver) else None
        if nuc is None or not nuc.is_alive:
            return
        name = nuc.effective_name or f"Nuc{nuc.index}"
        if name in self._shown_labels:
            self._shown_labels.discard(name)
        else:
            self._shown_labels.add(name)
        self._update_label_display()

    def _update_trail(self, timepoint: int | None = None) -> None:
        """Update 3D ghost trail for the selected cell."""
        if timepoint is None:
            timepoint = self.view_time
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

        selection_resolver = getattr(self.app, "get_selected_cell", None)
        cell = (
            selection_resolver()
            if callable(selection_resolver)
            else self.app.manager.get_cell(cell_name)
        )
        if cell is None:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        trail_len = vi.trail_length
        start = max(cell.start_time, timepoint - trail_len)

        coords = []
        sizes = []
        colors = []

        for t in range(start, timepoint):
            nuc = cell.get_nucleus_at(t)
            if nuc is None:
                continue
            age = timepoint - t
            alpha = max(0.15, 0.6 * (1.0 - age / (trail_len + 1)))
            coords.append([self._stack_z_from_plane(nuc.z), nuc.y, nuc.x])
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

    # ── Tracking proposal preview ──

    def set_tracking_preview_state(
        self,
        preview: ExpandedTrackingPreview | None,
        calibration: Calibration | None,
        *,
        visible: bool,
        stale: bool,
        highlight: str | None,
    ) -> None:
        """Receive the main viewer's immutable proposal presentation state."""

        self._tracking_preview = preview
        self._tracking_preview_calibration = calibration
        self._tracking_preview_visible = bool(visible)
        self._tracking_preview_stale = bool(stale)
        self._tracking_preview_highlight = highlight
        # A new run, stale flag, or highlight can change while time is fixed;
        # update only draft layers so review does not reread image stacks.
        self._update_tracking_preview()

    def clear_tracking_preview(self) -> None:
        """Clear proposal visuals while leaving curated points untouched."""

        self.set_tracking_preview_state(
            None,
            None,
            visible=False,
            stale=False,
            highlight=None,
        )

    def set_detector_preview_state(
        self,
        preview: ExpandedTrackingPreview | None,
        calibration: Calibration | None,
        *,
        visible: bool,
    ) -> None:
        """Receive the main workbench's non-committable detector test."""

        self._detector_preview = preview
        self._detector_preview_calibration = calibration
        self._detector_preview_visible = bool(visible)
        self._update_detector_preview()

    def clear_detector_preview(self) -> None:
        self.set_detector_preview_state(None, None, visible=False)

    def set_tracking_preview_visible(self, visible: bool) -> None:
        self._tracking_preview_visible = bool(visible)
        self.refresh(force=True)

    def set_tracking_preview_stale(self, stale: bool) -> None:
        self._tracking_preview_stale = bool(stale)
        self.refresh(force=True)

    def highlight_tracking_preview(self, preview_id: str | None) -> None:
        self._tracking_preview_highlight = preview_id
        self._update_tracking_preview()

    def center_tracking_preview(self, preview_id: str) -> bool:
        """Center this 3D camera without enabling main-view time sync."""

        preview = self._tracking_preview
        calibration = self._tracking_preview_calibration
        if preview is None or calibration is None or self._viewer is None:
            return False
        spot = preview.by_id.get(preview_id)
        if spot is None:
            return False

        if self._chk_sync.isChecked():
            self._local_time = int(spot.frame)
            self.app.set_time(spot.frame)
        else:
            self._local_time = int(spot.frame)
            self._set_time_controls(spot.frame)
        self.refresh(force=True)

        x_px, y_px, z_plane = calibration.physical_to_pixel(
            spot.x_um,
            spot.y_um,
            spot.z_um,
        )
        z_scale = calibration.z_um / calibration.xy_um
        try:
            self._viewer.camera.center = (
                (z_plane - calibration.plane_start) * z_scale,
                y_px,
                x_px,
            )
        except Exception:
            logger.debug("Could not center detached 3D preview", exc_info=True)
        return True

    def _update_tracking_preview(self) -> None:
        """Render the current local frame as read-only 3D Points and paths."""

        self._forget_removed_tracking_preview_layers()
        points_layer = self._tracking_preview_points_layer
        paths_layer = self._tracking_preview_paths_layer
        if points_layer is not None:
            points_layer.data = np.empty((0, 3))
            points_layer.visible = False
            self._make_preview_layer_read_only(points_layer)
        if paths_layer is not None:
            paths_layer.data = []
            paths_layer.visible = False
            self._make_preview_layer_read_only(paths_layer)

        preview = self._tracking_preview
        calibration = self._tracking_preview_calibration
        if (
            self._viewer is None
            or preview is None
            or calibration is None
            or not self._tracking_preview_visible
        ):
            return

        self._ensure_tracking_preview_layers(calibration)
        points_layer = self._tracking_preview_points_layer
        paths_layer = self._tracking_preview_paths_layer
        if points_layer is None or paths_layer is None:
            return
        points_layer.visible = True
        paths_layer.visible = True

        coords = []
        sizes = []
        face_colors = []
        border_colors = []
        symbols = []
        ids = []
        kinds = []
        current_time = self.view_time
        for spot in _tracking_review_spots(preview):
            if spot.frame != current_time or spot.kind == "seed":
                continue
            x_px, y_px, z_plane = calibration.physical_to_pixel(
                spot.x_um,
                spot.y_um,
                spot.z_um,
            )
            coords.append([z_plane - calibration.plane_start, y_px, x_px])
            diameter = max(1.0, 2.0 * spot.radius_um / calibration.xy_um)
            if spot.preview_id == self._tracking_preview_highlight:
                diameter *= 1.25
            sizes.append(diameter)
            color, symbol = self._tracking_preview_style(spot)
            border_colors.append(color)
            face_colors.append([color[0], color[1], color[2], min(0.35, color[3])])
            symbols.append(symbol)
            ids.append(spot.preview_id)
            kinds.append(spot.kind)

        points_layer.data = (
            np.asarray(coords, dtype=float) if coords else np.empty((0, 3))
        )
        points_layer.size = np.asarray(sizes, dtype=float)
        if coords:
            points_layer.face_color = np.asarray(face_colors, dtype=float)
            points_layer.border_color = np.asarray(border_colors, dtype=float)
            try:
                points_layer.symbol = np.asarray(symbols, dtype=object)
            except Exception:
                points_layer.symbol = "ring"
        try:
            points_layer.features = {"preview_id": ids, "kind": kinds}
        except Exception:
            pass
        self._make_preview_layer_read_only(points_layer)

        by_id = preview.by_id
        paths = []
        colors = []
        widths = []
        for link in preview.links:
            target = by_id[link.target_id]
            if target.frame != current_time or target.kind == "seed":
                continue
            source = by_id[link.source_id]
            sx, sy, sz = calibration.physical_to_pixel(
                source.x_um,
                source.y_um,
                source.z_um,
            )
            tx, ty, tz = calibration.physical_to_pixel(
                target.x_um,
                target.y_um,
                target.z_um,
            )
            paths.append(
                np.array(
                    [
                        [sz - calibration.plane_start, sy, sx],
                        [tz - calibration.plane_start, ty, tx],
                    ],
                    dtype=float,
                )
            )
            if self._tracking_preview_stale:
                colors.append([0.95, 0.65, 0.2, 0.65])
            elif link.kind == "gap":
                colors.append([1.0, 0.72, 0.2, 0.85])
            else:
                colors.append([0.0, 0.9, 1.0, 0.8])
            widths.append(
                3.0 if target.preview_id == self._tracking_preview_highlight else 2.0
            )
        search_region = getattr(preview, "search_region", None)
        if search_region is not None and search_region.frame == current_time:
            from .viewer_integration import _search_region_paths_3d

            search_paths = _search_region_paths_3d(search_region, calibration)
            paths.extend(search_paths)
            search_color = (
                [0.95, 0.65, 0.2, 0.65]
                if self._tracking_preview_stale
                else [1.0, 0.35, 0.75, 0.7]
            )
            colors.extend([search_color] * len(search_paths))
            widths.extend([1.25] * len(search_paths))
        paths_layer.data = []
        if paths:
            try:
                paths_layer.add(
                    paths,
                    shape_type="path",
                    edge_color=colors,
                    edge_width=widths,
                )
            except Exception as exc:
                logger.debug("Error drawing detached 3D tracking paths: %s", exc)
        self._make_preview_layer_read_only(paths_layer)

    def _update_detector_preview(self) -> None:
        """Render detector-test rings without touching draft or curated layers."""

        self._forget_removed_tracking_preview_layers()
        layer = self._detector_preview_points_layer
        if layer is not None:
            layer.data = np.empty((0, 3))
            layer.visible = False
            self._make_preview_layer_read_only(layer)

        preview = self._detector_preview
        calibration = self._detector_preview_calibration
        if (
            self._viewer is None
            or preview is None
            or calibration is None
            or not self._detector_preview_visible
        ):
            return
        self._ensure_detector_preview_layer(calibration)
        layer = self._detector_preview_points_layer
        if layer is None:
            return

        coords = []
        sizes = []
        ids = []
        for spot in preview.spots:
            if spot.frame != self.view_time:
                continue
            x_px, y_px, z_plane = calibration.physical_to_pixel(
                spot.x_um,
                spot.y_um,
                spot.z_um,
            )
            coords.append([z_plane - calibration.plane_start, y_px, x_px])
            sizes.append(max(1.0, 2.0 * spot.radius_um / calibration.xy_um))
            ids.append(spot.preview_id)
        layer.data = np.asarray(coords, dtype=float) if coords else np.empty((0, 3))
        layer.size = np.asarray(sizes, dtype=float)
        if coords:
            colors = np.tile(
                np.asarray([[0.72, 0.47, 1.0, 0.95]]),
                (len(coords), 1),
            )
            layer.face_color = np.column_stack(
                (colors[:, :3], np.full(len(coords), 0.2))
            )
            layer.border_color = colors
            try:
                layer.symbol = np.asarray(["ring"] * len(coords), dtype=object)
            except Exception:
                layer.symbol = "ring"
        try:
            layer.features = {
                "preview_id": ids,
                "kind": ["detector_test"] * len(ids),
            }
        except Exception:
            pass
        layer.visible = True
        self._make_preview_layer_read_only(layer)

    def _ensure_detector_preview_layer(self, calibration: Calibration) -> None:
        viewer = self._viewer
        if viewer is None:
            return
        self._forget_removed_tracking_preview_layers()
        try:
            previous_active = viewer.layers.selection.active
        except Exception:
            previous_active = None
        scale = (calibration.z_um / calibration.xy_um, 1.0, 1.0)
        if self._detector_preview_points_layer is None:
            kwargs = dict(
                size=np.empty((0,), dtype=float),
                face_color="transparent",
                border_color="#b879ff",
                symbol="ring",
                name="Detector Test Positions 3D",
                scale=scale,
                opacity=0.95,
                visible=False,
            )
            try:
                layer = viewer.add_points(np.empty((0, 3)), editable=False, **kwargs)
            except TypeError:
                layer = viewer.add_points(np.empty((0, 3)), **kwargs)
            self._detector_preview_points_layer = layer
            self._make_preview_layer_read_only(layer)
        else:
            self._detector_preview_points_layer.scale = scale
        target = previous_active
        from .viewer_integration import _viewer_contains_layer

        if (
            target is None
            or target is self._detector_preview_points_layer
            or not _viewer_contains_layer(viewer, target)
        ):
            target = self._points_layer
        if target is not None:
            try:
                viewer.layers.selection.active = target
            except Exception:
                pass

    def _ensure_tracking_preview_layers(self, calibration: Calibration) -> None:
        viewer = self._viewer
        if viewer is None:
            return
        self._forget_removed_tracking_preview_layers()
        try:
            previous_active = viewer.layers.selection.active
        except Exception:
            previous_active = None
        scale = (calibration.z_um / calibration.xy_um, 1.0, 1.0)
        if self._tracking_preview_points_layer is None:
            kwargs = dict(
                size=np.empty((0,), dtype=float),
                face_color="transparent",
                border_color="cyan",
                symbol="ring",
                name="Tracking Draft Positions 3D",
                scale=scale,
                opacity=0.95,
                visible=False,
            )
            try:
                layer = viewer.add_points(np.empty((0, 3)), editable=False, **kwargs)
            except TypeError:
                layer = viewer.add_points(np.empty((0, 3)), **kwargs)
            self._tracking_preview_points_layer = layer
            self._make_preview_layer_read_only(layer)
        else:
            self._tracking_preview_points_layer.scale = scale

        if self._tracking_preview_paths_layer is None:
            dummy = [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])]
            kwargs = dict(
                data=dummy,
                shape_type="path",
                name="Tracking Draft Paths 3D",
                edge_color="cyan",
                edge_width=2,
                opacity=0.85,
                visible=False,
            )
            try:
                layer = viewer.add_shapes(editable=False, scale=scale, **kwargs)
            except TypeError:
                layer = viewer.add_shapes(scale=scale, **kwargs)
            layer.data = []
            self._tracking_preview_paths_layer = layer
            self._make_preview_layer_read_only(layer)
        else:
            self._tracking_preview_paths_layer.scale = scale

        from .viewer_integration import _viewer_contains_layer

        target = previous_active
        if (
            target is None
            or target is self._tracking_preview_points_layer
            or target is self._tracking_preview_paths_layer
            or target is getattr(self, "_detector_preview_points_layer", None)
            or not _viewer_contains_layer(viewer, target)
        ):
            target = self._points_layer
        if target is not None:
            try:
                viewer.layers.selection.active = target
            except Exception:
                pass

    def _forget_removed_tracking_preview_layers(self) -> None:
        """Drop references to draft layers removed from this napari viewer."""

        viewer = self._viewer
        if viewer is None:
            return
        from .viewer_integration import _viewer_contains_layer

        for attribute in (
            "_tracking_preview_points_layer",
            "_tracking_preview_paths_layer",
            "_detector_preview_points_layer",
        ):
            layer = getattr(self, attribute, None)
            if layer is not None and not _viewer_contains_layer(viewer, layer):
                setattr(self, attribute, None)

    def _tracking_preview_style(self, spot) -> tuple[list[float], str]:
        kind = getattr(spot, "kind", "detection")
        if kind == "interpolated":
            color, symbol = [1.0, 0.72, 0.2, 0.95], "diamond"
        elif kind == "candidate":
            color, symbol = [1.0, 0.35, 0.75, 0.95], "cross"
        else:
            color, symbol = [0.0, 0.9, 1.0, 0.95], "ring"
        if self._tracking_preview_stale:
            color = [0.95, 0.65, 0.2, 0.85]
        if spot.preview_id == self._tracking_preview_highlight:
            color = [1.0, 1.0, 1.0, 1.0]
        return color, symbol

    @staticmethod
    def _make_preview_layer_read_only(layer) -> None:
        try:
            layer.editable = False
        except Exception:
            pass
        try:
            layer.selected_data = set()
        except Exception:
            pass
        try:
            layer.mode = "pan_zoom"
        except Exception:
            pass

    def _set_time_controls(self, timepoint: int) -> None:
        for control in (self._time_spin, self._time_slider):
            control.blockSignals(True)
            control.setValue(int(timepoint))
            control.blockSignals(False)

    # ── Time controls ──

    def _on_time_spin(self, value: int) -> None:
        self._local_time = int(value)
        self._time_slider.blockSignals(True)
        self._time_slider.setValue(value)
        self._time_slider.blockSignals(False)
        if self._chk_sync.isChecked():
            self.app.set_time(value)
        else:
            self.refresh(force=True)

    def _on_time_slider(self, value: int) -> None:
        self._local_time = int(value)
        self._time_spin.blockSignals(True)
        self._time_spin.setValue(value)
        self._time_spin.blockSignals(False)
        if self._chk_sync.isChecked():
            self.app.set_time(value)
        else:
            self.refresh(force=True)

    def _on_sync_toggled(self, synced: bool) -> None:
        """Keep the shown frame stable while changing time ownership."""

        if synced:
            self._local_time = int(self.app.current_time)
        else:
            try:
                self._local_time = int(self._time_spin.value())
            except Exception:
                self._local_time = int(self.app.current_time)
        self.refresh(force=True)

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


def _tracking_review_spots(preview) -> tuple:
    """Include diagnostic candidates when the preview model provides them."""

    return tuple(getattr(preview, "review_spots", preview.spots))
