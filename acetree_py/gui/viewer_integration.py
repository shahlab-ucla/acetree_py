"""Viewer integration — nucleus overlay on napari image layers.

Draws nucleus circles as a napari Shapes layer with:
- Size proportional to projected diameter at the current z-plane
- Color indicating selection state (white=selected, purple=named, gray=unnamed)
- Text labels showing cell names

Uses polygon circles (like Java's EUtils.pCircle) instead of bounding-box
ellipses to ensure perfect circles regardless of viewer aspect ratio.

Also handles click-to-select: clicking near a nucleus selects it.

Ported from: org.rhwlab.image.ImageWindow.showCentroids() + showAnnotations()
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QLabel

if TYPE_CHECKING:
    from ..tracking.api import Calibration, Detection, TrackingResult
    from .app import AceTreeApp
    from .tracking_preview import ExpandedTrackingPreview

logger = logging.getLogger(__name__)

# Number of vertices in each circle polygon (matches Java's pCircle density)
CIRCLE_VERTICES = 32


class ViewerIntegration:
    """Manages nucleus overlay layers on the napari viewer.

    Creates and updates:
    - A Shapes layer for nucleus circles (polygon approximations)
    - Text labels showing cell names

    Attributes:
        app: The parent AceTreeApp.
    """

    def __init__(self, app: AceTreeApp) -> None:
        self.app = app
        self._shapes_layer = None
        # Label visibility model:
        #   _shown_labels: set of cell names whose labels are individually shown
        #   _labels_global_visible: master toggle (True = shown labels are drawn)
        self._shown_labels: set[str] = set()
        self._labels_global_visible: bool = True
        # Division line layer (Feature 3)
        self._division_line_layer = None
        # Ghost trail layer — shows past positions of selected cell
        self._trails_layer = None
        self._trails_visible: bool = False
        self._trail_length: int = 10  # how many past timepoints to show
        # Non-destructive Auto Forward proposal layers.  These are kept
        # separate from ``Nuclei`` so draft points cannot be selected, moved,
        # or deleted as if they were curated records.
        self._tracking_preview_spots_layer = None
        self._tracking_preview_links_layer = None
        self._tracking_preview_3d_spots_layer = None
        self._tracking_preview_3d_links_layer = None
        self._tracking_preview: ExpandedTrackingPreview | None = None
        self._tracking_preview_calibration: Calibration | None = None
        self._tracking_preview_visible: bool = False
        self._tracking_preview_stale: bool = False
        self._tracking_preview_highlight: str | None = None
        # A current-frame detector test is not an accept-capable draft. Keep
        # it in dedicated read-only layers so it can never be confused with
        # the positions and links represented by ``_tracking_preview``.
        self._detector_preview_spots_layer = None
        self._detector_preview_3d_spots_layer = None
        self._detector_preview: ExpandedTrackingPreview | None = None
        self._detector_preview_calibration: Calibration | None = None
        self._detector_preview_visible: bool = False
        # Hover tooltip for cell info
        self._tooltip: QLabel | None = None
        self._tooltip_timer: QTimer | None = None
        self._last_hover_name: str | None = None
        self._hover_delay_ms: int = 300  # ms before tooltip appears

    def setup_layers(self) -> None:
        """Create the napari layers for nucleus overlay."""
        viewer = self.app.viewer
        if viewer is None:
            return

        # Shapes layer for nucleus circles — use polygon type
        # Start with a dummy polygon then clear it
        dummy = [np.array([[0, 0], [1, 0], [0, 1]])]
        self._shapes_layer = viewer.add_shapes(
            data=dummy,
            shape_type="polygon",
            name="Nuclei",
            edge_color="purple",
            face_color="transparent",
            edge_width=1,
            opacity=0.9,
        )
        # Clear the dummy
        self._shapes_layer.data = []

        # Connect mouse callback for click-to-select / label toggle
        self._shapes_layer.mouse_drag_callbacks.append(self._on_click)

        # Override napari's built-in Shapes-layer Delete binding — without
        # this, pressing Delete while this layer is active calls the
        # layer's ``remove_selected`` which pops shapes from layer.data
        # only, leaving ``nuclei_record`` untouched.  That makes the
        # circle vanish visually but the nucleus reappears on the next
        # display rebuild (e.g. after time scrub).  Delegate Delete to
        # the app's RemoveNucleus path instead so it's persisted + undoable.
        app = self.app

        @self._shapes_layer.bind_key("Delete", overwrite=True)
        def _delete(layer):  # noqa: ARG001 — napari binding signature
            app._delete_active_nucleus()

        @self._shapes_layer.bind_key("Backspace", overwrite=True)
        def _delete_bs(layer):  # noqa: ARG001
            app._delete_active_nucleus()

        # Division line layer (for Feature 3: daughter connection line)
        dummy_line = [np.array([[0, 0], [1, 1]])]
        self._division_line_layer = viewer.add_shapes(
            data=dummy_line,
            shape_type="line",
            name="Division Lines",
            edge_color="yellow",
            edge_width=2,
            opacity=0.8,
        )
        self._division_line_layer.data = []

        # Ghost trail layer — semi-transparent past positions + connecting line
        dummy_trail = [np.array([[0, 0], [1, 0], [0, 1]])]
        self._trails_layer = viewer.add_shapes(
            data=dummy_trail,
            shape_type="polygon",
            name="Trails",
            edge_color=[0.3, 0.8, 1.0, 0.4],
            face_color="transparent",
            edge_width=1,
            opacity=0.6,
        )
        self._trails_layer.data = []

        # Auto Forward draft layers are read-only and hidden until a proposal
        # is ready.  They never receive mouse callbacks and never become the
        # active napari layer.
        self._tracking_preview_links_layer = viewer.add_shapes(
            data=dummy_line,
            shape_type="line",
            name="Tracking Draft Links",
            edge_color="cyan",
            edge_width=2,
            opacity=0.9,
            visible=False,
        )
        self._tracking_preview_links_layer.data = []
        self._tracking_preview_links_layer.editable = False

        self._tracking_preview_spots_layer = viewer.add_shapes(
            data=dummy_trail,
            shape_type="polygon",
            name="Tracking Draft Positions",
            edge_color="cyan",
            face_color="transparent",
            edge_width=2,
            opacity=0.95,
            visible=False,
        )
        self._tracking_preview_spots_layer.data = []
        self._tracking_preview_spots_layer.editable = False

        # Set Nuclei as the active layer so clicks always reach it
        self._ensure_nuclei_active()

        # Create hover tooltip widget (parented to the napari window)
        self._setup_tooltip()

        # Connect mouse move callback for hover detection
        self._shapes_layer.mouse_move_callbacks.append(self._on_mouse_move)

    def _ensure_nuclei_active(self) -> None:
        """Keep the Nuclei shapes layer as the active layer."""
        viewer = self.app.viewer
        if viewer is not None and self._shapes_layer is not None:
            try:
                viewer.layers.selection.active = self._shapes_layer
            except Exception:
                pass

    def update_overlays(self) -> None:
        """Refresh the nucleus overlay for the current view state."""
        # Preview rendering is independent of curated nuclei.  In particular,
        # it must remain visible on frames where the nuclei record is empty.
        self._update_tracking_preview()
        # Detector rings are Z-slice intersections, so normal plane scrubbing
        # must recompute them through the same refresh path as curated nuclei.
        self._update_detector_preview()
        overlay = self.app.get_nucleus_overlay_data()

        if self._shapes_layer is None:
            return

        centers = overlay["centers"]
        radii = overlay["radii"]
        colors = overlay["colors"]
        names = overlay["names"]
        selected_idx = overlay["selected_idx"]

        if len(centers) == 0:
            self._shapes_layer.data = []
            return

        # Build polygon circle data for napari Shapes layer.
        # Using polygons instead of bounding-box ellipses ensures perfect
        # circles regardless of viewer aspect ratio or scaling.
        polygons = []
        edge_colors = []
        face_colors = []

        for i in range(len(centers)):
            cy, cx = centers[i]
            r = radii[i]

            # Skip extremely small circles (< 1 pixel radius)
            if r < 0.5:
                continue

            circle = make_circle_polygon(cx, cy, r, CIRCLE_VERTICES)
            polygons.append(circle)
            edge_colors.append(colors[i])
            face_colors.append([0, 0, 0, 0])  # Transparent fill

        if not polygons:
            self._shapes_layer.data = []
            return

        # Rebuild selected_idx after skipping tiny circles
        # (selected_idx from overlay refers to the pre-filter list)
        new_selected_idx = -1
        if selected_idx >= 0:
            # Find the selected nucleus in the filtered list
            filter_idx = 0
            for i in range(len(centers)):
                if radii[i] < 0.5:
                    continue
                if i == selected_idx:
                    new_selected_idx = filter_idx
                    break
                filter_idx += 1

        # Edge width: thin for normal, slightly thicker for selected
        edge_widths = np.full(len(polygons), 1.0)
        if new_selected_idx >= 0:
            edge_widths[new_selected_idx] = 2.0

        # Filter names to match polygons (skip tiny circles).
        # Label model: only show labels for cells in _shown_labels,
        # and only when _labels_global_visible is True.
        filtered_names = []
        for i in range(len(centers)):
            if radii[i] >= 0.5:
                name = names[i]
                if self._labels_global_visible and name in self._shown_labels:
                    filtered_names.append(name)
                else:
                    filtered_names.append("")

        try:
            # Clear and re-add shapes
            self._shapes_layer.data = []
            self._shapes_layer.add(
                polygons,
                shape_type="polygon",
                edge_color=edge_colors,
                face_color=face_colors,
                edge_width=edge_widths,
            )

            # Add text labels for named nuclei
            if filtered_names:
                self._shapes_layer.text = {
                    "string": filtered_names,
                    "color": "white",
                    "size": 8,
                    "anchor": "upper_left",
                }
        except Exception as e:
            # Shapes layer can be finicky with empty/mismatched data
            logger.debug("Error updating shapes layer: %s", e)

        # ── Feature 3: division line for active cell's daughters ──
        self._update_division_line()

        # ── Ghost trail for selected cell ──
        self._update_ghost_trail()

    def show_tracking_preview(
        self,
        proposal: TrackingResult,
        calibration: Calibration,
        *,
        visible: bool = True,
        stale: bool = False,
    ) -> None:
        """Display an immutable tracking proposal without editing the dataset."""

        from .tracking_preview import expand_tracking_preview

        previous_active = self._active_layer()
        self._tracking_preview = expand_tracking_preview(proposal)
        self._tracking_preview_calibration = calibration
        self._tracking_preview_visible = visible
        self._tracking_preview_stale = stale
        self._tracking_preview_highlight = None
        self._update_tracking_preview()
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def show_detector_preview(
        self,
        detections: tuple[Detection, ...],
        calibration: Calibration,
        *,
        visible: bool = True,
    ) -> None:
        """Display one detector test without creating a tracking proposal."""

        from .tracking_preview import expand_detector_preview

        previous_active = self._active_layer()
        self._detector_preview = expand_detector_preview(tuple(detections))
        self._detector_preview_calibration = calibration
        self._detector_preview_visible = bool(visible)
        self._update_detector_preview()
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def clear_detector_preview(self) -> None:
        """Clear transient detector-test layers in every viewer."""

        previous_active = self._active_layer()
        self._detector_preview = None
        self._detector_preview_calibration = None
        self._detector_preview_visible = False
        self._forget_removed_tracking_preview_layers()
        for layer in (
            self._detector_preview_spots_layer,
            self._detector_preview_3d_spots_layer,
        ):
            if layer is None:
                continue
            if layer is self._detector_preview_3d_spots_layer:
                layer.data = np.empty((0, 3))
            else:
                layer.data = []
            layer.visible = False
            self._make_preview_layer_read_only(layer)
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def set_detector_preview_visible(self, visible: bool) -> None:
        """Show or hide the current-frame detector test."""

        previous_active = self._active_layer()
        self._detector_preview_visible = bool(visible)
        self._update_detector_preview()
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def clear_tracking_preview(self) -> None:
        """Remove every temporary Auto Forward layer from all viewers."""

        previous_active = self._active_layer()
        self._tracking_preview = None
        self._tracking_preview_calibration = None
        self._tracking_preview_visible = False
        self._tracking_preview_stale = False
        self._tracking_preview_highlight = None
        self._forget_removed_tracking_preview_layers()
        for layer in (
            self._tracking_preview_spots_layer,
            self._tracking_preview_links_layer,
            self._tracking_preview_3d_spots_layer,
            self._tracking_preview_3d_links_layer,
        ):
            if layer is not None:
                if layer is self._tracking_preview_3d_spots_layer:
                    layer.data = np.empty((0, 3))
                else:
                    layer.data = []
                layer.visible = False
                self._make_preview_layer_read_only(layer)
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def set_tracking_preview_visible(self, visible: bool) -> None:
        """Show or hide a draft without discarding its review state."""

        previous_active = self._active_layer()
        self._tracking_preview_visible = bool(visible)
        self._update_tracking_preview()
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def set_tracking_preview_stale(self, stale: bool) -> None:
        """Update stale styling without rebuilding or accepting a draft."""

        previous_active = self._active_layer()
        self._tracking_preview_stale = bool(stale)
        self._update_tracking_preview()
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def highlight_tracking_preview(self, preview_id: str | None) -> None:
        """Emphasize the table-selected draft position in the image overlay."""

        previous_active = self._active_layer()
        self._tracking_preview_highlight = preview_id
        self._update_tracking_preview()
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def refresh_tracking_preview(self) -> None:
        """Refresh draft layers after a main-view time or 2D/3D transition.

        This is intentionally a public, side-effect-free display hook.  The app
        can call it after changing time or ``dims.ndisplay`` without teaching
        the transition code anything about proposal layer implementation.
        """

        previous_active = self._active_layer()
        self._update_tracking_preview()
        self._update_detector_preview()
        self._restore_editing_layer(previous_active)
        self._notify_detached_tracking_preview()

    def sync_tracking_preview_window(self, window) -> None:
        """Copy the current proposal display state into one detached viewer."""

        setter = getattr(window, "set_tracking_preview_state", None)
        if setter is None:
            return
        setter(
            self._tracking_preview,
            self._tracking_preview_calibration,
            visible=self._tracking_preview_visible,
            stale=self._tracking_preview_stale,
            highlight=self._tracking_preview_highlight,
        )
        detector_setter = getattr(window, "set_detector_preview_state", None)
        if detector_setter is not None:
            detector_setter(
                self._detector_preview,
                self._detector_preview_calibration,
                visible=self._detector_preview_visible,
            )

    def center_tracking_preview(self, preview_id: str) -> bool:
        """Navigate and center the main camera on a draft point.

        In 2D this selects the point's image plane and centers in Y/X.  In 3D
        it leaves volume mode intact and centers in calibrated Z/Y/X world
        coordinates.  ``False`` means the draft point no longer exists.
        """

        preview = self._tracking_preview
        calibration = self._tracking_preview_calibration
        viewer = self.app.viewer
        if preview is None or calibration is None or viewer is None:
            return False
        spot = preview.by_id.get(preview_id)
        if spot is None:
            return False

        self.app.set_time(spot.frame)
        x_px, y_px, z_plane = calibration.physical_to_pixel(
            spot.x_um,
            spot.y_um,
            spot.z_um,
        )
        is_3d = bool(getattr(self.app, "_3d_mode", False))
        if is_3d:
            z_scale = calibration.z_um / calibration.xy_um
            center = (
                (z_plane - calibration.plane_start) * z_scale,
                y_px,
                x_px,
            )
        else:
            self.app.set_plane(round(z_plane))
            center = (y_px, x_px)
        try:
            viewer.camera.center = center
        except Exception:
            logger.debug("Could not center camera on tracking preview", exc_info=True)
        return True

    def center_tracking_position(
        self,
        x_um: float,
        y_um: float,
        z_um: float,
    ) -> bool:
        """Center the main camera on an arbitrary physical tracking location."""

        calibration = self._tracking_preview_calibration
        viewer = self.app.viewer
        if calibration is None or viewer is None:
            return False
        x_px, y_px, z_plane = calibration.physical_to_pixel(x_um, y_um, z_um)
        if bool(getattr(self.app, "_3d_mode", False)):
            z_scale = calibration.z_um / calibration.xy_um
            center = (
                (z_plane - calibration.plane_start) * z_scale,
                y_px,
                x_px,
            )
        else:
            self.app.set_plane(round(z_plane))
            center = (y_px, x_px)
        try:
            viewer.camera.center = center
        except Exception:
            logger.debug("Could not center camera on tracking location", exc_info=True)
            return False
        return True

    @property
    def has_tracking_preview(self) -> bool:
        return self._tracking_preview is not None

    def capture_image_channel_visibility(self) -> list[tuple[object, bool]]:
        """Snapshot channel visibility in the main and detached viewers."""

        return [
            (layer, bool(getattr(layer, "visible", True)))
            for group in self._image_layer_groups()
            for layer in group
        ]

    def set_detection_channel_solo(self, channel_index: int) -> None:
        """Show one detector channel consistently in every open viewer."""

        for group in self._image_layer_groups():
            for index, layer in enumerate(group):
                try:
                    layer.visible = index == int(channel_index)
                except RuntimeError:
                    pass

    @staticmethod
    def restore_image_channel_visibility(
        snapshot: list[tuple[object, bool]],
    ) -> None:
        """Restore a visibility snapshot captured for tracking review."""

        for layer, visible in snapshot:
            try:
                layer.visible = visible
            except RuntimeError:
                pass

    def _image_layer_groups(self) -> tuple[tuple, ...]:
        groups = [tuple(getattr(self.app, "_image_layers", ()))]
        for window in tuple(getattr(self.app, "_3d_windows", ())):
            groups.append(tuple(getattr(window, "_image_layers", ())))
        return tuple(groups)

    def _update_tracking_preview(self) -> None:
        """Render the current frame in the main viewer's active display mode."""

        self._forget_removed_tracking_preview_layers()
        spots_layer = self._tracking_preview_spots_layer
        links_layer = self._tracking_preview_links_layer
        if spots_layer is not None:
            spots_layer.data = []
            spots_layer.visible = False
            self._make_preview_layer_read_only(spots_layer)
        if links_layer is not None:
            links_layer.data = []
            links_layer.visible = False
            self._make_preview_layer_read_only(links_layer)
        if self._tracking_preview_3d_spots_layer is not None:
            self._tracking_preview_3d_spots_layer.data = np.empty((0, 3))
            self._tracking_preview_3d_spots_layer.visible = False
            self._make_preview_layer_read_only(self._tracking_preview_3d_spots_layer)
        if self._tracking_preview_3d_links_layer is not None:
            self._tracking_preview_3d_links_layer.data = []
            self._tracking_preview_3d_links_layer.visible = False
            self._make_preview_layer_read_only(self._tracking_preview_3d_links_layer)

        preview = self._tracking_preview
        calibration = self._tracking_preview_calibration
        visible = bool(preview is not None and calibration is not None)
        visible = visible and self._tracking_preview_visible
        if not visible:
            return

        if bool(getattr(self.app, "_3d_mode", False)):
            self._ensure_tracking_preview_3d_layers(calibration)
            self._update_tracking_preview_3d(preview, calibration)
            return

        self._ensure_tracking_preview_2d_layers()
        spots_layer = self._tracking_preview_spots_layer
        links_layer = self._tracking_preview_links_layer
        if spots_layer is None or links_layer is None:
            return
        spots_layer.visible = True
        links_layer.visible = True
        self._update_tracking_preview_2d(preview, calibration)

    def _update_detector_preview(self) -> None:
        """Render the transient detector test in its own native napari layer."""

        self._forget_removed_tracking_preview_layers()
        if self._detector_preview_spots_layer is not None:
            self._detector_preview_spots_layer.data = []
            self._detector_preview_spots_layer.visible = False
            self._make_preview_layer_read_only(self._detector_preview_spots_layer)
        if self._detector_preview_3d_spots_layer is not None:
            self._detector_preview_3d_spots_layer.data = np.empty((0, 3))
            self._detector_preview_3d_spots_layer.visible = False
            self._make_preview_layer_read_only(self._detector_preview_3d_spots_layer)

        preview = self._detector_preview
        calibration = self._detector_preview_calibration
        if (
            preview is None
            or calibration is None
            or not self._detector_preview_visible
        ):
            return
        if bool(getattr(self.app, "_3d_mode", False)):
            self._ensure_detector_preview_3d_layer(calibration)
            self._update_detector_preview_3d(preview, calibration)
        else:
            self._ensure_detector_preview_2d_layer()
            self._update_detector_preview_2d(preview, calibration)

    def _ensure_detector_preview_2d_layer(self) -> None:
        viewer = self.app.viewer
        if viewer is None:
            return
        previous_active = self._active_layer()
        if self._detector_preview_spots_layer is None:
            layer = viewer.add_shapes(
                data=[np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
                shape_type="polygon",
                name="Detector Test Positions",
                edge_color="#b879ff",
                face_color="transparent",
                edge_width=2.25,
                opacity=0.95,
                visible=False,
            )
            layer.data = []
            self._detector_preview_spots_layer = layer
            self._make_preview_layer_read_only(layer)
        self._restore_editing_layer(previous_active)

    def _update_detector_preview_2d(self, preview, calibration) -> None:
        layer = self._detector_preview_spots_layer
        if layer is None:
            return
        current_time = self.app.current_time
        current_z_um = (
            float(self.app.current_plane) - calibration.plane_start
        ) * calibration.z_um
        polygons = []
        for spot in preview.spots:
            if spot.frame != current_time:
                continue
            dz_um = abs(spot.z_um - current_z_um)
            if dz_um >= spot.radius_um:
                continue
            radius_px = math.sqrt(max(0.0, spot.radius_um**2 - dz_um**2))
            radius_px /= calibration.xy_um
            if radius_px < 0.5:
                continue
            x_px, y_px, _ = calibration.physical_to_pixel(
                spot.x_um,
                spot.y_um,
                spot.z_um,
            )
            polygons.append(make_circle_polygon(x_px, y_px, radius_px, CIRCLE_VERTICES))
        layer.visible = True
        if polygons:
            try:
                layer.add(
                    polygons,
                    shape_type="polygon",
                    edge_color=[[0.72, 0.47, 1.0, 0.95]] * len(polygons),
                    face_color=[[0.0, 0.0, 0.0, 0.0]] * len(polygons),
                    edge_width=[2.25] * len(polygons),
                )
            except Exception as exc:
                logger.debug("Error drawing detector test positions: %s", exc)
        self._make_preview_layer_read_only(layer)

    def _ensure_detector_preview_3d_layer(self, calibration) -> None:
        viewer = self.app.viewer
        if viewer is None:
            return
        previous_active = self._active_layer()
        scale = (calibration.z_um / calibration.xy_um, 1.0, 1.0)
        if self._detector_preview_3d_spots_layer is None:
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
            self._detector_preview_3d_spots_layer = layer
            self._make_preview_layer_read_only(layer)
        else:
            self._detector_preview_3d_spots_layer.scale = scale
        self._restore_editing_layer(previous_active)

    def _update_detector_preview_3d(self, preview, calibration) -> None:
        layer = self._detector_preview_3d_spots_layer
        if layer is None:
            return
        coords = []
        sizes = []
        ids = []
        for spot in preview.spots:
            if spot.frame != self.app.current_time:
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
            colors = np.tile(np.asarray([[0.72, 0.47, 1.0, 0.95]]), (len(coords), 1))
            layer.face_color = np.column_stack((colors[:, :3], np.full(len(coords), 0.2)))
            layer.border_color = colors
            try:
                layer.symbol = np.asarray(["ring"] * len(coords), dtype=object)
            except Exception:
                layer.symbol = "ring"
        try:
            layer.features = {"preview_id": ids, "kind": ["detector_test"] * len(ids)}
        except Exception:
            pass
        layer.visible = True
        self._make_preview_layer_read_only(layer)

    def _ensure_tracking_preview_2d_layers(self) -> None:
        """Recreate read-only 2D draft layers if a user removed them."""

        viewer = self.app.viewer
        if viewer is None:
            return
        previous_active = self._active_layer()
        if self._tracking_preview_links_layer is None:
            layer = viewer.add_shapes(
                data=[np.array([[0.0, 0.0], [1.0, 1.0]])],
                shape_type="line",
                name="Tracking Draft Links",
                edge_color="cyan",
                edge_width=2,
                opacity=0.9,
                visible=False,
            )
            layer.data = []
            self._tracking_preview_links_layer = layer
            self._make_preview_layer_read_only(layer)
        if self._tracking_preview_spots_layer is None:
            layer = viewer.add_shapes(
                data=[np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
                shape_type="polygon",
                name="Tracking Draft Positions",
                edge_color="cyan",
                face_color="transparent",
                edge_width=2,
                opacity=0.95,
                visible=False,
            )
            layer.data = []
            self._tracking_preview_spots_layer = layer
            self._make_preview_layer_read_only(layer)
        self._restore_editing_layer(previous_active)

    def _update_tracking_preview_2d(self, preview, calibration) -> None:
        """Render projected, read-only proposal shapes on the current Z plane."""

        spots_layer = self._tracking_preview_spots_layer
        links_layer = self._tracking_preview_links_layer
        if spots_layer is None or links_layer is None:
            return

        current_time = self.app.current_time
        current_z_um = (
            float(self.app.current_plane) - calibration.plane_start
        ) * calibration.z_um
        polygons = []
        edge_colors = []
        face_colors = []
        edge_widths = []
        visible_ids: set[str] = set()

        for spot in _tracking_review_spots(preview):
            if spot.frame != current_time or spot.kind == "seed":
                continue
            dz_um = abs(spot.z_um - current_z_um)
            if dz_um >= spot.radius_um:
                continue
            radius_px = math.sqrt(max(0.0, spot.radius_um**2 - dz_um**2))
            radius_px /= calibration.xy_um
            if radius_px < 0.5:
                continue
            x_px, y_px, _ = calibration.physical_to_pixel(
                spot.x_um,
                spot.y_um,
                spot.z_um,
            )
            polygons.append(_preview_polygon_2d(spot.kind, x_px, y_px, radius_px))
            color, width, _symbol = self._preview_spot_style(spot)
            edge_colors.append(color)
            face_colors.append([0.0, 0.0, 0.0, 0.0])
            edge_widths.append(width)
            visible_ids.add(spot.preview_id)

        search_region = getattr(preview, "search_region", None)
        search_crosshair: list[np.ndarray] = []
        if search_region is not None and search_region.frame == current_time:
            dz_um = abs(search_region.z_um - current_z_um)
            if dz_um < search_region.radius_um:
                radius_px = math.sqrt(
                    max(0.0, search_region.radius_um**2 - dz_um**2)
                ) / calibration.xy_um
                x_px, y_px, _ = calibration.physical_to_pixel(
                    search_region.x_um,
                    search_region.y_um,
                    search_region.z_um,
                )
                polygons.append(
                    make_circle_polygon(x_px, y_px, radius_px, CIRCLE_VERTICES)
                )
                search_color = (
                    [0.95, 0.65, 0.2, 0.75]
                    if self._tracking_preview_stale
                    else [1.0, 0.35, 0.75, 0.75]
                )
                edge_colors.append(search_color)
                face_colors.append([0.0, 0.0, 0.0, 0.0])
                edge_widths.append(1.25)
                arm = max(3.0, min(10.0, radius_px * 0.25))
                search_crosshair.extend(
                    (
                        np.array([[y_px, x_px - arm], [y_px, x_px + arm]]),
                        np.array([[y_px - arm, x_px], [y_px + arm, x_px]]),
                    )
                )

        if polygons:
            try:
                spots_layer.add(
                    polygons,
                    shape_type="polygon",
                    edge_color=edge_colors,
                    face_color=face_colors,
                    edge_width=edge_widths,
                )
            except Exception as exc:
                logger.debug("Error drawing tracking draft positions: %s", exc)

        by_id = preview.by_id
        lines = []
        line_colors = []
        line_widths = []
        for link in preview.links:
            target = by_id[link.target_id]
            if target.frame != current_time or target.preview_id not in visible_ids:
                continue
            source = by_id[link.source_id]
            source_x, source_y, _ = calibration.physical_to_pixel(
                source.x_um,
                source.y_um,
                source.z_um,
            )
            target_x, target_y, _ = calibration.physical_to_pixel(
                target.x_um,
                target.y_um,
                target.z_um,
            )
            lines.append(np.array([[source_y, source_x], [target_y, target_x]]))
            if self._tracking_preview_stale:
                line_colors.append([0.95, 0.65, 0.2, 0.65])
            elif link.kind == "gap":
                line_colors.append([1.0, 0.72, 0.2, 0.85])
            else:
                line_colors.append([0.0, 0.9, 1.0, 0.8])
            line_widths.append(
                2.5
                if target.preview_id == self._tracking_preview_highlight
                else 2.0
            )

        if search_crosshair:
            search_color = (
                [0.95, 0.65, 0.2, 0.75]
                if self._tracking_preview_stale
                else [1.0, 0.35, 0.75, 0.75]
            )
            lines.extend(search_crosshair)
            line_colors.extend([search_color] * len(search_crosshair))
            line_widths.extend([1.25] * len(search_crosshair))

        if lines:
            try:
                links_layer.add(
                    lines,
                    shape_type="line",
                    edge_color=line_colors,
                    edge_width=line_widths,
                )
            except Exception as exc:
                logger.debug("Error drawing tracking draft links: %s", exc)
        # napari Shapes may re-enter selection mode when data is appended;
        # enforce the proposal boundary after every rebuild.
        self._make_preview_layer_read_only(spots_layer)
        self._make_preview_layer_read_only(links_layer)

    def _ensure_tracking_preview_3d_layers(self, calibration) -> None:
        """Lazily add napari-native Points and Shapes proposal layers."""

        viewer = self.app.viewer
        if viewer is None:
            return
        self._forget_removed_tracking_preview_layers()
        previous_active = self._active_layer()
        scale = (calibration.z_um / calibration.xy_um, 1.0, 1.0)
        if self._tracking_preview_3d_spots_layer is None:
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
            self._tracking_preview_3d_spots_layer = layer
            self._make_preview_layer_read_only(layer)
        else:
            self._tracking_preview_3d_spots_layer.scale = scale

        if self._tracking_preview_3d_links_layer is None:
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
            self._tracking_preview_3d_links_layer = layer
            self._make_preview_layer_read_only(layer)
        else:
            self._tracking_preview_3d_links_layer.scale = scale
        self._restore_editing_layer(previous_active)

    def _update_tracking_preview_3d(self, preview, calibration) -> None:
        """Render proposal points and movement paths for a 3D stack."""

        points_layer = self._tracking_preview_3d_spots_layer
        paths_layer = self._tracking_preview_3d_links_layer
        if points_layer is None or paths_layer is None:
            return
        points_layer.visible = True
        paths_layer.visible = True
        current_time = self.app.current_time

        coords = []
        sizes = []
        face_colors = []
        border_colors = []
        symbols = []
        ids = []
        kinds = []
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
            color, _width, symbol = self._preview_spot_style(spot)
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
                # Older napari releases accept one symbol only.  Keeping the
                # distinct colors still makes the fallback unambiguous.
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
                logger.debug("Error drawing 3D tracking draft paths: %s", exc)
        self._make_preview_layer_read_only(paths_layer)

    def _preview_spot_style(self, spot) -> tuple[list[float], float, str]:
        """Return redundant color/weight/symbol styling for one proposal spot."""

        kind = getattr(spot, "kind", "detection")
        if kind == "interpolated":
            color, width, symbol = [1.0, 0.72, 0.2, 0.95], 2.0, "diamond"
        elif kind == "candidate":
            color, width, symbol = [1.0, 0.35, 0.75, 0.95], 2.0, "cross"
        else:
            color, width, symbol = [0.0, 0.9, 1.0, 0.95], 2.5, "ring"
        if self._tracking_preview_stale:
            color, width = [0.95, 0.65, 0.2, 0.85], 2.0
        if spot.preview_id == self._tracking_preview_highlight:
            color, width = [1.0, 1.0, 1.0, 1.0], 3.5
        return color, width, symbol

    @staticmethod
    def _make_preview_layer_read_only(layer) -> None:
        """Lock a proposal layer and clear any accidental napari selection."""

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

    def _active_layer(self):
        try:
            return self.app.viewer.layers.selection.active
        except Exception:
            return None

    def _restore_editing_layer(self, previous_active) -> None:
        """Restore selection after napari makes a newly-added preview active."""

        preview_layers = tuple(
            layer
            for layer in (
                self._tracking_preview_spots_layer,
                self._tracking_preview_links_layer,
                self._tracking_preview_3d_spots_layer,
                self._tracking_preview_3d_links_layer,
                self._detector_preview_spots_layer,
                self._detector_preview_3d_spots_layer,
            )
            if layer is not None
        )
        target = previous_active
        if (
            target is None
            or any(target is layer for layer in preview_layers)
            or not _viewer_contains_layer(self.app.viewer, target)
        ):
            if bool(getattr(self.app, "_3d_mode", False)):
                target = getattr(self.app, "_points_layer", None)
            else:
                target = self._shapes_layer
        if target is None:
            return
        try:
            self.app.viewer.layers.selection.active = target
        except Exception:
            pass

    def _forget_removed_tracking_preview_layers(self) -> None:
        """Drop references to preview layers no longer owned by napari."""

        viewer = self.app.viewer
        if viewer is None:
            return
        for attribute in (
            "_tracking_preview_spots_layer",
            "_tracking_preview_links_layer",
            "_tracking_preview_3d_spots_layer",
            "_tracking_preview_3d_links_layer",
            "_detector_preview_spots_layer",
            "_detector_preview_3d_spots_layer",
        ):
            layer = getattr(self, attribute)
            if layer is not None and not _viewer_contains_layer(viewer, layer):
                setattr(self, attribute, None)

    def _notify_detached_tracking_preview(self) -> None:
        """Propagate proposal state, including same-frame visual changes."""

        for window in tuple(getattr(self.app, "_3d_windows", ())):
            try:
                self.sync_tracking_preview_window(window)
            except RuntimeError:
                pass

    def _update_division_line(self) -> None:
        """Draw a line connecting daughter cells for one frame after division.

        Checks both the selected cell and its parent, because tracking
        auto-follows to a daughter when time advances past a division —
        so by the time this runs, current_cell_name is typically the
        daughter, not the parent that divided.

        The line disappears on any z-plane change, time change, or selection
        change (because update_overlays is called in all those cases and
        re-evaluates the condition).
        """
        if self._division_line_layer is None:
            return

        self._division_line_layer.data = []

        cell_name = self.app.current_cell_name
        if not cell_name:
            return

        cell = self.app.manager.get_cell(cell_name)
        if cell is None:
            return

        cur_time = self.app.current_time

        # Find the dividing cell: either the selected cell itself,
        # or its parent (if tracking just followed into a daughter).
        dividing = None
        if cur_time == cell.end_time + 1 and len(cell.children) == 2:
            dividing = cell
        elif cell.parent is not None and cur_time == cell.parent.end_time + 1 \
                and len(cell.parent.children) == 2:
            dividing = cell.parent

        if dividing is None:
            return

        child_a, child_b = dividing.children
        nuc_a = child_a.get_nucleus_at(cur_time)
        nuc_b = child_b.get_nucleus_at(cur_time)

        if nuc_a is None or nuc_b is None:
            return

        # Draw line in (row, col) = (y, x) coordinate system
        line = np.array([[nuc_a.y, nuc_a.x], [nuc_b.y, nuc_b.x]])

        try:
            self._division_line_layer.add(
                [line],
                shape_type="line",
                edge_color="yellow",
                edge_width=2,
            )
        except Exception as e:
            logger.debug("Error drawing division line: %s", e)

    def _update_ghost_trail(self) -> None:
        """Draw ghost trail for the selected cell's past positions.

        Shows semi-transparent circles at past timepoints connected by
        a thin line, giving a visual trace of cell movement over time.
        """
        if self._trails_layer is None:
            return

        self._trails_layer.data = []

        if not self._trails_visible:
            return

        cell_name = self.app.current_cell_name
        if not cell_name:
            return

        cell = self.app.manager.get_cell(cell_name)
        if cell is None:
            return

        cur_time = self.app.current_time
        start = max(cell.start_time, cur_time - self._trail_length)

        # Collect past positions
        trail_shapes = []
        trail_types = []
        trail_edge_colors = []
        trail_face_colors = []
        trail_widths = []
        path_points = []

        for t in range(start, cur_time):
            nuc = cell.get_nucleus_at(t)
            if nuc is None:
                continue

            # Project diameter at current viewing plane
            diam = self.app.manager.nucleus_diameter(nuc, self.app.current_plane)
            if diam <= 0:
                # Still include in path even if not visible on this z-plane
                path_points.append([nuc.y, nuc.x])
                continue

            # Fade alpha based on age: older = more transparent
            age = cur_time - t
            alpha = max(0.15, 0.6 * (1.0 - age / (self._trail_length + 1)))

            radius = diam / 2.0
            circle = make_circle_polygon(nuc.x, nuc.y, radius, CIRCLE_VERTICES)
            trail_shapes.append(circle)
            trail_types.append("polygon")
            trail_edge_colors.append([0.3, 0.8, 1.0, alpha])
            trail_face_colors.append([0, 0, 0, 0])
            trail_widths.append(1.0)

            path_points.append([nuc.y, nuc.x])

        # Add connecting path line if we have 2+ points
        if len(path_points) >= 2:
            trail_shapes.append(np.array(path_points))
            trail_types.append("path")
            trail_edge_colors.append([0.3, 0.8, 1.0, 0.35])
            trail_face_colors.append([0, 0, 0, 0])
            trail_widths.append(1.5)

        if not trail_shapes:
            return

        try:
            self._trails_layer.add(
                trail_shapes,
                shape_type=trail_types,
                edge_color=trail_edge_colors,
                face_color=trail_face_colors,
                edge_width=trail_widths,
            )
        except Exception as e:
            logger.debug("Error drawing ghost trail: %s", e)

    def toggle_trails(self, visible: bool | None = None) -> None:
        """Toggle or set ghost trail visibility.

        Args:
            visible: If given, sets visibility directly. If None, toggles.
        """
        if visible is None:
            self._trails_visible = not self._trails_visible
        else:
            self._trails_visible = visible
        self.update_overlays()

    @property
    def trails_visible(self) -> bool:
        return self._trails_visible

    @property
    def trail_length(self) -> int:
        return self._trail_length

    def set_trail_length(self, length: int) -> None:
        """Set how many past timepoints the ghost trail covers."""
        self._trail_length = max(1, length)
        if self._trails_visible:
            self.update_overlays()

    def _on_click(self, layer, event):
        """Handle mouse clicks on the shapes layer.

        Left-click:  Toggle the clicked cell's label on/off.
        Right-click: Select the clicked cell and make it active (also shows label).

        This is a generator callback (yields once) so that napari properly
        finalises the drag/pan cycle after the click is handled.  Without
        the yield, napari can get stuck in pan mode after actions like
        relink confirmation that open modal dialogs.
        """
        if event.type != "mouse_press":
            return

        # Get click position in data coordinates
        coords = event.position
        if len(coords) < 2:
            return

        # napari coords are (row, col) = (y, x)
        y, x = coords[-2], coords[-1]

        # Check for relink pick mode first (consumes any click).
        # Defer the callback via QTimer so the yield happens first —
        # the callback opens a modal dialog which would block napari's
        # drag cycle finalisation and leave the canvas stuck in pan mode.
        if self.app._relink_pick_mode:
            nuc = self.app.manager.find_closest_nucleus(
                x, y, float(self.app.current_plane), self.app.current_time,
                require_hit=True, image_plane=self.app.current_plane,
            )
            if nuc is not None and self.app._relink_pick_callback is not None:
                cb = self.app._relink_pick_callback
                t = self.app.current_time
                self.app.exit_relink_pick_mode()
                QTimer.singleShot(0, lambda: cb(t, nuc))
            yield  # release drag cycle
            return

        button = event.button  # 1 = left, 2 = right

        # Check for add mode (consumes left-click)
        if self.app._add_mode and button == 1:
            self.app._handle_add_click(x, y)
            yield
            return

        # Check for placement mode (consumes right-click)
        if self.app._placement_mode and button == 2:
            self.app._handle_placement_click(x, y)
            yield
            return

        if button == 2:
            # Right-click: select cell and show its label
            self.app.select_cell_at_position(x, y)
            if self.app.current_cell_name:
                self._shown_labels.add(self.app.current_cell_name)
        else:
            # Left-click: toggle label for nearest cell without selecting
            nuc = self.app.manager.find_closest_nucleus(
                x, y, float(self.app.current_plane), self.app.current_time,
                require_hit=True, image_plane=self.app.current_plane,
            )
            if nuc is not None:
                name = nuc.effective_name or f"Nuc{nuc.index}"
                if name in self._shown_labels:
                    self._shown_labels.discard(name)
                else:
                    self._shown_labels.add(name)
                self.update_overlays()

        yield  # release drag cycle

    # ── Hover tooltip ─────────────────────────────────────────────

    def _setup_tooltip(self) -> None:
        """Create the QLabel tooltip widget for cell info on hover."""
        viewer = self.app.viewer
        if viewer is None or not hasattr(viewer, "window"):
            return

        # Parent to the napari main window so it floats above the canvas
        parent_widget = viewer.window._qt_window
        self._tooltip = QLabel(parent_widget)
        self._tooltip.setWindowFlags(
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self._tooltip.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(30, 30, 30, 220);"
            "  color: #e0e0e0;"
            "  border: 1px solid #555;"
            "  border-radius: 4px;"
            "  padding: 6px 8px;"
            "  font-family: monospace;"
            "  font-size: 11px;"
            "}"
        )
        self._tooltip.setTextFormat(Qt.PlainText)
        self._tooltip.hide()

        # Timer to debounce hover detection
        self._tooltip_timer = QTimer()
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.timeout.connect(self._show_tooltip)

    def _on_mouse_move(self, layer, event):
        """Handle mouse movement over the shapes layer for hover tooltip."""
        coords = event.position
        if len(coords) < 2:
            self._hide_tooltip()
            return

        y, x = coords[-2], coords[-1]

        # Find the nucleus under the cursor
        nuc = self.app.manager.find_closest_nucleus(
            x, y, float(self.app.current_plane), self.app.current_time,
            require_hit=True, image_plane=self.app.current_plane,
        )

        if nuc is None:
            self._hide_tooltip()
            return

        name = nuc.effective_name or f"Nuc{nuc.index}"

        # Same cell — tooltip already showing or timer already running
        if name == self._last_hover_name and self._tooltip is not None and (
            self._tooltip.isVisible() or self._tooltip_timer.isActive()
        ):
            # Update position to follow cursor
            if self._tooltip.isVisible():
                self._position_tooltip()
            return

        # New cell — restart the delay timer
        self._last_hover_name = name
        if self._tooltip_timer is not None:
            self._tooltip_timer.start(self._hover_delay_ms)

    def _show_tooltip(self) -> None:
        """Display the tooltip with cell info after the hover delay."""
        if self._tooltip is None or self._last_hover_name is None:
            return

        # Use app's existing cell info method if hovering over selected cell,
        # otherwise build a quick summary for the hovered cell
        text = self._get_hover_info(self._last_hover_name)
        if not text:
            self._hide_tooltip()
            return

        self._tooltip.setText(text)
        self._tooltip.adjustSize()
        self._position_tooltip()
        self._tooltip.show()

    def _position_tooltip(self) -> None:
        """Position the tooltip near the cursor with a small offset."""
        if self._tooltip is None:
            return
        pos = QCursor.pos()
        # Offset to the right and below the cursor
        self._tooltip.move(pos.x() + 16, pos.y() + 16)

    def _hide_tooltip(self) -> None:
        """Hide the tooltip and cancel any pending timer."""
        self._last_hover_name = None
        if self._tooltip_timer is not None:
            self._tooltip_timer.stop()
        if self._tooltip is not None:
            self._tooltip.hide()

    def _get_hover_info(self, cell_name: str) -> str:
        """Build concise cell info text for hover tooltip.

        If the hovered cell is the currently selected cell, delegates to
        ``app.get_cell_info_text()`` for the full info. Otherwise builds
        a shorter summary.
        """
        # Full info for selected cell
        if cell_name == self.app.current_cell_name:
            return self.app.get_cell_info_text()

        # Short info for any other cell
        cell = self.app.manager.get_cell(cell_name)
        if cell is None:
            return cell_name

        nuc = cell.get_nucleus_at(self.app.current_time)
        if nuc is None:
            return (
                f"{cell_name}\n"
                f"Not present at t={self.app.current_time}\n"
                f"Exists: t={cell.start_time} - {cell.end_time}"
            )

        lines = [
            cell_name,
            f"Position: ({nuc.x}, {nuc.y}, {nuc.z:.1f})",
            f"Size: {nuc.size}",
            f"Lifetime: t={cell.start_time} - {cell.end_time}",
            f"Fate: {cell.end_fate.name}",
        ]
        if cell.children:
            child_names = ", ".join(c.name for c in cell.children)
            lines.append(f"Children: {child_names}")
        return "\n".join(lines)

    # ── Label visibility controls ─────────────────────────────────

    def toggle_labels_global(self) -> None:
        """Toggle global label visibility on/off."""
        self._labels_global_visible = not self._labels_global_visible
        self.update_overlays()

    def clear_labels(self) -> None:
        """Clear all individually shown labels."""
        self._shown_labels.clear()
        self.update_overlays()

    @property
    def labels_visible(self) -> bool:
        """Whether labels are currently globally visible."""
        return self._labels_global_visible


def _preview_polygon_2d(kind: str, cx: float, cy: float, radius: float) -> np.ndarray:
    """Use geometry as well as color to distinguish proposal point kinds."""

    if kind == "interpolated":
        return np.array(
            [
                [cy - radius, cx],
                [cy, cx + radius],
                [cy + radius, cx],
                [cy, cx - radius],
            ],
            dtype=float,
        )
    if kind == "candidate":
        return np.array(
            [
                [cy - radius, cx - radius],
                [cy - radius, cx + radius],
                [cy + radius, cx + radius],
                [cy + radius, cx - radius],
            ],
            dtype=float,
        )
    return make_circle_polygon(cx, cy, radius, CIRCLE_VERTICES)


def _tracking_review_spots(preview) -> tuple:
    """Include diagnostic candidates when the preview model provides them."""

    return tuple(getattr(preview, "review_spots", preview.spots))


def _search_region_paths_3d(region, calibration) -> list[np.ndarray]:
    """Return three calibrated wireframe rings for a physical search sphere."""

    x_px = float(region.x_um) / calibration.xy_um
    y_px = float(region.y_um) / calibration.xy_um
    z_px = float(region.z_um) / calibration.z_um
    radius_xy = float(region.radius_um) / calibration.xy_um
    radius_z = float(region.radius_um) / calibration.z_um
    angles = np.linspace(0.0, 2.0 * np.pi, 65)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    return [
        np.column_stack(
            (
                np.full_like(angles, z_px),
                y_px + radius_xy * sine,
                x_px + radius_xy * cosine,
            )
        ),
        np.column_stack(
            (
                z_px + radius_z * sine,
                np.full_like(angles, y_px),
                x_px + radius_xy * cosine,
            )
        ),
        np.column_stack(
            (
                z_px + radius_z * sine,
                y_px + radius_xy * cosine,
                np.full_like(angles, x_px),
            )
        ),
    ]


def _viewer_contains_layer(viewer, layer) -> bool:
    """Return whether *layer* is still attached, using identity semantics."""

    try:
        return any(candidate is layer for candidate in viewer.layers)
    except Exception:
        # Lightweight test viewers may not expose an iterable layer list. In
        # that case retaining the reference is safer than creating duplicates.
        return True


def make_circle_polygon(cx: float, cy: float, radius: float,
                         n_vertices: int = CIRCLE_VERTICES) -> np.ndarray:
    """Create a circle polygon as an Nx2 array of (y, x) vertices.

    Matches Java's EUtils.pCircle(): generates a polygon approximation
    of a circle with evenly spaced vertices. This produces true circles
    regardless of display scaling (unlike bounding-box ellipses).

    Args:
        cx: Center x (column).
        cy: Center y (row).
        radius: Circle radius in pixels.
        n_vertices: Number of polygon vertices.

    Returns:
        Nx2 array of (row, col) vertex positions.
    """
    angles = np.linspace(0, 2 * math.pi, n_vertices, endpoint=False)
    vertices = np.column_stack([
        cy + radius * np.sin(angles),  # row
        cx + radius * np.cos(angles),  # col
    ])
    return vertices
