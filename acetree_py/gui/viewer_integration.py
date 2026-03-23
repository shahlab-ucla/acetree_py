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

if TYPE_CHECKING:
    import napari

    from .app import AceTreeApp

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
        # Set of cell names whose labels are hidden (toggled off by left-click)
        self._hidden_labels: set[str] = set()
        # Division line layer (Feature 3)
        self._division_line_layer = None

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

    def update_overlays(self) -> None:
        """Refresh the nucleus overlay for the current view state."""
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
            sel_name = names[selected_idx] if selected_idx < len(names) else ""
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

        # Filter names to match polygons (skip tiny circles)
        # Hide labels that have been toggled off via left-click
        filtered_names = []
        for i in range(len(centers)):
            if radii[i] >= 0.5:
                name = names[i]
                if name in self._hidden_labels:
                    filtered_names.append("")  # Hidden — show blank
                else:
                    filtered_names.append(name)

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

    def _update_division_line(self) -> None:
        """Draw a line connecting daughter cells for one frame after division.

        Only drawn when:
        - A cell is selected
        - current_time == cell.end_time + 1 (the frame right after division)
        - The cell has exactly 2 children (division, not death)
        - Both daughter nuclei are visible at the current time

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

        # Only show on the frame immediately after division
        if self.app.current_time != cell.end_time + 1:
            return

        if len(cell.children) != 2:
            return

        child_a, child_b = cell.children
        nuc_a = child_a.get_nucleus_at(self.app.current_time)
        nuc_b = child_b.get_nucleus_at(self.app.current_time)

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

    def _on_click(self, layer, event) -> None:
        """Handle mouse clicks on the shapes layer.

        Left-click:  Toggle the clicked cell's label visibility on/off.
        Right-click: Select the clicked cell and make it active.
        """
        if event.type != "mouse_press":
            return

        # Get click position in data coordinates
        coords = event.position
        if len(coords) < 2:
            return

        # napari coords are (row, col) = (y, x)
        y, x = coords[-2], coords[-1]

        # Find the closest nucleus at this position
        nuc = self.app.manager.find_closest_nucleus(
            x, y, float(self.app.current_plane), self.app.current_time
        )

        if nuc is None:
            return

        button = event.button  # 1 = left, 2 = right

        if button == 2:
            # Right-click: check if we're in relink pick mode first
            if self.app._handle_relink_pick(x, y):
                return  # Pick mode consumed the click
            # Otherwise: select cell and make active (original behavior)
            self.app.select_cell_at_position(x, y)
        else:
            # Left-click: toggle label visibility for this cell
            name = nuc.effective_name or f"Nuc{nuc.index}"
            if name in self._hidden_labels:
                self._hidden_labels.discard(name)
            else:
                self._hidden_labels.add(name)
            self.update_overlays()


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
