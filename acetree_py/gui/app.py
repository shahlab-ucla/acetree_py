"""Main application — creates napari viewer and wires up all components.

This is the entry point for the GUI. It creates the napari viewer,
loads image data via ImageProvider, overlays nucleus annotations,
and adds dock widgets for player controls, cell info, and contrast.

Usage:
    from acetree_py.gui.app import AceTreeApp
    app = AceTreeApp.from_config(config_path)
    app.run()

Ported from: org.rhwlab.acetree.AceTree (the monolithic 4000+ line Java class)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari

from ..core.nuclei_manager import NucleiManager
from ..editing.history import EditHistory
from ..io.config import AceTreeConfig, load_config
from ..io.image_provider import ImageProvider, create_image_provider_from_config

logger = logging.getLogger(__name__)

# Nucleus z-index offset used by Java AceTree (nuc.z is 0-based internally
# but 1-based in the display; Java adds this constant)
NUCZINDEXOFFSET = 1


class AceTreeApp:
    """Main AceTree application with napari viewer.

    Coordinates between:
    - NucleiManager (data)
    - EditHistory (undo/redo)
    - ImageProvider (image loading)
    - ViewerIntegration (nucleus overlay)
    - PlayerControls (navigation widget)
    - CellInfoPanel (selected cell info)
    - ContrastTools (channel contrast)

    Attributes:
        viewer: The napari viewer instance.
        manager: The NucleiManager holding all nuclei/lineage data.
        image_provider: The image data source.
        edit_history: Undo/redo manager.
        current_time: Current timepoint (1-based).
        current_plane: Current z-plane (1-based).
        current_cell_name: Name of the currently selected/tracked cell.
        tracking: If True, the viewer follows the selected cell through time.
    """

    def __init__(
        self,
        manager: NucleiManager,
        image_provider: ImageProvider | None = None,
    ) -> None:
        self.manager = manager
        self.image_provider = image_provider
        self.edit_history = EditHistory(
            manager.nuclei_record,
            on_edit=self._on_edit,
        )

        # Navigation state
        self.current_time: int = 1
        self.current_plane: int = 1
        self.current_cell_name: str = ""
        self.tracking: bool = True

        # GUI components (initialized in launch())
        self.viewer: napari.Viewer | None = None
        self._viewer_integration = None
        self._player_controls = None
        self._cell_info_panel = None
        self._contrast_tools = None
        self._edit_panel = None
        self._lineage_widget = None
        self._lineage_list = None

        # Cached image layer
        self._image_layer = None

        # Relink pick mode state (Feature 4)
        self._relink_pick_mode: bool = False
        self._relink_pick_callback = None  # callable(time, nuc) when target picked

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        image_provider: ImageProvider | None = None,
    ) -> AceTreeApp:
        """Create an AceTreeApp from a config file.

        Args:
            config_path: Path to the XML config file.
            image_provider: Optional image provider (auto-detected if None).

        Returns:
            A fully initialized AceTreeApp (data loaded, not yet launched).
        """
        config = load_config(Path(config_path))
        manager = NucleiManager.from_config(config)
        manager.process()

        # Auto-create image provider if not provided
        if image_provider is None:
            logger.info("Auto-detecting image provider from config...")
            image_provider = create_image_provider_from_config(config)
            if image_provider is not None:
                logger.info("Image provider created: %s (planes=%d)",
                            type(image_provider).__name__,
                            image_provider.num_planes)
            else:
                logger.warning("No image provider could be created from config")

        app = cls(manager, image_provider)
        app.current_time = 1
        # Set initial plane to middle of stack
        if image_provider is not None and image_provider.num_planes > 0:
            app.current_plane = max(1, image_provider.num_planes // 2)
        else:
            app.current_plane = max(1, (manager.movie.num_planes or 30) // 2)
        return app

    def launch(self) -> None:
        """Create the napari viewer and add all dock widgets.

        Call this to open the GUI window. After calling, use napari.run()
        to start the Qt event loop.
        """
        try:
            import napari
        except ImportError:
            raise ImportError(
                "napari is required for the GUI: pip install 'acetree-py[gui]'"
            )

        from .cell_info_panel import CellInfoPanel
        from .contrast_tools import ContrastTools
        from .edit_panel import EditPanel
        from .lineage_list import LineageListWidget
        from .lineage_widget import LineageWidget
        from .player_controls import PlayerControls
        from .viewer_integration import ViewerIntegration

        self.viewer = napari.Viewer(title="AceTree")

        # Set up image layer
        self._load_image()

        # Set up nucleus overlay
        self._viewer_integration = ViewerIntegration(self)
        self._viewer_integration.setup_layers()

        # Add dock widgets
        self._player_controls = PlayerControls(self)
        self.viewer.window.add_dock_widget(
            self._player_controls,
            name="Player Controls",
            area="bottom",
        )

        self._cell_info_panel = CellInfoPanel(self)
        self.viewer.window.add_dock_widget(
            self._cell_info_panel,
            name="Cell Info",
            area="left",
        )

        self._contrast_tools = ContrastTools(self)
        self.viewer.window.add_dock_widget(
            self._contrast_tools,
            name="Contrast",
            area="right",
        )

        self._edit_panel = EditPanel(self)
        self.viewer.window.add_dock_widget(
            self._edit_panel,
            name="Edit Tools",
            area="right",
        )

        # Lineage tree view (graphical Sulston tree)
        self._lineage_widget = LineageWidget(self)
        self.viewer.window.add_dock_widget(
            self._lineage_widget,
            name="Lineage Tree",
            area="bottom",
        )

        # Lineage list view (hierarchical JTree-style list)
        self._lineage_list = LineageListWidget(self)
        self.viewer.window.add_dock_widget(
            self._lineage_list,
            name="Lineage List",
            area="left",
        )

        # Keyboard shortcuts
        self._bind_keys()

        # Initial display
        self.update_display()

        logger.info("AceTree GUI launched")

    def run(self) -> None:
        """Launch the viewer and start the Qt event loop."""
        self.launch()
        import napari
        napari.run()

    # ── Save ──────────────────────────────────────────────────────

    @property
    def _default_save_path(self) -> Path | None:
        """Return the original nuclei ZIP path from config, if available."""
        if self.manager.config and str(self.manager.config.zip_file):
            zf = self.manager.config.zip_file
            # Path() defaults to '.' — treat as unset
            if zf != Path() and str(zf) not in ("", "."):
                return zf
        return None

    def save(self) -> Path | None:
        """Save nuclei to the original ZIP file (overwrite).

        If no original path is known, falls through to save_as().

        Returns:
            The Path that was saved to, or None if cancelled/failed.
        """
        path = self._default_save_path
        if path is None:
            return self.save_as()
        return self._do_save(path)

    def save_as(self) -> Path | None:
        """Save nuclei to a user-chosen ZIP file via a file dialog.

        Returns:
            The Path that was saved to, or None if cancelled/failed.
        """
        if self.viewer is None:
            logger.warning("Cannot show save dialog — no viewer")
            return None

        from qtpy.QtWidgets import QFileDialog

        default = str(self._default_save_path) if self._default_save_path else ""
        path_str, _ = QFileDialog.getSaveFileName(
            self.viewer.window._qt_window,
            "Save Nuclei As",
            default,
            "ZIP archives (*.zip);;All files (*)",
        )
        if not path_str:
            return None  # User cancelled

        return self._do_save(Path(path_str))

    def _do_save(self, path: Path) -> Path | None:
        """Write nuclei_record to *path* and report success/failure."""
        try:
            self.manager.save(path)
            logger.info("Saved nuclei to %s", path)
            return path
        except Exception:
            logger.exception("Failed to save nuclei to %s", path)
            if self.viewer is not None:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self.viewer.window._qt_window,
                    "Save Failed",
                    f"Could not save to:\n{path}\n\nSee log for details.",
                )
            return None

    # ── Navigation ────────────────────────────────────────────────

    def set_time(self, time: int) -> None:
        """Navigate to a specific timepoint.

        Args:
            time: 1-based timepoint.
        """
        time = max(1, min(time, self.manager.num_timepoints))
        if time == self.current_time:
            return
        self.current_time = time

        # Track cell across time
        if self.tracking and self.current_cell_name:
            self._track_cell_at_time()

        self.update_display()

    def set_plane(self, plane: int) -> None:
        """Navigate to a specific z-plane.

        Args:
            plane: 1-based z-plane index.
        """
        max_planes = self.image_provider.num_planes if self.image_provider else 30
        plane = max(1, min(plane, max_planes))
        if plane == self.current_plane:
            return
        self.current_plane = plane
        self.update_display()

    def next_time(self) -> None:
        """Advance to the next timepoint."""
        self.set_time(self.current_time + 1)

    def prev_time(self) -> None:
        """Go back to the previous timepoint."""
        self.set_time(self.current_time - 1)

    def next_plane(self) -> None:
        """Go to the next z-plane."""
        self.set_plane(self.current_plane + 1)

    def prev_plane(self) -> None:
        """Go to the previous z-plane."""
        self.set_plane(self.current_plane - 1)

    def select_cell(self, name: str, time: int | None = None) -> None:
        """Select a cell by name, optionally jumping to a specific time.

        Args:
            name: Cell name (e.g. "ABala").
            time: Optional timepoint to jump to.
        """
        cell = self.manager.get_cell(name)
        if cell is None:
            logger.warning("Cell '%s' not found in lineage tree", name)
            return

        self.current_cell_name = name

        if time is not None:
            self.current_time = max(cell.start_time, min(time, cell.end_time))
        elif self.current_time < cell.start_time or self.current_time > cell.end_time:
            self.current_time = cell.start_time

        # Track to cell's z-plane
        if self.tracking:
            self._track_cell_at_time()

        self.update_display()

    def select_cell_at_position(self, x: float, y: float) -> None:
        """Select the closest cell to an (x, y) position on the current image.

        Used for click-to-select in the viewer.

        Args:
            x: X coordinate in image pixels.
            y: Y coordinate in image pixels.
        """
        nuc = self.manager.find_closest_nucleus(
            x, y, float(self.current_plane), self.current_time
        )
        if nuc and nuc.effective_name:
            self.select_cell(nuc.effective_name, self.current_time)
        elif nuc:
            # Unnamed nucleus — just highlight it
            self.current_cell_name = nuc.identity or f"idx={nuc.index}"
            self.update_display()

    # ── Relink pick mode (Feature 4) ─────────────────────────────

    def enter_relink_pick_mode(self, callback) -> None:
        """Enter pick mode for interactive relink.

        While in pick mode, right-clicking in the image selects a target
        nucleus and calls *callback(time, nucleus)* with the pick result.

        Args:
            callback: Called with (time: int, nuc: Nucleus) when user picks.
        """
        self._relink_pick_mode = True
        self._relink_pick_callback = callback

    def exit_relink_pick_mode(self) -> None:
        """Exit pick mode without choosing a target."""
        self._relink_pick_mode = False
        self._relink_pick_callback = None

    def _handle_relink_pick(self, x: float, y: float) -> bool:
        """If in pick mode, handle a right-click as a pick event.

        Returns True if the click was consumed by pick mode.
        """
        if not self._relink_pick_mode or self._relink_pick_callback is None:
            return False

        nuc = self.manager.find_closest_nucleus(
            x, y, float(self.current_plane), self.current_time
        )
        if nuc is not None:
            cb = self._relink_pick_callback
            self.exit_relink_pick_mode()
            cb(self.current_time, nuc)
        return True

    # ── Display ───────────────────────────────────────────────────

    def update_display(self) -> None:
        """Refresh all visual components for the current state."""
        self._load_image()

        if self._viewer_integration:
            self._viewer_integration.update_overlays()

        if self._cell_info_panel:
            self._cell_info_panel.refresh()

        if self._player_controls:
            self._player_controls.refresh()

        if self._edit_panel:
            self._edit_panel.refresh()

        if self._lineage_widget:
            self._lineage_widget.refresh_selection()

        if self._lineage_list:
            self._lineage_list.refresh_selection()

    def get_cell_info_text(self) -> str:
        """Build the cell info display text for the currently selected cell.

        Returns:
            Formatted string with cell details (name, position, fate, etc.).
        """
        if not self.current_cell_name:
            return "No cell selected"

        cell = self.manager.get_cell(self.current_cell_name)
        if cell is None:
            return f"Cell '{self.current_cell_name}' not in lineage tree"

        nuc = cell.get_nucleus_at(self.current_time)
        if nuc is None:
            return (
                f"{self.current_cell_name}\n"
                f"Not present at t={self.current_time}\n"
                f"Exists: t={cell.start_time} - {cell.end_time}"
            )

        # Count alive cells at this timepoint
        alive_count = len(self.manager.alive_nuclei_at(self.current_time))

        # Compute projected diameter
        diam = self.manager.nucleus_diameter(nuc, self.current_plane)

        lines = [
            f"{self.current_cell_name}",
            f"One of {alive_count} cells at t={self.current_time}",
            "",
            f"Position: ({nuc.x}, {nuc.y}, {nuc.z:.1f})",
            f"Size: {nuc.size}  (displayed: {diam:.1f})",
            f"Index: {nuc.index}",
            "",
            f"Expression: weight={nuc.weight}, rweight={nuc.rweight}",
            "",
            f"Lifetime: t={cell.start_time} - {cell.end_time}",
            f"Fate: {cell.end_fate.name}",
            f"Depth: {cell.depth()} divisions from P0",
        ]

        if cell.parent:
            lines.append(f"Parent: {cell.parent.name}")
        if cell.children:
            child_names = ", ".join(c.name for c in cell.children)
            lines.append(f"Children: {child_names}")

        return "\n".join(lines)

    def get_nucleus_overlay_data(self) -> dict:
        """Compute nucleus overlay data for the current view.

        Returns a dict with arrays needed to draw nucleus circles on the image:
        - centers: Nx2 array of (y, x) positions
        - radii: N-element array of projected radii
        - colors: Nx4 array of RGBA colors
        - names: list of N name strings
        - selected_idx: index of the selected cell in the arrays (or -1)
        """
        nuclei = self.manager.alive_nuclei_at(self.current_time)
        if not nuclei:
            return {
                "centers": np.empty((0, 2)),
                "radii": np.empty(0),
                "colors": np.empty((0, 4)),
                "names": [],
                "selected_idx": -1,
            }

        centers = []
        radii = []
        colors = []
        names = []
        selected_idx = -1

        for nuc in nuclei:
            diam = self.manager.nucleus_diameter(nuc, self.current_plane)
            if diam <= 0:
                continue

            centers.append([nuc.y, nuc.x])  # napari uses (row, col) = (y, x)
            radii.append(diam / 2.0)
            names.append(nuc.effective_name or f"Nuc{nuc.index}")

            # Color: selected = white, named = blue/purple, unnamed = gray
            if nuc.effective_name == self.current_cell_name:
                selected_idx = len(centers) - 1
                colors.append([1.0, 1.0, 1.0, 1.0])  # White
            elif nuc.effective_name:
                colors.append([0.55, 0.27, 1.0, 0.8])  # Purple (matches Java blue)
            else:
                colors.append([0.5, 0.5, 0.5, 0.5])  # Gray

        return {
            "centers": np.array(centers) if centers else np.empty((0, 2)),
            "radii": np.array(radii) if radii else np.empty(0),
            "colors": np.array(colors) if colors else np.empty((0, 4)),
            "names": names,
            "selected_idx": selected_idx,
        }

    # ── Internal methods ──────────────────────────────────────────

    def _load_image(self) -> None:
        """Load the current image plane into the napari viewer."""
        if self.viewer is None or self.image_provider is None:
            return

        try:
            plane_data = self.image_provider.get_plane(
                self.current_time, self.current_plane
            )
        except (FileNotFoundError, IndexError) as e:
            logger.warning("Could not load image: %s", e)
            return

        if self._image_layer is None:
            self._image_layer = self.viewer.add_image(
                plane_data,
                name="Image",
                colormap="gray",
            )
        else:
            self._image_layer.data = plane_data

    def _track_cell_at_time(self) -> None:
        """Update current_plane to follow the tracked cell's z position."""
        if not self.current_cell_name:
            return

        cell = self.manager.get_cell(self.current_cell_name)
        if cell is None:
            return

        # Handle time beyond cell lifetime
        if self.current_time > cell.end_time:
            if cell.children:
                # Follow first daughter
                self.current_cell_name = cell.children[0].name
                cell = cell.children[0]
            else:
                return
        elif self.current_time < cell.start_time:
            if cell.parent:
                self.current_cell_name = cell.parent.name
                cell = cell.parent
            else:
                return

        nuc = cell.get_nucleus_at(self.current_time)
        if nuc:
            self.current_plane = max(1, round(nuc.z + NUCZINDEXOFFSET))

    def _on_edit(self) -> None:
        """Callback after any edit command — rebuild tree and refresh display."""
        self.manager.set_all_successors()
        self.manager.process()
        self.update_display()

    def _bind_keys(self) -> None:
        """Bind keyboard shortcuts to the napari viewer."""
        if self.viewer is None:
            return

        @self.viewer.bind_key("Right")
        def _next_time(viewer):
            self.next_time()

        @self.viewer.bind_key("Left")
        def _prev_time(viewer):
            self.prev_time()

        @self.viewer.bind_key("Up")
        def _next_plane(viewer):
            self.next_plane()

        @self.viewer.bind_key("Down")
        def _prev_plane(viewer):
            self.prev_plane()

        @self.viewer.bind_key("Control-s")
        def _save(viewer):
            self.save()

        @self.viewer.bind_key("Control-Shift-s")
        def _save_as(viewer):
            self.save_as()

        @self.viewer.bind_key("Control-z")
        def _undo(viewer):
            self.edit_history.undo()

        @self.viewer.bind_key("Control-y")
        def _redo(viewer):
            self.edit_history.redo()
