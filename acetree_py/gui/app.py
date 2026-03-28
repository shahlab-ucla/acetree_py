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
        self._lineage_widgets: list = []  # Multiple lineage tree panels
        self._lineage_list = None

        # Cached image layer
        self._image_layer = None

        # 3D view state
        self._3d_mode: bool = False
        self._points_layer = None  # napari Points layer for 3D nuclei

        # Relink pick mode state (Feature 4)
        self._relink_pick_mode: bool = False
        self._relink_pick_callback = None  # callable(time, nuc) when target picked

        # Click-to-place nucleus mode (Track button)
        self._placement_mode: bool = False
        self._placement_parent_name: str | None = None  # None = root mode
        self._placement_default_size: int = 20

        # Click-to-add nucleus mode (Add button)
        self._add_mode: bool = False

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

    @classmethod
    def from_new_dataset(
        cls,
        config: AceTreeConfig,
        num_timepoints: int,
        output_dir: Path,
    ) -> AceTreeApp:
        """Create an AceTreeApp for a brand-new dataset (empty nuclei).

        Writes an empty nuclei ZIP and config XML to *output_dir*,
        then opens the GUI for manual annotation.

        Args:
            config: Configuration built from DatasetCreationDialog.
            num_timepoints: Number of timepoints detected from images.
            output_dir: Where to save the nuclei ZIP and config XML.

        Returns:
            A fully initialized AceTreeApp ready for manual annotation.
        """
        from ..io.config_writer import write_config_xml
        from ..io.nuclei_writer import write_nuclei_zip

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manager = NucleiManager.new_empty(config, num_timepoints)

        # Write initial empty ZIP
        dataset_name = config.zip_file.stem if str(config.zip_file) not in ("", ".") else "nuclei"
        zip_path = output_dir / f"{dataset_name}.zip"
        write_nuclei_zip(manager.nuclei_record, zip_path)
        config.zip_file = zip_path

        # Write config XML
        xml_path = output_dir / f"{dataset_name}.xml"
        config.config_file = xml_path
        write_config_xml(config, xml_path)

        # Create image provider
        image_provider = create_image_provider_from_config(config)
        if image_provider is not None:
            logger.info("Image provider created: %s (planes=%d)",
                        type(image_provider).__name__,
                        image_provider.num_planes)

        app = cls(manager, image_provider)
        app.current_time = 1
        if image_provider is not None and image_provider.num_planes > 0:
            app.current_plane = max(1, image_provider.num_planes // 2)
        else:
            app.current_plane = max(1, (manager.movie.num_planes or 30) // 2)
        return app

    @classmethod
    def from_dialog(cls) -> AceTreeApp | None:
        """Show the dataset creation dialog and create an app if accepted.

        Returns:
            An AceTreeApp if the user completes the dialog, None if cancelled.
        """
        from .dataset_dialog import DatasetCreationDialog

        # Need a QApplication for the dialog
        from qtpy.QtWidgets import QApplication
        qt_app = QApplication.instance()
        if qt_app is None:
            qt_app = QApplication([])

        dlg = DatasetCreationDialog()
        if dlg.exec_() != dlg.Accepted:
            return None

        config = dlg.get_config()
        output_dir = dlg.get_output_directory()
        dataset_name = dlg.get_dataset_name()
        num_timepoints = dlg.get_num_timepoints()

        # Set zip_file name from dataset name
        config.zip_file = output_dir / f"{dataset_name}.zip"

        return cls.from_new_dataset(config, num_timepoints, output_dir)

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

        # Lineage tree view (graphical Sulston tree) — first default panel
        self.add_lineage_panel()

        # Lineage list view (hierarchical JTree-style list)
        self._lineage_list = LineageListWidget(self)
        self.viewer.window.add_dock_widget(
            self._lineage_list,
            name="Lineage List",
            area="left",
        )

        # Add toggle actions to Window menu so closed panels can be reopened
        self._add_panel_menu_actions()

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
        self._update_display_plane_only()

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

        # Show label for the selected cell
        if self._viewer_integration is not None:
            self._viewer_integration._shown_labels.add(name)

        self.update_display()

    def select_cell_at_position(self, x: float, y: float) -> None:
        """Select the closest cell to an (x, y) position on the current image.

        Used for click-to-select in the viewer.  The click must land within
        (or on) the nucleus's projected circle — clicks in empty space are
        ignored to avoid accidentally selecting distant cells.

        Args:
            x: X coordinate in image pixels.
            y: Y coordinate in image pixels.
        """
        nuc = self.manager.find_closest_nucleus(
            x, y, float(self.current_plane), self.current_time,
            require_hit=True, image_plane=self.current_plane,
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
            x, y, float(self.current_plane), self.current_time,
            require_hit=True, image_plane=self.current_plane,
        )
        if nuc is not None:
            cb = self._relink_pick_callback
            self.exit_relink_pick_mode()
            cb(self.current_time, nuc)
        return True

    # ── Click-to-add nucleus mode (Add button) ────────────────────

    def enter_add_mode(self) -> None:
        """Enter click-to-add mode. Left-click places a nucleus."""
        self._add_mode = True

    def exit_add_mode(self) -> None:
        """Exit click-to-add mode."""
        self._add_mode = False

    def _handle_add_click(self, x: float, y: float) -> bool:
        """Handle a left-click in add mode — place a nucleus at (x, y).

        Uses the currently selected cell as predecessor if one is active.
        For gap == 1, sets predecessor directly. For gap > 1, creates the
        nucleus then auto-interpolates to fill the gap. Inherits diameter
        from the parent cell's last nucleus when available.
        """
        if not self._add_mode:
            return False

        from ..editing.commands import AddNucleus, RelinkWithInterpolation

        ix, iy = round(x), round(y)
        iz = float(self.current_plane)
        time = self.current_time
        size = self._placement_default_size  # fallback

        identity = ""
        predecessor = -1  # NILLI
        parent_end_time = None
        parent_end_index = None

        # If a cell is selected, link to it
        parent_name = self.current_cell_name
        if parent_name:
            cell = self.manager.get_cell(parent_name)
            if cell is not None:
                parent_end_time = cell.end_time
                gap = time - parent_end_time
                if gap <= 0:
                    # Same or earlier timepoint — treat as root
                    parent_name = None
                else:
                    identity = parent_name
                    parent_nuc = cell.get_nucleus_at(parent_end_time)
                    if parent_nuc is not None:
                        parent_end_index = parent_nuc.index
                        size = parent_nuc.size  # inherit diameter
                        if gap == 1:
                            predecessor = parent_end_index

        cmd = AddNucleus(
            time=time,
            x=ix,
            y=iy,
            z=iz,
            size=size,
            identity=identity,
            predecessor=predecessor,
        )
        self.edit_history.do(cmd)
        new_index = cmd._added_index

        # Fill gap > 1 with interpolation
        if (parent_name and parent_end_time is not None
                and parent_end_index is not None):
            gap = time - parent_end_time
            if gap > 1:
                interp_cmd = RelinkWithInterpolation(
                    start_time=parent_end_time,
                    start_index=parent_end_index,
                    end_time=time,
                    end_index=new_index,
                )
                self.edit_history.do(interp_cmd)

        return True

    # ── Click-to-place nucleus mode (Track button) ──────────────

    def enter_placement_mode(
        self,
        parent_name: str | None = None,
        default_size: int = 20,
    ) -> None:
        """Enter click-to-place mode for adding nuclei.

        Args:
            parent_name: Name of parent cell to extend, or None for root mode.
            default_size: Default nucleus diameter for placed nuclei.
        """
        self._placement_mode = True
        self._placement_parent_name = parent_name
        self._placement_default_size = default_size

    def exit_placement_mode(self) -> None:
        """Exit click-to-place mode."""
        self._placement_mode = False
        self._placement_parent_name = None

    def _handle_placement_click(self, x: float, y: float) -> bool:
        """Handle a click in placement mode — create a nucleus at (x, y).

        Returns True if the click was consumed.
        """
        if not self._placement_mode:
            return False

        from ..editing.commands import AddNucleus, RelinkWithInterpolation

        ix, iy = round(x), round(y)
        iz = float(self.current_plane)
        time = self.current_time
        parent_name = self._placement_parent_name
        size = self._placement_default_size

        identity = ""
        predecessor = NILLI = -1

        # Determine linking if we have a parent
        parent_end_time = None
        parent_end_index = None
        if parent_name:
            cell = self.manager.get_cell(parent_name)
            if cell is not None:
                parent_end_time = cell.end_time
                gap = time - parent_end_time
                if gap <= 0:
                    # Same or earlier timepoint as parent — can't link;
                    # treat as independent root placement (e.g. single-frame
                    # annotation where multiple nuclei exist at t=1).
                    parent_name = None
                else:
                    identity = parent_name
                    parent_nuc = cell.get_nucleus_at(parent_end_time)
                    if parent_nuc is not None:
                        parent_end_index = parent_nuc.index
                        size = parent_nuc.size  # inherit diameter
                        if gap == 1:
                            # Adjacent: set predecessor directly
                            predecessor = parent_end_index
                        # gap > 1 handled after AddNucleus via interpolation

        # Create the nucleus
        cmd = AddNucleus(
            time=time,
            x=ix,
            y=iy,
            z=iz,
            size=size,
            identity=identity,
            predecessor=predecessor,
        )
        self.edit_history.do(cmd)
        new_index = cmd._added_index

        # Handle gap > 1 with interpolation
        if (parent_name and parent_end_time is not None
                and parent_end_index is not None):
            gap = time - parent_end_time
            if gap > 1:
                interp_cmd = RelinkWithInterpolation(
                    start_time=parent_end_time,
                    start_index=parent_end_index,
                    end_time=time,
                    end_index=new_index,
                )
                self.edit_history.do(interp_cmd)

        # Mode continuation
        if parent_name is None:
            # Root mode: exit after single placement
            self.exit_placement_mode()
        # else: stay in placement mode for continued tracking

        return True

    # ── 3D view toggle ─────────────────────────────────────────────

    def toggle_3d(self) -> None:
        """Toggle between 2D slice view and 3D volume view."""
        if self.viewer is None:
            return
        self._3d_mode = not self._3d_mode
        if self._3d_mode:
            self._enter_3d()
        else:
            self._exit_3d()

    def _enter_3d(self) -> None:
        """Switch to 3D volume rendering with nucleus spheres."""
        if self.viewer is None or self.image_provider is None:
            self._3d_mode = False
            return

        z_scale = self.manager.z_pix_res

        # Load full z-stack and replace 2D image layer data
        try:
            stack = self.image_provider.get_stack(self.current_time)
        except (FileNotFoundError, IndexError) as e:
            logger.warning("Could not load 3D stack: %s", e)
            self._3d_mode = False
            return

        if self._image_layer is not None:
            self._image_layer.data = stack
            self._image_layer.scale = (z_scale, 1.0, 1.0)

        # Hide 2D shapes overlay
        if self._viewer_integration and self._viewer_integration._shapes_layer:
            self._viewer_integration._shapes_layer.visible = False
        if self._viewer_integration and self._viewer_integration._division_line_layer:
            self._viewer_integration._division_line_layer.visible = False

        # Build 3D Points layer for nuclei
        self._update_3d_points()

        # Switch viewer to 3D
        self.viewer.dims.ndisplay = 3

    def _exit_3d(self) -> None:
        """Switch back to 2D slice view."""
        if self.viewer is None:
            return

        # Switch to 2D first
        self.viewer.dims.ndisplay = 2

        # Remove 3D points layer
        if self._points_layer is not None:
            try:
                self.viewer.layers.remove(self._points_layer)
            except ValueError:
                pass
            self._points_layer = None

        # Restore 2D image layer
        if self._image_layer is not None:
            self._image_layer.scale = (1.0, 1.0)

        # Show 2D shapes overlay
        if self._viewer_integration and self._viewer_integration._shapes_layer:
            self._viewer_integration._shapes_layer.visible = True
        if self._viewer_integration and self._viewer_integration._division_line_layer:
            self._viewer_integration._division_line_layer.visible = True

        # Reload 2D plane
        self.update_display()

    def _update_3d_points(self) -> None:
        """Create or update the 3D Points layer for nucleus positions."""
        if self.viewer is None:
            return

        nuclei = self.manager.alive_nuclei_at(self.current_time)
        z_scale = self.manager.z_pix_res

        coords = []
        sizes = []
        colors = []
        names_list = []

        for nuc in nuclei:
            # Points coords in (z, y, x) — z in pixel units, scaled by layer
            coords.append([nuc.z, nuc.y, nuc.x])
            sizes.append(nuc.size)
            names_list.append(nuc.effective_name or f"Nuc{nuc.index}")

            name = nuc.effective_name or ""
            if name == self.current_cell_name and name:
                colors.append([1.0, 1.0, 1.0, 1.0])  # White — selected
            elif name.startswith("Nuc"):
                colors.append([1.0, 0.6, 0.15, 0.8])  # Orange — unnamed (Nuc*)
            elif name:
                colors.append([0.55, 0.27, 1.0, 0.8])  # Purple — named
            else:
                colors.append([0.5, 0.5, 0.5, 0.5])  # Gray — no name at all

        if not coords:
            if self._points_layer is not None:
                self._points_layer.data = np.empty((0, 3))
            return

        coords_arr = np.array(coords)
        sizes_arr = np.array(sizes)
        colors_arr = np.array(colors)

        # Determine which labels to show
        shown = self._viewer_integration._shown_labels if self._viewer_integration else set()
        display_names = []
        for n in names_list:
            if self._viewer_integration and self._viewer_integration._labels_global_visible and n in shown:
                display_names.append(n)
            else:
                display_names.append("")

        if self._points_layer is None:
            self._points_layer = self.viewer.add_points(
                coords_arr,
                size=sizes_arr,
                face_color=colors_arr,
                border_color="transparent",
                name="Nuclei 3D",
                scale=(z_scale, 1.0, 1.0),
                opacity=0.7,
            )
            self._points_layer.features = {"name": display_names}
            self._points_layer.text = {
                "string": "{name}",
                "color": "white",
                "size": 10,
            }
            # Click callback for 3D selection
            self._points_layer.mouse_drag_callbacks.append(self._on_3d_click)
        else:
            self._points_layer.data = coords_arr
            self._points_layer.size = sizes_arr
            self._points_layer.face_color = colors_arr
            self._points_layer.features = {"name": display_names}

    def _on_3d_click(self, layer, event) -> None:
        """Handle click on 3D Points layer to select a cell.

        Supports relink pick mode and placement (track) mode in 3D — when
        either is active, clicking a point fulfils the pick/place action
        instead of doing a plain selection.
        """
        if event.type != "mouse_press":
            return

        # Get the index of the clicked point
        idx = layer.get_value(event.position, world=True)
        nuc = None
        if idx is not None and isinstance(idx, (int, np.integer)):
            nuclei = self.manager.alive_nuclei_at(self.current_time)
            if 0 <= idx < len(nuclei):
                nuc = nuclei[idx]

        # --- Relink pick mode (right-click selects relink target) ---
        # Defer callback via QTimer so napari finalises the click event
        # before the modal confirmation dialog opens.
        if self._relink_pick_mode and self._relink_pick_callback is not None:
            if event.button == 2 and nuc is not None:
                cb = self._relink_pick_callback
                t = self.current_time
                self.exit_relink_pick_mode()
                from qtpy.QtCore import QTimer
                QTimer.singleShot(0, lambda: cb(t, nuc))
            return  # consume all clicks while in pick mode

        # --- Placement / track mode (right-click places a nucleus) ---
        if self._placement_mode and event.button == 2:
            # In 3D we can't get reliable (x,y) from event.position easily,
            # so placement is only supported in 2D.  Just ignore.
            pass

        # --- Normal selection (right-click) ---
        if event.button != 2:
            return
        if nuc is not None:
            name = nuc.effective_name
            if name:
                self.current_cell_name = name
                if self._viewer_integration:
                    self._viewer_integration._shown_labels.add(name)
                self._update_3d_points()
                if self._cell_info_panel:
                    self._cell_info_panel.refresh()
                for lw in self._lineage_widgets:
                    lw.refresh_selection()
                if self._lineage_list:
                    self._lineage_list.refresh_selection()

    # ── Display ───────────────────────────────────────────────────

    def update_display(self) -> None:
        """Refresh all visual components for the current state."""
        self._load_image()

        if self._viewer_integration and not self._3d_mode:
            self._viewer_integration.update_overlays()

        if self._cell_info_panel:
            self._cell_info_panel.refresh()

        if self._player_controls:
            self._player_controls.refresh()

        if self._edit_panel:
            self._edit_panel.refresh()

        for lw in self._lineage_widgets:
            lw.refresh_selection()

        if self._lineage_list:
            self._lineage_list.refresh_selection()

    def _update_display_plane_only(self) -> None:
        """Refresh only z-plane-sensitive components (skip lineage tree).

        When only the z-plane changes, the lineage tree and list are
        unaffected — only the image and nucleus overlay need updating.
        """
        if self._3d_mode:
            return  # z-plane changes don't apply in 3D mode
        self._load_image()

        if self._viewer_integration:
            self._viewer_integration.update_overlays()

        if self._cell_info_panel:
            self._cell_info_panel.refresh()

        if self._player_controls:
            self._player_controls.refresh()

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

            # Color: selected=white, named=purple, Nuc*=orange, none=gray
            ename = nuc.effective_name or ""
            if ename == self.current_cell_name and ename:
                selected_idx = len(centers) - 1
                colors.append([1.0, 1.0, 1.0, 1.0])  # White — selected
            elif ename.startswith("Nuc"):
                colors.append([1.0, 0.6, 0.15, 0.8])  # Orange — unnamed
            elif ename:
                colors.append([0.55, 0.27, 1.0, 0.8])  # Purple — named
            else:
                colors.append([0.5, 0.5, 0.5, 0.5])  # Gray — no name

        return {
            "centers": np.array(centers) if centers else np.empty((0, 2)),
            "radii": np.array(radii) if radii else np.empty(0),
            "colors": np.array(colors) if colors else np.empty((0, 4)),
            "names": names,
            "selected_idx": selected_idx,
        }

    # ── Internal methods ──────────────────────────────────────────

    def _load_image(self) -> None:
        """Load the current image plane (or stack in 3D mode) into the viewer."""
        if self.viewer is None or self.image_provider is None:
            return

        if self._3d_mode:
            # In 3D mode, load full stack and update points
            try:
                stack = self.image_provider.get_stack(self.current_time)
            except (FileNotFoundError, IndexError) as e:
                logger.warning("Could not load 3D stack: %s", e)
                return
            if self._image_layer is not None:
                self._image_layer.data = stack
            self._update_3d_points()
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
        cmd = self.edit_history.last_command
        is_structural = cmd is None or cmd.structural

        if is_structural:
            self.manager.set_all_successors()
            self.manager.process()

            # Structural edits (relink, kill, add) change the lineage tree,
            # so all lineage tree panels need a full rebuild.
            for lw in self._lineage_widgets:
                lw.rebuild_tree()
            if self._lineage_list:
                self._lineage_list.rebuild()

        self.update_display()

    # ── Multi-panel lineage management ──────────────────────────

    def add_lineage_panel(
        self,
        *,
        root_cell_name: str | None = None,
        time_start: int | None = None,
        time_end: int | None = None,
        expr_min: float = -500.0,
        expr_max: float = 5000.0,
        cmap_name: str | None = None,
    ) -> None:
        """Create and dock a new lineage tree panel.

        Args:
            root_cell_name: Root cell to display (None = auto-detect best root).
            time_start: First timepoint to display (None = from root cell).
            time_end: Last timepoint to display (None = full range).
            expr_min: Expression color range minimum.
            expr_max: Expression color range maximum.
            cmap_name: Matplotlib colormap name (None = legacy green-to-red).
        """
        from .lineage_widget import LineageWidget

        widget = LineageWidget(
            self,
            root_cell_name=root_cell_name,
            time_start=time_start,
            time_end=time_end,
            expr_min=expr_min,
            expr_max=expr_max,
            cmap_name=cmap_name,
        )
        self._lineage_widgets.append(widget)

        # Determine panel title
        panel_num = len(self._lineage_widgets)
        title = widget.panel_title()
        if panel_num > 1:
            title = f"{title} ({panel_num})"
        else:
            title = "Lineage Tree"

        if self.viewer is not None:
            self.viewer.window.add_dock_widget(
                widget,
                name=title,
                area="bottom",
            )

    def remove_lineage_panel(self, widget) -> None:
        """Remove a lineage panel from the app."""
        if widget in self._lineage_widgets:
            self._lineage_widgets.remove(widget)
        if self.viewer is not None:
            self.viewer.window.remove_dock_widget(widget)

    def _add_panel_menu_actions(self) -> None:
        """Add show/hide toggle actions and panel management to the Window menu."""
        qt_window = self.viewer.window._qt_window
        menu_bar = qt_window.menuBar()
        # Find the Window menu (napari creates it via app-model as "&Window")
        window_menu = None
        for action in menu_bar.actions():
            if action.menu() and "window" in action.text().lower().replace("&", ""):
                window_menu = action.menu()
                break
        if window_menu is None:
            return

        window_menu.addSeparator()
        # Add toggle actions for each of our dock widgets.
        # We need the QDockWidget wrappers (not inner widgets) for
        # toggleViewAction(), so use the private dict with warning suppressed.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            dock_wrappers = self.viewer.window._dock_widgets
        for dock_widget in dock_wrappers.values():
            toggle = dock_widget.toggleViewAction()
            toggle.setText(dock_widget.name)
            window_menu.addAction(toggle)

        # Add "New Lineage Panel" action
        window_menu.addSeparator()
        from qtpy.QtWidgets import QAction
        add_panel_action = QAction("New Lineage Panel...", qt_window)
        add_panel_action.triggered.connect(self._on_new_lineage_panel)
        window_menu.addAction(add_panel_action)

    def _on_new_lineage_panel(self) -> None:
        """Show config dialog and create a new lineage panel."""
        from .lineage_widget import LineagePanelConfigDialog, LineageWidget

        # Create a temporary widget to host the dialog with defaults
        temp = LineageWidget.__new__(LineageWidget)
        temp.app = self
        temp.root_cell_name = None
        temp.time_start = None
        temp.time_end = None
        temp._expr_min = -500.0
        temp._expr_max = 5000.0
        temp.cmap_name = None

        dlg = LineagePanelConfigDialog(temp)
        if dlg.exec_():
            config = dlg.get_config()
            self.add_lineage_panel(
                root_cell_name=config["root_cell_name"],
                time_start=config["time_start"],
                time_end=config["time_end"],
                expr_min=config["expr_min"],
                expr_max=config["expr_max"],
                cmap_name=config["cmap_name"],
            )

    def _delete_active_nucleus(self) -> None:
        """Delete the active cell's nucleus at the current timepoint only."""
        if not self.current_cell_name:
            return

        cell = self.manager.get_cell(self.current_cell_name)
        if cell is None:
            return

        nuc = cell.get_nucleus_at(self.current_time)
        if nuc is None:
            return

        from ..editing.validators import validate_remove_nucleus

        errors = validate_remove_nucleus(
            self.edit_history.nuclei_record, self.current_time, nuc.index
        )
        if errors:
            return

        from ..editing.commands import RemoveNucleus

        cmd = RemoveNucleus(time=self.current_time, index=nuc.index)
        self.edit_history.do(cmd)

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

        @self.viewer.bind_key("3")
        def _toggle_3d(viewer):
            self.toggle_3d()

        @self.viewer.bind_key("Escape")
        def _exit_modes(viewer):
            if self._add_mode:
                self.exit_add_mode()
                if self._edit_panel:
                    self._edit_panel._btn_add.setChecked(False)
                    self._edit_panel._status_label.setText("Exited add mode")
            elif self._placement_mode:
                self.exit_placement_mode()
                if self._edit_panel:
                    self._edit_panel._btn_track.setChecked(False)
                    self._edit_panel._status_label.setText("Exited tracking mode")
            elif self._relink_pick_mode:
                self.exit_relink_pick_mode()

        @self.viewer.bind_key("Delete")
        def _delete_nucleus(viewer):
            self._delete_active_nucleus()
