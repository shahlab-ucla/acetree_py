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
from .color_rules import ColorRuleEngine

logger = logging.getLogger(__name__)

# Java AceTree stored this constant as `NUCZINDEXOFFSET = 1`, but in our
# Python port ``nuc.z`` and ``current_plane`` are in the *same* 1-based
# coordinate system (see NucleiManager.find_closest_nucleus /
# nucleus_diameter, which compare ``nuc.z - image_plane`` directly).
# Adding +1 to the snap target therefore lands the viewer one plane above
# the true centroid — visible symptom: the slice follows a selected cell
# across time but stops one plane short of the nucleus.  Set to 0.
NUCZINDEXOFFSET = 0


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

        # Cached image layers (one per channel)
        self._image_layers: list = []
        # Default colormaps for multi-channel display (green/magenta)
        self._channel_colormaps = ["green", "magenta", "cyan", "yellow", "red", "blue"]

        # 3D view state
        self._3d_mode: bool = False
        self._points_layer = None  # napari Points layer for 3D nuclei
        self._trail_points_layer = None  # 3D ghost trail Points layer

        # Relink pick mode state (Feature 4)
        self._relink_pick_mode: bool = False
        self._relink_pick_callback = None  # callable(time, nuc) when target picked

        # Click-to-place nucleus mode (Track button)
        self._placement_mode: bool = False
        self._placement_parent_name: str | None = None  # None = root mode
        self._placement_default_size: int = 20

        # Click-to-add nucleus mode (Add button)
        self._add_mode: bool = False

        # Visualization mode — when True, uses ColorRuleEngine for coloring;
        # when False, uses the hardcoded editing palette (white/purple/orange/gray).
        self._viz_mode: bool = False
        self._color_engine: ColorRuleEngine | None = None

        # Detached 3D viewer windows
        self._3d_windows: list = []

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

        from .contrast_tools import ContrastTools
        from .edit_panel import EditPanel
        from .lineage_list import LineageListWidget
        from .lineage_widget import LineageWidget
        from .player_controls import PlayerControls
        from .viewer_integration import ViewerIntegration

        self.viewer = napari.Viewer(title="AceTree")

        # Hide napari's default layer list and layer controls — they're
        # rarely needed and consume valuable dock space.  Still accessible
        # via the Window menu toggle actions.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for dw in list(self.viewer.window._dock_widgets.values()):
                if dw.objectName() in ("layer list", "layer controls"):
                    dw.setVisible(False)

        # Set up image layer
        self._load_image()

        # Set up nucleus overlay
        self._viewer_integration = ViewerIntegration(self)
        self._viewer_integration.setup_layers()

        # ── Dock widgets ──────────────────────────────────────────
        # Bottom: Player Controls, then Lineage Tree
        self._player_controls = PlayerControls(self)
        self.viewer.window.add_dock_widget(
            self._player_controls,
            name="Player Controls",
            area="bottom",
        )

        # Left: Contrast (compact), then Lineage List
        self._contrast_tools = ContrastTools(self)
        self.viewer.window.add_dock_widget(
            self._contrast_tools,
            name="Contrast",
            area="left",
        )

        self._lineage_list = LineageListWidget(self)
        self.viewer.window.add_dock_widget(
            self._lineage_list,
            name="Lineage List",
            area="left",
        )

        # Right: Edit Tools (compact — D-pad and history are popups)
        self._edit_panel = EditPanel(self)
        self.viewer.window.add_dock_widget(
            self._edit_panel,
            name="Edit Tools",
            area="right",
        )

        # Bottom: Lineage tree view (graphical Sulston tree)
        self.add_lineage_panel()

        # Cell Info is now a hover tooltip, not a dock widget.
        # Keep a reference for the tooltip builder but don't dock it.
        self._cell_info_panel = None

        # Add toggle actions to Window menu so closed panels can be reopened
        self._add_panel_menu_actions()
        # Add File → Measure… action
        self._add_file_menu_actions()

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

    # ── Screenshot + export ─────────────────────────────────────

    def screenshot(self, path: Path | None = None) -> Path | None:
        """Capture the current viewer canvas as a PNG image.

        Args:
            path: Destination file path.  If None, opens a file dialog.

        Returns:
            The Path that was saved to, or None if cancelled.
        """
        if self.viewer is None:
            return None

        if path is None:
            from qtpy.QtWidgets import QFileDialog

            path_str, _ = QFileDialog.getSaveFileName(
                self.viewer.window._qt_window,
                "Save Screenshot",
                f"screenshot_t{self.current_time:04d}.png",
                "PNG images (*.png);;All files (*)",
            )
            if not path_str:
                return None
            path = Path(path_str)

        try:
            self.viewer.screenshot(str(path), canvas_only=True)
            logger.info("Screenshot saved to %s", path)
            return path
        except Exception:
            logger.exception("Failed to save screenshot to %s", path)
            return None

    def record_sequence(
        self,
        start_time: int,
        end_time: int,
        step: int = 1,
        output_dir: str | Path = ".",
    ) -> int:
        """Export a sequence of screenshots across a timepoint range.

        Args:
            start_time: First timepoint (1-based).
            end_time: Last timepoint (1-based, inclusive).
            step: Timepoint increment between frames.
            output_dir: Directory to write PNG files into.

        Returns:
            Number of frames exported.
        """
        if self.viewer is None:
            return 0

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        original_time = self.current_time
        count = 0

        for t in range(start_time, end_time + 1, step):
            self.set_time(t)
            # Force a synchronous repaint so the screenshot captures the
            # updated frame.
            self.viewer.window._qt_window.repaint()

            frame_path = out / f"frame_{t:04d}.png"
            try:
                self.viewer.screenshot(str(frame_path), canvas_only=True)
                count += 1
            except Exception:
                logger.exception("Failed to capture frame at t=%d", t)

        # Restore original timepoint
        self.set_time(original_time)
        logger.info("Recorded %d frames to %s", count, out)
        return count

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

        Manual Z navigation disables auto-tracking (the slice should stop
        snapping to the selected cell's centroid on time-scrubs) but
        keeps ``current_cell_name`` set.  Previously we cleared the cell
        name entirely — which broke Add/Track modes where the user wants
        to adjust the Z slice to place a new nucleus while still
        inheriting the selected cell's name as the predecessor.  The
        selection is only explicitly cleared by clicking empty space,
        selecting a different cell, or pressing Escape.

        Args:
            plane: 1-based z-plane index.
        """
        max_planes = self.image_provider.num_planes if self.image_provider else 30
        plane = max(1, min(plane, max_planes))
        if plane == self.current_plane:
            return
        self.current_plane = plane
        # Preserve current_cell_name so Add/Track modes keep their
        # predecessor.  Just freeze auto-tracking so scrubbing time next
        # doesn't yank Z back to the cell's centroid.
        self.tracking = False
        if self.current_cell_name:
            self.update_display()
        else:
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
        # Explicitly selecting a cell re-enables follow-mode.  This undoes
        # any prior ↑/↓ Z nudge that disabled tracking, so subsequent
        # time-scrubbing snaps the slice back to the selected cell.
        self.tracking = True

        if time is not None:
            self.current_time = max(cell.start_time, min(time, cell.end_time))
        elif self.current_time < cell.start_time or self.current_time > cell.end_time:
            self.current_time = cell.start_time

        # Track to cell's z-plane
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
            # Unnamed nucleus — highlight it and re-enable tracking so
            # subsequent time-scrubbing still snaps Z to follow it (via
            # the predecessor/successor chain fallback in
            # _track_cell_at_time).
            self.current_cell_name = nuc.effective_name or f"idx={nuc.index}"
            self.tracking = True
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
        self._focus_viewer_canvas()

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
        self._focus_viewer_canvas()

    def exit_add_mode(self) -> None:
        """Exit click-to-add mode."""
        self._add_mode = False

    def _focus_viewer_canvas(self) -> None:
        """Return keyboard focus to the napari canvas.

        The viewer-level key bindings registered via ``@viewer.bind_key(...)``
        only fire when the canvas has focus.  After the user clicks a toolbar
        button (Add, Track, Relink) focus moves to that button, which eats
        Escape.  Calling this after entering any mode restores the canvas as
        the active focus target so Escape, Delete, and arrow keys behave.

        Guarded by try/except because the exact napari attribute path has
        varied across versions.
        """
        if self.viewer is None:
            return
        try:
            qt_viewer = self.viewer.window.qt_viewer  # type: ignore[attr-defined]
        except Exception:
            return
        try:
            canvas = qt_viewer.canvas.native  # type: ignore[attr-defined]
        except Exception:
            canvas = None
        for target in (canvas, qt_viewer):
            try:
                if target is not None:
                    target.setFocus()
                    return
            except Exception:
                continue

    @property
    def _image_layer(self):
        """Backward-compatible accessor: first image layer (or None)."""
        return self._image_layers[0] if self._image_layers else None

    # ── Visualization mode ───────────────────────────────────────

    @property
    def color_engine(self) -> ColorRuleEngine:
        """Lazily-created color rule engine for visualization mode."""
        if self._color_engine is None:
            self._color_engine = ColorRuleEngine()
        return self._color_engine

    def set_viz_mode(self, enabled: bool) -> None:
        """Switch between editing and visualization color modes.

        Args:
            enabled: True for visualization mode (rule-based coloring),
                     False for editing mode (status-based palette).
        """
        self._viz_mode = enabled
        if self._3d_mode:
            self._update_3d_points()
        else:
            self.update_display()

    def open_3d_window(self) -> None:
        """Open a new detached 3D viewer window."""
        from .viewer_3d_window import Viewer3DWindow

        win = Viewer3DWindow(self)
        self._3d_windows.append(win)
        win.show()

    def _say(self, msg: str) -> None:
        """Set a one-line status message on the napari status bar.

        Silently no-ops when ``self.viewer`` is None (test / headless
        contexts) or when napari's status attribute is unavailable.
        """
        try:
            if self.viewer is not None:
                self.viewer.status = msg
        except Exception:
            pass

    def _division_suffixes(
        self,
        first_pos: tuple[float, float, float],
        new_pos: tuple[float, float, float],
        time: int,
    ) -> tuple[str, str]:
        """Decide which of two division daughters gets the ``"a"`` suffix.

        Projects both daughter positions onto the embryo's AP direction
        (resolved via ``NucleiManager.get_ap_direction_at``) and returns
        ``(suffix_for_first_daughter, suffix_for_new_daughter)`` — always
        one ``"a"`` and one ``"p"``.

        Falls back to Java AceTree's "+X is anterior" convention when no
        axis information is available — see ``get_ap_direction_at``'s
        4-source priority order.
        """
        import numpy as np
        ap = self.manager.get_ap_direction_at(time)
        p1 = np.asarray(first_pos, dtype=float)
        p2 = np.asarray(new_pos, dtype=float)
        proj_first = float(np.dot(p1, ap))
        proj_new = float(np.dot(p2, ap))
        # Larger projection along AP = more anterior = "a".
        if proj_first >= proj_new:
            return ("a", "p")  # first_daughter is anterior
        return ("p", "a")

    def _handle_add_click(self, x: float, y: float) -> bool:
        """Handle a left-click in add mode — place a nucleus at (x, y).

        Uses the currently selected cell as predecessor if one is active.
        For gap == 1, sets predecessor directly. For gap > 1, creates the
        nucleus then auto-interpolates to fill the gap. Inherits diameter
        from the parent cell's last nucleus when available.

        Manual-division handling: if the click is the second successor of
        the selected parent, the two daughters are named via the AP axis
        (``parent_name + "a"`` / ``parent_name + "p"``) so they don't
        collide on the parent's forced name.  Triple-successor attempts
        are rejected with a status message.
        """
        if not self._add_mode:
            return False

        from ..editing.commands import AddNucleus, RelinkWithInterpolation, RenameCell
        from ..editing.validators import validate_add_nucleus

        ix, iy = round(x), round(y)
        iz = float(self.current_plane)
        time = self.current_time
        size = self._placement_default_size  # fallback

        identity = ""
        assigned_id = ""
        predecessor = -1  # NILLI
        parent_end_time = None
        parent_end_index = None
        parent_nuc_ref = None  # live ref to parent nucleus (for division detection)

        # If a cell is selected, link to it.  The user's intent when clicking
        # Add with a parent selected is "extend this cell forward in time".
        # If they happen to click at the parent's end_time (or earlier), auto-
        # shift the new nucleus to end_time + 1 so the extension actually
        # happens — matching the "Predecessor: <name>" hint shown in the
        # status bar.  Also advance ``current_time`` so the user sees the
        # newly placed nucleus.
        parent_name = self.current_cell_name
        if parent_name:
            cell = self.manager.get_cell(parent_name)
            # Guard against phantom cells (created by the dummy-ancestor
            # scaffold in lineage.py or by _track_cell_at_time following a
            # phantom child).  If current_cell_name maps to a cell with no
            # real nuclei, walk up the parent chain until we find an
            # ancestor that actually has nuclei — that's the cell the user
            # meant to extend.
            while cell is not None and not cell.nuclei:
                cell = cell.parent
            if cell is not None:
                parent_name = cell.name
                # Three modes, disambiguated by where click_time sits
                # relative to the selected cell's lifetime:
                #  (a) Division — a nucleus of this cell lives at click
                #      time AND click_time < cell.end_time (mid-life).
                #      The cell continues past this time, so placing a
                #      sibling here unambiguously creates a division.
                #      Link to the shared predecessor at click_time - 1.
                #      Do NOT auto-advance.  Distance is NOT checked —
                #      refinement of a mid-life nucleus is the Move/
                #      Resize tool's job, not Add.
                #  (b) Extension with auto-advance — no nucleus at
                #      click_time, OR click_time == cell.end_time and
                #      the click is near the existing terminal nucleus
                #      (user extending the cell one frame forward).
                #      Place at cell.end_time + 1.
                #  (c) Extension without auto-advance — click_time is
                #      strictly after cell.end_time, i.e. there's a gap.
                #      Link to cell.end_time's nucleus; RelinkWith-
                #      Interpolation fills the gap.
                #
                # At click_time == end_time, the distance heuristic is
                # kept: a click FAR from the terminal nucleus is still
                # treated as a terminal division.
                existing_here = cell.get_nucleus_at(time)
                is_division_click = False
                if existing_here is not None:
                    if time < cell.end_time:
                        is_division_click = True
                    else:
                        # time == end_time: fall back to distance
                        # heuristic — far click = terminal division,
                        # close click = extend past end_time.
                        dx_ex = ix - existing_here.x
                        dy_ex = iy - existing_here.y
                        threshold_sq = float(existing_here.size) ** 2
                        if (dx_ex * dx_ex + dy_ex * dy_ex) > threshold_sq:
                            is_division_click = True

                if is_division_click and existing_here is not None:
                    # (a) Division mode: link to the shared predecessor.
                    if existing_here.predecessor != -1 and time > 1:
                        parent_end_time = time - 1
                        parent_end_index = existing_here.predecessor
                    # else: no shared predecessor — can't link; placement
                    # goes through as a root at this timepoint.
                else:
                    # (b)/(c) Extension mode — possibly auto-advance.
                    parent_end_time = cell.end_time
                    if time <= parent_end_time:
                        new_time = parent_end_time + 1
                        if new_time <= self.manager.num_timepoints:
                            time = new_time
                            self.current_time = new_time
                            logger.info(
                                "Add: auto-advanced to t=%d to extend cell '%s' "
                                "(which ends at t=%d)",
                                new_time, parent_name, parent_end_time,
                            )
                        else:
                            parent_name = None
                    if parent_name:
                        pnuc = cell.get_nucleus_at(parent_end_time)
                        if pnuc is not None:
                            parent_end_index = pnuc.index

                gap = time - parent_end_time if (parent_name and parent_end_time) else 0
                if parent_name and gap > 0 and parent_end_index is not None:
                    identity = parent_name
                    nr = self.manager.nuclei_record
                    t_idx_p = parent_end_time - 1
                    if 0 <= t_idx_p < len(nr):
                        p_idx_p = parent_end_index - 1
                        if 0 <= p_idx_p < len(nr[t_idx_p]):
                            parent_nuc = nr[t_idx_p][p_idx_p]
                            parent_nuc_ref = parent_nuc
                            size = parent_nuc.size  # inherit diameter
                            # Plant the parent's effective name as a forced
                            # name (assigned_id) on the new nucleus.  The
                            # naming pipeline's _propagate_assigned_ids
                            # will sweep it backward/forward through the
                            # continuation chain, unifying the cell across
                            # timepoints.
                            assigned_id = parent_nuc.effective_name or parent_name
                            if gap == 1:
                                predecessor = parent_end_index

        # Validate BEFORE creating the command.  Blocks e.g. a third
        # successor (the parent already has 2 children), which would
        # otherwise leave a floating nucleus that set_all_successors
        # silently drops during the post-edit rebuild.
        errors = validate_add_nucleus(
            self.edit_history.nuclei_record, time, predecessor,
        )
        if errors:
            self._say(errors[0])
            logger.info("Add rejected: %s", errors[0])
            return False

        # Manual-division detection.  If the click is making the parent
        # dividing (parent already has exactly one successor and we're
        # about to add a second), rename both daughters with "a"/"p"
        # suffixes along the embryo's AP axis so they don't collide on
        # the parent's forced name.
        rename_first_to: str | None = None
        if (parent_nuc_ref is not None
                and predecessor != -1
                and parent_nuc_ref.successor1 != -1
                and parent_nuc_ref.successor2 == -1):
            # The existing successor-1 is the first daughter.
            nr = self.manager.nuclei_record
            t_idx = time - 1
            first_idx = parent_nuc_ref.successor1 - 1
            if 0 <= t_idx < len(nr) and 0 <= first_idx < len(nr[t_idx]):
                first_daughter = nr[t_idx][first_idx]
                base = parent_nuc_ref.effective_name or parent_name
                first_pos = (float(first_daughter.x), float(first_daughter.y),
                             float(first_daughter.z))
                new_pos = (float(ix), float(iy), float(iz))
                first_sfx, new_sfx = self._division_suffixes(
                    first_pos, new_pos, time,
                )
                # Only rename the first daughter if it's still carrying
                # the naive "extension" name (same as the parent).  If
                # the user has customised its name, respect that and
                # only suffix the new daughter's name.
                if first_daughter.assigned_id == base:
                    rename_first_to = base + first_sfx
                assigned_id = base + new_sfx
                identity = assigned_id
                self._say(
                    f"Division: {base} \u2192 "
                    f"{rename_first_to or first_daughter.effective_name} + {assigned_id}",
                )

        # Issue AddNucleus FIRST so that set_all_successors marks the
        # parent as dividing (successor2 set to the new nucleus).  The
        # subsequent RenameCell on the first daughter then walks only
        # that daughter's continuation chain — the backward walk stops
        # at the now-dividing parent instead of bleeding into the
        # parent cell.  Issuing them in the opposite order would
        # rename the parent too (it and the first daughter were a
        # single continuation chain until the second daughter landed).
        cmd = AddNucleus(
            time=time,
            x=ix,
            y=iy,
            z=iz,
            size=size,
            identity=identity,
            predecessor=predecessor,
            assigned_id=assigned_id,
        )
        self.edit_history.do(cmd)
        new_index = cmd._added_index

        if rename_first_to is not None:
            self.edit_history.do(
                RenameCell(time=time, index=first_idx + 1,
                           new_name=rename_first_to),
            )

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
        self._focus_viewer_canvas()

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

        from ..editing.commands import AddNucleus, RelinkWithInterpolation, RenameCell
        from ..editing.validators import validate_add_nucleus

        ix, iy = round(x), round(y)
        iz = float(self.current_plane)
        time = self.current_time
        parent_name = self._placement_parent_name
        size = self._placement_default_size

        identity = ""
        assigned_id = ""
        predecessor = NILLI = -1
        parent_nuc_ref = None

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
                        parent_nuc_ref = parent_nuc
                        parent_end_index = parent_nuc.index
                        size = parent_nuc.size  # inherit diameter
                        # Plant parent's effective name as a forced name on
                        # the new nucleus so the naming pipeline propagates
                        # it through the continuation chain (see bug 3).
                        assigned_id = parent_nuc.effective_name or parent_name
                        if gap == 1:
                            # Adjacent: set predecessor directly
                            predecessor = parent_end_index
                        # gap > 1 handled after AddNucleus via interpolation

        # Validate BEFORE creating the command — reject triple-successor
        # attempts with a status message instead of silently letting
        # set_all_successors drop the third link.
        errors = validate_add_nucleus(
            self.edit_history.nuclei_record, time, predecessor,
        )
        if errors:
            self._say(errors[0])
            logger.info("Placement rejected: %s", errors[0])
            return False

        # Manual-division detection (mirrors _handle_add_click).  If the
        # new nucleus is the second successor of the parent, rename the
        # first daughter and the new daughter with axis-aware "a"/"p"
        # suffixes so they don't collide on the parent's forced name.
        rename_first_to: str | None = None
        first_idx = -1
        if (parent_nuc_ref is not None
                and predecessor != NILLI
                and parent_nuc_ref.successor1 != NILLI
                and parent_nuc_ref.successor2 == NILLI):
            nr = self.manager.nuclei_record
            t_idx = time - 1
            first_idx = parent_nuc_ref.successor1 - 1
            if 0 <= t_idx < len(nr) and 0 <= first_idx < len(nr[t_idx]):
                first_daughter = nr[t_idx][first_idx]
                base = parent_nuc_ref.effective_name or parent_name
                first_pos = (float(first_daughter.x), float(first_daughter.y),
                             float(first_daughter.z))
                new_pos = (float(ix), float(iy), float(iz))
                first_sfx, new_sfx = self._division_suffixes(
                    first_pos, new_pos, time,
                )
                if first_daughter.assigned_id == base:
                    rename_first_to = base + first_sfx
                assigned_id = base + new_sfx
                identity = assigned_id
                self._say(
                    f"Division: {base} \u2192 "
                    f"{rename_first_to or first_daughter.effective_name} + {assigned_id}",
                )

        # AddNucleus first, RenameCell second — see _handle_add_click for
        # why the ordering matters (parent-vs-daughter continuation chain).
        cmd = AddNucleus(
            time=time,
            x=ix,
            y=iy,
            z=iz,
            size=size,
            identity=identity,
            predecessor=predecessor,
            assigned_id=assigned_id,
        )
        self.edit_history.do(cmd)
        new_index = cmd._added_index

        if rename_first_to is not None:
            self.edit_history.do(
                RenameCell(time=time, index=first_idx + 1,
                           new_name=rename_first_to),
            )

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

        # Load full z-stacks for all channels
        n_ch = self.image_provider.num_channels
        for ch in range(n_ch):
            try:
                stack = self.image_provider.get_stack(
                    self.current_time, channel=ch
                )
            except (FileNotFoundError, IndexError) as e:
                logger.warning("Could not load 3D stack ch%d: %s", ch, e)
                self._3d_mode = False
                return

            if ch < len(self._image_layers):
                self._image_layers[ch].data = stack
                self._image_layers[ch].scale = (z_scale, 1.0, 1.0)

        # Hide 2D shapes overlay (incl. trails)
        if self._viewer_integration:
            if self._viewer_integration._shapes_layer:
                self._viewer_integration._shapes_layer.visible = False
            if self._viewer_integration._division_line_layer:
                self._viewer_integration._division_line_layer.visible = False
            if self._viewer_integration._trails_layer:
                self._viewer_integration._trails_layer.visible = False

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

        # Remove 3D trail layer
        if self._trail_points_layer is not None:
            try:
                self.viewer.layers.remove(self._trail_points_layer)
            except ValueError:
                pass
            self._trail_points_layer = None

        # Restore 2D image layers
        for layer in self._image_layers:
            layer.scale = (1.0, 1.0)

        # Show 2D shapes overlay (incl. trails)
        if self._viewer_integration:
            if self._viewer_integration._shapes_layer:
                self._viewer_integration._shapes_layer.visible = True
            if self._viewer_integration._division_line_layer:
                self._viewer_integration._division_line_layer.visible = True
            if self._viewer_integration._trails_layer:
                self._viewer_integration._trails_layer.visible = True

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
        names_list = []

        for nuc in nuclei:
            # Points coords in (z, y, x) — z in pixel units, scaled by layer
            coords.append([nuc.z, nuc.y, nuc.x])
            sizes.append(nuc.size)
            names_list.append(nuc.effective_name or f"Nuc{nuc.index}")

        if self._viz_mode:
            # Visualization mode — batch rule-engine colors
            colors = [
                list(c) for c in self.color_engine.colors_for_frame(
                    nuclei, self.manager, self.current_time,
                    selected_name=self.current_cell_name,
                )
            ]
        else:
            # Editing mode — status-based palette
            colors = []
            for nuc in nuclei:
                name = nuc.effective_name or ""
                if name == self.current_cell_name and name:
                    colors.append([1.0, 1.0, 1.0, 1.0])  # White — selected
                elif name.startswith("Nuc"):
                    colors.append([1.0, 0.6, 0.15, 0.8])  # Orange — unnamed
                elif name:
                    colors.append([0.55, 0.27, 1.0, 0.8])  # Purple — named
                else:
                    colors.append([0.5, 0.5, 0.5, 0.5])  # Gray — no name

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

        # Ghost trail in 3D
        self._update_3d_trail()

    def _update_3d_trail(self) -> None:
        """Update 3D ghost trail points for the selected cell's past positions."""
        vi = self._viewer_integration
        if self.viewer is None or vi is None or not vi.trails_visible:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        cell_name = self.current_cell_name
        if not cell_name:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        cell = self.manager.get_cell(cell_name)
        if cell is None:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        trail_len = vi.trail_length
        start = max(cell.start_time, self.current_time - trail_len)

        coords = []
        sizes = []
        colors = []

        for t in range(start, self.current_time):
            nuc = cell.get_nucleus_at(t)
            if nuc is None:
                continue
            age = self.current_time - t
            alpha = max(0.15, 0.6 * (1.0 - age / (trail_len + 1)))
            coords.append([nuc.z, nuc.y, nuc.x])
            sizes.append(nuc.size * 0.6)  # slightly smaller than live nuclei
            colors.append([0.3, 0.8, 1.0, alpha])

        z_scale = self.manager.z_pix_res

        if not coords:
            if self._trail_points_layer is not None:
                self._trail_points_layer.data = np.empty((0, 3))
            return

        coords_arr = np.array(coords)
        sizes_arr = np.array(sizes)
        colors_arr = np.array(colors)

        if self._trail_points_layer is None:
            self._trail_points_layer = self.viewer.add_points(
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

    def _on_3d_click(self, layer, event):
        """Handle click on 3D Points layer to select or label a cell.

        Left-click:  Toggle the clicked cell's label on/off.
        Right-click: Select the clicked cell and make it active (also shows label).

        Also supports relink pick mode and placement (track) mode in 3D.

        This is a generator callback (yields once) so that napari properly
        finalises the drag/pan cycle after the click is handled.
        """
        if event.type != "mouse_press":
            return

        # ── 3D ray-based point picking ──
        # The event carries view_direction and dims_displayed from the
        # camera; passing them to get_value enables proper 3D ray casting
        # instead of falling back to unreliable 2D projection.
        view_direction = getattr(event, "view_direction", None)
        dims_displayed = getattr(event, "dims_displayed", None)
        idx = layer.get_value(
            event.position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=True,
        )
        nuc = None
        if idx is not None and isinstance(idx, (int, np.integer)):
            nuclei = self.manager.alive_nuclei_at(self.current_time)
            if 0 <= idx < len(nuclei):
                nuc = nuclei[idx]

        # --- Relink pick mode (any click selects relink target) ---
        # Defer callback via QTimer so napari finalises the click event
        # before the modal confirmation dialog opens.
        if self._relink_pick_mode and self._relink_pick_callback is not None:
            if nuc is not None:
                cb = self._relink_pick_callback
                t = self.current_time
                self.exit_relink_pick_mode()
                from qtpy.QtCore import QTimer
                QTimer.singleShot(0, lambda: cb(t, nuc))
            yield  # release drag cycle
            return

        button = event.button  # 1 = left, 2 = right

        # --- Placement / track mode (right-click places a nucleus) ---
        # In 3D we cannot reliably determine the (x, y, z) data position
        # from the click ray, so placement is only supported in 2D.
        if self._placement_mode and button == 2:
            yield
            return

        if button == 2:
            # --- Right-click: select cell and show its label ---
            if nuc is not None:
                name = nuc.effective_name
                if name:
                    self.current_cell_name = name
                    if self._viewer_integration:
                        self._viewer_integration._shown_labels.add(name)
                    self._update_3d_points()
                    for lw in self._lineage_widgets:
                        lw.refresh_selection()
                    if self._lineage_list:
                        self._lineage_list.refresh_selection()
        else:
            # --- Left-click: toggle label for clicked cell ---
            if nuc is not None and self._viewer_integration:
                name = nuc.effective_name or f"Nuc{nuc.index}"
                if name in self._viewer_integration._shown_labels:
                    self._viewer_integration._shown_labels.discard(name)
                else:
                    self._viewer_integration._shown_labels.add(name)
                self._update_3d_points()

        yield  # release drag cycle

    # ── Display ───────────────────────────────────────────────────

    def update_display(self) -> None:
        """Refresh all visual components for the current state."""
        self._load_image()

        if self._viewer_integration and not self._3d_mode:
            self._viewer_integration.update_overlays()

        if self._contrast_tools:
            self._contrast_tools.refresh()

        if self._player_controls:
            self._player_controls.refresh()

        if self._edit_panel:
            self._edit_panel.refresh()

        for lw in self._lineage_widgets:
            lw.refresh_selection()

        if self._lineage_list:
            self._lineage_list.refresh_selection()

        # Refresh any detached 3D viewer windows
        for win in self._3d_windows:
            try:
                if win.isVisible():
                    win.refresh()
            except RuntimeError:
                pass  # window was deleted

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

        # Pre-compute visualization-mode colors for the whole frame
        # (batched for efficiency; skipped in editing mode).
        if self._viz_mode:
            viz_colors = self.color_engine.colors_for_frame(
                nuclei, self.manager, self.current_time,
                selected_name=self.current_cell_name,
            )

        centers = []
        radii = []
        colors = []
        names = []
        selected_idx = -1
        viz_idx = 0  # tracks position in the unfiltered nuclei list

        for nuc in nuclei:
            diam = self.manager.nucleus_diameter(nuc, self.current_plane)
            if diam <= 0:
                viz_idx += 1
                continue

            centers.append([nuc.y, nuc.x])  # napari uses (row, col) = (y, x)
            radii.append(diam / 2.0)
            ename = nuc.effective_name or f"Nuc{nuc.index}"
            names.append(ename)

            if self._viz_mode:
                # Visualization mode — rule-engine colors
                r, g, b, a = viz_colors[viz_idx]
                colors.append([r, g, b, a])
                if ename == self.current_cell_name and ename:
                    selected_idx = len(centers) - 1
            else:
                # Editing mode — status-based palette
                if ename == self.current_cell_name and ename:
                    selected_idx = len(centers) - 1
                    colors.append([1.0, 1.0, 1.0, 1.0])  # White — selected
                elif ename.startswith("Nuc"):
                    colors.append([1.0, 0.6, 0.15, 0.8])  # Orange — unnamed
                elif ename:
                    colors.append([0.55, 0.27, 1.0, 0.8])  # Purple — named
                else:
                    colors.append([0.5, 0.5, 0.5, 0.5])  # Gray — no name

            viz_idx += 1

        return {
            "centers": np.array(centers) if centers else np.empty((0, 2)),
            "radii": np.array(radii) if radii else np.empty(0),
            "colors": np.array(colors) if colors else np.empty((0, 4)),
            "names": names,
            "selected_idx": selected_idx,
        }

    # ── Internal methods ──────────────────────────────────────────

    def _load_image(self) -> None:
        """Load the current image plane (or stack in 3D mode) into the viewer.

        Creates one napari Image layer per channel.  For single-channel
        data a gray colormap is used; for multi-channel, green/magenta
        (standard fluorescence convention).
        """
        if self.viewer is None or self.image_provider is None:
            return

        n_ch = self.image_provider.num_channels

        if self._3d_mode:
            for ch in range(n_ch):
                try:
                    stack = self.image_provider.get_stack(
                        self.current_time, channel=ch
                    )
                except (FileNotFoundError, IndexError) as e:
                    logger.warning("Could not load 3D stack ch%d: %s", ch, e)
                    continue
                if ch < len(self._image_layers):
                    self._image_layers[ch].data = stack
                else:
                    cmap = "gray" if n_ch == 1 else self._channel_colormaps[ch % len(self._channel_colormaps)]
                    layer = self.viewer.add_image(
                        stack,
                        name=f"Ch{ch + 1}" if n_ch > 1 else "Image",
                        colormap=cmap,
                        blending="additive" if n_ch > 1 else "translucent",
                    )
                    self._image_layers.append(layer)
            self._update_3d_points()
            return

        for ch in range(n_ch):
            try:
                plane_data = self.image_provider.get_plane(
                    self.current_time, self.current_plane, channel=ch
                )
            except (FileNotFoundError, IndexError) as e:
                logger.warning("Could not load image ch%d: %s", ch, e)
                continue

            if ch < len(self._image_layers):
                self._image_layers[ch].data = plane_data
            else:
                cmap = "gray" if n_ch == 1 else self._channel_colormaps[ch % len(self._channel_colormaps)]
                layer = self.viewer.add_image(
                    plane_data,
                    name=f"Ch{ch + 1}" if n_ch > 1 else "Image",
                    colormap=cmap,
                    blending="additive" if n_ch > 1 else "translucent",
                )
                self._image_layers.append(layer)

    def _track_cell_at_time(self) -> None:
        """Update current_plane to follow the tracked cell's z position.

        Primary path: ``cell.get_nucleus_at(current_time)``.

        Fallback: if the cell's in-memory nucleus dict doesn't have an entry
        for the target timepoint (common for manually-added nuclei whose
        continuation chain hasn't yet been stitched into one multi-timepoint
        cell by the naming pipeline), walk the predecessor/successor chain
        in ``nuclei_record`` starting from the cell's known nuclei.  This
        keeps the slice snapping to the right Z across time even for cells
        that aren't fully materialised in the lineage tree.
        """
        if not self.current_cell_name:
            return

        cell = self.manager.get_cell(self.current_cell_name)
        if cell is None:
            return

        # Handle time beyond cell lifetime: follow to a daughter / parent
        # if the tree knows one with real nuclei.  IMPORTANT: skip phantom
        # children/parents created by the dummy-ancestor scaffold in
        # lineage.py (e.g. AB automatically gets ABa/ABp children even
        # when the user hasn't placed any nuclei for them).  Following a
        # phantom corrupts ``current_cell_name`` and breaks subsequent
        # Add clicks, which then think the parent is the phantom and
        # fail to link.
        if self.current_time > cell.end_time:
            real_child = next(
                (c for c in cell.children if c.nuclei), None
            )
            if real_child is not None:
                self.current_cell_name = real_child.name
                cell = real_child
        elif self.current_time < cell.start_time:
            if cell.parent is not None and cell.parent.nuclei:
                self.current_cell_name = cell.parent.name
                cell = cell.parent

        nuc = cell.get_nucleus_at(self.current_time)
        if nuc is None:
            nuc = self._find_nucleus_via_chain(cell, self.current_time)
        if nuc:
            self.current_plane = max(1, round(nuc.z + NUCZINDEXOFFSET))

    def _find_nucleus_via_chain(self, cell, target_time: int):
        """Walk the predecessor / successor chain in ``nuclei_record``
        starting from the cell's known nuclei, looking for a nucleus at
        ``target_time``.  Returns the ``Nucleus`` or ``None``.

        Used as a fallback when ``cell.get_nucleus_at(t)`` returns None —
        e.g. a continuation chain added manually that hasn't been glued
        into a single multi-timepoint cell yet.  Stops at divisions and
        at dead nuclei.
        """
        nr = self.manager.nuclei_record
        if not cell.nuclei or not nr:
            return None
        known = sorted(cell.nuclei, key=lambda tn: tn[0])

        if target_time > known[-1][0]:
            # Walk forward from the latest known nucleus via successor1
            t, nuc = known[-1]
            while t < target_time:
                s = nuc.successor1
                # Stop at divisions and on broken links
                if s <= 0 or nuc.successor2 > 0:
                    return None
                t_next = t + 1
                t_idx = t_next - 1
                if not (0 <= t_idx < len(nr)):
                    return None
                idx = s - 1
                if not (0 <= idx < len(nr[t_idx])):
                    return None
                nuc = nr[t_idx][idx]
                if nuc.status < 1:
                    return None
                t = t_next
                if t == target_time:
                    return nuc
        elif target_time < known[0][0]:
            # Walk backward from the earliest known nucleus via predecessor
            t, nuc = known[0]
            while t > target_time:
                p = nuc.predecessor
                if p <= 0:
                    return None
                t_prev = t - 1
                if t_prev < 1:
                    return None
                t_idx = t_prev - 1
                idx = p - 1
                if not (0 <= t_idx < len(nr)) or not (0 <= idx < len(nr[t_idx])):
                    return None
                nuc = nr[t_idx][idx]
                if nuc.status < 1:
                    return None
                t = t_prev
                if t == target_time:
                    return nuc
        return None

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

        # After a rename or swap (or undo of either), the tracked cell's
        # name in the tree may have changed.  Read the nucleus's current
        # effective_name to get the correct post-rebuild name (works for
        # both execute and undo paths).
        from ..editing.commands import RenameCell, SwapCellNames

        anchors: list[tuple[int, int]] = []
        if isinstance(cmd, RenameCell):
            anchors = [(cmd.time, cmd.index)]
        elif isinstance(cmd, SwapCellNames):
            anchors = [(cmd.time_a, cmd.index_a), (cmd.time_b, cmd.index_b)]

        if anchors and self.current_cell_name:
            nr = self.manager.nuclei_record
            old_name = self.current_cell_name
            for t_1based, idx_1based in anchors:
                t_idx = t_1based - 1
                n_idx = idx_1based - 1
                if 0 <= t_idx < len(nr) and 0 <= n_idx < len(nr[t_idx]):
                    nuc = nr[t_idx][n_idx]
                    new_name = nuc.effective_name
                    if new_name and self.manager.get_cell(new_name) is not None:
                        # For a swap, pick whichever anchor's new name is
                        # actually in the tree.  If neither matches the
                        # old tracked name, the first valid anchor wins.
                        self.current_cell_name = new_name
                        if self._viewer_integration is not None:
                            self._viewer_integration._shown_labels.discard(old_name)
                            self._viewer_integration._shown_labels.add(new_name)
                        break

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

    def _add_file_menu_actions(self) -> None:
        """Add a 'Measure…' action under the File menu.

        Walks the napari menubar for a File menu and appends the
        Measure action.  Silently no-ops if the menubar isn't
        available (e.g. when running headlessly under pytest).
        """
        try:
            qt_window = self.viewer.window._qt_window
            menu_bar = qt_window.menuBar()
        except Exception:
            return

        file_menu = None
        for action in menu_bar.actions():
            if action.menu() and "file" in action.text().lower().replace("&", ""):
                file_menu = action.menu()
                break
        if file_menu is None:
            return

        from qtpy.QtWidgets import QAction
        file_menu.addSeparator()
        measure_action = QAction("Measure…", qt_window)
        measure_action.triggered.connect(self._on_measure)
        file_menu.addAction(measure_action)

    def _on_measure(self) -> None:
        """Run the Measure orchestrator from a File → Measure… dialog.

        Opens :class:`MeasureDialog`, shows a progress dialog while
        :func:`run_measure` iterates every channel × timepoint, then
        rebuilds every lineage panel so the refreshed ``rweight``
        values show up in the tree colors.
        """
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QApplication,
            QMessageBox,
            QProgressDialog,
        )

        if self.image_provider is None:
            QMessageBox.warning(
                None,
                "Measure",
                "No image data loaded — cannot run Measure.",
            )
            return
        if not self.manager.nuclei_record:
            QMessageBox.warning(
                None,
                "Measure",
                "No nuclei loaded — cannot run Measure.",
            )
            return

        from .measure_dialog import MeasureDialog
        qt_window = self.viewer.window._qt_window if self.viewer else None
        dlg = MeasureDialog(self, parent=qt_window)
        if not dlg.exec_():
            return
        values = dlg.get_values()
        at_channel: int = values["at_channel"]
        output_dir: Path = values["output_dir"]
        correction_method: str = values.get("correction_method", "global")

        n_channels = int(self.image_provider.num_channels)
        n_timepoints = len(self.manager.nuclei_record)
        total_steps = max(1, n_channels * n_timepoints)

        progress = QProgressDialog(
            "Measuring…", "Cancel", 0, total_steps, qt_window,
        )
        progress.setWindowTitle("Measure")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def progress_cb(c_idx: int, n_ch: int, t_1based: int, n_tp: int) -> bool:
            step = c_idx * n_tp + t_1based
            progress.setValue(step)
            progress.setLabelText(
                f"Measuring channel {c_idx + 1}/{n_ch}, "
                f"timepoint {t_1based}/{n_tp}…"
            )
            QApplication.processEvents()
            return not progress.wasCanceled()

        from ..analysis.measure_runner import run_measure
        try:
            written = run_measure(
                self.manager,
                self.image_provider,
                output_dir,
                at_channel,
                progress_cb=progress_cb,
                correction_method=correction_method,
            )
        except RuntimeError as e:
            # User-cancelled or orchestrator-raised runtime error
            progress.close()
            QMessageBox.information(None, "Measure", str(e))
            return
        except Exception as e:  # noqa: BLE001 — surface unknown errors
            progress.close()
            logger.exception("Measure failed")
            QMessageBox.critical(
                None,
                "Measure failed",
                f"Measure could not complete:\n{e}",
            )
            return
        finally:
            progress.setValue(total_steps)
            progress.close()

        # Re-color lineage trees with the fresh rweight values.
        for lw in self._lineage_widgets:
            try:
                lw.rebuild_tree()
            except Exception:
                logger.exception("Failed to rebuild lineage widget")

        msg = (
            f"Measured {len(written)} channel(s); "
            f"wrote CSV(s) to {output_dir}"
        )
        try:
            self.viewer.status = msg
        except Exception:
            pass
        QMessageBox.information(None, "Measure complete", msg)

    def _delete_active_nucleus(self) -> None:
        """Delete the selected nucleus at the current timepoint.

        Resolves the target nucleus in this priority order:

        1. ``current_cell_name`` resolves to a real Cell → use that cell's
           nucleus at the current timepoint.
        2. ``current_cell_name`` has the ``idx=N`` fallback form (set by
           ``select_cell_at_position`` for unnamed manually-added nuclei)
           → parse N and delete that nucleus directly.

        Surfaces a status message when nothing can be deleted so failures
        aren't silent.
        """
        _say = self._say

        if not self.current_cell_name:
            _say("No nucleus selected to delete")
            return

        index: int | None = None

        # Path 1: real named cell in the lineage tree
        cell = self.manager.get_cell(self.current_cell_name)
        if cell is not None:
            nuc = cell.get_nucleus_at(self.current_time)
            if nuc is not None:
                index = nuc.index

        # Path 2: raw idx= fallback (unnamed manually-added nucleus)
        if index is None and self.current_cell_name.startswith("idx="):
            try:
                index = int(self.current_cell_name[4:])
            except ValueError:
                index = None

        if index is None:
            _say(f"Cannot locate '{self.current_cell_name}' at t={self.current_time}")
            return

        from ..editing.validators import validate_remove_nucleus

        errors = validate_remove_nucleus(
            self.edit_history.nuclei_record, self.current_time, index
        )
        if errors:
            _say(f"Delete failed: {errors[0]}")
            return

        from ..editing.commands import RemoveNucleus

        deleted_cell_name = self.current_cell_name
        deleted_at_time = self.current_time

        cmd = RemoveNucleus(time=deleted_at_time, index=index)
        self.edit_history.do(cmd)
        _say(f"Removed nucleus at t={deleted_at_time} idx={index}")

        # Chain-delete UX: step the view back one timepoint and re-anchor
        # on the cell if it still has any nuclei.  That way repeatedly
        # pressing Delete walks backward along the cell's continuation
        # chain, killing one timepoint per press.  If the cell is now
        # empty (we just killed its last nucleus) OR the selection was
        # an "idx=N" raw-nucleus fallback (no lineage entry), deselect.
        if deleted_at_time > 1:
            self.current_time = deleted_at_time - 1

        if deleted_cell_name and not deleted_cell_name.startswith("idx="):
            cell_after = self.manager.get_cell(deleted_cell_name)
            if cell_after is not None and cell_after.nuclei:
                # Cell still exists — stay selected, snap Z to its
                # nucleus at the new (earlier) timepoint.
                self.current_cell_name = deleted_cell_name
                # _track_cell_at_time also runs from update_display via
                # _on_edit, but that fires at the pre-step_back time;
                # call it now with the updated current_time so the Z
                # slice lands on the cell at t-1.
                self._track_cell_at_time()
            else:
                self.current_cell_name = ""
        else:
            self.current_cell_name = ""

        # Refresh the viewer to reflect the new time + selection.  The
        # edit already triggered one update_display via _on_edit, but
        # that ran before we stepped current_time back.
        self.update_display()

    def _exit_all_modes(self) -> None:
        """Exit every interaction mode and reset every toolbar button.

        Called from both the napari viewer keybinding and a global
        ``QShortcut`` attached to the main window so Escape works
        regardless of which widget has keyboard focus (see ``_bind_keys``).
        """
        changed = False
        if self._add_mode:
            self.exit_add_mode()
            changed = True
        if self._placement_mode:
            self.exit_placement_mode()
            changed = True
        if self._relink_pick_mode:
            self.exit_relink_pick_mode()
            changed = True

        if self._edit_panel:
            try:
                self._edit_panel._btn_add.setChecked(False)
            except Exception:
                pass
            try:
                self._edit_panel._btn_track.setChecked(False)
            except Exception:
                pass
            if changed:
                self._edit_panel._status_label.setText("Exited mode")

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

        @self.viewer.bind_key("a")
        def _prev_time_a(viewer):
            self.prev_time()

        @self.viewer.bind_key("z")
        def _prev_time_z(viewer):
            self.prev_time()

        @self.viewer.bind_key("d")
        def _next_time_d(viewer):
            self.next_time()

        @self.viewer.bind_key("w")
        def _inc_z(viewer):
            self.next_plane()

        @self.viewer.bind_key("s")
        def _dec_z(viewer):
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
            # Canvas-focused Escape path.  See _exit_all_modes() for the
            # shared implementation used by both this handler and the
            # application-wide QShortcut below.
            self._exit_all_modes()

        @self.viewer.bind_key("Delete")
        def _delete_nucleus(viewer):
            self._delete_active_nucleus()

        # ── Application-wide Escape shortcut ───────────────────────
        # Napari's @viewer.bind_key("Escape") only fires when the canvas
        # has keyboard focus.  Clicking a toolbar button (Add, Track)
        # moves focus to the button, which absorbs or ignores Escape.
        # Install a QShortcut on the main window with ApplicationShortcut
        # context so Escape works regardless of which widget has focus.
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtGui import QKeySequence
            from qtpy.QtWidgets import QShortcut

            qt_window = self.viewer.window._qt_window  # type: ignore[attr-defined]
            self._escape_shortcut = QShortcut(QKeySequence("Escape"), qt_window)
            self._escape_shortcut.setContext(Qt.ApplicationShortcut)
            self._escape_shortcut.activated.connect(self._exit_all_modes)
        except Exception as e:
            logger.warning("Could not install application-wide Escape shortcut: %s", e)
