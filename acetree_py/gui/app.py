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
import os
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari
    from ..core.nucleus import Nucleus
    from ..core.roi_manager import RoiManager
    from ..tracking.api import (
        Calibration,
        ComponentSpec,
        Detection,
        TrackingRequest,
        TrackingResult,
    )

from ..core.nuclei_manager import NucleiManager
from ..editing.history import EditHistory, PostCommitCallbackError
from ..io.config import AceTreeConfig, load_config
from ..io.image_provider import (
    ImageProvider,
    clone_image_provider_for_worker,
    close_worker_image_provider,
    create_image_provider_from_config,
)
from .color_rules import ColorRuleEngine
from .marker_layers import (
    configure_curated_points_layer,
    passed_drag_threshold,
    point_anchor,
    pointer_position,
    replace_points_layer,
)

logger = logging.getLogger(__name__)


def _stage_config_xml(config: AceTreeConfig, destination: Path) -> Path:
    """Serialize a config to a private sibling for a coordinated Save."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".save-config.tmp",
    )
    os.close(descriptor)
    staged = Path(name)
    try:
        if destination.exists():
            os.chmod(staged, stat.S_IMODE(destination.stat().st_mode))
        from ..io.config_writer import write_config_xml

        # The sibling lives in the destination directory, so relative paths in
        # the staged XML are exactly those of the eventual config file.
        write_config_xml(config, staged)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged

# Java AceTree stored this constant as `NUCZINDEXOFFSET = 1`, but in our
# Python port ``nuc.z`` and ``current_plane`` are in the *same* 1-based
# coordinate system (see NucleiManager.find_closest_nucleus /
# nucleus_diameter, which compare ``nuc.z - image_plane`` directly).
# Adding +1 to the snap target therefore lands the viewer one plane above
# the true centroid — visible symptom: the slice follows a selected cell
# across time but stops one plane short of the nucleus.  Set to 0.
NUCZINDEXOFFSET = 0


@dataclass(frozen=True, slots=True)
class TrackingAnalysisSnapshot:
    """Immutable document context consumed by a background tracking run.

    Every nucleus is copied on the GUI thread before the worker starts. The
    provider is a template: built-in providers are cloned with independent
    file-handle caches in the worker. The monotonic change counter catches
    edit→undo sequences that return to the same history revision.
    """

    request: TrackingRequest
    image_provider: ImageProvider
    calibration: Calibration
    nuclei_record: list[list[Nucleus]]
    revision: int
    change_counter: int


@dataclass(frozen=True, slots=True)
class DetectorPreviewSnapshot:
    """Minimal immutable context for one background detector test.

    Unlike a tracking snapshot, this deliberately carries no nuclei-record
    copy because current-frame detection reads only the image and calibration.
    """

    detector: ComponentSpec
    frame: int
    image_provider: ImageProvider
    calibration: Calibration
    revision: int
    change_counter: int


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
        roi_manager: RoiManager | None = None,
    ) -> None:
        from ..analysis.roi_measurements import RoiMeasurementEngine
        from ..core.roi_manager import RoiManager

        self.manager = manager
        self.image_provider = image_provider
        self.roi_manager = roi_manager if roi_manager is not None else RoiManager()
        self.roi_measurement_engine = RoiMeasurementEngine(image_provider)
        self.edit_history = EditHistory(
            manager.nuclei_record,
            on_edit=self._on_edit,
        )

        # Navigation state
        self.current_time: int = 1
        self.current_plane: int = 1
        self.current_cell_name: str = ""
        # Stable physical anchor for the selection.  Cell names are mutable:
        # automatic naming, manual overrides, relinks, and undo/redo can all
        # change them.  Keeping the nucleus that was actually picked prevents
        # a rename elsewhere from stealing the selection and makes unnamed
        # selections safe across time navigation.
        self.selection_anchor: tuple[int, int] | None = None
        self.current_roi_object_id = None
        self.current_roi_class_id = None
        self.tracking: bool = True

        # Save As becomes the target for subsequent Save operations even for
        # headless/new managers that do not yet own an AceTreeConfig.
        self._save_path_override: Path | None = None

        # Accepted image-analysis runs are retained as provenance and written
        # to the optional tracking sidecar on Save.  They never replace the
        # legacy XML/nuclei ZIP contract.
        self._tracking_results: list[TrackingResult] = []
        self._tracking_sidecar_managed: bool = False
        # A rendering failure happens after an edit has already crossed the
        # history boundary. Retain the most recent failure so proposal
        # acceptance can make one redraw retry without executing the command
        # again (which would duplicate detections/markers).
        self._last_post_commit_refresh_error: Exception | None = None
        # Whole-dataset tracking is always proposal-first.  Dataset creation
        # retains the wizard request until the viewer exists so the result can
        # be inspected in the same 2D/3D overlays used for curation.
        self._pending_initial_tracking_request: TrackingRequest | None = None
        self._last_global_tracking_request: TrackingRequest | None = None
        self._global_tracking_dialog = None
        self._global_tracking_jobs: dict[tuple[int, int], tuple] = {}
        # A worker can be complete while its QThread is still draining queued
        # teardown events. Keep those Qt objects alive without treating the
        # analysis as active or blocking the next workbench.
        self._global_tracking_retiring_jobs: dict[tuple[int, int], tuple] = {}
        self._tracking_shutdown_connected = False

        # GUI components (initialized in launch())
        self.viewer: napari.Viewer | None = None
        self._viewer_integration = None
        self._player_controls = None
        self._cell_info_panel = None
        self._contrast_tools = None
        self._edit_panel = None
        self._subcellular_objects_panel = None
        self._roi_viewer_integration = None
        self._tracking_menu = None
        self._tracking_menu_actions: dict[str, object] = {}
        self._lineage_widgets: list = []  # Multiple lineage tree panels
        self._expression_plot_windows: list = []
        self._expression_plot_window_counter: int = 0
        self._expression_comparison_windows: list = []
        self._expression_comparison_window_counter: int = 0
        self._roi_scalar_plot_windows: list = []
        self._roi_profile_windows: list = []
        self._expression_dataset_repository = None
        self._expression_repository_shutdown_connected = False
        self._panel_menu_actions: dict[str, object] = {}
        self._lineage_list = None
        # 0-based image channel most recently chosen by File -> Measure.
        self.current_expression_channel: int = 0

        # Cached image layers (one per channel)
        self._image_layers: list = []
        # Default colormaps for multi-channel display (green/magenta)
        self._channel_colormaps = ["green", "magenta", "cyan", "yellow", "red", "blue"]

        # 3D view state
        self._3d_mode: bool = False
        self._changing_ndisplay: bool = False
        self._points_layer = None  # napari Points layer for 3D nuclei
        self._trail_points_layer = None  # 3D ghost trail Points layer

        # Relink pick mode state (Feature 4)
        self._relink_pick_mode: bool = False
        self._relink_pick_callback = None  # callable(time, nuc) when target picked

        # Click-to-place nucleus mode (Track button)
        self._placement_mode: bool = False
        self._placement_parent_name: str | None = None  # None = root mode
        self._placement_parent_anchor: tuple[int, int] | None = None
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

        from ..core.roi_manager import RoiManager

        roi_manager = RoiManager.from_config(
            config,
            image_provider=image_provider,
            num_timepoints=manager.num_timepoints,
        )
        app = cls(manager, image_provider, roi_manager=roi_manager)
        app.roi_manager.reconcile_cells(app._resolve_roi_cell_anchor)
        tracking_sidecar = config.zip_file.with_suffix(".tracking.json")
        if tracking_sidecar.exists():
            try:
                from ..tracking.persistence import read_tracking_proposal

                app._tracking_results.append(
                    read_tracking_proposal(tracking_sidecar)
                )
                app._tracking_sidecar_managed = True
            except Exception:
                # Tracking provenance is optional and must never make a
                # backward-compatible nuclei dataset impossible to open.
                logger.warning(
                    "Could not read tracking sidecar %s",
                    tracking_sidecar,
                    exc_info=True,
                )
        app.current_time = 1
        # Set initial plane to middle of stack
        plane_start = int(config.plane_start)
        if image_provider is not None and image_provider.num_planes > 0:
            app.current_plane = plane_start + (image_provider.num_planes - 1) // 2
        else:
            app.current_plane = plane_start + max(
                0, ((manager.movie.num_planes or 30) - 1) // 2
            )
        return app

    @classmethod
    def from_new_dataset(
        cls,
        config: AceTreeConfig,
        num_timepoints: int,
        output_dir: Path,
        tracking_request: TrackingRequest | None = None,
    ) -> AceTreeApp:
        """Create an AceTreeApp for a brand-new dataset (empty nuclei).

        Writes an empty nuclei ZIP and config XML to *output_dir*,
        then opens the GUI for manual annotation.

        Args:
            config: Configuration built from DatasetCreationDialog.
            num_timepoints: Number of timepoints detected from images.
            output_dir: Where to save the nuclei ZIP and config XML.
            tracking_request: Optional automated draft settings. The request
                remains uncommitted until the launched viewer's global review
                workbench explicitly accepts it. ``None`` preserves the
                manual-annotation workflow.

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

        from ..core.roi_manager import RoiManager

        app = cls(
            manager,
            image_provider,
            roi_manager=RoiManager.from_config(
                config,
                image_provider=image_provider,
                num_timepoints=manager.num_timepoints,
            ),
        )
        app.current_time = 1
        plane_start = int(config.plane_start)
        if image_provider is not None and image_provider.num_planes > 0:
            app.current_plane = plane_start + (image_provider.num_planes - 1) // 2
        else:
            app.current_plane = plane_start + max(
                0, ((manager.movie.num_planes or 30) - 1) // 2
            )
        if tracking_request is not None:
            app._pending_initial_tracking_request = tracking_request
            app._last_global_tracking_request = tracking_request
        return app

    @classmethod
    def from_dialog(cls) -> AceTreeApp | None:
        """Show the dataset creation dialog and create an app if accepted.

        Returns:
            An AceTreeApp if the user completes the dialog, None if cancelled.
        """
        from .dataset_dialog import DatasetCreationDialog

        # Need a QApplication for the dialog
        from qtpy.QtWidgets import QApplication, QDialog
        qt_app = QApplication.instance()
        if qt_app is None:
            qt_app = QApplication([])

        dlg = DatasetCreationDialog()
        if dlg.exec_() != QDialog.Accepted:
            return None

        config = dlg.get_config()
        output_dir = dlg.get_output_directory()
        dataset_name = dlg.get_dataset_name()
        num_timepoints = dlg.get_num_timepoints()
        tracking_request = dlg.get_tracking_request()

        # Set zip_file name from dataset name
        config.zip_file = output_dir / f"{dataset_name}.zip"

        return cls.from_new_dataset(
            config,
            num_timepoints,
            output_dir,
            tracking_request=tracking_request,
        )

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
        from .player_controls import PlayerControls
        from .roi_viewer_integration import RoiViewerIntegration
        from .subcellular_objects_panel import SubcellularObjectsPanel
        from .viewer_integration import ViewerIntegration

        self.viewer = napari.Viewer(title="AceTree")
        if not self._tracking_shutdown_connected:
            from qtpy.QtWidgets import QApplication

            qt_app = QApplication.instance()
            if qt_app is not None:
                qt_app.aboutToQuit.connect(self._shutdown_global_tracking_workers)
                self._tracking_shutdown_connected = True
        if not self._expression_repository_shutdown_connected:
            from qtpy.QtWidgets import QApplication

            qt_app = QApplication.instance()
            if qt_app is not None:
                qt_app.aboutToQuit.connect(
                    self._shutdown_expression_dataset_repository
                )
                self._expression_repository_shutdown_connected = True
        try:
            self.viewer.dims.events.ndisplay.connect(self._on_native_ndisplay_changed)
        except (AttributeError, TypeError):
            logger.debug("napari does not expose an ndisplay change event")

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
        self._roi_viewer_integration = RoiViewerIntegration(self)
        self._roi_viewer_integration.setup_layers()

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

        # Right: Edit & Tracking Tools (scrollable; D-pad/history are popups)
        self._edit_panel = EditPanel(self)
        self.viewer.window.add_dock_widget(
            self._edit_panel,
            name="Edit & Tracking Tools",
            area="right",
        )

        self._subcellular_objects_panel = SubcellularObjectsPanel(
            self,
            browse_only=False,
        )
        self._subcellular_objects_panel.objectSelected.connect(
            self._on_roi_object_selected
        )
        self._subcellular_objects_panel.modeChanged.connect(
            self._on_roi_mode_changed
        )
        self._subcellular_objects_panel.actionRequested.connect(
            self._on_roi_action_requested
        )
        self.viewer.window.add_dock_widget(
            self._subcellular_objects_panel,
            name="Subcellular Objects",
            area="right",
        )

        # Bottom: Lineage tree view (graphical Sulston tree)
        self.add_lineage_panel()

        # Cell Info is now a hover tooltip, not a dock widget.
        # Keep a reference for the tooltip builder but don't dock it.
        self._cell_info_panel = None

        # Add toggle actions to Window menu so closed panels can be reopened
        self._add_panel_menu_actions()
        # Tracking entry points must remain reachable even if the Edit &
        # Tracking dock is closed or scrolled on a small display.
        self._add_tracking_menu_actions()
        # Add File → Measure… action
        self._add_file_menu_actions()
        self._add_objects_menu_actions()

        # Keyboard shortcuts
        self._bind_keys()

        # Initial display
        self.update_display()

        if self._pending_initial_tracking_request is not None:
            # The review workbench needs ViewerIntegration's preview layers,
            # and image analysis must not start before Qt's event loop is able
            # to deliver progress and cancellation signals.
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, self._open_pending_initial_tracking)

        logger.info("AceTree GUI launched")

    def run(self) -> None:
        """Launch the viewer and start the Qt event loop."""
        self.launch()
        import napari
        napari.run()

    def run_tracking_request(
        self,
        request: TrackingRequest,
        *,
        progress=None,
        cancelled=None,
    ) -> TrackingResult:
        """Analyze images and accept the result as one undoable draft edit."""
        proposal, revision = self.analyze_tracking_request(
            request,
            progress=progress,
            cancelled=cancelled,
        )
        self.accept_tracking_proposal(proposal, expected_revision=revision)
        return proposal

    def analyze_tracking_request(
        self,
        request: TrackingRequest,
        *,
        progress=None,
        cancelled=None,
    ) -> tuple[TrackingResult, int]:
        """Return an uncommitted proposal and its source document revision.

        Synchronous callers retain the original API.  GUI workbenches call
        :meth:`prepare_tracking_analysis` on the GUI thread, then execute the
        returned snapshot in a worker with :meth:`analyze_prepared_tracking`.
        """

        snapshot = self.prepare_tracking_analysis(request)
        proposal = self.analyze_prepared_tracking(
            snapshot,
            progress=progress,
            cancelled=cancelled,
        )
        return proposal, snapshot.revision

    def prepare_tracking_analysis(
        self,
        request: TrackingRequest,
    ) -> TrackingAnalysisSnapshot:
        """Capture a worker-safe, immutable view of the current document."""

        if self.image_provider is None:
            raise ValueError("This dataset has no readable image source")
        config = self.manager.config
        if config is None:
            raise ValueError("Tracking requires dataset calibration")
        if (
            request.tracker.plugin_id == "acetree.starrynite_legacy_exact"
            and self.manager.num_timepoints != self.image_provider.num_timepoints
        ):
            raise ValueError(
                "Exact StarryNite whole-movie tracking requires the nuclei record "
                "and image source to describe the same number of timepoints "
                f"(record: {self.manager.num_timepoints}; images: "
                f"{self.image_provider.num_timepoints})."
            )

        from ..tracking.api import Calibration

        revision = self.edit_history.revision
        change_counter = self.edit_history.change_counter
        calibration = Calibration(
            xy_um=config.xy_res,
            z_um=config.z_res,
            plane_start=config.plane_start,
        )
        nuclei_snapshot = [
            [nucleus.copy() for nucleus in frame]
            for frame in self.manager.nuclei_record
        ]
        return TrackingAnalysisSnapshot(
            request=request,
            image_provider=self.image_provider,
            calibration=calibration,
            nuclei_record=nuclei_snapshot,
            revision=revision,
            change_counter=change_counter,
        )

    def analyze_prepared_tracking(
        self,
        snapshot: TrackingAnalysisSnapshot,
        *,
        progress=None,
        cancelled=None,
    ) -> TrackingResult:
        """Analyze a previously captured snapshot, normally in a worker."""

        from ..tracking.pipeline import TrackingPipeline

        if (
            self.edit_history.revision != snapshot.revision
            or self.edit_history.change_counter != snapshot.change_counter
        ):
            raise RuntimeError(
                "The dataset changed before tracking started; recompute the draft"
            )
        worker_provider = clone_image_provider_for_worker(snapshot.image_provider)
        owns_provider = worker_provider is not None
        if worker_provider is None:
            # Compatibility for external/in-memory providers that predate the
            # clone contract. Built-in disk providers never take this path.
            worker_provider = snapshot.image_provider
        try:
            proposal = TrackingPipeline().run(
                worker_provider,
                snapshot.calibration,
                snapshot.request,
                nuclei_record=snapshot.nuclei_record,
                progress=progress,
                cancelled=cancelled,
            )
        finally:
            if owns_provider:
                close_worker_image_provider(worker_provider)
        if (
            self.edit_history.revision != snapshot.revision
            or self.edit_history.change_counter != snapshot.change_counter
        ):
            raise RuntimeError(
                "The dataset changed while tracking was running; recompute the draft"
            )
        return proposal

    def prepare_detector_preview(
        self,
        detector: ComponentSpec,
        frame: int,
    ) -> DetectorPreviewSnapshot:
        """Capture the small worker-safe context for a one-frame detector test."""

        if self.image_provider is None:
            raise ValueError("This dataset has no readable image source")
        config = self.manager.config
        if config is None:
            raise ValueError("Detector preview requires dataset calibration")
        frame = int(frame)
        if frame < 1 or frame > self.image_provider.num_timepoints:
            raise ValueError(
                f"Detector preview frame {frame} is outside the image source"
            )

        from ..tracking.api import Calibration

        return DetectorPreviewSnapshot(
            detector=detector,
            frame=frame,
            image_provider=self.image_provider,
            calibration=Calibration(
                xy_um=config.xy_res,
                z_um=config.z_res,
                plane_start=config.plane_start,
            ),
            revision=self.edit_history.revision,
            change_counter=self.edit_history.change_counter,
        )

    def analyze_prepared_detector_preview(
        self,
        snapshot: DetectorPreviewSnapshot,
        *,
        progress=None,
        cancelled=None,
    ) -> tuple[Detection, ...]:
        """Run one detector without copying nuclei or constructing a tracker."""

        from ..tracking.pipeline import TrackingPipeline

        if (
            self.edit_history.revision != snapshot.revision
            or self.edit_history.change_counter != snapshot.change_counter
        ):
            raise RuntimeError(
                "The dataset changed before detector preview started; run it again"
            )
        worker_provider = clone_image_provider_for_worker(snapshot.image_provider)
        owns_provider = worker_provider is not None
        if worker_provider is None:
            worker_provider = snapshot.image_provider
        try:
            detections = TrackingPipeline().detect_frame(
                worker_provider,
                snapshot.calibration,
                snapshot.detector,
                frame=snapshot.frame,
                progress=progress,
                cancelled=cancelled,
            )
        finally:
            if owns_provider:
                close_worker_image_provider(worker_provider)
        if (
            self.edit_history.revision != snapshot.revision
            or self.edit_history.change_counter != snapshot.change_counter
        ):
            raise RuntimeError(
                "The dataset changed while detector preview was running; run it again"
            )
        return detections

    def accept_tracking_proposal(
        self,
        proposal: TrackingResult,
        *,
        expected_revision: int,
    ) -> dict[str, tuple[int, int]]:
        """Commit a reviewed proposal and return detection-to-nucleus locations."""
        if self.edit_history.revision != expected_revision:
            raise RuntimeError(
                "The dataset changed after preview; recompute the tracking draft"
            )
        config = self.manager.config
        if config is None:
            raise ValueError("Tracking requires dataset calibration")

        from ..tracking.api import Calibration
        from ..tracking.integration import ApplyTrackingProposal

        calibration = Calibration(
            xy_um=config.xy_res,
            z_um=config.z_res,
            plane_start=config.plane_start,
        )
        command = ApplyTrackingProposal(result=proposal, calibration=calibration)
        self._last_post_commit_refresh_error = None
        try:
            self.edit_history.do(command)
        except PostCommitCallbackError as error:
            # ``EditHistory`` raises this only after data, revision, and undo
            # state have committed. Reading the mapping also verifies that
            # this particular proposal is applied before we report success.
            # Retry only the post-commit work -- never execute the command a
            # second time, which would create overlapping duplicate nuclei.
            mapping = command.detection_mapping
            self._report_committed_refresh_failure(
                command,
                error.__cause__ or error,
            )
            try:
                self.edit_history.retry_post_commit(error)
            except Exception as retry_error:
                self._report_committed_refresh_failure(command, retry_error)
            return mapping

        mapping = command.detection_mapping
        if self._last_post_commit_refresh_error is not None:
            # Structural/model rebuilding completed, but the presentation
            # refresh failed. One direct redraw retry is safe and avoids an
            # expensive second naming pass. A failed retry remains a warning
            # because the proposal itself is already committed and undoable.
            try:
                self.update_display()
            except Exception as retry_error:
                self._report_committed_refresh_failure(command, retry_error)
        return mapping

    def _open_pending_initial_tracking(self) -> None:
        """Open the wizard request and test its detector on one frame first."""

        request = self._pending_initial_tracking_request
        if request is not None:
            self.set_time(request.scope.start_frame)
            dialog = self.open_global_tracking_workbench(
                initial_request=request,
                auto_preview_current_frame=True,
            )
            if dialog is not None:
                self._pending_initial_tracking_request = None

    def open_global_tracking_workbench(
        self,
        *,
        initial_request: TrackingRequest | None = None,
        auto_start: bool = False,
        auto_preview_current_frame: bool = False,
    ):
        """Show proposal-first whole-dataset tracking for an empty dataset."""

        if self._global_tracking_dialog is not None:
            try:
                if self._global_tracking_dialog.isVisible():
                    self._global_tracking_dialog.raise_()
                    self._global_tracking_dialog.activateWindow()
                    return self._global_tracking_dialog
            except RuntimeError:
                self._global_tracking_dialog = None

        from qtpy.QtWidgets import QMessageBox

        if self._global_tracking_jobs:
            parent = None
            if self.viewer is not None:
                parent = self.viewer.window._qt_window
            QMessageBox.information(
                parent,
                "Previous Analysis Is Stopping",
                "Please wait for the previous whole-movie analysis to finish "
                "canceling before starting another run.",
            )
            return None
        if self.viewer is None:
            logger.warning("Global tracking review requires the launched viewer")
            return None

        if self.image_provider is None:
            QMessageBox.warning(
                self.viewer.window._qt_window,
                "Images Unavailable",
                "Whole-dataset tracking needs a readable image source.",
            )
            return None
        if any(frame for frame in self.manager.nuclei_record):
            QMessageBox.information(
                self.viewer.window._qt_window,
                "Dataset Is Not Empty",
                "Whole-dataset tracking currently adds a new initial draft and is only "
                "available before curation begins. Undo the accepted initial draft, or "
                "use Track Selected Cell for a selected lineage.",
            )
            return None

        from qtpy.QtCore import QTimer

        from ..tracking.api import Calibration
        from .global_tracking_dialog import GlobalTrackingDialog

        config = self.manager.config
        if config is None:
            QMessageBox.warning(
                self.viewer.window._qt_window,
                "Calibration Unavailable",
                "Whole-dataset tracking needs pixel and Z calibration.",
            )
            return None
        record_timepoints = self.manager.num_timepoints
        image_timepoints = self.image_provider.num_timepoints
        end_time = min(record_timepoints, image_timepoints)
        exact_scope_error = ""
        if record_timepoints != image_timepoints:
            exact_scope_error = (
                "Exact StarryNite whole-movie tracking is unavailable because "
                f"the nuclei record has {record_timepoints} timepoint(s), while "
                f"the image source has {image_timepoints}. Reopen the dataset with "
                "matching movie and nuclei ranges."
            )
        request = initial_request or self._last_global_tracking_request
        dialog = GlobalTrackingDialog(
            start_time=1,
            end_time=max(1, end_time),
            num_channels=max(1, self.image_provider.num_channels),
            parent=self.viewer.window._qt_window,
            initial_request=request,
            viewer_integration=self._viewer_integration,
            calibration=Calibration(
                config.xy_res,
                config.z_res,
                config.plane_start,
            ),
            analysis_starter=self._start_global_tracking_analysis,
            detector_preview_starter=self._start_global_detector_preview,
            accept_callback=lambda proposal, revision: self.accept_tracking_proposal(
                proposal,
                expected_revision=int(revision),
            ),
            revision_getter=lambda: (
                self.edit_history.revision,
                self.edit_history.change_counter,
            ),
            navigate_to_frame=self.set_time,
            current_frame_getter=lambda: self.current_time,
            dataset_empty_getter=lambda: not any(
                frame for frame in self.manager.nuclei_record
            ),
            exact_scope_error=exact_scope_error,
        )
        dialog.finished.connect(
            lambda _result, dlg=dialog: self._global_tracking_dialog_closed(dlg)
        )
        dialog.draftAccepted.connect(
            lambda count: self._set_tracking_status(
                f"Accepted {count} reviewed whole-movie positions"
            )
        )
        dialog.draftDiscarded.connect(
            lambda: self._set_tracking_status(
                "Discarded the whole-movie draft; no positions were added"
            )
        )
        self._global_tracking_dialog = dialog
        self._pause_playback_for_tracking_review()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        if auto_start:
            QTimer.singleShot(0, dialog.start_analysis)
        elif auto_preview_current_frame:
            QTimer.singleShot(0, dialog.start_detector_preview)
        return dialog

    def _start_global_tracking_analysis(self, request, run_id: int, dialog):
        """Start one cancellable worker and return its cancellation handle."""

        snapshot = self.prepare_tracking_analysis(request)

        def analysis(progress, cancelled):
            proposal = self.analyze_prepared_tracking(
                snapshot,
                progress=progress,
                cancelled=cancelled,
            )
            return (
                proposal,
                snapshot.revision,
                run_id,
                (snapshot.revision, snapshot.change_counter),
            )

        return self._start_global_tracking_worker(
            analysis,
            run_id,
            dialog,
            success_slot=dialog.finish_worker_result,
            failure_slot=dialog.fail_analysis,
        )

    def _start_global_detector_preview(
        self,
        detector,
        frame: int,
        run_id: int,
        dialog,
    ):
        """Start one lightweight detector-only current-frame worker."""

        snapshot = self.prepare_detector_preview(detector, frame)

        def analysis(progress, cancelled):
            detections = self.analyze_prepared_detector_preview(
                snapshot,
                progress=progress,
                cancelled=cancelled,
            )
            return (
                detections,
                snapshot.frame,
                run_id,
                (snapshot.revision, snapshot.change_counter),
            )

        return self._start_global_tracking_worker(
            analysis,
            run_id,
            dialog,
            success_slot=dialog.finish_detector_preview_worker_result,
            failure_slot=dialog.fail_detector_preview,
        )

    def _start_global_tracking_worker(
        self,
        analysis,
        run_id: int,
        dialog,
        *,
        success_slot,
        failure_slot,
    ):
        """Run either tracking mode with kind-safe queued Qt callbacks."""

        from threading import Event

        from qtpy.QtCore import QThread

        from .tracking_worker import TrackingAnalysisWorker, TrackingWorkerRelay

        cancel_event = Event()

        def cancel_matching_run(requested_run_id: int) -> None:
            # The worker can enter plugin code before ``start_analysis`` has
            # received and stored the returned cancellation handle. Keep a
            # direct signal path so closing during that startup window cannot
            # strand a background job.
            if int(requested_run_id) == int(run_id):
                cancel_event.set()

        thread = QThread()
        worker = TrackingAnalysisWorker(analysis, cancel_event)
        job_key = (id(dialog), run_id)
        relay = TrackingWorkerRelay(
            run_id,
            dialog.update_analysis_progress,
            success_slot,
            failure_slot,
            lambda key=job_key: self._global_tracking_worker_finished(key),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(relay.progress)
        worker.succeeded.connect(relay.succeeded)
        worker.failed.connect(relay.failed)
        # This queued GUI-thread callback clears the active-job guard as soon
        # as plugin code returns. The tuple moves to a retiring collection so
        # the QThread cannot be destroyed before its own ``finished`` signal.
        worker.finished.connect(relay.finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda key=job_key: self._global_tracking_thread_finished(key)
        )
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(relay.deleteLater)
        dialog.cancelRequested.connect(cancel_matching_run)
        self._global_tracking_jobs[job_key] = (
            thread,
            worker,
            cancel_event,
            relay,
            cancel_matching_run,
        )
        thread.start()
        return cancel_event.set

    def _global_tracking_worker_finished(self, job_key: tuple[int, int]) -> None:
        """Mark completed plugin code inactive while retaining Qt teardown refs."""

        job = self._global_tracking_jobs.pop(job_key, None)
        if job is not None:
            self._global_tracking_retiring_jobs[job_key] = job

    def _global_tracking_thread_finished(self, job_key: tuple[int, int]) -> None:
        """Release worker references after Qt has dispatched ``finished``.

        Dropping the last Python reference to a ``QThread`` from inside its own
        ``finished`` signal can destroy the wrapper while Qt is still unwinding
        that signal.  This is especially easy to hit when a canceled analysis
        is immediately followed by another workbench.  Keep the retiring tuple
        alive for one GUI turn so destruction happens from the ordinary event
        loop instead of the thread's completion callback.
        """

        from qtpy.QtCore import QTimer

        QTimer.singleShot(
            0,
            lambda key=job_key: self._release_global_tracking_job(key),
        )

    def _release_global_tracking_job(self, job_key: tuple[int, int]) -> None:
        """Drop references for a worker whose Qt completion signal has returned."""

        self._global_tracking_jobs.pop(job_key, None)
        self._global_tracking_retiring_jobs.pop(job_key, None)

    def _global_tracking_dialog_closed(self, dialog) -> None:
        # Closing is allowed during the tiny interval between ``thread.start``
        # and the dialog receiving its callable cancellation handle. Reinforce
        # the dialog signal at the host boundary for every run it owns.
        for (dialog_id, _run_id), job in tuple(self._global_tracking_jobs.items()):
            if dialog_id == id(dialog):
                job[2].set()
        try:
            self._last_global_tracking_request = dialog.get_request()
        except (AttributeError, RuntimeError, ValueError):
            pass
        if self._global_tracking_dialog is dialog:
            self._global_tracking_dialog = None

    def _shutdown_global_tracking_workers(self, timeout_ms: int = 1500) -> None:
        """Cancel global workers and allow a bounded cooperative shutdown."""

        import time

        jobs = [
            *self._global_tracking_jobs.values(),
            *self._global_tracking_retiring_jobs.values(),
        ]
        for _thread, _worker, cancel_event, *_relay in jobs:
            cancel_event.set()
        deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
        for thread, _worker, _cancel_event, *_relay in jobs:
            try:
                if not thread.isRunning():
                    continue
                thread.quit()
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if remaining_ms:
                    thread.wait(remaining_ms)
                if thread.isRunning():
                    logger.warning(
                        "A tracking plugin did not stop within the shutdown timeout"
                    )
            except RuntimeError:
                # Qt may already have released a thread that finished while
                # shutdown callbacks were being delivered.
                continue

    def _pause_playback_for_tracking_review(self) -> None:
        player = self._player_controls
        if player is not None and bool(getattr(player, "_playing", False)):
            stop = getattr(player, "_stop_play", None)
            if callable(stop):
                stop()

    def _set_tracking_status(self, message: str) -> None:
        if self._edit_panel is not None:
            self._edit_panel._status_label.setText(message)

    # ── Save ──────────────────────────────────────────────────────

    @property
    def _default_save_path(self) -> Path | None:
        """Return the original nuclei ZIP path from config, if available."""
        if self._save_path_override is not None:
            return self._save_path_override
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

        target_path = Path(path_str)
        config = self.manager.config
        config_path = Path(config.config_file) if config is not None else Path()
        has_xml = config_path != Path() and config_path.suffix.lower() == ".xml"
        saved_path = self._do_save(
            target_path,
            mark_saved=False,
            include_roi=not has_xml,
            roi_destination=(
                None if has_xml else target_path.with_suffix(".subcellular-rois.json")
            ),
        )
        if saved_path is None:
            return None

        old_zip_path = config.zip_file if config is not None else None
        if config is not None:
            config.zip_file = saved_path
            if has_xml:
                config_stage = None
                roi_stage = None
                try:
                    from ..io.dataset_transaction import DatasetTransaction
                    from ..io.roi_sidecar import (
                        RoiSidecarConflictError,
                        read_roi_sidecar,
                        roi_sidecar_path,
                    )

                    config_stage = _stage_config_xml(config, config_path)
                    transaction = DatasetTransaction()
                    if self.roi_manager.is_dirty:
                        roi_destination = (
                            self.roi_manager.sidecar_path
                            or roi_sidecar_path(config_path)
                        )
                        roi_stage = self.roi_manager.stage_save(roi_destination)
                        sidecar = roi_stage.sidecar
                        precondition = None
                        if not roi_stage.replaces_protected_sidecar:
                            expected = sidecar.expected_token
                            destination = sidecar.destination

                            def ensure_roi_unchanged() -> None:
                                loaded = read_roi_sidecar(destination)
                                current = None if loaded is None else loaded.token
                                if current != expected:
                                    raise RoiSidecarConflictError(
                                        "ROI sidecar changed externally immediately "
                                        f"before Save As: {destination}"
                                    )

                            precondition = ensure_roi_unchanged
                        transaction.add(
                            sidecar.temp_path,
                            sidecar.destination,
                            precondition=precondition,
                        )
                    transaction.add(config_stage, config_path)
                    transaction.commit()
                    if roi_stage is not None:
                        self.roi_manager.finalize_external_commit(roi_stage)
                except Exception:
                    # The target ZIP is a valid standalone copy, but Save As
                    # is not a successful retarget unless the source config
                    # will reopen it.  Keep both in-memory and on-disk config
                    # pointing at the previous dataset and leave history dirty.
                    config.zip_file = old_zip_path
                    logger.exception(
                        "Saved nuclei to %s but could not update config %s",
                        saved_path,
                        config_path,
                    )
                    from qtpy.QtWidgets import QMessageBox

                    QMessageBox.critical(
                        self.viewer.window._qt_window,
                        "Save As Incomplete",
                        "The nuclei copy was written, but the dataset config "
                        "could not be updated. The current Save target was "
                        "not changed.",
                    )
                    return None
                finally:
                    if roi_stage is not None:
                        self.roi_manager.discard_save(roi_stage)
                    if config_stage is not None:
                        config_stage.unlink(missing_ok=True)

        self.manager._config_dirty = False
        self._save_path_override = saved_path
        self.edit_history.mark_saved()
        return saved_path

    def _do_save(
        self,
        path: Path,
        *,
        mark_saved: bool = True,
        include_roi: bool = True,
        roi_destination: Path | None = None,
    ) -> Path | None:
        """Write authoritative dataset artifacts and report success/failure."""
        config_stage: Path | None = None
        roi_stage = None
        try:
            config = self.manager.config
            final_commit = None
            if (
                mark_saved
                and config is not None
                and bool(getattr(self.manager, "_config_dirty", False))
            ):
                config_path = Path(config.config_file)
                if config_path != Path() and config_path.suffix.lower() == ".xml":
                    config_stage = _stage_config_xml(config, config_path)
                    staged_path = config_stage

                    def commit_config() -> None:
                        os.replace(staged_path, config_path)

                    final_commit = commit_config

            # A clean sidecar still belongs to the old dataset until Save As
            # publishes it at the new destination. Do not create optional empty
            # sidecars for datasets that have never contained ROI state.
            roi_retarget = (
                roi_destination is not None
                and (
                    self.roi_manager.sidecar_path is None
                    or roi_destination.resolve(strict=False)
                    != self.roi_manager.sidecar_path.resolve(strict=False)
                )
                and bool(
                    self.roi_manager.sidecar_path
                    or self.roi_manager.objects
                    or self.roi_manager.classes
                )
            )
            if include_roi and (self.roi_manager.is_dirty or roi_retarget):
                if roi_destination is None:
                    roi_destination = self.roi_manager.sidecar_path
                if roi_destination is None:
                    from ..io.roi_sidecar import roi_sidecar_path

                    roi_destination = roi_sidecar_path(None, path)
                roi_stage = self.roi_manager.stage_save(roi_destination)

            if roi_stage is not None or config_stage is not None:
                from ..io.dataset_transaction import DatasetTransaction
                from ..io.roi_sidecar import (
                    RoiSidecarConflictError,
                    read_roi_sidecar,
                )

                transaction = DatasetTransaction()
                if roi_stage is not None:
                    sidecar = roi_stage.sidecar
                    precondition = None
                    if not roi_stage.replaces_protected_sidecar:
                        expected = sidecar.expected_token
                        destination = sidecar.destination

                        def ensure_roi_unchanged() -> None:
                            loaded = read_roi_sidecar(destination)
                            current = None if loaded is None else loaded.token
                            if current != expected:
                                raise RoiSidecarConflictError(
                                    "ROI sidecar changed externally immediately "
                                    f"before save: {destination}"
                                )

                        precondition = ensure_roi_unchanged
                    transaction.add(
                        sidecar.temp_path,
                        sidecar.destination,
                        precondition=precondition,
                    )
                if config_stage is not None:
                    transaction.add(config_stage, config_path)

                def commit_authoritative_stages() -> None:
                    transaction.commit()

                final_commit = commit_authoritative_stages

            if final_commit is None:
                self.manager.save(path)
            else:
                self.manager.save(path, final_commit=final_commit)
            if roi_stage is not None:
                self.roi_manager.finalize_external_commit(roi_stage)
            if config_stage is not None:
                self.manager._config_dirty = False
            if self._tracking_results:
                try:
                    from ..tracking.persistence import (
                        tracking_sidecar_path,
                        write_tracking_proposal,
                    )

                    write_tracking_proposal(
                        tracking_sidecar_path(path),
                        self._tracking_results[-1],
                    )
                    self._tracking_sidecar_managed = True
                except Exception:
                    logger.warning(
                        "Authoritative data saved, but tracking provenance did not",
                        exc_info=True,
                    )
            elif self._tracking_sidecar_managed:
                try:
                    from ..tracking.persistence import tracking_sidecar_path

                    tracking_sidecar_path(path).unlink(missing_ok=True)
                    self._tracking_sidecar_managed = False
                except OSError:
                    logger.warning(
                        "Authoritative data saved, but stale tracking provenance "
                        "could not be removed",
                        exc_info=True,
                    )
            if mark_saved:
                self.edit_history.mark_saved()
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
        finally:
            if roi_stage is not None:
                self.roi_manager.discard_save(roi_stage)
            if config_stage is not None:
                try:
                    config_stage.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Could not remove staged config after Save: %s",
                        config_stage,
                        exc_info=True,
                    )

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
        self._exit_roi_mode()
        self.current_time = time

        # Track cell across time
        if self.tracking and self.current_cell_name:
            self._track_cell_at_time()

        self.update_display()

    def set_plane(self, plane: int, *, user_initiated: bool = False) -> None:
        """Navigate to a specific z-plane.

        The active cell is bound to the displayed slice: selecting a nucleus
        snaps the view onto its centroid z (see ``_snap_plane_to_nucleus``).
        Java AceTree therefore treats a user's Z move as leaving that cell —
        once the user scrolls off the nucleus's slice, the highlighted cell no
        longer corresponds to what is on screen.  We match that: when
        *user_initiated* is true (Up/Down and W/S keys, the ``z=`` spinbox, and
        the ± plane buttons), an actual plane change clears the selection and
        stops follow mode.

        Programmatic navigation passes ``user_initiated=False`` (the default)
        and preserves the selection: auto-tracking review and tracking-preview
        centering move the slice *on behalf of* the active cell, so dropping
        the selection there would defeat the feature.

        3D mode is exempt for the same reason the rule exists.  There is no
        slice on screen to scroll off, so a Z gesture cannot walk the view
        away from the active cell — deselecting would just look like the
        selection vanished for no reason.

        Args:
            plane: 1-based z-plane index.
            user_initiated: True when the move came straight from a user Z
                navigation gesture, which deselects the active cell.
        """
        plane_start, plane_end = self._plane_bounds()
        plane = max(plane_start, min(plane, plane_end))
        if plane == self.current_plane:
            return
        self._exit_roi_mode()
        had_selection = bool(self.current_cell_name) or self.selection_anchor is not None
        self.current_plane = plane
        if user_initiated and had_selection and not self._3d_mode:
            # Clear the selection *after* committing the new plane and refresh
            # exactly once: deselect_cell() would redraw a second time, and a
            # redraw ordered before the plane assignment would render the old
            # slice.  The user's plane is what must survive this call.
            self._clear_selection_state()
            self.update_display()
        elif self.current_cell_name:
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
        """Go to the next z-plane (user gesture: Up / W key, ▲ button)."""
        self.set_plane(self.current_plane + 1, user_initiated=True)

    def prev_plane(self) -> None:
        """Go to the previous z-plane (user gesture: Down / S key, ▼ button)."""
        self.set_plane(self.current_plane - 1, user_initiated=True)

    def _nucleus_at_anchor(self, anchor: tuple[int, int] | None = None):
        """Return the raw nucleus at an immutable ``(time, index)`` anchor."""
        if anchor is None:
            anchor = self.selection_anchor
        if anchor is None:
            return None
        time, index = anchor
        t_idx = time - 1
        n_idx = index - 1
        nr = self.manager.nuclei_record
        if not (0 <= t_idx < len(nr)):
            return None
        if not (0 <= n_idx < len(nr[t_idx])):
            return None
        return nr[t_idx][n_idx]

    def _cell_for_nucleus(self, time: int, nuc):
        """Resolve the lineage Cell containing *nuc* without using its name."""
        tree = self.manager.lineage_tree
        if tree is None:
            return None
        if nuc.hash_key:
            cell = tree.cells_by_hash.get(nuc.hash_key)
            if cell is not None:
                return cell

        # Defensive fallback for hand-built trees without nucleus hash keys.
        for cell in tree.all_cells():
            if any(t == time and candidate is nuc for t, candidate in cell.nuclei):
                return cell
        return None

    def _resolve_roi_cell_anchor(self, time: int, index: int):
        """Resolve one persisted physical ROI association fail-closed."""

        from ..core.subcellular_roi import CellResolution, NucleusAnchor

        t0 = int(time) - 1
        i0 = int(index) - 1
        record = self.manager.nuclei_record
        if not (0 <= t0 < len(record)) or not (0 <= i0 < len(record[t0])):
            return None
        nucleus = record[t0][i0]
        if not nucleus.is_alive:
            return None
        cell = self._cell_for_nucleus(int(time), nucleus)
        if cell is None:
            return None
        birth_time, birth_nucleus = min(cell.nuclei, key=lambda item: item[0])
        return CellResolution(
            value=cell,
            name=cell.name,
            birth_anchor=NucleusAnchor(
                timepoint=int(birth_time),
                index=int(birth_nucleus.index),
            ),
            centroid_xyz_px=(
                float(nucleus.x),
                float(nucleus.y),
                float(nucleus.z),
            ),
        )

    def _selection_name(self, time: int, nuc) -> str:
        """Return the current display name for a physically anchored nucleus."""
        if not nuc.effective_name:
            # A bare ``idx=N`` can target an unrelated nucleus after a time
            # scrub, so raw fallbacks are always time-qualified.
            return f"idx={time}:{nuc.index}"
        cell = self._cell_for_nucleus(time, nuc)
        return cell.name if cell is not None else nuc.effective_name

    def _set_selection_from_nucleus(self, time: int, nuc) -> None:
        """Select a concrete nucleus and derive its mutable name from it."""
        old_name = self.current_cell_name
        self.selection_anchor = (time, nuc.index)
        self.current_cell_name = self._selection_name(time, nuc)
        self.tracking = True
        if self._viewer_integration is not None:
            if old_name and old_name != self.current_cell_name:
                self._viewer_integration._shown_labels.discard(old_name)
            if self.current_cell_name:
                self._viewer_integration._shown_labels.add(self.current_cell_name)

    def _resolve_selection_after_rebuild(self) -> None:
        """Re-resolve the selected name from its stable physical anchor."""
        if self.selection_anchor is None:
            # Compatibility for callers/tests that still assign the name
            # directly.  Normal GUI selection paths set the anchor eagerly.
            if not self.current_cell_name:
                return
            cell = self.manager.get_cell(self.current_cell_name)
            if cell is None:
                return
            nuc = cell.get_nucleus_at(self.current_time)
            if nuc is None and cell.nuclei:
                anchor_time, nuc = min(
                    cell.nuclei, key=lambda item: abs(item[0] - self.current_time)
                )
            else:
                anchor_time = self.current_time
            if nuc is None:
                return
            self.selection_anchor = (anchor_time, nuc.index)

        nuc = self._nucleus_at_anchor()
        if nuc is None or not nuc.is_alive:
            self.current_cell_name = ""
            self.selection_anchor = None
            self.tracking = False
            return

        anchor_time, _ = self.selection_anchor
        old_name = self.current_cell_name
        self.current_cell_name = self._selection_name(anchor_time, nuc)
        if self._viewer_integration is not None and old_name != self.current_cell_name:
            if old_name:
                self._viewer_integration._shown_labels.discard(old_name)
            if self.current_cell_name:
                self._viewer_integration._shown_labels.add(self.current_cell_name)

    def get_selected_nucleus(self, time: int | None = None):
        """Return ``(nucleus, time, index)`` for the stable selection."""
        target_time = self.current_time if time is None else time

        if self.selection_anchor is None and self.current_cell_name:
            cell = self.manager.get_cell(self.current_cell_name)
            if cell is not None:
                nuc = cell.get_nucleus_at(target_time)
                if nuc is None and cell.nuclei:
                    anchor_time, nuc = min(
                        cell.nuclei,
                        key=lambda item: abs(item[0] - target_time),
                    )
                else:
                    anchor_time = target_time
                if nuc is not None:
                    self.selection_anchor = (anchor_time, nuc.index)
            elif self.current_cell_name.startswith("idx="):
                raw = self.current_cell_name[4:]
                try:
                    if ":" in raw:
                        anchor_time, anchor_index = (
                            int(value) for value in raw.split(":", 1)
                        )
                    else:  # qualify legacy in-memory state immediately
                        anchor_time, anchor_index = target_time, int(raw)
                    self.selection_anchor = (anchor_time, anchor_index)
                except ValueError:
                    return None

        anchor = self.selection_anchor
        anchor_nuc = self._nucleus_at_anchor(anchor)
        if anchor is None or anchor_nuc is None:
            return None
        anchor_time, _ = anchor
        if target_time == anchor_time:
            return anchor_nuc, target_time, anchor_nuc.index

        cell = self._cell_for_nucleus(anchor_time, anchor_nuc)
        if cell is not None:
            nuc = cell.get_nucleus_at(target_time)
            if nuc is not None:
                return nuc, target_time, nuc.index
        return None

    def get_selected_cell(self):
        """Return the physically selected lineage cell.

        Names are intentionally not used when a selection anchor exists:
        disconnected cells can temporarily share an effective name while a
        conflict is being corrected.  Name lookup remains only as a legacy
        bridge for programmatic callers that set ``current_cell_name``
        directly without selecting a nucleus.
        """
        if self.selection_anchor is None:
            # Qualify legacy name-only state into a physical anchor while the
            # current tree still provides the lookup context.
            if self.get_selected_nucleus() is None:
                return None

        anchor = self.selection_anchor
        nuc = self._nucleus_at_anchor(anchor)
        if anchor is None or nuc is None or not nuc.is_alive:
            return None
        anchor_time, _ = anchor
        return self._cell_for_nucleus(anchor_time, nuc)

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

        self.current_cell_name = cell.name
        # Explicitly selecting a cell re-enables follow-mode.  This undoes
        # any prior ↑/↓ Z nudge that disabled tracking, so subsequent
        # time-scrubbing snaps the slice back to the selected cell.
        self.tracking = True

        if time is not None:
            self.current_time = max(cell.start_time, min(time, cell.end_time))
        elif self.current_time < cell.start_time or self.current_time > cell.end_time:
            self.current_time = cell.start_time

        nuc = cell.get_nucleus_at(self.current_time)
        if nuc is None and cell.nuclei:
            anchor_time, nuc = min(
                cell.nuclei, key=lambda item: abs(item[0] - self.current_time)
            )
        else:
            anchor_time = self.current_time
        self.selection_anchor = (
            (anchor_time, nuc.index) if nuc is not None else None
        )

        # Track to cell's z-plane
        self._track_cell_at_time()

        # Show label for the selected cell
        if self._viewer_integration is not None:
            self._viewer_integration._shown_labels.add(self.current_cell_name)

        self.update_display()

    def _clear_selection_state(self) -> None:
        """Drop the active selection without refreshing the display.

        Split out of ``deselect_cell()`` so callers that already own a redraw
        (notably user-initiated ``set_plane()``) can clear the selection and
        still emit exactly one ``update_display()``.
        """
        self.current_cell_name = ""
        self.selection_anchor = None
        self.tracking = False

    def deselect_cell(self) -> None:
        """Clear the current cell selection and stop follow-mode."""
        self._clear_selection_state()
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
        if nuc:
            # Unnamed nucleus — highlight it and re-enable tracking so
            # subsequent time-scrubbing still snaps Z to follow it (via
            # the predecessor/successor chain fallback in
            # _track_cell_at_time).
            self._set_selection_from_nucleus(self.current_time, nuc)
            self.update_display()
        else:
            self.deselect_cell()

    # ── Relink pick mode (Feature 4) ─────────────────────────────

    def enter_relink_pick_mode(self, callback) -> None:
        """Enter pick mode for interactive relink.

        While in pick mode, right-clicking in the image selects a target
        nucleus and calls *callback(time, nucleus)* with the pick result.

        Args:
            callback: Called with (time: int, nuc: Nucleus) when user picks.
        """
        # Interaction modes are exclusive.  A relink target click must never
        # also be interpreted as an Add/Track gesture afterward.
        self._exit_roi_mode()
        self.exit_add_mode()
        self.exit_placement_mode()
        if self._edit_panel is not None:
            self._edit_panel._btn_add.setChecked(False)
            self._edit_panel._btn_track.setChecked(False)
        self._relink_pick_mode = True
        self._relink_pick_callback = callback
        self._focus_viewer_canvas()

    def exit_relink_pick_mode(self) -> None:
        """Exit pick mode without choosing a target."""
        self._relink_pick_mode = False
        self._relink_pick_callback = None

    def cancel_relink_pick_mode(self) -> None:
        """Cancel relink and synchronize the panel's pending source state."""
        self.exit_relink_pick_mode()
        if self._edit_panel is not None:
            cancel = getattr(self._edit_panel, "_on_relink_cancelled", None)
            if cancel is not None:
                cancel()

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
        switch_from_3d = self._3d_mode
        self._exit_roi_mode()
        if self._relink_pick_mode:
            self.cancel_relink_pick_mode()
        self.exit_placement_mode()
        if self._edit_panel is not None:
            self._edit_panel._btn_track.setChecked(False)
        self._add_mode = True
        if switch_from_3d:
            # A 3D camera ray does not supply an unambiguous Z placement.
            # Arm Add before switching so the toolbar remains checked when
            # the 2D view refreshes.
            self.set_3d_mode(False)
            self._say("Add placement uses the 2D slice view")
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
        if self._3d_mode and self._points_layer is not None:
            self._make_curated_points_read_only(self._points_layer)
            try:
                self.viewer.layers.selection.active = self._points_layer
            except Exception:
                pass
        elif self._viewer_integration is not None:
            self._viewer_integration._ensure_nuclei_active()
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

    def open_expression_plot_window(self) -> None:
        """Open an independent, modeless expression plot window."""

        from .expression_plot_window import ExpressionPlotWindow

        self._expression_plot_window_counter += 1
        parent = None
        try:
            parent = self.viewer.window._qt_window if self.viewer is not None else None
        except (AttributeError, RuntimeError):
            parent = None
        window = ExpressionPlotWindow(
            self,
            window_number=self._expression_plot_window_counter,
            parent=parent,
        )
        self._expression_plot_windows.append(window)
        window.show()

    def expression_dataset_repository(self):
        """Return the application-scoped cross-dataset expression cache."""

        if self._expression_dataset_repository is None:
            from ..analysis.expression_dataset_repository import (
                ExpressionDatasetRepository,
            )

            self._expression_dataset_repository = ExpressionDatasetRepository()
        return self._expression_dataset_repository

    def open_expression_comparison_window(self) -> None:
        """Open an independent multi-dataset expression comparison window."""

        from .expression_comparison_window import ExpressionComparisonWindow

        self._expression_comparison_window_counter += 1
        parent = None
        try:
            parent = self.viewer.window._qt_window if self.viewer is not None else None
        except (AttributeError, RuntimeError):
            parent = None
        window = ExpressionComparisonWindow(
            self,
            repository=self.expression_dataset_repository(),
            window_number=self._expression_comparison_window_counter,
            parent=parent,
        )
        self._expression_comparison_windows.append(window)
        window.show()

    def open_expression_comparison_result_window(self, path=None):
        """Open a portable comparison in a new source-independent window.

        Loading and validating the result happens before any window-list or
        counter mutation. Legacy v1 captures remain fixed and never instantiate
        the shared repository. Full v2 measurement sets use the application
        repository only for optional XML attachment/recomputation; all plotting,
        retargeting, and export remain available from embedded caches offline.
        """

        from qtpy.QtWidgets import QFileDialog, QMessageBox

        from ..analysis.expression_comparison_result import (
            EXPRESSION_COMPARISON_RESULT_SUFFIX,
            load_expression_comparison_result,
        )
        from .expression_comparison_window import ExpressionComparisonWindow

        parent = None
        try:
            parent = self.viewer.window._qt_window if self.viewer is not None else None
        except (AttributeError, RuntimeError):
            parent = None
        if path is None:
            path, _selected_filter = QFileDialog.getOpenFileName(
                parent,
                "Open expression measurement set or legacy result",
                "",
                "AceTree expression sets and results (*.aceexpr)",
            )
            if not path:
                return None
        try:
            result = load_expression_comparison_result(path)
            next_number = self._expression_comparison_window_counter + 1
            window = ExpressionComparisonWindow(
                self,
                # Schema-v2 measurement sets remain source-independent for
                # plotting, but they may be extended with new XML datasets or
                # refreshed from attached sources.  Give those windows the
                # same application-scoped repository as live comparisons.
                # Legacy schema-v1 captures intentionally keep the old fixed,
                # repository-free boundary.
                repository=(
                    self.expression_dataset_repository()
                    if result.measurement_caches
                    else None
                ),
                result=result,
                result_path=str(path),
                window_number=next_number,
                parent=parent,
            )
        except Exception as error:  # noqa: BLE001 - malformed files fail closed
            logger.exception("Could not open portable expression result %s", path)
            QMessageBox.warning(
                parent,
                "Cannot open expression result",
                f"The selected {EXPRESSION_COMPARISON_RESULT_SUFFIX} file could not "
                f"be opened:\n{error}",
            )
            return None
        self._expression_comparison_window_counter = next_number
        self._expression_comparison_windows.append(window)
        window.show()
        return window

    def _shutdown_expression_dataset_repository(self) -> None:
        repository = self._expression_dataset_repository
        self._expression_dataset_repository = None
        if repository is not None:
            try:
                repository.close()
            except Exception:  # noqa: BLE001 - best-effort application teardown
                logger.exception("Could not close expression dataset repository")

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

    def _suggest_division_names_safe(
        self,
        parent,
        first_pos: tuple[float, float, float],
        new_pos: tuple[float, float, float],
        time: int,
    ):
        """Ask the naming model for daughter names, or safely defer to it.

        Older managers do not expose the suggestion API.  In that case the
        click still creates the division with unlocked daughter identities;
        the normal post-edit naming pass determines their names.  No GUI-level
        axis or ``a/p`` assumption is made.
        """
        suggest = getattr(self.manager, "suggest_division_names", None)
        if suggest is None:
            return None
        try:
            result = suggest(parent, first_pos, new_pos, time)
        except Exception:
            logger.exception("Division-name suggestion failed at t=%d", time)
            return None
        if not result or not result.first_name or not result.second_name:
            return None
        if result.first_name == result.second_name:
            logger.warning(
                "Ignoring non-distinct division-name suggestion '%s' at t=%d",
                result.first_name, time,
            )
            return None
        return result

    @staticmethod
    def _reconcile_division_suggestion_with_first_override(
        suggestion,
        first_daughter,
        inherited_parent_lock: bool,
    ):
        """Make a preview/commit agree with a preserved first-daughter lock.

        A cell may have been explicitly named while it still looked like a
        continuation.  Once a second successor proves that it is a daughter,
        preserve a genuine daughter override and swap the automatic pair when
        that override names the proposed second daughter.  Inherited parent
        locks are cleared by the existing division workflow and do not reorder
        the pair.
        """
        if suggestion is None or inherited_parent_lock:
            return suggestion
        locked_name = (first_daughter.assigned_id or "").strip()
        if not locked_name or locked_name == suggestion.first_name:
            return suggestion
        if locked_name == suggestion.second_name:
            return replace(
                suggestion,
                first_name=suggestion.second_name,
                second_name=suggestion.first_name,
                confidence=0.0,
                source="forced first-daughter override",
                ambiguous=True,
            )
        # A foreign curator name is an intentional exception to the canonical
        # pair.  Report the actual effective first name instead of previewing a
        # placement that the assigned_id will immediately mask.
        return replace(
            suggestion,
            first_name=locked_name,
            confidence=0.0,
            source="forced first-daughter override",
            ambiguous=True,
        )

    def _handle_add_click(self, x: float, y: float) -> bool:
        """Handle a left-click in add mode — place a nucleus at (x, y).

        Uses the currently selected cell as predecessor if one is active.
        For gap == 1, sets predecessor directly. For gap > 1, creates the
        nucleus then auto-interpolates to fill the gap. Inherits diameter
        from the parent cell's last nucleus when available.

        Manual-division handling: if the click is the second successor of
        the selected parent, daughter identities are suggested by the same
        lineage/geometry model used by automated naming. Triple-successor
        attempts are rejected with a status message.
        """
        if not self._add_mode:
            return False

        from ..editing.commands import (
            AddNucleus,
            CompositeCommand,
            RelinkWithInterpolation,
            SetCellNameState,
        )
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
        cell = self.get_selected_cell()
        parent_name = cell.name if cell is not None else None
        if self.selection_anchor is not None and cell is None:
            self._say("Selected nucleus is no longer in the lineage tree")
            return False
        if cell is not None:
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
                            # Only a genuinely manual parent override is
                            # inherited.  Copying an automatic effective name
                            # into assigned_id would silently lock the chain.
                            assigned_id = parent_nuc.assigned_id
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

        # Daughter names come from the manager's lineage/axis-aware naming
        # model and remain automatic identities.  A first daughter that
        # inherited its parent's manual lock while it looked like a
        # continuation is unlocked once the second daughter proves division.
        first_name_state = None
        first_idx = -1
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
                first_pos = (float(first_daughter.x), float(first_daughter.y),
                             float(first_daughter.z))
                new_pos = (float(ix), float(iy), float(iz))
                suggestion = self._suggest_division_names_safe(
                    parent_nuc_ref, first_pos, new_pos, time,
                )
                inherited_parent_lock = bool(
                    parent_nuc_ref.assigned_id
                    and first_daughter.assigned_id == parent_nuc_ref.assigned_id
                )
                suggestion = self._reconcile_division_suggestion_with_first_override(
                    suggestion,
                    first_daughter,
                    inherited_parent_lock,
                )
                # The second successor proves that the apparent continuation
                # is a daughter.  Always clear its inherited automatic parent
                # identity when geometry is unavailable; otherwise a partial
                # dataset would retain a duplicate parent name on one branch.
                first_name_state = SetCellNameState(
                    time=time,
                    index=first_idx + 1,
                    identity=suggestion.first_name if suggestion else "",
                    assigned_id=(
                        "" if inherited_parent_lock
                        else first_daughter.assigned_id
                    ),
                )

                assigned_id = ""
                identity = suggestion.second_name if suggestion else ""
                if suggestion:
                    self._say(
                        f"Division: {suggestion.first_name} + "
                        f"{suggestion.second_name} "
                        f"({suggestion.axis_label}, {suggestion.source}, "
                        f"confidence {suggestion.confidence:.0%})"
                    )
                else:
                    self._say(
                        "Division created; biological daughter ordering is "
                        "deferred until a complete body frame is available"
                    )

        # Issue AddNucleus first so the parent becomes a division before the
        # first daughter's continuation component is updated.  Reversing the
        # order would let SetCellNameState walk backward into the parent.
        nr = self.edit_history.nuclei_record
        predicted_index = (
            len(nr[time - 1]) + 1 if 0 <= time - 1 < len(nr) else 1
        )
        add_cmd = AddNucleus(
            time=time,
            x=ix,
            y=iy,
            z=iz,
            size=size,
            identity=identity,
            predecessor=predecessor,
            assigned_id=assigned_id,
        )
        commands = [add_cmd]
        if first_name_state is not None:
            commands.append(first_name_state)

        # Fill gap > 1 with interpolation
        if (parent_name and parent_end_time is not None
                and parent_end_index is not None):
            gap = time - parent_end_time
            if gap > 1:
                interp_cmd = RelinkWithInterpolation(
                    start_time=parent_end_time,
                    start_index=parent_end_index,
                    end_time=time,
                    end_index=predicted_index,
                )
                commands.append(interp_cmd)

        command = CompositeCommand(
            commands=commands,
            label=f"Add nucleus at t={time}",
        )
        self._run_edit_action(self.edit_history.do, command)

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
        switch_from_3d = self._3d_mode
        self._exit_roi_mode()
        if self._relink_pick_mode:
            self.cancel_relink_pick_mode()
        self.exit_add_mode()
        if self._edit_panel is not None:
            self._edit_panel._btn_add.setChecked(False)
        self._placement_mode = True
        self._placement_parent_name = parent_name
        self._placement_parent_anchor = None
        if parent_name is not None:
            selected_cell = self.get_selected_cell()
            if selected_cell is not None and selected_cell.name == parent_name:
                self._placement_parent_anchor = self.selection_anchor
        self._placement_default_size = default_size
        if switch_from_3d:
            # Track placement needs the current image plane for a definite Z.
            # Arm Track before switching so its checked state survives the
            # 2D-view refresh and the next right click works immediately.
            self.set_3d_mode(False)
            self._say("Manual Track placement uses the 2D slice view")
        self._focus_viewer_canvas()

    def exit_placement_mode(self) -> None:
        """Exit click-to-place mode."""
        self._placement_mode = False
        self._placement_parent_name = None
        self._placement_parent_anchor = None

    def _handle_placement_click(self, x: float, y: float) -> bool:
        """Handle a click in placement mode — create a nucleus at (x, y).

        Returns True if the click was consumed.
        """
        if not self._placement_mode:
            return False

        from ..editing.commands import (
            AddNucleus,
            CompositeCommand,
            RelinkWithInterpolation,
            SetCellNameState,
        )
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
            cell = None
            if self._placement_parent_anchor is not None:
                anchor = self._placement_parent_anchor
                anchor_nuc = self._nucleus_at_anchor(anchor)
                if anchor_nuc is not None and anchor_nuc.is_alive:
                    cell = self._cell_for_nucleus(anchor[0], anchor_nuc)
                if cell is None:
                    self._say("Tracked parent is no longer in the lineage tree")
                    return False
            else:
                # Compatibility for programmatic callers that start Track
                # with a name but no GUI selection.
                cell = self.manager.get_cell(parent_name)
            if cell is not None:
                parent_name = cell.name
                existing_here = cell.get_nucleus_at(time)
                is_division_click = False
                if (existing_here is not None
                        and time > 1
                        and existing_here.predecessor != NILLI):
                    # As in Add mode, a placement during the cell's lifetime
                    # is a retroactive division.  At the terminal frame a far
                    # click is a division while a close click is left as an
                    # unlinked placement (Track extensions should be made at
                    # a later frame selected by the user).
                    if time < cell.end_time:
                        is_division_click = True
                    else:
                        dx_ex = ix - existing_here.x
                        dy_ex = iy - existing_here.y
                        is_division_click = (
                            dx_ex * dx_ex + dy_ex * dy_ex
                            > float(existing_here.size) ** 2
                        )

                if is_division_click:
                    parent_end_time = time - 1
                    parent_end_index = existing_here.predecessor
                    nr = self.manager.nuclei_record
                    t0 = parent_end_time - 1
                    j0 = parent_end_index - 1
                    if 0 <= t0 < len(nr) and 0 <= j0 < len(nr[t0]):
                        parent_nuc_ref = nr[t0][j0]
                        size = parent_nuc_ref.size
                        identity = parent_nuc_ref.effective_name
                        assigned_id = parent_nuc_ref.assigned_id
                        predecessor = parent_end_index
                    else:
                        parent_name = None
                else:
                    parent_end_time = cell.end_time
                    gap = time - parent_end_time
                    if gap <= 0:
                        # Same/earlier close placement is independent (for
                        # example, adding multiple roots at the first frame).
                        parent_name = None
                    else:
                        identity = parent_name
                        parent_nuc = cell.get_nucleus_at(parent_end_time)
                        if parent_nuc is not None:
                            parent_nuc_ref = parent_nuc
                            parent_end_index = parent_nuc.index
                            size = parent_nuc.size  # inherit diameter
                            assigned_id = parent_nuc.assigned_id
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

        # Manual-division detection mirrors Add mode and delegates all
        # biological name choice to the manager.
        first_name_state = None
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
                first_pos = (float(first_daughter.x), float(first_daughter.y),
                             float(first_daughter.z))
                new_pos = (float(ix), float(iy), float(iz))
                suggestion = self._suggest_division_names_safe(
                    parent_nuc_ref, first_pos, new_pos, time,
                )
                inherited_parent_lock = bool(
                    parent_nuc_ref.assigned_id
                    and first_daughter.assigned_id == parent_nuc_ref.assigned_id
                )
                suggestion = self._reconcile_division_suggestion_with_first_override(
                    suggestion,
                    first_daughter,
                    inherited_parent_lock,
                )
                first_name_state = SetCellNameState(
                    time=time,
                    index=first_idx + 1,
                    identity=suggestion.first_name if suggestion else "",
                    assigned_id=(
                        "" if inherited_parent_lock
                        else first_daughter.assigned_id
                    ),
                )
                assigned_id = ""
                identity = suggestion.second_name if suggestion else ""
                if suggestion:
                    self._say(
                        f"Division: {suggestion.first_name} + "
                        f"{suggestion.second_name} "
                        f"({suggestion.axis_label}, {suggestion.source}, "
                        f"confidence {suggestion.confidence:.0%})"
                    )
                else:
                    self._say(
                        "Division created; biological daughter ordering is "
                        "deferred until a complete body frame is available"
                    )

        # AddNucleus first, SetCellNameState second — see
        # _handle_add_click for why the structural ordering matters.
        nr = self.edit_history.nuclei_record
        predicted_index = (
            len(nr[time - 1]) + 1 if 0 <= time - 1 < len(nr) else 1
        )
        add_cmd = AddNucleus(
            time=time,
            x=ix,
            y=iy,
            z=iz,
            size=size,
            identity=identity,
            predecessor=predecessor,
            assigned_id=assigned_id,
        )
        commands = [add_cmd]
        if first_name_state is not None:
            commands.append(first_name_state)

        # Handle gap > 1 with interpolation
        if (parent_name and parent_end_time is not None
                and parent_end_index is not None):
            gap = time - parent_end_time
            if gap > 1:
                interp_cmd = RelinkWithInterpolation(
                    start_time=parent_end_time,
                    start_index=parent_end_index,
                    end_time=time,
                    end_index=predicted_index,
                )
                commands.append(interp_cmd)

        command = CompositeCommand(
            commands=commands,
            label=f"Track nucleus at t={time}",
        )
        self._run_edit_action(self.edit_history.do, command)

        # Mode continuation
        if parent_name is None:
            # Root mode: exit after single placement
            self.exit_placement_mode()
            if self._edit_panel is not None:
                self._edit_panel.refresh()
        # else: stay in placement mode for continued tracking

        return True

    # ── 3D view toggle ─────────────────────────────────────────────

    def stack_z_from_plane(self, plane: float) -> float:
        """Translate AceTree's absolute Z-plane coordinate to stack-local Z."""

        config = self.manager.config
        plane_start = config.plane_start if config is not None else 1
        return float(plane) - float(plane_start)

    def physical_to_stack_coordinates(
        self,
        x_um: float,
        y_um: float,
        z_um: float,
    ) -> tuple[float, float, float]:
        """Return napari ``(z, y, x)`` coordinates for a physical position."""

        config = self.manager.config
        if config is None:
            return float(z_um), float(y_um), float(x_um)
        return (
            float(z_um) / config.z_res,
            float(y_um) / config.xy_res,
            float(x_um) / config.xy_res,
        )

    def toggle_3d(self) -> None:
        """Toggle between 2D slice view and 3D volume view."""
        self.set_3d_mode(not self._3d_mode)

    def set_3d_mode(self, enabled: bool) -> None:
        """Idempotently synchronize AceTree and napari display modes."""

        if self.viewer is None:
            self._3d_mode = False
            return
        enabled = bool(enabled)
        if enabled:
            self._exit_roi_mode()
        if enabled and (self._add_mode or self._placement_mode):
            # Manual placement requires a definite image plane. Do not carry
            # an armed 2D click mode into 3D, where the same buttons would be
            # interpreted as label/camera interactions.
            self.exit_add_mode()
            self.exit_placement_mode()
            if self._edit_panel is not None:
                try:
                    self._edit_panel._btn_add.setChecked(False)
                    self._edit_panel._btn_track.setChecked(False)
                except Exception:
                    pass
            self._say(
                "Exited Add/Manual Track because placement uses the 2D slice view"
            )
        if enabled == self._3d_mode:
            expected = 3 if enabled else 2
            if int(getattr(self.viewer.dims, "ndisplay", expected)) == expected:
                return
        self._3d_mode = enabled
        self._changing_ndisplay = True
        try:
            if enabled:
                self._enter_3d()
            else:
                self._exit_3d()
        finally:
            self._changing_ndisplay = False
        if self._roi_viewer_integration is not None:
            self._roi_viewer_integration.set_three_dimensional(enabled)
        if self._player_controls is not None:
            self._player_controls.refresh()

    def _on_native_ndisplay_changed(self, event) -> None:
        """Route napari's built-in 2D/3D button through AceTree setup."""

        if self._changing_ndisplay:
            return
        value = int(getattr(event, "value", getattr(self.viewer.dims, "ndisplay", 2)))
        desired = value == 3
        if desired != self._3d_mode:
            self.set_3d_mode(desired)

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
            if self._viewer_integration._tracking_preview_spots_layer:
                self._viewer_integration._tracking_preview_spots_layer.visible = False
            if self._viewer_integration._tracking_preview_links_layer:
                self._viewer_integration._tracking_preview_links_layer.visible = False

        # Build 3D Points layer for nuclei
        self._update_3d_points()

        # Switch viewer to 3D
        self.viewer.dims.ndisplay = 3
        if self._viewer_integration:
            self._viewer_integration.refresh_tracking_preview()

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
            self._viewer_integration.refresh_tracking_preview()

        # Reload 2D plane
        self.update_display()

    def _update_3d_points(self) -> None:
        """Create or update the 3D Points layer for nucleus positions."""
        if self.viewer is None:
            return

        nuclei = self.manager.alive_nuclei_at(self.current_time)
        z_scale = self.manager.z_pix_res
        selection_resolver = getattr(self, "get_selected_nucleus", None)
        resolved_selection = (
            selection_resolver(self.current_time)
            if callable(selection_resolver)
            else None
        )
        selected_nucleus = (
            resolved_selection[0]
            if resolved_selection is not None
            and resolved_selection[1] == self.current_time
            else None
        )

        coords = []
        sizes = []
        names_list = []

        for nuc in nuclei:
            # Points coords in (z, y, x) — z in pixel units, scaled by layer
            coords.append([self.stack_z_from_plane(nuc.z), nuc.y, nuc.x])
            sizes.append(nuc.size)
            names_list.append(nuc.effective_name or f"Nuc{nuc.index}")

        if self._viz_mode:
            # Visualization mode — batch rule-engine colors
            colors = [
                list(c) for c in self.color_engine.colors_for_frame(
                    nuclei, self.manager, self.current_time,
                    # Forced names need not be unique while a conflict is
                    # being corrected. Highlight the physically anchored
                    # nucleus below instead of every matching name.
                    selected_name="",
                )
            ]
            selected_color = list(
                getattr(
                    self.color_engine,
                    "selected_color",
                    (1.0, 1.0, 1.0, 1.0),
                )
            )
            for i, nuc in enumerate(nuclei):
                if nuc is selected_nucleus:
                    colors[i] = selected_color
        else:
            # Editing mode — status-based palette
            colors = []
            for nuc in nuclei:
                name = nuc.effective_name or ""
                if nuc is selected_nucleus:
                    colors.append([1.0, 1.0, 1.0, 1.0])  # White — selected
                elif name.startswith("Nuc"):
                    colors.append([1.0, 0.6, 0.15, 0.8])  # Orange — unnamed
                elif name:
                    colors.append([0.55, 0.27, 1.0, 0.8])  # Purple — named
                else:
                    colors.append([0.5, 0.5, 0.5, 0.5])  # Gray — no name

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
            # Trails are a separate native layer.  They still need to be
            # cleared when the current frame contains no live nuclei;
            # otherwise positions from the previous frame remain visible.
            self._update_3d_trail()
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
        point_features = {
            "name": display_names,
            "acetree_time": [self.current_time] * len(nuclei),
            "acetree_index": [nuc.index for nuc in nuclei],
        }

        if self._points_layer is None:
            # Create once, then lock the returned layer. Retrying a broad
            # TypeError with different kwargs can duplicate a layer if napari
            # partially completed the first constructor call.
            layer = self.viewer.add_points(
                coords_arr,
                size=sizes_arr,
                face_color=colors_arr,
                border_color="transparent",
                name="Nuclei 3D",
                scale=(z_scale, 1.0, 1.0),
                opacity=0.7,
                features=point_features,
            )
            self._points_layer = layer
            configure_curated_points_layer(
                layer,
                callback=self._on_3d_click,
                lock=self._make_curated_points_read_only,
            )
        else:
            # Retry text/callback setup before replacing marker state. A
            # transient failure during initial creation must not leave this
            # layer permanently non-interactive.
            configure_curated_points_layer(
                self._points_layer,
                callback=self._on_3d_click,
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

        # ``Nuclei 3D`` is a projection of the curated record, not an editing
        # surface.  Keep custom click/label callbacks, but prevent napari's
        # native point add/move/delete modes from creating marker-only edits.
        self._make_curated_points_read_only(self._points_layer)

        # Ghost trail in 3D
        self._update_3d_trail()

    @staticmethod
    def _make_curated_points_read_only(layer) -> None:
        """Lock a curated 3D marker layer without removing click callbacks."""

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

        cell = self.get_selected_cell()
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
            coords.append([self.stack_z_from_plane(nuc.z), nuc.y, nuc.x])
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

        Picking happens on press, but selection/redraw is queued only after
        mouse release so a camera drag cannot toggle a label or strand
        napari's active drag generator.
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
        anchor = None
        if idx is not None and isinstance(idx, (int, np.integer)):
            anchor = point_anchor(layer, int(idx))
            if anchor is None:
                # Compatibility with a layer created before anchor features
                # were introduced; the next redraw will publish them.
                nuclei = self.manager.alive_nuclei_at(self.current_time)
                if 0 <= idx < len(nuclei):
                    anchor = (self.current_time, nuclei[int(idx)].index)

        button = event.button  # 1 = left, 2 = right
        time = self.current_time
        history = getattr(self, "edit_history", None)
        change_counter = getattr(history, "change_counter", None)
        relink_callback = self._relink_pick_callback
        mode_context = (
            self._relink_pick_mode,
            self._add_mode,
            self._placement_mode,
        )

        if self._relink_pick_mode:
            intent = "relink"
        elif self._placement_mode and button == 2:
            # A ray does not define one unambiguous placement depth.
            intent = None
        elif button == 2:
            intent = "select"
        else:
            intent = "label"

        press_pointer = pointer_position(event)
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = dragged or passed_drag_threshold(event, press_pointer)
            yield
        dragged = dragged or passed_drag_threshold(event, press_pointer)
        if dragged or intent is None:
            return

        from qtpy.QtCore import QTimer

        QTimer.singleShot(
            0,
            lambda: self._apply_deferred_3d_click(
                layer=layer,
                intent=intent,
                anchor=anchor,
                time=time,
                change_counter=change_counter,
                relink_callback=relink_callback,
                mode_context=mode_context,
            ),
        )

    def _apply_deferred_3d_click(
        self,
        *,
        layer,
        intent: str,
        anchor: tuple[int, int] | None,
        time: int,
        change_counter: int | None,
        relink_callback,
        mode_context: tuple[bool, bool, bool],
    ) -> None:
        """Apply a stable 3D pick after napari has closed the drag cycle."""

        history = getattr(self, "edit_history", None)
        if (
            self._points_layer is not layer
            or self.current_time != time
            or getattr(history, "change_counter", None) != change_counter
            or anchor is None
            or anchor[0] != time
        ):
            return
        nuc = self._nucleus_at_anchor(anchor)
        if nuc is None or not nuc.is_alive:
            return

        if intent == "relink":
            if (
                not self._relink_pick_mode
                or self._relink_pick_callback is not relink_callback
                or relink_callback is None
            ):
                return
            self.exit_relink_pick_mode()
            self._run_edit_action(relink_callback, time, nuc)
            return

        current_modes = (
            self._relink_pick_mode,
            self._add_mode,
            self._placement_mode,
        )
        if current_modes != mode_context:
            return

        if intent == "select":
            self._set_selection_from_nucleus(time, nuc)
            # Keep the hidden 2D slice in step with the new selection so
            # toggling out of 3D lands on the active cell instead of whatever
            # plane was showing before.  _snap_plane_to_nucleus() only touches
            # current_plane — it must not kick off a 2D redraw while the
            # viewer is in 3D mode.
            self._snap_plane_to_nucleus(nuc)
            if self._viewer_integration:
                display_name = nuc.effective_name or f"Nuc{nuc.index}"
                self._viewer_integration._shown_labels.add(display_name)
            self._update_3d_points()
            for lineage_widget in self._lineage_widgets:
                lineage_widget.refresh_selection()
            if self._lineage_list:
                self._lineage_list.refresh_selection()
            return

        if intent == "label" and self._viewer_integration:
            name = nuc.effective_name or f"Nuc{nuc.index}"
            if name in self._viewer_integration._shown_labels:
                self._viewer_integration._shown_labels.discard(name)
            else:
                self._viewer_integration._shown_labels.add(name)
            self._update_3d_points()

    # ── Display ───────────────────────────────────────────────────

    def update_display(self) -> None:
        """Refresh all visual components for the current state."""
        self._load_image()

        if self._viewer_integration:
            if self._3d_mode:
                self._viewer_integration.refresh_tracking_preview()
            else:
                self._viewer_integration.update_overlays()

        if self._roi_viewer_integration:
            self._roi_viewer_integration.update_overlay()

        if self._contrast_tools:
            self._contrast_tools.refresh()

        if self._player_controls:
            self._player_controls.refresh()

        if self._edit_panel:
            self._edit_panel.refresh()

        if self._subcellular_objects_panel:
            self._subcellular_objects_panel.refresh()

        if self._global_tracking_dialog is not None:
            try:
                self._global_tracking_dialog.sync_document_revision()
                self._global_tracking_dialog.sync_viewer_position(self.current_time)
            except RuntimeError:
                self._global_tracking_dialog = None

        for lw in self._lineage_widgets:
            lw.refresh_selection()

        if self._lineage_list:
            self._lineage_list.refresh_selection()

        # Refresh any detached 3D viewer windows
        for win in self._3d_windows:
            try:
                if win.isVisible():
                    win.refresh()
            except RuntimeError as error:
                # Qt raises RuntimeError when its C++ widget was deleted.
                # Other RuntimeErrors (including an atomic centroid redraw
                # rollback) are real refresh failures and must reach the
                # post-commit warning boundary instead of being hidden.
                message = str(error).lower()
                deleted_qt_object = (
                    "deleted" in message
                    and ("c/c++ object" in message or "c++ object" in message)
                )
                if not deleted_qt_object:
                    raise

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

        if self._roi_viewer_integration:
            self._roi_viewer_integration.update_overlay()

        if self._player_controls:
            self._player_controls.refresh()

        if self._subcellular_objects_panel:
            self._subcellular_objects_panel.refresh()

    def _on_roi_object_selected(self, object_id) -> None:
        """Keep ROI and nucleus selection as independent UI state."""

        self.current_roi_object_id = object_id
        track = self.roi_manager.get_object(object_id) if object_id is not None else None
        self.current_roi_class_id = None if track is None else track.class_id

    def _on_roi_mode_changed(self, mode: str) -> None:
        """Keep ROI authoring mutually exclusive with other canvas modes."""

        if mode == "inspect":
            return
        if self._3d_mode:
            self.set_3d_mode(False)
        if self._add_mode:
            self.exit_add_mode()
        if self._placement_mode:
            self.exit_placement_mode()
        if self._relink_pick_mode:
            self.cancel_relink_pick_mode()
        if self._edit_panel is not None:
            try:
                self._edit_panel._btn_add.setChecked(False)
                self._edit_panel._btn_track.setChecked(False)
            except Exception:
                pass
        if mode in {"draw_polygon", "draw_polyline", "draw_contour_stack"}:
            try:
                self._begin_roi_drawing(mode)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                self._say(str(error))
                if self._roi_viewer_integration is not None:
                    self._roi_viewer_integration.cancel_edit()
                if self._subcellular_objects_panel is not None:
                    self._subcellular_objects_panel.set_mode("inspect")

    def _begin_roi_drawing(self, mode: str) -> None:
        """Start a model-free draft for a new track or an undecided frame."""

        panel = self._subcellular_objects_panel
        integration = self._roi_viewer_integration
        if panel is None or integration is None:
            raise RuntimeError("The subcellular object editor is unavailable")
        from ..core.subcellular_roi import ContourStack3D

        track = self.roi_manager.get_object(self.current_roi_object_id)
        frame = None if track is None else track.frames.get(int(self.current_time))
        geometry = None if frame is None else frame.geometry
        continuing_stack = (
            mode == "draw_contour_stack"
            and isinstance(geometry, ContourStack3D)
        )
        if track is not None and (
            frame is None or frame.presence.value == "absent" or continuing_stack
        ):
            object_id = track.object_id
            class_id = track.class_id
        else:
            object_id = None
            class_id = panel.selected_class_id or self.current_roi_class_id
        if class_id is None:
            raise ValueError("Choose an object class before drawing")
        object_class = self.roi_manager.get_class(class_id)
        if object_class is None:
            raise ValueError("The selected object class no longer exists")
        integration.begin_drawing(
            class_id=class_id,
            object_id=object_id,
            timepoint=int(self.current_time),
            z_plane=int(self.current_plane),
            kind=mode,
            cell_ref=self._roi_cell_ref_from_selected(),
            geometry=geometry if continuing_stack else None,
        )
        target = (
            f"{object_class.name} #{track.instance_index}"
            if object_id is not None
            else f"new {object_class.name} object"
        )
        self._say(
            f"Drawing {target} at t={self.current_time}, z={self.current_plane}; "
            "double-click to close the shape, then press Finish or Enter"
        )

    def _roi_cell_ref_from_selected(self):
        """Capture the selected same-frame cell as a persistent ROI anchor."""

        selected = self.get_selected_nucleus(int(self.current_time))
        if selected is None:
            return None
        nucleus, selected_time, _index = selected
        return self._roi_cell_ref_for_nucleus(selected_time, nucleus)

    def _roi_cell_ref_for_nucleus(self, timepoint: int, nucleus):
        """Build a stable same-frame ROI association for a picked nucleus."""

        selected_time = int(timepoint)
        cell = self._cell_for_nucleus(selected_time, nucleus)
        if cell is None:
            return None
        from ..core.subcellular_roi import CellRef, NucleusAnchor

        birth_time, birth_nucleus = min(cell.nuclei, key=lambda item: item[0])
        return CellRef(
            nucleus_anchor=NucleusAnchor(selected_time, nucleus.index),
            cell_birth_anchor=NucleusAnchor(birth_time, birth_nucleus.index),
            name_snapshot=cell.name,
            centroid_snapshot_xyz_px=(nucleus.x, nucleus.y, nucleus.z),
        )

    def _exit_roi_mode(self) -> bool:
        """Cancel transient ROI authoring and restore the dock to Inspect."""

        integration = self._roi_viewer_integration
        panel = self._subcellular_objects_panel
        editing = bool(integration is not None and integration.editing)
        mode = getattr(panel, "mode", "inspect") if panel is not None else "inspect"
        mode_value = str(getattr(mode, "value", mode))
        changed = editing or mode_value != "inspect"
        if integration is not None:
            integration.cancel_edit()
        if panel is not None:
            current_mode = getattr(panel, "mode", "inspect")
            if str(getattr(current_mode, "value", current_mode)) != "inspect":
                panel.set_mode("inspect")
        return changed

    def _on_roi_action_requested(self, action: str, object_id) -> None:
        """Dispatch browse/curation actions from the Objects dock."""

        from ..editing.roi_commands import (
            AssociateRoiFrame,
            CopyRoiFrameDraft,
            DeleteRoiFrame,
            MarkRoiFrameAbsent,
            MarkRoiFrameReviewed,
        )

        timepoint = int(self.current_time)
        try:
            if action == "finish_drawing":
                if self._roi_viewer_integration is None:
                    raise RuntimeError("The ROI editor is unavailable")
                self._roi_viewer_integration.finish_edit()
                return
            if action == "cancel_drawing":
                self._exit_roi_mode()
                return
            if action == "manage_classes":
                self._create_roi_class_from_dialog()
                return
            if action == "set_visibility":
                return
            if object_id is None:
                return
            track = self.roi_manager.get_object(object_id)
            if track is None:
                raise ValueError("The selected ROI object no longer exists")
            if action == "pick_cell":
                picked_time = timepoint

                def associate_picked_cell(selected_time, nucleus) -> None:
                    if int(selected_time) != picked_time:
                        self._say(
                            "The view time changed while picking an association; "
                            "pick the cell again"
                        )
                        return
                    cell_ref = self._roi_cell_ref_for_nucleus(
                        selected_time,
                        nucleus,
                    )
                    if cell_ref is None:
                        self._say("The picked nucleus has no current cell")
                        return
                    self._run_edit_action(
                        self.edit_history.do,
                        AssociateRoiFrame(
                            self.roi_manager,
                            object_id,
                            picked_time,
                            cell_ref,
                        ),
                    )

                self.enter_relink_pick_mode(associate_picked_cell)
                self._say(
                    "Right-click a nucleus at this timepoint to associate it "
                    "with the selected subcellular object; Escape cancels"
                )
                return
            if action == "mark_reviewed":
                self._run_edit_action(
                    self.edit_history.do,
                    MarkRoiFrameReviewed(self.roi_manager, object_id, timepoint),
                )
            elif action == "mark_absent":
                self._run_edit_action(
                    self.edit_history.do,
                    MarkRoiFrameAbsent(self.roi_manager, object_id, timepoint),
                )
            elif action == "delete_frame":
                self._run_edit_action(
                    self.edit_history.do,
                    DeleteRoiFrame(self.roi_manager, object_id, timepoint),
                )
            elif action in {"use_selected_cell", "clear_association"}:
                cell_ref = None
                if action == "use_selected_cell":
                    cell_ref = self._roi_cell_ref_from_selected()
                    if cell_ref is None:
                        raise ValueError("Select a live cell at this timepoint first")
                self._run_edit_action(
                    self.edit_history.do,
                    AssociateRoiFrame(
                        self.roi_manager, object_id, timepoint, cell_ref
                    ),
                )
            elif action == "copy_previous":
                previous = max(
                    (value for value in track.segmented_times if value < timepoint),
                    default=None,
                )
                if previous is None:
                    raise ValueError("There is no earlier segmented frame to copy")
                self._run_edit_action(
                    self.edit_history.do,
                    CopyRoiFrameDraft(
                        self.roi_manager,
                        object_id,
                        previous,
                        timepoint,
                    ),
                )
            elif action in {"previous", "next"}:
                times = track.segmented_times
                candidates = (
                    [value for value in times if value < timepoint]
                    if action == "previous"
                    else [value for value in times if value > timepoint]
                )
                if candidates:
                    self.set_time(max(candidates) if action == "previous" else min(candidates))
            elif action == "edit":
                frame = track.frames.get(timepoint)
                if frame is None or frame.geometry is None:
                    raise ValueError("The selected object has no geometry at this timepoint")
                self._roi_viewer_integration.enter_edit_mode(
                    object_id,
                    timepoint,
                    frame.geometry,
                    z_plane=self.current_plane,
                )
                if self._subcellular_objects_panel is not None:
                    self._subcellular_objects_panel.set_mode("edit")
            elif action == "measure":
                snapshot = self.roi_measurement_engine.measure(
                    self.roi_manager,
                    object_ids=(object_id,),
                )
                self._refresh_roi_scalar_plot_windows(snapshot)
                self._say(
                    f"Measured {len(snapshot.samples)} ROI channel samples"
                )
            elif action == "plot_track":
                from .roi_scalar_plot_window import RoiScalarPlotWindow

                parent = (
                    self.viewer.window._qt_window
                    if self.viewer is not None
                    else None
                )
                window = RoiScalarPlotWindow.from_app(
                    self,
                    object_ids=(object_id,),
                    parent=parent,
                )
                window.destroyed.connect(
                    lambda *_args, item=window: (
                        self._roi_scalar_plot_windows.remove(item)
                        if item in self._roi_scalar_plot_windows
                        else None
                    )
                )
                self._roi_scalar_plot_windows.append(window)
                window.show()
            elif action == "plot_profiles":
                snapshot = self.roi_measurement_engine.latest_snapshot
                if snapshot is None:
                    raise ValueError("Measure this object with spatial profiles first")
                from .roi_profile_window import RoiProfileSeries, RoiProfileWindow

                profiles = tuple(
                    RoiProfileSeries(
                        label=f"t={sample.timepoint}, channel {sample.image_channel + 1}",
                        profile=sample.profile,
                        object_id=sample.object_id,
                        timepoint=sample.timepoint,
                        image_channel=sample.image_channel,
                    )
                    for sample in snapshot.samples.values()
                    if sample.object_id == str(object_id) and sample.profile is not None
                )
                if not profiles:
                    raise ValueError(
                        "No line profiles are available; remeasure with Profiles enabled"
                    )
                window = RoiProfileWindow(profiles)
                window.destroyed.connect(
                    lambda *_args, item=window: (
                        self._roi_profile_windows.remove(item)
                        if item in self._roi_profile_windows
                        else None
                    )
                )
                self._roi_profile_windows.append(window)
                window.show()
        except (KeyError, ValueError, RuntimeError) as error:
            self._say(str(error))

    def _refresh_roi_scalar_plot_windows(self, snapshot=None) -> None:
        """Refresh live scalar plots only after a measurement snapshot publishes."""

        for window in tuple(self._roi_scalar_plot_windows):
            try:
                window.on_measurements_updated(snapshot)
            except RuntimeError as error:
                if "deleted" in str(error).lower():
                    try:
                        self._roi_scalar_plot_windows.remove(window)
                    except ValueError:
                        pass
                else:
                    logger.exception("Failed to refresh ROI scalar plot window")
            except Exception:  # noqa: BLE001 - measurement remains successful
                logger.exception("Failed to refresh ROI scalar plot window")

    def _create_roi_class_from_dialog(self) -> None:
        """Create one class through the minimum safe class-management flow."""

        if self.viewer is None:
            raise RuntimeError("Object classes can be created after the GUI opens")
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QColorDialog, QInputDialog

        from ..editing.roi_commands import CreateObjectClass

        parent = self.viewer.window._qt_window
        name, accepted = QInputDialog.getText(
            parent,
            "New Subcellular Object Class",
            "Class name:",
        )
        name = str(name).strip()
        if not accepted or not name:
            return
        color = QColorDialog.getColor(
            QColor("#2ec4b6"),
            parent,
            "Class color",
        )
        if not color.isValid():
            return
        red, green, blue, alpha = color.getRgbF()
        command = CreateObjectClass(
            self.roi_manager,
            name,
            (float(red), float(green), float(blue), float(alpha)),
        )
        self._run_edit_action(self.edit_history.do, command)
        if self._subcellular_objects_panel is not None:
            self._subcellular_objects_panel.refresh()
            self._subcellular_objects_panel.select_class(command.created_class_id)

    def get_cell_info_text(self) -> str:
        """Build the cell info display text for the currently selected cell.

        Returns:
            Formatted string with cell details (name, position, fate, etc.).
        """
        if not self.current_cell_name:
            return "No cell selected"

        cell = self.get_selected_cell()
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

        resolved_selection = self.get_selected_nucleus(self.current_time)
        selected_nucleus = (
            resolved_selection[0]
            if resolved_selection is not None
            and resolved_selection[1] == self.current_time
            else None
        )

        # Pre-compute visualization-mode colors for the whole frame
        # (batched for efficiency; skipped in editing mode).
        if self._viz_mode:
            viz_colors = self.color_engine.colors_for_frame(
                nuclei, self.manager, self.current_time,
                selected_name="",
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
            is_selected = nuc is selected_nucleus

            if self._viz_mode:
                # Visualization mode — rule-engine colors
                if is_selected:
                    r, g, b, a = self.color_engine.selected_color
                else:
                    r, g, b, a = viz_colors[viz_idx]
                colors.append([r, g, b, a])
                if is_selected:
                    selected_idx = len(centers) - 1
            else:
                # Editing mode — status-based palette
                if is_selected:
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
                config = self.manager.config
                plane_start = int(config.plane_start) if config is not None else 1
                provider_plane = self.current_plane - plane_start + 1
                plane_data = self.image_provider.get_plane(
                    self.current_time, provider_plane, channel=ch
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

    def _plane_bounds(self) -> tuple[int, int]:
        """Return the inclusive ``(first, last)`` z-plane of the current movie."""
        config = self.manager.config
        plane_start = int(config.plane_start) if config is not None else 1
        if self.image_provider is not None:
            plane_end = plane_start + max(0, int(self.image_provider.num_planes) - 1)
        elif config is not None:
            plane_end = int(config.plane_end)
        else:
            plane_end = plane_start + 29
        return plane_start, plane_end

    def _snap_plane_to_nucleus(self, nuc) -> None:
        """Force the displayed slice onto *nuc*'s centroid z.

        This is the single binding between the active cell and the image
        display: whenever a nucleus becomes active, the slice follows it.
        Callers that must not trigger a redraw can rely on this touching
        only ``current_plane`` -- it never calls ``update_display()``.
        """
        if nuc is None:
            return
        plane_start, plane_end = self._plane_bounds()
        self.current_plane = max(
            plane_start,
            min(round(nuc.z + NUCZINDEXOFFSET), plane_end),
        )

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

        Last resort: if neither lookup finds a nucleus at this timepoint,
        snap to the selection's anchored nucleus so the slice still tracks
        the active cell.
        """
        cell = self.get_selected_cell()
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
                cell = real_child
        elif self.current_time < cell.start_time:
            if cell.parent is not None and cell.parent.nuclei:
                cell = cell.parent

        nuc = cell.get_nucleus_at(self.current_time)
        if nuc is None:
            nuc = self._find_nucleus_via_chain(cell, self.current_time)
        if nuc:
            self._set_selection_from_nucleus(self.current_time, nuc)
            self._snap_plane_to_nucleus(nuc)
            return

        # Last resort: keep the display bound to the active cell even when
        # this timepoint has no nucleus to follow.  _find_nucleus_via_chain
        # only walks *outside* the cell's known range, so an interior gap
        # (present at T3 and T5, missing at T4) lands here — as does a time
        # past a cell's death with no real daughter.  Snapping to the
        # anchored nucleus puts the slice at the best Z we know for this
        # cell instead of stranding the user on an unrelated plane.  The
        # anchor itself is deliberately left alone: there is no nucleus at
        # ``current_time`` to re-anchor onto.
        self._snap_plane_to_nucleus(self._nucleus_at_anchor())

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
        """Route one committed edit to only the derived state it affects."""
        from ..editing.commands import EditEffect

        cmd = self.edit_history.last_command
        self._last_post_commit_refresh_error = None
        self._sync_tracking_provenance(cmd)
        effects = (
            frozenset((EditEffect.NUCLEI_TOPOLOGY,))
            if cmd is None
            else cmd.effects
        )
        nuclear_effects = {
            EditEffect.NUCLEI_TOPOLOGY,
            EditEffect.NUCLEUS_GEOMETRY,
        }
        has_nuclear_effect = bool(effects & nuclear_effects)
        is_structural = EditEffect.NUCLEI_TOPOLOGY in effects

        if has_nuclear_effect:
            # Nuclear expression values are derived from nucleus geometry and
            # topology. ROI-only edits must not make them stale.
            self.manager.mark_data_edited(
                (id(self.edit_history), self.edit_history.change_counter)
            )

        if EditEffect.ROI_GEOMETRY in effects:
            object_id = getattr(cmd, "object_id", None)
            timepoint = getattr(cmd, "timepoint", getattr(cmd, "time", None))
            if object_id is not None and timepoint is not None:
                self.roi_measurement_engine.invalidate_frame(object_id, timepoint)
            else:
                self.roi_measurement_engine.cache.clear()
        elif EditEffect.ROI_ASSOCIATION in effects:
            object_id = getattr(cmd, "object_id", None)
            timepoint = getattr(cmd, "timepoint", getattr(cmd, "time", None))
            if object_id is not None and timepoint is not None:
                self.roi_measurement_engine.invalidate_association(
                    object_id, timepoint
                )
        if EditEffect.CONFIG in effects:
            self.roi_measurement_engine.invalidate_calibration()

        if is_structural:
            # A few programmatic callers still set ``current_cell_name``
            # directly.  Capture its physical nucleus while the pre-edit tree
            # is still available; after a rename/process that old lookup name
            # may no longer exist.  Normal click/select paths are already
            # anchored, so this cannot redirect them to the command target.
            if self.selection_anchor is None and self.current_cell_name:
                self.get_selected_nucleus()
            self.manager.set_all_successors()
            self.manager.process()

            # Naming and topology edits can change every display name.  The
            # physical selection anchor, rather than the command being undone,
            # determines which cell remains selected.
            self._resolve_selection_after_rebuild()

            # Structural edits (relink, kill, add) change the lineage tree,
            # so all lineage tree panels need a full rebuild.
            for lw in self._lineage_widgets:
                lw.rebuild_tree()
            if self._lineage_list:
                self._lineage_list.rebuild()

        if is_structural or EditEffect.ROI_ASSOCIATION in effects:
            self.roi_manager.reconcile_cells(self._resolve_roi_cell_anchor)

        # Rendering is an observer of the curated data, not part of the edit
        # transaction. At this point the command, revision, provenance, and
        # undo entry have already committed. Do not let a napari/layer redraw
        # failure masquerade as a rejected edit (which can leave a cyan draft
        # drawn over the newly curated marker).
        if has_nuclear_effect:
            for window in tuple(self._expression_plot_windows):
                try:
                    window.on_document_edited(structural=is_structural)
                except RuntimeError as error:
                    if "deleted" in str(error).lower():
                        try:
                            self._expression_plot_windows.remove(window)
                        except ValueError:
                            pass
                    else:
                        self._report_committed_refresh_failure(cmd, error)
                except Exception as error:  # noqa: BLE001 - optional observer
                    self._report_committed_refresh_failure(cmd, error)
        try:
            self.update_display()
        except Exception as error:
            self._report_committed_refresh_failure(cmd, error)

    def _report_committed_refresh_failure(self, command, error: Exception) -> None:
        """Surface an observer failure without changing committed edit state."""

        self._last_post_commit_refresh_error = error
        description = command.description if command is not None else "Edit"
        message = (
            "The data change is committed and undoable, but the display "
            "refresh failed. Refresh the view before continuing."
        )
        logger.warning(
            "Post-commit refresh failed after %s",
            description,
            exc_info=(type(error), error, error.__traceback__),
        )
        self._say(message)

    def _run_edit_action(self, action, *args, **kwargs):
        """Run a GUI edit action without replaying a committed command.

        A PostCommitCallbackError means the data and undo entry already
        exist. Retrying the action would create duplicate nuclei or links;
        only rebuild/redraw observers are safe to retry.
        """

        try:
            return action(*args, **kwargs)
        except PostCommitCallbackError as error:
            command = error.command
            self._report_committed_refresh_failure(
                command,
                error.__cause__ or error,
            )
            try:
                self.edit_history.retry_post_commit(error)
            except Exception as retry_error:
                self._report_committed_refresh_failure(command, retry_error)
            # Undo/redo normally return the affected command. Preserve that
            # contract so the UI reports the operation that already committed.
            return command

    # ── Multi-panel lineage management ──────────────────────────

    def _sync_tracking_provenance(self, command) -> None:
        """Keep the saved proposal aligned with tracking-command undo/redo."""
        if (
            command is None
            or command.__class__.__name__ != "ApplyTrackingProposal"
            or not command.__class__.__module__.endswith(".tracking.integration")
        ):
            return
        try:
            command.detection_mapping
            applied = True
        except RuntimeError:
            applied = False
        present = any(item is command.result for item in self._tracking_results)
        if applied and not present:
            self._tracking_results.append(command.result)
        elif not applied and present:
            self._tracking_results = [
                item for item in self._tracking_results
                if item is not command.result
            ]

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
            # Napari/app-model menu labels and construction order can vary by
            # version or locale.  These are primary feature entry points, so
            # never silently omit them merely because no discoverable Window
            # menu existed yet.
            logger.warning(
                "No existing Window menu was discoverable; creating an AceTree one"
            )
            window_menu = menu_bar.addMenu("&Window")

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

        # Add independent visualization-window actions.
        window_menu.addSeparator()
        from qtpy.QtWidgets import QAction
        add_panel_action = QAction("New Lineage Panel...", qt_window)
        add_panel_action.triggered.connect(self._on_new_lineage_panel)
        window_menu.addAction(add_panel_action)
        expression_action = QAction("New Expression Plot…", qt_window)
        expression_action.setStatusTip(
            "Plot one or more cells from any measured image channel"
        )
        expression_action.triggered.connect(self.open_expression_plot_window)
        window_menu.addAction(expression_action)
        comparison_action = QAction("New Expression Comparison…", qt_window)
        comparison_action.setStatusTip(
            "Compare one cell across multiple AceTree XML datasets"
        )
        comparison_action.triggered.connect(self.open_expression_comparison_window)
        window_menu.addAction(comparison_action)
        open_comparison_result_action = QAction(
            "Open Expression Measurement Set / Result…", qt_window
        )
        open_comparison_result_action.setStatusTip(
            "Open an offline .aceexpr full measurement set or legacy fixed comparison"
        )
        open_comparison_result_action.triggered.connect(
            lambda _checked=False: self.open_expression_comparison_result_window()
        )
        window_menu.addAction(open_comparison_result_action)
        self._panel_menu_actions = {
            "new_lineage": add_panel_action,
            "new_expression_plot": expression_action,
            "new_expression_comparison": comparison_action,
            "open_expression_result": open_comparison_result_action,
        }

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

    def _add_tracking_menu_actions(self) -> None:
        """Add stable, plain-language tracking entry points to the menu bar."""

        if self.viewer is None or self._edit_panel is None:
            return
        try:
            qt_window = self.viewer.window._qt_window
            menu_bar = qt_window.menuBar()
        except Exception:
            return

        tracking_menu = None
        for action in menu_bar.actions():
            if action.menu() and action.text().lower().replace("&", "") == "tracking":
                tracking_menu = action.menu()
                break
        if tracking_menu is None:
            tracking_menu = menu_bar.addMenu("&Tracking")

        from qtpy.QtWidgets import QAction

        manual_action = QAction("Manual Track / Place Nuclei", qt_window)
        manual_action.setStatusTip(
            "Toggle manual right-click placement from the selected cell"
        )
        manual_action.triggered.connect(
            lambda _checked=False: self._edit_panel._btn_track.click()
        )
        selected_action = QAction("Track Selected Cell Forward…", qt_window)
        selected_action.setStatusTip(
            "Build and review a sparse forward draft; divisions are supported"
        )
        selected_action.triggered.connect(
            lambda _checked=False: self._edit_panel._on_auto_track_forward()
        )
        whole_action = QAction("Track Whole Movie…", qt_window)
        whole_action.setStatusTip(
            "Open reviewed Modern StarryNite, LoG + LAP, DoG + LAP, or advanced "
            "legacy exact whole-movie tracking"
        )
        whole_action.triggered.connect(
            lambda _checked=False: self._edit_panel._on_global_track()
        )
        relink_action = QAction("Relink Selected Cells…", qt_window)
        relink_action.setStatusTip(
            "Choose a source and target nucleus and create an undoable link"
        )
        relink_action.triggered.connect(
            lambda _checked=False: self._edit_panel._btn_relink.click()
        )
        show_panel_action = QAction("Show Edit & Tracking Tools", qt_window)
        show_panel_action.setStatusTip(
            "Reveal the scrollable dock containing all manual and automated tools"
        )
        show_panel_action.triggered.connect(
            lambda _checked=False: self._show_edit_tracking_panel()
        )

        tracking_menu.addAction(manual_action)
        tracking_menu.addAction(selected_action)
        tracking_menu.addAction(whole_action)
        tracking_menu.addSeparator()
        tracking_menu.addAction(relink_action)
        tracking_menu.addSeparator()
        tracking_menu.addAction(show_panel_action)
        self._tracking_menu = tracking_menu
        self._tracking_menu_actions = {
            "manual": manual_action,
            "selected_forward": selected_action,
            "whole_movie": whole_action,
            "relink": relink_action,
            "show_panel": show_panel_action,
        }

    def _show_edit_tracking_panel(self) -> None:
        """Reveal the Edit & Tracking dock without relying on the Window menu."""

        if self.viewer is None or self._edit_panel is None:
            return
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                dock_wrappers = tuple(self.viewer.window._dock_widgets.values())
            dock = next(
                (
                    candidate
                    for candidate in dock_wrappers
                    if candidate.widget() is self._edit_panel
                ),
                None,
            )
            if dock is not None:
                dock.setVisible(True)
                dock.raise_()
            self._edit_panel.setVisible(True)
        except (AttributeError, RuntimeError):
            logger.debug("Could not reveal the Edit & Tracking Tools dock")

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

    def _add_objects_menu_actions(self) -> None:
        """Add dedicated ROI measurement and dock entry points."""

        try:
            qt_window = self.viewer.window._qt_window
            menu_bar = qt_window.menuBar()
        except Exception:
            return
        objects_menu = None
        for action in menu_bar.actions():
            if action.menu() and action.text().lower().replace("&", "") == "objects":
                objects_menu = action.menu()
                break
        if objects_menu is None:
            objects_menu = menu_bar.addMenu("&Objects")

        from qtpy.QtWidgets import QAction

        measure_action = QAction("Measure Subcellular Objects…", qt_window)
        measure_action.setStatusTip(
            "Measure raw scalar intensities and optional thick-line profiles"
        )
        measure_action.triggered.connect(self._on_measure_rois)
        objects_menu.addAction(measure_action)
        show_action = QAction("Show Subcellular Objects", qt_window)
        show_action.triggered.connect(self._show_subcellular_objects_panel)
        objects_menu.addAction(show_action)

    def _show_subcellular_objects_panel(self) -> None:
        if self.viewer is None or self._subcellular_objects_panel is None:
            return
        try:
            dock = next(
                (
                    item
                    for item in self.viewer.window._dock_widgets.values()
                    if item.widget() is self._subcellular_objects_panel
                ),
                None,
            )
            if dock is not None:
                dock.setVisible(True)
                dock.raise_()
        except (AttributeError, RuntimeError):
            logger.debug("Could not reveal the Subcellular Objects dock")

    def _on_measure_rois(self) -> None:
        if self.viewer is None or self.image_provider is None:
            self._say("No image source is available for ROI measurement")
            return
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication, QDialog, QProgressDialog

        from .roi_measure_dialog import RoiMeasureDialog

        dialog = RoiMeasureDialog(self, parent=self.viewer.window._qt_window)
        if dialog.exec_() != QDialog.Accepted:
            return
        request = dialog.build_request()
        progress = QProgressDialog(
            "Measuring subcellular objects…",
            "Cancel",
            0,
            1,
            self.viewer.window._qt_window,
        )
        progress.setWindowTitle("Measure Subcellular Objects")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def progress_cb(completed: int, total: int) -> bool:
            progress.setMaximum(max(1, total))
            progress.setValue(completed)
            progress.setLabelText(
                f"Measuring ROI channel sample {completed}/{max(1, total)}…"
            )
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            snapshot = self.roi_measurement_engine.measure(
                request,
                progress_cb=progress_cb,
            )
        except Exception as error:
            logger.exception("ROI measurement failed")
            self._say(f"ROI measurement failed: {error}")
            return
        finally:
            progress.close()
        self._refresh_roi_scalar_plot_windows(snapshot)
        self._say(f"Measured {len(snapshot.samples)} ROI channel samples")

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
        self.current_expression_channel = at_channel
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

        completed_steps = 0

        def progress_cb(c_idx: int, n_ch: int, t_1based: int, n_tp: int) -> bool:
            nonlocal completed_steps
            completed_steps += 1
            progress.setValue(min(total_steps, completed_steps))
            progress.setLabelText(
                "Reading movie once for all channels "
                f"(selected correction: {correction_method}; "
                f"{completed_steps}/{total_steps})…"
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

        for window in tuple(self._expression_plot_windows):
            try:
                window.on_measurements_updated()
            except RuntimeError as error:
                if "deleted" in str(error).lower():
                    try:
                        self._expression_plot_windows.remove(window)
                    except ValueError:
                        pass
                else:
                    logger.exception("Failed to refresh expression plot window")
            except Exception:  # noqa: BLE001 - completed Measure remains successful
                logger.exception("Failed to refresh expression plot window")

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

        selected = self.get_selected_nucleus()
        if selected is None:
            _say(f"Cannot locate '{self.current_cell_name}' at t={self.current_time}")
            return
        nuc, _, index = selected

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
        self._run_edit_action(self.edit_history.do, cmd)
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
            self.cancel_relink_pick_mode()
            changed = True
        if self._exit_roi_mode():
            changed = True

        if self._edit_panel:
            dialog = getattr(self._edit_panel, "_auto_track_dialog", None)
            if dialog is not None:
                try:
                    if dialog.isVisible():
                        dialog.reject()
                        changed = True
                except RuntimeError:
                    self._edit_panel._auto_track_dialog = None
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

        dialog = self._global_tracking_dialog
        if dialog is not None:
            try:
                if dialog.isVisible():
                    dialog.reject()
                    changed = True
            except RuntimeError:
                self._global_tracking_dialog = None

    def _handle_space_shortcut(self) -> None:
        """Keep Space available for temporary pan while editing an ROI."""

        integration = self._roi_viewer_integration
        if integration is not None and integration.editing:
            return
        self.deselect_cell()

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
            self._run_edit_action(self.edit_history.undo)

        @self.viewer.bind_key("Control-y")
        def _redo(viewer):
            self._run_edit_action(self.edit_history.redo)

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

        # ── Application-wide Escape + WASD/Z shortcuts ─────────────
        # Napari's @viewer.bind_key("Escape") only fires when the canvas
        # has keyboard focus.  Clicking a toolbar button (Add, Track)
        # moves focus to the button, which absorbs or ignores Escape.
        # WASD/Z are additionally shadowed by napari's default layer
        # bindings (Shapes/Points layers bind 'a', 's', 'd' to their own
        # modes), so @viewer.bind_key never sees the letter at all.
        # Install QShortcuts on the main window so these keys work
        # regardless of which widget has focus.  WindowShortcut context
        # still lets QLineEdit / QSpinBox inside dialogs consume the
        # letter key first, so typing in text fields isn't hijacked.
        try:
            from qtpy.QtCore import Qt
            from qtpy.QtGui import QKeySequence
            from qtpy.QtWidgets import QShortcut

            qt_window = self.viewer.window._qt_window  # type: ignore[attr-defined]
            self._escape_shortcut = QShortcut(QKeySequence("Escape"), qt_window)
            self._escape_shortcut.setContext(Qt.ApplicationShortcut)
            self._escape_shortcut.activated.connect(self._exit_all_modes)

            self._nav_shortcuts = []
            nav_bindings = [
                ("A", self.prev_time),
                ("Z", self.prev_time),
                ("D", self.next_time),
                ("W", self.next_plane),
                ("S", self.prev_plane),
                ("Space", self._handle_space_shortcut),
            ]
            for key, handler in nav_bindings:
                sc = QShortcut(QKeySequence(key), qt_window)
                sc.setContext(Qt.WindowShortcut)
                sc.activated.connect(handler)
                self._nav_shortcuts.append(sc)
                if key == "Space":
                    self._space_shortcut = sc
        except Exception as e:
            logger.warning("Could not install application-wide shortcuts: %s", e)
