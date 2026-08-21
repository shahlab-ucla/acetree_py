"""NucleiManager — central data store and processing pipeline.

This is the main orchestrator that coordinates between nuclei loading,
naming, lineage tree building, and data access. It holds the
nuclei_record and provides the API that the GUI layer uses.

Processing pipeline:
    1. load() — read nuclei from ZIP file
    2. set_all_successors() — compute predecessor/successor links
    3. compute_red_weights() — apply expression corrections
    4. process() — run naming + build lineage tree

Ported from: org.rhwlab.snight.NucleiMgr
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io.auxinfo import (
    AuxInfo,
    auxinfo_from_axes,
    auxinfo_v2_path,
    is_manual_auxinfo_v2,
    load_auxinfo,
    stage_auxinfo_v2,
)
from ..io.config import AceTreeConfig
from ..io.nuclei_reader import read_nuclei_zip
from ..io.nuclei_writer import stage_nuclei_zip
from ..naming.identity import NEWCANONICAL, IdentityAssigner
from ..naming.division_caller import DivisionCaller
from ..naming.rules import RuleManager
from ..naming.validation import NamingWarning, validate_naming
from .cell import Cell
from .lineage import LineageTree, build_lineage_tree
from .movie import Movie
from .nucleus import NILLI, Nucleus

logger = logging.getLogger(__name__)


def _unused_sibling_path(destination: Path, *, suffix: str) -> Path:
    """Reserve an unused sibling name for a same-filesystem rollback rename."""
    fd, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
    )
    os.close(fd)
    path = Path(name)
    path.unlink()
    return path


def _discard_staged_file(path: Path | None) -> None:
    """Remove a private staged file without masking the save result."""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove staged save file: %s", path, exc_info=True)


@dataclass(frozen=True)
class DivisionSuggestion:
    """Preview of a geometry-based daughter assignment for a manual edit."""

    first_name: str
    second_name: str
    confidence: float
    axis_label: str
    source: str
    ambiguous: bool = False


class NucleiManager:
    """Central data store and processing pipeline for nuclei data.

    Holds the nuclei_record (list of timepoints, each a list of Nucleus)
    and coordinates the naming and lineage tree building.

    Attributes:
        nuclei_record: The raw nuclei data, indexed by [timepoint][index].
        lineage_tree: The built lineage tree (None until process() is called).
        movie: Temporal/spatial bounds of the dataset.
        config: The configuration used to load this data.
        auxinfo: AuxInfo data (v1/v2 orientation info).
    """

    def __init__(self) -> None:
        self.nuclei_record: list[list[Nucleus]] = []
        self.lineage_tree: LineageTree | None = None
        self.movie: Movie = Movie()
        self.config: AceTreeConfig | None = None
        self.auxinfo: AuxInfo | None = None
        self._naming_method: int = NEWCANONICAL
        self._expr_corr: str = "none"
        # Set when an in-memory operation changes XML-backed configuration.
        # Ordinary Save rewrites the XML only while this flag is set, avoiding
        # needless churn of legacy configs for nuclei-only edits.
        self._config_dirty: bool = False
        # Monotonic document revision used to bind derived measurements to
        # the exact nuclei geometry/topology they were computed from.  Loading
        # starts at revision 0; every edit (including undo/redo) advances it.
        self._data_revision: int = 0
        self._last_data_edit_token: object | None = None
        self.expression_measurements = None
        # False for reloaded legacy nuclei: the file format contains values
        # but no proof that they were measured after the last saved geometry.
        self.expression_measurement_freshness_known: bool = True
        self.naming_warnings: list[NamingWarning] = []
        # The last-run IdentityAssigner is kept so the GUI can reach the
        # topology-inferred per-timepoint axes (via
        # ``identity_assigner.division_caller._get_local_axes``).  Used by
        # ``get_ap_direction_at`` to resolve the AP unit vector for manual
        # division-daughter naming.
        self.identity_assigner: IdentityAssigner | None = None
        # Frame-level cache for alive_nuclei_at() — avoids recomputing
        # the same filter 3+ times per display update cycle.
        self._alive_cache_time: int = -1
        self._alive_cache_result: list[Nucleus] = []

    @property
    def data_revision(self) -> int:
        """Monotonic revision of data that can affect pixel measurements."""

        return self._data_revision

    def mark_data_edited(self, edit_token: object | None = None) -> int:
        """Advance and return the document revision after an edit commits.

        ``edit_token`` makes post-commit observer retries idempotent.  A retry
        may call the GUI refresh boundary twice for one already-committed edit;
        it must not masquerade as a second data change.
        """

        if edit_token is not None and edit_token == self._last_data_edit_token:
            return self._data_revision
        self._data_revision += 1
        self._last_data_edit_token = edit_token
        return self._data_revision

    @classmethod
    def new_empty(cls, config: AceTreeConfig, num_timepoints: int) -> NucleiManager:
        """Create an empty NucleiManager for a new dataset (no nuclei yet).

        Args:
            config: The dataset configuration (resolution, planes, etc.).
            num_timepoints: Number of timepoints to pre-allocate.

        Returns:
            A NucleiManager with empty nuclei_record, ready for manual annotation.
        """
        mgr = cls()
        mgr.config = config
        mgr.movie = Movie(
            xy_res=config.xy_res,
            z_res=config.z_res,
            num_planes=config.plane_end,
        )
        mgr._naming_method = config.naming_method.value
        mgr._expr_corr = config.expr_corr
        mgr.nuclei_record = [[] for _ in range(num_timepoints)]
        if config.axis_given:
            candidate = AuxInfo(
                version=1,
                data={"axis": config.axis_given.upper(), "ang": "0"},
            )
            if candidate.has_orientation:
                mgr.auxinfo = candidate
        return mgr

    @classmethod
    def from_config(cls, config: AceTreeConfig) -> NucleiManager:
        """Create a NucleiManager from an AceTreeConfig and load data.

        Args:
            config: The parsed configuration.

        Returns:
            A fully loaded NucleiManager (nuclei read, successors set,
            red weights computed). Call process() to run naming + tree building.
        """
        mgr = cls()
        mgr.config = config

        # Set parameters from config
        mgr.movie = Movie(
            xy_res=config.xy_res,
            z_res=config.z_res,
            num_planes=config.plane_end,
        )
        mgr._naming_method = config.naming_method.value
        mgr._expr_corr = config.expr_corr

        # Load nuclei from ZIP
        mgr.load(config.zip_file)

        # Load AuxInfo — try alongside the zip file first, then alongside
        # the config file (handles cases where the zip has been copied to a
        # different location than what the config originally referenced).
        for candidate in (
            config.zip_file.with_suffix(""),
            config.config_file.parent / config.zip_file.with_suffix("").name,
        ):
            ai = load_auxinfo(candidate)
            if ai.has_orientation:
                mgr.auxinfo = ai
                break

        # Legacy XML orientation is the final explicit-metadata fallback.
        if mgr.auxinfo is None and config.axis_given:
            candidate = AuxInfo(
                version=1,
                data={"axis": config.axis_given.upper(), "ang": "0"},
            )
            if candidate.has_orientation:
                mgr.auxinfo = candidate

        return mgr

    def load(self, zip_path: Path) -> None:
        """Load nuclei from a ZIP archive.

        Args:
            zip_path: Path to the nuclei ZIP file.
        """
        logger.info("Loading nuclei from %s", zip_path)
        self._data_revision = 0
        self._last_data_edit_token = None
        self._config_dirty = False
        self.nuclei_record = read_nuclei_zip(zip_path)
        self.expression_measurements = None
        self.expression_measurement_freshness_known = False

        if self.nuclei_record:
            self.movie.start_time = 1
            self.movie.end_time = len(self.nuclei_record)
            logger.info(
                "Loaded %d timepoints (%d-%d)",
                len(self.nuclei_record),
                self.movie.start_time,
                self.movie.end_time,
            )

        # Set successor links and compute red weights
        self.set_all_successors()
        self.compute_red_weights()

    def invalidate_alive_cache(self) -> None:
        """Invalidate the alive-nuclei cache (call after edits)."""
        self._alive_cache_time = -1

    def process(self, do_identity: bool = True) -> None:
        """Run the full processing pipeline: naming + tree building.

        Args:
            do_identity: If True, run identity assignment (naming).
        """
        self.invalidate_alive_cache()
        if not self.nuclei_record:
            logger.warning("No nuclei loaded; nothing to process")
            return

        # Step 1: Identity assignment (naming)
        if do_identity:
            self._run_naming()

        # Step 2: Build lineage tree
        self._build_tree()

        logger.info(
            "Processing complete: %d cells in lineage tree",
            self.lineage_tree.num_cells if self.lineage_tree else 0,
        )

    def save(
        self,
        zip_path: Path,
        start_time: int = 1,
        *,
        final_commit: Callable[[], None] | None = None,
    ) -> None:
        """Save the nuclei archive and orientation sidecar as one transaction.

        Args:
            zip_path: Output path for the ZIP file.
            start_time: Starting timepoint number for file naming.
            final_commit: Optional atomic final replacement coordinated with
                the nuclei save. If it raises before changing its destination,
                the previous archive and AuxInfo sidecar are restored. The GUI
                uses this to commit a pre-staged dirty XML configuration.
        """
        zip_path = Path(zip_path)
        aux_base = zip_path.with_suffix("")

        # Prepare every new byte before changing a user-visible file.  The
        # archive is normally the final atomic replacement; a coordinated
        # final commit retains the old archive until that added step succeeds.
        archive_stage = stage_nuclei_zip(
            self.nuclei_record,
            zip_path,
            start_time=start_time,
        )
        sidecar_path = auxinfo_v2_path(aux_base)
        sidecar_stage: Path | None = None
        sidecar_operation = False
        sidecar_backup: Path | None = None
        sidecar_changed = False
        archive_backup: Path | None = None
        archive_changed = False

        try:
            if (
                self.auxinfo is not None
                and self.auxinfo.is_v2
                and self.auxinfo.has_orientation
            ):
                sidecar_path, sidecar_stage = stage_auxinfo_v2(
                    self.auxinfo,
                    aux_base,
                )
                sidecar_operation = True
            elif is_manual_auxinfo_v2(aux_base):
                # Undoing a manual orientation removes only the sidecar that
                # AceTree created; acquisition metadata is never deleted.
                sidecar_operation = True

            if sidecar_operation:
                if sidecar_path.exists():
                    sidecar_backup = _unused_sibling_path(
                        sidecar_path,
                        suffix=".rollback",
                    )
                    os.replace(sidecar_path, sidecar_backup)
                if sidecar_stage is not None:
                    os.replace(sidecar_stage, sidecar_path)
                sidecar_changed = True

            if final_commit is not None and zip_path.exists():
                archive_backup = _unused_sibling_path(
                    zip_path,
                    suffix=".rollback",
                )
                os.replace(zip_path, archive_backup)
            os.replace(archive_stage, zip_path)
            archive_changed = True
            if final_commit is not None:
                final_commit()
        except BaseException:
            rollback_errors: list[BaseException] = []
            if final_commit is not None:
                try:
                    if archive_changed:
                        zip_path.unlink(missing_ok=True)
                    if archive_backup is not None and archive_backup.exists():
                        os.replace(archive_backup, zip_path)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
                    logger.exception(
                        "Save failed and the archive rollback also failed for %s",
                        zip_path,
                    )
            try:
                if sidecar_changed:
                    sidecar_path.unlink(missing_ok=True)
                if sidecar_backup is not None and sidecar_backup.exists():
                    os.replace(sidecar_backup, sidecar_path)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
                logger.exception(
                    "Save failed and the AuxInfo rollback also failed for %s",
                    zip_path,
                )
            if rollback_errors:
                raise RuntimeError(
                    "Save failed and one or more previous dataset files could "
                    "not be restored"
                ) from rollback_errors[0]
            raise
        finally:
            _discard_staged_file(archive_stage)
            _discard_staged_file(sidecar_stage)

        if sidecar_backup is not None:
            try:
                sidecar_backup.unlink(missing_ok=True)
            except OSError:
                # Both committed files are already valid.  A hidden backup is
                # safer than reporting a failed save after durable commit.
                logger.warning(
                    "Could not remove completed-save backup: %s",
                    sidecar_backup,
                    exc_info=True,
                )
        if archive_backup is not None:
            try:
                archive_backup.unlink(missing_ok=True)
            except OSError:
                # Both the new archive and final commit are already valid. A
                # hidden backup is safer than reporting a failed durable save.
                logger.warning(
                    "Could not remove completed-save backup: %s",
                    archive_backup,
                    exc_info=True,
                )
        logger.info("Saved nuclei to %s", zip_path)

    # ── Data access ────────────────────────────────────────────────

    @property
    def num_timepoints(self) -> int:
        """Number of timepoints in the dataset."""
        return len(self.nuclei_record)

    @property
    def z_pix_res(self) -> float:
        """Z pixel resolution (z_res / xy_res)."""
        return self.movie.z_pix_res

    @property
    def ending_index(self) -> int:
        """1-based ending index."""
        if self.config:
            return min(self.config.ending_index, len(self.nuclei_record))
        return len(self.nuclei_record)

    def nuclei_at(self, time: int) -> list[Nucleus]:
        """Get nuclei at a timepoint (1-based).

        Args:
            time: 1-based timepoint.

        Returns:
            List of Nucleus objects at that timepoint, or empty list.
        """
        idx = time - 1
        if 0 <= idx < len(self.nuclei_record):
            return self.nuclei_record[idx]
        return []

    def alive_nuclei_at(self, time: int) -> list[Nucleus]:
        """Get only alive nuclei at a timepoint (1-based).

        Result is cached so that multiple calls within the same display
        update cycle (overlay, cell info, find_closest) don't re-filter.
        """
        if time == self._alive_cache_time:
            return self._alive_cache_result
        result = [n for n in self.nuclei_at(time) if n.is_alive]
        self._alive_cache_time = time
        self._alive_cache_result = result
        return result

    def find_closest_nucleus(
        self,
        x: float,
        y: float,
        z: float,
        time: int,
        z_scale: bool = True,
        require_hit: bool = False,
        image_plane: int | None = None,
    ) -> Nucleus | None:
        """Find the nucleus closest to (x, y, z) at a given timepoint.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
            z: Z coordinate (plane number).
            time: 1-based timepoint.
            z_scale: If True, scale z by z_pix_res for distance calculation.
            require_hit: If True, only return a nucleus if the click is
                within (or on) its projected circle at *image_plane*.
                When False (default), the closest nucleus is always returned.
            image_plane: The z-plane being viewed.  Required when
                *require_hit* is True; used to compute projected radii.

        Returns:
            The closest alive Nucleus, or None if no nuclei at that time
            (or none within projected radius when *require_hit* is True).
        """
        nuclei = self.alive_nuclei_at(time)
        if not nuclei:
            return None

        z_factor = self.z_pix_res if z_scale else 1.0
        best_nuc = None
        best_dist = float("inf")

        for nuc in nuclei:
            dx = nuc.x - x
            dy = nuc.y - y
            dz = (nuc.z - z) * z_factor
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if require_hit and image_plane is not None:
                # Only consider this nucleus if the click's 2D distance
                # is within the nucleus's projected circle on the image.
                proj_diam = self.nucleus_diameter(nuc, image_plane)
                if proj_diam <= 0:
                    continue  # Not visible at this z-plane
                proj_radius = proj_diam / 2.0
                dist_2d = math.sqrt(dx * dx + dy * dy)
                if dist_2d > proj_radius:
                    continue  # Click is outside the drawn circle

            if dist < best_dist:
                best_dist = dist
                best_nuc = nuc

        return best_nuc

    def find_closest_nucleus_2d(
        self,
        x: float,
        y: float,
        time: int,
        require_hit: bool = False,
        image_plane: int | None = None,
    ) -> Nucleus | None:
        """Find the nucleus closest to (x, y) at a given timepoint (2D only).

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
            time: 1-based timepoint.
            require_hit: If True, only return a nucleus if the click is
                within (or on) its projected circle at *image_plane*.
            image_plane: The z-plane being viewed.  Required when
                *require_hit* is True.

        Returns:
            The closest alive Nucleus, or None if no nuclei at that time
            (or none within projected radius when *require_hit* is True).
        """
        nuclei = self.alive_nuclei_at(time)
        if not nuclei:
            return None

        best_nuc = None
        best_dist = float("inf")

        for nuc in nuclei:
            dx = nuc.x - x
            dy = nuc.y - y
            dist = dx * dx + dy * dy  # No need for sqrt, just comparing

            if require_hit and image_plane is not None:
                proj_diam = self.nucleus_diameter(nuc, image_plane)
                if proj_diam <= 0:
                    continue
                proj_radius = proj_diam / 2.0
                if math.sqrt(dist) > proj_radius:
                    continue

            if dist < best_dist:
                best_dist = dist
                best_nuc = nuc

        return best_nuc

    def get_cell(self, name: str) -> Cell | None:
        """Look up a cell by name in the lineage tree.

        Args:
            name: Cell name (e.g. "ABala").

        Returns:
            The Cell object, or None if not found or tree not built.
        """
        if self.lineage_tree is None:
            return None
        return self.lineage_tree.get_cell(name)

    def get_nucleus_for_cell(self, name: str, time: int) -> Nucleus | None:
        """Get the Nucleus for a named cell at a specific timepoint.

        Args:
            name: Cell name.
            time: 1-based timepoint.

        Returns:
            The Nucleus, or None if not found.
        """
        cell = self.get_cell(name)
        if cell is None:
            return None
        return cell.get_nucleus_at(time)

    def nucleus_diameter(
        self,
        nuc: Nucleus,
        image_plane: int,
    ) -> float:
        """Compute the projected 2D diameter of a nucleus at a given image plane.

        The diameter shrinks as the nucleus is further from the focal plane.

        Args:
            nuc: The Nucleus.
            image_plane: The z-plane being viewed.

        Returns:
            The projected diameter in pixels, or 0 if the nucleus is too far.
        """
        radius = nuc.size / 2.0
        dz = abs(nuc.z - image_plane) * self.z_pix_res

        if dz >= radius:
            return 0.0

        # Pythagorean theorem for circle cross-section
        projected_radius = math.sqrt(radius * radius - dz * dz)
        return projected_radius * 2.0

    def has_circle(self, nuc: Nucleus, image_plane: int) -> bool:
        """Check if a nucleus is visible (has a non-zero projected diameter) at a plane."""
        return self.nucleus_diameter(nuc, image_plane) > 0

    # ── Internal processing methods ────────────────────────────────

    def set_all_successors(self) -> None:
        """Compute successor links from predecessor links.

        Iterates through all timepoints and uses the predecessor field
        in each nucleus to set the successor1/successor2 fields in the
        previous timepoint's nuclei.

        This is the inverse of the predecessor links that are stored in the
        nuclei files.

        Important: Only alive nuclei (status >= 1) contribute successor
        links. Dead nuclei's predecessor links are preserved in the data
        (for undo/resurrect) but are NOT used to build successor links,
        because stale predecessor links on dead nuclei create false
        division signals (e.g., a dead nucleus pointing to the same parent
        as an alive nucleus makes the parent appear to divide).
        """
        for t in range(len(self.nuclei_record) - 1):
            current = self.nuclei_record[t]
            next_nuclei = self.nuclei_record[t + 1]

            # Reset successors for current timepoint
            for nuc in current:
                nuc.successor1 = NILLI
                nuc.successor2 = NILLI

            # Only process alive nuclei — dead nuclei's predecessor links
            # are stale and create false division signals.
            alive_indices = []
            for j, next_nuc in enumerate(next_nuclei):
                if next_nuc.predecessor == NILLI:
                    continue
                if next_nuc.status >= 1:
                    alive_indices.append(j)

            for j in alive_indices:
                next_nuc = next_nuclei[j]
                pred_idx = next_nuc.predecessor - 1  # 1-based to 0-based
                if not (0 <= pred_idx < len(current)):
                    continue

                parent = current[pred_idx]
                if not parent.is_alive:
                    logger.warning(
                        "Live nucleus at t=%d idx=%d references dead predecessor "
                        "at t=%d idx=%d; link ignored",
                        t + 2, j + 1, t + 1, pred_idx + 1,
                    )
                    continue
                next_idx_1based = j + 1

                if parent.successor1 == NILLI:
                    parent.successor1 = next_idx_1based
                elif parent.successor2 == NILLI:
                    parent.successor2 = next_idx_1based
                else:
                    logger.warning(
                        "Nucleus at t=%d idx=%d already has 2 successors; "
                        "ignoring additional successor at t=%d idx=%d",
                        t + 1, pred_idx + 1, t + 2, next_idx_1based,
                    )

    def compute_red_weights(self) -> None:
        """Compute corrected red weights based on the expression correction method."""
        method = self._expr_corr
        if method == "none":
            return

        count = 0
        for nuclei in self.nuclei_record:
            for nuc in nuclei:
                if nuc.rwraw <= 0:
                    continue
                nuc.rweight = nuc.corrected_red(method)
                count += 1

        logger.info("Computed red weights for %d nuclei (method=%s)", count, method)

    def _run_naming(self) -> None:
        """Run the identity assignment pipeline."""
        assigner = IdentityAssigner(
            nuclei_record=self.nuclei_record,
            auxinfo=self.auxinfo,
            naming_method=self._naming_method,
            starting_index=0,
            ending_index=self.ending_index,
            z_pix_res=self.z_pix_res,
        )
        assigner.assign_identities()
        # Keep the assigner around so the GUI's manual-division path can
        # reach the topology-inferred per-timepoint AP axes via
        # ``self.get_ap_direction_at``.
        self.identity_assigner = assigner
        logger.info("Identity assignment complete")

    def _legacy_get_ap_direction_at(self, time: int) -> np.ndarray:
        """Return the AP unit-ish direction vector in pixel space at ``time``.

        Priority order (most-specific to fallback):

        1. **Topology-inferred per-timepoint axes** computed by the naming
           pipeline (``DivisionCaller._get_local_axes``) when the 4-cell
           stage was identified.  These are the axes the division-calling
           code itself uses, so daughter naming lines up with the canonical
           pipeline when it succeeded.
        2. **AuxInfo v2** — ``auxinfo.ap_orientation``.
        3. **AuxInfo v1** — ``auxinfo.axis[0]`` as ``A`` or ``P``.
        4. **Default** — ``+X`` (pixel-space) is anterior, matching Java
           AceTree's out-of-the-box convention.

        Used by the GUI to orient manual division-daughter naming ("a" vs
        "p") for cells the user annotates by hand.

        Args:
            time: 1-based timepoint.

        Returns:
            A 3-element numpy array (x, y, z in pixel space).  Not
            guaranteed to be unit length — callers should treat it as
            a *direction* and use a dot-product sign check rather than
            precise magnitude.
        """
        # 1. Topology-based local axes
        if (self.identity_assigner is not None
                and self.identity_assigner.division_caller is not None):
            try:
                axes = self.identity_assigner.division_caller._get_local_axes(time)
            except Exception:
                axes = None
            if axes is not None:
                ap = axes[0]
                if ap is not None:
                    return np.asarray(ap, dtype=float)

        # 2. AuxInfo v2 AP orientation vector
        if self.auxinfo is not None and self.auxinfo.is_v2:
            ap = self.auxinfo.ap_orientation
            if ap is not None:
                return np.asarray(ap, dtype=float)

        # 3. AuxInfo v1 axis string (first character is A or P)
        if self.auxinfo is not None:
            axis_str = self.auxinfo.axis or ""
            if axis_str and axis_str[0] in ("A", "P"):
                sign = 1.0 if axis_str[0] == "A" else -1.0
                return np.array([sign, 0.0, 0.0])

        # 4. Default: +X is anterior (Java AceTree convention)
        return np.array([1.0, 0.0, 0.0])

    def get_body_axes_at(
        self, time: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return anatomical ``(AP, LR, DV)`` axes at a 1-based timepoint.

        Explicit AuxInfo/manual vectors take precedence. Otherwise the last
        naming run's inferred frame is used. The one-based/zero-based
        conversion is centralised here so GUI edits cannot use the following
        frame's geometry by accident.
        """
        if time < 1:
            return None

        if self.auxinfo is not None and self.auxinfo.is_v2 and self.auxinfo.has_orientation:
            from ..naming.body_axes import BodyAxisFrame, BodyAxisValidationError

            try:
                frame = BodyAxisFrame.from_auxinfo_vectors(
                    self.auxinfo.ap_orientation,
                    self.auxinfo.lr_orientation,
                    provenance=("manual" if self.auxinfo.is_manual else "auxinfo_v2"),
                    reference_time=self.auxinfo.reference_time or None,
                )
                return frame.ap.copy(), frame.lr.copy(), frame.dv.copy()
            except BodyAxisValidationError:
                logger.warning("Ignoring invalid AuxInfo v2 body axes", exc_info=True)

        caller = self.identity_assigner.division_caller if self.identity_assigner else None
        if caller is not None:
            if caller.is_lineage_mode:
                try:
                    axes = caller._get_local_axes(time - 1)
                except Exception:
                    axes = None
                if axes is not None:
                    ap, lr, dv = axes
                    return (
                        np.asarray(ap, dtype=float),
                        np.asarray(lr, dtype=float),
                        np.asarray(dv, dtype=float),
                    )
                fallback = caller._lineage_fallback_frame()
                if fallback is not None:
                    ap, lr, dv = fallback
                    return (
                        np.asarray(ap, dtype=float),
                        np.asarray(lr, dtype=float),
                        np.asarray(dv, dtype=float),
                    )
            if (
                caller.founder_ap is not None
                and caller.founder_lr is not None
                and caller.founder_dv is not None
            ):
                return (
                    np.asarray(caller.founder_ap, dtype=float),
                    np.asarray(caller.founder_lr, dtype=float),
                    np.asarray(caller.founder_dv, dtype=float),
                )

        # Invert the sign/angle transform used for legacy v1 classification
        # so previews and automatic naming share exactly one convention.
        if self.auxinfo is not None and self.auxinfo.has_orientation and not self.auxinfo.is_v2:
            axis = self.auxinfo.axis.upper()
            signs = np.array([
                1.0 if axis[0] == "A" else -1.0,
                1.0 if axis[1] == "D" else -1.0,
                1.0 if axis[2] == "L" else -1.0,
            ])
            angle = math.radians(self.auxinfo.angle)
            rotation = np.array([
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ])
            ap = rotation @ np.array([signs[0], 0.0, 0.0])
            dv = rotation @ np.array([0.0, signs[1], 0.0])
            lr = rotation @ np.array([0.0, 0.0, signs[2]])
            return ap, lr, dv

        return None

    def get_ap_direction_at(self, time: int) -> np.ndarray:
        """Return AP at a 1-based time; use +X only when orientation is unknown."""
        axes = self.get_body_axes_at(time)
        return axes[0] if axes is not None else np.array([1.0, 0.0, 0.0])

    def suggest_division_names(
        self,
        parent: Nucleus,
        first_pos: tuple[float, float, float],
        second_pos: tuple[float, float, float],
        time: int,
    ) -> DivisionSuggestion:
        """Use the same rule and body frame as automation for a manual division."""
        rule_manager = RuleManager()
        rule = rule_manager.get_rule(parent.effective_name)
        # Founder daughters such as E/MS and C/P3 do not expose a directional
        # suffix.  Report the dominant component of the actual rule vector,
        # which is also what the classifier dots against.
        dominant = int(np.argmax(np.abs(rule.axis_vector)))
        axis_label = ("AP", "DV", "LR")[dominant]
        first = Nucleus(
            x=round(first_pos[0]), y=round(first_pos[1]), z=float(first_pos[2]), status=1,
        )
        second = Nucleus(
            x=round(second_pos[0]), y=round(second_pos[1]), z=float(second_pos[2]), status=1,
        )

        caller = self.identity_assigner.division_caller if self.identity_assigner else None
        source = "unknown"
        if caller is not None:
            if caller.is_v2:
                source = "manual axes" if (self.auxinfo and self.auxinfo.is_manual) else "AuxInfo v2"
            elif caller.is_lineage_mode:
                source = "inferred body frame"
            elif caller.is_founder_mode:
                source = "four-cell frame"
            else:
                source = "AuxInfo v1"
        else:
            axes = self.get_body_axes_at(time)
            if axes is not None:
                ap, lr, dv = axes
                caller = DivisionCaller(
                    rule_manager=rule_manager,
                    z_pix_res=self.z_pix_res,
                    founder_ap=ap,
                    founder_lr=lr,
                    founder_dv=dv,
                )
                source = (
                    "manual axes"
                    if self.auxinfo is not None and self.auxinfo.is_manual
                    else "body frame"
                )

        if caller is None:
            return DivisionSuggestion(
                first_name=rule.daughter1,
                second_name=rule.daughter2,
                confidence=0.0,
                axis_label=axis_label,
                source="canonical family (body axes unavailable)",
                ambiguous=True,
            )

        before = len(caller.classifications)
        has_complete_body_frame = getattr(caller, "has_complete_body_frame", None)
        used_body_frame = not (
            caller.is_lineage_mode
            and callable(has_complete_body_frame)
            and not has_complete_body_frame(time - 1)
        )
        name1, name2 = caller.assign_names(parent, first, second, timepoint=time - 1)
        classification = caller.classifications[-1] if len(caller.classifications) > before else None
        confidence = classification.confidence if classification is not None else 0.0
        expected = {rule.daughter1, rule.daughter2}
        if name1 == name2 or {name1, name2} != expected:
            name1, name2 = rule.daughter1, rule.daughter2
            confidence = 0.0
            source = "canonical family (body axes unavailable)"
        elif not used_body_frame:
            source = "canonical family (body axes unavailable)"
        return DivisionSuggestion(
            first_name=name1,
            second_name=name2,
            confidence=confidence,
            axis_label=axis_label,
            source=source,
            ambiguous=not name1 or not name2 or confidence < 0.3,
        )

    def set_manual_body_axes(self, frame) -> None:
        """Install a validated BodyAxisFrame for subsequent naming and saving."""
        ap, lr = frame.to_auxinfo_vectors()
        self.auxinfo = auxinfo_from_axes(
            ap,
            lr,
            z_pix_res=self.z_pix_res,
            reference_time=frame.reference_time or 0,
            quality=frame.quality,
            series_name=(self.config.config_file.stem if self.config else "manual"),
        )
        self.identity_assigner = None

    def _build_tree(self) -> None:
        """Build the lineage tree from the nuclei record."""
        self.lineage_tree = build_lineage_tree(
            nuclei_record=self.nuclei_record,
            starting_index=0,
            ending_index=self.ending_index,
            create_dummy_ancestors=True,
        )
        logger.info(
            "Lineage tree built: %d cells, root=%s",
            self.lineage_tree.num_cells,
            self.lineage_tree.root.name if self.lineage_tree.root else "None",
        )

        # Run naming validation
        if self.lineage_tree is not None:
            self.naming_warnings = validate_naming(
                self.lineage_tree, self.nuclei_record,
            )
            if self.naming_warnings:
                n_errors = sum(1 for w in self.naming_warnings if w.severity == "error")
                n_warns = sum(1 for w in self.naming_warnings if w.severity == "warning")
                logger.info(
                    "Naming validation: %d errors, %d warnings",
                    n_errors, n_warns,
                )
