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
from pathlib import Path

from ..io.auxinfo import AuxInfo, load_auxinfo
from ..io.config import AceTreeConfig, NamingMethod
from ..io.nuclei_reader import read_nuclei_zip
from ..io.nuclei_writer import write_nuclei_zip
from ..naming.identity import MANUAL, NEWCANONICAL, IdentityAssigner
from ..naming.validation import NamingWarning, validate_naming
from .cell import Cell
from .lineage import LineageTree, build_lineage_tree
from .movie import Movie
from .nucleus import NILLI, Nucleus

logger = logging.getLogger(__name__)


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
        self.naming_warnings: list[NamingWarning] = []

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

        # Load AuxInfo
        if config.zip_file.exists():
            base = config.zip_file.with_suffix("")
            mgr.auxinfo = load_auxinfo(base)

        return mgr

    def load(self, zip_path: Path) -> None:
        """Load nuclei from a ZIP archive.

        Args:
            zip_path: Path to the nuclei ZIP file.
        """
        logger.info("Loading nuclei from %s", zip_path)
        self.nuclei_record = read_nuclei_zip(zip_path)

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

    def process(self, do_identity: bool = True) -> None:
        """Run the full processing pipeline: naming + tree building.

        Args:
            do_identity: If True, run identity assignment (naming).
        """
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

    def save(self, zip_path: Path, start_time: int = 1) -> None:
        """Save nuclei back to a ZIP archive.

        Args:
            zip_path: Output path for the ZIP file.
            start_time: Starting timepoint number for file naming.
        """
        write_nuclei_zip(self.nuclei_record, zip_path, start_time=start_time)
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
        """Get only alive nuclei at a timepoint (1-based)."""
        return [n for n in self.nuclei_at(time) if n.is_alive]

    def find_closest_nucleus(
        self,
        x: float,
        y: float,
        z: float,
        time: int,
        z_scale: bool = True,
    ) -> Nucleus | None:
        """Find the nucleus closest to (x, y, z) at a given timepoint.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
            z: Z coordinate (plane number).
            time: 1-based timepoint.
            z_scale: If True, scale z by z_pix_res for distance calculation.

        Returns:
            The closest alive Nucleus, or None if no nuclei at that time.
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
            if dist < best_dist:
                best_dist = dist
                best_nuc = nuc

        return best_nuc

    def find_closest_nucleus_2d(
        self,
        x: float,
        y: float,
        time: int,
    ) -> Nucleus | None:
        """Find the nucleus closest to (x, y) at a given timepoint (2D only).

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
            time: 1-based timepoint.

        Returns:
            The closest alive Nucleus, or None if no nuclei at that time.
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

        Important: alive nuclei (status >= 1) are processed first, so that
        dead nuclei cannot steal successor slots from living cells. Dead
        nuclei with predecessor links are still processed (for completeness)
        but only if their parent still has available successor slots.
        """
        for t in range(len(self.nuclei_record) - 1):
            current = self.nuclei_record[t]
            next_nuclei = self.nuclei_record[t + 1]

            # Reset successors for current timepoint
            for nuc in current:
                nuc.successor1 = NILLI
                nuc.successor2 = NILLI

            # Process alive nuclei first, then dead ones.
            # This ensures dead cells with stale predecessor links
            # cannot steal successor slots from living daughters.
            alive_indices = []
            dead_indices = []
            for j, next_nuc in enumerate(next_nuclei):
                if next_nuc.predecessor == NILLI:
                    continue
                if next_nuc.status >= 1:
                    alive_indices.append(j)
                else:
                    dead_indices.append(j)

            for j in alive_indices + dead_indices:
                next_nuc = next_nuclei[j]
                pred_idx = next_nuc.predecessor - 1  # 1-based to 0-based
                if not (0 <= pred_idx < len(current)):
                    continue

                parent = current[pred_idx]
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
        logger.info("Identity assignment complete")

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
