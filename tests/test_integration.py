"""Integration tests — full pipeline: load → name → build tree → verify."""

from __future__ import annotations

from pathlib import Path

import pytest

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.io.nuclei_writer import write_nuclei_zip


class TestFullPipeline:
    """Test the complete load → process → query pipeline."""

    def test_load_process_query(self, sample_nuclei_zip: Path):
        """Full pipeline: load ZIP → process → query cells."""
        mgr = NucleiManager()
        mgr.load(sample_nuclei_zip)
        mgr.process(do_identity=False)

        # Verify data loaded
        assert mgr.num_timepoints == 3

        # Verify tree built
        assert mgr.lineage_tree is not None
        assert mgr.lineage_tree.root is not None

        # Verify cell lookup works
        p0 = mgr.get_cell("P0")
        assert p0 is not None

    def test_round_trip_preserves_structure(self, sample_nuclei_zip: Path, tmp_path: Path):
        """Load → process → save → reload → process → verify same structure."""
        mgr1 = NucleiManager()
        mgr1.load(sample_nuclei_zip)
        mgr1.process(do_identity=False)

        # Save
        output = tmp_path / "round_trip.zip"
        mgr1.save(output)

        # Reload
        mgr2 = NucleiManager()
        mgr2.load(output)
        mgr2.process(do_identity=False)

        # Compare
        assert mgr2.num_timepoints == mgr1.num_timepoints
        assert mgr2.lineage_tree is not None
        assert mgr2.lineage_tree.num_cells > 0

    def test_spatial_queries_after_processing(self, sample_nuclei_zip: Path):
        """Spatial queries should work after processing."""
        mgr = NucleiManager()
        mgr.load(sample_nuclei_zip)

        # Find closest nucleus at T1 (P0 at ~300, 250)
        closest = mgr.find_closest_nucleus_2d(305, 255, 1)
        assert closest is not None
        assert closest.identity == "P0"

        # Find closest at T2 (AB at ~280, P1 at ~320)
        closest = mgr.find_closest_nucleus_2d(285, 245, 2)
        assert closest is not None
        assert closest.identity == "AB"


class TestSyntheticLineage:
    """Test with a synthetic multi-generation lineage."""

    def _build_synthetic_data(self) -> list[list[Nucleus]]:
        """Build a 5-timepoint synthetic lineage.

        T0: P0 (divides at T1)
        T1: AB (divides at T2), P1
        T2: ABa, ABp, P1 (divides at T3)
        T3: ABa, ABp, EMS, P2
        T4: ABa, ABp, EMS, P2 (all continuing)
        """
        def nuc(idx, x, y, z, name, pred=NILLI, s1=NILLI, s2=NILLI):
            return Nucleus(
                index=idx, x=x, y=y, z=z, size=20,
                identity=name, status=1,
                predecessor=pred, successor1=s1, successor2=s2,
                weight=5000,
            )

        return [
            # T0
            [nuc(1, 300, 250, 15.0, "P0")],
            # T1
            [
                nuc(1, 280, 240, 14.0, "AB", pred=1),
                nuc(2, 320, 260, 16.0, "P1", pred=1),
            ],
            # T2
            [
                nuc(1, 260, 230, 13.0, "ABa", pred=1),
                nuc(2, 300, 250, 15.0, "ABp", pred=1),
                nuc(3, 340, 270, 17.0, "P1", pred=2),
            ],
            # T3
            [
                nuc(1, 255, 225, 12.5, "ABa", pred=1),
                nuc(2, 295, 245, 14.5, "ABp", pred=2),
                nuc(3, 330, 265, 16.5, "EMS", pred=3),
                nuc(4, 350, 275, 17.5, "P2", pred=3),
            ],
            # T4
            [
                nuc(1, 250, 220, 12.0, "ABa", pred=1),
                nuc(2, 290, 240, 14.0, "ABp", pred=2),
                nuc(3, 325, 260, 16.0, "EMS", pred=3),
                nuc(4, 345, 270, 17.0, "P2", pred=4),
            ],
        ]

    def test_synthetic_lineage_builds(self):
        mgr = NucleiManager()
        mgr.nuclei_record = self._build_synthetic_data()
        mgr.set_all_successors()
        mgr.process(do_identity=False)

        assert mgr.lineage_tree is not None
        assert mgr.lineage_tree.num_cells > 0

    def test_synthetic_cell_counts(self):
        mgr = NucleiManager()
        mgr.nuclei_record = self._build_synthetic_data()
        mgr.set_all_successors()
        mgr.process(do_identity=False)

        tree = mgr.lineage_tree
        assert tree is not None
        assert tree.cell_counts[0] == 1  # T0: P0
        assert tree.cell_counts[1] == 2  # T1: AB, P1
        assert tree.cell_counts[2] == 3  # T2: ABa, ABp, P1
        assert tree.cell_counts[3] == 4  # T3: ABa, ABp, EMS, P2
        assert tree.cell_counts[4] == 4  # T4: same 4

    def test_synthetic_division_detected(self):
        mgr = NucleiManager()
        mgr.nuclei_record = self._build_synthetic_data()
        mgr.set_all_successors()

        # Verify P0 has two successors
        p0 = mgr.nuclei_record[0][0]
        assert p0.successor1 == 1
        assert p0.successor2 == 2

        # Verify AB has two successors (divides at T1->T2)
        ab = mgr.nuclei_record[1][0]
        assert ab.successor1 == 1
        assert ab.successor2 == 2

        # Verify P1 at T2 has two successors (divides at T2->T3)
        p1_t2 = mgr.nuclei_record[2][2]
        assert p1_t2.successor1 == 3
        assert p1_t2.successor2 == 4

    def test_synthetic_write_reload(self, tmp_path: Path):
        mgr = NucleiManager()
        mgr.nuclei_record = self._build_synthetic_data()
        mgr.set_all_successors()

        # Save
        output = tmp_path / "synthetic.zip"
        mgr.save(output)

        # Reload
        mgr2 = NucleiManager()
        mgr2.load(output)

        assert mgr2.num_timepoints == 5
        assert len(mgr2.nuclei_at(4)) == 4
