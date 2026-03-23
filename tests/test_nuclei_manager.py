"""Tests for acetree_py.core.nuclei_manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.core.nuclei_manager import NucleiManager


def _nuc(
    index: int,
    x: int = 300,
    y: int = 250,
    z: float = 15.0,
    identity: str = "",
    status: int = 1,
    pred: int = NILLI,
    succ1: int = NILLI,
    succ2: int = NILLI,
    rwraw: int = 0,
) -> Nucleus:
    return Nucleus(
        index=index, x=x, y=y, z=z, size=20,
        identity=identity, status=status,
        predecessor=pred, successor1=succ1, successor2=succ2,
        weight=5000, rwraw=rwraw, rwcorr1=10, rwcorr2=15,
        rwcorr3=12, rwcorr4=8,
    )


def _make_mgr_with_data() -> NucleiManager:
    """Create a NucleiManager with a simple 3-timepoint lineage."""
    mgr = NucleiManager()
    mgr.nuclei_record = [
        # T0: P0 dividing
        [_nuc(1, x=300, identity="P0", pred=NILLI)],
        # T1: AB and P1
        [
            _nuc(1, x=280, identity="AB", pred=1),
            _nuc(2, x=320, identity="P1", pred=1),
        ],
        # T2: ABa, ABp, P1
        [
            _nuc(1, x=260, identity="ABa", pred=1),
            _nuc(2, x=300, identity="ABp", pred=1),
            _nuc(3, x=340, identity="P1", pred=2),
        ],
    ]
    mgr.set_all_successors()
    return mgr


class TestNucleiManagerBasics:
    """Test basic NucleiManager operations."""

    def test_num_timepoints(self):
        mgr = _make_mgr_with_data()
        assert mgr.num_timepoints == 3

    def test_nuclei_at(self):
        mgr = _make_mgr_with_data()
        t1 = mgr.nuclei_at(1)  # 1-based
        assert len(t1) == 1
        assert t1[0].identity == "P0"

        t2 = mgr.nuclei_at(2)
        assert len(t2) == 2

    def test_nuclei_at_out_of_range(self):
        mgr = _make_mgr_with_data()
        assert mgr.nuclei_at(0) == []
        assert mgr.nuclei_at(100) == []

    def test_alive_nuclei_at(self):
        mgr = NucleiManager()
        mgr.nuclei_record = [
            [
                _nuc(1, identity="Live", status=1),
                _nuc(2, identity="Dead", status=-1),
            ],
        ]
        alive = mgr.alive_nuclei_at(1)
        assert len(alive) == 1
        assert alive[0].identity == "Live"


class TestSetAllSuccessors:
    """Test successor link computation."""

    def test_sets_successors_correctly(self):
        mgr = _make_mgr_with_data()
        # T0 P0 should have successor1 and successor2 set
        p0 = mgr.nuclei_record[0][0]
        assert p0.successor1 == 1  # AB at index 1 (1-based)
        assert p0.successor2 == 2  # P1 at index 2 (1-based)

    def test_non_dividing_cell_has_one_successor(self):
        mgr = _make_mgr_with_data()
        # T1 P1 (index 2, 1-based) should have one successor
        p1 = mgr.nuclei_record[1][1]
        assert p1.successor1 == 3  # P1 at T2 index 3 (1-based)
        assert p1.successor2 == NILLI

    def test_dividing_cell_has_two_successors(self):
        mgr = _make_mgr_with_data()
        # T1 AB should have two successors (ABa, ABp)
        ab = mgr.nuclei_record[1][0]
        assert ab.successor1 == 1  # ABa at T2
        assert ab.successor2 == 2  # ABp at T2

    def test_last_timepoint_has_no_successors(self):
        mgr = _make_mgr_with_data()
        # T2 cells should have no successors (last timepoint not processed)
        for nuc in mgr.nuclei_record[2]:
            assert nuc.successor1 == NILLI
            assert nuc.successor2 == NILLI


class TestFindClosestNucleus:
    """Test spatial nucleus lookup."""

    def test_find_closest_2d(self):
        mgr = _make_mgr_with_data()
        # P0 is at (300, 250)
        closest = mgr.find_closest_nucleus_2d(305, 255, 1)
        assert closest is not None
        assert closest.identity == "P0"

    def test_find_closest_among_multiple(self):
        mgr = _make_mgr_with_data()
        # At T2: ABa at 260, ABp at 300, P1 at 340
        closest = mgr.find_closest_nucleus_2d(265, 250, 3)
        assert closest is not None
        assert closest.identity == "ABa"

        closest = mgr.find_closest_nucleus_2d(335, 250, 3)
        assert closest is not None
        assert closest.identity == "P1"

    def test_find_closest_3d(self):
        mgr = _make_mgr_with_data()
        closest = mgr.find_closest_nucleus(300, 250, 15.0, 1)
        assert closest is not None
        assert closest.identity == "P0"

    def test_find_closest_empty_timepoint(self):
        mgr = NucleiManager()
        mgr.nuclei_record = [[]]
        assert mgr.find_closest_nucleus_2d(100, 100, 1) is None


class TestComputeRedWeights:
    """Test red weight computation."""

    def test_compute_blot_correction(self):
        mgr = NucleiManager()
        mgr._expr_corr = "blot"
        nuc = _nuc(1, rwraw=120)
        mgr.nuclei_record = [[nuc]]
        mgr.compute_red_weights()

        # blot = rwraw - rwcorr3 = 120 - 12 = 108
        assert nuc.rweight == 108

    def test_skip_zero_rwraw(self):
        mgr = NucleiManager()
        mgr._expr_corr = "global"
        nuc = _nuc(1, rwraw=0)
        mgr.nuclei_record = [[nuc]]
        mgr.compute_red_weights()
        assert nuc.rweight == 0

    def test_no_correction(self):
        mgr = NucleiManager()
        mgr._expr_corr = "none"
        nuc = _nuc(1, rwraw=120)
        mgr.nuclei_record = [[nuc]]
        mgr.compute_red_weights()
        # rweight should be unchanged (default 0)
        assert nuc.rweight == 0


class TestNucleusDiameter:
    """Test projected nucleus diameter calculation."""

    def test_at_same_plane(self):
        mgr = NucleiManager()
        nuc = _nuc(1, z=15.0)
        nuc.size = 20  # diameter = 20, radius = 10

        # At the same plane, diameter should be full size
        diam = mgr.nucleus_diameter(nuc, 15)
        assert diam == pytest.approx(20.0, abs=0.1)

    def test_at_distant_plane(self):
        mgr = NucleiManager()
        nuc = _nuc(1, z=15.0)
        nuc.size = 20

        # Far away -> diameter should be 0
        diam = mgr.nucleus_diameter(nuc, 100)
        assert diam == 0.0

    def test_has_circle(self):
        mgr = NucleiManager()
        nuc = _nuc(1, z=15.0)
        nuc.size = 20

        assert mgr.has_circle(nuc, 15)
        assert not mgr.has_circle(nuc, 100)


class TestSetAllSuccessorsAliveFirst:
    """Test that alive nuclei get priority over dead nuclei for successor slots."""

    def test_dead_cell_cannot_steal_successor_slot(self):
        """Dead nuclei should NOT take a successor slot from alive daughters."""
        mgr = NucleiManager()
        # T0: parent cell that divides
        parent = _nuc(1, x=300, identity="AB", pred=NILLI)
        # T1: two alive daughters + one dead cell that also claims same parent
        alive1 = _nuc(1, x=280, identity="ABa", status=1, pred=1)
        alive2 = _nuc(2, x=320, identity="ABp", status=1, pred=1)
        dead = _nuc(3, x=300, status=0, pred=1)  # dead cell with stale pred link

        mgr.nuclei_record = [[parent], [alive1, alive2, dead]]
        mgr.set_all_successors()

        # Parent should have the two alive daughters as successors, not the dead one
        assert parent.successor1 in (1, 2)  # 1-based
        assert parent.successor2 in (1, 2)  # 1-based
        assert parent.successor1 != parent.successor2
        # The dead cell should NOT have taken a successor slot
        assert 3 not in (parent.successor1, parent.successor2)

    def test_dead_cell_gets_slot_if_room(self):
        """Dead cells should still get successor slots if parent has room."""
        mgr = NucleiManager()
        parent = _nuc(1, x=300, pred=NILLI)
        alive1 = _nuc(1, x=280, status=1, pred=1)
        dead = _nuc(2, x=320, status=0, pred=1)

        mgr.nuclei_record = [[parent], [alive1, dead]]
        mgr.set_all_successors()

        # Parent should have alive as successor1, dead as successor2
        assert parent.successor1 == 1  # alive first
        assert parent.successor2 == 2  # dead second

    def test_three_alive_drops_third(self):
        """If 3 alive cells claim same parent, third is dropped with warning."""
        mgr = NucleiManager()
        parent = _nuc(1, x=300, pred=NILLI)
        a1 = _nuc(1, x=280, status=1, pred=1)
        a2 = _nuc(2, x=320, status=1, pred=1)
        a3 = _nuc(3, x=300, status=1, pred=1)

        mgr.nuclei_record = [[parent], [a1, a2, a3]]
        mgr.set_all_successors()

        # Only first two should get slots
        assert parent.successor1 == 1
        assert parent.successor2 == 2


class TestLoadAndProcess:
    """Test loading from ZIP and processing."""

    def test_load_from_zip(self, sample_nuclei_zip: Path):
        mgr = NucleiManager()
        mgr.load(sample_nuclei_zip)

        assert mgr.num_timepoints == 3
        assert len(mgr.nuclei_at(1)) == 1
        assert len(mgr.nuclei_at(2)) == 2
        assert len(mgr.nuclei_at(3)) == 3

    def test_successors_set_after_load(self, sample_nuclei_zip: Path):
        mgr = NucleiManager()
        mgr.load(sample_nuclei_zip)

        # P0 at T1 should have successors set
        p0 = mgr.nuclei_at(1)[0]
        assert p0.successor1 != NILLI or p0.successor2 != NILLI

    def test_process_builds_tree(self, sample_nuclei_zip: Path):
        mgr = NucleiManager()
        mgr.load(sample_nuclei_zip)
        mgr.process(do_identity=False)

        assert mgr.lineage_tree is not None
        assert mgr.lineage_tree.num_cells > 0

    def test_get_cell_after_process(self, sample_nuclei_zip: Path):
        mgr = NucleiManager()
        mgr.load(sample_nuclei_zip)
        mgr.process(do_identity=False)

        # P0 should be in the tree (as a dummy or real cell)
        p0 = mgr.get_cell("P0")
        assert p0 is not None

    def test_save_and_reload(self, sample_nuclei_zip: Path, tmp_path: Path):
        """Load, save, reload — data should survive."""
        mgr = NucleiManager()
        mgr.load(sample_nuclei_zip)

        output = tmp_path / "saved.zip"
        mgr.save(output)

        mgr2 = NucleiManager()
        mgr2.load(output)

        assert mgr2.num_timepoints == mgr.num_timepoints
        for t in range(1, mgr.num_timepoints + 1):
            assert len(mgr2.nuclei_at(t)) == len(mgr.nuclei_at(t))
