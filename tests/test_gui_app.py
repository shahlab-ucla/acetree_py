"""Tests for the GUI application logic (non-Qt parts).

Tests the pure-Python logic in AceTreeApp: navigation, cell selection,
overlay computation, cell info text, and tracking. Does NOT require
napari or Qt to be installed — only tests the data-layer logic.
"""

import math

import numpy as np
import pytest

from acetree_py.core.cell import Cell, CellFate
from acetree_py.core.lineage import LineageTree, build_lineage_tree
from acetree_py.core.movie import Movie
from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.gui.app import NUCZINDEXOFFSET, AceTreeApp
from acetree_py.gui.viewer_integration import make_circle_polygon


# ── Fixtures ─────────────────────────────────────────────────────


def _make_nuc(index, x, y, z, size=20, identity="", status=1,
              predecessor=NILLI, successor1=NILLI, successor2=NILLI):
    return Nucleus(
        index=index, x=x, y=y, z=z, size=size,
        identity=identity, status=status,
        predecessor=predecessor, successor1=successor1, successor2=successor2,
    )


def _build_test_manager():
    """Build a NucleiManager with synthetic data (no file I/O).

    Creates 5 timepoints with P0 dividing into AB and P1:
      T1: P0 at (150, 150, 15)
      T2: P0 at (150, 150, 15)
      T3: AB at (100, 150, 15), P1 at (200, 150, 15) — division
      T4: AB at (100, 150, 15), P1 at (200, 150, 15)
      T5: AB at (100, 150, 15), P1 at (200, 150, 15)
    """
    mgr = NucleiManager()
    mgr.movie = Movie(xy_res=0.1, z_res=1.0, num_planes=30)

    mgr.nuclei_record = [
        [  # T1
            _make_nuc(1, 150, 150, 15.0, identity="P0", successor1=1),
        ],
        [  # T2
            _make_nuc(1, 150, 150, 15.0, identity="P0", predecessor=1,
                     successor1=1, successor2=2),
        ],
        [  # T3 — division
            _make_nuc(1, 100, 150, 15.0, identity="AB", predecessor=1),
            _make_nuc(2, 200, 150, 15.0, identity="P1", predecessor=1),
        ],
        [  # T4
            _make_nuc(1, 100, 150, 15.0, identity="AB", predecessor=1),
            _make_nuc(2, 200, 150, 15.0, identity="P1", predecessor=2),
        ],
        [  # T5
            _make_nuc(1, 100, 150, 14.0, identity="AB", predecessor=1),
            _make_nuc(2, 200, 150, 16.0, identity="P1", predecessor=2),
        ],
    ]

    # Build lineage tree manually
    mgr.lineage_tree = build_lineage_tree(
        mgr.nuclei_record,
        starting_index=0,
        ending_index=5,
        create_dummy_ancestors=False,
    )

    return mgr


def _make_app():
    """Create an AceTreeApp with test data (no viewer)."""
    mgr = _build_test_manager()
    app = AceTreeApp(mgr, image_provider=None)
    app.current_time = 1
    app.current_plane = 15
    return app


# ── Navigation tests ─────────────────────────────────────────────


class TestNavigation:
    def test_set_time(self):
        app = _make_app()
        app.set_time(3)
        assert app.current_time == 3

    def test_set_time_clamps(self):
        app = _make_app()
        app.set_time(0)
        assert app.current_time == 1
        app.set_time(999)
        assert app.current_time == 5

    def test_next_prev_time(self):
        app = _make_app()
        app.set_time(3)
        app.next_time()
        assert app.current_time == 4
        app.prev_time()
        assert app.current_time == 3

    def test_set_plane(self):
        app = _make_app()
        app.set_plane(20)
        assert app.current_plane == 20

    def test_set_plane_clamps(self):
        app = _make_app()
        app.set_plane(0)
        assert app.current_plane == 1

    def test_next_prev_plane(self):
        app = _make_app()
        app.current_plane = 15
        app.next_plane()
        assert app.current_plane == 16
        app.prev_plane()
        assert app.current_plane == 15


# ── Cell selection tests ─────────────────────────────────────────


class TestCellSelection:
    def test_select_cell_by_name(self):
        app = _make_app()
        app.select_cell("P0")
        assert app.current_cell_name == "P0"

    def test_select_cell_jumps_to_start(self):
        app = _make_app()
        app.current_time = 5
        app.select_cell("P0")
        # P0 ends at T2 (before division at T3), so should jump to P0's range
        assert app.current_time <= 2

    def test_select_cell_with_time(self):
        app = _make_app()
        app.select_cell("AB", time=4)
        assert app.current_cell_name == "AB"
        assert app.current_time == 4

    def test_select_nonexistent_cell(self):
        app = _make_app()
        app.select_cell("NONEXISTENT")
        assert app.current_cell_name == ""  # Unchanged (was empty)

    def test_select_cell_at_position(self):
        app = _make_app()
        app.current_time = 3
        # AB is at (100, 150), P1 at (200, 150)
        # Click near AB
        app.select_cell_at_position(105, 155)
        assert app.current_cell_name == "AB"

    def test_select_cell_at_position_other(self):
        app = _make_app()
        app.current_time = 3
        # Click near P1
        app.select_cell_at_position(195, 145)
        assert app.current_cell_name == "P1"


# ── Tracking tests ───────────────────────────────────────────────


class TestTracking:
    def test_tracking_follows_z(self):
        app = _make_app()
        app.tracking = True
        app.select_cell("AB")
        app.set_time(5)
        # AB at T5 is at z=14.0, so plane should be 14 (+ 0 offset).
        assert app.current_plane == round(14.0 + NUCZINDEXOFFSET)

    def test_tracking_lands_on_centroid_not_off_by_one(self):
        """Regression: NUCZINDEXOFFSET used to be 1, which shifted the
        displayed plane one above the nucleus centroid.  With the offset
        fixed to 0, the slice must land exactly on the integer-rounded
        nucleus z value."""
        assert NUCZINDEXOFFSET == 0
        app = _make_app()
        app.tracking = True
        app.select_cell("AB")
        # AB at T5 lives at z=14.0 — the display must snap to plane 14.
        app.set_time(5)
        assert app.current_plane == 14
        # P1 at T5 lives at z=16.0 — the display must snap to plane 16.
        app.select_cell("P1")
        app.set_time(5)
        assert app.current_plane == 16

    def test_tracking_off_preserves_plane(self):
        app = _make_app()
        app.tracking = False
        app.current_plane = 20
        app.current_cell_name = "AB"
        app.set_time(5)
        assert app.current_plane == 20  # Not changed

    def test_z_nav_preserves_cell_selection(self):
        """Regression: set_plane used to clear current_cell_name and
        set tracking=False.  This broke Add/Track modes where the user
        wants to nudge the Z slice to position the new nucleus while
        still inheriting the selected cell as predecessor.  After the
        fix, Z nav only disables auto-tracking — the cell stays
        selected so the Add path still sees a parent."""
        app = _make_app()
        app.current_cell_name = "AB"
        app.tracking = True
        app.current_plane = 15
        app.set_plane(20)
        # Cell stays selected, tracking is frozen
        assert app.current_cell_name == "AB"
        assert app.tracking is False
        assert app.current_plane == 20

    def test_tracking_follows_daughter(self):
        app = _make_app()
        app.tracking = True
        app.select_cell("P0", time=2)
        # Now advance past P0's end time — should follow to a daughter
        app.set_time(3)
        # Should have switched to AB (first daughter)
        assert app.current_cell_name in ("AB", "P1")


# ── Add-mode auto-advance tests ──────────────────────────────────


class TestAddModeAutoAdvance:
    """_handle_add_click should extend the selected cell forward rather
    than silently drop the parent link when the user clicks at the same
    timepoint as the parent's end_time."""

    def test_add_at_parent_end_time_auto_advances(self):
        """User selects AB (ends at t=5) and clicks Add while viewing t=5.
        The new nucleus should be placed at t=6 with AB as predecessor."""
        app = _make_app()
        app.enter_add_mode()
        app.select_cell("AB")  # AB ends at t=5
        app.current_time = 5   # same as parent end_time
        # Extend the record to have a t=6 slot so num_timepoints allows it.
        app.manager.nuclei_record.append([])
        # Click at some (x, y).
        app._handle_add_click(100.0, 150.0)

        # current_time advanced to 6
        assert app.current_time == 6
        # New nucleus landed at t=6
        assert len(app.manager.nuclei_record[5]) == 1
        new_nuc = app.manager.nuclei_record[5][0]
        # Linked to AB's nucleus at t=5 (idx 1)
        assert new_nuc.predecessor == 1
        # And inherited AB's name via assigned_id so the naming pipeline
        # treats the chain as one cell.
        assert new_nuc.assigned_id == "AB"

    def test_add_without_selection_is_root(self):
        """No cell selected → new nucleus is a root, no predecessor,
        no auto-advance."""
        app = _make_app()
        app.enter_add_mode()
        app.current_cell_name = ""
        app.current_time = 3
        t_before = app.current_time
        before_len = len(app.manager.nuclei_record[2])
        app._handle_add_click(300.0, 300.0)
        assert app.current_time == t_before
        assert len(app.manager.nuclei_record[2]) == before_len + 1
        new_nuc = app.manager.nuclei_record[2][-1]
        assert new_nuc.predecessor == -1  # NILLI
        assert new_nuc.assigned_id == ""


# ── Phantom-cell scaffold handling ────────────────────────────────


class TestPhantomAncestorAvoidance:
    """lineage.build_lineage_tree attaches canonical phantom children
    (ABa, ABp, EMS, P2, …) to any cell whose name matches the standard
    lineage scaffold.  These phantoms have no nuclei and must not be
    followed by time-tracking or treated as parents by Add clicks —
    otherwise a user who renames their first cell to "AB" sees their
    subsequent Add at t=2 produce an orphan (the code would point at
    phantom "ABa" instead of real "AB").
    """

    @staticmethod
    def _fresh_manual_app(num_timepoints: int = 10):
        from acetree_py.core.nuclei_manager import NucleiManager
        from acetree_py.editing.commands import AddNucleus, RenameCell
        from acetree_py.gui.app import AceTreeApp
        from acetree_py.io.config import AceTreeConfig, NamingMethod

        cfg = AceTreeConfig(
            naming_method=NamingMethod.NEWCANONICAL,
            plane_end=30, xy_res=0.1, z_res=1.0,
        )
        mgr = NucleiManager.new_empty(cfg, num_timepoints=num_timepoints)
        mgr.process()
        app = AceTreeApp(mgr, image_provider=None)
        app.current_time = 1
        app.current_plane = 5
        # Add a cell at t=1
        app.enter_add_mode()
        app._handle_add_click(100.0, 100.0)
        app.current_cell_name = mgr.nuclei_record[0][0].effective_name
        # Rename to AB
        app.edit_history.do(RenameCell(time=1, index=1, new_name="AB"))
        return app

    def test_tracking_does_not_follow_phantom_daughter(self):
        """After renaming the single-timepoint cell to "AB", the tree
        contains phantom ABa/ABp children.  Time-stepping to t=2 must
        NOT switch current_cell_name to "ABa"."""
        app = self._fresh_manual_app()
        app.set_time(2)
        assert app.current_cell_name == "AB"

    def test_add_after_rename_and_advance_links_to_real_parent(self):
        """Exactly the user-reported flow: add cell, rename to AB,
        press Right (advance to t=2), click Add.  The new nucleus must
        be linked to AB — not produce a Nuc_... orphan."""
        app = self._fresh_manual_app()
        app.set_time(2)
        app._handle_add_click(102.0, 100.0)

        t2 = app.manager.nuclei_record[1][0]
        assert t2.predecessor == 1
        assert t2.assigned_id == "AB"
        assert t2.effective_name == "AB"

        cell_ab = app.manager.get_cell("AB")
        assert cell_ab is not None
        assert cell_ab.start_time == 1
        assert cell_ab.end_time == 2
        assert len(cell_ab.nuclei) == 2

    def test_add_walks_up_through_phantom_to_real_ancestor(self):
        """Even if tracking or user action left current_cell_name
        pointing at a phantom, _handle_add_click must walk up the parent
        chain to find a real ancestor with nuclei."""
        app = self._fresh_manual_app()
        # Simulate a drift: user's tracker or tree-click lands on phantom ABa
        app.current_cell_name = "ABa"
        app.set_time(2)
        app._handle_add_click(104.0, 100.0)

        t2 = app.manager.nuclei_record[1][0]
        # Should still have linked to real AB at t=1
        assert t2.predecessor == 1
        assert t2.assigned_id == "AB"


# ── Manual division tests ────────────────────────────────────────


class TestManualDivision:
    """Pressing Add a second time on a cell that already has a first
    daughter at the current timepoint creates a manual division.  The
    two daughters get distinct ``+"a"`` / ``+"p"`` suffixes along the
    embryo's AP axis so they don't collide on the parent's forced name
    (and the lineage tree doesn't fall back to the ``_{n}`` alias)."""

    @staticmethod
    def _fresh_app(num_timepoints: int = 10):
        from acetree_py.core.nuclei_manager import NucleiManager
        from acetree_py.editing.commands import RenameCell
        from acetree_py.gui.app import AceTreeApp
        from acetree_py.io.config import AceTreeConfig, NamingMethod

        cfg = AceTreeConfig(
            naming_method=NamingMethod.NEWCANONICAL,
            plane_end=30, xy_res=0.1, z_res=1.0,
        )
        mgr = NucleiManager.new_empty(cfg, num_timepoints=num_timepoints)
        mgr.process()
        app = AceTreeApp(mgr, image_provider=None)
        app.current_plane = 5
        app.enter_add_mode()
        # Seed P2 at t=1
        app._handle_add_click(100.0, 100.0)
        app.current_cell_name = mgr.nuclei_record[0][0].effective_name
        app.edit_history.do(RenameCell(time=1, index=1, new_name="P2"))
        # Extend to t=2
        app.set_time(2)
        app._handle_add_click(110.0, 100.0)
        app.current_time = 2
        return app

    def test_second_add_creates_axis_aware_division(self):
        """With the default AP direction (+X is anterior), clicking Add
        at (x=50, y=100) — far from the existing first daughter at
        (110, 100) — splits the cell into P2 + P2a (anterior) + P2p
        (posterior), not P2 and P2_2."""
        app = self._fresh_app()
        app._handle_add_click(50.0, 100.0)
        mgr = app.manager

        # t=1 parent cell still named P2, now marked as dividing
        assert mgr.nuclei_record[0][0].effective_name == "P2"
        assert mgr.nuclei_record[0][0].successor1 != -1
        assert mgr.nuclei_record[0][0].successor2 != -1

        # t=2 has two daughters with P2a / P2p names
        nucs = mgr.nuclei_record[1]
        assert len(nucs) == 2
        names = {n.effective_name for n in nucs}
        assert names == {"P2a", "P2p"}

        # Larger-X daughter is anterior (default AP = +X), so P2a is at x=110
        for n in nucs:
            if n.effective_name == "P2a":
                assert n.x == 110
            elif n.effective_name == "P2p":
                assert n.x == 50

        # No P2_2 collision cell in the tree
        assert mgr.get_cell("P2_2") is None
        assert mgr.get_cell("P2a") is not None
        assert mgr.get_cell("P2p") is not None

    def test_division_flips_when_ap_direction_flipped(self):
        """If AuxInfo says ``axis[0] = "P"`` (canonical +X is posterior),
        the daughter at larger X gets the ``"p"`` suffix instead."""
        from acetree_py.io.auxinfo import AuxInfo

        app = self._fresh_app()
        # Force a v1 AuxInfo with P as first axis character.
        app.manager.auxinfo = AuxInfo(version=1, data={"axis": "PDL"})
        # Also clear the stored identity_assigner so get_ap_direction_at
        # falls through to the AuxInfo path instead of the topology path.
        app.manager.identity_assigner = None

        app._handle_add_click(50.0, 100.0)
        nucs = app.manager.nuclei_record[1]
        for n in nucs:
            if n.x == 110:
                # Larger X is now posterior under PDL axis
                assert n.effective_name == "P2p"
            elif n.x == 50:
                assert n.effective_name == "P2a"

    def test_click_near_existing_at_end_time_extends(self):
        """At click_time == cell.end_time, a close click preserves the
        extend-past-end workflow: auto-advance to end_time + 1 rather
        than creating a division.  (At click_time < cell.end_time, by
        contrast, any click creates a division — see
        test_mid_life_close_click_creates_division.)"""
        app = self._fresh_app()
        # P2 has nuclei at t=1 and t=2, so end_time=2 and current_time=2.
        # First daughter is at (110, 100), size=20.  Click close to it.
        app._handle_add_click(112.0, 101.0)
        assert app.current_time == 3  # auto-advanced past the extension
        # Only one nucleus still at t=2 (no division happened)
        assert len(app.manager.nuclei_record[1]) == 1

    def test_mid_life_close_click_creates_division(self):
        """At click_time < cell.end_time, clicking close to the existing
        nucleus still creates a division.  The old XY-distance heuristic
        used to route these close clicks into extension-with-auto-
        advance, yanking the user to end_time + 1 — which was wrong
        because the cell continues past click_time, so the only sensible
        interpretation of a new nucleus at click_time is a sibling."""
        app = self._fresh_app()
        # Extend P2 one more frame so click_time=2 is mid-life.
        app.set_time(3)
        app._handle_add_click(115.0, 100.0)
        # Now P2 exists at t=1, t=2, t=3 → end_time=3.
        app.current_cell_name = "P2"
        app.set_time(2)  # click at t=2, strictly mid-life
        # P2's t=2 nucleus is at (110, 100), size=20.  Click CLOSE to it.
        app._handle_add_click(112.0, 101.0)
        # No auto-advance — we stay at t=2.
        assert app.current_time == 2
        # Two nuclei at t=2: the original P2 and the new sibling.
        assert len(app.manager.nuclei_record[1]) == 2


# ── Triple-successor rejection ───────────────────────────────────


class TestTripleSuccessorRejection:
    """Add / Placement viewer clicks must refuse to create a third
    successor.  Previously they by-passed the validator and left a
    floating nucleus when set_all_successors silently dropped the
    third link."""

    @staticmethod
    def _app_with_dividing_parent():
        """Build an app where P2 at t=1 already has two children at
        t=2 (P2a, P2p) and user might click Add again with P2 still
        selected."""
        app = TestManualDivision._fresh_app()
        app._handle_add_click(50.0, 100.0)  # create division
        # Re-select the parent P2 (user clicks on the parent at t=1)
        app.current_cell_name = "P2"
        app.current_time = 1
        return app

    def test_add_click_rejects_third_successor(self):
        app = self._app_with_dividing_parent()
        app.enter_add_mode()
        # Try to add another child at t=2 — P2 already has 2 kids
        app.current_time = 2
        before = [len(ts) for ts in app.manager.nuclei_record]
        app._handle_add_click(200.0, 200.0)
        after = [len(ts) for ts in app.manager.nuclei_record]
        # Nothing added anywhere
        assert before == after

    def test_placement_click_rejects_third_successor(self):
        app = self._app_with_dividing_parent()
        app.exit_add_mode()
        app.enter_placement_mode(parent_name="P2")
        app.current_time = 2
        before = [len(ts) for ts in app.manager.nuclei_record]
        app._handle_placement_click(200.0, 200.0)
        after = [len(ts) for ts in app.manager.nuclei_record]
        assert before == after


# ── Chain-delete tests ───────────────────────────────────────────


class TestChainDelete:
    """After pressing Delete on a cell's nucleus, the view should step
    one timepoint back and keep the cell selected so pressing Delete
    again kills that cell's previous-timepoint nucleus.  This makes it
    easy to chain-delete a tracked cell backward without re-selecting
    between each press."""

    @staticmethod
    def _build_chain_app():
        """Build an app where cell "AB" is a forced-name continuation
        chain across t=1..3.  Using assigned_id ensures the naming
        pipeline's rebuild preserves the cell name across each delete,
        so successive deletes can actually walk the same cell back."""
        from acetree_py.core.lineage import build_lineage_tree
        from acetree_py.core.movie import Movie
        from acetree_py.core.nuclei_manager import NucleiManager
        from acetree_py.core.nucleus import Nucleus
        from acetree_py.gui.app import AceTreeApp

        mgr = NucleiManager()
        mgr.movie = Movie(xy_res=0.1, z_res=1.0, num_planes=30)
        mgr.nuclei_record = [
            [Nucleus(index=1, x=100, y=100, z=5.0, size=10,
                     identity="AB", assigned_id="AB", status=1,
                     predecessor=-1)],
            [Nucleus(index=1, x=100, y=100, z=5.0, size=10,
                     identity="AB", assigned_id="AB", status=1,
                     predecessor=1)],
            [Nucleus(index=1, x=100, y=100, z=5.0, size=10,
                     identity="AB", assigned_id="AB", status=1,
                     predecessor=1)],
        ]
        mgr.set_all_successors()
        mgr.lineage_tree = build_lineage_tree(
            mgr.nuclei_record, starting_index=0, ending_index=3,
            create_dummy_ancestors=False,
        )
        app = AceTreeApp(mgr, image_provider=None)
        return app

    def test_delete_steps_back_and_keeps_cell_selected(self):
        """Select AB (assigned_id), jump to T3, Delete → current_time
        drops to T2, AB remains selected."""
        app = self._build_chain_app()
        app.select_cell("AB")
        app.current_time = 3
        app._delete_active_nucleus()
        assert app.current_time == 2
        assert app.current_cell_name == "AB"
        assert app.manager.nuclei_record[2][0].status < 0

    def test_delete_chain_walks_backward(self):
        """Repeatedly pressing Delete walks the cell backward, one
        timepoint per press, while the cell keeps its forced name."""
        app = self._build_chain_app()
        app.select_cell("AB")
        app.current_time = 3
        # Delete T3
        app._delete_active_nucleus()
        assert app.current_time == 2
        assert app.manager.nuclei_record[2][0].status < 0
        assert app.current_cell_name == "AB"
        # Delete T2
        app._delete_active_nucleus()
        assert app.current_time == 1
        assert app.manager.nuclei_record[1][0].status < 0
        assert app.current_cell_name == "AB"

    def test_delete_last_nucleus_clears_selection(self):
        """If a Delete removes the cell's only remaining nucleus, the
        cell vanishes and current_cell_name is cleared so subsequent
        navigation doesn't chase a dead reference."""
        from acetree_py.core.nuclei_manager import NucleiManager
        from acetree_py.core.nucleus import Nucleus
        from acetree_py.core.movie import Movie
        from acetree_py.core.lineage import build_lineage_tree
        from acetree_py.gui.app import AceTreeApp

        mgr = NucleiManager()
        mgr.movie = Movie(xy_res=0.1, z_res=1.0, num_planes=30)
        mgr.nuclei_record = [
            [Nucleus(index=1, x=100, y=100, z=5.0, size=10,
                     identity="Solo", status=1)],
            [],
        ]
        mgr.lineage_tree = build_lineage_tree(
            mgr.nuclei_record, starting_index=0, ending_index=2,
            create_dummy_ancestors=False,
        )
        app = AceTreeApp(mgr, image_provider=None)
        app.current_time = 1
        app.current_cell_name = "Solo"
        app._delete_active_nucleus()
        # Only nucleus gone → cell vanishes → selection cleared
        assert app.current_cell_name == ""

    def test_delete_at_t1_does_not_step_below_one(self):
        """Guard: deleting at t=1 must not set current_time to 0."""
        from acetree_py.core.nuclei_manager import NucleiManager
        from acetree_py.core.nucleus import Nucleus
        from acetree_py.core.movie import Movie
        from acetree_py.core.lineage import build_lineage_tree
        from acetree_py.gui.app import AceTreeApp

        mgr = NucleiManager()
        mgr.movie = Movie(xy_res=0.1, z_res=1.0, num_planes=30)
        # Two nuclei at t=1 so "Solo" exists in lineage_tree before delete
        mgr.nuclei_record = [
            [Nucleus(index=1, x=100, y=100, z=5.0, size=10,
                     identity="A", status=1)],
        ]
        mgr.lineage_tree = build_lineage_tree(
            mgr.nuclei_record, starting_index=0, ending_index=1,
            create_dummy_ancestors=False,
        )
        app = AceTreeApp(mgr, image_provider=None)
        app.current_time = 1
        app.current_cell_name = "A"
        app._delete_active_nucleus()
        assert app.current_time == 1  # stayed at t=1, didn't go to 0


# ── Overlay data tests ───────────────────────────────────────────


class TestOverlayData:
    def test_empty_timepoint(self):
        app = _make_app()
        # No alive nuclei at T0 (time 0 is invalid, clamped to T1)
        # Let's just verify with valid data
        app.current_time = 1
        data = app.get_nucleus_overlay_data()
        assert len(data["centers"]) > 0

    def test_overlay_at_t3(self):
        app = _make_app()
        app.current_time = 3
        app.current_plane = 15  # Same z as nuclei

        data = app.get_nucleus_overlay_data()
        assert len(data["centers"]) == 2
        assert len(data["radii"]) == 2
        assert len(data["names"]) == 2
        assert "AB" in data["names"]
        assert "P1" in data["names"]

    def test_overlay_selected_cell(self):
        app = _make_app()
        app.current_time = 3
        app.current_plane = 15
        app.current_cell_name = "AB"

        data = app.get_nucleus_overlay_data()
        idx = data["selected_idx"]
        assert idx >= 0
        # Selected cell should be white
        assert np.allclose(data["colors"][idx], [1.0, 1.0, 1.0, 1.0])

    def test_overlay_far_plane_filters_nuclei(self):
        app = _make_app()
        app.current_time = 3
        app.current_plane = 1  # Very far from z=15

        data = app.get_nucleus_overlay_data()
        # Nuclei at z=15 with size=20 (radius=10) should not be visible at plane=1
        # because dz = |15 - 1| * z_pix_res = 14 * 10 = 140 >> radius=10
        assert len(data["centers"]) == 0

    def test_overlay_centers_are_yx(self):
        """napari uses (row, col) = (y, x) convention."""
        app = _make_app()
        app.current_time = 1
        app.current_plane = 15

        data = app.get_nucleus_overlay_data()
        # P0 is at (x=150, y=150), so center should be (y=150, x=150)
        assert data["centers"][0][0] == 150  # y/row
        assert data["centers"][0][1] == 150  # x/col


# ── Cell info text tests ──────────────────────────────────────────


class TestCellInfoText:
    def test_no_selection(self):
        app = _make_app()
        text = app.get_cell_info_text()
        assert "No cell selected" in text

    def test_selected_cell_info(self):
        app = _make_app()
        app.current_cell_name = "P0"
        app.current_time = 1
        text = app.get_cell_info_text()

        assert "P0" in text
        assert "Position" in text
        assert "150" in text  # x or y coordinate
        assert "Fate" in text

    def test_cell_not_present(self):
        app = _make_app()
        app.current_cell_name = "AB"
        app.current_time = 1  # AB doesn't exist at T1
        text = app.get_cell_info_text()

        assert "Not present" in text or "not in lineage" in text.lower()

    def test_cell_info_shows_children(self):
        app = _make_app()
        app.current_cell_name = "P0"
        app.current_time = 2  # P0's last time before division
        text = app.get_cell_info_text()

        # P0 should show children AB and P1
        assert "Children" in text or "DIVIDED" in text


# ── Circle polygon utility tests ─────────────────────────────────


class TestCirclePolygon:
    def test_circle_polygon_shape(self):
        poly = make_circle_polygon(100, 200, 10, n_vertices=16)
        assert poly.shape == (16, 2)

    def test_circle_polygon_center(self):
        poly = make_circle_polygon(100, 200, 10, n_vertices=100)
        # Mean of vertices should be close to center
        mean_y = poly[:, 0].mean()
        mean_x = poly[:, 1].mean()
        assert abs(mean_y - 200) < 0.5  # cy=200 (row)
        assert abs(mean_x - 100) < 0.5  # cx=100 (col)

    def test_circle_polygon_radius(self):
        cx, cy, r = 50, 100, 25
        poly = make_circle_polygon(cx, cy, r, n_vertices=100)
        # All vertices should be radius distance from center
        distances = np.sqrt((poly[:, 0] - cy) ** 2 + (poly[:, 1] - cx) ** 2)
        assert np.allclose(distances, r, atol=0.1)

    def test_circle_polygon_zero_radius(self):
        poly = make_circle_polygon(0, 0, 0, n_vertices=8)
        assert np.allclose(poly, 0)
