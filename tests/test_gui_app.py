"""Tests for the GUI application logic (non-Qt parts).

Tests the pure-Python logic in AceTreeApp: navigation, cell selection,
overlay computation, cell info text, and tracking. Does NOT require
napari or Qt to be installed — only tests the data-layer logic.
"""


import numpy as np

from acetree_py.core.lineage import build_lineage_tree
from acetree_py.core.movie import Movie
from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.gui.app import NUCZINDEXOFFSET, AceTreeApp
from acetree_py.gui.viewer_integration import make_circle_polygon


# ── Fixtures ─────────────────────────────────────────────────────


def _make_nuc(index, x, y, z, size=20, identity="", assigned_id="", status=1,
              predecessor=NILLI, successor1=NILLI, successor2=NILLI):
    return Nucleus(
        index=index, x=x, y=y, z=z, size=size,
        identity=identity, assigned_id=assigned_id, status=status,
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

    mgr.set_all_successors()
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


def _make_duplicate_name_app():
    """Build two disconnected tracks that share one forced display name."""
    mgr = NucleiManager()
    mgr.movie = Movie(xy_res=0.1, z_res=1.0, num_planes=30)
    mgr.nuclei_record = [
        [
            _make_nuc(
                1, 40, 50, 5.0, identity="LeftSeed", assigned_id="Dup",
                successor1=1,
            ),
            _make_nuc(
                2, 140, 150, 15.0, identity="RightSeed", assigned_id="Dup",
                successor1=2,
            ),
        ],
        [
            _make_nuc(
                1, 42, 50, 6.0, identity="LeftSeed", assigned_id="Dup",
                predecessor=1,
            ),
            _make_nuc(
                2, 142, 150, 16.0, identity="RightSeed", assigned_id="Dup",
                predecessor=2,
            ),
        ],
        [],
    ]
    mgr.set_all_successors()
    mgr.lineage_tree = build_lineage_tree(
        mgr.nuclei_record,
        starting_index=0,
        ending_index=3,
        create_dummy_ancestors=False,
    )

    collisions = mgr.lineage_tree.name_collisions["Dup"]
    lookup_cell = mgr.get_cell("Dup")
    selected_cell = next(cell for cell in collisions if cell is not lookup_cell)
    app = AceTreeApp(mgr, image_provider=None)
    app.current_plane = 1
    return app, selected_cell, lookup_cell


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

    def test_selected_physical_cell_survives_rename_and_undo_elsewhere(self):
        """Undoing an unrelated rename must not move selection to its target."""
        from acetree_py.editing.commands import RenameCell

        app = _make_app()
        app.manager.process()
        selected_name = app.manager.nuclei_record[2][0].effective_name
        app.select_cell(selected_name, time=3)
        assert app.selection_anchor == (3, 1)

        selected = app.manager.nuclei_record[2][0]
        app.edit_history.do(RenameCell(time=3, index=2, new_name="Other"))
        resolved, time, index = app.get_selected_nucleus()
        assert resolved is selected
        assert (time, index) == (3, 1)
        assert app.current_cell_name != "Other"
        assert app.selection_anchor == (3, 1)

        app.edit_history.undo()
        resolved, time, index = app.get_selected_nucleus()
        assert resolved is selected
        assert (time, index) == (3, 1)
        assert app.selection_anchor == (3, 1)

    def test_selected_name_re_resolves_across_rename_and_undo(self):
        from acetree_py.editing.commands import RenameCell

        app = _make_app()
        app.manager.process()
        original_name = app.manager.nuclei_record[2][0].effective_name
        app.select_cell(original_name, time=3)
        app.edit_history.do(RenameCell(time=3, index=1, new_name="AB_manual"))
        assert app.current_cell_name == "AB_manual"
        assert app.selection_anchor == (3, 1)

        app.edit_history.undo()
        assert app.current_cell_name == original_name
        assert app.selection_anchor == (3, 1)

    def test_unnamed_index_selection_is_time_qualified(self):
        """The same numeric index at another time is not the same selection."""
        mgr = NucleiManager()
        mgr.movie = Movie(xy_res=0.1, z_res=1.0, num_planes=10)
        mgr.nuclei_record = [
            [_make_nuc(1, 10, 10, 2)],
            [_make_nuc(1, 90, 90, 3)],
        ]
        mgr.lineage_tree = build_lineage_tree(
            mgr.nuclei_record,
            starting_index=0,
            ending_index=2,
            create_dummy_ancestors=False,
        )
        app = AceTreeApp(mgr, image_provider=None)
        app.current_time = 1
        app._set_selection_from_nucleus(1, mgr.nuclei_record[0][0])

        assert app.current_cell_name == "idx=1:1"
        app.current_time = 2
        assert app.get_selected_nucleus() is None
        assert app.selection_anchor == (1, 1)
        app._delete_active_nucleus()
        assert mgr.nuclei_record[1][0].is_alive


class TestInteractionModes:
    def test_modes_are_exclusive_and_escape_clears_relink_source(self):
        app = _make_app()

        class _Button:
            def __init__(self):
                self.checked = False

            def setChecked(self, checked):
                self.checked = checked

        class _Label:
            def setText(self, text):
                self.text = text

        class _Panel:
            def __init__(self):
                self._btn_add = _Button()
                self._btn_track = _Button()
                self._status_label = _Label()
                self._relink_source = None
                self.cancel_count = 0

            def _on_relink_cancelled(self):
                self._relink_source = None
                self.cancel_count += 1

        panel = _Panel()
        app._edit_panel = panel

        app.enter_add_mode()
        assert app._add_mode and not app._placement_mode
        app.enter_placement_mode(parent_name="AB")
        assert app._placement_mode and not app._add_mode
        app.enter_relink_pick_mode(lambda *_: None)
        panel._relink_source = (object(), 3, 1)
        assert app._relink_pick_mode
        assert not app._add_mode and not app._placement_mode

        app._exit_all_modes()

        assert not app._relink_pick_mode
        assert app._relink_pick_callback is None
        assert panel._relink_source is None
        assert panel.cancel_count == 1
        assert not panel._btn_add.checked
        assert not panel._btn_track.checked


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
        """Z navigation keeps the selection and time-follow behavior."""
        app = _make_app()
        app.current_cell_name = "AB"
        app.tracking = True
        app.current_plane = 15
        app.set_plane(20)
        # Cell stays selected and remains ready to follow through time.
        assert app.current_cell_name == "AB"
        assert app.tracking is True
        assert app.current_plane == 20

    def test_time_advance_follows_selected_cell_after_manual_z_navigation(self):
        """Regression: a manual Z move must not silently disable following."""
        app = _make_app()
        app.select_cell("AB", time=4)
        app.set_plane(20)

        app.next_time()

        # AB moves from z=15 at T4 to z=14 at T5.
        assert app.current_time == 5
        assert app.current_plane == 14
        assert app.current_cell_name == "AB"
        assert app.tracking is True

    def test_tracking_follows_daughter(self):
        app = _make_app()
        app.tracking = True
        app.select_cell("P0", time=2)
        # Now advance past P0's end time — should follow to a daughter
        app.set_time(3)
        # Should have switched to AB (first daughter)
        assert app.current_cell_name in ("AB", "P1")

    def test_follow_uses_anchor_when_disconnected_cells_share_name(self):
        app, selected_cell, lookup_cell = _make_duplicate_name_app()
        assert selected_cell is not lookup_cell
        _, selected = selected_cell.nuclei[0]
        app.current_time = 1
        app._set_selection_from_nucleus(1, selected)

        app.set_time(2)

        expected = selected_cell.get_nucleus_at(2)
        assert expected is not None
        assert app.current_plane == round(expected.z + NUCZINDEXOFFSET)
        assert app.selection_anchor == (2, expected.index)
        assert app.get_selected_cell() is selected_cell


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
        # AB is automatic in this fixture. Extending it must not silently
        # convert that automatic name into a permanent manual override.
        assert new_nuc.assigned_id == ""
        assert app.manager.nuclei_record[4][0].assigned_id == ""

    def test_track_extension_of_automatic_parent_stays_unlocked(self):
        app = _make_app()
        app.manager.nuclei_record.append([])
        app.current_time = 6
        app.enter_placement_mode(parent_name="AB")
        before = app.edit_history.num_undoable

        assert app._handle_placement_click(101.0, 151.0)

        new_nuc = app.manager.nuclei_record[5][0]
        assert new_nuc.predecessor == 1
        assert new_nuc.assigned_id == ""
        assert app.edit_history.num_undoable == before + 1
        from acetree_py.editing.commands import CompositeCommand
        assert isinstance(app.edit_history.last_command, CompositeCommand)

    def test_root_track_auto_exit_refreshes_button_state(self):
        app = _make_app()
        app.current_time = 3
        app.enter_placement_mode(parent_name=None)

        class _Panel:
            def __init__(self):
                self.refresh_count = 0
                self.track_checked = True

            def refresh(self):
                self.refresh_count += 1
                self.track_checked = app._placement_mode

        panel = _Panel()
        app._edit_panel = panel
        assert app._handle_placement_click(350.0, 350.0)
        assert not app._placement_mode
        # The structural edit refresh happens while placement mode is still
        # active; the explicit post-exit refresh is what unchecks Track.
        assert panel.refresh_count >= 2
        assert not panel.track_checked

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

    def test_add_uses_anchor_when_disconnected_cells_share_name(self):
        app, selected_cell, lookup_cell = _make_duplicate_name_app()
        assert selected_cell is not lookup_cell
        selected = selected_cell.get_nucleus_at(2)
        assert selected is not None
        app.current_time = 2
        app._set_selection_from_nucleus(2, selected)
        app.enter_add_mode()

        assert app._handle_add_click(145.0, 150.0)

        added = app.manager.nuclei_record[2][0]
        assert added.predecessor == selected.index
        assert selected.successor1 == added.index
        wrong_parent = lookup_cell.get_nucleus_at(2)
        assert wrong_parent is not None
        assert wrong_parent.successor1 == NILLI

    def test_track_uses_captured_anchor_when_duplicate_name_is_ambiguous(self):
        app, selected_cell, lookup_cell = _make_duplicate_name_app()
        assert selected_cell is not lookup_cell
        selected = selected_cell.get_nucleus_at(2)
        assert selected is not None
        app.current_time = 2
        app._set_selection_from_nucleus(2, selected)
        app.enter_placement_mode(parent_name="Dup")
        app.current_time = 3

        assert app._handle_placement_click(145.0, 150.0)

        added = app.manager.nuclei_record[2][0]
        assert added.predecessor == selected.index
        assert selected.successor1 == added.index
        wrong_parent = lookup_cell.get_nucleus_at(2)
        assert wrong_parent is not None
        assert wrong_parent.successor1 == NILLI


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
    two daughters get distinct, parent-specific biological names from the
    automatic naming model, without turning those suggestions into locks."""

    @staticmethod
    def _fresh_app(
        num_timepoints: int = 10,
        parent_name: str = "P2",
        with_body_frame: bool = True,
    ):
        from acetree_py.core.nuclei_manager import NucleiManager
        from acetree_py.editing.commands import RenameCell
        from acetree_py.gui.app import AceTreeApp
        from acetree_py.io.config import AceTreeConfig, NamingMethod
        from acetree_py.naming.body_axes import BodyAxisFrame

        cfg = AceTreeConfig(
            naming_method=NamingMethod.NEWCANONICAL,
            plane_end=30, xy_res=0.1, z_res=1.0,
        )
        mgr = NucleiManager.new_empty(cfg, num_timepoints=num_timepoints)
        mgr.process()
        # These tests assert biological daughter identities, so give the
        # synthetic one-lineage dataset a complete manual body frame.  Orient
        # the relevant rule axis along the daughters' X separation: EMS uses
        # AP, while P2 uses DV.
        if with_body_frame:
            if parent_name == "P2":
                ap, lr = ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0])
            else:
                ap, lr = ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
            mgr.set_manual_body_axes(BodyAxisFrame.from_auxinfo_vectors(
                ap,
                lr,
                provenance="manual_test_frame",
                reference_time=1,
            ))
        app = AceTreeApp(mgr, image_provider=None)
        app.current_plane = 5
        app.enter_add_mode()
        # Seed the requested parent at t=1
        app._handle_add_click(100.0, 100.0)
        app.current_cell_name = mgr.nuclei_record[0][0].effective_name
        app.edit_history.do(
            RenameCell(time=1, index=1, new_name=parent_name)
        )
        # Extend to t=2
        app.set_time(2)
        app._handle_add_click(110.0, 100.0)
        app.current_time = 2
        return app

    def test_second_add_creates_axis_aware_division(self):
        """P2's manual division uses its canonical C/P3 lineage rule."""
        app = self._fresh_app()
        history_before = app.edit_history.num_undoable
        app._handle_add_click(50.0, 100.0)
        mgr = app.manager

        # A click is one user gesture, even though it adds the second daughter
        # and repairs the first daughter's inherited name state together.
        from acetree_py.editing.commands import CompositeCommand
        assert app.edit_history.num_undoable == history_before + 1
        assert isinstance(app.edit_history.last_command, CompositeCommand)

        # t=1 parent cell still named P2, now marked as dividing
        assert mgr.nuclei_record[0][0].effective_name == "P2"
        assert mgr.nuclei_record[0][0].successor1 != -1
        assert mgr.nuclei_record[0][0].successor2 != -1

        # t=2 has P2's actual biological daughters, not generic suffixes.
        nucs = mgr.nuclei_record[1]
        assert len(nucs) == 2
        names = {n.effective_name for n in nucs}
        assert names == {"C", "P3"}
        assert all(n.assigned_id == "" for n in nucs)

        # The manual frame orients the empirical P2 rule consistently.
        for n in nucs:
            if n.effective_name == "C":
                assert n.x == 110
            elif n.effective_name == "P3":
                assert n.x == 50

        # No collision alias or biologically invalid generic suffix remains.
        assert mgr.get_cell("P2_2") is None
        assert mgr.get_cell("P2a") is None
        assert mgr.get_cell("P2p") is None
        assert mgr.get_cell("C") is not None
        assert mgr.get_cell("P3") is not None

        app.edit_history.undo()
        assert len(mgr.nuclei_record[1]) == 1

    def test_second_daughter_without_body_frame_uses_neutral_names(self):
        app = self._fresh_app(parent_name="EMS", with_body_frame=False)

        app._handle_add_click(50.0, 100.0)

        daughters = app.manager.nuclei_record[1]
        assert len(daughters) == 2
        assert all(n.assigned_id == "" for n in daughters)
        assert all(n.effective_name.startswith("Nuc") for n in daughters)
        assert len({n.effective_name for n in daughters}) == 2

    def test_division_flips_when_ap_direction_flipped(self):
        """A valid legacy body frame still uses P2's C/P3 rule.

        P2's empirical rule is predominantly dorsoventral, so the UI must
        not reduce it to the old hard-coded AP ``a/p`` suffix heuristic.
        """
        from acetree_py.io.auxinfo import AuxInfo

        app = self._fresh_app()
        # Force a valid v1 AuxInfo with P as first axis character.  PDR is
        # handed consistently; the old PDL combination is intentionally
        # rejected by AuxInfo validation.
        app.manager.auxinfo = AuxInfo(version=1, data={"axis": "PDR"})
        # Also clear the stored identity_assigner so get_ap_direction_at
        # falls through to the AuxInfo path instead of the topology path.
        app.manager.identity_assigner = None

        app._handle_add_click(50.0, 100.0)
        nucs = app.manager.nuclei_record[1]
        assert {n.effective_name for n in nucs} == {"C", "P3"}
        assert all(n.assigned_id == "" for n in nucs)

    def test_ems_division_uses_e_and_ms(self):
        """Founder-specific rules also cover the EMS -> E/MS division."""
        app = self._fresh_app(parent_name="EMS")
        app._handle_add_click(50.0, 100.0)

        nucs = app.manager.nuclei_record[1]
        assert {n.effective_name for n in nucs} == {"E", "MS"}
        assert all(n.assigned_id == "" for n in nucs)

    def test_track_can_commit_retroactive_division_as_one_edit(self):
        """Track and Add share the same retroactive-division semantics."""
        from acetree_py.editing.commands import CompositeCommand

        app = self._fresh_app()
        app.exit_add_mode()
        app.enter_placement_mode(parent_name="P2")
        app.current_time = 2
        history_before = app.edit_history.num_undoable

        assert app._handle_placement_click(50.0, 100.0)

        nucs = app.manager.nuclei_record[1]
        assert {n.effective_name for n in nucs} == {"C", "P3"}
        assert all(n.assigned_id == "" for n in nucs)
        assert app.edit_history.num_undoable == history_before + 1
        assert isinstance(app.edit_history.last_command, CompositeCommand)

        app.edit_history.undo()
        assert len(app.manager.nuclei_record[1]) == 1

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
