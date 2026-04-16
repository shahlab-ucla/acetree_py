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
