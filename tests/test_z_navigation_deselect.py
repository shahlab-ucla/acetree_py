"""Z-navigation deselect semantics (Java AceTree parity).

The active cell is bound to the image slice it lives on: selecting a nucleus
snaps ``current_plane`` onto its centroid z.  Java AceTree therefore treats a
user's Z move as *leaving* that cell — once the user scrolls off the nucleus's
slice, the highlighted cell no longer matches what is on screen.

These tests pin both halves of that contract:

* user-initiated Z navigation (Up/Down and W/S keys, the ``z=`` spinbox, and
  the ± plane buttons) clears the selection and stops follow mode;
* programmatic Z navigation (auto-tracking review, tracking-preview centering)
  leaves the selection alone, because it moves the slice *on behalf of* the
  active cell.

Pure data-layer tests — no Qt or napari required.
"""


from acetree_py.gui.app import NUCZINDEXOFFSET

from tests.test_gui_app import _make_app


def _select_ab_at_t4():
    """App with AB selected and following at T4 (AB's centroid is z=15)."""
    app = _make_app()
    app.select_cell("AB", time=4)
    assert app.current_cell_name == "AB"
    assert app.selection_anchor is not None
    assert app.tracking is True
    assert app.current_plane == round(15.0 + NUCZINDEXOFFSET)
    return app


# ── User-initiated navigation deselects ──────────────────────────


def test_user_set_plane_clears_selection_and_keeps_new_plane():
    app = _select_ab_at_t4()

    app.set_plane(20, user_initiated=True)

    assert app.current_cell_name == ""
    assert app.selection_anchor is None
    assert app.tracking is False
    # Deselection must not revert the plane the user asked for.
    assert app.current_plane == 20


def test_user_next_plane_deselects():
    app = _select_ab_at_t4()
    start_plane = app.current_plane

    app.next_plane()

    assert app.current_plane == start_plane + 1
    assert app.current_cell_name == ""
    assert app.selection_anchor is None
    assert app.tracking is False


def test_user_prev_plane_deselects():
    app = _select_ab_at_t4()
    start_plane = app.current_plane

    app.prev_plane()

    assert app.current_plane == start_plane - 1
    assert app.current_cell_name == ""
    assert app.selection_anchor is None
    assert app.tracking is False


def test_user_set_plane_to_same_plane_keeps_selection():
    """The early return fires before any deselect: nothing actually moved."""
    app = _select_ab_at_t4()
    plane = app.current_plane

    app.set_plane(plane, user_initiated=True)

    assert app.current_plane == plane
    assert app.current_cell_name == "AB"
    assert app.selection_anchor is not None
    assert app.tracking is True


def test_user_set_plane_clamped_onto_current_plane_keeps_selection():
    """Clamping onto the plane already shown is a no-op, not a deselect.

    Holding at the top of the stack while mashing ▲ must not silently drop the
    selection on every repeat.
    """
    app = _select_ab_at_t4()
    # Park on the last plane of the stack without going through set_plane so
    # the selection is still live when the clamped request arrives.
    _, plane_end = app._plane_bounds()
    app.current_plane = plane_end

    app.set_plane(plane_end + 5, user_initiated=True)

    assert app.current_plane == plane_end
    assert app.current_cell_name == "AB"
    assert app.selection_anchor is not None
    assert app.tracking is True

    # And the ▲ button at the ceiling behaves the same way.
    app.next_plane()

    assert app.current_plane == plane_end
    assert app.current_cell_name == "AB"
    assert app.tracking is True


def test_user_set_plane_without_selection_does_not_crash():
    app = _make_app()
    assert app.current_cell_name == ""
    assert app.selection_anchor is None
    # ``tracking`` is armed by default and is only meaningful once something
    # is selected, so a no-selection Z move must leave it alone rather than
    # stomping the flag.
    tracking_before = app.tracking

    app.set_plane(20, user_initiated=True)

    assert app.current_plane == 20
    assert app.current_cell_name == ""
    assert app.selection_anchor is None
    assert app.tracking is tracking_before


def test_next_plane_without_selection_does_not_crash():
    app = _make_app()
    start_plane = app.current_plane

    app.next_plane()
    app.prev_plane()

    assert app.current_plane == start_plane
    assert app.current_cell_name == ""


# ── Programmatic navigation preserves the selection ──────────────


def test_programmatic_set_plane_preserves_selection():
    """Auto-tracking review relies on this path keeping the active cell."""
    app = _select_ab_at_t4()

    app.set_plane(20)

    assert app.current_plane == 20
    assert app.current_cell_name == "AB"
    assert app.selection_anchor is not None
    assert app.tracking is True


def test_programmatic_set_plane_preserves_follow_mode_across_time():
    app = _select_ab_at_t4()
    app.set_plane(20)

    app.next_time()

    # Still following: AB moves from z=15 at T4 to z=14 at T5.
    assert app.current_time == 5
    assert app.current_plane == round(14.0 + NUCZINDEXOFFSET)
    assert app.current_cell_name == "AB"
    assert app.tracking is True


# ── Follow mode really is off after a user Z move ────────────────


def test_next_time_does_not_snap_z_after_user_deselect():
    """Nothing is being followed, so the user's slice survives time travel."""
    app = _select_ab_at_t4()
    app.set_plane(20, user_initiated=True)

    app.next_time()

    assert app.current_time == 5
    # AB sits at z=14 at T5 — the slice must NOT snap back to it.
    assert app.current_plane == 20
    assert app.current_cell_name == ""
    assert app.selection_anchor is None
    assert app.tracking is False


def test_next_time_does_not_snap_z_after_arrow_key_deselect():
    app = _select_ab_at_t4()
    app.next_plane()
    plane_after_nav = app.current_plane

    app.next_time()

    assert app.current_time == 5
    assert app.current_plane == plane_after_nav
    assert app.current_cell_name == ""
    assert app.tracking is False


def test_reselecting_after_user_z_nav_restores_follow_mode():
    """Deselect is not sticky — a fresh selection re-arms Z following."""
    app = _select_ab_at_t4()
    app.set_plane(20, user_initiated=True)
    assert app.tracking is False

    app.select_cell("AB", time=4)

    assert app.current_cell_name == "AB"
    assert app.tracking is True
    assert app.current_plane == round(15.0 + NUCZINDEXOFFSET)

    app.next_time()
    assert app.current_plane == round(14.0 + NUCZINDEXOFFSET)


def test_user_z_nav_in_3d_mode_keeps_selection():
    """3D mode has no displayed slice to scroll off, so Z must not deselect.

    The deselect rule exists because a user Z move walks the 2D view away
    from the active cell's centroid.  In 3D the whole volume is on screen,
    so the premise does not hold and dropping the selection would look
    like an unexplained glitch.
    """
    app = _make_app()
    app.select_cell("AB", time=4)
    app._3d_mode = True

    app.set_plane(20, user_initiated=True)

    assert app.current_plane == 20
    assert app.current_cell_name == "AB"
    assert app.tracking is True
