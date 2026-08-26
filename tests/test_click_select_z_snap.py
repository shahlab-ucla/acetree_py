"""The active cell is bound to the displayed slice, from every entry point.

Java AceTree keeps the highlighted nucleus and the visible z-plane locked
together: whatever you make active, the image jumps to its centroid plane.
Three user gestures make a cell active and all three must snap:

- right-click on the image window (``ViewerIntegration._apply_deferred_click``)
- click in the lineage tree list (``AceTreeApp.select_cell``)
- click in the interactive lineage tree (also ``AceTreeApp.select_cell``)

And one gesture deliberately must *not* snap: a left-click label toggle only
flips an overlay, so yanking the slice there would move the view out from
under whatever the user was actually inspecting.

The image-window cases drive ``_apply_deferred_click`` directly with an
explicit press anchor, mirroring ``tests/test_centroid_click_ordering.py`` --
the generator half of ``_on_click`` is already covered there, and going
straight to the deferred body keeps these tests about z-snapping.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from qtpy.QtCore import Qt  # noqa: E402

from acetree_py.gui.app import AceTreeApp  # noqa: E402
from acetree_py.gui.viewer_integration import ViewerIntegration  # noqa: E402
from tests.test_gui_app import _build_test_manager, _make_app  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────
#
# The synthetic movie from ``tests/test_gui_app`` has 30 planes and puts the
# interesting z differences at T5:
#
#   T1-T2  P0  z=15.0
#   T3-T4  AB  z=15.0   P1  z=15.0
#   T5     AB  z=14.0   P1  z=16.0
#
# so a click at T5 must move the slice off 15 in opposite directions
# depending on which nucleus was hit.


def _click_harness(*, time: int, plane: int):
    """Build a real app + integration wired for deferred-click tests.

    ``update_display`` and ``update_overlays`` are replaced by recorders that
    capture ``current_plane`` *at redraw time*.  That is the assertion that
    matters: the snap has to land before the redraw, otherwise the user sees
    the old slice until something else happens to repaint.
    """
    app = _make_app()
    app.current_time = time
    app.current_plane = plane
    integration = ViewerIntegration(app)
    app._viewer_integration = integration
    layer = object()
    integration._shapes_layer = layer

    redraws: list[int] = []
    overlays: list[int] = []
    app.update_display = lambda: redraws.append(app.current_plane)
    integration.update_overlays = lambda: overlays.append(app.current_plane)
    return app, integration, layer, redraws, overlays


def _deferred_click(integration, layer, app, *, intent, target_anchor,
                    x: float = 100.0, y: float = 150.0) -> None:
    """Run one deferred click with a press context matching the live app.

    Everything except *intent* / *target_anchor* is snapshotted from the app
    so the staleness guards at the top of ``_apply_deferred_click`` pass; the
    guards themselves are covered by ``test_centroid_click_ordering``.
    """
    integration._apply_deferred_click(
        layer=layer,
        intent=intent,
        x=x,
        y=y,
        time=app.current_time,
        plane=app.current_plane,
        change_counter=app.edit_history.change_counter,
        target_anchor=target_anchor,
        selection_anchor=app.selection_anchor,
        current_name=app.current_cell_name,
        relink_callback=None,
        placement_context=(
            app._placement_parent_name,
            app._placement_parent_anchor,
            app._placement_default_size,
        ),
        mode_context=(
            app._relink_pick_mode,
            app._add_mode,
            app._placement_mode,
        ),
    )


# ── Image window (right-click select) ────────────────────────────


class TestImageWindowSelect:
    def test_right_click_select_snaps_plane_before_the_single_redraw(self):
        """A hit several planes off-centroid still brings the cell into focus.

        ``find_closest_nucleus`` scores a click against the nucleus *sphere*,
        so plane 15 can legitimately hit AB whose centroid sits on 14.  The
        selection has to drag the slice with it.
        """
        app, integration, layer, redraws, _overlays = _click_harness(
            time=5, plane=15
        )

        _deferred_click(
            integration, layer, app, intent="select", target_anchor=(5, 1)
        )

        assert app.current_cell_name == "AB"
        assert app.selection_anchor == (5, 1)
        assert app.current_plane == 14
        # Exactly one repaint, and it already saw the snapped plane.
        assert redraws == [14]

    def test_right_click_select_snaps_in_either_direction(self):
        """P1 sits above the current slice where AB sits below it."""
        app, integration, layer, redraws, _overlays = _click_harness(
            time=5, plane=15
        )

        _deferred_click(
            integration, layer, app, intent="select", target_anchor=(5, 2)
        )

        assert app.current_cell_name == "P1"
        assert app.current_plane == 16
        assert redraws == [16]

    def test_selecting_a_second_cell_re_snaps_from_the_new_slice(self):
        """Consecutive selections each rebind the slice to their own cell."""
        app, integration, layer, redraws, _overlays = _click_harness(
            time=5, plane=15
        )

        _deferred_click(
            integration, layer, app, intent="select", target_anchor=(5, 1)
        )
        assert app.current_plane == 14
        # The follow-up click is pressed on the *new* plane, which is what
        # the staleness guard compares against.
        _deferred_click(
            integration, layer, app, intent="select", target_anchor=(5, 2)
        )

        assert app.current_cell_name == "P1"
        assert app.current_plane == 16
        assert redraws == [14, 16]

    def test_right_click_on_empty_space_deselects_without_moving_the_slice(
        self,
    ):
        """No nucleus under the cursor means nothing to bind the slice to."""
        app, integration, layer, redraws, _overlays = _click_harness(
            time=5, plane=15
        )
        _deferred_click(
            integration, layer, app, intent="select", target_anchor=(5, 1)
        )
        assert app.current_plane == 14
        redraws.clear()

        _deferred_click(
            integration, layer, app, intent="select", target_anchor=None
        )

        assert app.current_cell_name == ""
        assert app.selection_anchor is None
        assert app.tracking is False
        # Deselecting is not a navigation gesture: the user keeps looking at
        # the slice they were on.
        assert app.current_plane == 14
        assert redraws == [14]

    def test_right_click_on_a_dead_nucleus_changes_nothing(self):
        """A stale anchor pointing at a killed nucleus must not move the view."""
        app, integration, layer, redraws, _overlays = _click_harness(
            time=5, plane=15
        )
        app.manager.nuclei_record[4][0].status = -1  # kill AB at T5

        _deferred_click(
            integration, layer, app, intent="select", target_anchor=(5, 1)
        )

        assert app.current_plane == 15
        assert app.current_cell_name == ""
        assert redraws == []


# ── Image window (left-click label toggle) ───────────────────────


class TestImageWindowLabelToggle:
    def test_label_toggle_never_moves_the_slice(self):
        """Labelling is an overlay change, not an activation."""
        app, integration, layer, redraws, overlays = _click_harness(
            time=5, plane=15
        )

        _deferred_click(
            integration, layer, app, intent="label", target_anchor=(5, 1)
        )

        assert "AB" in integration._shown_labels
        assert app.current_plane == 15
        # Overlay-only repaint; no full display refresh, no z move.
        assert overlays == [15]
        assert redraws == []

        # Toggling back off is equally inert.
        _deferred_click(
            integration, layer, app, intent="label", target_anchor=(5, 1)
        )
        assert "AB" not in integration._shown_labels
        assert app.current_plane == 15
        assert overlays == [15, 15]
        assert redraws == []

    def test_label_toggle_does_not_change_the_active_cell(self):
        """The previously selected cell keeps both the selection and the slice."""
        app, integration, layer, _redraws, _overlays = _click_harness(
            time=5, plane=15
        )
        _deferred_click(
            integration, layer, app, intent="select", target_anchor=(5, 1)
        )
        assert (app.current_cell_name, app.current_plane) == ("AB", 14)

        # Label the *other* nucleus, whose centroid is two planes away.
        _deferred_click(
            integration, layer, app, intent="label", target_anchor=(5, 2)
        )

        assert integration._shown_labels >= {"P1"}
        assert app.current_cell_name == "AB"
        assert app.selection_anchor == (5, 1)
        assert app.current_plane == 14


# ── Lineage tree list / interactive lineage tree ─────────────────
#
# Both windows route their clicks through ``AceTreeApp.select_cell``, which
# snaps via ``_track_cell_at_time`` -> ``_snap_plane_to_nucleus``.


class TestSelectCellSnapsZ:
    def test_select_cell_snaps_to_the_centroid_at_the_requested_time(self):
        """Left-click in the list: select at ``cell.start_time``."""
        app = _make_app()
        app.current_plane = 25

        cell = app.manager.get_cell("AB")
        app.select_cell("AB", cell.start_time)

        assert app.current_time == 3
        assert app.current_plane == 15

    def test_select_cell_at_end_time_snaps_to_that_timepoint_z(self):
        """Right-click in the list: select at ``cell.end_time``.

        AB and P1 share z=15 for their whole lifetime except the last frame,
        so the end-time jump is the one that proves the snap uses the
        requested timepoint rather than the start.
        """
        app = _make_app()
        app.current_plane = 25

        ab = app.manager.get_cell("AB")
        app.select_cell("AB", ab.end_time)
        assert (app.current_time, app.current_plane) == (5, 14)

        p1 = app.manager.get_cell("P1")
        app.select_cell("P1", p1.end_time)
        assert (app.current_time, app.current_plane) == (5, 16)

    def test_select_cell_without_a_time_jumps_to_start_and_snaps(self):
        """Callers that pass no time still land on a bound slice."""
        app = _make_app()
        app.current_time = 1
        app.current_plane = 2

        app.select_cell("AB")

        assert app.current_time == 3  # outside AB's lifetime -> start_time
        assert app.current_plane == 15

    def test_select_cell_keeps_an_in_range_time_and_snaps_there(self):
        """An already-valid timepoint is preserved, and Z follows it."""
        app = _make_app()
        app.current_time = 5
        app.current_plane = 1

        app.select_cell("P1")

        assert app.current_time == 5
        assert app.current_plane == 16

    def test_select_cell_re_enables_follow_mode(self):
        """A Z nudge disables tracking; selecting a cell must restore it.

        Otherwise the slice would stay unbound while scrubbing time, which is
        the same bug as not snapping in the first place, just delayed.
        """
        app = _make_app()
        app.tracking = False

        app.select_cell("AB", 5)

        assert app.tracking is True
        assert app.current_plane == 14

    @pytest.mark.parametrize(
        ("z", "expected_plane"),
        [(999.0, 30), (-40.0, 1)],
    )
    def test_select_cell_clamps_the_snap_to_the_movie_plane_bounds(
        self, z, expected_plane
    ):
        """A corrupt/out-of-range centroid must not scroll off the stack."""
        mgr = _build_test_manager()
        mgr.nuclei_record[4][0].z = z  # AB at T5
        app = AceTreeApp(mgr, image_provider=None)
        app.current_time = 1
        app.current_plane = 15

        app.select_cell("AB", 5)

        assert app.current_plane == expected_plane

    def test_select_cell_snaps_even_when_the_cell_has_an_interior_gap(self):
        """Regression: an interior hole must not strand the slice.

        ``_find_nucleus_via_chain`` walks the predecessor/successor chain
        only for times OUTSIDE the cell's known range, so a cell missing a
        mid-track nucleus (e.g. one removed by an edit) used to end up
        selected, with a valid anchor, on whatever slice the user happened
        to be on.  ``_track_cell_at_time`` now falls back to the anchored
        nucleus so the display stays bound to the active cell.
        """
        app = _make_app()
        ab = app.manager.get_cell("AB")  # T3, T4, T5
        # Punch a hole at T4, as an edit that removes one nucleus would.
        ab.nuclei = [(t, n) for (t, n) in ab.nuclei if t != 4]
        ab._nuclei_by_time.pop(4, None)
        app.current_plane = 25

        app.select_cell("AB", 4)

        assert app.current_time == 4
        assert app.selection_anchor == (3, 1)  # anchored to the nearest nucleus
        # The slice follows the anchor's Z rather than staying at 25.
        assert app.current_plane == 15

    def test_select_cell_on_a_nucleus_less_cell_cannot_snap(self):
        """Documents the one case where there is genuinely nothing to snap to.

        ``build_lineage_tree(create_dummy_ancestors=True)`` scaffolds named
        descendants (ABa, ABp, EMS, ...) that hold no nuclei at all.  Selecting
        one names it and enables follow-mode but leaves the slice alone, since
        no centroid exists.  Asserted so a future change here is a deliberate
        one rather than a silent regression.
        """
        from acetree_py.core.lineage import build_lineage_tree

        mgr = _build_test_manager()
        mgr.lineage_tree = build_lineage_tree(
            mgr.nuclei_record,
            starting_index=0,
            ending_index=5,
            create_dummy_ancestors=True,
        )
        app = AceTreeApp(mgr, image_provider=None)
        app.current_time = 1
        app.current_plane = 25
        phantom = next(
            c for c in mgr.lineage_tree.cells_by_name.values() if not c.nuclei
        )

        app.select_cell(phantom.name)

        assert app.current_cell_name == phantom.name
        assert app.selection_anchor is None
        assert app.current_plane == 25

    def test_select_cell_on_an_unknown_name_is_inert(self):
        """No cell means no selection and, importantly, no slice movement."""
        app = _make_app()
        app.current_plane = 20

        app.select_cell("NOT_A_CELL")

        assert app.current_plane == 20
        assert app.current_cell_name == ""


# ── Lineage list widget: right-click must not double-fire ────────


class TestLineageListButtonGuard:
    """``itemClicked`` fires on right-button release in some Qt versions.

    Without a guard a single right-click runs ``_on_item_clicked`` (select at
    start_time) *and* ``_on_right_click`` (select at end_time), so the user
    sees the view jump twice -- in time and, now that selection snaps Z, in
    plane too -- and pays for two full redraws.
    """

    def _widget(self, qtbot):
        from acetree_py.gui.lineage_list import LineageListWidget

        app = _make_app()
        widget = LineageListWidget(app)
        qtbot.addWidget(widget)
        calls: list[tuple[str, int | None]] = []
        app.select_cell = lambda name, time=None: calls.append((name, time))
        return app, widget, calls

    def test_viewport_press_records_the_button(self, qtbot):
        """The event filter is actually installed on the live viewport."""
        _app, widget, _calls = self._widget(qtbot)
        viewport = widget._tree.viewport()

        qtbot.mousePress(viewport, Qt.RightButton, pos=viewport.rect().center())
        assert widget._last_press_button == Qt.RightButton

        qtbot.mousePress(viewport, Qt.LeftButton, pos=viewport.rect().center())
        assert widget._last_press_button == Qt.LeftButton

    def test_right_click_does_not_also_run_the_left_click_handler(self, qtbot):
        _app, widget, calls = self._widget(qtbot)
        viewport = widget._tree.viewport()
        item = widget._items["AB"]

        qtbot.mousePress(viewport, Qt.RightButton, pos=viewport.rect().center())
        # Simulate the spurious ``itemClicked`` some Qt versions emit on the
        # right-button release that follows.
        widget._on_item_clicked(item, 0)

        assert calls == []

    def test_left_click_still_selects_at_start_time(self, qtbot):
        _app, widget, calls = self._widget(qtbot)
        viewport = widget._tree.viewport()
        item = widget._items["AB"]

        qtbot.mousePress(viewport, Qt.LeftButton, pos=viewport.rect().center())
        widget._on_item_clicked(item, 0)

        assert calls == [("AB", 3)]

    def test_right_click_handler_still_selects_at_end_time(self, qtbot):
        _app, widget, calls = self._widget(qtbot)
        widget._tree.expandAll()
        item = widget._items["AB"]
        position = widget._tree.visualItemRect(item).center()

        widget._on_right_click(position)

        assert calls == [("AB", 5)]
