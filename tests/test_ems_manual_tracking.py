"""Focused regression coverage for manually curated EMS sublineages."""

from __future__ import annotations

from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.editing.commands import RenameCell, SetBodyAxes
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.config import AceTreeConfig, NamingMethod
from acetree_py.naming.body_axes import BodyAxisFrame


def _manual_ems_app() -> AceTreeApp:
    """Create a sparse movie with one forced EMS anchor and manual axes."""
    config = AceTreeConfig(
        naming_method=NamingMethod.NEWCANONICAL,
        plane_end=30,
        xy_res=0.1,
        z_res=1.0,
    )
    manager = NucleiManager.new_empty(config, num_timepoints=5)
    manager.process()
    app = AceTreeApp(manager, image_provider=None)
    app.current_time = 1
    app.current_plane = 10

    app.enter_add_mode()
    assert app._handle_add_click(200.0, 200.0)
    app.edit_history.do(RenameCell(time=1, index=1, new_name="EMS"))

    # Lab coordinates already match the transform's canonical convention:
    # AP is posterior -> anterior (-X), LR is right -> left (+Z).
    frame = BodyAxisFrame.from_auxinfo_vectors(
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        provenance="manual_landmarks",
        reference_time=1,
    )
    app.edit_history.do(SetBodyAxes(manager, frame))
    app.exit_add_mode()
    return app


def test_forced_ems_track_out_automatically_names_e_ms_then_ea_ep():
    """A forced founder anchor feeds unlocked, axis-named descendants.

    This is the manual-curation workflow exercised through Track placement:
    force EMS, place its two daughters, select E, then place E's daughters.
    """
    app = _manual_ems_app()

    # Track the first apparent EMS continuation, then add its far sibling.
    # The second click proves that the first placement was actually a daughter,
    # so the inherited EMS override must be removed from both daughters.
    app.set_time(2)
    app.enter_placement_mode(parent_name="EMS")
    app.current_plane = 10
    assert app._handle_placement_click(200.0, 200.0)
    app.current_plane = 11
    assert app._handle_placement_click(140.0, 197.0)

    ems = app.manager.nuclei_record[0][0]
    daughters = app.manager.nuclei_record[1]
    assert ems.effective_name == "EMS"
    assert ems.assigned_id == "EMS"
    assert {n.effective_name for n in daughters} == {"E", "MS"}
    assert all(n.assigned_id == "" for n in daughters)
    assert next(n for n in daughters if n.x == 200).effective_name == "E"

    # Continue E by one placement, then make the second placement at the same
    # timepoint far enough away to commit the E division.
    app.exit_placement_mode()
    app.set_time(3)
    app.enter_placement_mode(parent_name="E")
    app.current_plane = 10
    assert app._handle_placement_click(210.0, 195.0)
    app.current_plane = 8
    assert app._handle_placement_click(270.0, 180.0)

    granddaughters = app.manager.nuclei_record[2]
    assert {n.effective_name for n in granddaughters} == {"Ea", "Ep"}
    assert all(n.assigned_id == "" for n in granddaughters)
    assert next(n for n in granddaughters if n.x == 210).effective_name == "Ea"


def test_reversing_manual_ap_axis_reverses_ea_ep_spatial_assignment():
    """Correcting the manual AP direction changes ordering, not identities."""
    app = _manual_ems_app()

    # Seed E as an automatic child of the forced EMS anchor.
    app.set_time(2)
    app.enter_placement_mode(parent_name="EMS")
    app.current_plane = 10
    assert app._handle_placement_click(200.0, 200.0)
    app.current_plane = 11
    assert app._handle_placement_click(140.0, 197.0)

    # Reverse AP and LR together to retain a right-handed frame. This models
    # correcting two swapped anatomical endpoint pairs in the GUI.
    corrected = BodyAxisFrame.from_auxinfo_vectors(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        provenance="manual_landmarks",
        reference_time=2,
    )
    app.edit_history.do(SetBodyAxes(app.manager, corrected))

    app.exit_placement_mode()
    app.set_time(3)
    app.enter_placement_mode(parent_name="E")
    app.current_plane = 10
    assert app._handle_placement_click(210.0, 195.0)
    app.current_plane = 8
    assert app._handle_placement_click(270.0, 180.0)

    granddaughters = app.manager.nuclei_record[2]
    assert {n.effective_name for n in granddaughters} == {"Ea", "Ep"}
    assert next(n for n in granddaughters if n.x == 210).effective_name == "Ep"
    assert all(n.assigned_id == "" for n in granddaughters)
