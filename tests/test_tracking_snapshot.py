"""Safety contracts for background tracking document snapshots."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import Nucleus
from acetree_py.editing.commands import AddNucleus
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.config import AceTreeConfig
from acetree_py.io.image_provider import NumpyProvider
from acetree_py.tracking.api import ComponentSpec, TrackingRequest, TrackingScope


def test_prepared_tracking_is_invalid_after_edit_then_undo() -> None:
    """Undo restoring the same revision must not revive an observed draft."""

    manager = NucleiManager.new_empty(
        AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=3),
        num_timepoints=1,
    )
    provider = NumpyProvider(np.zeros((1, 3, 8, 8), dtype=np.float32))
    app = AceTreeApp(manager, provider)
    penalties = {"quality": 0.25}
    request = TrackingRequest(
        detector=ComponentSpec("acetree.dog3d", {"TARGET_CHANNEL": 1}),
        tracker=ComponentSpec("acetree.simple_lap", {"LINKING_FEATURE_PENALTIES": penalties}),
        scope=TrackingScope("global", 1, 1),
    )
    snapshot = app.prepare_tracking_analysis(request)
    penalties["quality"] = 99
    assert snapshot.request.tracker.settings["LINKING_FEATURE_PENALTIES"]["quality"] == 0.25

    app.edit_history.do(AddNucleus(time=1, x=3, y=4, z=2.0, size=4))
    app.edit_history.undo()

    assert app.edit_history.revision == snapshot.revision
    assert app.edit_history.change_counter != snapshot.change_counter
    with pytest.raises(RuntimeError, match="dataset changed before tracking started"):
        app.analyze_prepared_tracking(snapshot)


@pytest.mark.parametrize("kind", ["global", "selected_forward"])
def test_tracking_snapshot_copies_nuclei_only_for_seeded_analysis(kind):
    manager = NucleiManager.new_empty(
        AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=3), num_timepoints=1,
    )
    nucleus = Nucleus(index=1, x=4, status=1)
    manager.nuclei_record[0].append(nucleus)
    app = AceTreeApp(manager, NumpyProvider(np.zeros((1, 3, 8, 8), dtype=np.float32)))
    request = TrackingRequest(
        ComponentSpec("acetree.dog3d"), ComponentSpec("acetree.simple_lap"),
        TrackingScope(kind, 1, 1, seed_anchors=((1, 1),) if kind == "selected_forward" else ()),
    )

    snapshot = app.prepare_tracking_analysis(request)
    nucleus.x = 99
    if kind == "global":
        assert snapshot.nuclei_record is None
    else:
        assert snapshot.nuclei_record[0][0].x == 4
        assert snapshot.nuclei_record[0][0] is not nucleus
