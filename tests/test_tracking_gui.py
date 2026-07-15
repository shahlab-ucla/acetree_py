"""Small Qt contract tests for the manual/automated tracking workflow."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from acetree_py.gui.dataset_dialog import DatasetCreationDialog
from acetree_py.gui.edit_panel import AutoTrackForwardDialog
from acetree_py.core.nucleus import Nucleus
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.editing.commands import AddNucleus
from acetree_py.editing.history import EditHistory
from acetree_py.gui.app import AceTreeApp
from acetree_py.gui.viewer_integration import ViewerIntegration
from acetree_py.io.config import AceTreeConfig
from acetree_py.io.image_provider import NumpyProvider
from acetree_py.tracking.api import Calibration, Detection, TrackEdge, TrackingResult


def test_dataset_wizard_defaults_to_manual_and_has_tracking_page(qtbot):
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)

    assert dialog._stack.count() == 5
    assert dialog.get_tracking_request() is None
    assert dialog._radio_tracking_manual.isChecked()


def test_dataset_wizard_builds_trackmate_keyed_global_request(qtbot):
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    dialog._radio_tracking_auto.setChecked(True)
    dialog._tracking_gap_spin.setValue(1)

    request = dialog.get_tracking_request()

    assert request is not None
    assert request.scope.kind == "global"
    assert request.detector.plugin_id in {"acetree.dog3d", "acetree.log3d"}
    assert request.detector.settings["TARGET_CHANNEL"] == 1
    assert request.tracker.plugin_id == "acetree.simple_lap"
    # The UI speaks in missed frames; TrackMate MAX_FRAME_GAP is the frame delta.
    assert request.tracker.settings["MAX_FRAME_GAP"] == 2
    assert request.tracker.settings["ALLOW_TRACK_SPLITTING"] is False


def test_selected_forward_dialog_keeps_physical_seed_and_local_scope(qtbot):
    dialog = AutoTrackForwardDialog(start_time=7, end_time=20, num_channels=2)
    qtbot.addWidget(dialog)
    dialog._end_spin.setValue(15)
    dialog._channel_spin.setValue(2)

    request = dialog.get_request((7, 3))

    assert request.scope.kind == "selected_forward"
    assert request.scope.seed_anchors == ((7, 3),)
    assert request.scope.end_frame == 15
    assert request.scope.roi_radius_um > 0
    assert request.detector.settings["TARGET_CHANNEL"] == 2


class _PreviewSpy:
    def __init__(self):
        self.shown = []
        self.cleared = 0
        self.visible = True
        self.highlighted = None

    def show_tracking_preview(self, proposal, calibration, **options):
        self.shown.append((proposal, calibration, options))

    def clear_tracking_preview(self):
        self.cleared += 1

    def set_tracking_preview_visible(self, visible):
        self.visible = visible

    def highlight_tracking_preview(self, preview_id):
        self.highlighted = preview_id


class _TrackingDialogApp:
    def __init__(self):
        record = [[Nucleus(index=1, x=5, y=5, z=2.0, size=4, status=1)], [], []]
        self.manager = SimpleNamespace(
            config=AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=3),
            nuclei_record=record,
        )
        self.edit_history = EditHistory(record)
        self.current_time = 1
        self.current_plane = 2
        self.current_cell_name = "EMS"
        self.selection_anchor = (1, 1)
        self.tracking = True
        self.viewer = None
        self._viewer_integration = _PreviewSpy()
        self.analysis_calls = []
        self.accept_calls = []

    def analyze_tracking_request(self, request, *, progress=None, cancelled=None):
        self.analysis_calls.append(request)
        run = len(self.analysis_calls)
        if progress is not None:
            progress(1, 1, f"Tracking selected cell at time {run + 1}")
        assert cancelled is None or not cancelled()
        seed = Detection("seed", 1, 5.0, 5.0, 1.0, 2.0, 1.0)
        target = Detection(
            f"run-{run}",
            2,
            5.0 + run,
            5.0,
            1.0,
            2.0,
            10.0,
        )
        result = TrackingResult(
            request=request,
            detections=(seed, target),
            edges=(TrackEdge("seed", target.detection_id, 1.0),),
            existing_anchors={"seed": (1, 1)},
        )
        return result, self.edit_history.revision

    def accept_tracking_proposal(self, proposal, *, expected_revision):
        self.accept_calls.append((proposal, expected_revision))
        return {}

    def set_time(self, value):
        self.current_time = value

    def set_plane(self, value):
        self.current_plane = value

    def update_display(self):
        pass


class _FakeShapesLayer:
    def __init__(self):
        self.data = []
        self.visible = False
        self.editable = False
        self.last_add = None

    def add(self, data, **kwargs):
        self.data = list(data)
        self.last_add = kwargs


def test_auto_forward_starts_safe_and_supports_adjust_rerun(qtbot):
    app = _TrackingDialogApp()
    dialog = AutoTrackForwardDialog(
        1,
        3,
        app=app,
        seed_anchor=(1, 1),
        seed_label="EMS",
    )
    qtbot.addWidget(dialog)

    assert dialog.state == dialog.CONFIGURING
    assert not dialog._accept_button.isEnabled()

    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    first = dialog.proposal
    assert dialog.state == dialog.READY
    assert dialog._accept_button.isEnabled()
    assert len(app._viewer_integration.shown) == 1

    dialog._threshold_spin.setValue(dialog._threshold_spin.value() + 1)
    assert dialog.state == dialog.OUTDATED
    assert not dialog._accept_button.isEnabled()
    assert app._viewer_integration.shown[-1][2]["stale"] is True

    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    assert dialog.state == dialog.READY
    assert dialog.proposal is not first
    assert len(app.analysis_calls) == 2

    dialog._accept_button.click()
    qtbot.waitUntil(lambda: app._viewer_integration.cleared == 1)
    assert len(app.accept_calls) == 1
    assert app.accept_calls[0][0] is not first
    assert app._viewer_integration.cleared == 1


def test_auto_forward_discard_restores_view_and_never_commits(qtbot):
    app = _TrackingDialogApp()
    dialog = AutoTrackForwardDialog(
        1,
        3,
        app=app,
        seed_anchor=(1, 1),
        seed_label="EMS",
    )
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    dialog._navigate_to_row(1)
    assert app.current_time == 2

    dialog.reject()
    qtbot.waitUntil(lambda: app._viewer_integration.cleared == 1)

    assert app.current_time == 1
    assert app.current_plane == 2
    assert app.selection_anchor == (1, 1)
    assert app.accept_calls == []
    assert app._viewer_integration.cleared == 1


def test_cancelled_auto_forward_stays_open_for_refinement(qtbot):
    from acetree_py.tracking.pipeline import TrackingCancelled

    app = _TrackingDialogApp()

    def cancel_analysis(*_args, **_kwargs):
        raise TrackingCancelled("cancelled")

    app.analyze_tracking_request = cancel_analysis
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)

    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state != dialog.RUNNING)

    assert dialog.state == dialog.CONFIGURING
    assert not dialog._accept_button.isEnabled()
    assert "No changes" in dialog._banner.text()
    assert app.accept_calls == []
    dialog.reject()


def test_close_during_background_analysis_requests_cancel_before_cleanup(qtbot):
    from acetree_py.tracking.pipeline import TrackingCancelled

    app = _TrackingDialogApp()
    started = threading.Event()
    worker_threads = []

    def slow_analysis(request, *, progress=None, cancelled=None):
        worker_threads.append(threading.get_ident())
        started.set()
        while cancelled is None or not cancelled():
            time.sleep(0.005)
        raise TrackingCancelled("cancelled")

    app.analyze_tracking_request = slow_analysis
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog.show()
    main_thread = threading.get_ident()

    dialog._preview_button.click()
    qtbot.waitUntil(started.is_set)
    dialog.reject()

    assert dialog._close_after_run is True
    assert worker_threads != [main_thread]
    qtbot.waitUntil(lambda: app._viewer_integration.cleared == 1)


def test_keyboard_row_navigation_and_bounded_draft_playback_follow_viewer(qtbot):
    app = _TrackingDialogApp()
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog._review_timer.setInterval(10)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)

    dialog._table.setCurrentCell(1, 0)
    assert app.current_time == 2
    assert not dialog._next_button.isEnabled()
    assert dialog._previous_button.isEnabled()

    dialog._navigate_to_row(0)
    dialog._play_button.click()
    qtbot.waitUntil(lambda: app.current_time == 2)
    qtbot.waitUntil(lambda: not dialog._review_timer.isActive())
    assert not dialog._play_button.isChecked()
    dialog.reject()


def test_no_continuation_keeps_settings_available_for_rerun(qtbot):
    app = _TrackingDialogApp()

    def empty_analysis(request, **_kwargs):
        seed = Detection("seed", 1, 5.0, 5.0, 1.0, 2.0, 1.0)
        return (
            TrackingResult(
                request=request,
                detections=(seed,),
                edges=(),
                existing_anchors={"seed": (1, 1)},
                warnings=("Stopped at t=2: no unique candidate passed the distance gate",),
            ),
            app.edit_history.revision,
        )

    app.analyze_tracking_request = empty_analysis
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.EMPTY)

    assert dialog.state == dialog.EMPTY
    assert not dialog._accept_button.isEnabled()
    assert dialog._settings_widget.isEnabled()
    dialog._threshold_spin.setValue(1.0)
    assert dialog.state == dialog.OUTDATED
    dialog.reject()


def test_document_edit_makes_auto_forward_draft_stale_even_after_undo(qtbot):
    app = _TrackingDialogApp()
    dialog = AutoTrackForwardDialog(
        1,
        3,
        app=app,
        seed_anchor=(1, 1),
    )
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    assert dialog.state == dialog.READY

    app.edit_history.do(AddNucleus(time=2, x=1, y=1, z=1.0, size=2))
    app.edit_history.undo()
    assert app.edit_history.revision == dialog._proposal_revision
    dialog.sync_document_revision()

    assert dialog.state == dialog.STALE
    assert not dialog._accept_button.isEnabled()
    dialog.reject()


def test_auto_forward_overlay_draws_only_current_draft_and_never_steals_selection(qtbot):
    source_app = _TrackingDialogApp()
    request_dialog = AutoTrackForwardDialog(1, 3, seed_anchor=(1, 1))
    qtbot.addWidget(request_dialog)
    request = request_dialog.get_request((1, 1))
    seed = Detection("seed", 1, 5.0, 5.0, 1.0, 2.0, 1.0)
    target = Detection("target", 2, 6.0, 5.0, 1.0, 2.0, 10.0)
    proposal = TrackingResult(
        request=request,
        detections=(seed, target),
        edges=(TrackEdge("seed", "target", 1.0),),
        existing_anchors={"seed": (1, 1)},
    )
    nuclei_layer = object()
    selection = SimpleNamespace(active=None)
    source_app.viewer = SimpleNamespace(
        layers=SimpleNamespace(selection=selection),
    )
    source_app.current_time = 2
    source_app.current_plane = 2
    integration = ViewerIntegration(source_app)
    integration._shapes_layer = nuclei_layer
    integration._tracking_preview_spots_layer = _FakeShapesLayer()
    integration._tracking_preview_links_layer = _FakeShapesLayer()

    integration.show_tracking_preview(proposal, Calibration(1.0, 1.0, 1))

    assert len(integration._tracking_preview_spots_layer.data) == 1
    assert len(integration._tracking_preview_links_layer.data) == 1
    assert selection.active is nuclei_layer
    integration.clear_tracking_preview()
    assert integration._tracking_preview_spots_layer.data == []
    assert integration._tracking_preview_links_layer.data == []


def test_auto_forward_button_accepts_real_selection_tuple(qtbot):
    from acetree_py.gui.edit_panel import EditPanel

    config = AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=3)
    manager = NucleiManager.new_empty(config, 3)
    manager.nuclei_record[0].append(
        Nucleus(
            index=1,
            x=5,
            y=5,
            z=2.0,
            size=4,
            identity="EMS",
            assigned_id="EMS",
            status=1,
        )
    )
    manager.process()
    provider = NumpyProvider(np.zeros((3, 3, 12, 12), dtype=np.float32))
    app = AceTreeApp(manager, provider)
    app.selection_anchor = (1, 1)
    app.current_cell_name = "EMS"
    panel = EditPanel(app)
    app._edit_panel = panel
    qtbot.addWidget(panel)

    panel._on_auto_track_forward()

    assert panel._auto_track_dialog is not None
    assert panel._auto_track_dialog._seed_anchor == (1, 1)
    panel._auto_track_dialog.reject()


def test_auto_forward_resolves_linear_endpoint_but_never_chooses_at_division():
    from acetree_py.gui.edit_panel import EditPanel

    linear = [
        [Nucleus(index=1, successor1=1, status=1)],
        [Nucleus(index=1, predecessor=1, status=1)],
    ]
    stub = SimpleNamespace(app=SimpleNamespace(manager=SimpleNamespace(nuclei_record=linear)))
    nucleus, time, index = EditPanel._resolve_auto_track_seed(stub, 1, 1)
    assert nucleus is linear[1][0]
    assert (time, index) == (2, 1)

    division = [
        [Nucleus(index=1, successor1=1, successor2=2, status=1)],
        [
            Nucleus(index=1, predecessor=1, status=1),
            Nucleus(index=2, predecessor=1, status=1),
        ],
    ]
    stub.app.manager.nuclei_record = division
    assert EditPanel._resolve_auto_track_seed(stub, 1, 1) is None
