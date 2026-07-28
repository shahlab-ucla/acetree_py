"""Small Qt contract tests for the manual/automated tracking workflow."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from acetree_py.gui.dataset_dialog import DatasetCreationDialog
from acetree_py.gui.edit_panel import AutoTrackForwardDialog
from acetree_py.gui.auto_tracking_dialog import _unique_new_terminal
from acetree_py.core.nucleus import Nucleus
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.editing.commands import AddNucleus
from acetree_py.editing.history import EditHistory
from acetree_py.gui.app import AceTreeApp
from acetree_py.gui.viewer_integration import ViewerIntegration
from acetree_py.io.config import AceTreeConfig
from acetree_py.io.image_provider import NumpyProvider
from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)
from acetree_py.tracking.registry import ComponentDescriptor, TrackingRegistry
from acetree_py.tracking.starrynite import read_parameter_file


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
    assert dialog._tracking_workflow_combo.currentData() == "modern_starrynite"
    assert request.detector.plugin_id == "acetree.starrynite_detector"
    assert request.detector.settings["TARGET_CHANNEL"] == 1
    assert request.detector.settings["THRESHOLD"] == 0.0
    assert request.tracker.plugin_id == "acetree.starrynite_division"
    # The UI speaks in missed frames; TrackMate MAX_FRAME_GAP is the frame delta.
    assert request.tracker.settings["MAX_FRAME_GAP"] == 2
    assert request.tracker.settings["ALLOW_TRACK_SPLITTING"] is True


def test_dataset_wizard_enables_reviewed_divisions_for_starrynite(qtbot):
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    dialog._radio_tracking_auto.setChecked(True)

    assert dialog._tracking_division_check.isEnabled()
    assert dialog._tracking_division_check.isChecked()
    assert "can propose two-daughter divisions" in (
        dialog._tracking_capability_label.text()
    )

    lap_index = dialog._tracking_workflow_combo.findData("log_lap")
    assert lap_index >= 0
    dialog._tracking_workflow_combo.setCurrentIndex(lap_index)

    assert not dialog._tracking_division_check.isEnabled()
    assert not dialog._tracking_division_check.isChecked()
    assert "does not propose divisions" in dialog._tracking_capability_label.text()

    modern_index = dialog._tracking_workflow_combo.findData("modern_starrynite")
    dialog._tracking_workflow_combo.setCurrentIndex(modern_index)

    request = dialog.get_tracking_request()

    assert dialog._tracking_division_check.isEnabled()
    assert dialog._tracking_division_check.isChecked()
    assert "can propose two-daughter divisions" in (
        dialog._tracking_capability_label.text()
    )
    assert request is not None
    assert request.tracker.settings["ALLOW_TRACK_SPLITTING"] is True
    assert request.tracker.settings["ALLOW_TRACK_MERGING"] is False
    assert "divisions=on" in dialog._tracking_description()

    dialog._tracking_division_check.setChecked(False)
    request = dialog.get_tracking_request()
    assert request is not None
    assert request.tracker.settings["ALLOW_TRACK_SPLITTING"] is False


def test_dataset_wizard_preserves_starrynite_localization_default(qtbot):
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    dialog._radio_tracking_auto.setChecked(True)
    detector_index = dialog._tracking_detector_combo.findData(
        "acetree.starrynite_detector"
    )
    dialog._tracking_detector_combo.setCurrentIndex(detector_index)

    request = dialog.get_tracking_request()

    assert request is not None
    assert request.detector.settings["DO_SUBPIXEL_LOCALIZATION"] is False


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
    assert request.scope.branch_policy == "stop"
    assert request.detector.settings["TARGET_CHANNEL"] == 2
    assert request.tracker.settings["ALLOW_TRACK_SPLITTING"] is False
    assert dialog._workflow_combo.currentData() == "modern_starrynite"
    assert dialog._branch_policy_combo.model().item(
        dialog._follow_both_index
    ).isEnabled()


def test_selected_forward_enables_follow_both_for_splitting_tracker(
    qtbot,
    monkeypatch,
):
    registry = TrackingRegistry()
    registry.register_detector(
        ComponentDescriptor(
            "example.detector",
            "detector",
            "Example detector",
            settings_schema={
                "TARGET_CHANNEL": {"default": 1},
                "RADIUS": {"default": 4.0},
                "THRESHOLD": {"default": 0.0},
                "DO_SUBPIXEL_LOCALIZATION": {"default": True},
                "DO_MEDIAN_FILTERING": {"default": False},
            },
        ),
        lambda: SimpleNamespace(detect=lambda *_args, **_kwargs: ()),
    )
    registry.register_tracker(
        ComponentDescriptor(
            "example.splitting_tracker",
            "tracker",
            "Splitting tracker",
            settings_schema={
                "LINKING_MAX_DISTANCE": {"default": 8.0},
                "ALLOW_GAP_CLOSING": {"default": True},
                "GAP_CLOSING_MAX_DISTANCE": {"default": 8.0},
                "MAX_FRAME_GAP": {"default": 2},
                "ALLOW_TRACK_SPLITTING": {"default": False},
                "ALLOW_TRACK_MERGING": {"default": False},
            },
            capabilities=("splitting",),
        ),
        lambda: SimpleNamespace(track=lambda *_args, **_kwargs: ()),
    )
    import acetree_py.tracking.registry as registry_module

    monkeypatch.setattr(registry_module, "get_default_registry", lambda: registry)
    dialog = AutoTrackForwardDialog(
        start_time=3,
        end_time=8,
        initial_settings={"branch_policy": "follow_both"},
    )
    qtbot.addWidget(dialog)

    assert dialog._branch_policy_combo.model().item(
        dialog._follow_both_index
    ).isEnabled()
    request = dialog.get_request((3, 2))
    assert request.scope.branch_policy == "follow_both"
    assert request.tracker.settings["ALLOW_TRACK_SPLITTING"] is True
    assert request.tracker.settings["ALLOW_TRACK_MERGING"] is False
    assert request.tracker.settings["ALLOW_GAP_CLOSING"] is False
    assert not dialog._gap_spin.isEnabled()
    assert dialog.export_settings()["branch_policy"] == "follow_both"


def test_multiple_daughter_terminals_do_not_choose_an_arbitrary_endpoint():
    request = TrackingRequest(
        ComponentSpec("example.detector", {}),
        ComponentSpec("example.splitting_tracker", {}),
        TrackingScope(
            "selected_forward",
            1,
            2,
            seed_anchors=((1, 1),),
            branch_policy="follow_both",
        ),
    )
    seed = Detection("seed", 1, 0.0, 0.0, 0.0, 1.0, 1.0)
    first = Detection("first", 2, -1.0, 0.0, 0.0, 1.0, 1.0)
    second = Detection("second", 2, 1.0, 0.0, 0.0, 1.0, 1.0)
    proposal = TrackingResult(
        request=request,
        detections=(seed, first, second),
        edges=(
            TrackEdge("seed", "first", 1.0, kind="split"),
            TrackEdge("seed", "second", 1.0, kind="split"),
        ),
        existing_anchors={"seed": (1, 1)},
    )

    assert _unique_new_terminal(proposal) is None


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


class _CommittingTrackingDialogApp(_TrackingDialogApp):
    """Small app double that crosses the real atomic tracking-history boundary."""

    def __init__(self, saved_path: Path):
        super().__init__()
        self.saved_path = saved_path
        self.save_calls = 0

    def accept_tracking_proposal(self, proposal, *, expected_revision):
        self.accept_calls.append((proposal, expected_revision))
        return AceTreeApp.accept_tracking_proposal(
            self,
            proposal,
            expected_revision=expected_revision,
        )

    def save(self):
        self.save_calls += 1
        return self.saved_path


def test_auto_forward_parameter_preset_uses_alive_cell_count_for_stage(
    qtbot,
    tmp_path,
):
    app = _TrackingDialogApp()
    app.manager.nuclei_record[0] = [
        Nucleus(index=index, x=index, y=5, z=2.0, size=4, status=1)
        for index in range(1, 82)
    ]
    parameter_path = tmp_path / "late-stage.txt"
    parameter_path.write_text(
        "xyres=.25;\n"
        "firsttimestepdiam=40;\n"
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[10,20,30];\n"
        "load 'missing-model.mat';\n",
        encoding="utf-8",
    )
    dialog = AutoTrackForwardDialog(
        1,
        3,
        app=app,
        seed_anchor=(1, 1),
    )
    qtbot.addWidget(dialog)

    dialog.load_starrynite_parameter_file(str(parameter_path))
    request = dialog.get_request()

    assert dialog._threshold_spin.value() == pytest.approx(30.0)
    assert request.detector.plugin_id == "acetree.starrynite_detector"
    assert request.detector.settings["STARRYNITE_CELL_COUNT"] == 81
    assert request.detector.settings["STARRYNITE_STAGE_INDEX"] == 2
    assert request.detector.settings["THRESHOLD"] == 0.0
    assert request.detector.settings["INTENSITY_THRESHOLD"] == pytest.approx(30.0)
    assert request.detector.settings["DO_SUBPIXEL_LOCALIZATION"] is False
    assert not dialog._subpixel_check.isChecked()
    assert dialog._starrynite_save_button.isEnabled()
    assert "Preset ready" in dialog._starrynite_file_label.text()
    assert "stage 3" in dialog._starrynite_file_label.text()
    dialog._advanced_toggle.setChecked(True)
    assert "not applied by this workbench" in dialog._starrynite_file_label.text()

    dialog._generated_settings = dialog.export_settings()
    follow_both_index = dialog._branch_policy_combo.findData("follow_both")
    dialog._branch_policy_combo.setCurrentIndex(follow_both_index)
    assert "division behavior" in dialog._changed_setting_labels()
    stop_index = dialog._branch_policy_combo.findData("stop")

    dialog._radius_spin.setValue(6.0)
    dialog._threshold_spin.setValue(33.0)
    dialog._gap_spin.setValue(2)
    dialog._roi_spin.setValue(27.0)
    dialog._distance_spin.setValue(13.0)
    dialog._ambiguity_spin.setValue(1.65)
    dialog._branch_policy_combo.setCurrentIndex(stop_index)
    dialog._subpixel_check.setChecked(True)
    dialog._median_check.setChecked(True)
    saved_path = tmp_path / "late-stage-tuned.txt"
    warnings = dialog.save_starrynite_parameter_file(str(saved_path))
    saved = read_parameter_file(saved_path)

    assert warnings == ()
    assert saved.normalized_settings["parameters.intensitythreshold"] == (
        10,
        20,
        33.0,
    )
    assert saved.normalized_settings["firsttimestepdiam"] == pytest.approx(48.0)
    assert saved.normalized_settings["trackingparameters.temporalcutoff"] == 3
    assert dialog.recent_starrynite_parameter_file() == saved_path.resolve()
    assert "expected radius" in dialog._starrynite_save_explanation.text()
    assert dialog._roi_spin.value() == pytest.approx(27.0)
    assert dialog._distance_spin.value() == pytest.approx(13.0)
    assert dialog._ambiguity_spin.value() == pytest.approx(1.65)
    assert dialog._branch_policy_combo.currentData() == "stop"
    assert dialog._subpixel_check.isChecked()
    assert dialog._median_check.isChecked()

    reopened = AutoTrackForwardDialog(
        1,
        3,
        app=app,
        seed_anchor=(1, 1),
    )
    qtbot.addWidget(reopened)
    assert reopened.recent_starrynite_parameter_file() == saved_path.resolve()
    assert not reopened._starrynite_recent_button.isHidden()
    assert saved_path.name in reopened._starrynite_recent_button.text()

    reopened._starrynite_recent_button.click()
    assert reopened._starrynite_parameter_path == saved_path.resolve()
    assert reopened._starrynite_recent_button.isHidden()


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
    assert dialog._analysis_thread is None
    assert dialog._accept_button.isEnabled()
    assert len(app._viewer_integration.shown) == 1

    dialog._threshold_spin.setValue(dialog._threshold_spin.value() + 1)
    assert dialog.state == dialog.OUTDATED
    assert not dialog._accept_button.isEnabled()
    assert app._viewer_integration.shown[-1][2]["stale"] is True

    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    assert dialog.state == dialog.READY
    assert dialog._analysis_thread is None
    assert dialog.proposal is not first
    assert len(app.analysis_calls) == 2

    dialog._accept_button.click()
    qtbot.waitUntil(lambda: app._viewer_integration.cleared == 1)
    assert len(app.accept_calls) == 1
    assert app.accept_calls[0][0] is not first
    assert app._viewer_integration.cleared == 1
    assert dialog.state == dialog.APPLIED
    assert not dialog._undo_applied_button.isHidden()
    assert not dialog._save_dataset_button.isHidden()
    assert dialog._discard_button.text().replace("&", "") == "Close"
    dialog.reject()


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
    assert dialog._analysis_thread is None
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


def test_selected_forward_defaults_to_short_basic_tuning_and_next_frame_test(qtbot):
    app = _TrackingDialogApp()
    dialog = AutoTrackForwardDialog(1, 30, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)

    assert dialog._end_spin.value() == 11
    assert not dialog._advanced_toggle.isChecked()
    assert dialog._roi_spin.isHidden()
    assert dialog._distance_spin.isHidden()
    assert dialog._gap_spin.isHidden()
    assert dialog._ambiguity_spin.isHidden()

    dialog._quick_preview_button.click()
    qtbot.waitUntil(
        lambda: dialog._analysis_thread is None
        and dialog.state == dialog.CONFIGURING
    )
    assert app.analysis_calls[-1].scope.end_frame == 2
    assert not dialog._accept_button.isEnabled()
    assert "review-only" in dialog._quick_preview_button.toolTip()

    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    assert app.analysis_calls[-1].scope.end_frame == 11
    assert "Modern StarryNite" in dialog._generated_label.text()
    assert "acetree.starrynite_detector" not in dialog._generated_label.text()
    dialog.reject()


def test_sparse_selected_forward_stage_can_be_overridden_explicitly(
    qtbot,
    tmp_path,
):
    app = _TrackingDialogApp()
    parameter_path = tmp_path / "staged.txt"
    parameter_path.write_text(
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[10,20,30];\n",
        encoding="utf-8",
    )
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)

    dialog.load_starrynite_parameter_file(str(parameter_path))
    assert dialog._threshold_spin.value() == pytest.approx(10.0)
    assert dialog._starrynite_stage_combo.currentData() is None
    assert "26–80 cells" in dialog._starrynite_stage_combo.itemText(2)
    stage_three = dialog._starrynite_stage_combo.findData(2)
    assert stage_three >= 0

    dialog._starrynite_stage_combo.setCurrentIndex(stage_three)
    request = dialog.get_request()

    assert dialog._threshold_spin.value() == pytest.approx(30.0)
    assert request.detector.settings["STARRYNITE_STAGE_INDEX"] == 2
    assert request.detector.settings["STARRYNITE_CELL_COUNT"] == 1
    assert not dialog._starrynite_stage_hint.isHidden()
    dialog.reject()


def test_restore_defaults_reloads_recommended_preset_and_invalidates_draft(qtbot):
    from acetree_py.tracking.starrynite import DEFAULT_BUNDLED_PRESET_ID

    app = _TrackingDialogApp()
    dialog = AutoTrackForwardDialog(1, 20, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    log_index = dialog._workflow_combo.findData("log_lap")
    dialog._workflow_combo.setCurrentIndex(log_index)
    dialog._advanced_toggle.setChecked(True)
    dialog._end_spin.setValue(20)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)

    dialog._restore_defaults()

    assert dialog.state == dialog.OUTDATED
    assert dialog._workflow_combo.currentData() == "modern_starrynite"
    assert dialog._starrynite_preset_combo.currentData() == DEFAULT_BUNDLED_PRESET_ID
    assert dialog._starrynite_profile is not None
    assert dialog._end_spin.value() == 11
    assert not dialog._advanced_toggle.isChecked()
    assert not dialog._accept_button.isEnabled()
    dialog.reject()


def test_restore_defaults_clears_a_pre_preview_failure(qtbot):
    dialog = AutoTrackForwardDialog(1, 20, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog._show_failure("Bad custom settings.", ValueError("broken"))
    assert dialog.state == dialog.FAILED

    dialog._restore_defaults()

    assert dialog.state == dialog.CONFIGURING
    assert "defaults restored" in dialog._banner.text()
    assert dialog._warning_label.isHidden()
    dialog.reject()


def test_native_forward_settings_persist_with_relative_horizon(
    qtbot,
    tmp_path,
    monkeypatch,
):
    from qtpy.QtCore import QSettings

    store = QSettings(str(tmp_path / "selected-forward.ini"), QSettings.IniFormat)
    store.clear()
    monkeypatch.setattr(
        AutoTrackForwardDialog,
        "_settings_store",
        staticmethod(lambda: store),
    )
    first = AutoTrackForwardDialog(2, 40, seed_anchor=(2, 1))
    qtbot.addWidget(first)
    dog_index = first._workflow_combo.findData("dog_lap")
    first._workflow_combo.setCurrentIndex(dog_index)
    first._end_spin.setValue(17)
    first._threshold_spin.setValue(12.5)
    first._roi_spin.setValue(31.0)
    first._advanced_toggle.setChecked(True)
    AutoTrackForwardDialog.persist_native_settings(first.export_settings())

    restored_settings = AutoTrackForwardDialog.persisted_native_settings()
    second = AutoTrackForwardDialog(
        10,
        40,
        seed_anchor=(10, 1),
        initial_settings=restored_settings,
    )
    qtbot.addWidget(second)

    assert second._workflow_combo.currentData() == "dog_lap"
    assert second._end_spin.value() == 25
    assert second._threshold_spin.value() == pytest.approx(12.5)
    assert second._roi_spin.value() == pytest.approx(31.0)
    assert second._advanced_toggle.isChecked()
    first.reject()
    second.reject()


def test_division_stop_offers_direct_follow_both_rerun(qtbot, monkeypatch):
    app = _TrackingDialogApp()
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    request = dialog.get_request()
    seed = Detection("seed", 1, 5.0, 5.0, 1.0, 2.0, 1.0)
    prefix = Detection("prefix", 2, 6.0, 5.0, 1.0, 2.0, 8.0)
    candidates = (
        Detection("daughter-a", 3, 6.5, 4.0, 1.0, 2.0, 7.0),
        Detection("daughter-b", 3, 6.5, 6.0, 1.0, 2.0, 6.0),
    )
    proposal = TrackingResult(
        request=request,
        detections=(seed, prefix),
        edges=(TrackEdge("seed", "prefix", 1.0),),
        existing_anchors={"seed": (1, 1)},
        warnings=("Stopped at t=3: two candidates form a probable division",),
        outcome=TrackingOutcome(
            "division",
            3,
            2,
            (6.5, 5.0, 1.0),
            12.0,
            candidates,
        ),
    )
    dialog._analysis_is_quick = False
    dialog._apply_analysis_succeeded(
        (proposal, app.edit_history.revision, app.edit_history.change_counter)
    )
    reruns = []
    monkeypatch.setattr(
        dialog,
        "_run_preview",
        lambda *, quick=False: reruns.append(quick),
    )

    assert not dialog._follow_both_rerun_button.isHidden()
    assert "Rerun Following Both Daughters" in dialog._warning_label.text()
    dialog._follow_both_rerun_button.click()

    assert dialog._branch_policy_combo.currentData() == "follow_both"
    assert reruns == [False]
    dialog.reject()


def test_accept_through_selected_frame_can_save_then_undo(qtbot, tmp_path):
    app = _CommittingTrackingDialogApp(tmp_path / "dataset.zip")

    def three_frame_analysis(request, **_kwargs):
        seed = Detection("seed", 1, 5.0, 5.0, 1.0, 2.0, 1.0)
        second = Detection("second", 2, 6.0, 5.0, 1.0, 2.0, 10.0)
        third = Detection("third", 3, 7.0, 5.0, 1.0, 2.0, 9.0)
        return (
            TrackingResult(
                request=request,
                detections=(seed, second, third),
                edges=(
                    TrackEdge("seed", "second", 1.0),
                    TrackEdge("second", "third", 1.0),
                ),
                existing_anchors={"seed": (1, 1)},
            ),
            app.edit_history.revision,
        )

    app.analyze_tracking_request = three_frame_analysis
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    dialog._table.setCurrentCell(1, 0)

    assert dialog._accept_through_button.isEnabled()
    assert "t=2" in dialog._accept_through_button.text()
    dialog._accept_through_button.click()

    assert dialog.state == dialog.APPLIED
    assert len(app.accept_calls) == 1
    committed = app.accept_calls[0][0]
    assert committed.request.scope.end_frame == 2
    assert [item.detection_id for item in committed.detections] == ["seed", "second"]
    assert dialog._undo_applied_button.isEnabled()
    assert dialog._save_dataset_button.isEnabled()
    assert len(app.manager.nuclei_record[1]) == 1
    assert len(app.manager.nuclei_record[2]) == 0

    dialog._save_dataset_button.click()
    assert app.save_calls == 1
    assert "dataset.zip" in dialog._banner.text()

    dialog._undo_applied_button.click()
    assert dialog.state == dialog.CONFIGURING
    assert len(app.manager.nuclei_record[1]) == 0
    assert dialog._proposal is None
    assert not dialog._accepted
    assert dialog._accept_button.text().replace("&", "") == "Accept Draft"
    assert dialog._accept_through_button.text() == "Accept through selected frame"
    dialog.reject()


def test_main_history_undo_updates_open_applied_workbench(qtbot, tmp_path):
    app = _CommittingTrackingDialogApp(tmp_path / "dataset.zip")
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    dialog._accept_button.click()
    assert dialog.state == dialog.APPLIED

    app.edit_history.undo()
    dialog.sync_document_revision()

    assert dialog.state == dialog.CONFIGURING
    assert "undone from the main Edit history" in dialog._banner.text()
    assert dialog._proposal is None
    assert not dialog._accepted
    assert dialog._save_dataset_button.isHidden()
    dialog.reject()


def test_undo_button_recovers_after_intervening_edit_is_undone(qtbot, tmp_path):
    app = _CommittingTrackingDialogApp(tmp_path / "dataset.zip")
    dialog = AutoTrackForwardDialog(1, 3, app=app, seed_anchor=(1, 1))
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    qtbot.waitUntil(lambda: dialog.state == dialog.READY)
    dialog._accept_button.click()
    assert dialog._undo_applied_button.isEnabled()

    app.edit_history.do(AddNucleus(time=3, x=2, y=2, z=1.0, size=2))
    dialog.sync_document_revision()
    assert not dialog._undo_applied_button.isEnabled()

    app.edit_history.undo()
    dialog.sync_document_revision()
    assert dialog.state == dialog.APPLIED
    assert dialog._undo_applied_button.isEnabled()
    dialog.reject()
