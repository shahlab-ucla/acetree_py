"""Focused Qt contracts for global tracking review and wizard channel safety."""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("qtpy")

from acetree_py.gui.dataset_dialog import DatasetCreationDialog
from acetree_py.gui.global_tracking_dialog import GlobalTrackingDialog
from acetree_py.gui.tracking_preview import expand_tracking_preview
from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)
from acetree_py.tracking.registry import TrackingRegistry, build_default_registry
from acetree_py.tracking.starrynite import read_parameter_file


class _ViewerApp:
    def __init__(self) -> None:
        self.current_time = 1
        self.current_plane = 2
        self.current_cell_name = ""
        self.selection_anchor = None
        self.tracking = False
        self.updated = 0

    def set_time(self, value: int) -> None:
        self.current_time = value

    def update_display(self) -> None:
        self.updated += 1


class _PreviewSpy:
    def __init__(self) -> None:
        self.app = _ViewerApp()
        self.shown = []
        self.cleared = 0
        self.visible = True
        self.highlighted = None
        self.detector_shown = []
        self.detector_cleared = 0
        self.detector_visible = True

    def show_tracking_preview(self, proposal, calibration, **options) -> None:
        self.shown.append((proposal, calibration, options))

    def clear_tracking_preview(self) -> None:
        self.cleared += 1

    def set_tracking_preview_visible(self, visible: bool) -> None:
        self.visible = visible

    def highlight_tracking_preview(self, preview_id) -> None:
        self.highlighted = preview_id

    def show_detector_preview(self, detections, calibration, **options) -> None:
        self.detector_shown.append((tuple(detections), calibration, options))

    def clear_detector_preview(self) -> None:
        self.detector_cleared += 1

    def set_detector_preview_visible(self, visible: bool) -> None:
        self.detector_visible = visible


def _global_result(request) -> TrackingResult:
    first = Detection("first", 1, 3.0, 4.0, 1.0, 2.0, 9.0)
    second_root = Detection("second-root", 1, 10.0, 12.0, 1.0, 2.0, 6.0)
    target = Detection("target", 3, 5.0, 4.0, 1.0, 2.0, 8.0)
    return TrackingResult(
        request=request,
        detections=(first, second_root, target),
        edges=(TrackEdge("first", "target", 1.0, kind="gap"),),
    )


def _detector_hits(frame: int = 2) -> tuple[Detection, ...]:
    return (
        Detection("preview-a", frame, 3.0, 4.0, 1.0, 2.0, 9.0),
        Detection("preview-b", frame, 10.0, 12.0, 1.0, 2.0, 6.0),
    )


def test_global_starrynite_threshold_control_maps_to_absolute_legacy_threshold(qtbot):
    registry = build_default_registry(discover_plugins=False)
    dialog = GlobalTrackingDialog(1, 3, registry=registry)
    qtbot.addWidget(dialog)
    detector_index = dialog._detector_combo.findData("acetree.starrynite_detector")
    dialog._detector_combo.setCurrentIndex(detector_index)
    dialog._threshold_spin.setValue(18.0)

    detector = dialog.get_detector_spec()

    assert detector.settings["THRESHOLD"] == 0.0
    assert detector.settings["INTENSITY_THRESHOLD"] == pytest.approx(18.0)
    assert detector.settings["DO_SUBPIXEL_LOCALIZATION"] is False

    dialog._subpixel_check.setChecked(True)
    assert dialog.get_detector_spec().settings["DO_SUBPIXEL_LOCALIZATION"] is True


def test_global_starrynite_parameter_load_save_restore_and_recent(
    qtbot,
    tmp_path,
):
    model_path = tmp_path / "legacy-model.mat"
    model_path.write_bytes(b"legacy classifier provenance")
    parameter_path = tmp_path / "standard-parameters.txt"
    parameter_path.write_text(
        "% preserve this legacy comment\n"
        "firsttimestepnumcells=81;\n"
        "xyres=.25;\n"
        "firsttimestepdiam=40;\n"
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[10,20,30];\n"
        "trackingparameters.temporalcutoff=[2,3,4];\n"
        "trackingparameters.candidateCutoff=1.4;\n"
        "load 'legacy-model.mat';\n",
        encoding="utf-8",
    )
    registry = build_default_registry(discover_plugins=False)
    dialog = GlobalTrackingDialog(1, 5, registry=registry)
    qtbot.addWidget(dialog)

    dialog.load_starrynite_parameter_file(str(parameter_path))
    request = dialog.get_request()

    assert dialog._detector_combo.currentData() == "acetree.starrynite_detector"
    assert dialog._tracker_combo.currentData() == "acetree.starrynite_division"
    assert dialog._radius_spin.value() == pytest.approx(5.0)
    assert dialog._threshold_spin.value() == pytest.approx(30.0)
    assert dialog._gap_spin.value() == 3
    assert not dialog._subpixel_check.isChecked()
    assert dialog._division_check.isChecked()
    assert dialog._starrynite_save_button.isEnabled()
    assert not dialog._starrynite_behavior_label.isHidden()
    assert "provenance only" in dialog._starrynite_behavior_label.text()
    assert "native geometry scorer" in dialog._starrynite_behavior_label.text()
    assert "provenance-only" in dialog._starrynite_file_label.text()
    assert request.detector.settings["STARRYNITE_STAGE_INDEX"] == 2
    assert request.detector.settings["STARRYNITE_CELL_COUNT"] == 81
    assert request.detector.settings["INTENSITY_THRESHOLD"] == pytest.approx(30.0)
    assert request.tracker.settings["CANDIDATE_CUTOFF"] == pytest.approx(1.4)
    assert request.tracker.settings["STARRYNITE_MODEL_FILE"] == str(
        model_path.resolve()
    )
    assert len(request.tracker.settings["STARRYNITE_MODEL_SHA256"]) == 64

    dialog._radius_spin.setValue(6.0)
    dialog._threshold_spin.setValue(33.0)
    dialog._gap_spin.setValue(2)
    dialog._distance_spin.setValue(13.5)
    dialog._division_check.setChecked(False)
    dialog._subpixel_check.setChecked(True)
    dialog._median_check.setChecked(True)
    saved_path = tmp_path / "standard-parameters-tuned.txt"
    assert dialog.save_starrynite_parameter_file(str(saved_path)) == ()
    saved = read_parameter_file(saved_path)

    assert saved.source.startswith(read_parameter_file(parameter_path).source)
    assert saved.normalized_settings["firsttimestepdiam"] == pytest.approx(48.0)
    assert saved.normalized_settings["parameters.intensitythreshold"] == (
        10,
        20,
        33.0,
    )
    assert saved.normalized_settings["trackingparameters.temporalcutoff"] == (
        2,
        3,
        3,
    )
    assert dialog.recent_starrynite_parameter_file() == saved_path.resolve()
    assert dialog._distance_spin.value() == pytest.approx(13.5)
    assert not dialog._division_check.isChecked()
    assert dialog._subpixel_check.isChecked()
    assert dialog._median_check.isChecked()
    saved_request = dialog.get_request()
    assert saved_request.detector.settings["STARRYNITE_PARAMETER_FILE"] == str(
        saved_path.resolve()
    )

    restored = GlobalTrackingDialog(
        1,
        5,
        registry=registry,
        initial_request=saved_request,
    )
    qtbot.addWidget(restored)
    assert restored._starrynite_parameter_path == saved_path.resolve()
    assert restored._threshold_spin.value() == pytest.approx(33.0)
    assert restored.get_request().tracker.settings["CANDIDATE_CUTOFF"] == (
        pytest.approx(1.4)
    )

    recent = GlobalTrackingDialog(1, 5, registry=registry)
    qtbot.addWidget(recent)
    assert recent.recent_starrynite_parameter_file() == saved_path.resolve()
    assert not recent._starrynite_recent_button.isHidden()
    recent._starrynite_recent_button.click()
    assert recent._starrynite_parameter_path == saved_path.resolve()
    assert recent._starrynite_recent_button.isHidden()


def test_exact_tracker_interactive_selection_uses_full_movie_and_restore_fails_closed(
    qtbot,
) -> None:
    registry = build_default_registry(discover_plugins=False)
    dialog = GlobalTrackingDialog(1, 5, registry=registry)
    qtbot.addWidget(dialog)
    dialog._start_spin.setValue(2)
    dialog._end_spin.setValue(4)

    dialog._tracker_combo.setCurrentIndex(
        dialog._tracker_combo.findData("acetree.starrynite_legacy_exact")
    )

    assert dialog._detector_combo.currentData() == "acetree.starrynite_detector"
    assert dialog._start_spin.value() == 1
    assert dialog._end_spin.value() == 5

    partial_request = TrackingRequest(
        detector=ComponentSpec("acetree.starrynite_detector", {}),
        tracker=ComponentSpec(
            "acetree.starrynite_legacy_exact",
            {"STARRYNITE_COMPATIBILITY_MODE": "legacy_exact_refinement"},
        ),
        scope=TrackingScope("global", 1, 4),
    )
    restored = GlobalTrackingDialog(
        1,
        5,
        registry=registry,
        initial_request=partial_request,
    )
    qtbot.addWidget(restored)

    error = restored._settings_validation_error()
    assert "complete movie" in error
    assert "t=1–5" in error
    assert not restored._preview_button.isEnabled()


def test_global_initial_request_rebases_profile_metadata_and_warns_on_calibration(
    qtbot,
    tmp_path,
) -> None:
    model_path = tmp_path / "legacy-model.mat"
    model_path.write_bytes(b"initial model")
    parameter_path = tmp_path / "rebase-parameters.m"
    parameter_path.write_text(
        "firsttimestepnumcells=30;\n"
        "xyres=.25;\n"
        "zres=1;\n"
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[5,8,12];\n"
        "trackingparameters.candidateCutoff=1.4;\n"
        "load 'legacy-model.mat';\n",
        encoding="utf-8",
    )
    registry = build_default_registry(discover_plugins=False)
    original = GlobalTrackingDialog(1, 3, registry=registry)
    qtbot.addWidget(original)
    original.load_starrynite_parameter_file(str(parameter_path))
    original_request = original.get_request()

    parameter_path.write_text(
        "firsttimestepnumcells=90;\n"
        "xyres=.25;\n"
        "zres=1;\n"
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[15,22,33];\n"
        "trackingparameters.candidateCutoff=2.2;\n"
        "load 'legacy-model.mat';\n",
        encoding="utf-8",
    )
    model_path.write_bytes(b"changed model")

    restored = GlobalTrackingDialog(
        1,
        3,
        registry=registry,
        initial_request=original_request,
        calibration=Calibration(1.0, 2.0),
    )
    qtbot.addWidget(restored)
    request = restored.get_request()

    assert restored._threshold_spin.value() == pytest.approx(8.0)
    assert request.detector.settings["STARRYNITE_CELL_COUNT"] == 90
    assert request.detector.settings["STARRYNITE_STAGE_INDEX"] == 2
    assert request.detector.settings["STARRYNITE_PARAMETER_SHA256"] == (
        hashlib.sha256(parameter_path.read_bytes()).hexdigest()
    )
    assert request.tracker.settings["STARRYNITE_MODEL_SHA256"] == (
        hashlib.sha256(model_path.read_bytes()).hexdigest()
    )
    assert request.tracker.settings["CANDIDATE_CUTOFF"] == pytest.approx(2.2)
    assert "Source/model changed" in restored._starrynite_file_label.text()
    assert "dataset uses" in restored._starrynite_file_label.toolTip()


def test_global_review_surfaces_division_control_and_event_counts(qtbot) -> None:
    dialog = GlobalTrackingDialog(
        1,
        2,
        registry=build_default_registry(discover_plugins=False),
    )
    qtbot.addWidget(dialog)
    parent = Detection("parent", 1, 3.0, 4.0, 1.0, 2.0, 9.0)
    first = Detection("daughter-a", 2, 2.5, 4.0, 1.0, 2.0, 8.0)
    second = Detection("daughter-b", 2, 3.5, 4.0, 1.0, 2.0, 8.0)
    result = TrackingResult(
        request=dialog.get_request(),
        detections=(parent, first, second),
        edges=(
            TrackEdge("parent", "daughter-a", 1.0, kind="split"),
            TrackEdge("parent", "daughter-b", 1.0, kind="split"),
        ),
    )
    dialog._proposal = result
    dialog._expanded_preview = expand_tracking_preview(result)
    dialog._populate_review()

    assert not dialog._advanced_widget.isAncestorOf(dialog._division_check)
    assert "1 proposed division" in dialog._summary_label.text()
    assert "1 proposed division" in dialog._table.item(0, 6).text()
    assert "forked path" in dialog._legend_label.text()


def test_global_workbench_uses_host_slots_and_reviews_every_frame(qtbot):
    preview = _PreviewSpy()
    token = [(4, 0)]
    dialog = GlobalTrackingDialog(
        1,
        3,
        num_channels=2,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        revision_getter=lambda: token[0],
    )
    qtbot.addWidget(dialog)
    requested = []
    dialog.analysisRequested.connect(lambda request, run_id: requested.append((request, run_id)))

    dialog._preview_button.click()

    assert dialog.state == dialog.RUNNING
    assert len(requested) == 1
    request, run_id = requested[0]
    assert request.scope.kind == "global"
    assert request.detector.settings["TARGET_CHANNEL"] == 1
    assert dialog.update_analysis_progress(2, 3, "Detecting t=2", run_id)

    assert dialog.finish_analysis(
        _global_result(request),
        expected_revision=4,
        run_id=run_id,
        document_token=token[0],
    )

    assert dialog.state == dialog.READY
    assert dialog._accept_button.isEnabled()
    assert dialog._table.rowCount() == 3
    # The middle frame has no detector endpoint but will receive the exact
    # interpolated position that acceptance materializes.
    assert dialog._table.item(1, 1).text() == "0"
    assert dialog._table.item(1, 2).text() == "1"
    assert "interpolated" in dialog._table.item(1, 6).text()
    assert "3 detected spots" in dialog._summary_label.text()
    assert preview.shown[-1][2]["stale"] is False

    dialog._table.cellClicked.emit(1, 0)
    assert preview.app.current_time == 2
    assert preview.highlighted is not None

    dialog._threshold_spin.setValue(dialog._threshold_spin.value() + 1)
    assert dialog.state == dialog.OUTDATED
    assert not dialog._accept_button.isEnabled()
    assert preview.shown[-1][2]["stale"] is True
    dialog.reject()


def test_global_workbench_cancel_and_safe_close_ignore_late_worker_result(qtbot):
    preview = _PreviewSpy()
    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
    )
    qtbot.addWidget(dialog)
    requested = []
    canceled = []
    dialog.analysisRequested.connect(lambda request, run_id: requested.append((request, run_id)))
    dialog.cancelRequested.connect(canceled.append)

    dialog._preview_button.click()
    request, run_id = requested[0]
    dialog._cancel_run_button.click()
    assert dialog.state == dialog.CANCELING
    assert canceled == [run_id]
    assert dialog.analysis_cancelled(run_id)
    assert dialog.state == dialog.CONFIGURING
    assert dialog.proposal is None

    dialog._preview_button.click()
    request, run_id = requested[-1]
    dialog.reject()
    assert canceled[-1] == run_id
    assert preview.cleared == 1
    assert not dialog.finish_analysis(_global_result(request), 0, run_id)


def test_result_delivery_uses_live_document_token(qtbot):
    token = [(3, 0)]
    dialog = GlobalTrackingDialog(
        1,
        3,
        revision_getter=lambda: token[0],
    )
    qtbot.addWidget(dialog)
    requested = []
    dialog.analysisRequested.connect(
        lambda request, run_id: requested.append((request, run_id))
    )

    dialog._preview_button.click()
    request, run_id = requested[0]
    token[0] = (3, 1)

    assert dialog.finish_analysis(
        _global_result(request),
        expected_revision=3,
        run_id=run_id,
        document_token=(3, 0),
    )
    assert dialog.state == dialog.STALE
    assert not dialog._accept_button.isEnabled()
    dialog.reject()


def test_global_workbench_stale_guard_and_explicit_accept(qtbot):
    preview = _PreviewSpy()
    token = [(7, 0)]
    accepted = []

    def starter(request, _run_id, _dialog):
        return _global_result(request), 7

    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        analysis_starter=starter,
        accept_callback=lambda proposal, revision: accepted.append((proposal, revision)),
        revision_getter=lambda: token[0],
    )
    qtbot.addWidget(dialog)

    dialog._preview_button.click()
    assert dialog.state == dialog.READY
    token[0] = (7, 1)
    assert not dialog.sync_document_revision()
    assert dialog.state == dialog.STALE
    assert not dialog._accept_button.isEnabled()

    dialog._preview_button.click()
    assert dialog.state == dialog.READY
    dialog._accept_button.click()

    assert len(accepted) == 1
    assert accepted[0][1] == 7
    assert preview.cleared == 1


def test_current_frame_detector_test_is_independent_and_never_accept_capable(qtbot):
    preview = _PreviewSpy()
    preview.app.current_time = 2
    registry = build_default_registry(discover_plugins=False)
    registry._trackers.clear()
    started = []

    def detector_starter(detector, frame, run_id, _dialog):
        started.append((detector, frame, run_id))
        return (_detector_hits(frame),)

    dialog = GlobalTrackingDialog(
        1,
        3,
        registry=registry,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        detector_preview_starter=detector_starter,
        current_frame_getter=lambda: preview.app.current_time,
    )
    qtbot.addWidget(dialog)
    dialog._start_spin.setValue(3)
    dialog._end_spin.setValue(1)

    assert not dialog._preview_button.isEnabled()
    assert dialog._detector_preview_button.isEnabled()
    dialog._detector_preview_button.click()

    assert started[0][1] == 2
    assert started[0][0].settings["TARGET_CHANNEL"] == 1
    assert dialog.state == dialog.DETECTOR_READY
    assert dialog.proposal is None
    assert dialog._table.rowCount() == 0
    assert not dialog._accept_button.isEnabled()
    assert len(preview.detector_shown) == 1
    assert len(preview.detector_shown[0][0]) == 2
    assert "cannot be accepted" in dialog._detector_status_label.text()

    cleared = preview.detector_cleared
    dialog._distance_spin.setValue(dialog._distance_spin.value() + 1)
    assert preview.detector_cleared == cleared
    dialog._threshold_spin.setValue(dialog._threshold_spin.value() + 1)
    assert preview.detector_cleared == cleared + 1
    assert dialog.state == dialog.CONFIGURING
    dialog.reject()


def test_detector_test_ignores_late_result_after_viewer_moves(qtbot):
    preview = _PreviewSpy()
    preview.app.current_time = 2
    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        current_frame_getter=lambda: preview.app.current_time,
    )
    qtbot.addWidget(dialog)
    requested = []
    dialog.detectorPreviewRequested.connect(
        lambda detector, frame, run_id: requested.append((detector, frame, run_id))
    )

    dialog._detector_preview_button.click()
    _detector, frame, run_id = requested[0]
    preview.app.current_time = 3
    dialog.sync_viewer_position(3)

    assert not dialog.finish_detector_preview(_detector_hits(frame), frame, run_id)
    assert dialog.proposal is None
    assert not preview.detector_shown
    assert "late result was ignored" in dialog._banner.text()
    dialog.reject()


def test_completed_detector_test_clears_when_viewer_moves(qtbot):
    preview = _PreviewSpy()
    preview.app.current_time = 2
    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        current_frame_getter=lambda: preview.app.current_time,
        detector_preview_starter=lambda _detector, frame, _run_id, _dialog: (
            _detector_hits(frame),
        ),
    )
    qtbot.addWidget(dialog)

    dialog._detector_preview_button.click()
    assert dialog.state == dialog.DETECTOR_READY
    assert len(preview.detector_shown) == 1

    preview.app.current_time = 3
    dialog.sync_viewer_position(3)

    assert dialog.state == dialog.CONFIGURING
    assert preview.detector_cleared >= 1
    assert "overlay was cleared" in dialog._detector_status_label.text()
    dialog.reject()


def test_detector_test_rejects_document_and_setting_races(qtbot):
    preview = _PreviewSpy()
    preview.app.current_time = 2
    token = [(4, 0)]
    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        current_frame_getter=lambda: preview.app.current_time,
        revision_getter=lambda: token[0],
    )
    qtbot.addWidget(dialog)
    requested = []
    dialog.detectorPreviewRequested.connect(
        lambda detector, frame, run_id: requested.append((detector, frame, run_id))
    )

    dialog._detector_preview_button.click()
    _detector, frame, run_id = requested[-1]
    token[0] = (4, 1)
    assert not dialog.finish_detector_preview(_detector_hits(frame), frame, run_id)
    assert not preview.detector_shown
    assert "dataset changed" in dialog._banner.text()

    token[0] = (4, 1)
    dialog._detector_preview_button.click()
    _detector, frame, run_id = requested[-1]
    dialog._threshold_spin.setValue(dialog._threshold_spin.value() + 1)
    assert not dialog.finish_detector_preview(_detector_hits(frame), frame, run_id)
    assert not preview.detector_shown
    assert "settings changed" in dialog._banner.text()
    dialog.reject()


def test_detector_cancel_and_close_ignore_late_results(qtbot):
    preview = _PreviewSpy()
    preview.app.current_time = 2
    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        current_frame_getter=lambda: preview.app.current_time,
    )
    qtbot.addWidget(dialog)
    requested = []
    canceled = []
    dialog.detectorPreviewRequested.connect(
        lambda detector, frame, run_id: requested.append((detector, frame, run_id))
    )
    dialog.cancelRequested.connect(canceled.append)

    dialog._detector_preview_button.click()
    _detector, frame, run_id = requested[-1]
    dialog._cancel_run_button.click()
    assert dialog.analysis_cancelled(run_id)
    assert canceled == [run_id]
    assert not dialog.finish_detector_preview(_detector_hits(frame), frame, run_id)
    assert not preview.detector_shown

    dialog._detector_preview_button.click()
    _detector, frame, run_id = requested[-1]
    dialog.reject()
    assert canceled[-1] == run_id
    assert not dialog.finish_detector_preview(_detector_hits(frame), frame, run_id)
    assert not preview.detector_shown


def test_whole_dataset_draft_rechecks_empty_record_before_accept(qtbot):
    preview = _PreviewSpy()
    empty = [True]

    def starter(request, _run_id, _dialog):
        return _global_result(request), 7

    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        analysis_starter=starter,
        revision_getter=lambda: (7, 0),
        dataset_empty_getter=lambda: empty[0],
    )
    qtbot.addWidget(dialog)
    dialog._preview_button.click()
    assert dialog.state == dialog.READY

    empty[0] = False
    assert not dialog.sync_document_revision()
    assert dialog.state == dialog.STALE
    assert not dialog._accept_button.isEnabled()
    assert not dialog._preview_button.isEnabled()

    empty[0] = True
    assert not dialog.sync_document_revision()
    assert dialog.state == dialog.STALE
    assert dialog._preview_button.isEnabled()
    assert not dialog._accept_button.isEnabled()
    assert not dialog._settings_error.text()
    dialog.reject()


def test_whole_dataset_workbench_recovers_after_record_is_emptied_again(qtbot):
    empty = [True]
    dialog = GlobalTrackingDialog(
        1,
        3,
        dataset_empty_getter=lambda: empty[0],
    )
    qtbot.addWidget(dialog)

    empty[0] = False
    assert not dialog.sync_document_revision()
    assert dialog.state == dialog.STALE
    assert not dialog._preview_button.isEnabled()

    empty[0] = True
    assert dialog.sync_document_revision()
    assert dialog.state == dialog.CONFIGURING
    assert dialog._preview_button.isEnabled()
    assert not dialog._settings_error.text()
    dialog.reject()


def test_detector_result_stays_noncommittable_if_curated_positions_appear(qtbot):
    preview = _PreviewSpy()
    preview.app.current_time = 2
    empty = [True]
    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        current_frame_getter=lambda: preview.app.current_time,
        dataset_empty_getter=lambda: empty[0],
    )
    qtbot.addWidget(dialog)
    requested = []
    dialog.detectorPreviewRequested.connect(
        lambda detector, frame, run_id: requested.append((detector, frame, run_id))
    )

    dialog._detector_preview_button.click()
    _detector, frame, run_id = requested[0]
    empty[0] = False
    assert dialog.finish_detector_preview(_detector_hits(frame), frame, run_id)

    assert dialog.state == dialog.STALE
    assert len(preview.detector_shown) == 1
    assert not dialog._preview_button.isEnabled()
    assert not dialog._accept_button.isEnabled()
    assert "Whole-dataset tracking is disabled" in dialog._banner.text()
    dialog.reject()


def test_whole_dataset_draft_rechecks_empty_record_on_worker_delivery(qtbot):
    preview = _PreviewSpy()
    empty = [True]
    dialog = GlobalTrackingDialog(
        1,
        3,
        viewer_integration=preview,
        calibration=Calibration(1.0, 1.0),
        dataset_empty_getter=lambda: empty[0],
    )
    qtbot.addWidget(dialog)
    requested = []
    dialog.analysisRequested.connect(
        lambda request, run_id: requested.append((request, run_id))
    )

    dialog._preview_button.click()
    request, run_id = requested[0]
    empty[0] = False
    assert dialog.finish_analysis(_global_result(request), 7, run_id)

    assert dialog.state == dialog.STALE
    assert not dialog._accept_button.isEnabled()
    assert preview.shown[-1][2]["stale"] is True
    assert "added while analysis was running" in dialog._banner.text()
    dialog.reject()


def test_global_workbench_blocks_missing_plugins(qtbot):
    dialog = GlobalTrackingDialog(1, 3, registry=TrackingRegistry())
    qtbot.addWidget(dialog)

    assert not dialog._preview_button.isEnabled()
    assert "No compatible detector" in dialog._settings_error.text()
    with pytest.raises(ValueError, match="No compatible detector"):
        dialog.get_request()
    dialog.reject()


def test_dataset_wizard_tracking_channel_follows_layout_immediately(qtbot):
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    dialog._radio_tracking_auto.setChecked(True)

    assert dialog._tracking_channel_spin.maximum() == 1

    dialog._radio_split.setChecked(True)
    assert dialog._tracking_channel_spin.maximum() == 2
    dialog._tracking_channel_spin.setValue(2)
    assert dialog.get_tracking_request().detector.settings["TARGET_CHANNEL"] == 2

    dialog._radio_multistack.setChecked(True)
    dialog._n_channels_spin.setValue(4)
    assert dialog._tracking_channel_spin.maximum() == 4
    dialog._tracking_channel_spin.setValue(4)

    dialog._radio_single.setChecked(True)
    assert dialog._tracking_channel_spin.maximum() == 1
    assert dialog._tracking_channel_spin.value() == 1
    assert "review workbench" in dialog._tracking_explanation_label.text()
    assert "Undo" not in dialog._tracking_explanation_label.text()


def test_dataset_wizard_blocks_unusable_multichannel_layout(qtbot):
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    dialog._detected = {"num_planes": 5}
    dialog._radio_multistack.setChecked(True)
    dialog._n_channels_spin.setValue(2)
    dialog._radio_tracking_auto.setChecked(True)
    dialog._stack.setCurrentWidget(dialog._page4)
    dialog._refresh_tracking_validation()

    assert "cannot be divided evenly" in dialog._tracking_validation_label.text()
    assert not dialog._btn_next.isEnabled()
    with pytest.raises(ValueError, match="cannot be divided evenly"):
        dialog.get_tracking_request()

    dialog._n_channels_spin.setValue(5)
    assert not dialog._tracking_validation_label.isVisible()
    assert dialog._btn_next.isEnabled()
