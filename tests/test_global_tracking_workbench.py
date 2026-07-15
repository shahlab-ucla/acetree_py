"""Focused Qt contracts for global tracking review and wizard channel safety."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("qtpy")

from acetree_py.gui.dataset_dialog import DatasetCreationDialog
from acetree_py.gui.global_tracking_dialog import GlobalTrackingDialog
from acetree_py.tracking.api import (
    Calibration,
    Detection,
    TrackEdge,
    TrackingResult,
)
from acetree_py.tracking.registry import TrackingRegistry, build_default_registry


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
