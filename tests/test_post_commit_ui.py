"""Regression tests for the edit/UI post-commit boundary."""

from __future__ import annotations

import logging

import pytest

from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import Nucleus
from acetree_py.editing.commands import AddNucleus, MoveNucleus
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.config import AceTreeConfig
from acetree_py.tracking.api import (
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)
from acetree_py.tracking.integration import TrackingProposalConflict


def _empty_app(num_timepoints: int = 2) -> AceTreeApp:
    config = AceTreeConfig(
        starting_index=1,
        ending_index=num_timepoints,
        xy_res=1.0,
        z_res=1.0,
        plane_start=1,
        plane_end=8,
    )
    return AceTreeApp(NucleiManager.new_empty(config, num_timepoints))


def _proposal(*, invalid_anchor: bool = False) -> TrackingResult:
    request = TrackingRequest(
        detector=ComponentSpec("test.detector"),
        tracker=ComponentSpec("test.tracker"),
        scope=TrackingScope("global", 1, 2),
    )
    first = Detection("d1", 1, 10.0, 20.0, 0.0, 3.0, 9.0)
    second = Detection("d2", 2, 11.0, 20.0, 0.0, 3.0, 8.0)
    return TrackingResult(
        request=request,
        detections=(first, second),
        edges=(TrackEdge("d1", "d2", 1.0),),
        existing_anchors={"d1": (1, 99)} if invalid_anchor else {},
        provenance={"run_id": "post-commit-regression"},
    )


def test_tracking_display_failure_commits_once_retries_redraw_and_returns_mapping(
    monkeypatch,
    caplog,
):
    app = _empty_app()
    result = _proposal()
    messages: list[str] = []
    redraw_calls = 0

    def flaky_redraw() -> None:
        nonlocal redraw_calls
        redraw_calls += 1
        if redraw_calls == 1:
            raise RuntimeError("layer replacement failed")

    monkeypatch.setattr(app, "update_display", flaky_redraw)
    monkeypatch.setattr(app, "_say", messages.append)
    initial_revision = app.edit_history.revision
    initial_changes = app.edit_history.change_counter

    with caplog.at_level(logging.WARNING):
        mapping = app.accept_tracking_proposal(
            result,
            expected_revision=initial_revision,
        )

    assert mapping == {"d1": (1, 1), "d2": (2, 1)}
    assert [len(frame) for frame in app.manager.nuclei_record] == [1, 1]
    assert app.manager.nuclei_record[0][0].successor1 == 1
    assert app.manager.nuclei_record[1][0].predecessor == 1
    assert app.edit_history.revision != initial_revision
    assert app.edit_history.change_counter == initial_changes + 1
    assert app.edit_history.num_undoable == 1
    assert app.edit_history.num_redoable == 0
    assert app._tracking_results == [result]
    assert redraw_calls == 2
    assert messages and "committed and undoable" in messages[-1]
    assert "Post-commit refresh failed" in caplog.text


def test_tracking_callback_failure_rebuilds_once_without_reapplying(monkeypatch):
    app = _empty_app()
    result = _proposal()
    messages: list[str] = []
    process_calls = 0
    real_process = app.manager.process

    def flaky_process(*args, **kwargs) -> None:
        nonlocal process_calls
        process_calls += 1
        if process_calls == 1:
            raise RuntimeError("temporary lineage rebuild failure")
        real_process(*args, **kwargs)

    monkeypatch.setattr(app.manager, "process", flaky_process)
    monkeypatch.setattr(app, "_say", messages.append)
    initial_revision = app.edit_history.revision

    mapping = app.accept_tracking_proposal(
        result,
        expected_revision=initial_revision,
    )

    assert mapping == {"d1": (1, 1), "d2": (2, 1)}
    assert process_calls == 2
    assert [len(frame) for frame in app.manager.nuclei_record] == [1, 1]
    assert app.edit_history.num_undoable == 1
    assert app.edit_history.change_counter == 1
    assert app._tracking_results == [result]
    assert messages and "committed and undoable" in messages[0]


def test_interactive_edit_retries_only_post_commit_observer(monkeypatch):
    app = _empty_app(num_timepoints=1)
    messages: list[str] = []
    process_calls = 0
    real_process = app.manager.process

    def flaky_process(*args, **kwargs) -> None:
        nonlocal process_calls
        process_calls += 1
        if process_calls == 1:
            raise RuntimeError("temporary manual-edit rebuild failure")
        real_process(*args, **kwargs)

    monkeypatch.setattr(app.manager, "process", flaky_process)
    monkeypatch.setattr(app, "_say", messages.append)
    command = AddNucleus(time=1, x=12, y=20, z=3.0, size=8)

    app._run_edit_action(app.edit_history.do, command)

    assert process_calls == 2
    assert len(app.manager.nuclei_record[0]) == 1
    assert app.manager.nuclei_record[0][0].x == 12
    assert app.edit_history.change_counter == 1
    assert app.edit_history.num_undoable == 1
    assert messages and "committed and undoable" in messages[0]


def test_safe_wrapper_returns_committed_undo_and_redo_command(monkeypatch):
    app = _empty_app(num_timepoints=1)
    command = AddNucleus(time=1, x=12, y=20, z=3.0, size=8)
    app.edit_history.do(command)
    real_process = app.manager.process
    fail_next = True

    def flaky_process(*args, **kwargs) -> None:
        nonlocal fail_next
        if fail_next:
            fail_next = False
            raise RuntimeError("temporary history observer failure")
        real_process(*args, **kwargs)

    monkeypatch.setattr(app.manager, "process", flaky_process)

    undone = app._run_edit_action(app.edit_history.undo)
    assert undone is command
    assert app.edit_history.num_undoable == 0
    assert app.edit_history.num_redoable == 1
    assert app.edit_history.change_counter == 2

    fail_next = True
    redone = app._run_edit_action(app.edit_history.redo)
    assert redone is command
    assert app.edit_history.num_undoable == 1
    assert app.edit_history.num_redoable == 0
    assert app.edit_history.change_counter == 3
    assert len(app.manager.nuclei_record[0]) == 1


def test_detached_redraw_failure_is_not_mistaken_for_deleted_window():
    app = _empty_app(num_timepoints=1)

    class RedrawFailure:
        @staticmethod
        def isVisible() -> bool:
            return True

        @staticmethod
        def refresh() -> None:
            raise RuntimeError("Centroid marker redraw failed after rollback")

    app._3d_windows = [RedrawFailure()]

    with pytest.raises(RuntimeError, match="Centroid marker redraw failed"):
        app.update_display()


def test_deleted_detached_qt_window_is_still_ignored():
    app = _empty_app(num_timepoints=1)

    class DeletedWindow:
        @staticmethod
        def isVisible() -> bool:
            raise RuntimeError("wrapped C/C++ object has been deleted")

    app._3d_windows = [DeletedWindow()]
    app.update_display()


def test_tracking_command_conflict_still_propagates_without_commit(monkeypatch):
    app = _empty_app()
    result = _proposal(invalid_anchor=True)
    messages: list[str] = []
    initial_revision = app.edit_history.revision
    initial_changes = app.edit_history.change_counter

    monkeypatch.setattr(app, "_say", messages.append)

    with pytest.raises(TrackingProposalConflict):
        app.accept_tracking_proposal(
            result,
            expected_revision=initial_revision,
        )

    assert app.manager.nuclei_record == [[], []]
    assert app.edit_history.revision == initial_revision
    assert app.edit_history.change_counter == initial_changes
    assert app.edit_history.num_undoable == 0
    assert app._tracking_results == []
    assert messages == []


def test_manual_move_display_failure_stays_committed_undoable_and_warns(
    monkeypatch,
    caplog,
):
    app = _empty_app(num_timepoints=1)
    nucleus = Nucleus(
        index=1,
        x=10,
        y=20,
        z=3.0,
        size=8,
        identity="P0",
        assigned_id="P0",
        status=1,
    )
    app.manager.nuclei_record[0].append(nucleus)
    app.manager.process()
    messages: list[str] = []

    def broken_redraw() -> None:
        raise RuntimeError("napari redraw failed")

    monkeypatch.setattr(app, "update_display", broken_redraw)
    monkeypatch.setattr(app, "_say", messages.append)
    initial_revision = app.edit_history.revision

    with caplog.at_level(logging.WARNING):
        app.edit_history.do(MoveNucleus(time=1, index=1, new_x=42))

    assert nucleus.x == 42
    assert app.edit_history.revision != initial_revision
    assert app.edit_history.num_undoable == 1
    assert app.edit_history.modified
    assert messages and "committed and undoable" in messages[-1]
    assert "Post-commit refresh failed" in caplog.text

    app.edit_history.undo()

    assert nucleus.x == 10
    assert app.edit_history.revision == initial_revision
    assert app.edit_history.num_undoable == 0
    assert app.edit_history.num_redoable == 1
