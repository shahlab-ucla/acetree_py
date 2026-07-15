"""Qt worker used by tracking review workbenches.

Image analysis can take seconds or minutes.  It must not run in the GUI
thread, and progress must cross the thread boundary through queued Qt signals
instead of by manually pumping the event loop.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import Any

from qtpy.QtCore import QObject, Signal, Slot


AnalysisCallable = Callable[
    [Callable[[int, int, str], None], Callable[[], bool]],
    Any,
]


class TrackingAnalysisWorker(QObject):
    """Run one cancellable analysis call outside the GUI thread."""

    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()

    def __init__(
        self,
        analysis: AnalysisCallable,
        cancel_event: Event,
    ) -> None:
        super().__init__()
        self._analysis = analysis
        self._cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        try:
            result = self._analysis(self.progress.emit, self._cancel_event.is_set)
        except Exception as exc:  # delivered to the GUI for human-readable handling
            self.failed.emit(exc)
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()


class TrackingWorkerRelay(QObject):
    """Deliver contextual worker callbacks on the relay's GUI thread."""

    def __init__(
        self,
        run_id: int,
        progress_callback,
        success_callback,
        failure_callback,
    ) -> None:
        super().__init__()
        self._run_id = int(run_id)
        self._progress_callback = progress_callback
        self._success_callback = success_callback
        self._failure_callback = failure_callback

    @Slot(int, int, str)
    def progress(self, done: int, total: int, message: str) -> None:
        self._progress_callback(done, total, message, self._run_id)

    @Slot(object)
    def succeeded(self, payload: object) -> None:
        self._success_callback(payload)

    @Slot(object)
    def failed(self, error: object) -> None:
        self._failure_callback(error, self._run_id)
