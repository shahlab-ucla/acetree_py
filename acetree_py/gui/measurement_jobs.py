"""One cancellable measurement job per application, with GUI-only publication."""

from __future__ import annotations

from threading import Event
from typing import Callable

from qtpy.QtCore import QEvent, QObject, QThread, QTimer, Qt, Slot
from qtpy.QtWidgets import QApplication, QProgressDialog

from .tracking_worker import TrackingAnalysisWorker


_NO_RESULT = object()


class MeasurementJobs(QObject):
    """Own ROI/nuclear progress, cancellation, publication, and Qt teardown."""

    def __init__(self, window, report: Callable[[str], None]) -> None:
        super().__init__()
        self._window = window
        self._report = report
        self._thread = None
        self._worker = None
        self._cancel = Event()
        self._progress = None
        self._publish = None
        self._discard = None
        self._closing = False
        self._pending_result = _NO_RESULT
        if window is not None:
            window.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    @property
    def active(self) -> bool:
        return self._thread is not None

    def start(self, title: str, compute, publish, *, discard=None) -> bool:
        if self.active or self._closing:
            self._report("A measurement is already running; cancel it or wait for completion")
            return False
        self._cancel = Event()
        self._publish = publish
        self._discard = discard
        self._progress = QProgressDialog("Preparing measurement…", "Cancel", 0, 0, self._window)
        self._progress.setWindowTitle(title)
        self._progress.setWindowModality(Qt.NonModal)
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.setAutoReset(False)
        self._progress.canceled.connect(self.cancel)
        self._progress.show()
        def run(progress, cancelled):
            result = compute(progress, cancelled)
            if cancelled():
                if discard is not None:
                    discard(result)
                raise RuntimeError("Measurement cancelled")
            self._pending_result = result
            return result

        self._thread = QThread()
        self._worker = TrackingAnalysisWorker(run, self._cancel)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()
        return True

    @Slot()
    def cancel(self) -> None:
        self._cancel.set()
        if self._progress is not None:
            self._progress.setLabelText("Cancelling measurement…")

    @Slot(int, int, str)
    def _on_progress(self, done: int, total: int, message: str) -> None:
        if self._progress is not None and not self._cancel.is_set():
            self._progress.setMaximum(max(1, total))
            self._progress.setValue(done)
            self._progress.setLabelText(message)

    @Slot(object)
    def _on_success(self, result) -> None:
        if self._pending_result is _NO_RESULT:
            return
        self._pending_result = _NO_RESULT
        try:
            if self._cancel.is_set() or self._closing:
                self._report("Measurement cancelled; previous results retained")
            else:
                self._publish(result)
        except Exception as error:
            self._report(f"Measurement failed: {error}")
        finally:
            if self._discard is not None:
                try:
                    self._discard(result)
                except Exception as error:
                    self._report(f"Could not remove temporary measurement files: {error}")

    @Slot(object)
    def _on_failure(self, error) -> None:
        if self._cancel.is_set():
            self._report("Measurement cancelled; previous results retained")
        else:
            self._report(f"Measurement failed: {error}")

    @Slot()
    def _on_finished(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress.deleteLater()
            self._progress = None

    @Slot()
    def _on_thread_finished(self) -> None:
        # Keep wrappers alive until Qt has finished delivering this signal.
        QTimer.singleShot(0, self._release)

    def _release(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        self._publish = None
        self._discard = None
        if thread is not None:
            thread.deleteLater()

    def eventFilter(self, watched, event) -> bool:
        if watched is self._window and event.type() == QEvent.Close:
            self._closing = True
            self.cancel()
            QTimer.singleShot(0, self._check_window_closed)
        return super().eventFilter(watched, event)

    def _check_window_closed(self) -> None:
        try:
            self._closing = not self._window.isVisible()
        except RuntimeError:
            self._closing = True

    @Slot()
    def shutdown(self, timeout_ms: int = 1500) -> None:
        self._closing = True
        self.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(max(0, timeout_ms))
            if self._pending_result is not _NO_RESULT:
                self._on_success(self._pending_result)
