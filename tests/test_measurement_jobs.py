"""Outcome-level checks for cancellable measurement and GUI publication."""

from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QWidget

from acetree_py.analysis import roi_measurement_job
from acetree_py.analysis.roi_measurements import RoiMeasurementRequest
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.roi_manager import RoiManager
from acetree_py.core.subcellular_roi import Polygon2D
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.image_provider import NumpyProvider, StackTiffProvider, close_worker_image_provider
from tests.test_roi_cache import _mixed_movie_document, polygon


@pytest.mark.parametrize("outcome", ["success", "cancel", "edit", "images", "close"])
def test_roi_job_is_responsive_and_publishes_only_current_results(qtbot, tmp_path, monkeypatch, outcome):
    for time in (1, 2, 3):
        tifffile.imwrite(tmp_path / f"t{time:03d}.tif", np.full((2, 8, 8), time, dtype=np.uint16),
                         photometric="minisblack")
    provider = StackTiffProvider(tmp_path, pattern="t{time:03d}.tif")
    provider.get_stack(1)
    manager = RoiManager(_mixed_movie_document())
    app = AceTreeApp(NucleiManager(), provider, manager)
    previous = app.roi_measurement_engine.measure(manager)
    original_handle = provider._open_tif
    window = QWidget()
    if outcome != "close":
        qtbot.addWidget(window)
    window.show()
    app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
    notices = []
    published = []
    app._say = notices.append
    app._refresh_roi_scalar_plot_windows = published.append
    entered, release = Event(), Event()
    clones, reader_threads = [], []
    original_clone = roi_measurement_job.clone_image_provider_for_worker

    def clone_for_test(source):
        clone = original_clone(source)
        clones.append(clone)
        for name in ("get_plane", "get_stack"):
            read = getattr(clone, name)
            def gated_read(*args, read=read, **kwargs):
                reader_threads.append(get_ident())
                entered.set()
                assert release.wait(5), "Test did not release background image read"
                return read(*args, **kwargs)
            setattr(clone, name, gated_read)
        return clone

    monkeypatch.setattr(roi_measurement_job, "clone_image_provider_for_worker", clone_for_test)
    jobs = app._measurement_job_controller()
    try:
        assert app.start_roi_measurement(RoiMeasurementRequest(manager))
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert jobs.active
        assert not app.start_roi_measurement(RoiMeasurementRequest(manager))
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        qtbot.waitUntil(lambda: bool(heartbeat))
        assert app.roi_measurement_engine.latest_snapshot is previous
        assert clones[0] is not provider
        assert clones[0].image_shape == provider.image_shape == (8, 8)
        assert clones[0].num_planes == provider.num_planes == 2
        assert provider._open_tif is original_handle
        assert set(reader_threads) == {reader_threads[0]} and get_ident() not in reader_threads

        if outcome == "cancel":
            jobs.cancel()
        elif outcome == "edit":
            manager.update_frame_geometry(manager.objects[0].object_id, 1,
                                          Polygon2D(1, polygon(2).exterior_xy_px))
        elif outcome == "images":
            (tmp_path / "t001.tif").write_bytes(b"external source changed")
        elif outcome == "close":
            window.setAttribute(Qt.WA_DeleteOnClose, True)
            window.close()
        release.set()
        qtbot.waitUntil(lambda: not jobs.active, timeout=10000)
        assert clones[0]._open_tif is None
        if outcome == "success":
            assert len(published) == 1
            assert app.roi_measurement_engine.latest_snapshot is published[0]
            assert published[0].is_current(manager, image_provider=provider)
            assert all(sample.metric("intensity.mean").value == sample.timepoint
                       for sample in published[0].samples.values())
            # Independent wrappers around in-memory movies keep original-source
            # provenance, so a successful background result is immediately usable.
            app.image_provider = NumpyProvider(np.ones((3, 2, 8, 8)))
            assert app.start_roi_measurement(RoiMeasurementRequest(manager))
            qtbot.waitUntil(lambda: not jobs.active, timeout=10000)
            assert len(published) == 2
            assert published[-1].is_current(manager, image_provider=app.image_provider)
        else:
            assert not published
            assert app.roi_measurement_engine.latest_snapshot is previous
    finally:
        release.set()
        jobs.shutdown()
        qtbot.waitUntil(lambda: not jobs.active, timeout=10000)
        close_worker_image_provider(provider)


@pytest.mark.parametrize("outcome", ["success", "cancel", "edit", "provider", "close"])
def test_nuclear_job_stages_privately_and_commits_only_current_outputs(qtbot, tmp_path, monkeypatch, outcome):
    from acetree_py.analysis import nuclear_measurement_job
    from acetree_py.analysis.measure_runner import run_measure
    from tests.test_expression_measurements import _manager, _provider

    manager = _manager()
    output = tmp_path / "measurements"
    written = run_measure(manager, _provider(), output, 0, correction_method="none")
    previous_store = manager.expression_measurements
    previous_files = {path: path.read_bytes() for path in written}
    nucleus = manager.nuclei_record[0][0]
    previous_fields = (nucleus.rwraw, nucleus.rweight)
    app = AceTreeApp(manager, _provider(220, 440))
    assert not app._nuclear_measurement_unsaved
    # A cancelled/stale job must also preserve an earlier unsaved publication.
    previous_unsaved = outcome != "success"
    app._nuclear_measurement_unsaved = previous_unsaved
    app.current_expression_channel = 1
    window = QWidget()
    if outcome != "close":
        qtbot.addWidget(window)
    window.show()
    app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
    notices, refreshed = [], []
    app._say = notices.append
    app._refresh_nuclear_measurement_windows = lambda paths, directory: refreshed.append(paths)
    prepared_event, release = Event(), Event()
    original_prepare = nuclear_measurement_job.prepare_measure_publication
    compute_threads = []

    def stage_then_wait(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        compute_threads.append(get_ident())
        prepared_event.set()
        assert release.wait(5), "Test did not release staged measurement"
        return result

    monkeypatch.setattr(nuclear_measurement_job, "prepare_measure_publication", stage_then_wait)
    jobs = app._measurement_job_controller()
    try:
        assert app.start_nuclear_measurement(output, 0, "none")
        qtbot.waitUntil(prepared_event.is_set, timeout=5000)
        assert not app.start_roi_measurement(RoiMeasurementRequest(app.roi_manager))
        assert compute_threads and get_ident() not in compute_threads
        assert manager.expression_measurements is previous_store
        assert (nucleus.rwraw, nucleus.rweight) == previous_fields
        assert {path: path.read_bytes() for path in written} == previous_files
        assert list(output.glob("*.tmp"))
        assert app.current_expression_channel == 1
        assert app._nuclear_measurement_unsaved is previous_unsaved

        if outcome == "cancel":
            jobs.cancel()
        elif outcome == "edit":
            # Even edit/undo returning to identical geometry advances freshness.
            manager.mark_data_edited()
        elif outcome == "provider":
            app.image_provider = _provider(330, 550)
        elif outcome == "close":
            window.setAttribute(Qt.WA_DeleteOnClose, True)
            window.close()
        release.set()
        qtbot.waitUntil(lambda: not jobs.active, timeout=10000)
        assert not list(output.glob("*.tmp"))
        assert not list(output.glob("*.bak"))
        assert manager.nuclei_record[0][0] is nucleus
        if outcome == "success":
            assert len(refreshed) == 1
            assert manager.expression_measurements is not previous_store
            assert nucleus.rwraw == 220_000
            assert app.current_expression_channel == 0
            assert b"220000" in written[0].read_bytes()
            assert b"440000" in written[1].read_bytes()
            assert app._nuclear_measurement_unsaved
            assert not app.edit_history.modified
            assert not manager._config_dirty  # The correction stayed unchanged.

            copy_path = tmp_path / "measurement-copy.zip"
            assert app._do_save(copy_path, mark_saved=False) == copy_path
            assert app._nuclear_measurement_unsaved
            with monkeypatch.context() as failed_save:
                from qtpy.QtWidgets import QMessageBox

                def fail_write(*args, **kwargs):
                    raise OSError("save unavailable")

                failed_save.setattr(manager, "save", fail_write)
                failed_save.setattr(QMessageBox, "critical", lambda *args: None)
                assert app._do_save(tmp_path / "failed.zip") is None
                assert app._nuclear_measurement_unsaved
            save_path = tmp_path / "measurement-saved.zip"
            assert app._do_save(save_path) == save_path
            assert not app._nuclear_measurement_unsaved
            assert not app.edit_history.modified
        else:
            assert not refreshed
            assert manager.expression_measurements is previous_store
            assert (nucleus.rwraw, nucleus.rweight) == previous_fields
            assert {path: path.read_bytes() for path in written} == previous_files
            assert app.current_expression_channel == 1
            assert app._nuclear_measurement_unsaved is previous_unsaved
    finally:
        release.set()
        jobs.shutdown()
        qtbot.waitUntil(lambda: not jobs.active, timeout=10000)


def test_application_quit_drains_owned_worker_and_discards_unpublished_result():
    # A real application exit needs its own process: abandoning a running Qt
    # thread can abort Python before cleanup and cannot be tested in this runner.
    import os
    import subprocess
    import sys

    script = """
import time
from threading import Event
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QWidget
from acetree_py.gui.measurement_jobs import MeasurementJobs
app = QApplication([])
window = QWidget()
window.show()
started = Event()
jobs = MeasurementJobs(window, print)
def compute(progress, cancelled):
    started.set()
    try:
        time.sleep(1.8)
        return 'result'
    finally:
        print('provider cleanup completed', flush=True)
def quit_when_started():
    if started.is_set():
        app.quit()
    else:
        QTimer.singleShot(10, quit_when_started)
jobs.start('Measure', compute, lambda result: print('unexpected publish'),
           discard=lambda result: print('staged output discarded', flush=True))
QTimer.singleShot(10, quit_when_started)
app.exec_()
assert not jobs._thread.isRunning()
print('clean exit', flush=True)
"""
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, "-B", "-c", script], env=environment,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "provider cleanup completed" in result.stdout
    assert "staged output discarded" in result.stdout
    assert "clean exit" in result.stdout
    assert "unexpected publish" not in result.stdout
