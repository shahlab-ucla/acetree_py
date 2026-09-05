"""Outcome-level checks for cancellable measurement and GUI publication."""

from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget

from acetree_py.analysis import roi_measurement_job
from acetree_py.analysis.roi_measurements import RoiMeasurementEngine, RoiMeasurementRequest
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
