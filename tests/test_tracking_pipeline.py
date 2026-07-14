"""End-to-end tests for global and selected-forward tracking orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.core.nucleus import Nucleus
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.config import AceTreeConfig
from acetree_py.io.image_provider import NumpyProvider
from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    TrackingRequest,
    TrackingScope,
)
from acetree_py.tracking.pipeline import TrackingCancelled, TrackingPipeline
from acetree_py.tracking.registry import build_default_registry


def _blob_movie(centers, shape=(7, 25, 25), sigma=1.0):
    z, y, x = np.indices(shape, dtype=float)
    frames = []
    for frame_centers in centers:
        image = np.zeros(shape, dtype=np.float32)
        for cz, cy, cx in frame_centers:
            image += 100.0 * np.exp(
                -((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
                / (2.0 * sigma**2)
            )
        frames.append(image)
    return np.stack(frames)


def _request(scope):
    return TrackingRequest(
        detector=ComponentSpec(
            "acetree.dog3d",
            {
                "TARGET_CHANNEL": 1,
                "RADIUS": 1.7,
                "THRESHOLD": 1.0,
                "DO_SUBPIXEL_LOCALIZATION": True,
                "DO_MEDIAN_FILTERING": False,
            },
        ),
        tracker=ComponentSpec(
            "acetree.simple_lap",
            {
                "LINKING_MAX_DISTANCE": 3.0,
                "ALLOW_GAP_CLOSING": True,
                "GAP_CLOSING_MAX_DISTANCE": 4.0,
                "MAX_FRAME_GAP": 2,
                "ALLOW_TRACK_SPLITTING": False,
                "ALLOW_TRACK_MERGING": False,
            },
        ),
        scope=scope,
    )


def _pipeline():
    return TrackingPipeline(build_default_registry(discover_plugins=False))


def test_global_pipeline_detects_and_links_moving_blob():
    movie = _blob_movie([
        [(3, 10, 10)],
        [(3, 10, 11)],
        [(3, 10, 12)],
    ])
    result = _pipeline().run(
        NumpyProvider(movie),
        Calibration(1.0, 1.0),
        _request(TrackingScope("global", 1, 3)),
    )

    assert len(result.detections) == 3
    assert len(result.edges) == 2
    assert result.existing_anchors == {}
    assert result.provenance["detector"]["plugin_id"] == "acetree.dog3d"


def test_selected_forward_uses_existing_anchor_and_local_roi():
    movie = _blob_movie([
        [(3, 10, 10), (3, 20, 20)],
        [(3, 10, 11), (3, 19, 20)],
        [(3, 10, 12), (3, 18, 20)],
    ])
    record = [
        [Nucleus(index=1, x=10, y=10, z=4.0, size=4, status=1)],
        [],
        [],
    ]
    request = _request(
        TrackingScope(
            "selected_forward",
            1,
            3,
            seed_anchors=((1, 1),),
            roi_radius_um=5.0,
            ambiguity_ratio=1.2,
        )
    )
    result = _pipeline().run(
        NumpyProvider(movie),
        Calibration(1.0, 1.0),
        request,
        nuclei_record=record,
    )

    assert len(result.existing_anchors) == 1
    assert len(result.new_detections) == 2
    assert len(result.edges) == 2
    assert all(detection.y_um < 15 for detection in result.new_detections)
    # Analysis is proposal-only.
    assert record[1:] == [[], []]


def test_selected_forward_stops_instead_of_guessing_between_equal_candidates():
    movie = _blob_movie([
        [(3, 10, 10)],
        [(3, 10, 8), (3, 10, 12)],
    ])
    record = [
        [Nucleus(index=1, x=10, y=10, z=4.0, size=4, status=1)],
        [],
    ]
    result = _pipeline().run(
        NumpyProvider(movie),
        Calibration(1.0, 1.0),
        _request(
            TrackingScope(
                "selected_forward",
                1,
                2,
                seed_anchors=((1, 1),),
                roi_radius_um=6.0,
                ambiguity_ratio=1.2,
            )
        ),
        nuclei_record=record,
    )

    assert len(result.new_detections) == 0
    assert any("similar assignment costs" in warning for warning in result.warnings)


def test_cancelled_global_run_returns_no_partial_result_or_edit():
    movie = _blob_movie([[(3, 10, 10)], [(3, 10, 11)]])
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 1

    with pytest.raises(TrackingCancelled):
        _pipeline().run(
            NumpyProvider(movie),
            Calibration(1.0, 1.0),
            _request(TrackingScope("global", 1, 2)),
            cancelled=cancelled,
        )


def test_pipeline_preflights_frame_and_channel_bounds():
    provider = NumpyProvider(_blob_movie([[(3, 10, 10)]]))
    with pytest.raises(ValueError, match="has 1 timepoint"):
        _pipeline().run(
            provider,
            Calibration(1.0, 1.0),
            _request(TrackingScope("global", 1, 2)),
        )

    request = _request(TrackingScope("global", 1, 1))
    request = TrackingRequest(
        detector=ComponentSpec(
            request.detector.plugin_id,
            {**request.detector.settings, "TARGET_CHANNEL": 2},
        ),
        tracker=request.tracker,
        scope=request.scope,
    )
    with pytest.raises(ValueError, match="has 1 channel"):
        _pipeline().run(provider, Calibration(1.0, 1.0), request)


def test_app_accepts_run_as_one_undo_step_and_tracks_provenance(tmp_path):
    movie = _blob_movie([[(3, 10, 10)], [(3, 10, 11)]])
    config = AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=7)
    manager = NucleiManager.new_empty(config, 2)
    app = AceTreeApp(manager, NumpyProvider(movie))

    result = app.run_tracking_request(
        _request(TrackingScope("global", 1, 2))
    )

    assert sum(len(frame) for frame in manager.nuclei_record) == 2
    assert app.edit_history.num_undoable == 1
    assert app._tracking_results == [result]

    app.edit_history.undo()
    assert manager.nuclei_record == [[], []]
    assert app._tracking_results == []

    app.edit_history.redo()
    assert sum(len(frame) for frame in manager.nuclei_record) == 2
    assert app._tracking_results == [result]

    destination = tmp_path / "embryo.zip"
    assert app._do_save(destination) == destination
    assert (tmp_path / "embryo.tracking.json").exists()

    app.edit_history.undo()
    assert app._do_save(destination) == destination
    assert not (tmp_path / "embryo.tracking.json").exists()


def test_selected_forward_preserves_forced_seed_name_via_naming_pipeline():
    movie = _blob_movie([
        [(3, 10, 10)],
        [(3, 10, 11)],
        [(3, 10, 12)],
    ])
    config = AceTreeConfig(xy_res=1.0, z_res=1.0, plane_end=7)
    manager = NucleiManager.new_empty(config, 3)
    manager.nuclei_record[0].append(
        Nucleus(
            index=1,
            x=10,
            y=10,
            z=4.0,
            size=4,
            identity="EMS",
            assigned_id="EMS",
            status=1,
        )
    )
    manager.process()
    app = AceTreeApp(manager, NumpyProvider(movie))
    request = _request(
        TrackingScope(
            "selected_forward",
            1,
            3,
            seed_anchors=((1, 1),),
            roi_radius_um=5.0,
            ambiguity_ratio=1.2,
        )
    )

    app.run_tracking_request(request)

    assert [frame[0].effective_name for frame in manager.nuclei_record] == [
        "EMS",
        "EMS",
        "EMS",
    ]
    assert [frame[0].assigned_id for frame in manager.nuclei_record] == [
        "EMS",
        "EMS",
        "EMS",
    ]
