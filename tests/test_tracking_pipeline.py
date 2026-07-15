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
    Detection,
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


class _FrameDetector:
    """Deterministic selected-forward detector for stop-diagnostic tests."""

    def __init__(self, detections_by_frame):
        self._detections_by_frame = detections_by_frame

    def detect(self, stack, frame, calibration, settings, **kwargs):
        return tuple(self._detections_by_frame.get(frame, ()))


def _review_candidate(
    detection_id,
    *,
    frame=2,
    x_um,
    radius_um=1.0,
    quality=10.0,
):
    return Detection(
        detection_id=detection_id,
        frame=frame,
        x_um=x_um,
        y_um=10.0,
        z_um=3.0,
        radius_um=radius_um,
        quality=quality,
    )


def _run_selected_with_candidates(
    monkeypatch,
    detections_by_frame,
    *,
    nuclei_record=None,
    end_frame=2,
    roi_radius_um=8.0,
):
    if nuclei_record is None:
        nuclei_record = [
            [Nucleus(index=1, x=10, y=10, z=4.0, size=4, status=1)],
            *([] for _ in range(end_frame - 1)),
        ]
    pipeline = _pipeline()
    detector = _FrameDetector(detections_by_frame)
    monkeypatch.setattr(
        pipeline.registry,
        "create_detector",
        lambda plugin_id: detector,
    )
    movie = np.zeros((end_frame, 7, 25, 25), dtype=np.float32)
    request = _request(
        TrackingScope(
            "selected_forward",
            1,
            end_frame,
            seed_anchors=((1, 1),),
            roi_radius_um=roi_radius_um,
            ambiguity_ratio=1.2,
        )
    )
    return pipeline.run(
        NumpyProvider(movie),
        Calibration(1.0, 1.0),
        request,
        nuclei_record=nuclei_record,
    )


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
    assert result.outcome is None


def test_current_frame_detector_preview_reads_one_frame_and_never_builds_tracker(
    monkeypatch,
):
    movie = _blob_movie(
        [
            [(3, 10, 9)],
            [(3, 10, 11)],
            [(3, 10, 13)],
        ]
    )

    class RecordingProvider(NumpyProvider):
        def __init__(self, data):
            super().__init__(data)
            self.calls = []

        def get_stack(self, time, channel=0):
            self.calls.append((time, channel))
            return super().get_stack(time, channel)

    provider = RecordingProvider(movie)
    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline.registry,
        "create_tracker",
        lambda _plugin_id: pytest.fail("detector preview constructed a tracker"),
    )
    detector = _request(TrackingScope("global", 1, 3)).detector

    detections = pipeline.detect_frame(
        provider,
        Calibration(1.0, 1.0),
        detector,
        frame=2,
    )

    assert provider.calls == [(2, 0)]
    assert detections
    assert {detection.frame for detection in detections} == {2}
    assert list(detections) == sorted(
        detections,
        key=lambda detection: (
            detection.z_um,
            detection.y_um,
            detection.x_um,
            detection.detection_id,
        ),
    )


def test_current_frame_detector_preview_validates_bounds_and_cancellation(monkeypatch):
    provider = NumpyProvider(np.zeros((2, 3, 5, 5), dtype=np.float32))
    pipeline = _pipeline()
    detector = _request(TrackingScope("global", 1, 2)).detector

    with pytest.raises(ValueError, match="frame 3 is unavailable"):
        pipeline.detect_frame(
            provider,
            Calibration(1.0, 1.0),
            detector,
            frame=3,
        )
    detector_constructions = []
    create_detector = pipeline.registry.create_detector

    def record_detector_construction(plugin_id):
        detector_constructions.append(plugin_id)
        return create_detector(plugin_id)

    monkeypatch.setattr(
        pipeline.registry,
        "create_detector",
        record_detector_construction,
    )
    with pytest.raises(TrackingCancelled):
        pipeline.detect_frame(
            provider,
            Calibration(1.0, 1.0),
            detector,
            frame=1,
            cancelled=lambda: True,
        )
    assert detector_constructions == []
    bad_channel = ComponentSpec(
        detector.plugin_id,
        {**dict(detector.settings), "TARGET_CHANNEL": 2},
    )
    with pytest.raises(ValueError, match="TARGET_CHANNEL 2 is unavailable"):
        pipeline.detect_frame(
            provider,
            Calibration(1.0, 1.0),
            bad_channel,
            frame=1,
        )


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
    assert result.outcome is not None
    assert result.outcome.code == "completed"
    assert result.outcome.stop_frame is None
    assert result.outcome.last_accepted_frame == 3
    # Analysis is proposal-only.
    assert record[1:] == [[], []]


def test_selected_forward_reports_ambiguity_with_review_candidates(monkeypatch):
    candidates = (
        _review_candidate("left", x_um=7.0),
        _review_candidate("right", x_um=13.0),
    )
    result = _run_selected_with_candidates(monkeypatch, {2: candidates})

    assert len(result.new_detections) == 0
    assert any("similar assignment costs" in warning for warning in result.warnings)
    assert result.outcome is not None
    assert result.outcome.code == "ambiguity"
    assert result.outcome.stop_frame == 2
    assert result.outcome.last_accepted_frame == 1
    assert result.outcome.predicted_position_um == pytest.approx((10.0, 10.0, 3.0))
    assert result.outcome.search_radius_um == pytest.approx(8.0)
    assert result.outcome.review_candidates == candidates
    assert all(candidate not in result.detections for candidate in candidates)


def test_ambiguity_preview_contains_only_the_triggering_pair(monkeypatch):
    near_left = _review_candidate("near-left", x_um=8.0)
    near_right = _review_candidate("near-right", x_um=12.1)
    unrelated = _review_candidate("unrelated", x_um=7.0)

    result = _run_selected_with_candidates(
        monkeypatch,
        {2: (unrelated, near_right, near_left)},
    )

    assert result.outcome is not None
    assert result.outcome.code == "ambiguity"
    assert result.outcome.review_candidates == (near_left, near_right)


def test_probable_division_takes_priority_over_generic_ambiguity(monkeypatch):
    candidates = (
        _review_candidate("daughter-a", x_um=8.0, radius_um=2.0),
        _review_candidate("daughter-b", x_um=12.0, radius_um=2.0),
    )

    result = _run_selected_with_candidates(monkeypatch, {2: candidates})

    assert result.outcome is not None
    assert result.outcome.code == "division"
    assert result.outcome.review_candidates == candidates
    assert any("probable division" in warning for warning in result.warnings)
    assert not any("similar assignment costs" in warning for warning in result.warnings)


def test_selected_forward_reports_curated_overlap_as_conflict(monkeypatch):
    seed = Nucleus(index=1, x=10, y=10, z=4.0, size=4, status=1)
    curated = Nucleus(index=1, x=11, y=10, z=4.0, size=4, status=1)
    candidate = _review_candidate("overlap", x_um=11.0)

    result = _run_selected_with_candidates(
        monkeypatch,
        {2: (candidate,)},
        nuclei_record=[[seed], [curated]],
    )

    assert result.outcome is not None
    assert result.outcome.code == "conflict"
    assert result.outcome.review_candidates == (candidate,)
    assert any("existing curated nucleus" in warning for warning in result.warnings)


def test_selected_forward_reports_lost_at_unclosed_end_gap(monkeypatch):
    outside_gate = _review_candidate("too-far", x_um=14.0)

    result = _run_selected_with_candidates(
        monkeypatch,
        {2: (outside_gate,)},
    )

    assert result.outcome is not None
    assert result.outcome.code == "lost"
    assert result.outcome.stop_frame == 2
    assert result.outcome.last_accepted_frame == 1
    assert result.outcome.review_candidates == (outside_gate,)
    assert any("distance gate" in warning for warning in result.warnings)


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
