"""Real-napari smoke coverage for tracking proposal layers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

napari = pytest.importorskip("napari")

from acetree_py.gui.viewer_integration import ViewerIntegration
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


def _diagnostic_proposal() -> TrackingResult:
    request = TrackingRequest(
        ComponentSpec("acetree.dog3d", {"TARGET_CHANNEL": 1}),
        ComponentSpec("acetree.simple_lap", {}),
        TrackingScope(
            "selected_forward",
            1,
            3,
            seed_anchors=((1, 1),),
            roi_radius_um=6.0,
        ),
    )
    seed = Detection("seed", 1, 6.0, 8.0, 4.0, 2.0, 1.0)
    accepted = Detection("accepted", 2, 8.0, 10.0, 8.0, 2.0, 10.0)
    candidates = (
        Detection("candidate-a", 3, 9.0, 9.0, 8.0, 1.5, 9.0),
        Detection("candidate-b", 3, 9.0, 11.0, 8.0, 1.5, 8.0),
    )
    return TrackingResult(
        request=request,
        detections=(seed, accepted),
        edges=(TrackEdge("seed", "accepted", 1.0),),
        existing_anchors={"seed": (1, 1)},
        outcome=TrackingOutcome(
            "ambiguity",
            stop_frame=3,
            last_accepted_frame=2,
            predicted_position_um=(9.0, 10.0, 8.0),
            search_radius_um=6.0,
            review_candidates=candidates,
        ),
    )


def test_tracking_preview_round_trips_real_napari_2d_and_3d():
    # ViewerModel exercises napari's real layer models without requiring an
    # OpenGL canvas, so this remains reliable on headless CI and Windows.
    viewer = napari.components.ViewerModel()
    app = SimpleNamespace(
        viewer=viewer,
        current_time=3,
        current_plane=7,
        _3d_mode=False,
        _3d_windows=[],
    )

    def set_time(value):
        app.current_time = int(value)

    def set_plane(value):
        app.current_plane = int(value)

    app.set_time = set_time
    app.set_plane = set_plane
    integration = ViewerIntegration(app)
    integration.setup_layers()
    integration.show_tracking_preview(
        _diagnostic_proposal(),
        Calibration(2.0, 4.0, plane_start=5),
    )
    integration.show_detector_preview(
        (Detection("detector-test", 3, 8.0, 10.0, 8.0, 2.0, 7.0),),
        Calibration(2.0, 4.0, plane_start=5),
    )

    assert integration._tracking_preview_spots_layer.visible is True
    assert len(integration._tracking_preview_spots_layer.data) == 3
    assert len(integration._tracking_preview_links_layer.data) == 2
    assert integration._tracking_preview_spots_layer.editable is False
    assert integration._detector_preview_spots_layer.visible is True
    assert len(integration._detector_preview_spots_layer.data) == 1
    assert integration._detector_preview_spots_layer.editable is False

    app._3d_mode = True
    viewer.dims.ndisplay = 3
    integration.refresh_tracking_preview()
    points = integration._tracking_preview_3d_spots_layer
    paths = integration._tracking_preview_3d_links_layer
    assert len(points.data) == 2
    assert points.data[0, 0] == pytest.approx(2.0)
    assert len(paths.data) == 3
    assert points.editable is False
    assert paths.editable is False
    detector_points = integration._detector_preview_3d_spots_layer
    assert len(detector_points.data) == 1
    assert detector_points.data[0, 0] == pytest.approx(2.0)
    assert detector_points.editable is False

    app._3d_mode = False
    viewer.dims.ndisplay = 2
    integration.refresh_tracking_preview()
    assert integration._tracking_preview_spots_layer.visible is True
    assert integration._tracking_preview_3d_spots_layer.visible is False
    integration.clear_detector_preview()
    assert integration._detector_preview_spots_layer.visible is False
    assert integration._detector_preview_3d_spots_layer.visible is False
