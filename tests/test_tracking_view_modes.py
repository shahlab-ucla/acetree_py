"""Focused contracts for proposal rendering across 2D and 3D viewers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from acetree_py.gui.viewer_3d_window import Viewer3DWindow  # noqa: E402
from acetree_py.gui.viewer_integration import ViewerIntegration  # noqa: E402


@dataclass(frozen=True)
class _PreviewSpot:
    preview_id: str
    detection_id: str | None
    frame: int
    x_um: float
    y_um: float
    z_um: float
    radius_um: float
    quality: float
    kind: str


@dataclass(frozen=True)
class _PreviewLink:
    source_id: str
    target_id: str
    kind: str
    cost: float


@dataclass(frozen=True)
class _SearchRegion:
    frame: int
    x_um: float
    y_um: float
    z_um: float
    radius_um: float
    outcome_code: str = "ambiguity"


@dataclass(frozen=True)
class _ExpandedPreview:
    spots: tuple[_PreviewSpot, ...]
    links: tuple[_PreviewLink, ...]
    candidates: tuple[_PreviewSpot, ...] = ()
    search_region: _SearchRegion | None = None

    @property
    def by_id(self):
        return {spot.preview_id: spot for spot in self.review_spots}

    @property
    def review_spots(self):
        return (*self.spots, *self.candidates)


@dataclass(frozen=True)
class _Calibration:
    xy_um: float
    z_um: float
    plane_start: int = 1

    def physical_to_pixel(self, x_um, y_um, z_um):
        return (
            x_um / self.xy_um,
            y_um / self.xy_um,
            z_um / self.z_um + self.plane_start,
        )


class _FakeLayer:
    def __init__(self, data=(), **properties):
        self.data = list(data) if not isinstance(data, np.ndarray) else data.copy()
        self.visible = properties.pop("visible", True)
        self.editable = properties.pop("editable", True)
        self.selected_data = {0}
        self.mode = "select"
        self.last_add = None
        for name, value in properties.items():
            setattr(self, name, value)

    def add(self, data, **properties):
        self.data = list(data)
        self.last_add = properties


class _FakeLayerList:
    def __init__(self, active=None):
        self._items = [] if active is None else [active]
        self.selection = SimpleNamespace(active=active)

    def __iter__(self):
        return iter(self._items)

    def append(self, layer):
        self._items.append(layer)

    def remove(self, layer):
        self._items.remove(layer)


class _FakeViewer:
    def __init__(self, active=None):
        self.layers = _FakeLayerList(active)
        self.camera = SimpleNamespace(center=None)
        self.added_points = []
        self.added_shapes = []

    def add_points(self, data, **properties):
        layer = _FakeLayer(data, **properties)
        self.added_points.append(layer)
        self.layers.append(layer)
        # Adding a layer normally makes it active in napari.
        self.layers.selection.active = layer
        return layer

    def add_shapes(self, data, **properties):
        layer = _FakeLayer(data, **properties)
        self.added_shapes.append(layer)
        self.layers.append(layer)
        self.layers.selection.active = layer
        return layer


class _FakeControl:
    def __init__(self, value=1, *, checked=False):
        self._value = value
        self._checked = checked

    def blockSignals(self, _blocked):
        pass

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value

    def isChecked(self):
        return self._checked


def _spot(preview_id, kind, *, frame=2, x=8.0, y=10.0, z=8.0):
    return _PreviewSpot(
        preview_id=preview_id,
        detection_id=None if kind == "interpolated" else preview_id,
        frame=frame,
        x_um=x,
        y_um=y,
        z_um=z,
        radius_um=2.0,
        quality=10.0,
        kind=kind,
    )


def _preview_with_all_symbols():
    return _ExpandedPreview(
        spots=(
            _spot("seed", "seed", frame=1, x=6.0, y=8.0, z=4.0),
            _spot("detected", "detection"),
            _spot("gap", "interpolated", x=10.0),
        ),
        links=(_PreviewLink("seed", "detected", "link", 1.0),),
        candidates=(_spot("candidate", "candidate", x=12.0),),
    )


def _main_integration(*, is_3d=True):
    editing_layer = object()
    viewer = _FakeViewer(active=editing_layer)
    app = SimpleNamespace(
        viewer=viewer,
        current_time=2,
        current_plane=7,
        _3d_mode=is_3d,
        _3d_windows=[],
    )

    def set_time(value):
        app.current_time = value

    def set_plane(value):
        app.current_plane = value
        app.plane_calls.append(value)

    app.set_time = set_time
    app.set_plane = set_plane
    app.plane_calls = []
    integration = ViewerIntegration(app)
    integration._shapes_layer = editing_layer
    integration._tracking_preview_spots_layer = _FakeLayer()
    integration._tracking_preview_links_layer = _FakeLayer()
    viewer.layers.append(integration._tracking_preview_spots_layer)
    viewer.layers.append(integration._tracking_preview_links_layer)
    integration._tracking_preview = _preview_with_all_symbols()
    integration._tracking_preview_calibration = _Calibration(2.0, 4.0, plane_start=5)
    integration._tracking_preview_visible = True
    return integration, app, editing_layer


def test_main_3d_preview_uses_locked_native_layers_and_calibrated_plane_start():
    integration, app, editing_layer = _main_integration(is_3d=True)

    integration.refresh_tracking_preview()

    points = integration._tracking_preview_3d_spots_layer
    paths = integration._tracking_preview_3d_links_layer
    np.testing.assert_allclose(points.data[0], [2.0, 5.0, 4.0])
    assert list(points.symbol) == ["ring", "diamond", "cross"]
    assert points.scale == (2.0, 1.0, 1.0)
    assert points.editable is False
    assert points.mode == "pan_zoom"
    assert paths.editable is False
    assert paths.last_add["shape_type"] == "path"
    np.testing.assert_allclose(paths.data[0][1], [2.0, 5.0, 4.0])
    assert app.viewer.layers.selection.active is editing_layer
    assert integration._tracking_preview_spots_layer.visible is False


def test_main_preview_mode_transition_and_centering_are_mode_aware():
    integration, app, editing_layer = _main_integration(is_3d=True)
    integration.refresh_tracking_preview()

    app._3d_mode = False
    integration.refresh_tracking_preview()
    assert integration._tracking_preview_spots_layer.visible is True
    assert integration._tracking_preview_3d_spots_layer.visible is False
    assert app.viewer.layers.selection.active is editing_layer

    assert integration.center_tracking_preview("detected") is True
    assert app.plane_calls == [7]
    assert app.viewer.camera.center == (5.0, 4.0)

    app._3d_mode = True
    assert integration.center_tracking_preview("detected") is True
    assert app.plane_calls == [7]  # volume mode must not collapse to a Z slice
    assert app.viewer.camera.center == (4.0, 5.0, 4.0)


def test_detector_test_uses_separate_locked_layers_in_2d_and_main_3d():
    integration, app, editing_layer = _main_integration(is_3d=True)
    integration._detector_preview = _ExpandedPreview(
        spots=(_spot("detector-test", "detector_test"),),
        links=(),
    )
    integration._detector_preview_calibration = _Calibration(
        2.0,
        4.0,
        plane_start=5,
    )
    integration._detector_preview_visible = True

    integration.refresh_tracking_preview()

    points = integration._detector_preview_3d_spots_layer
    assert points.name == "Detector Test Positions 3D"
    assert points.editable is False
    assert list(points.symbol) == ["ring"]
    np.testing.assert_allclose(points.data[0], [2.0, 5.0, 4.0])
    assert integration._tracking_preview_3d_spots_layer is not points
    assert app.viewer.layers.selection.active is editing_layer

    app._3d_mode = False
    integration.refresh_tracking_preview()
    shapes = integration._detector_preview_spots_layer
    assert shapes.name == "Detector Test Positions"
    assert shapes.visible is True
    assert shapes.editable is False
    assert len(shapes.data) == 1
    assert points.visible is False
    assert app.viewer.layers.selection.active is editing_layer


def test_detector_test_recomputes_slice_rings_during_normal_z_navigation():
    integration, app, _editing_layer = _main_integration(is_3d=False)
    integration._detector_preview = _ExpandedPreview(
        spots=(_spot("detector-test", "detector_test"),),
        links=(),
    )
    integration._detector_preview_calibration = _Calibration(
        2.0,
        4.0,
        plane_start=5,
    )
    integration._detector_preview_visible = True
    integration.refresh_tracking_preview()
    shapes = integration._detector_preview_spots_layer
    assert len(shapes.data) == 1

    # The production plane-only display path calls update_overlays(). Avoid
    # exercising curated-shape rendering in this preview-focused harness.
    integration._shapes_layer = None
    app.get_nucleus_overlay_data = lambda: {}
    app.current_plane = 8
    integration.update_overlays()
    assert len(shapes.data) == 0

    app.current_plane = 7
    integration.update_overlays()
    assert len(shapes.data) == 1


def test_search_region_is_visible_as_crosshair_in_2d_and_wireframe_in_3d():
    integration, app, _editing_layer = _main_integration(is_3d=False)
    integration._tracking_preview = replace(
        integration._tracking_preview,
        search_region=_SearchRegion(2, 8.0, 10.0, 8.0, 6.0),
    )

    integration.refresh_tracking_preview()
    assert len(integration._tracking_preview_spots_layer.data) == 4
    assert len(integration._tracking_preview_links_layer.data) == 3

    app._3d_mode = True
    integration.refresh_tracking_preview()
    paths = integration._tracking_preview_3d_links_layer.data
    assert len(paths) == 4  # one movement path plus three search-sphere rings
    assert all(path.shape == (65, 3) for path in paths[-3:])


class _DetachedPreviewHarness:
    view_time = Viewer3DWindow.view_time
    _update_tracking_preview = Viewer3DWindow._update_tracking_preview
    _ensure_tracking_preview_layers = Viewer3DWindow._ensure_tracking_preview_layers
    _forget_removed_tracking_preview_layers = (
        Viewer3DWindow._forget_removed_tracking_preview_layers
    )
    _tracking_preview_style = Viewer3DWindow._tracking_preview_style
    _make_preview_layer_read_only = staticmethod(
        Viewer3DWindow._make_preview_layer_read_only
    )
    _update_detector_preview = Viewer3DWindow._update_detector_preview
    _ensure_detector_preview_layer = Viewer3DWindow._ensure_detector_preview_layer


def test_detached_preview_uses_local_unsynced_time_and_same_symbols():
    editing_layer = object()
    harness = _DetachedPreviewHarness()
    harness.app = SimpleNamespace(current_time=9)
    harness._chk_sync = _FakeControl(checked=False)
    harness._local_time = 2
    harness._viewer = _FakeViewer(active=editing_layer)
    harness._points_layer = editing_layer
    harness._tracking_preview_points_layer = None
    harness._tracking_preview_paths_layer = None
    harness._tracking_preview = _preview_with_all_symbols()
    harness._tracking_preview_calibration = _Calibration(2.0, 4.0, plane_start=5)
    harness._tracking_preview_visible = True
    harness._tracking_preview_stale = False
    harness._tracking_preview_highlight = None

    harness._update_tracking_preview()

    points = harness._tracking_preview_points_layer
    assert len(points.data) == 3
    assert list(points.symbol) == ["ring", "diamond", "cross"]
    np.testing.assert_allclose(points.data[0], [2.0, 5.0, 4.0])
    assert harness._viewer.layers.selection.active is editing_layer


def test_detached_detector_test_uses_local_time_and_separate_ring_layer():
    editing_layer = object()
    harness = _DetachedPreviewHarness()
    harness.app = SimpleNamespace(current_time=9)
    harness._chk_sync = _FakeControl(checked=False)
    harness._local_time = 2
    harness._viewer = _FakeViewer(active=editing_layer)
    harness._points_layer = editing_layer
    harness._tracking_preview_points_layer = None
    harness._tracking_preview_paths_layer = None
    harness._detector_preview_points_layer = None
    harness._detector_preview = _ExpandedPreview(
        spots=(_spot("detector-test", "detector_test"),),
        links=(),
    )
    harness._detector_preview_calibration = _Calibration(
        2.0,
        4.0,
        plane_start=5,
    )
    harness._detector_preview_visible = True

    harness._update_detector_preview()

    layer = harness._detector_preview_points_layer
    assert layer.name == "Detector Test Positions 3D"
    assert layer.editable is False
    assert len(layer.data) == 1
    np.testing.assert_allclose(layer.data[0], [2.0, 5.0, 4.0])
    assert harness._viewer.layers.selection.active is editing_layer

    harness._local_time = 3
    harness._update_detector_preview()
    assert len(layer.data) == 0


class _RefreshHarness:
    view_time = Viewer3DWindow.view_time
    refresh = Viewer3DWindow.refresh
    _on_time_spin = Viewer3DWindow._on_time_spin

    def __init__(self):
        self.app = SimpleNamespace(current_time=9, set_time=self._set_main_time)
        self._viewer = object()
        self._chk_sync = _FakeControl(checked=False)
        self._time_spin = _FakeControl(2)
        self._time_slider = _FakeControl(2)
        self._local_time = 2
        self._last_time = 2
        self.calls = []

    def _set_main_time(self, value):
        self.calls.append(("main", value))

    def _load_stacks(self, value):
        self.calls.append(("stack", value))

    def _update_points(self, value):
        self.calls.append(("points", value))

    def _update_tracking_preview(self):
        self.calls.append(("preview", self.view_time))


def test_detached_refresh_can_force_same_frame_and_controls_stay_local():
    harness = _RefreshHarness()

    harness.refresh()
    assert harness.calls == []
    harness.refresh(force=True)
    assert harness.calls == [("stack", 2), ("points", 2), ("preview", 2)]

    harness.calls.clear()
    harness._on_time_spin(4)
    assert harness._local_time == 4
    assert ("main", 4) not in harness.calls
    assert harness.calls == [("stack", 4), ("points", 4), ("preview", 4)]


def test_preview_state_propagates_and_clear_forces_detached_refresh():
    integration, app, _editing_layer = _main_integration(is_3d=False)

    class WindowSpy:
        def __init__(self):
            self.states = []

        def set_tracking_preview_state(self, preview, calibration, **state):
            self.states.append((preview, calibration, state))

    window = WindowSpy()
    app._3d_windows = [window]
    integration.set_tracking_preview_stale(True)
    integration.highlight_tracking_preview("detected")
    integration.set_tracking_preview_visible(False)
    integration.clear_tracking_preview()

    assert window.states[0][2]["stale"] is True
    assert window.states[1][2]["highlight"] == "detected"
    assert window.states[2][2]["visible"] is False
    preview, calibration, state = window.states[3]
    assert preview is None and calibration is None
    assert state == {"visible": False, "stale": False, "highlight": None}


def test_detached_state_change_updates_only_preview_when_frame_is_unchanged():
    calls = []
    harness = SimpleNamespace(_update_tracking_preview=lambda: calls.append("preview"))

    Viewer3DWindow.set_tracking_preview_state(
        harness,
        _preview_with_all_symbols(),
        _Calibration(1.0, 1.0),
        visible=True,
        stale=True,
        highlight="detected",
    )

    assert calls == ["preview"]


def test_main_3d_preview_restores_curated_points_and_recovers_removed_layer():
    integration, app, _editing_layer = _main_integration(is_3d=True)
    curated_points = object()
    app._points_layer = curated_points
    app.viewer.layers.append(curated_points)
    integration.refresh_tracking_preview()
    removed = integration._tracking_preview_3d_spots_layer
    app.viewer.layers.selection.active = removed
    app.viewer.layers.remove(removed)

    integration.refresh_tracking_preview()

    assert integration._tracking_preview_3d_spots_layer is not removed
    assert app.viewer.layers.selection.active is curated_points


def test_detached_curated_points_use_stack_local_z_for_nondefault_plane_start():
    harness = SimpleNamespace(
        app=SimpleNamespace(
            manager=SimpleNamespace(config=SimpleNamespace(plane_start=5))
        )
    )

    assert Viewer3DWindow._stack_z_from_plane(harness, 7.0) == pytest.approx(2.0)


def test_solo_detection_channel_covers_main_and_detached_images():
    main_layers = [SimpleNamespace(visible=False), SimpleNamespace(visible=True)]
    detached_layers = [SimpleNamespace(visible=True), SimpleNamespace(visible=False)]
    app = SimpleNamespace(
        _image_layers=main_layers,
        _3d_windows=[SimpleNamespace(_image_layers=detached_layers)],
    )
    integration = ViewerIntegration(app)
    snapshot = integration.capture_image_channel_visibility()

    integration.set_detection_channel_solo(0)

    assert [layer.visible for layer in main_layers] == [True, False]
    assert [layer.visible for layer in detached_layers] == [True, False]
    integration.restore_image_channel_visibility(snapshot)
    assert [layer.visible for layer in main_layers] == [False, True]
    assert [layer.visible for layer in detached_layers] == [True, False]
