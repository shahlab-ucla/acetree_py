"""Focused contracts for curated 2D centroid marker rendering."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from acetree_py.gui.viewer_integration import ViewerIntegration


def _overlay(
    *,
    centers: list[tuple[float, float]],
    radii: list[float],
    names: list[str] | None = None,
) -> dict:
    count = len(centers)
    return {
        "centers": np.asarray(centers, dtype=float).reshape(count, 2),
        "radii": np.asarray(radii, dtype=float),
        "colors": np.tile([0.55, 0.27, 1.0, 0.8], (count, 1)),
        "names": names if names is not None else [f"Nuc{i + 1}" for i in range(count)],
        "selected_idx": -1,
    }


class _FakeShapesLayer:
    """Small Shapes double with an add that can fail after partial work."""

    def __init__(self) -> None:
        self.data = [
            np.array(
                [[8.0, 9.0], [9.0, 10.0], [10.0, 9.0]],
                dtype=float,
            )
        ]
        self.shape_type = ["polygon"]
        self.edge_color = np.array([[0.2, 0.4, 0.6, 1.0]])
        self.face_color = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.edge_width = [3.0]
        self.text = {
            "string": ["Old"],
            "color": "yellow",
            "size": 11,
            "anchor": "center",
        }
        self.editable = True
        self.selected_data = {0}
        self.mode = "select"
        self.fail_next_add = False
        self.partial_adds = 0

    def add(
        self,
        shapes,
        *,
        shape_type,
        edge_color,
        face_color,
        edge_width,
    ) -> None:
        shapes = [np.array(shape, copy=True) for shape in shapes]
        types = (
            [shape_type] * len(shapes)
            if isinstance(shape_type, str)
            else list(shape_type)
        )
        edges = np.asarray(edge_color, dtype=float).copy()
        faces = np.asarray(face_color, dtype=float).copy()
        widths = np.asarray(edge_width, dtype=float).tolist()

        # Mirror napari's behavior closely enough for the contract: clearing
        # data also clears the parallel presentation arrays, and a programmatic
        # add can leave the layer editable again.
        if not self.data:
            self.shape_type = []
            self.edge_color = np.empty((0, 4))
            self.face_color = np.empty((0, 4))
            self.edge_width = []
        self.editable = True

        if self.fail_next_add:
            self.fail_next_add = False
            self.partial_adds += 1
            if shapes:
                self.data.append(shapes[0])
                self.shape_type.append(types[0])
                self.edge_color = edges[:1]
                self.face_color = faces[:1]
                self.edge_width = widths[:1]
                self.text = {
                    "string": ["Partial"],
                    "color": "red",
                    "size": 20,
                    "anchor": "center",
                }
            raise RuntimeError("simulated partial Shapes.add failure")

        self.data.extend(shapes)
        self.shape_type.extend(types)
        self.edge_color = edges
        self.face_color = faces
        self.edge_width = widths


def _fake_integration(layer: _FakeShapesLayer, overlay: dict) -> ViewerIntegration:
    app = SimpleNamespace(
        viewer=None,
        get_nucleus_overlay_data=lambda: overlay,
    )
    integration = ViewerIntegration(app)
    integration._shapes_layer = layer
    integration._update_tracking_preview = lambda: None
    integration._update_detector_preview = lambda: None
    return integration


def test_viewer_model_curated_layer_is_locked_and_click_workflows_survive(
    monkeypatch,
) -> None:
    napari = pytest.importorskip("napari")
    viewer = napari.components.ViewerModel()
    calls: list[tuple] = []
    queued: list = []
    nucleus = SimpleNamespace(index=1, effective_name="AB", is_alive=True)
    app = SimpleNamespace(
        viewer=viewer,
        _delete_active_nucleus=lambda: calls.append(("delete",)),
        _relink_pick_mode=False,
        _relink_pick_callback=None,
        _add_mode=False,
        _placement_mode=False,
        _placement_parent_name=None,
        _placement_parent_anchor=None,
        _placement_default_size=20,
        selection_anchor=None,
        current_time=1,
        current_plane=1,
        current_cell_name="",
        edit_history=SimpleNamespace(change_counter=0),
        manager=SimpleNamespace(find_closest_nucleus=lambda *_args, **_kwargs: nucleus),
    )
    app._nucleus_at_anchor = lambda anchor: nucleus if anchor == (1, 1) else None
    app.update_display = lambda: calls.append(("redraw",))
    app.deselect_cell = lambda: calls.append(("deselect",))
    integration = ViewerIntegration(app)
    app._viewer_integration = integration

    def set_selection(time, selected) -> None:
        calls.append(("select", time, selected.index))
        app.current_cell_name = "AB"
        integration._shown_labels.add("AB")

    app._set_selection_from_nucleus = set_selection
    integration.setup_layers()
    monkeypatch.setattr(
        "acetree_py.gui.viewer_integration.QTimer.singleShot",
        lambda _delay, callback: queued.append(callback),
    )

    layer = integration._shapes_layer
    assert layer is not None
    assert layer.editable is False
    assert layer.mode == "pan_zoom"
    assert not layer.selected_data
    assert integration._on_click in layer.mouse_drag_callbacks
    assert viewer.layers.selection.active is layer

    def release(event):
        generator = integration._on_click(layer, event)
        next(generator)
        assert not queued
        event.type = "mouse_release"
        with pytest.raises(StopIteration):
            next(generator)
        assert queued
        queued.pop(0)()

    event = SimpleNamespace(
        type="mouse_press",
        position=(11.0, 13.0),
        pos=(11.0, 13.0),
        button=1,
    )
    app._handle_add_click = lambda x, y: calls.append(("add", x, y))
    app._add_mode = True
    release(event)
    assert calls[-1] == ("add", 13.0, 11.0)

    app._add_mode = False
    app._placement_mode = True
    app._handle_placement_click = lambda x, y: calls.append(("track", x, y))
    event.button = 2
    event.type = "mouse_press"
    release(event)
    assert calls[-1] == ("track", 13.0, 11.0)

    app._placement_mode = False
    event.type = "mouse_press"
    release(event)
    assert ("select", 1, 1) in calls
    assert "AB" in integration._shown_labels
    assert layer.editable is False


def test_viewer_model_redraw_is_idempotent_and_relocks_layer() -> None:
    napari = pytest.importorskip("napari")
    viewer = napari.components.ViewerModel()
    overlay = _overlay(centers=[(20.0, 30.0)], radii=[4.0], names=["AB"])
    app = SimpleNamespace(
        viewer=viewer,
        _delete_active_nucleus=lambda: None,
        current_cell_name="",
        get_nucleus_overlay_data=lambda: overlay,
    )
    integration = ViewerIntegration(app)
    integration.setup_layers()
    integration._update_tracking_preview = lambda: None
    integration._update_detector_preview = lambda: None
    integration._update_division_line = lambda: None
    integration._update_ghost_trail = lambda: None

    for _ in range(5):
        integration.update_overlays()

    layer = integration._shapes_layer
    assert layer is not None
    assert len(layer.data) == 1
    np.testing.assert_allclose(np.mean(layer.data[0], axis=0), [20.0, 30.0])
    assert len(layer.edge_color) == len(layer.face_color) == len(layer.edge_width) == 1
    assert layer.editable is False
    assert layer.mode == "pan_zoom"
    assert not layer.selected_data


@pytest.mark.parametrize(
    "overlay",
    [
        _overlay(centers=[], radii=[]),
        _overlay(centers=[(20.0, 30.0)], radii=[0.1]),
    ],
    ids=["empty-frame", "subpixel-projection"],
)
def test_empty_marker_sets_still_clear_division_and_trail_layers(overlay) -> None:
    layer = _FakeShapesLayer()
    integration = _fake_integration(layer, overlay)
    division = SimpleNamespace(data=["stale division"])
    trail = SimpleNamespace(data=["stale trail"])
    calls: list[str] = []

    def clear_division() -> None:
        calls.append("division")
        division.data = []

    def clear_trail() -> None:
        calls.append("trail")
        trail.data = []

    integration._update_division_line = clear_division
    integration._update_ghost_trail = clear_trail
    integration.update_overlays()

    assert layer.data == []
    assert division.data == []
    assert trail.data == []
    assert calls == ["division", "trail"]
    assert layer.editable is False


def test_partial_add_failure_restores_complete_previous_marker_state(caplog) -> None:
    layer = _FakeShapesLayer()
    previous_data = [np.array(shape, copy=True) for shape in layer.data]
    previous_edge_color = layer.edge_color.copy()
    previous_face_color = layer.face_color.copy()
    previous_edge_width = list(layer.edge_width)
    previous_text = dict(layer.text)
    layer.fail_next_add = True

    overlay = _overlay(
        centers=[(20.0, 30.0), (40.0, 50.0)],
        radii=[4.0, 5.0],
        names=["NewA", "NewB"],
    )
    integration = _fake_integration(layer, overlay)
    auxiliary_updates: list[str] = []
    integration._update_division_line = lambda: auxiliary_updates.append("division")
    integration._update_ghost_trail = lambda: auxiliary_updates.append("trail")

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="previous complete marker set"):
            integration.update_overlays()

    assert layer.partial_adds == 1
    assert len(layer.data) == len(previous_data) == 1
    np.testing.assert_allclose(layer.data[0], previous_data[0])
    assert layer.shape_type == ["polygon"]
    np.testing.assert_allclose(layer.edge_color, previous_edge_color)
    np.testing.assert_allclose(layer.face_color, previous_face_color)
    assert layer.edge_width == previous_edge_width
    assert layer.text == previous_text
    assert "restored the prior complete marker set" in caplog.text
    assert auxiliary_updates == []
    assert layer.editable is False
    assert layer.mode == "pan_zoom"
    assert not layer.selected_data


def test_partial_add_and_rollback_failure_is_never_silenced(monkeypatch) -> None:
    layer = _FakeShapesLayer()
    layer.fail_next_add = True
    integration = _fake_integration(
        layer,
        _overlay(centers=[(20.0, 30.0)], radii=[4.0]),
    )

    def broken_restore(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated rollback failure")

    monkeypatch.setattr(integration, "_restore_curated_shapes", broken_restore)

    with pytest.raises(RuntimeError, match="redraw and rollback both failed"):
        integration.update_overlays()

    assert layer.editable is False
    assert layer.mode == "pan_zoom"
    assert not layer.selected_data
