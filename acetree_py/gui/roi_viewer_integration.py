"""Napari boundary for subcellular ROI projections and transient editing.

Authoritative geometry always lives in ``RoiManager`` in model ``(x, y)``
coordinates.  The permanent overlay is rebuilt read-only for the current
time/Z, while the editor layer contains at most one staged geometry and emits
one undo command only when the user finishes the gesture.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import logging
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from ..editing.roi_commands import SetRoiFrameGeometry

logger = logging.getLogger(__name__)

OVERLAY_LAYER_NAME = "Subcellular ROI Overlay"
EDITOR_LAYER_NAME = "Subcellular ROI Editor"


def model_xy_to_napari(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert model/display ``(x, y)`` points to napari ``(y, x)``."""

    array = np.asarray(tuple(tuple(point) for point in points), dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("ROI points must be an N x 2 sequence")
    return array[:, ::-1].copy()


def napari_yx_to_model(points: Iterable[Iterable[float]]) -> tuple[tuple[float, float], ...]:
    """Convert napari ``(y, x)`` points to immutable model ``(x, y)``."""

    array = np.asarray(tuple(tuple(point) for point in points), dtype=float)
    if array.size == 0:
        return ()
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Napari ROI points must be an N x 2 sequence")
    return tuple((float(x), float(y)) for y, x in array)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _manager_items(manager: Any, public: str, document_field: str) -> tuple[Any, ...]:
    value = getattr(manager, public, None)
    if callable(value):
        value = value()
    if value is None:
        document = getattr(manager, "document", None)
        value = getattr(document, document_field, ()) if document is not None else ()
    return tuple(value or ())


def _geometry_kind(geometry: Any) -> str:
    if geometry is None:
        return ""
    if isinstance(geometry, dict):
        return str(geometry.get("kind", ""))
    name = type(geometry).__name__.lower()
    if hasattr(geometry, "points_xy_px") or "polyline" in name:
        return "thick_polyline_2d"
    if hasattr(geometry, "slices") or "contourstack" in name:
        return "contour_stack_3d"
    if hasattr(geometry, "exterior_xy_px") or "polygon" in name:
        return "polygon_2d"
    return name


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _contour_at(geometry: Any, z_plane: int) -> Any:
    for contour in _field(geometry, "slices", ()):
        if int(_field(contour, "z_plane", -1)) == int(z_plane):
            return contour
    return None


@dataclass(frozen=True)
class RoiOverlayShape:
    data_yx: np.ndarray
    shape_type: str
    edge_color: tuple[float, float, float, float]
    edge_width: float
    object_id: Any
    frame_id: Any
    label: str


def roi_overlay_shapes(
    manager: Any,
    timepoint: int,
    z_plane: int,
    *,
    object_ids: Iterable[Any] | None = None,
) -> tuple[RoiOverlayShape, ...]:
    """Project model records for exactly one current time and Z plane."""

    classes = {
        getattr(value, "class_id", None): value
        for value in _manager_items(manager, "classes", "object_classes")
    }
    visible_ids = None if object_ids is None else frozenset(str(item) for item in object_ids)
    result: list[RoiOverlayShape] = []
    for track in _manager_items(manager, "objects", "objects"):
        if visible_ids is not None and str(getattr(track, "object_id", None)) not in visible_ids:
            continue
        frame = (getattr(track, "frames", {}) or {}).get(int(timepoint))
        if frame is None or _enum_value(getattr(frame, "presence", "")) != "segmented":
            continue
        geometry = getattr(frame, "geometry", None)
        kind = _geometry_kind(geometry)
        shape_type = "polygon"
        width = 2.0
        if kind == "polygon_2d":
            if int(_field(geometry, "z_plane", -1)) != int(z_plane):
                continue
            points = _field(geometry, "exterior_xy_px", ())
        elif kind == "thick_polyline_2d":
            if int(_field(geometry, "z_plane", -1)) != int(z_plane):
                continue
            points = _field(geometry, "points_xy_px", ())
            shape_type = "path"
            thickness = _field(geometry, "thickness")
            value = float(_field(thickness, "value", 1.0))
            unit = _enum_value(_field(thickness, "unit", "px"))
            if unit in {"um", "µm"}:
                document = getattr(manager, "document", None)
                coordinate_space = getattr(document, "coordinate_space", None)
                xy_res = getattr(coordinate_space, "xy_res", None)
                if xy_res:
                    value /= float(xy_res)
            width = max(1.0, value)
        elif kind == "contour_stack_3d":
            contour = _contour_at(geometry, z_plane)
            if contour is None:
                continue
            points = _field(contour, "exterior_xy_px", ())
        else:
            continue
        data = model_xy_to_napari(points)
        if len(data) < (2 if shape_type == "path" else 3):
            continue
        object_class = classes.get(getattr(track, "class_id", None))
        color = tuple(getattr(object_class, "color_rgba", (0.95, 0.6, 0.1, 1.0)))
        name = getattr(object_class, "name", "ROI")
        label = f"{name} #{getattr(track, 'instance_index', '?')}"
        result.append(
            RoiOverlayShape(
                data_yx=data,
                shape_type=shape_type,
                edge_color=color,  # type: ignore[arg-type]
                edge_width=width,
                object_id=getattr(track, "object_id", None),
                frame_id=getattr(frame, "frame_id", None),
                label=label,
            )
        )
    return tuple(result)


@dataclass
class _EditorSession:
    object_id: Any
    timepoint: int
    z_plane: int
    geometry: Any | None
    kind: str
    class_id: Any = None
    cell_ref: Any = None
    create_object: bool = False
    drawing: bool = False


class RoiViewerIntegration:
    """Own the model-derived overlay and a single transient editor layer."""

    def __init__(
        self,
        app: Any,
        roi_manager: Any | None = None,
        *,
        command_sink: Any | None = None,
    ) -> None:
        self.app = app
        self.manager = roi_manager or getattr(app, "roi_manager", None)
        self.command_sink = command_sink
        self.overlay_layer: Any = None
        self.editor_layer: Any = None
        # Compatibility aliases for call sites that keep layer fields private.
        self._overlay_layer: Any = None
        self._editor_layer: Any = None
        self._editor_session: _EditorSession | None = None
        self._three_dimensional = False
        self._overlay_visible = True
        self._visible_object_ids: frozenset[str] | None = None

    @property
    def editing(self) -> bool:
        return self._editor_session is not None

    @property
    def editor_banner(self) -> str:
        session = self._editor_session
        if session is None:
            return "MODE: INSPECT"
        return (
            f"Editing ROI {session.object_id} at t={session.timepoint}, "
            f"z={session.z_plane} — Enter: finish; Escape: cancel"
        )

    def setup_layers(self) -> None:
        viewer = getattr(self.app, "viewer", None)
        if viewer is None:
            return
        if self.overlay_layer is None:
            self.overlay_layer = self._add_shapes_layer(
                viewer,
                name=OVERLAY_LAYER_NAME,
                edge_color="orange",
                edge_width=2,
                visible=self._overlay_visible and not self._three_dimensional,
            )
            self._overlay_layer = self.overlay_layer
        if self.editor_layer is None:
            self.editor_layer = self._add_shapes_layer(
                viewer,
                name=EDITOR_LAYER_NAME,
                edge_color="white",
                edge_width=2,
                visible=False,
            )
            self._editor_layer = self.editor_layer
            self._connect_editor_interactions()
        self._lock_overlay()
        self._clear_editor()
        self.update_overlay()
        # ``viewer.add_shapes`` selects each newly created layer.  The hidden
        # editor is therefore active at this point unless we explicitly hand
        # control back to the Nuclei layer, whose mouse callbacks implement
        # the normal left/right-click cell interactions.
        self._restore_cell_interaction_layer()

    @staticmethod
    def _add_shapes_layer(viewer: Any, **kwargs: Any) -> Any:
        dummy = [np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])]
        try:
            layer = viewer.add_shapes(
                data=dummy,
                shape_type="polygon",
                face_color="transparent",
                opacity=0.95,
                **kwargs,
            )
        except TypeError:
            layer = viewer.add_shapes(data=dummy, shape_type="polygon", **kwargs)
        layer.data = []
        return layer

    def _lock_overlay(self) -> None:
        if self.overlay_layer is None:
            return
        try:
            self.overlay_layer.editable = False
        except Exception:
            pass
        try:
            self.overlay_layer.mode = "pan_zoom"
        except Exception:
            pass

    def _connect_editor_interactions(self) -> None:
        layer = self.editor_layer
        if layer is None:
            return
        callbacks = getattr(layer, "mouse_drag_callbacks", None)
        if callbacks is not None and self._on_editor_drag not in callbacks:
            callbacks.append(self._on_editor_drag)
        binder = getattr(layer, "bind_key", None)
        if not callable(binder):
            return

        @binder("Enter", overwrite=True)
        def _finish(_layer):
            try:
                self.finish_edit()
            except (RuntimeError, TypeError, ValueError) as error:
                self._report_editor_error(error)

        @binder("Escape", overwrite=True)
        def _cancel(_layer):
            self.cancel_edit()

        @binder("Space", overwrite=True)
        def _temporary_pan(bound_layer):
            prior_mode = getattr(bound_layer, "mode", "select")
            bound_layer.mode = "pan_zoom"
            yield
            if self.editing:
                bound_layer.mode = prior_mode

    def _on_editor_drag(self, _layer: Any, event: Any):
        """Commit one command at mouse release, never per mutation event."""

        session = self._editor_session
        if session is None or session.drawing:
            return
        yield
        while getattr(event, "type", "") == "mouse_move":
            yield
        if getattr(event, "type", "") == "mouse_release" and self.editing:
            try:
                self.finish_edit()
            except (RuntimeError, TypeError, ValueError) as error:
                # Invalid staged geometry remains available for repair.
                self._report_editor_error(error)

    def _report_editor_error(self, error: Exception) -> None:
        reporter = getattr(self.app, "_say", None)
        if callable(reporter):
            reporter(str(error))
        else:
            logger.warning("ROI editor commit failed: %s", error)

    def update_overlay(
        self,
        *,
        timepoint: int | None = None,
        z_plane: int | None = None,
    ) -> tuple[RoiOverlayShape, ...]:
        """Atomically redraw the permanent current-time/current-Z projection."""

        if self.overlay_layer is None or self.manager is None:
            return ()
        timepoint = int(timepoint or getattr(self.app, "current_time", 1))
        z_plane = int(z_plane or getattr(self.app, "current_plane", 1))
        shapes = roi_overlay_shapes(
            self.manager, timepoint, z_plane, object_ids=self._visible_object_ids,
        )
        previous = self._snapshot_layer(self.overlay_layer)
        try:
            self.overlay_layer.data = []
            if shapes:
                self.overlay_layer.add(
                    [item.data_yx for item in shapes],
                    shape_type=[item.shape_type for item in shapes],
                    edge_color=[item.edge_color for item in shapes],
                    face_color=[(0.0, 0.0, 0.0, 0.0) for _ in shapes],
                    edge_width=[item.edge_width for item in shapes],
                )
            properties = {
                "object_id": [str(item.object_id) for item in shapes],
                "frame_id": [str(item.frame_id) for item in shapes],
                "label": [item.label for item in shapes],
            }
            if hasattr(self.overlay_layer, "properties"):
                self.overlay_layer.properties = properties
            if hasattr(self.overlay_layer, "text"):
                self.overlay_layer.text = {
                    "string": [item.label for item in shapes],
                    "color": "white",
                    "size": 8,
                    "anchor": "upper_left",
                }
        except Exception as error:
            try:
                self._restore_layer(self.overlay_layer, previous)
            except Exception as restore_error:
                raise RuntimeError(
                    "ROI overlay redraw and rollback both failed; refresh the view"
                ) from restore_error
            raise RuntimeError(
                "ROI overlay redraw failed; the prior complete projection was restored"
            ) from error
        finally:
            self._lock_overlay()
        return shapes

    # Existing app integration uses the plural spelling for its nucleus layer.
    update_overlays = update_overlay
    refresh = update_overlay

    @staticmethod
    def _snapshot_layer(layer: Any) -> dict[str, Any]:
        text = getattr(layer, "text", None)
        if hasattr(text, "dict"):
            text = text.dict()
        return {
            "data": [np.array(item, copy=True) for item in getattr(layer, "data", ())],
            "shape_type": deepcopy(getattr(layer, "shape_type", [])),
            "edge_color": deepcopy(getattr(layer, "edge_color", [])),
            "face_color": deepcopy(getattr(layer, "face_color", [])),
            "edge_width": deepcopy(getattr(layer, "edge_width", [])),
            "properties": deepcopy(getattr(layer, "properties", {})),
            "text": deepcopy(text),
            "visible": bool(getattr(layer, "visible", True)),
        }

    @staticmethod
    def _restore_layer(layer: Any, snapshot: dict[str, Any]) -> None:
        layer.data = [np.array(item, copy=True) for item in snapshot["data"]]
        fields = (
            "shape_type",
            "edge_color",
            "face_color",
            "edge_width",
            "properties",
            "text",
            "visible",
        )
        for name in fields:
            if hasattr(layer, name):
                setattr(layer, name, deepcopy(snapshot[name]))

    def enter_edit_mode(
        self,
        object_id: Any,
        timepoint: int,
        geometry: Any,
        *,
        z_plane: int | None = None,
    ) -> None:
        """Stage one frame/contour in the editor without mutating the model."""

        if self._three_dimensional:
            raise RuntimeError("ROI authoring is available only in the main 2D view")
        if self.editor_layer is None:
            self.setup_layers()
        if self.editor_layer is None:
            raise RuntimeError("ROI editor layer is unavailable")
        self._editor_session = None
        self._clear_editor()
        z_plane = int(z_plane or getattr(self.app, "current_plane", 1))
        kind = _geometry_kind(geometry)
        if kind == "polygon_2d":
            if int(_field(geometry, "z_plane", z_plane)) != z_plane:
                raise ValueError("Polygon is not on the current Z plane")
            points = _field(geometry, "exterior_xy_px", ())
            shape_type = "polygon"
        elif kind == "thick_polyline_2d":
            if int(_field(geometry, "z_plane", z_plane)) != z_plane:
                raise ValueError("Polyline is not on the current Z plane")
            points = _field(geometry, "points_xy_px", ())
            shape_type = "path"
        elif kind == "contour_stack_3d":
            contour = _contour_at(geometry, z_plane)
            if contour is None:
                raise ValueError("Contour stack has no contour on the current Z plane")
            points = _field(contour, "exterior_xy_px", ())
            shape_type = "polygon"
        else:
            raise TypeError(f"Unsupported ROI geometry: {type(geometry).__name__}")
        self.editor_layer.data = []
        self.editor_layer.add(
            [model_xy_to_napari(points)],
            shape_type=shape_type,
            edge_color="white",
            face_color="transparent",
            edge_width=2,
        )
        self.editor_layer.visible = True
        self.editor_layer.editable = True
        try:
            self.editor_layer.mode = "select"
        except Exception:
            pass
        self._editor_session = _EditorSession(
            object_id=object_id,
            timepoint=int(timepoint),
            z_plane=z_plane,
            geometry=geometry,
            kind=kind,
        )
        self._set_space_shortcut_enabled(False)
        self._make_editor_active()

    # Concise alias used by controllers.
    begin_edit = enter_edit_mode

    def begin_drawing(
        self,
        *,
        class_id: Any,
        timepoint: int,
        kind: str,
        z_plane: int,
        object_id: Any = None,
        cell_ref: Any = None,
        geometry: Any = None,
    ) -> Any:
        """Open an empty transient editor for a new object or missing frame."""

        if self._three_dimensional:
            raise RuntimeError("ROI authoring is available only in the main 2D view")
        if self.editor_layer is None:
            self.setup_layers()
        if self.editor_layer is None:
            raise RuntimeError("ROI editor layer is unavailable")
        normalized = str(kind).lower()
        aliases = {
            "draw_polygon": "polygon_2d",
            "polygon": "polygon_2d",
            "draw_polyline": "thick_polyline_2d",
            "polyline": "thick_polyline_2d",
            "draw_contour_stack": "contour_stack_3d",
            "contour_stack": "contour_stack_3d",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {
            "polygon_2d",
            "thick_polyline_2d",
            "contour_stack_3d",
        }:
            raise ValueError(f"Unsupported ROI drawing kind: {kind}")
        self._editor_session = None
        self._clear_editor()
        create_object = object_id is None
        object_id = uuid4() if create_object else object_id
        shape_mode = "add_path" if normalized == "thick_polyline_2d" else "add_polygon"
        self.editor_layer.data = []
        self.editor_layer.visible = True
        self.editor_layer.editable = True
        self.editor_layer.mode = shape_mode
        self._editor_session = _EditorSession(
            object_id=object_id,
            class_id=class_id,
            timepoint=int(timepoint),
            z_plane=int(z_plane),
            geometry=geometry,
            kind=normalized,
            cell_ref=cell_ref,
            create_object=create_object,
            drawing=True,
        )
        self._set_space_shortcut_enabled(False)
        self._make_editor_active()
        return object_id

    def finish_edit(self) -> Any:
        """Validate/commit staged geometry as exactly one history command."""

        session = self._editor_session
        if session is None or self.editor_layer is None:
            raise RuntimeError("No ROI edit is active")
        data = list(getattr(self.editor_layer, "data", ()))
        if len(data) != 1:
            raise ValueError("The transient ROI editor must contain exactly one shape")
        points = napari_yx_to_model(data[0])
        minimum = 2 if session.kind == "thick_polyline_2d" else 3
        if len(set(points)) < minimum:
            raise ValueError(f"ROI geometry requires at least {minimum} distinct points")
        geometry = self._geometry_with_points(session, points)
        geometry_command = SetRoiFrameGeometry(
            manager=self.manager,
            object_id=session.object_id,
            timepoint=session.timepoint,
            geometry=geometry,
            cell_ref=session.cell_ref,
        )
        command: Any = geometry_command
        if session.create_object:
            from ..editing.commands import CompositeCommand
            from ..editing.roi_commands import CreateRoiObject

            command = CompositeCommand(
                commands=[
                    CreateRoiObject(
                        manager=self.manager,
                        class_id=session.class_id,
                        object_id=session.object_id,
                    ),
                    geometry_command,
                ],
                label=f"Create and draw ROI object at t={session.timepoint}",
            )
        self._submit(command)
        self._clear_editor()
        self._editor_session = None
        self._set_space_shortcut_enabled(True)
        self._restore_cell_interaction_layer()
        if session.create_object:
            selected_id = session.object_id
            panel = getattr(self.app, "_subcellular_objects_panel", None)
            if panel is not None:
                panel.refresh()
                panel.select_object(session.object_id)
                selected_id = getattr(panel, "current_object_id", selected_id)
            selector = getattr(self.app, "_on_roi_object_selected", None)
            if callable(selector):
                selector(selected_id)
        self._sync_panel_inspect()
        return command

    commit_edit = finish_edit

    def _geometry_with_points(
        self,
        session: _EditorSession,
        points: tuple[tuple[float, float], ...],
    ) -> Any:
        geometry = session.geometry
        if geometry is None:
            from ..core.subcellular_roi import (
                ContourSlice,
                ContourStack3D,
                Polygon2D,
                ThickPolyline2D,
                Thickness,
            )

            if session.kind == "polygon_2d":
                return Polygon2D(session.z_plane, points)
            if session.kind == "contour_stack_3d":
                return ContourStack3D((ContourSlice(session.z_plane, points),))
            coordinate_space = getattr(
                getattr(self.manager, "document", None),
                "coordinate_space",
                None,
            )
            calibrated = getattr(coordinate_space, "xy_res", None) is not None
            thickness = Thickness(0.8, "um") if calibrated else Thickness(3.0, "px")
            return ThickPolyline2D(session.z_plane, points, thickness)
        if isinstance(geometry, dict):
            updated = deepcopy(geometry)
            if session.kind == "contour_stack_3d":
                for contour in updated.get("slices", ()):
                    if int(contour.get("z_plane", -1)) == session.z_plane:
                        contour["exterior_xy_px"] = points
                        break
            else:
                key = (
                    "points_xy_px"
                    if session.kind == "thick_polyline_2d"
                    else "exterior_xy_px"
                )
                updated[key] = points
            return updated
        if session.kind == "polygon_2d":
            return replace(geometry, exterior_xy_px=points)
        if session.kind == "thick_polyline_2d":
            return replace(geometry, points_xy_px=points)
        if session.kind == "contour_stack_3d":
            contours = []
            found = False
            for contour in geometry.slices:
                if int(contour.z_plane) == session.z_plane:
                    contours.append(replace(contour, exterior_xy_px=points))
                    found = True
                else:
                    contours.append(contour)
            if not found:
                from ..core.subcellular_roi import ContourSlice

                contours.append(ContourSlice(session.z_plane, points))
            return replace(geometry, slices=tuple(contours))
        raise TypeError(f"Unsupported ROI geometry: {type(geometry).__name__}")

    def _submit(self, command: Any) -> None:
        if callable(self.command_sink):
            self.command_sink(command)
            return
        history = getattr(self.app, "edit_history", None)
        if history is None:
            raise RuntimeError("No edit history is available for ROI commit")
        history.do(command)

    def cancel_edit(self) -> None:
        self._editor_session = None
        self._clear_editor()
        self._set_space_shortcut_enabled(True)
        self._restore_cell_interaction_layer()
        self._sync_panel_inspect()

    exit_edit_mode = cancel_edit

    def _sync_panel_inspect(self) -> None:
        panel = getattr(self.app, "_subcellular_objects_panel", None)
        if panel is None:
            return
        mode = getattr(panel, "mode", "inspect")
        if str(getattr(mode, "value", mode)) != "inspect":
            panel.set_mode("inspect")

    def _set_space_shortcut_enabled(self, enabled: bool) -> None:
        shortcut = getattr(self.app, "_space_shortcut", None)
        if shortcut is not None:
            try:
                shortcut.setEnabled(bool(enabled))
            except RuntimeError:
                pass

    def _clear_editor(self) -> None:
        if self.editor_layer is None:
            return
        try:
            self.editor_layer.data = []
            self.editor_layer.editable = False
            self.editor_layer.visible = False
            self.editor_layer.mode = "pan_zoom"
        except Exception:
            logger.exception("Could not fully clear the transient ROI editor layer")

    def _make_editor_active(self) -> None:
        viewer = getattr(self.app, "viewer", None)
        if viewer is None or self.editor_layer is None:
            return
        try:
            viewer.layers.selection.active = self.editor_layer
        except Exception:
            pass

    def _restore_cell_interaction_layer(self) -> None:
        """Return mouse ownership to the curated Nuclei layer."""

        integration = getattr(self.app, "_viewer_integration", None)
        restore = getattr(integration, "_ensure_nuclei_active", None)
        if callable(restore):
            restore()

    def on_view_changed(self, *, reason: str = "View changed") -> None:
        """Leave editing explicitly, then redraw for a time/Z navigation."""

        if self.editing:
            logger.info("Cancelled ROI edit: %s", reason)
            self.cancel_edit()
        self.update_overlay()

    def set_visible_object_ids(self, object_ids: Iterable[Any] | None) -> None:
        """Apply the Objects browser filter to the permanent image projection."""
        visible_ids = None if object_ids is None else frozenset(str(item) for item in object_ids)
        if visible_ids != self._visible_object_ids:
            self._visible_object_ids = visible_ids
            self.update_overlay()

    def set_overlay_visible(self, visible: bool) -> None:
        """Remember explicit visibility independently of temporary 3D suppression."""
        self._overlay_visible = bool(visible)
        if self.overlay_layer is not None:
            self.overlay_layer.visible = self._overlay_visible and not self._three_dimensional

    def set_three_dimensional(self, active: bool) -> None:
        """Keep detached/3D previews read-only and mode-safe."""

        self._three_dimensional = bool(active)
        if active:
            self.cancel_edit()
        if self.overlay_layer is not None:
            self.overlay_layer.visible = self._overlay_visible and not active
        self._lock_overlay()
        if not active:
            self.update_overlay()

    set_3d_mode = set_three_dimensional

    def cleanup(self) -> None:
        """Explicitly leave every ROI authoring mode without touching data."""

        self.cancel_edit()
        self._three_dimensional = False
        self._lock_overlay()


__all__ = [
    "EDITOR_LAYER_NAME",
    "OVERLAY_LAYER_NAME",
    "RoiOverlayShape",
    "RoiViewerIntegration",
    "model_xy_to_napari",
    "napari_yx_to_model",
    "roi_overlay_shapes",
]
