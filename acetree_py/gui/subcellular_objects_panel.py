"""Browse-first dock for the manually curated subcellular object stream.

The panel is intentionally useful before authoring is enabled: classes,
tracks, frame/review state, association, filtering, and sidecar protection are
all inspectable.  Editing controls remain present but disabled in browse-only
or protected/read-only sessions so the UI never suggests that a mutation was
accepted when it cannot be saved.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import AceTreeApp


class RoiInteractionMode(str, Enum):
    INSPECT = "inspect"
    DRAW_POLYGON = "draw_polygon"
    DRAW_POLYLINE = "draw_polyline"
    DRAW_CONTOUR_STACK = "draw_contour_stack"
    EDIT = "edit"


@dataclass(frozen=True)
class ObjectBrowserRow:
    object_id: Any
    class_id: Any
    label: str
    association: str
    first_time: int | None
    last_time: int | None
    expected_start_time: int | None
    expected_end_time: int | None
    segmented_count: int
    reviewed_count: int
    needs_review: bool
    current_state: str
    current_time: int
    reviewed_decision_count: int = 0
    expected_count: int = 0

    @property
    def status_text(self) -> str:
        if self.needs_review:
            return "Needs review"
        if self.expected_count and self.reviewed_decision_count == self.expected_count:
            return "Complete"
        return "In progress"

    @property
    def display_text(self) -> str:
        glyph = "!" if self.needs_review else "✓" if self.status_text == "Complete" else "•"
        association = f"  {self.association}" if self.association else "  Unassociated"
        if self.first_time is None:
            span = "No frames"
        elif self.first_time == self.last_time:
            span = f"t{self.first_time}"
        else:
            span = f"t{self.first_time}–{self.last_time}"
        return (
            f"{glyph} {self.label}{association}  {span}  "
            f"{self.reviewed_decision_count}/{self.expected_count} reviewed"
        )


def _items(manager: Any, public: str, document_field: str) -> tuple[Any, ...]:
    value = getattr(manager, public, None)
    if callable(value):
        value = value()
    if value is None:
        document = getattr(manager, "document", None)
        value = getattr(document, document_field, ()) if document is not None else ()
    return tuple(value or ())


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def object_browser_rows(manager: Any, current_time: int) -> tuple[ObjectBrowserRow, ...]:
    """Build deterministic, Qt-free rows from a manager-like object."""

    classes = {
        getattr(item, "class_id", None): item
        for item in _items(manager, "classes", "object_classes")
    }
    rows: list[ObjectBrowserRow] = []
    for track in _items(manager, "objects", "objects"):
        frames = getattr(track, "frames", {}) or {}
        object_class = classes.get(getattr(track, "class_id", None))
        class_name = getattr(object_class, "name", "Unknown class")
        index = getattr(track, "instance_index", "?")
        segmented = [
            frame
            for frame in frames.values()
            if _enum_value(getattr(frame, "presence", "")) == "segmented"
        ]
        reviewed = [
            frame
            for frame in segmented
            if _enum_value(getattr(frame, "review_state", "")) == "reviewed"
        ]
        needs_review = any(
            _enum_value(getattr(frame, "review_state", "")) == "needs_review"
            for frame in frames.values()
        )
        current = frames.get(current_time)
        times = sorted(int(timepoint) for timepoint in frames)
        expected_start = getattr(track, "expected_start_time", None)
        expected_end = getattr(track, "expected_end_time", None)
        span_start = expected_start if expected_start is not None else (times[0] if times else None)
        span_end = expected_end if expected_end is not None else (times[-1] if times else None)
        expected_count = (
            max(0, span_end - span_start + 1)
            if span_start is not None and span_end is not None else 0
        )
        rows.append(
            ObjectBrowserRow(
                object_id=getattr(track, "object_id", None),
                class_id=getattr(track, "class_id", None),
                label=f"{class_name} #{index}",
                association=_frame_association(current),
                first_time=times[0] if times else None,
                last_time=times[-1] if times else None,
                expected_start_time=expected_start,
                expected_end_time=expected_end,
                segmented_count=len(segmented),
                reviewed_count=len(reviewed),
                needs_review=needs_review,
                current_state=_frame_state(current),
                current_time=current_time,
                reviewed_decision_count=sum(
                    _enum_value(getattr(frame, "review_state", "")) == "reviewed"
                    for frame in frames.values()
                ),
                expected_count=expected_count,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.label.casefold(), str(row.object_id))))


def _frame_state(frame: Any) -> str:
    if frame is None:
        return "missing"
    if _enum_value(getattr(frame, "presence", "")) == "absent":
        return "absent"
    return _enum_value(getattr(frame, "review_state", "draft"))


def _frame_association(frame: Any) -> str:
    cell_ref = getattr(frame, "cell_ref", None)
    name = getattr(cell_ref, "name_snapshot", "") or ""
    if name:
        return name
    anchor = getattr(cell_ref, "nucleus_anchor", None)
    index = getattr(anchor, "index", None)
    return "" if index is None else f"Nucleus {index}"


class ObjectBrowserModel:
    """Reuse immutable document summaries while projecting the current time."""

    def __init__(self) -> None:
        self._document: Any = None
        self._revision: Any = None
        self._time: int | None = None
        self._rows: tuple[ObjectBrowserRow, ...] = ()
        self._tracks: dict[Any, Any] = {}

    def rows(self, manager: Any, current_time: int) -> tuple[ObjectBrowserRow, ...]:
        document = getattr(manager, "document", None)
        revision = getattr(manager, "roi_revision", None)
        if document is None or document is not self._document or revision != self._revision:
            self._rows = object_browser_rows(manager, current_time)
            self._tracks = {
                track.object_id: track for track in _items(manager, "objects", "objects")
            }
            self._document = document
            self._revision = revision
        elif current_time != self._time:
            self._rows = tuple(
                replace(
                    row,
                    current_time=current_time,
                    current_state=_frame_state(self._tracks[row.object_id].frames.get(current_time)),
                    association=_frame_association(
                        self._tracks[row.object_id].frames.get(current_time)
                    ),
                )
                for row in self._rows
            )
        self._time = current_time
        return self._rows


def filter_object_rows(
    rows: Iterable[ObjectBrowserRow],
    *,
    class_id: Any = None,
    state: str = "all",
    cell_scope: str = "all",
    selected_cell_name: str = "",
    search: str = "",
) -> tuple[ObjectBrowserRow, ...]:
    needle = search.strip().casefold()
    state = state.casefold()
    result: list[ObjectBrowserRow] = []
    for row in rows:
        if class_id is not None and row.class_id != class_id:
            continue
        if state != "all" and row.current_state != state:
            continue
        if cell_scope == "current" and (
            not selected_cell_name or row.association != selected_cell_name
        ):
            continue
        haystack = f"{row.label} {row.association} {row.status_text}".casefold()
        if needle and needle not in haystack:
            continue
        result.append(row)
    return tuple(result)


try:
    from qtpy.QtCore import Qt, Signal
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


class SubcellularObjectsPanel(QWidget):  # type: ignore[misc]
    """Right-side Objects dock content with an explicit protection state."""

    if _QT_AVAILABLE:
        objectSelected = Signal(object)
        actionRequested = Signal(str, object)
        modeChanged = Signal(str)

    def __init__(
        self,
        app: AceTreeApp | Any,
        parent=None,
        *,
        browse_only: bool = True,
    ) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")
        super().__init__(parent)
        self.app = app
        self.manager = getattr(app, "roi_manager", app)
        self.current_object_id: Any = None
        self.mode = RoiInteractionMode.INSPECT
        self._requested_browse_only = bool(browse_only)
        self._rows: dict[Any, ObjectBrowserRow] = {}
        self._visible_object_ids: frozenset[Any] = frozenset()
        self._browser_model = ObjectBrowserModel()
        self._rows_snapshot: tuple[ObjectBrowserRow, ...] | None = None
        self._class_snapshot: tuple[Any, ...] | None = None
        self._filtered_cell_name = ""
        self._build_ui()
        self.refresh()

    @property
    def browse_only(self) -> bool:
        return (
            self._requested_browse_only
            or bool(getattr(self.manager, "read_only", False))
            or bool(getattr(self.manager, "is_write_protected", False))
        )

    def set_browse_only(self, value: bool, reason: str = "") -> None:
        self._requested_browse_only = bool(value)
        self._apply_editability(reason)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setAccessibleName("Object tools and details")
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._scroll_area.setWidget(content)
        outer.addWidget(self._scroll_area, 1)

        self._context_label = QLabel()
        self._context_label.setWordWrap(True)
        self._context_label.setAccessibleName("Current ROI view context")
        layout.addWidget(self._context_label)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class"))
        self._class_combo = QComboBox()
        self._class_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._class_combo.setMinimumContentsLength(10)
        self._class_combo.setAccessibleName("Object class filter and authoring class")
        class_row.addWidget(self._class_combo, 1)
        self._btn_manage_classes = self._button("Manage...", "Manage object classes")
        class_row.addWidget(self._btn_manage_classes)
        layout.addLayout(class_row)

        draw_row = QHBoxLayout()
        self._btn_polygon = self._button("Polygon", "Draw a two-dimensional polygon")
        self._btn_polyline = self._button("Thick line", "Draw a thick polyline")
        self._btn_contours = self._button("3D contours", "Draw a contour stack")
        for button in (self._btn_polygon, self._btn_polyline, self._btn_contours):
            draw_row.addWidget(button)
        layout.addLayout(draw_row)

        self._mode_label = QLabel("MODE: INSPECT")
        self._mode_label.setWordWrap(True)
        self._mode_label.setAccessibleName("ROI interaction mode")
        layout.addWidget(self._mode_label)
        self._drawing_actions = QWidget()
        mode_actions = QHBoxLayout(self._drawing_actions)
        mode_actions.setContentsMargins(0, 0, 0, 0)
        self._btn_finish_drawing = self._button(
            "Finish", "Finish and commit the active ROI drawing",
        )
        self._btn_cancel_drawing = self._button(
            "Cancel", "Cancel the active ROI drawing without changing annotations",
        )
        mode_actions.addWidget(self._btn_finish_drawing)
        mode_actions.addWidget(self._btn_cancel_drawing)
        layout.addWidget(self._drawing_actions)
        self._drawing_actions.hide()

        self._show_checkbox = QCheckBox("Show ROIs")
        self._show_checkbox.setChecked(True)
        self._show_checkbox.setAccessibleName("Show subcellular ROI overlay")
        filter_row = QHBoxLayout()
        filter_row.addWidget(self._show_checkbox)
        self._cell_combo = QComboBox()
        self._cell_combo.addItem("All cells", userData="all")
        self._cell_combo.addItem("Current cell", userData="current")
        self._cell_combo.setAccessibleName("Cell association filter")
        filter_row.addWidget(self._cell_combo, 1)
        self._state_combo = QComboBox()
        for label, value in (
            ("All states", "all"), ("Missing", "missing"), ("Draft", "draft"),
            ("Reviewed", "reviewed"), ("Needs review", "needs_review"), ("Absent", "absent"),
        ):
            self._state_combo.addItem(label, userData=value)
        self._state_combo.setAccessibleName("Frame state filter")
        filter_row.addWidget(self._state_combo, 1)
        layout.addLayout(filter_row)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search class, index, or associated cell")
        self._search_edit.setAccessibleName("Search subcellular objects")
        layout.addWidget(self._search_edit)
        self._track_list = QListWidget()
        self._track_list.setAccessibleName("Subcellular object tracks")
        self._track_list.setMinimumHeight(100)
        self._track_list.setMaximumHeight(130)
        layout.addWidget(self._track_list)
        self._empty_label = QLabel()
        self._empty_label.setWordWrap(True)
        self._empty_label.setAccessibleName("Object filter status")
        layout.addWidget(self._empty_label)

        # Primary actions remain visible while the secondary inspector scrolls.
        action_grid = QGridLayout()
        self._btn_edit = self._button("Edit geometry", "Edit selected ROI geometry")
        self._btn_measure = self._button("Measure", "Measure selected ROI")
        self._btn_plot = self._button("Plot track", "Plot selected ROI scalar track")
        self._btn_profiles = self._button("Plot profiles", "Plot selected ROI spatial profiles")
        for index, button in enumerate((
            self._btn_edit, self._btn_measure, self._btn_plot, self._btn_profiles,
        )):
            action_grid.addWidget(button, index // 2, index % 2)
        outer.addLayout(action_grid)

        selected = QGroupBox("Selected object")
        selected_form = QFormLayout(selected)
        self._identity_label = QLabel("None")
        self._geometry_label = QLabel("-")
        self._association_label = QLabel("-")
        self._span_label = QLabel("-")
        self._frame_state_label = QLabel("-")
        for label in (self._identity_label, self._geometry_label, self._association_label,
                      self._span_label, self._frame_state_label):
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        selected_form.addRow("Identity", self._identity_label)
        selected_form.addRow("Geometry", self._geometry_label)
        selected_form.addRow("Association", self._association_label)
        span_row = QHBoxLayout()
        span_row.addWidget(self._span_label, 1)
        self._btn_span = self._button("Set span...", "Set expected object time span")
        span_row.addWidget(self._btn_span)
        selected_form.addRow("Expected span", span_row)
        selected_form.addRow("Frame state", self._frame_state_label)
        layout.addWidget(selected)

        association_row = QHBoxLayout()
        self._btn_use_cell = self._button("Use selected cell", "Associate with selected cell")
        self._btn_pick_cell = self._button("Pick cell", "Pick cell association from image")
        self._btn_clear_cell = self._button("Clear", "Clear cell association")
        for button in (self._btn_use_cell, self._btn_pick_cell, self._btn_clear_cell):
            association_row.addWidget(button)
        layout.addLayout(association_row)

        review_row = QHBoxLayout()
        self._btn_review = self._button("Mark reviewed", "Mark selected ROI frame reviewed")
        self._btn_absent = self._button("Mark absent", "Mark ROI explicitly absent")
        review_row.addWidget(self._btn_review)
        review_row.addWidget(self._btn_absent)
        layout.addLayout(review_row)
        temporal_row = QHBoxLayout()
        self._btn_previous = self._button("Previous", "Previous segmented ROI frame")
        self._btn_next = self._button("Next", "Next segmented ROI frame")
        self._btn_copy = self._button("Copy previous", "Copy previous geometry as draft")
        for button in (self._btn_previous, self._btn_next, self._btn_copy):
            temporal_row.addWidget(button)
        layout.addLayout(temporal_row)
        self._btn_delete = self._button("Delete frame...", "Delete selected ROI frame")
        layout.addWidget(self._btn_delete)
        self._file_status_label = QLabel()
        self._file_status_label.setWordWrap(True)
        self._file_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._file_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._file_status_label.setAccessibleName("ROI file status")
        layout.addWidget(self._file_status_label)
        layout.addStretch(1)

        self._class_combo.currentIndexChanged.connect(self._apply_filters)
        self._cell_combo.currentIndexChanged.connect(self._apply_filters)
        self._state_combo.currentIndexChanged.connect(self._apply_filters)
        self._search_edit.textChanged.connect(self._apply_filters)
        self._show_checkbox.toggled.connect(self._on_visibility_changed)
        self._track_list.currentItemChanged.connect(self._on_selection_changed)
        self._btn_polygon.clicked.connect(
            lambda: self.set_mode(RoiInteractionMode.DRAW_POLYGON)
        )
        self._btn_polyline.clicked.connect(
            lambda: self.set_mode(RoiInteractionMode.DRAW_POLYLINE)
        )
        self._btn_contours.clicked.connect(
            lambda: self.set_mode(RoiInteractionMode.DRAW_CONTOUR_STACK)
        )
        for name, button in self._action_buttons().items():
            button.clicked.connect(lambda _checked=False, action=name: self._emit_action(action))

    @staticmethod
    def _button(text: str, accessible_name: str) -> Any:
        button = QPushButton(text)
        button.setAccessibleName(accessible_name)
        button.setToolTip(accessible_name)
        return button

    def _action_buttons(self) -> dict[str, Any]:
        return {
            "use_selected_cell": self._btn_use_cell,
            "pick_cell": self._btn_pick_cell,
            "clear_association": self._btn_clear_cell,
            "manage_classes": self._btn_manage_classes,
            "set_span": self._btn_span,
            "mark_reviewed": self._btn_review,
            "previous": self._btn_previous,
            "next": self._btn_next,
            "copy_previous": self._btn_copy,
            "mark_absent": self._btn_absent,
            "edit": self._btn_edit,
            "measure": self._btn_measure,
            "plot_track": self._btn_plot,
            "plot_profiles": self._btn_profiles,
            "delete_frame": self._btn_delete,
            "finish_drawing": self._btn_finish_drawing,
            "cancel_drawing": self._btn_cancel_drawing,
        }

    def _emit_action(self, action: str) -> None:
        self.actionRequested.emit(action, self.current_object_id)

    def _on_visibility_changed(self, visible: bool) -> None:
        integration = getattr(self.app, "_roi_viewer_integration", None)
        setter = getattr(integration, "set_overlay_visible", None)
        if callable(setter):
            setter(visible)
        else:
            layer = getattr(integration, "overlay_layer", None)
            if layer is not None:
                layer.visible = bool(visible)
        self.actionRequested.emit("set_visibility", bool(visible))

    def set_mode(self, mode: RoiInteractionMode | str) -> bool:
        mode = RoiInteractionMode(mode)
        if self.browse_only and mode is not RoiInteractionMode.INSPECT:
            return False
        self.mode = mode
        self._drawing_actions.setVisible(mode is not RoiInteractionMode.INSPECT)
        self._mode_label.setText(f"MODE: {mode.value.replace('_', ' ').upper()}")
        self._apply_editability()
        self.modeChanged.emit(mode.value)
        return True

    @property
    def selected_class_id(self) -> Any:
        """Concrete authoring class, falling back to the selected track."""

        class_id = self._class_combo.currentData()
        if class_id is not None:
            return class_id
        row = self._rows.get(self.current_object_id)
        return None if row is None else row.class_id

    def select_class(self, class_id: Any) -> bool:
        for index in range(self._class_combo.count()):
            if self._class_combo.itemData(index) == class_id:
                self._class_combo.setCurrentIndex(index)
                return True
        return False

    def refresh(self) -> None:
        current_time = int(getattr(self.app, "current_time", 1))
        current_plane = int(getattr(self.app, "current_plane", 1))
        selected_cell = str(getattr(self.app, "current_cell_name", "") or "None")
        self._context_label.setText(
            f"t={current_time}  z={current_plane}  Selected cell: {selected_cell}"
        )
        classes = _items(self.manager, "classes", "object_classes")
        if classes != self._class_snapshot:
            self._refresh_classes()
            self._class_snapshot = classes
        rows = self._browser_model.rows(self.manager, current_time)
        cell_name = str(getattr(self.app, "current_cell_name", "") or "")
        if rows is not self._rows_snapshot or cell_name != self._filtered_cell_name:
            self._rows_snapshot = rows
            self._rows = {row.object_id: row for row in rows}
            self._apply_filters()
        path = getattr(self.manager, "sidecar_path", None)
        state = "Unsaved" if bool(getattr(self.manager, "is_dirty", False)) else "Saved"
        if getattr(self.manager, "load_error", None):
            state = "Protected — load error"
        self._file_status_label.setText(
            f"ROI file: {state}" + (f" — {path}" if path else "")
        )
        self._apply_editability()

    def _refresh_classes(self) -> None:
        selected = self._class_combo.currentData()
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("All", userData=None)
        for item in sorted(
            _items(self.manager, "classes", "object_classes"),
            key=lambda value: str(getattr(value, "name", "")).casefold(),
        ):
            self._class_combo.addItem(
                str(getattr(item, "name", "Unnamed")),
                userData=getattr(item, "class_id", None),
            )
        for index in range(self._class_combo.count()):
            if self._class_combo.itemData(index) == selected:
                self._class_combo.setCurrentIndex(index)
                break
        self._class_combo.blockSignals(False)

    def _apply_filters(self) -> None:
        selected = self.current_object_id
        scroll_position = self._track_list.verticalScrollBar().value()
        self._filtered_cell_name = str(getattr(self.app, "current_cell_name", "") or "")
        visible = filter_object_rows(
            self._rows.values(),
            class_id=self._class_combo.currentData(),
            state=str(self._state_combo.currentData() or "all"),
            cell_scope=str(self._cell_combo.currentData() or "all"),
            selected_cell_name=str(getattr(self.app, "current_cell_name", "") or ""),
            search=self._search_edit.text(),
        )
        self._visible_object_ids = frozenset(row.object_id for row in visible)
        self._track_list.blockSignals(True)
        self._track_list.clear()
        for row in visible:
            item = QListWidgetItem(row.display_text)
            item.setData(Qt.UserRole, row.object_id)
            item.setToolTip(f"{row.label}; current frame: {row.current_state}")
            self._track_list.addItem(item)
            if row.object_id == selected:
                self._track_list.setCurrentItem(item)
        self._track_list.blockSignals(False)
        self.select_object(selected)
        self._track_list.verticalScrollBar().setValue(scroll_position)
        no_cell = (
            self._cell_combo.currentData() == "current"
            and not getattr(self.app, "current_cell_name", "")
        )
        self._empty_label.setText(
            "Select a cell to see its associated objects."
            if no_cell else "No objects match these filters."
        )
        self._empty_label.setVisible(not visible)
        integration = getattr(self.app, "_roi_viewer_integration", None)
        setter = getattr(integration, "set_visible_object_ids", None)
        if callable(setter):
            setter(self._visible_object_ids)

    def select_object(self, object_id: Any) -> None:
        selected = object_id if object_id in self._visible_object_ids else None
        self._track_list.blockSignals(True)
        self._track_list.setCurrentItem(None)
        for index in range(self._track_list.count()):
            item = self._track_list.item(index)
            if item.data(Qt.UserRole) == selected:
                self._track_list.setCurrentItem(item)
                break
        self._track_list.blockSignals(False)
        self._set_selected_object(selected)

    def _on_selection_changed(self, current: Any, _previous: Any) -> None:
        self._set_selected_object(current.data(Qt.UserRole) if current is not None else None)

    def _set_selected_object(self, object_id: Any) -> None:
        changed = object_id != self.current_object_id
        self.current_object_id = object_id
        if changed and self.mode is not RoiInteractionMode.INSPECT:
            integration = getattr(self.app, "_roi_viewer_integration", None)
            cancel = getattr(integration, "cancel_edit", None)
            if callable(cancel):
                cancel()
            if self.mode is not RoiInteractionMode.INSPECT:
                self.set_mode(RoiInteractionMode.INSPECT)
        self._refresh_inspector()
        if changed:
            self.objectSelected.emit(object_id)

    def _refresh_inspector(self) -> None:
        row = self._rows.get(self.current_object_id)
        if row is None:
            for label in (
                self._identity_label,
                self._geometry_label,
                self._association_label,
                self._span_label,
                self._frame_state_label,
            ):
                label.setText("—")
            self._apply_editability()
            return
        self._identity_label.setText(row.label)
        self._association_label.setText(row.association or "Unassociated")
        state_text = row.current_state.replace("_", " ").title()
        start = row.expected_start_time
        end = row.expected_end_time
        self._span_label.setText(
            "—" if start is None and end is None else f"t={start or '?'}–{end or '?'}"
        )
        geometry_name = "—"
        getter = getattr(self.manager, "get_object", None)
        track = getter(row.object_id) if callable(getter) else None
        if track is not None:
            frame = getattr(track, "frames", {}).get(row.current_time)
            geometry = getattr(frame, "geometry", None)
            if geometry is not None:
                geometry_name = type(geometry).__name__
            if row.current_state == "absent":
                review = _enum_value(getattr(frame, "review_state", "draft"))
                state_text += f" ({review.replace('_', ' ')})"
        self._frame_state_label.setText(state_text)
        self._geometry_label.setText(geometry_name)
        self._apply_editability()

    def _apply_editability(self, reason: str = "") -> None:
        protected = self.browse_only
        has_selection = self.current_object_id is not None
        mutation_buttons = (
            self._btn_use_cell,
            self._btn_pick_cell,
            self._btn_clear_cell,
            self._btn_manage_classes,
            self._btn_polygon,
            self._btn_polyline,
            self._btn_contours,
            self._btn_review,
            self._btn_copy,
            self._btn_absent,
            self._btn_edit,
            self._btn_delete,
            self._btn_span,
        )
        for button in mutation_buttons:
            button.setEnabled(not protected and (has_selection or button in (
                self._btn_manage_classes,
                self._btn_polygon,
                self._btn_polyline,
                self._btn_contours,
            )))
            if protected:
                button.setToolTip(reason or "Browse-only: ROI annotations cannot be changed")
        active_authoring = self.mode is not RoiInteractionMode.INSPECT
        self._btn_finish_drawing.setEnabled(active_authoring and not protected)
        # Cancellation must remain available if protection changes mid-gesture.
        self._btn_cancel_drawing.setEnabled(active_authoring)
        self._btn_measure.setEnabled(has_selection)
        self._btn_plot.setEnabled(has_selection)
        self._btn_profiles.setEnabled(has_selection)
        self._btn_previous.setEnabled(has_selection)
        self._btn_next.setEnabled(has_selection)

    def next_instance_index(self, class_id: Any) -> int | None:
        """Return the persisted allocator value displayed for new objects."""

        getter = getattr(self.manager, "get_class", None)
        item = getter(class_id) if callable(getter) else None
        return getattr(item, "next_instance_index", None)

    @staticmethod
    def shortcuts_allowed(focus_widget: Any) -> bool:
        """Global letter shortcuts are unsafe while a value editor has focus."""

        if not _QT_AVAILABLE or focus_widget is None:
            return True
        return not isinstance(focus_widget, (QLineEdit, QComboBox))


__all__ = [
    "ObjectBrowserModel",
    "ObjectBrowserRow",
    "RoiInteractionMode",
    "SubcellularObjectsPanel",
    "filter_object_rows",
    "object_browser_rows",
]
