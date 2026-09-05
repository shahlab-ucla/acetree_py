"""Small editors for existing undoable ROI class and track metadata commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

try:
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import (
        QColorDialog,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
    )
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QDialog = object  # type: ignore[misc,assignment]


class RoiClassDialog(QDialog):  # type: ignore[misc]
    """Create, rename, recolor, or delete an unused class through edit history."""

    def __init__(
        self,
        manager: Any,
        command_runner: Callable[[Any], Any],
        *,
        selected_class_id: Any = None,
        parent=None,
    ) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("ROI class editing requires 'acetree-py[gui]'")
        super().__init__(parent)
        self.manager = manager
        self._run_command = command_runner
        self._color_rgba = (0.18, 0.77, 0.71, 1.0)
        self.setWindowTitle("Manage Object Classes")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._class_combo = QComboBox()
        self._class_combo.setAccessibleName("Object class to edit")
        form.addRow("Class", self._class_combo)
        self._name_edit = QLineEdit()
        self._name_edit.setAccessibleName("Object class name")
        form.addRow("Name", self._name_edit)
        self._color_button = QPushButton()
        self._color_button.setAccessibleName("Choose object class color")
        self._color_button.clicked.connect(self._choose_color)
        form.addRow("Color", self._color_button)
        layout.addLayout(form)
        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        actions = QHBoxLayout()
        self._save_button = QPushButton()
        self._save_button.clicked.connect(self._save_class)
        self._delete_button = QPushButton("Delete class")
        self._delete_button.clicked.connect(self._delete_class)
        actions.addWidget(self._save_button)
        actions.addWidget(self._delete_button)
        layout.addLayout(actions)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self._class_combo.currentIndexChanged.connect(self._load_class)
        self._refresh_classes(selected_class_id)

    @property
    def selected_class_id(self) -> Any:
        return self._class_combo.currentData()

    def _refresh_classes(self, class_id: Any) -> None:
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("New class…", None)
        for item in sorted(self.manager.classes, key=lambda value: value.name.casefold()):
            self._class_combo.addItem(item.name, item.class_id)
        self._class_combo.setCurrentIndex(max(0, self._class_combo.findData(class_id)))
        self._class_combo.blockSignals(False)
        self._load_class()

    def _load_class(self) -> None:
        current = self.manager.get_class(self.selected_class_id)
        self._name_edit.setText("" if current is None else current.name)
        self._color_rgba = (0.18, 0.77, 0.71, 1.0) if current is None else current.color_rgba
        self._refresh_color()
        used = sum(item.class_id == self.selected_class_id for item in self.manager.objects)
        protected = self.manager.is_write_protected
        self._save_button.setText("Create class" if current is None else "Save changes")
        self._save_button.setEnabled(not protected)
        self._delete_button.setEnabled(current is not None and not used and not protected)
        self._delete_button.setToolTip("Only classes with no object tracks can be deleted.")
        self._status.setText(
            "ROI annotations are protected from editing." if protected else
            f"Used by {used} object track(s). Classes containing objects cannot be deleted."
            if used else "Class changes can be undone with Undo after closing this dialog."
        )

    def _refresh_color(self) -> None:
        color = QColor.fromRgbF(*self._color_rgba)
        self._color_button.setText(f"{color.name()} · Choose…")

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(QColor.fromRgbF(*self._color_rgba), self, "Class color")
        if color.isValid():
            self._color_rgba = tuple(float(value) for value in color.getRgbF())
            self._refresh_color()

    def _save_class(self) -> None:
        from ..editing.roi_commands import CreateObjectClass, UpdateObjectClass

        name = self._name_edit.text().strip()
        if not name:
            self._status.setText("Enter a class name.")
            return
        class_id = self.selected_class_id
        command = (
            CreateObjectClass(self.manager, name, self._color_rgba)
            if class_id is None else
            UpdateObjectClass(self.manager, class_id, name=name, color_rgba=self._color_rgba)
        )
        try:
            self._run_command(command)
        except (ValueError, RuntimeError) as error:
            self._status.setText(str(error))
            return
        if class_id is None:
            class_id = command.created_class_id
        self._refresh_classes(class_id)

    def _delete_class(self) -> None:
        from ..editing.roi_commands import DeleteObjectClass

        if self.selected_class_id is None:
            return
        try:
            self._run_command(DeleteObjectClass(self.manager, self.selected_class_id))
        except (ValueError, RuntimeError) as error:
            self._status.setText(str(error))
            return
        self._refresh_classes(None)


class RoiSpanDialog(QDialog):  # type: ignore[misc]
    """Validate expected bounds without removing or moving any observations."""

    def __init__(self, track: Any, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("ROI span editing requires 'acetree-py[gui]'")
        super().__init__(parent)
        self.track = track
        self.setWindowTitle("Expected Object Time Span")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Set the timepoints that should have a reviewed segmentation or an explicit absence. "
            "Unspecified bounds use the first or last annotated observation."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        self._start = QSpinBox()
        self._end = QSpinBox()
        for spin, label, value in (
            (self._start, "Start timepoint", track.expected_start_time),
            (self._end, "End timepoint", track.expected_end_time),
        ):
            spin.setRange(0, 2147483647)
            spin.setSpecialValueText("Unspecified")
            spin.setValue(value or 0)
            spin.setAccessibleName(label)
            form.addRow(label, spin)
        layout.addLayout(form)
        self._status = QLabel()
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> dict[str, int | None]:
        return {
            "expected_start_time": self._start.value() or None,
            "expected_end_time": self._end.value() or None,
        }

    def _accept_if_valid(self) -> None:
        try:
            replace(self.track, **self.get_values())
        except ValueError as error:
            self._status.setText(str(error))
            return
        self.accept()
