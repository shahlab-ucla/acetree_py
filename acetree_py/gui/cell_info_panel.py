"""Cell info panel — displays details about the currently selected cell.

Shows cell name, position, expression values, lineage info, and lifecycle.
Updates whenever the selection or timepoint changes.

Ported from: org.rhwlab.acetree.AceTree.makeDisplayText()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import (
        QLabel,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


class CellInfoPanel(QWidget):  # type: ignore[misc]
    """Widget displaying information about the currently selected cell.

    Shows:
    - Cell name and population count
    - Position (x, y, z)
    - Size (raw and projected diameter)
    - Expression values (weight, rweight)
    - Lifecycle (start/end time, fate)
    - Lineage context (parent, children)
    """

    def __init__(self, app: AceTreeApp, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the panel layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._title = QLabel("Cell Info")
        self._title.setFont(QFont("Sans Serif", 12, QFont.Bold))
        layout.addWidget(self._title)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Monospace", 10))
        self._text.setMinimumWidth(220)
        layout.addWidget(self._text)

    def refresh(self) -> None:
        """Update the displayed text from the current app state."""
        info = self.app.get_cell_info_text()
        self._text.setPlainText(info)

        if self.app.current_cell_name:
            self._title.setText(f"Cell: {self.app.current_cell_name}")
        else:
            self._title.setText("Cell Info")
