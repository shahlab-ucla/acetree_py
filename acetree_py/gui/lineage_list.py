"""Lineage list widget — hierarchical JTree-style cell lineage browser.

Provides a QTreeWidget showing the cell lineage as an expandable tree,
similar to the main AceTree Java window's JTree panel. Each node shows
the cell name and lifetime. Supports:

- Click to select cell and jump to its start time
- Right-click to jump to the cell's end time (last timepoint)
- Double-click to expand/collapse
- Current selection highlighted in sync with the image viewer

Ported from: org.rhwlab.acetree.AceTree (main JTree panel)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import AceTreeApp

logger = logging.getLogger(__name__)

try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QTreeWidget,
        QTreeWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QWidget = object  # type: ignore[misc,assignment]


class LineageListWidget(QWidget):  # type: ignore[misc]
    """Hierarchical tree list showing the cell lineage.

    Each item in the tree shows:
    - Cell name
    - Lifetime range (start_time - end_time)
    - Fate indicator (DIVIDED / ALIVE / DIED)

    Interactions:
    - Left-click: select cell, jump to start time
    - Right-click: select cell, jump to end time
    - Search box to find cells by name
    """

    def __init__(self, app: AceTreeApp, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.app = app

        # Map cell name -> QTreeWidgetItem for quick lookup
        self._items: dict[str, QTreeWidgetItem] = {}
        self._current_selection: str = ""

        self._build_ui()
        self._populate_tree()

    def _build_ui(self) -> None:
        """Build the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Search bar
        search_layout = QHBoxLayout()
        search_layout.setSpacing(4)

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search cell name...")
        self._search_box.textChanged.connect(self._on_search)

        self._btn_clear = QPushButton("X")
        self._btn_clear.setFixedWidth(24)
        self._btn_clear.setToolTip("Clear search")
        self._btn_clear.clicked.connect(self._clear_search)

        search_layout.addWidget(self._search_box)
        search_layout.addWidget(self._btn_clear)
        layout.addLayout(search_layout)

        # Cell count label
        self._count_label = QLabel("")
        layout.addWidget(self._count_label)

        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Cell", "Time", "Fate"])
        self._tree.setColumnWidth(0, 120)
        self._tree.setColumnWidth(1, 80)
        self._tree.setColumnWidth(2, 60)
        self._tree.setAlternatingRowColors(False)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)

        # Style for dark-theme compatibility (napari uses a dark theme)
        self._tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: none;
            }
            QTreeWidget::item {
                padding: 2px;
            }
            QTreeWidget::item:alternate {
                background-color: #252525;
            }
            QTreeWidget::item:selected {
                background-color: #3a5fcd;
                color: #ffffff;
            }
            QTreeWidget::item:hover {
                background-color: #2a2a2a;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 4px;
                border: 1px solid #3a3a3a;
            }
        """)

        # Connect signals
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_right_click)

        layout.addWidget(self._tree)

    def _populate_tree(self) -> None:
        """Build the tree from the lineage data.

        Shows ALL root-level subtrees (cells with no parent) as
        top-level items. The largest named tree (e.g. P0) is
        expanded by default; smaller/unnamed trees are collapsed.
        """
        self._tree.clear()
        self._items.clear()

        tree = self.app.manager.lineage_tree
        if tree is None:
            self._count_label.setText("No lineage data")
            return

        # Find all root cells (cells with no parent), sorted by subtree size
        roots = []
        for cell in tree.cells_by_name.values():
            if cell.parent is None:
                subtree_size = sum(1 for _ in cell.iter_subtree_preorder())
                roots.append((subtree_size, cell))

        if not roots:
            self._count_label.setText("No lineage data")
            return

        # Sort: largest subtrees first, then by start time
        roots.sort(key=lambda x: (-x[0], x[1].start_time))

        for i, (size, root_cell) in enumerate(roots):
            root_item = self._build_item(root_cell)
            self._tree.addTopLevelItem(root_item)

            # Expand the first (largest) tree to depth 3
            if i == 0:
                self._expand_to_depth(root_item, 3)

        count = len(self._items)
        n_roots = len(roots)
        self._count_label.setText(f"{count} cells in {n_roots} trees")

    def _build_item(self, cell) -> QTreeWidgetItem:
        """Recursively build a QTreeWidgetItem from a Cell."""
        time_str = f"t{cell.start_time}-{cell.end_time}"
        fate_str = cell.end_fate.name if hasattr(cell.end_fate, 'name') else str(cell.end_fate)

        item = QTreeWidgetItem([cell.name, time_str, fate_str])
        item.setData(0, Qt.UserRole, cell.name)  # Store name for lookup

        # Tooltip with more detail
        lifetime = cell.end_time - cell.start_time + 1
        n_children = len(cell.children) if cell.children else 0
        item.setToolTip(0, f"{cell.name}\nLifetime: {lifetime} timepoints\nChildren: {n_children}")

        self._items[cell.name] = item

        # Add children
        for child in (cell.children or []):
            child_item = self._build_item(child)
            item.addChild(child_item)

        return item

    def _expand_to_depth(self, item: QTreeWidgetItem, depth: int) -> None:
        """Expand tree items to a given depth."""
        if depth <= 0:
            return
        item.setExpanded(True)
        for i in range(item.childCount()):
            self._expand_to_depth(item.child(i), depth - 1)

    def refresh_selection(self) -> None:
        """Update the tree selection to match the app's current cell."""
        cell_name = self.app.current_cell_name
        if cell_name == self._current_selection:
            return

        self._current_selection = cell_name

        if cell_name and cell_name in self._items:
            item = self._items[cell_name]
            # Block signals to avoid re-triggering selection
            self._tree.blockSignals(True)
            self._tree.setCurrentItem(item)
            self._tree.scrollToItem(item)
            self._tree.blockSignals(False)

    def rebuild(self) -> None:
        """Rebuild the entire tree (e.g. after edits)."""
        self._populate_tree()

    # ── Signal handlers ────────────────────────────────────────────

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle left-click on a tree item: select cell at start time."""
        cell_name = item.data(0, Qt.UserRole)
        if cell_name:
            cell = self.app.manager.get_cell(cell_name)
            if cell is not None:
                self.app.select_cell(cell_name, cell.start_time)
            else:
                self.app.select_cell(cell_name)

    def _on_right_click(self, position) -> None:
        """Handle right-click: select cell and jump to end time."""
        item = self._tree.itemAt(position)
        if item is None:
            return

        cell_name = item.data(0, Qt.UserRole)
        if cell_name:
            cell = self.app.manager.get_cell(cell_name)
            if cell is not None:
                self.app.select_cell(cell_name, cell.end_time)
            else:
                self.app.select_cell(cell_name)

    def _on_search(self, text: str) -> None:
        """Filter tree to show only cells matching the search text."""
        text = text.strip().lower()

        if not text:
            # Show all items
            for item in self._items.values():
                item.setHidden(False)
            return

        # Hide non-matching items, show matching ones and their ancestors
        visible = set()
        for name, item in self._items.items():
            if text in name.lower():
                # Show this item and all ancestors
                visible.add(name)
                self._show_ancestors(item, visible)

        for name, item in self._items.items():
            item.setHidden(name not in visible)

        # Expand to show matching items
        for name in visible:
            if text in name.lower() and name in self._items:
                parent = self._items[name].parent()
                while parent:
                    parent.setExpanded(True)
                    parent = parent.parent()

    def _show_ancestors(self, item: QTreeWidgetItem, visible: set) -> None:
        """Mark all ancestors of an item as visible."""
        parent = item.parent()
        while parent:
            name = parent.data(0, Qt.UserRole)
            if name:
                visible.add(name)
            parent = parent.parent()

    def _clear_search(self) -> None:
        """Clear the search box and show all items."""
        self._search_box.clear()
