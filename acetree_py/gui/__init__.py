"""GUI layer — napari-based image viewer with nucleus overlays and lineage display.

All GUI code is isolated here. The core/, naming/, editing/, and io/ packages
have ZERO GUI imports, allowing headless/batch usage.

Requires optional dependencies: napari, qtpy, pyqtgraph
Install with: pip install acetree-py[gui]
"""
