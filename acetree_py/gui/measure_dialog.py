"""Dialog for configuring the Measure run.

Lets the user pick which image channel becomes the "AT expression
channel" (i.e. the one whose measurements are written back onto
``Nucleus.rwraw``/``rwcorr1`` and drive the lineage-tree coloring)
and where per-channel CSVs should be written.

Pattern follows :class:`RenameCellDialog` / :class:`KillCellDialog`
in ``gui/edit_panel.py`` for layout and
:class:`LineagePanelConfigDialog` in ``gui/lineage_widget.py`` for the
channel combo.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import AceTreeApp

try:
    from qtpy.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False
    QDialog = object  # type: ignore[misc,assignment]


class MeasureDialog(QDialog):  # type: ignore[misc]
    """Pick the AT expression channel and the output directory.

    The caller constructs this with the live :class:`AceTreeApp`;
    the combo is populated from ``app.image_provider.num_channels``
    and the output directory defaults to a ``measurements`` folder
    alongside the currently-loaded nuclei zip, if any.
    """

    def __init__(self, app: AceTreeApp, parent=None) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("Qt is required: pip install 'acetree-py[gui]'")

        super().__init__(parent)
        self.setWindowTitle("Measure")
        self.setMinimumWidth(420)

        self._app = app

        n_ch = 1
        provider = getattr(app, "image_provider", None)
        if provider is not None:
            try:
                n_ch = int(provider.num_channels)
            except Exception:
                n_ch = 1
        n_ch = max(1, n_ch)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Measure pixel intensity for every nucleus in every channel.\n"
            "One CSV per channel will be written to the output folder.\n"
            "The selected AT channel's values also update the lineage "
            "tree colors."
        ))

        form = QFormLayout()

        # Channel selector: 1-based labels, 0-based values
        self._channel_combo = QComboBox()
        for c in range(n_ch):
            self._channel_combo.addItem(f"Channel {c + 1}", userData=c)
        # Default to the AceTreeApp's currently-shown expression channel
        # if it exposes one; otherwise channel 1.
        default_idx = 0
        current = getattr(app, "current_expression_channel", None)
        if isinstance(current, int) and 0 <= current < n_ch:
            default_idx = current
        self._channel_combo.setCurrentIndex(default_idx)
        form.addRow("AT channel:", self._channel_combo)

        # Output directory row
        default_dir = self._default_output_dir(app)
        self._dir_edit = QLineEdit(str(default_dir) if default_dir else "")
        self._dir_edit.setPlaceholderText("Output folder for CSVs")

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(browse_btn, 0)
        form.addRow("Output folder:", dir_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _default_output_dir(app: AceTreeApp) -> Path | None:
        """Pick a sensible default output directory.

        Prefers ``<zip_dir>/measurements`` when a nuclei zip is loaded;
        otherwise returns ``None`` and lets the user browse.
        """
        mgr = getattr(app, "manager", None)
        cfg = getattr(mgr, "config", None) if mgr is not None else None
        zip_file = getattr(cfg, "zip_file", None)
        if zip_file is not None:
            p = Path(zip_file)
            if str(p) not in ("", ".") and p != Path():
                return p.parent / "measurements"
        return None

    def _on_browse(self) -> None:
        """Open a directory picker and stash the selection into the line edit."""
        start = self._dir_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self, "Select output directory", start
        )
        if chosen:
            self._dir_edit.setText(chosen)

    def _on_accept(self) -> None:
        """Validate the output directory before accepting."""
        text = self._dir_edit.text().strip()
        if not text:
            # Re-open the browser if the user hit OK with an empty field
            self._on_browse()
            if not self._dir_edit.text().strip():
                return
        self.accept()

    def get_values(self) -> dict:
        """Return the dialog's selections.

        Keys:
            at_channel: 0-based channel index for the AT expression channel.
            output_dir: :class:`pathlib.Path` of the chosen output folder.
        """
        at_channel = int(self._channel_combo.currentData())
        output_dir = Path(self._dir_edit.text().strip())
        return {"at_channel": at_channel, "output_dir": output_dir}
