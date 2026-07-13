"""Tests for AceTreeApp save functionality (no Qt/napari required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.gui.app import AceTreeApp
from acetree_py.editing.commands import MoveNucleus
from acetree_py.io.config import AceTreeConfig, load_config
from acetree_py.io.config_writer import write_config_xml


def _nuc(index, x=300, y=250, z=15.0, identity="", status=1, pred=NILLI):
    return Nucleus(
        index=index, x=x, y=y, z=z, size=20,
        identity=identity, status=status,
        predecessor=pred, successor1=NILLI, successor2=NILLI,
    )


def _make_app(zip_path: Path | None = None) -> AceTreeApp:
    """Create a minimal AceTreeApp without launching the GUI."""
    mgr = NucleiManager()
    mgr.nuclei_record = [
        [_nuc(1, identity="P0")],
        [_nuc(1, identity="AB", pred=1), _nuc(2, identity="P1", pred=1)],
    ]
    mgr.set_all_successors()
    if zip_path is not None:
        config = AceTreeConfig()
        config.zip_file = zip_path
        mgr.config = config
    return AceTreeApp(mgr)


class TestDefaultSavePath:
    def test_no_config_returns_none(self):
        app = _make_app()
        assert app._default_save_path is None

    def test_with_config_returns_zip_path(self, tmp_path):
        expected = tmp_path / "data.zip"
        app = _make_app(zip_path=expected)
        assert app._default_save_path == expected

    def test_empty_zip_file_returns_none(self):
        app = _make_app()
        app.manager.config = AceTreeConfig()
        # config.zip_file defaults to Path() which is falsy
        assert app._default_save_path is None


class TestDoSave:
    def test_save_creates_file(self, tmp_path):
        output = tmp_path / "output.zip"
        app = _make_app()
        result = app._do_save(output)
        assert result == output
        assert output.exists()

    def test_save_roundtrip(self, tmp_path):
        output = tmp_path / "roundtrip.zip"
        app = _make_app()
        app._do_save(output)

        mgr2 = NucleiManager()
        mgr2.load(output)
        assert mgr2.num_timepoints == 2
        assert len(mgr2.nuclei_at(1)) == 1
        assert len(mgr2.nuclei_at(2)) == 2

    def test_successful_save_marks_current_history_state_saved(self, tmp_path):
        app = _make_app()
        app.edit_history.do(MoveNucleus(time=1, index=1, new_x=321))
        assert app.edit_history.modified

        assert app._do_save(tmp_path / "saved.zip") is not None
        assert not app.edit_history.modified

        # The savepoint is a real history state: moving away from it via Undo
        # makes the document dirty again.
        app.edit_history.undo()
        assert app.edit_history.modified


class TestSaveMethod:
    def test_save_with_known_path(self, tmp_path):
        target = tmp_path / "nuclei.zip"
        app = _make_app(zip_path=target)
        result = app.save()
        assert result == target
        assert target.exists()

    def test_save_without_path_returns_none_no_viewer(self):
        """Without a viewer, save_as cannot show a dialog and returns None."""
        app = _make_app()
        # No viewer → save_as() returns None
        result = app.save()
        assert result is None

    def test_save_as_retargets_subsequent_saves(self, tmp_path, monkeypatch):
        qt_widgets = pytest.importorskip("qtpy.QtWidgets")
        original = tmp_path / "original.zip"
        target = tmp_path / "new-location.zip"
        app = _make_app(zip_path=original)
        app.viewer = SimpleNamespace(
            window=SimpleNamespace(_qt_window=None)
        )
        monkeypatch.setattr(
            qt_widgets.QFileDialog,
            "getSaveFileName",
            lambda *args: (str(target), "ZIP archives (*.zip)"),
        )

        assert app.save_as() == target
        assert app.manager.config.zip_file == target
        assert app._default_save_path == target

        # Plain Save now follows the new destination, not the file originally
        # opened by the user.
        assert app.save() == target
        assert target.exists()

    def test_save_as_persists_retarget_in_source_config(self, tmp_path, monkeypatch):
        qt_widgets = pytest.importorskip("qtpy.QtWidgets")
        config_path = tmp_path / "embryo.xml"
        original = tmp_path / "original.zip"
        target = tmp_path / "new-location.zip"
        app = _make_app(zip_path=original)
        app.manager.config.config_file = config_path
        write_config_xml(app.manager.config, config_path)
        app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=None))
        monkeypatch.setattr(
            qt_widgets.QFileDialog,
            "getSaveFileName",
            lambda *args: (str(target), "ZIP archives (*.zip)"),
        )

        assert app.save_as() == target

        reopened = load_config(config_path)
        assert reopened.zip_file == target
        assert app.manager.config.zip_file == target
        assert app._default_save_path == target

    def test_config_write_failure_does_not_retarget_or_mark_saved(
        self,
        tmp_path,
        monkeypatch,
    ):
        qt_widgets = pytest.importorskip("qtpy.QtWidgets")
        import acetree_py.io.config_writer as config_writer_module

        config_path = tmp_path / "embryo.xml"
        original = tmp_path / "original.zip"
        target = tmp_path / "new-location.zip"
        app = _make_app(zip_path=original)
        app.manager.config.config_file = config_path
        write_config_xml(app.manager.config, config_path)
        app.edit_history.do(MoveNucleus(time=1, index=1, new_x=321))
        app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=None))
        monkeypatch.setattr(
            qt_widgets.QFileDialog,
            "getSaveFileName",
            lambda *args: (str(target), "ZIP archives (*.zip)"),
        )
        monkeypatch.setattr(qt_widgets.QMessageBox, "critical", lambda *args: None)
        monkeypatch.setattr(
            config_writer_module,
            "write_config_xml",
            lambda *args: (_ for _ in ()).throw(OSError("config write failed")),
        )

        assert app.save_as() is None

        assert target.exists()  # Valid copy, but not the current dataset target.
        assert app.manager.config.zip_file == original
        assert app._default_save_path == original
        assert app.edit_history.modified
        assert load_config(config_path).zip_file == original

    def test_archive_failure_never_attempts_config_retarget(
        self,
        tmp_path,
        monkeypatch,
    ):
        qt_widgets = pytest.importorskip("qtpy.QtWidgets")
        import acetree_py.io.config_writer as config_writer_module

        config_path = tmp_path / "embryo.xml"
        original = tmp_path / "original.zip"
        target = tmp_path / "new-location.zip"
        app = _make_app(zip_path=original)
        app.manager.config.config_file = config_path
        write_config_xml(app.manager.config, config_path)
        app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=None))
        monkeypatch.setattr(
            qt_widgets.QFileDialog,
            "getSaveFileName",
            lambda *args: (str(target), "ZIP archives (*.zip)"),
        )
        monkeypatch.setattr(qt_widgets.QMessageBox, "critical", lambda *args: None)
        monkeypatch.setattr(
            app.manager,
            "save",
            lambda *args: (_ for _ in ()).throw(OSError("archive write failed")),
        )
        config_write_called = False

        def record_config_write(*args):
            nonlocal config_write_called
            config_write_called = True

        monkeypatch.setattr(
            config_writer_module,
            "write_config_xml",
            record_config_write,
        )

        assert app.save_as() is None
        assert not config_write_called
        assert app.manager.config.zip_file == original
        assert load_config(config_path).zip_file == original
