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

    def test_plain_save_persists_expression_correction_in_config(self, tmp_path):
        config_path = tmp_path / "embryo.xml"
        target = tmp_path / "nuclei.zip"
        app = _make_app(zip_path=target)
        app.manager.config.config_file = config_path
        app.manager.config.expr_corr = "blot"
        write_config_xml(app.manager.config, config_path)

        app.manager.config.expr_corr = "global"
        app.manager._config_dirty = True
        assert app.save() == target

        assert load_config(config_path).expr_corr == "global"
        assert not app.manager._config_dirty

    def test_plain_save_config_staging_failure_preserves_previous_files(
        self,
        tmp_path,
        monkeypatch,
    ):
        import acetree_py.io.config_writer as config_writer_module

        config_path = tmp_path / "embryo.xml"
        target = tmp_path / "nuclei.zip"
        app = _make_app(zip_path=target)
        app.manager.config.config_file = config_path
        app.manager.config.expr_corr = "blot"
        write_config_xml(app.manager.config, config_path)
        assert app.save() == target
        previous_archive = target.read_bytes()
        previous_config = config_path.read_bytes()

        app.manager.nuclei_record[0][0].x = 777
        app.manager.config.expr_corr = "global"
        app.manager._config_dirty = True
        monkeypatch.setattr(
            config_writer_module,
            "write_config_xml",
            lambda *args: (_ for _ in ()).throw(OSError("config staging failed")),
        )

        assert app.save() is None
        assert target.read_bytes() == previous_archive
        assert config_path.read_bytes() == previous_config
        assert app.manager._config_dirty
        assert list(tmp_path.glob("*.save-config.tmp")) == []

    def test_plain_save_config_commit_failure_restores_previous_archive(
        self,
        tmp_path,
        monkeypatch,
    ):
        import acetree_py.gui.app as app_module

        config_path = tmp_path / "embryo.xml"
        target = tmp_path / "nuclei.zip"
        app = _make_app(zip_path=target)
        app.manager.config.config_file = config_path
        app.manager.config.expr_corr = "blot"
        write_config_xml(app.manager.config, config_path)
        assert app.save() == target
        previous_archive = target.read_bytes()
        previous_config = config_path.read_bytes()

        app.edit_history.do(MoveNucleus(time=1, index=1, new_x=777))
        app.manager.config.expr_corr = "global"
        app.manager._config_dirty = True
        real_replace = app_module.os.replace

        def fail_final_config_replace(source, destination):
            if Path(destination) == config_path:
                raise OSError("config commit failed")
            return real_replace(source, destination)

        monkeypatch.setattr(app_module.os, "replace", fail_final_config_replace)

        assert app.save() is None
        assert target.read_bytes() == previous_archive
        assert config_path.read_bytes() == previous_config
        assert app.manager._config_dirty
        assert app.edit_history.modified
        assert list(tmp_path.glob(".nuclei.zip.*.rollback")) == []
        assert list(tmp_path.glob("*.save-config.tmp")) == []

    def test_plain_save_archive_commit_failure_keeps_config_and_archive_together(
        self,
        tmp_path,
        monkeypatch,
    ):
        import acetree_py.core.nuclei_manager as nuclei_manager_module

        config_path = tmp_path / "embryo.xml"
        target = tmp_path / "nuclei.zip"
        app = _make_app(zip_path=target)
        app.manager.config.config_file = config_path
        app.manager.config.expr_corr = "blot"
        write_config_xml(app.manager.config, config_path)
        assert app.save() == target
        previous_archive = target.read_bytes()
        previous_config = config_path.read_bytes()

        app.manager.nuclei_record[0][0].x = 777
        app.manager.config.expr_corr = "global"
        app.manager._config_dirty = True
        real_replace = nuclei_manager_module.os.replace
        failed = False

        def fail_new_archive_install(source, destination):
            nonlocal failed
            if (
                not failed
                and Path(destination) == target
                and Path(source).suffix == ".tmp"
            ):
                failed = True
                raise OSError("archive commit failed")
            return real_replace(source, destination)

        monkeypatch.setattr(
            nuclei_manager_module.os,
            "replace",
            fail_new_archive_install,
        )

        assert app.save() is None
        assert failed
        assert target.read_bytes() == previous_archive
        assert config_path.read_bytes() == previous_config
        assert app.manager._config_dirty
        assert list(tmp_path.glob(".nuclei.zip.*.rollback")) == []
        assert list(tmp_path.glob("*.save-config.tmp")) == []

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
        app._nuclear_measurement_unsaved = True
        app.viewer = SimpleNamespace(
            window=SimpleNamespace(_qt_window=None)
        )
        monkeypatch.setattr(
            qt_widgets.QFileDialog,
            "getSaveFileName",
            lambda *args: (str(target), "ZIP archives (*.zip)"),
        )

        assert app.save_as() == target
        assert app._nuclear_measurement_unsaved is False
        assert app.manager.config.zip_file == target
        assert app._default_save_path == target

        # Plain Save now follows the new destination, not the file originally
        # opened by the user.
        assert app.save() == target
        assert target.exists()

    def test_save_as_copies_clean_roi_and_future_saves_leave_source_unchanged(
        self, tmp_path, monkeypatch
    ):
        from acetree_py.core.roi_manager import RoiManager
        from acetree_py.core.subcellular_roi import Polygon2D
        from acetree_py.io.roi_sidecar import read_roi_sidecar

        qt_widgets = pytest.importorskip("qtpy.QtWidgets")
        original = tmp_path / "original.zip"
        target = tmp_path / "new-location.zip"
        source_sidecar = original.with_suffix(".subcellular-rois.json")
        target_sidecar = target.with_suffix(".subcellular-rois.json")
        app = _make_app(zip_path=original)
        object_class = app.roi_manager.create_class("Golgi", (1, 0.5, 0, 1))
        track = app.roi_manager.create_object(object_class.class_id)
        geometry = Polygon2D(1, ((1, 1), (4, 1), (1, 4)))
        app.roi_manager.update_frame_geometry(track.object_id, 1, geometry)
        assert app.save() == original
        source_bytes = source_sidecar.read_bytes()
        app.roi_manager = RoiManager.from_load(read_roi_sidecar(source_sidecar))
        assert not app.roi_manager.is_dirty
        app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=None))
        monkeypatch.setattr(
            qt_widgets.QFileDialog, "getSaveFileName",
            lambda *args: (str(target), "ZIP archives (*.zip)"),
        )

        assert app.save_as() == target
        assert app.roi_manager.sidecar_path == target_sidecar
        reopened = RoiManager.from_config(app.manager.config)
        assert reopened.get_object(track.object_id).frames[1].geometry == geometry

        changed = Polygon2D(1, ((2, 2), (5, 2), (2, 5)))
        app.roi_manager.update_frame_geometry(track.object_id, 1, changed)
        assert app.save() == target
        assert source_sidecar.read_bytes() == source_bytes
        reopened = RoiManager.from_config(app.manager.config)
        assert reopened.get_object(track.object_id).frames[1].geometry == changed

    def test_save_as_persists_retarget_in_source_config(self, tmp_path, monkeypatch):
        qt_widgets = pytest.importorskip("qtpy.QtWidgets")
        config_path = tmp_path / "embryo.xml"
        original = tmp_path / "original.zip"
        target = tmp_path / "new-location.zip"
        app = _make_app(zip_path=original)
        app._nuclear_measurement_unsaved = True
        app.manager.config.config_file = config_path
        write_config_xml(app.manager.config, config_path)
        app.viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=None))
        monkeypatch.setattr(
            qt_widgets.QFileDialog,
            "getSaveFileName",
            lambda *args: (str(target), "ZIP archives (*.zip)"),
        )

        assert app.save_as() == target
        assert app._nuclear_measurement_unsaved is False

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
        app._nuclear_measurement_unsaved = True
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
        assert app._nuclear_measurement_unsaved is True

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
