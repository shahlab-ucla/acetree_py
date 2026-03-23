"""Tests for AceTreeApp save functionality (no Qt/napari required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.config import AceTreeConfig


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
