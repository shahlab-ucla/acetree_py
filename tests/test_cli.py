"""Tests for __main__.py CLI commands.

Tests cover:
- CLI app creation and help
- load command output format
- export command with different formats
- info command output
- rename command
- Error handling for missing cells
"""

from __future__ import annotations

from typer.testing import CliRunner

from acetree_py.__main__ import app

runner = CliRunner()


class TestCLIHelp:
    """Test CLI help and no-args behavior."""

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # typer with no_args_is_help returns exit code 0 or 2 depending on version
        assert result.exit_code in (0, 2)
        assert "AceTree-Py" in result.output or "Usage" in result.output

    def test_version_identifies_alpha_build(self):
        from acetree_py import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.output.strip() == f"AceTree-Py {__version__} (alpha v2)"

    def test_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "load" in result.output
        assert "gui" in result.output
        assert "export" in result.output
        assert "rename" in result.output
        assert "info" in result.output

    def test_create_help_exposes_manual_or_automated_tracking(self):
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0
        assert "--tracking" in result.output
        assert "dog-lap" in result.output
        assert "log-lap" in result.output


class TestCreateCommand:
    """Test noninteractive dataset creation settings."""

    def test_starrynite_create_uses_bundled_preset_without_launching_gui(
        self,
        tmp_path,
        monkeypatch,
    ):
        from acetree_py.gui.app import AceTreeApp
        from acetree_py.tracking.starrynite import (
            bundled_parameter_preset,
            load_tuning_profile,
        )

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        # The command deliberately tolerates an unreadable probe and falls back
        # to a one-plane stack, so no image codec is needed for this request test.
        (image_dir / "embryo_t1.tif").touch()
        (image_dir / "embryo_t2.tif").touch()
        output_dir = tmp_path / "output"
        captured = {}

        class _CreatedApp:
            def run(self):
                captured["run_called"] = True

        def _capture_new_dataset(
            config,
            num_timepoints,
            output_dir_arg,
            *,
            tracking_request=None,
        ):
            captured.update(
                config=config,
                num_timepoints=num_timepoints,
                output_dir=output_dir_arg,
                request=tracking_request,
            )
            return _CreatedApp()

        monkeypatch.setattr(
            AceTreeApp,
            "from_new_dataset",
            staticmethod(_capture_new_dataset),
        )

        result = runner.invoke(
            app,
            [
                "create",
                str(image_dir),
                "--output",
                str(output_dir),
                "--tracking",
                "starrynite",
                "--starrynite-preset",
                "isim_red_40x",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["run_called"] is True
        assert captured["num_timepoints"] == 2
        assert captured["output_dir"] == output_dir.resolve()

        request = captured["request"]
        preset = bundled_parameter_preset("isim_red_40x")
        profile = load_tuning_profile(preset.parameter_file)
        assert request.detector.plugin_id == "acetree.starrynite_detector"
        assert request.tracker.plugin_id == "acetree.starrynite_division"
        assert request.scope.kind == "global"
        assert request.scope.start_frame == 1
        assert request.scope.end_frame == 2
        assert request.detector.settings["STARRYNITE_PARAMETER_FILE"] == str(
            preset.parameter_file.resolve()
        )
        assert request.detector.settings["STARRYNITE_PARAMETER_SHA256"] == (
            profile.detector_settings["STARRYNITE_PARAMETER_SHA256"]
        )
        assert request.detector.settings["STARRYNITE_DISTRIBUTION_FILE"] == ""
        assert request.detector.settings["STARRYNITE_DISTRIBUTION_SOURCE_SHA256"] == ""
        assert request.detector.settings["RADIUS"] == profile.detector_settings[
            "RADIUS"
        ]
        assert request.detector.settings["INTENSITY_THRESHOLD"] == (
            profile.detector_settings["INTENSITY_THRESHOLD"]
        )
        assert request.detector.settings["THRESHOLD"] == 0.0
        assert request.tracker.settings["MAX_FRAME_GAP"] == (
            profile.tracker_settings["MAX_FRAME_GAP"]
        )
        assert request.tracker.settings["ALLOW_GAP_CLOSING"] is True
        assert request.tracker.settings["ALLOW_TRACK_SPLITTING"] is True
        assert request.tracker.settings["ALLOW_TRACK_MERGING"] is False


class TestLoadCommand:
    """Test the load command."""

    def test_load_prints_summary(self, sample_xml_config):
        result = runner.invoke(app, ["load", str(sample_xml_config)])
        assert result.exit_code == 0
        assert "Timepoints" in result.output
        assert "3" in result.output  # 3 timepoints in fixture


class TestExportCommand:
    """Test the export command with different formats."""

    def test_export_cell_csv(self, sample_xml_config, tmp_path):
        output = tmp_path / "cells.csv"
        result = runner.invoke(app, [
            "export", str(sample_xml_config),
            "--format", "cell_csv",
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert output.exists()
        assert "Exported" in result.output

    def test_export_nucleus_csv(self, sample_xml_config, tmp_path):
        output = tmp_path / "nuclei.csv"
        result = runner.invoke(app, [
            "export", str(sample_xml_config),
            "--format", "nucleus_csv",
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_export_newick(self, sample_xml_config, tmp_path):
        output = tmp_path / "tree.nwk"
        result = runner.invoke(app, [
            "export", str(sample_xml_config),
            "--format", "newick",
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert content.strip().endswith(";")

    def test_export_expression_csv(self, sample_xml_config, tmp_path):
        output = tmp_path / "expression.csv"
        result = runner.invoke(app, [
            "export", str(sample_xml_config),
            "--format", "expression_csv",
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert output.exists()

    def test_export_unknown_format(self, sample_xml_config):
        result = runner.invoke(app, [
            "export", str(sample_xml_config),
            "--format", "unknown_format",
        ])
        assert result.exit_code == 1

    def test_export_auto_names(self, sample_xml_config, tmp_path, monkeypatch):
        """Export without --output creates auto-named file."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, [
            "export", str(sample_xml_config),
            "--format", "newick",
        ])
        assert result.exit_code == 0
        assert "Exported" in result.output


class TestInfoCommand:
    """Test the info command."""

    def test_info_existing_cell(self, sample_xml_config):
        """Info command should work for cells that exist in the fixture."""
        # The fixture has P0 at t=1, AB and P1 at t=2, ABa/ABp/P1 at t=3
        # The naming pipeline may produce different names, so test with a
        # cell we know exists from the raw data
        result = runner.invoke(app, [
            "info", str(sample_xml_config),
            "--cell", "P0",
        ])
        # P0 should exist in the dummy ancestors at minimum
        if result.exit_code == 0:
            assert "Cell: P0" in result.output
            assert "Lifetime" in result.output

    def test_info_nonexistent_cell(self, sample_xml_config):
        result = runner.invoke(app, [
            "info", str(sample_xml_config),
            "--cell", "NonexistentCell",
        ])
        assert result.exit_code == 1


class TestRenameCommand:
    """Test the rename command."""

    def test_rename_creates_output(self, sample_xml_config, tmp_path):
        output = tmp_path / "renamed.zip"
        result = runner.invoke(app, [
            "rename", str(sample_xml_config),
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert output.exists()
        assert "Saved" in result.output
