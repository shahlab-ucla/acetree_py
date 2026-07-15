"""Regression tests for safe dataset-creation inputs and channel layouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy")
tifffile = pytest.importorskip("tifffile")

from qtpy.QtWidgets import QMessageBox

from acetree_py.gui import dataset_dialog as dataset_dialog_module
from acetree_py.gui.dataset_dialog import DatasetCreationDialog
from acetree_py.io.image_provider import create_image_provider_from_config


def _write_stack(directory: Path, name: str = "emb_t001.tif", pages: int = 4) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    tifffile.imwrite(
        path,
        np.zeros((pages, 8, 8), dtype=np.uint16),
        photometric="minisblack",
    )
    return path


def _set_primary_images(dialog: DatasetCreationDialog, directory: Path) -> None:
    dialog._dir_edit.setText(str(directory))
    dialog._run_auto_detect(directory)


def test_primary_image_directory_must_contain_readable_tiffs(
    qtbot,
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)

    assert not dialog._btn_next.isEnabled()
    _set_primary_images(dialog, empty)
    assert "contains no TIFF" in dialog._image_source_validation_error()
    assert not dialog._btn_next.isEnabled()
    with pytest.raises(ValueError, match="contains no TIFF"):
        dialog.get_config()

    _write_stack(empty)
    _set_primary_images(dialog, empty)
    assert dialog._image_source_validation_error() == ""
    assert dialog._btn_next.isEnabled()


def test_separate_channel_layout_never_silently_degrades_to_one_channel(
    qtbot,
    tmp_path: Path,
) -> None:
    primary = tmp_path / "channel1"
    second = tmp_path / "channel2"
    _write_stack(primary)
    second.mkdir()

    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    _set_primary_images(dialog, primary)
    dialog._radio_separate.setChecked(True)
    dialog._ch2_dir_edit.setText(str(second))
    dialog._stack.setCurrentWidget(dialog._page2)

    assert "contains no TIFF" in dialog._image_layout_validation_error()
    assert not dialog._btn_next.isEnabled()
    with pytest.raises(ValueError, match="contains no TIFF"):
        dialog.get_config()

    _write_stack(second)
    dialog._refresh_tracking_validation()
    dialog._radio_tracking_auto.setChecked(True)
    dialog._tracking_channel_spin.setValue(2)

    config = dialog.get_config()
    request = dialog.get_tracking_request()
    provider = create_image_provider_from_config(config)

    assert config.num_channels == 2
    assert set(config.image_channels) == {1, 2}
    assert provider is not None
    assert provider.num_channels == 2
    assert request is not None
    assert request.detector.settings["TARGET_CHANNEL"] == 2


def test_changing_directory_recomputes_multichannel_z_count(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detected = iter(
        (
            {"num_files": 1, "num_timepoints": 1, "num_planes": 10},
            {"num_files": 1, "num_timepoints": 1, "num_planes": 12},
        )
    )
    monkeypatch.setattr(
        dataset_dialog_module,
        "_auto_detect_format",
        lambda _directory: next(detected),
    )
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)

    dialog._run_auto_detect(Path("first"))
    dialog._radio_multistack.setChecked(True)
    dialog._n_channels_spin.setValue(2)
    assert dialog._planes_spin.value() == 5

    dialog._run_auto_detect(Path("second"))
    assert dialog._planes_spin.value() == 6


def test_create_requires_output_and_confirms_before_replacement(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    _write_stack(image_dir)
    output_dir.mkdir()

    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    _set_primary_images(dialog, image_dir)
    dialog._stack.setCurrentWidget(dialog._page5)
    dialog._update_nav_buttons()

    assert not dialog._btn_next.isEnabled()
    assert "Choose an output directory" in dialog._output_validation_label.text()

    dialog._output_edit.setText(str(output_dir))
    assert dialog._btn_next.isEnabled()
    existing = output_dir / "dataset.zip"
    existing.write_bytes(b"do not replace without confirmation")

    replies = iter((QMessageBox.No, QMessageBox.Yes))
    prompts: list[str] = []

    def question(_parent, _title, message, *_args):
        prompts.append(message)
        return next(replies)

    monkeypatch.setattr(QMessageBox, "question", question)
    accepted: list[bool] = []
    dialog.accepted.connect(lambda: accepted.append(True))

    dialog._go_next()
    assert not accepted
    assert existing.read_bytes() == b"do not replace without confirmation"
    assert "dataset.zip" in prompts[-1]

    dialog._go_next()
    assert accepted == [True]


def test_create_blocks_non_directory_output(qtbot, tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    dialog = DatasetCreationDialog()
    qtbot.addWidget(dialog)
    dialog._stack.setCurrentWidget(dialog._page5)
    dialog._output_edit.setText(str(output_file))

    assert "is a file" in dialog._output_validation_error()
    assert not dialog._btn_next.isEnabled()
    with pytest.raises(ValueError, match="is a file"):
        dialog.get_output_directory()
