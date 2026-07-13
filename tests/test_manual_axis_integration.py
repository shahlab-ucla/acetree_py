"""Integration tests for manual body orientation and division previews."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import acetree_py.core.nuclei_manager as nuclei_manager_module
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import Nucleus
from acetree_py.io.auxinfo import load_auxinfo
from acetree_py.io.config import AceTreeConfig
from acetree_py.naming.body_axes import BodyAxisFrame, BodyAxisLabels
from acetree_py.naming.rules import RuleManager
from acetree_py.editing.commands import SetBodyAxes


class _RecordingLineageCaller:
    is_lineage_mode = True
    is_founder_mode = False

    def __init__(self) -> None:
        self.calls: list[int] = []

    def _get_local_axes(self, time: int):
        self.calls.append(time)
        return (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )


def _manual_frame(reference_time: int = 1) -> BodyAxisFrame:
    return BodyAxisFrame.from_auxinfo_vectors(
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        provenance="manual_landmarks",
        reference_time=reference_time,
    )


def test_gui_time_is_converted_to_zero_based_axis_time_once():
    manager = NucleiManager()
    caller = _RecordingLineageCaller()
    manager.identity_assigner = SimpleNamespace(division_caller=caller)

    axes = manager.get_body_axes_at(7)

    assert axes is not None
    assert caller.calls == [6]


def test_explicit_manual_axes_take_precedence_over_inferred_axes():
    manager = NucleiManager()
    manager.set_manual_body_axes(_manual_frame(reference_time=4))
    caller = _RecordingLineageCaller()
    manager.identity_assigner = SimpleNamespace(division_caller=caller)

    ap, lr, dv = manager.get_body_axes_at(4)

    np.testing.assert_allclose(ap, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(lr, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(dv, [0.0, 1.0, 0.0])
    assert caller.calls == []


@pytest.mark.parametrize(
    ("parent_name", "expected", "expected_axis"),
    [
        ("AB", {"ABa", "ABp"}, "AP"),
        ("EMS", {"E", "MS"}, "AP"),
        ("P2", {"C", "P3"}, "DV"),
    ],
)
def test_manual_division_preview_uses_parent_specific_rules(
    parent_name: str,
    expected: set[str],
    expected_axis: str,
):
    manager = NucleiManager()
    manager.movie.xy_res = 1.0
    manager.movie.z_res = 2.0
    manager.set_manual_body_axes(_manual_frame())
    rule = RuleManager().get_rule(parent_name)
    scale = 30.0
    first = (100.0, 100.0, 10.0)
    second = (
        first[0] + rule.axis_vector[0] * scale,
        first[1] + rule.axis_vector[1] * scale,
        first[2] + rule.axis_vector[2] * scale / manager.z_pix_res,
    )
    parent = Nucleus(identity=parent_name, status=1)

    suggestion = manager.suggest_division_names(parent, first, second, time=2)

    assert {suggestion.first_name, suggestion.second_name} == expected
    assert suggestion.first_name == rule.daughter1
    assert suggestion.axis_label == expected_axis
    assert suggestion.source == "manual axes"
    assert suggestion.confidence > 0.9
    assert not suggestion.ambiguous


def test_division_preview_without_body_frame_does_not_guess_names():
    manager = NucleiManager()
    parent = Nucleus(identity="EMS", status=1)

    suggestion = manager.suggest_division_names(
        parent,
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        time=2,
    )

    assert suggestion.first_name == ""
    assert suggestion.second_name == ""
    assert suggestion.confidence == 0.0
    assert suggestion.ambiguous


def test_division_preview_preserves_empty_caller_result():
    caller = SimpleNamespace(
        is_v2=False,
        is_lineage_mode=True,
        is_founder_mode=False,
        classifications=[],
        assign_names=lambda *args, **kwargs: ("", ""),
    )
    manager = NucleiManager()
    manager.identity_assigner = SimpleNamespace(division_caller=caller)

    suggestion = manager.suggest_division_names(
        Nucleus(identity="E", status=1),
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        time=2,
    )

    assert suggestion.first_name == ""
    assert suggestion.second_name == ""
    assert suggestion.ambiguous


def test_landmark_z_is_scaled_before_solving_frame():
    labels = BodyAxisLabels(
        posterior=(0.0, 0.0, 0.0),
        anterior=(0.0, 0.0, 2.0),
        right=(0.0, 0.0, 0.0),
        left=(2.0, 0.0, 0.0),
    )
    frame = BodyAxisFrame.from_landmarks(labels, z_pix_res=5.0)

    np.testing.assert_allclose(frame.ap, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(frame.lr, [1.0, 0.0, 0.0])


def test_manual_axes_are_saved_as_reloadable_sidecar(tmp_path):
    manager = NucleiManager()
    manager.config = AceTreeConfig(config_file=tmp_path / "embryo.xml")
    manager.nuclei_record = [[Nucleus(index=1, status=1, identity="P0")]]
    manager.set_manual_body_axes(_manual_frame(reference_time=11))

    archive = tmp_path / "curated.zip"
    manager.save(archive)
    reloaded = load_auxinfo(archive.with_suffix(""))

    assert archive.exists()
    assert reloaded.is_manual
    assert reloaded.reference_time == 11
    assert reloaded.has_orientation
    np.testing.assert_allclose(reloaded.ap_orientation, [-1.0, 0.0, 0.0])


def test_undoing_manual_axes_removes_stale_manual_sidecar_on_next_save(tmp_path):
    manager = NucleiManager()
    manager.config = AceTreeConfig(config_file=tmp_path / "embryo.xml")
    manager.nuclei_record = [[Nucleus(index=1, status=1, identity="P0")]]
    archive = tmp_path / "curated.zip"
    command = SetBodyAxes(manager, _manual_frame(reference_time=3))

    command.execute(manager.nuclei_record)
    manager.save(archive)
    sidecar = tmp_path / "curatedAuxInfo_v2.csv"
    assert sidecar.exists()

    command.undo(manager.nuclei_record)
    manager.save(archive)
    assert not sidecar.exists()


def test_archive_commit_failure_restores_previous_sidecar_set(tmp_path, monkeypatch):
    manager = NucleiManager()
    manager.nuclei_record = [[Nucleus(index=1, status=1, identity="P0")]]
    manager.set_manual_body_axes(_manual_frame(reference_time=11))
    archive = tmp_path / "curated.zip"
    sidecar = tmp_path / "curatedAuxInfo_v2.csv"
    old_archive = b"previous archive"
    old_sidecar = b"previous sidecar"
    archive.write_bytes(old_archive)
    sidecar.write_bytes(old_sidecar)
    real_replace = os.replace

    def fail_archive_commit(source, destination):
        if Path(destination) == archive:
            raise OSError("simulated archive commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(nuclei_manager_module.os, "replace", fail_archive_commit)

    with pytest.raises(OSError, match="archive commit"):
        manager.save(archive)

    assert archive.read_bytes() == old_archive
    assert sidecar.read_bytes() == old_sidecar
    assert list(tmp_path.glob(".curated.zip.*.tmp")) == []
    assert list(tmp_path.glob(".curatedAuxInfo_v2.csv.*.tmp")) == []
    assert list(tmp_path.glob(".curatedAuxInfo_v2.csv.*.rollback")) == []


def test_sidecar_commit_failure_leaves_previous_save_set(tmp_path, monkeypatch):
    manager = NucleiManager()
    manager.nuclei_record = [[Nucleus(index=1, status=1, identity="P0")]]
    manager.set_manual_body_axes(_manual_frame(reference_time=11))
    archive = tmp_path / "curated.zip"
    sidecar = tmp_path / "curatedAuxInfo_v2.csv"
    old_archive = b"previous archive"
    old_sidecar = b"previous sidecar"
    archive.write_bytes(old_archive)
    sidecar.write_bytes(old_sidecar)
    real_replace = os.replace
    failed = False

    def fail_sidecar_commit(source, destination):
        nonlocal failed
        if (
            not failed
            and Path(destination) == sidecar
            and Path(source).suffix == ".tmp"
        ):
            failed = True
            raise OSError("simulated sidecar commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(nuclei_manager_module.os, "replace", fail_sidecar_commit)

    with pytest.raises(OSError, match="sidecar commit"):
        manager.save(archive)

    assert archive.read_bytes() == old_archive
    assert sidecar.read_bytes() == old_sidecar
    assert list(tmp_path.glob(".curated.zip.*.tmp")) == []
    assert list(tmp_path.glob(".curatedAuxInfo_v2.csv.*.tmp")) == []
    assert list(tmp_path.glob(".curatedAuxInfo_v2.csv.*.rollback")) == []


def test_archive_failure_restores_manual_sidecar_scheduled_for_removal(
    tmp_path,
    monkeypatch,
):
    manager = NucleiManager()
    manager.nuclei_record = [[Nucleus(index=1, status=1, identity="P0")]]
    archive = tmp_path / "curated.zip"
    sidecar = tmp_path / "curatedAuxInfo_v2.csv"
    old_archive = b"previous archive"
    old_sidecar = b"orientation_source\nmanual\n"
    archive.write_bytes(old_archive)
    sidecar.write_bytes(old_sidecar)
    real_replace = os.replace

    def fail_archive_commit(source, destination):
        if Path(destination) == archive:
            raise OSError("simulated archive commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(nuclei_manager_module.os, "replace", fail_archive_commit)

    with pytest.raises(OSError, match="archive commit"):
        manager.save(archive)

    assert archive.read_bytes() == old_archive
    assert sidecar.read_bytes() == old_sidecar
    assert list(tmp_path.glob(".curatedAuxInfo_v2.csv.*.rollback")) == []
