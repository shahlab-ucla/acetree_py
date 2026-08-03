"""Arbitrary-channel Measure retention and concurrency boundaries."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import acetree_py.analysis.measure_runner as measure_runner
from acetree_py.analysis.expression_measurements import legacy_expression_coverage
from acetree_py.analysis.measure_runner import SCALE, run_measure
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import Nucleus
from acetree_py.io.image_provider import NumpyProvider


def _manager() -> NucleiManager:
    manager = NucleiManager()
    manager.movie.xy_res = 1.0
    manager.movie.z_res = 1.0
    nucleus = Nucleus(
        index=1,
        x=8,
        y=8,
        z=2.0,
        size=6,
        status=1,
        identity="A",
    )
    manager.nuclei_record = [[nucleus]]
    manager.set_all_successors()
    manager.process(do_identity=False)
    return manager


def _provider(first: int = 100, second: int = 250) -> NumpyProvider:
    channel_1 = np.full((5, 16, 16), first, dtype=np.uint16)
    channel_2 = np.full((5, 16, 16), second, dtype=np.uint16)
    # T, C, Z, Y, X
    return NumpyProvider(np.stack([channel_1, channel_2], axis=0)[np.newaxis, ...])


def _manager_with_neighbor() -> NucleiManager:
    manager = _manager()
    manager.nuclei_record[0].append(
        Nucleus(
            index=2,
            x=12,
            y=8,
            z=2.0,
            size=6,
            status=1,
            identity="B",
        )
    )
    manager.set_all_successors()
    manager.process(do_identity=False)
    return manager


def test_run_measure_retains_every_channel_for_plotting(tmp_path: Path):
    manager = _manager()
    written = run_measure(
        manager,
        _provider(),
        tmp_path,
        at_channel=0,
        correction_method="none",
    )

    store = manager.expression_measurements
    assert store is not None
    assert store.is_current(manager)
    assert store.csv_paths == tuple(written)
    assert len(store.channels) == 2

    cell = manager.get_cell("A")
    assert cell is not None
    nucleus = cell.get_nucleus_at(1)
    assert nucleus is not None
    first = store.sample(manager, 0, 1, nucleus)
    second = store.sample(manager, 1, 1, nucleus)
    assert first is not None and second is not None
    assert first.value == pytest.approx(100 * SCALE)
    assert second.value == pytest.approx(250 * SCALE)

    channels = store.expression_channels(manager)
    assert [channel.key for channel in channels] == [
        "measured_channel_1",
        "measured_channel_2",
    ]
    assert channels[1].reader(cell, 1, nucleus) == pytest.approx(250 * SCALE)


def test_measurements_fail_closed_after_revision_or_geometry_change(tmp_path: Path):
    manager = _manager()
    run_measure(manager, _provider(), tmp_path, 0, correction_method="none")
    store = manager.expression_measurements
    assert store is not None
    nucleus = manager.nuclei_record[0][0]
    assert store.sample(manager, 0, 1, nucleus) is not None

    # Geometry signature catches mutation even if a caller bypasses history.
    nucleus.x += 1
    assert store.sample(manager, 0, 1, nucleus) is None
    nucleus.x -= 1
    assert store.sample(manager, 0, 1, nucleus) is not None

    manager.mark_data_edited()
    assert not store.is_current(manager)
    assert store.sample(manager, 0, 1, nucleus) is None


def test_cancelled_measure_preserves_previous_snapshot(tmp_path: Path):
    manager = _manager()
    run_measure(manager, _provider(), tmp_path / "first", 0, correction_method="none")
    previous = manager.expression_measurements

    with pytest.raises(RuntimeError, match="cancelled"):
        run_measure(
            manager,
            _provider(),
            tmp_path / "cancelled",
            0,
            progress_cb=lambda *_args: False,
            correction_method="none",
        )

    assert manager.expression_measurements is previous


def test_measure_aborts_if_dataset_changes_during_progress(tmp_path: Path):
    manager = _manager()
    run_measure(manager, _provider(), tmp_path / "first", 0, correction_method="none")
    previous = manager.expression_measurements
    prior_rweight = manager.nuclei_record[0][0].rweight
    mutated = False

    def mutate_once(*_args):
        nonlocal mutated
        if not mutated:
            manager.nuclei_record[0][0].x += 1
            mutated = True
        return True

    with pytest.raises(RuntimeError, match="Dataset changed while Measure"):
        run_measure(
            manager,
            _provider(400, 800),
            tmp_path / "changed",
            0,
            progress_cb=mutate_once,
            correction_method="none",
        )

    assert manager.expression_measurements is previous
    assert manager.nuclei_record[0][0].rweight == prior_rweight
    assert not list((tmp_path / "changed").glob("*.csv"))


def test_csv_failure_rolls_back_legacy_fields_and_store(tmp_path: Path, monkeypatch):
    manager = _manager()
    output_dir = tmp_path / "measure"
    prior_paths = run_measure(
        manager, _provider(), output_dir, 0, correction_method="none"
    )
    prior_contents = {path: path.read_bytes() for path in prior_paths}
    nucleus = manager.nuclei_record[0][0]
    previous_store = manager.expression_measurements
    previous_fields = (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr3,
        nucleus.rsum,
        nucleus.rcount,
        manager._expr_corr,
    )

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(measure_runner, "write_measure_csv", fail_write)
    with pytest.raises(OSError, match="disk full"):
        run_measure(
            manager,
            _provider(999, 777),
            output_dir,
            0,
            correction_method="global",
        )

    assert manager.expression_measurements is previous_store
    assert (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr3,
        nucleus.rsum,
        nucleus.rcount,
        manager._expr_corr,
    ) == previous_fields
    assert {path: path.read_bytes() for path in prior_paths} == prior_contents
    assert not list(output_dir.glob("*.tmp"))


def test_late_publication_failure_restores_csvs_fields_and_store(
    tmp_path: Path,
    monkeypatch,
):
    manager = _manager()
    output_dir = tmp_path / "measure"
    prior_paths = run_measure(
        manager, _provider(), output_dir, 0, correction_method="none"
    )
    prior_contents = {path: path.read_bytes() for path in prior_paths}
    nucleus = manager.nuclei_record[0][0]
    previous_store = manager.expression_measurements
    previous_freshness = manager.expression_measurement_freshness_known
    previous_fields = (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr3,
        nucleus.rsum,
        nucleus.rcount,
        manager._expr_corr,
    )

    def fail_after_legacy_apply(*_args, **_kwargs):
        raise RuntimeError("late publication failure")

    monkeypatch.setattr(
        measure_runner,
        "_set_measured_at_weights",
        fail_after_legacy_apply,
    )
    with pytest.raises(RuntimeError, match="late publication failure"):
        run_measure(
            manager,
            _provider(999, 777),
            output_dir,
            0,
            correction_method="global",
        )

    assert manager.expression_measurements is previous_store
    assert manager.expression_measurement_freshness_known is previous_freshness
    assert (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr3,
        nucleus.rsum,
        nucleus.rcount,
        manager._expr_corr,
    ) == previous_fields
    assert {path: path.read_bytes() for path in prior_paths} == prior_contents
    assert not any(
        ".csv.tmp" in path.name or ".csv.bak" in path.name
        for path in output_dir.iterdir()
    )


def test_blot_measurement_invalidates_when_unselected_neighbor_moves(tmp_path: Path):
    manager = _manager_with_neighbor()
    run_measure(
        manager,
        _provider(),
        tmp_path,
        0,
        correction_method="blot",
    )
    store = manager.expression_measurements
    selected = manager.get_cell("A")
    assert store is not None and selected is not None
    selected_nucleus = selected.get_nucleus_at(1)
    assert selected_nucleus is not None
    assert store.sample(manager, 0, 1, selected_nucleus) is not None

    # Blot masks all neighbouring projected disks. This bypasses edit history
    # deliberately and must still invalidate A's measured background.
    manager.nuclei_record[0][1].x += 1

    assert not store.dependencies_current(manager)
    assert store.sample(manager, 0, 1, selected_nucleus) is None


def test_valid_black_measurement_replaces_old_nonzero_legacy_value(tmp_path: Path):
    manager = _manager()
    nucleus = manager.nuclei_record[0][0]
    nucleus.rweight = 123456
    nucleus.rwraw = 123456

    written = run_measure(
        manager,
        _provider(0, 0),
        tmp_path,
        0,
        correction_method="none",
    )

    assert nucleus.rwraw == 0
    assert nucleus.rweight == 0
    sample = manager.expression_measurements.sample(manager, 0, 1, nucleus)
    assert sample is not None and sample.value == 0
    with written[0].open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["t1"] == "0"


def test_local_measure_uses_documented_global_fallback_everywhere(tmp_path: Path):
    manager = _manager()
    nucleus = manager.nuclei_record[0][0]
    nucleus.rwcorr2 = 999999

    run_measure(
        manager,
        _provider(100, 100),
        tmp_path,
        0,
        correction_method="local",
    )

    sample = manager.expression_measurements.sample(manager, 0, 1, nucleus)
    assert sample is not None
    assert nucleus.rweight == pytest.approx(sample.value)
    assert manager._expr_corr == "global"


def test_legacy_coverage_is_channel_specific_and_preserves_valid_zero():
    manager = _manager()
    cell = manager.get_cell("A")
    assert cell is not None
    nucleus = manager.nuclei_record[0][0]
    nucleus.rweight = 42
    nucleus.rwraw = 0
    nucleus.rcount = 0

    assert legacy_expression_coverage([cell], "rweight") == (1, 1)
    assert legacy_expression_coverage([cell], "rwraw") == (0, 1)
    assert legacy_expression_coverage([cell], "red_global") == (0, 1)
    assert legacy_expression_coverage([cell], "red_local") == (0, 1)
    assert legacy_expression_coverage([cell], "red_blot") == (0, 1)
    assert legacy_expression_coverage([cell], "red_cross") == (0, 1)

    nucleus.rweight = 0
    nucleus.rcount = 10
    assert legacy_expression_coverage([cell], "rweight") == (1, 1)
    assert legacy_expression_coverage([cell], "rwraw") == (1, 1)
    assert legacy_expression_coverage([cell], "red_global") == (1, 1)


def test_measured_coverage_requires_requested_background_metric(tmp_path: Path):
    manager = _manager()
    run_measure(manager, _provider(), tmp_path, 0, correction_method="global")
    store = manager.expression_measurements
    cell = manager.get_cell("A")
    assert store is not None and cell is not None
    channel = store.channels[0]
    sample = channel.samples[(1, 1)]
    missing_background = replace(
        sample,
        annulus_background=None,
        blot_background=None,
    )
    manager.expression_measurements = replace(
        store,
        channels=(
            replace(channel, samples={(1, 1): missing_background}),
            *store.channels[1:],
        ),
    )
    store = manager.expression_measurements

    assert store.coverage(manager, [cell], 0, metric="value") == (1, 1)
    assert store.coverage(manager, [cell], 0, metric="global") == (0, 1)
    assert store.coverage(manager, [cell], 0, metric="blot") == (0, 1)


def test_unknown_correction_method_is_rejected(tmp_path: Path):
    manager = _manager()
    with pytest.raises(ValueError, match="Unknown correction_method"):
        run_measure(
            manager,
            _provider(),
            tmp_path,
            0,
            correction_method="mystery",
        )
