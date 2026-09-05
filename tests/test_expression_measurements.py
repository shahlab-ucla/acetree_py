"""Arbitrary-channel Measure retention and concurrency boundaries."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import acetree_py.analysis.measure_runner as measure_runner
from acetree_py.analysis.expression_measurements import legacy_expression_coverage
from acetree_py.analysis.measure_runner import (
    SCALE,
    measure_expression_family,
    measure_expression_set,
    run_measure,
)
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import Nucleus
from acetree_py.io.config import AceTreeConfig
from acetree_py.io.image_provider import (
    NumpyProvider,
    SplitChannelProvider,
    StackTiffProvider,
)


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


def _two_timepoint_provider(value: int = 100) -> NumpyProvider:
    data = np.full((2, 1, 5, 16, 16), value, dtype=np.uint16)
    return NumpyProvider(data)


class _MissingSecondTimepointProvider:
    """One-channel provider with an unreadable second stack."""

    num_channels = 1
    num_timepoints = 2
    num_planes = 5

    def get_stack(self, time: int, channel: int) -> np.ndarray:
        assert channel == 0
        if time == 2:
            raise OSError("second stack unavailable")
        return np.full((5, 16, 16), 100, dtype=np.uint16)


class _UnreadableProvider:
    num_channels = 1
    num_timepoints = 1
    num_planes = 5

    def get_stack(self, _time: int, _channel: int) -> np.ndarray:
        raise OSError("all stacks unavailable")


def _two_timepoint_manager() -> NucleiManager:
    manager = NucleiManager()
    manager.movie.xy_res = 1.0
    manager.movie.z_res = 1.0
    first = Nucleus(
        index=1,
        x=8,
        y=8,
        z=2.0,
        size=6,
        status=1,
        identity="A",
        successor1=1,
    )
    second = Nucleus(
        index=1,
        x=8,
        y=8,
        z=2.0,
        size=6,
        status=1,
        identity="A",
        predecessor=1,
    )
    manager.nuclei_record = [[first], [second]]
    manager.set_all_successors()
    manager.process(do_identity=False)
    return manager


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


def test_measure_expression_set_is_nonmutating_and_retains_all_channels():
    manager = _manager()
    nucleus = manager.nuclei_record[0][0]
    nucleus.rweight = 123
    nucleus.rwraw = 456
    before = (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr3,
        manager._expr_corr,
        manager.expression_measurements,
        manager.expression_measurement_freshness_known,
    )

    store = measure_expression_set(
        manager,
        _provider(),
        correction_method="none",
    )

    after = (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr3,
        manager._expr_corr,
        manager.expression_measurements,
        manager.expression_measurement_freshness_known,
    )
    assert after == before
    assert len(store.channels) == 2
    assert store.csv_paths == ()
    assert store.sample(manager, 1, 1, nucleus).value == pytest.approx(250 * SCALE)


def test_measurement_snapshot_nested_mappings_are_immutable():
    manager = _manager()
    store = measure_expression_set(manager, _provider(), correction_method="none")

    with pytest.raises(TypeError):
        store.geometries[(1, 1)] = store.geometries[(1, 1)]  # type: ignore[index]
    with pytest.raises(TypeError):
        store.channels[0].samples[(1, 1)] = store.channels[0].samples[(1, 1)]  # type: ignore[index]


def test_all_failed_at_channel_aborts_without_clearing_prior_results(tmp_path: Path):
    manager = _manager()
    manager.config = AceTreeConfig(expr_corr="none")
    output_dir = tmp_path / "measure"
    prior_paths = run_measure(
        manager,
        _provider(),
        output_dir,
        0,
        correction_method="none",
    )
    prior_contents = {path: path.read_bytes() for path in prior_paths}
    nucleus = manager.nuclei_record[0][0]
    previous_fields = (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr2,
        nucleus.rwcorr3,
        nucleus.rwcorr4,
        nucleus.rsum,
        nucleus.rcount,
    )
    previous_store = manager.expression_measurements
    previous_correction = manager.config.expr_corr
    previous_dirty = manager._config_dirty

    with pytest.raises(RuntimeError, match="no valid samples"):
        run_measure(
            manager,
            _UnreadableProvider(),
            output_dir,
            0,
            correction_method="global",
        )

    assert manager.expression_measurements is previous_store
    assert manager.config.expr_corr == previous_correction
    assert manager._config_dirty is previous_dirty
    assert (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr2,
        nucleus.rwcorr3,
        nucleus.rwcorr4,
        nucleus.rsum,
        nucleus.rcount,
    ) == previous_fields
    assert {path: path.read_bytes() for path in prior_paths} == prior_contents


def test_prepublication_source_change_is_not_rolled_back(tmp_path: Path, monkeypatch):
    manager = _manager()
    nucleus = manager.nuclei_record[0][0]
    original_builder = measure_runner._build_expression_measurement_set

    def build_then_edit(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        nucleus.rweight = 987654
        return result

    monkeypatch.setattr(
        measure_runner,
        "_build_expression_measurement_set",
        build_then_edit,
    )

    with pytest.raises(RuntimeError, match="preparing results"):
        run_measure(
            manager,
            _provider(),
            tmp_path,
            0,
            correction_method="global",
        )

    assert nucleus.rweight == 987654
    assert manager.expression_measurements is None


def test_run_measure_updates_config_correction_for_later_save(tmp_path: Path):
    manager = _manager()
    manager.config = AceTreeConfig(expr_corr="none")

    run_measure(
        manager,
        _provider(),
        tmp_path,
        0,
        correction_method="blot",
    )

    assert manager._expr_corr == "blot"
    assert manager.config.expr_corr == "blot"
    assert manager._config_dirty


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
    manager = _two_timepoint_manager()
    output_dir = tmp_path / "measure"
    prior_paths = run_measure(
        manager, _two_timepoint_provider(), output_dir, 0, correction_method="none"
    )
    prior_contents = {path: path.read_bytes() for path in prior_paths}
    nucleus = manager.nuclei_record[1][0]
    nucleus.rwcorr2 = 222
    nucleus.rwcorr4 = 444
    # Force the next otherwise-successful Measure to publish a missing sample,
    # exercising rollback of every field cleared by that path.
    nucleus.x = 1000
    previous_store = manager.expression_measurements
    previous_freshness = manager.expression_measurement_freshness_known
    previous_fields = (
        nucleus.rweight,
        nucleus.rwraw,
        nucleus.rwcorr1,
        nucleus.rwcorr2,
        nucleus.rwcorr3,
        nucleus.rwcorr4,
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
            _two_timepoint_provider(999),
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
        nucleus.rwcorr2,
        nucleus.rwcorr3,
        nucleus.rwcorr4,
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


def test_partial_measure_clears_stale_legacy_sample_across_save_reload(
    tmp_path: Path,
):
    manager = _two_timepoint_manager()
    for nucleus in (frame[0] for frame in manager.nuclei_record):
        nucleus.rweight = 101
        nucleus.rsum = 102
        nucleus.rcount = 103
        nucleus.rwraw = 104
        nucleus.rwcorr1 = 105
        nucleus.rwcorr2 = 106
        nucleus.rwcorr3 = 107
        nucleus.rwcorr4 = 108

    written = run_measure(
        manager,
        _MissingSecondTimepointProvider(),
        tmp_path / "measure",
        0,
        correction_method="none",
    )

    first = manager.nuclei_record[0][0]
    missing = manager.nuclei_record[1][0]
    assert first.rcount > 0
    assert (
        missing.rweight,
        missing.rsum,
        missing.rcount,
        missing.rwraw,
        missing.rwcorr1,
        missing.rwcorr2,
        missing.rwcorr3,
        missing.rwcorr4,
    ) == (0, 0, 0, 0, 0, 0, 0, 0)

    cell = manager.get_cell("A")
    store = manager.expression_measurements
    assert cell is not None and store is not None
    assert store.coverage(manager, [cell], 0) == (1, 2)
    with written[0].open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["t1"] != ""
    assert row["t2"] == ""

    archive = tmp_path / "partial.zip"
    manager.save(archive)
    reopened = NucleiManager()
    reopened.load(archive)
    reopened.process(do_identity=False)
    reopened_cell = reopened.get_cell("A")
    assert reopened_cell is not None
    assert legacy_expression_coverage([reopened_cell], "rwraw") == (1, 2)
    assert legacy_expression_coverage([reopened_cell], "rweight") == (1, 2)


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


class _AllChannelCountingProvider:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.all_channel_calls: list[int] = []
        self.single_channel_calls: list[tuple[int, int]] = []

    @property
    def num_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def num_timepoints(self) -> int:
        return int(self.data.shape[0])

    @property
    def num_planes(self) -> int:
        return int(self.data.shape[2])

    @property
    def image_shape(self) -> tuple[int, int]:
        return (int(self.data.shape[-2]), int(self.data.shape[-1]))

    def get_all_channel_stacks(self, time: int) -> tuple[np.ndarray, ...]:
        self.all_channel_calls.append(time)
        return tuple(self.data[time - 1, channel] for channel in range(self.num_channels))

    def get_stack(self, time: int, channel: int) -> np.ndarray:
        self.single_channel_calls.append((time, channel))
        return self.data[time - 1, channel]


def test_measurement_family_loads_once_per_time_and_derives_all_corrections():
    manager = _two_timepoint_manager()
    data = np.empty((2, 2, 5, 16, 16), dtype=np.uint16)
    data[:, 0] = 100
    data[:, 1] = 250
    provider = _AllChannelCountingProvider(data)
    progress: list[tuple[int, int, int, int]] = []

    family = measure_expression_family(
        manager,
        provider,
        progress_cb=lambda *args: progress.append(args) or True,
    )

    assert provider.all_channel_calls == [1, 2]
    assert provider.single_channel_calls == []
    assert progress == [
        (0, 2, 1, 2),
        (1, 2, 1, 2),
        (0, 2, 2, 2),
        (1, 2, 2, 2),
    ]
    nucleus = manager.nuclei_record[0][0]
    sample = family.sample(manager, 1, 1, nucleus)
    assert sample is not None
    assert sample.raw == pytest.approx(250 * SCALE)
    global_value = sample.raw - (sample.annulus_background or 0.0)
    blot_value = sample.raw - (
        sample.blot_background
        if sample.blot_background is not None
        else (sample.annulus_background or 0.0)
    )
    assert sample.corrected_value("none") == pytest.approx(sample.raw)
    assert sample.corrected_value("global") == pytest.approx(global_value)
    assert sample.corrected_value("local") == pytest.approx(global_value)
    assert sample.corrected_value("cross") == pytest.approx(global_value)
    assert sample.corrected_value("blot") == pytest.approx(blot_value)


def test_measurement_family_fallback_provider_is_read_in_time_major_order():
    manager = _two_timepoint_manager()
    data = np.full((2, 2, 5, 16, 16), 100, dtype=np.uint16)

    class FallbackProvider:
        num_channels = 2
        num_timepoints = 2
        num_planes = 5
        image_shape = (16, 16)

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def get_stack(self, time: int, channel: int) -> np.ndarray:
            self.calls.append((time, channel))
            return data[time - 1, channel]

    provider = FallbackProvider()
    measure_expression_family(manager, provider)
    assert provider.calls == [(1, 0), (1, 1), (2, 0), (2, 1)]


def test_measurement_family_bulk_failure_falls_back_to_per_channel_reads():
    manager = _two_timepoint_manager()
    data = np.full((2, 2, 5, 16, 16), 100, dtype=np.uint16)

    class OptionalBulkProvider:
        num_channels = 2
        num_timepoints = 2
        num_planes = 5
        image_shape = (16, 16)

        def __init__(self) -> None:
            self.bulk_calls: list[int] = []
            self.channel_calls: list[tuple[int, int]] = []

        def get_all_channel_stacks(self, time: int) -> tuple[np.ndarray, ...]:
            self.bulk_calls.append(time)
            raise NotImplementedError("bulk loading is unavailable")

        def get_stack(self, time: int, channel: int) -> np.ndarray:
            self.channel_calls.append((time, channel))
            return data[time - 1, channel]

    provider = OptionalBulkProvider()
    family = measure_expression_family(manager, provider)

    assert provider.bulk_calls == [1, 2]
    assert provider.channel_calls == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert family.sample(manager, 0, 1, manager.nuclei_record[0][0]) is not None


def test_measurement_family_cancellation_stops_before_publication():
    manager = _two_timepoint_manager()
    data = np.full((2, 2, 5, 16, 16), 100, dtype=np.uint16)
    provider = _AllChannelCountingProvider(data)

    with pytest.raises(RuntimeError, match="cancelled"):
        measure_expression_family(
            manager,
            provider,
            progress_cb=lambda *_args: False,
        )

    assert provider.all_channel_calls == [1]
    assert manager.expression_measurements is None


def test_split_channel_family_reads_the_raw_stack_once():
    manager = _manager()
    raw = np.empty((5, 16, 32), dtype=np.uint16)
    raw[..., :16] = 10
    raw[..., 16:] = 20

    class InnerProvider:
        num_channels = 1
        num_timepoints = 1
        num_planes = 5
        image_shape = (16, 32)

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def get_stack(self, time: int, channel: int) -> np.ndarray:
            self.calls.append((time, channel))
            return raw

    inner = InnerProvider()
    family = measure_expression_family(
        manager,
        SplitChannelProvider(inner, split=True, flip=False),
    )

    assert inner.calls == [(1, 0)]
    nucleus = manager.nuclei_record[0][0]
    assert family.sample(manager, 0, 1, nucleus).raw == pytest.approx(10 * SCALE)
    assert family.sample(manager, 1, 1, nucleus).raw == pytest.approx(20 * SCALE)


def test_stack_tiff_all_channel_fast_path_decodes_each_page_once(tmp_path: Path):
    class Page:
        def __init__(self, value: int) -> None:
            self.value = value
            self.calls = 0

        def asarray(self) -> np.ndarray:
            self.calls += 1
            return np.full((3, 4), self.value, dtype=np.uint16)

    pages = [Page(value) for value in (10, 20, 30, 40)]
    fake_tiff = type("FakeTiff", (), {"pages": pages})()
    provider = StackTiffProvider(
        tmp_path,
        num_channels=2,
        channel_order="CZ",
    )
    provider._get_tiff_handle = lambda _time: fake_tiff

    channels = provider.get_all_channel_stacks(1)

    assert [page.calls for page in pages] == [1, 1, 1, 1]
    assert channels[0][:, 0, 0].tolist() == [10, 30]
    assert channels[1][:, 0, 0].tolist() == [20, 40]


def test_measurement_family_matches_single_correction_backends_on_nonuniform_data():
    manager = _manager_with_neighbor()
    z, y, x = np.indices((5, 16, 16))
    stack = (17 * z + 11 * y + 3 * x).astype(np.uint16)
    provider = NumpyProvider(stack[np.newaxis, np.newaxis, ...])
    family = measure_expression_family(manager, provider)

    for method in ("none", "global", "blot"):
        reference = measure_expression_set(
            manager,
            provider,
            correction_method=method,
        )
        for nucleus in manager.nuclei_record[0]:
            expected = reference.sample(manager, 0, 1, nucleus)
            assert expected is not None
            assert family.corrected_value(
                manager,
                0,
                1,
                nucleus,
                method,
            ) == pytest.approx(expected.value)


def test_stack_tiff_all_channel_fast_path_rejects_rgb_as_z_stack(tmp_path: Path):
    class RgbPage:
        samplesperpixel = 3

        @staticmethod
        def asarray() -> np.ndarray:
            return np.zeros((3, 8, 6), dtype=np.uint16)

    provider = StackTiffProvider(tmp_path)
    provider._get_tiff_handle = lambda _time: type(
        "FakeTiff", (), {"pages": [RgbPage()]}
    )()

    with pytest.raises(ValueError, match="RGB TIFF page"):
        provider.get_all_channel_stacks(1)


@pytest.mark.parametrize("plane_start", [1, 7])
@pytest.mark.parametrize("correction_method", ["none", "blot"])
def test_absolute_first_plane_measurement_and_origin_freshness(plane_start, correction_method):
    """An asymmetric stack binds raw/blot sampling to the dataset's Z origin."""
    from acetree_py.analysis.expression_measurements import freeze_expression_measurement_family

    manager = NucleiManager()
    manager.config = AceTreeConfig(
        xy_res=1.0, z_res=2.0, plane_start=plane_start, plane_end=plane_start + 2,
    )
    manager.movie.xy_res = 1.0
    manager.movie.z_res = 2.0
    manager.movie.num_planes = 3
    nucleus = Nucleus(index=1, x=4, y=4, z=float(plane_start), size=2, status=1, identity="A")
    manager.nuclei_record = [[nucleus]]
    manager.set_all_successors()
    manager.process(do_identity=False)
    stack = np.full((3, 9, 9), 2000, dtype=np.uint16)
    stack[0] = 10
    for y, x in ((4, 4), (3, 4), (5, 4), (4, 3), (4, 5)):
        stack[0, y, x] = 100
    provider = NumpyProvider(stack[np.newaxis, ...])

    measured = measure_expression_set(manager, provider, correction_method=correction_method)
    family = measure_expression_family(manager, provider)
    sample = measured.sample(manager, 0, 1, nucleus)
    assert sample.raw == pytest.approx(100 * SCALE)
    assert sample.value == pytest.approx((100 if correction_method == "none" else 90) * SCALE)
    assert family.corrected_value(manager, 0, 1, nucleus, correction_method) == sample.value
    assert measured.source_plane_start == family.source_plane_start == plane_start
    frozen = freeze_expression_measurement_family(
        manager, family, dataset_id="origin", source_uri="origin.xml",
        source_fingerprint="source", snapshot_token="snapshot", dataset_generation=0,
        image_manifest_token=None, measured_at="2026-09-05T00:00:00Z",
    )
    assert frozen.measurement_algorithm_version == 2
    assert frozen.is_current(manager)
    manager.config.plane_start += 1
    assert not measured.is_current(manager)
    assert not family.is_current(manager)
    assert not measured.dependencies_current(manager)
    assert not family.dependencies_current(manager, correction_method)
    assert not frozen.is_current(manager)
