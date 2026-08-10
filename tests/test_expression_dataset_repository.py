"""Detached multi-dataset expression repository boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
import csv
import zipfile

import numpy as np
import pytest

from acetree_py.analysis.expression_comparison import (
    BandStatistic,
    CenterStatistic,
    DatasetExpressionTrace,
    DatasetProvenance,
    ExpressionComparisonService,
    ExpressionDataset,
    SummarySpec,
    TraceAvailability,
    export_expression_comparison_tidy_csv,
)
from acetree_py.analysis.expression_dataset_repository import (
    CanonicalCellAmbiguousError,
    CanonicalCellNotFoundError,
    DatasetBusyError,
    DatasetSourceChangedError,
    ExpressionChannelUnavailableError,
    ExpressionDataIncompleteError,
    ExpressionDatasetRepository,
    ExpressionTraceFreshness,
    ExpressionTraceSource,
    ImageSourceUnavailableError,
    MeasurementComputationError,
    RecomputedCacheUnavailableError,
    RepositoryClosedError,
    source_fingerprint_for_config,
)
from acetree_py.analysis.expression_measurements import (
    ExpressionMeasurementSet,
    MeasuredExpressionChannel,
    MeasuredExpressionSample,
    NucleusGeometrySignature,
    expression_measurement_calibration,
    expression_measurement_dependency_fingerprint,
)
from acetree_py.core.nucleus import RED_CORRECTIONS, Nucleus
from acetree_py.io.config import AceTreeConfig, NamingMethod, load_config
from acetree_py.io.config_writer import write_config_xml
from acetree_py.io.nuclei_writer import write_nuclei_zip


class _FakeProvider:
    def __init__(self, num_channels: int = 2) -> None:
        self._num_channels = num_channels

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def num_timepoints(self) -> int:
        return 2

    @property
    def num_planes(self) -> int:
        return 5

    @property
    def image_shape(self) -> tuple[int, int]:
        return (16, 16)

    def get_stack(self, *_args, **_kwargs):
        raise AssertionError("The injected measurement function owns image access")

    def get_plane(self, *_args, **_kwargs):
        raise AssertionError("The injected measurement function owns image access")


class _CountingFamilyProvider:
    def __init__(
        self,
        *,
        fail_second_time_once: bool = False,
        source_files: tuple[Path, ...] = (),
    ) -> None:
        self.data = np.empty((2, 2, 5, 16, 16), dtype=np.uint16)
        self.data[:, 0] = 100
        self.data[:, 1] = 250
        self.all_channel_calls: list[int] = []
        self.single_channel_calls: list[tuple[int, int]] = []
        self.fail_second_time_once = fail_second_time_once
        self._failed_bulk_time: int | None = None
        self.source_files = source_files

    @property
    def num_channels(self) -> int:
        return 2

    @property
    def num_timepoints(self) -> int:
        return 2

    @property
    def num_planes(self) -> int:
        return 5

    @property
    def image_shape(self) -> tuple[int, int]:
        return (16, 16)

    def get_all_channel_stacks(self, time: int) -> tuple[np.ndarray, ...]:
        self.all_channel_calls.append(time)
        if self.fail_second_time_once and time == 2:
            self.fail_second_time_once = False
            self._failed_bulk_time = time
            raise OSError("transient second-timepoint failure")
        return tuple(self.data[time - 1, channel] for channel in range(2))

    def get_stack(self, time: int, channel: int) -> np.ndarray:
        self.single_channel_calls.append((time, channel))
        if self._failed_bulk_time == time:
            if channel == self.num_channels - 1:
                self._failed_bulk_time = None
            raise OSError("transient per-channel fallback failure")
        return self.data[time - 1, channel]

    def image_source_files(self, **_bounds) -> tuple[Path, ...]:
        return self.source_files


def _nucleus(index: int, name: str, *, predecessor: int = -1, complete: bool = True):
    if complete:
        expression = {
            "weight": 50 + index,
            "rweight": 90 + index,
            "rsum": 1000 + index,
            "rcount": 10,
            "rwraw": 100 + index,
            "rwcorr1": 10,
            "rwcorr2": 5,
            "rwcorr3": 8,
            "rwcorr4": 2,
        }
    else:
        expression = {
            "weight": 0,
            "rweight": 0,
            "rsum": 0,
            "rcount": 0,
            "rwraw": 0,
            "rwcorr1": 0,
            "rwcorr2": 0,
            "rwcorr3": 0,
            "rwcorr4": 0,
        }
    return Nucleus(
        index=index,
        x=6 + index,
        y=7,
        z=2.0,
        size=6,
        status=1,
        predecessor=predecessor,
        identity=name,
        assigned_id=name,
        **expression,
    )


def _write_dataset(
    root: Path,
    *,
    name: str = "embryo",
    complete: bool = True,
    duplicate: bool = False,
    second_cell: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / f"{name}_t1.tif"
    image_path.write_bytes(b"representative image")
    zip_path = root / f"{name}.zip"
    if duplicate:
        nuclei = [[_nucleus(1, "ABa"), _nucleus(2, "ABa")]]
    elif second_cell:
        nuclei = [
            [_nucleus(1, "ABa"), _nucleus(2, "ABp")],
            [
                _nucleus(1, "ABa", predecessor=1, complete=complete),
                _nucleus(2, "ABp", predecessor=2, complete=complete),
            ],
        ]
    else:
        nuclei = [
            [_nucleus(1, "ABa")],
            [_nucleus(1, "ABa", predecessor=1, complete=complete)],
        ]
    write_nuclei_zip(nuclei, zip_path)
    xml_path = root / f"{name}.xml"
    config = AceTreeConfig(
        config_file=xml_path,
        zip_file=zip_path,
        image_file=image_path,
        naming_method=NamingMethod.STANDARD,
        ending_index=len(nuclei),
        xy_res=1.0,
        z_res=1.0,
        plane_end=5,
        expr_corr="none",
        split=0,
        flip=0,
    )
    write_config_xml(config, xml_path)
    return xml_path


def _measurement_function(calls: list[str]):
    def measure(
        manager,
        provider,
        *,
        at_channel,
        correction_method,
        progress_cb,
    ):
        calls.append(correction_method)
        if progress_cb is not None:
            progress_cb(0, provider.num_channels, 1, manager.num_timepoints)
        geometries = {
            (time, nucleus.index): NucleusGeometrySignature.from_nucleus(nucleus)
            for time, nuclei in enumerate(manager.nuclei_record, start=1)
            for nucleus in nuclei
            if nucleus.status >= 1
        }
        channels = []
        for channel_index in range(provider.num_channels):
            samples = {}
            for time, nuclei in enumerate(manager.nuclei_record, start=1):
                for nucleus in nuclei:
                    value = float((channel_index + 1) * 100 + time)
                    samples[(time, nucleus.index)] = MeasuredExpressionSample(
                        value=value,
                        raw=value + 10.0,
                        annulus_background=10.0,
                        blot_background=5.0 if correction_method == "blot" else None,
                        inner_pixel_count=20,
                        annulus_pixel_count=10,
                        blot_pixel_count=8 if correction_method == "blot" else 0,
                    )
            channels.append(
                MeasuredExpressionChannel(
                    image_channel=channel_index,
                    label=f"Channel {channel_index + 1}",
                    samples=samples,
                )
            )
        return ExpressionMeasurementSet(
            source_revision=manager.data_revision,
            source_dependency_fingerprint=(
                expression_measurement_dependency_fingerprint(manager)
            ),
            source_calibration=expression_measurement_calibration(manager),
            correction_method=correction_method,
            at_channel=at_channel,
            channels=tuple(channels),
            geometries=geometries,
        )

    return measure


def test_load_deduplicates_canonical_xml_paths(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    repository = ExpressionDatasetRepository()

    first = repository.load_dataset(xml_path)
    alias = xml_path.parent / "." / xml_path.name
    second = repository.load_dataset(alias)

    assert first == second
    assert first.config_path == xml_path.resolve()
    assert first.source_fingerprint == source_fingerprint_for_config(xml_path)
    assert len(repository.statuses()) == 1
    repository.close()


def test_saved_trace_is_complete_immutable_and_explicitly_unverified(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    provider_calls = []

    def unexpected_provider(config):
        provider_calls.append(config)
        return None

    repository = ExpressionDatasetRepository(image_provider_factory=unexpected_provider)
    repository.load_dataset(xml_path)

    trace = repository.extract_saved_trace(xml_path, "ABa", "rweight")

    assert trace.timepoints == (1, 2)
    assert trace.values == (91.0, 91.0)
    assert trace.provenance.source is ExpressionTraceSource.SAVED_LEGACY
    assert trace.provenance.freshness is ExpressionTraceFreshness.UNVERIFIED
    assert trace.provenance.image_channel is None
    assert trace.provenance.correction_method is None
    assert not trace.provenance.channel_verified
    assert not trace.provenance.correction_verified
    assert provider_calls == []
    with pytest.raises(AttributeError):
        trace.values = ()  # type: ignore[misc]


def test_incomplete_saved_trace_fails_closed(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data", complete=False)
    repository = ExpressionDatasetRepository()
    repository.load_dataset(xml_path)

    with pytest.raises(ExpressionDataIncompleteError, match="1/2"):
        repository.extract_saved_trace(xml_path, "ABa", "rweight")


def test_absent_duplicate_and_unknown_saved_channel_fail_closed(tmp_path: Path):
    normal = _write_dataset(tmp_path / "normal")
    duplicate = _write_dataset(tmp_path / "duplicate", duplicate=True)
    repository = ExpressionDatasetRepository()
    repository.load_dataset(normal)
    repository.load_dataset(duplicate)

    with pytest.raises(CanonicalCellNotFoundError):
        repository.extract_saved_trace(normal, "NotACell", "rweight")
    with pytest.raises(CanonicalCellAmbiguousError):
        repository.extract_saved_trace(duplicate, "ABa", "rweight")
    with pytest.raises(ExpressionChannelUnavailableError):
        repository.extract_saved_trace(normal, "ABa", "measured_channel_1")


def test_recomputed_measurements_are_lazy_shared_and_cached_by_correction(
    tmp_path: Path,
):
    xml_path = _write_dataset(tmp_path / "data")
    provider = _FakeProvider(num_channels=2)
    providers_created = []
    providers_closed = []
    measure_calls: list[str] = []
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: providers_created.append(provider) or provider,
        image_provider_closer=providers_closed.append,
        measurement_function=_measurement_function(measure_calls),
    )
    repository.load_dataset(xml_path)
    assert not repository.status(xml_path).image_provider_loaded

    first = repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")
    second_window = repository.extract_recomputed_trace(xml_path, "ABa", 1, "global")
    blot = repository.extract_recomputed_trace(xml_path, "ABa", 0, "blot")

    assert first.values == (101.0, 102.0)
    assert second_window.values == (201.0, 202.0)
    assert blot.provenance.correction_method == "blot"
    assert first.provenance.freshness is ExpressionTraceFreshness.CURRENT_SESSION
    assert first.provenance.channel_verified
    assert first.provenance.correction_verified
    assert providers_created == [provider]
    assert measure_calls == ["global", "blot"]
    assert repository.status(xml_path).cached_corrections == ("blot", "global")
    assert not list(xml_path.parent.rglob("measure*.csv"))

    repository.close()
    repository.close()
    assert providers_closed == [provider]


def test_unavailable_recomputed_channel_fails_closed(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    calls: list[str] = []
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: _FakeProvider(num_channels=1),
        measurement_function=_measurement_function(calls),
    )
    repository.load_dataset(xml_path)

    with pytest.raises(ExpressionChannelUnavailableError):
        repository.extract_recomputed_trace(xml_path, "ABa", 4, "global")

    assert calls == []


def test_default_backend_uses_public_measure_expression_family(
    tmp_path: Path,
    monkeypatch,
):
    xml_path = _write_dataset(tmp_path / "data")
    calls: list[str] = []
    from acetree_py.analysis import measure_runner

    provider = _CountingFamilyProvider()
    real_measure = measure_runner.measure_expression_family

    def counting_measure(*args, **kwargs):
        calls.append("family")
        return real_measure(*args, **kwargs)

    monkeypatch.setattr(
        measure_runner,
        "measure_expression_family",
        counting_measure,
    )
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)

    trace = repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")

    assert trace.values == pytest.approx((0.0, 0.0))
    assert calls == ["family"]


def test_production_family_is_shared_across_cells_channels_and_corrections(
    tmp_path: Path,
    monkeypatch,
):
    xml_path = _write_dataset(tmp_path / "data", second_cell=True)
    provider = _CountingFamilyProvider()
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)

    global_trace = repository.extract_recomputed_trace(
        xml_path, "ABa", 0, "global"
    )
    assert provider.all_channel_calls == [1, 2]
    assert provider.single_channel_calls == []

    import acetree_py.analysis.expression_measurements as measurements_module

    fingerprint_calls = 0
    real_fingerprint = measurements_module.expression_measurement_dependency_fingerprint

    def counting_fingerprint(manager):
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return real_fingerprint(manager)

    monkeypatch.setattr(
        measurements_module,
        "expression_measurement_dependency_fingerprint",
        counting_fingerprint,
    )
    blot_trace = repository.extract_recomputed_trace(xml_path, "ABp", 1, "blot")
    none_trace = repository.extract_recomputed_trace(xml_path, "ABa", 0, "none")
    local_trace = repository.extract_recomputed_trace(xml_path, "ABp", 1, "local")
    cross_trace = repository.extract_recomputed_trace(xml_path, "ABp", 1, "cross")

    assert provider.all_channel_calls == [1, 2]
    assert provider.single_channel_calls == []
    # Only blot depends on every neighbouring nucleus. Raw/global/local/cross
    # reuse the family with calibration plus selected-geometry validation.
    assert fingerprint_calls == 1
    assert global_trace.values == pytest.approx((0.0, 0.0))
    assert blot_trace.values == pytest.approx((0.0, 0.0))
    assert none_trace.values == pytest.approx((100_000.0, 100_000.0))
    assert local_trace.values == pytest.approx(cross_trace.values)
    assert repository.status(xml_path).cached_corrections == (
        "blot",
        "cross",
        "global",
        "local",
        "none",
    )
    entry = next(iter(repository._entries.values()))
    assert entry.measurement_family is not None
    assert entry.measurements_by_correction == {}


def test_full_cache_materializes_every_cell_channel_and_correction_without_reread(
    tmp_path: Path,
):
    xml_path = _write_dataset(tmp_path / "data", second_cell=True)
    provider = _CountingFamilyProvider()
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)

    with pytest.raises(RecomputedCacheUnavailableError):
        repository.snapshot_recomputed_cache(xml_path)

    cache = repository.prepare_recomputed_cache(xml_path)
    assert set(cache.cell_names) == {"ABa", "ABp"}
    assert len(cache.cells) == 2
    assert len(cache.channels) == 2
    assert cache.available_corrections == RED_CORRECTIONS
    calls_after_prepare = list(provider.all_channel_calls)
    assert calls_after_prepare == [1, 2]

    for cell_name in cache.cell_names:
        for image_channel in range(2):
            cell = next(item for item in cache.cells if item.cell_name == cell_name)
            channel = cache.channel(image_channel)
            for correction in RED_CORRECTIONS:
                dataset = cache.materialize_dataset(
                    cell_name,
                    image_channel,
                    correction,
                    {"label": "Frozen embryo", "group_id": "control"},
                )
                assert dataset.provenance.label == "Frozen embryo"
                assert dataset.provenance.group_id == "control"
                assert len(dataset.traces) == 1
                expected = tuple(
                    channel.samples[key].corrected_value(correction)
                    for key in cell.sample_keys
                )
                assert dataset.traces[0].values == pytest.approx(expected)

    assert provider.all_channel_calls == calls_after_prepare
    assert repository.snapshot_recomputed_cache(xml_path) is cache


def test_full_cache_preserves_duplicate_cells_and_missing_sample_reasons(
    tmp_path: Path,
):
    duplicate_xml = _write_dataset(tmp_path / "duplicates", duplicate=True)
    duplicate_provider = _CountingFamilyProvider()
    duplicate_repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: duplicate_provider
    )
    duplicate_repository.load_dataset(duplicate_xml)
    duplicate_cache = duplicate_repository.prepare_recomputed_cache(duplicate_xml)

    assert [cell.cell_name for cell in duplicate_cache.cells] == ["ABa", "ABa"]
    assert len({cell.cell_id for cell in duplicate_cache.cells}) == 2
    ambiguous = duplicate_cache.materialize_dataset("ABa", 0, "global")
    assert not ambiguous.traces
    assert ambiguous.acquisition_statuses[0].availability is TraceAvailability.AMBIGUOUS

    incomplete_xml = _write_dataset(tmp_path / "incomplete")
    incomplete_provider = _CountingFamilyProvider(fail_second_time_once=True)
    incomplete_repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: incomplete_provider
    )
    incomplete_repository.load_dataset(incomplete_xml)
    incomplete_cache = incomplete_repository.prepare_recomputed_cache(incomplete_xml)
    trace = incomplete_cache.materialize_dataset("ABa", 0, "global").traces[0]

    assert trace.values[0] is not None
    assert trace.values[1] is None
    assert trace.missing_reasons[1] == "no measurable inner pixels"
    calls = list(incomplete_provider.all_channel_calls)
    incomplete_cache.materialize_dataset("ABa", 1, "blot")
    assert incomplete_provider.all_channel_calls == calls


def test_force_recompute_is_atomic_and_nonforce_reuses_snapshot(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data", second_cell=True)
    provider = _CountingFamilyProvider()
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)
    first = repository.prepare_recomputed_cache(xml_path)
    first_calls = list(provider.all_channel_calls)

    assert repository.prepare_recomputed_cache(xml_path) is first
    assert provider.all_channel_calls == first_calls

    with pytest.raises(MeasurementComputationError, match="cancelled"):
        repository.prepare_recomputed_cache(
            xml_path,
            force=True,
            progress_cb=lambda *_args: False,
        )
    assert repository.snapshot_recomputed_cache(xml_path) is first

    replacement = repository.prepare_recomputed_cache(xml_path, force=True)
    assert replacement is not first
    assert replacement.measured_at != first.measured_at
    assert repository.snapshot_recomputed_cache(xml_path) is replacement


def test_cached_family_trace_performs_one_source_validation(tmp_path: Path, monkeypatch):
    xml_path = _write_dataset(tmp_path / "data")
    provider = _CountingFamilyProvider()
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)
    repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")

    validation_calls = 0
    real_validate = repository._assert_source_current

    def count_validation(entry):
        nonlocal validation_calls
        validation_calls += 1
        return real_validate(entry)

    monkeypatch.setattr(repository, "_assert_source_current", count_validation)
    repository.extract_recomputed_trace(xml_path, "ABa", 0, "none")

    assert validation_calls == 1
    assert provider.all_channel_calls == [1, 2]


def test_fresh_family_uses_repository_manifest_lease(tmp_path: Path, monkeypatch):
    import acetree_py.analysis.expression_dataset_repository as repository_module
    import acetree_py.analysis.measure_runner as runner_module

    xml_path = _write_dataset(tmp_path / "data")
    representative = xml_path.with_name("embryo_t1.tif")
    sibling = xml_path.with_name("embryo_t2.tif")
    sibling.write_bytes(b"stable sibling")
    provider = _CountingFamilyProvider(source_files=(representative, sibling))
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)

    manifest_calls = 0
    real_manifest = repository_module.image_source_manifest_token

    def count_manifest(*args, **kwargs):
        nonlocal manifest_calls
        manifest_calls += 1
        return real_manifest(*args, **kwargs)

    monkeypatch.setattr(repository_module, "image_source_manifest_token", count_manifest)
    monkeypatch.setattr(runner_module, "image_source_manifest_token", count_manifest)

    repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")

    # Provider opening takes a before/after inventory; repository publication
    # takes one post-read inventory. The runner trusts that stronger lease.
    assert manifest_calls == 3


def test_incomplete_production_family_is_cached_until_explicit_force_retry(
    tmp_path: Path,
):
    xml_path = _write_dataset(tmp_path / "data")
    provider = _CountingFamilyProvider(fail_second_time_once=True)
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)

    with pytest.raises(ExpressionDataIncompleteError):
        repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")

    assert provider.all_channel_calls == [1, 2]
    assert provider.single_channel_calls == [(2, 0), (2, 1)]
    assert repository.status(xml_path).cached_corrections == (
        "blot",
        "cross",
        "global",
        "local",
        "none",
    )
    cached = repository.snapshot_recomputed_cache(xml_path)
    assert cached.materialize_dataset("ABa", 0, "global").traces[0].values[1] is None

    repository.prepare_recomputed_cache(xml_path, force=True)
    trace = repository.extract_recomputed_trace(xml_path, "ABa", 0, "none")
    assert trace.values == pytest.approx((100_000.0, 100_000.0))
    assert provider.all_channel_calls == [1, 2, 1, 2]
    assert repository.status(xml_path).cached_corrections == (
        "blot",
        "cross",
        "global",
        "local",
        "none",
    )


def test_cancelled_production_family_is_not_cached(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    provider = _CountingFamilyProvider()
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)

    with pytest.raises(MeasurementComputationError, match="cancelled"):
        repository.extract_recomputed_trace(
            xml_path,
            "ABa",
            0,
            "global",
            progress_cb=lambda *_args: False,
        )

    assert provider.all_channel_calls == [1]
    assert repository.status(xml_path).cached_corrections == ()


def test_production_family_manifest_change_clears_shared_cache(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    representative = xml_path.with_name("embryo_t1.tif")
    sibling = xml_path.with_name("embryo_t2.tif")
    sibling.write_bytes(b"initial sibling")
    provider = _CountingFamilyProvider(
        source_files=(representative, sibling),
    )
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider
    )
    repository.load_dataset(xml_path)
    repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")
    entry = next(iter(repository._entries.values()))
    assert entry.measurement_family is not None

    sibling.write_bytes(b"changed sibling with a different size")

    with pytest.raises(DatasetSourceChangedError, match="image file"):
        repository.status(xml_path)
    assert entry.measurement_family is None


def test_real_tiff_to_measurement_comparison_and_csv(tmp_path: Path):
    tifffile = pytest.importorskip("tifffile")
    root = tmp_path / "real"
    root.mkdir()
    first_image = root / "real_t1.tif"
    second_image = root / "real_t2.tif"
    tifffile.imwrite(
        first_image,
        np.full((5, 16, 16), 100, dtype=np.uint16),
        photometric="minisblack",
    )
    tifffile.imwrite(
        second_image,
        np.full((5, 16, 16), 250, dtype=np.uint16),
        photometric="minisblack",
    )
    nuclei = [
        [_nucleus(1, "ABa")],
        [_nucleus(1, "ABa", predecessor=1)],
    ]
    zip_path = root / "real.zip"
    write_nuclei_zip(nuclei, zip_path)
    xml_path = root / "real.xml"
    write_config_xml(
        AceTreeConfig(
            config_file=xml_path,
            zip_file=zip_path,
            image_file=first_image,
            naming_method=NamingMethod.STANDARD,
            ending_index=2,
            xy_res=1.0,
            z_res=1.0,
            plane_end=5,
            expr_corr="none",
            split=0,
            flip=0,
        ),
        xml_path,
    )
    repository = ExpressionDatasetRepository()
    status = repository.load_dataset(xml_path)

    native = repository.extract_recomputed_trace(
        xml_path,
        "ABa",
        0,
        "none",
    )
    assert native.timepoints == (1, 2)
    assert native.values == pytest.approx((100000.0, 250000.0))

    dataset = ExpressionDataset(
        provenance=DatasetProvenance(
            dataset_id="real",
            label="real",
            source_uri=str(xml_path),
            source_fingerprint=status.source_fingerprint,
            source_revision=native.dataset_generation,
        ),
        traces=(
            DatasetExpressionTrace(
                cell_name=native.cell_name,
                channel_key=native.channel_key,
                channel_label=native.channel_label,
                channel_unit=native.channel_unit,
                absolute_times=tuple(float(time) for time in native.timepoints),
                values=tuple(native.values),
                birth_time=float(native.start_time),
                end_time=float(native.end_time),
            ),
        ),
    )
    comparison = ExpressionComparisonService().build(
        [dataset],
        cell_names=["ABa"],
        channel_key=native.channel_key,
        summary=SummarySpec(CenterStatistic.MEAN, BandStatistic.NONE),
    )
    csv_path = tmp_path / "real-comparison.csv"
    export_expression_comparison_tidy_csv(comparison, csv_path)

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "aligned_sample" in csv_text
    assert "250000" in csv_text
    repository.close()


def test_t050_archive_remains_absolute_through_repository_and_csv(tmp_path: Path):
    root = tmp_path / "offset"
    root.mkdir()
    zip_path = root / "offset.zip"
    nucleus = _nucleus(1, "late")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("nuclei/t050-nuclei", nucleus.to_text_line() + "\n")
    image_path = root / "offset_t50.tif"
    image_path.write_bytes(b"unused saved-expression image")
    xml_path = root / "offset.xml"
    write_config_xml(
        AceTreeConfig(
            config_file=xml_path,
            zip_file=zip_path,
            image_file=image_path,
            naming_method=NamingMethod.STANDARD,
            starting_index=50,
            ending_index=50,
            xy_res=1.0,
            z_res=1.0,
            plane_end=5,
            expr_corr="none",
        ),
        xml_path,
    )
    repository = ExpressionDatasetRepository()
    repository.load_dataset(xml_path)

    native = repository.extract_saved_trace(xml_path, "late", "rweight")
    assert native.timepoints == (50,)
    assert native.start_time == 50
    assert native.end_time == 50

    comparison = ExpressionComparisonService().build(
        [
            ExpressionDataset(
                DatasetProvenance("offset", "offset"),
                (
                    DatasetExpressionTrace(
                        cell_name="late",
                        channel_key="rweight",
                        absolute_times=(50.0,),
                        values=native.values,
                        birth_time=50.0,
                        end_time=50.0,
                    ),
                ),
            )
        ],
        cell_names=["late"],
        channel_key="rweight",
        summary=SummarySpec(CenterStatistic.MEAN, BandStatistic.NONE),
    )
    csv_path = tmp_path / "offset.csv"
    export_expression_comparison_tidy_csv(comparison, csv_path)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    native_rows = [row for row in rows if row["record_type"] == "native_sample"]
    assert native_rows[0]["absolute_time"] == "50"
    assert native_rows[0]["x"] == "50"


def test_representative_image_stat_change_fails_closed(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    repository = ExpressionDatasetRepository()
    repository.load_dataset(xml_path)
    image_path = xml_path.with_name("embryo_t1.tif")
    image_path.write_bytes(b"replacement image with a different size")

    with pytest.raises(DatasetSourceChangedError):
        repository.extract_saved_trace(xml_path, "ABa", "rweight")


def test_nonrepresentative_movie_sibling_change_invalidates_measurement_cache(
    tmp_path: Path,
):
    xml_path = _write_dataset(tmp_path / "data")
    sibling = xml_path.with_name("embryo_t2.tif")
    sibling.write_bytes(b"original sibling")
    calls: list[str] = []
    repository = ExpressionDatasetRepository(
        measurement_function=_measurement_function(calls),
    )
    repository.load_dataset(xml_path)
    repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")

    sibling.write_bytes(b"changed non-representative sibling with another size")

    with pytest.raises(DatasetSourceChangedError, match="image file"):
        repository.status(xml_path)
    with pytest.raises(DatasetSourceChangedError):
        repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")
    assert calls == ["global"]


def test_manifest_only_reload_creates_a_new_snapshot_generation(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    sibling = xml_path.with_name("embryo_t2.tif")
    sibling.write_bytes(b"original sibling")
    calls: list[str] = []
    repository = ExpressionDatasetRepository(
        measurement_function=_measurement_function(calls),
    )
    repository.load_dataset(xml_path)
    old_trace = repository.extract_recomputed_trace(
        xml_path,
        "ABa",
        0,
        "global",
    )

    sibling.write_bytes(b"changed sibling with a different size")
    with pytest.raises(DatasetSourceChangedError):
        repository.status(xml_path)

    reloaded = repository.reload_dataset(xml_path)

    assert reloaded.source_fingerprint == old_trace.dataset_fingerprint
    assert reloaded.snapshot_token != old_trace.dataset_snapshot_token
    assert reloaded.generation > old_trace.dataset_generation


def test_reentrant_reload_is_rejected_while_recomputation_is_active(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    calls: list[str] = []
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: _FakeProvider(num_channels=1),
        measurement_function=_measurement_function(calls),
    )
    repository.load_dataset(xml_path)
    busy_errors: list[str] = []

    def progress(*_args):
        with pytest.raises(DatasetBusyError) as captured:
            repository.reload_dataset(xml_path)
        busy_errors.append(str(captured.value))
        return True

    trace = repository.extract_recomputed_trace(
        xml_path,
        "ABa",
        0,
        "global",
        progress_cb=progress,
    )

    assert trace.values == (101.0, 102.0)
    assert busy_errors and "busy" in busy_errors[0]
    assert calls == ["global"]


def test_incomplete_recomputation_cache_is_discarded_for_retry(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    calls: list[str] = []
    complete_measure = _measurement_function(calls)

    def transient_measure(*args, **kwargs):
        result = complete_measure(*args, **kwargs)
        if len(calls) != 1:
            return result
        first_channel = result.channels[0]
        partial_samples = dict(first_channel.samples)
        partial_samples.pop((2, 1))
        return replace(
            result,
            channels=(
                replace(first_channel, samples=partial_samples),
                *result.channels[1:],
            ),
        )

    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: _FakeProvider(num_channels=2),
        measurement_function=transient_measure,
    )
    repository.load_dataset(xml_path)

    with pytest.raises(ExpressionDataIncompleteError):
        repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")
    assert repository.status(xml_path).cached_corrections == ()

    trace = repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")

    assert trace.values == (101.0, 102.0)
    assert calls == ["global", "global"]


def test_close_waits_for_active_extraction_and_releases_provider_once(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    provider = _FakeProvider(num_channels=1)
    providers_created: list[_FakeProvider] = []
    providers_closed: list[_FakeProvider] = []
    started = Event()
    release = Event()
    calls: list[str] = []
    complete_measure = _measurement_function(calls)

    def blocking_measure(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return complete_measure(*args, **kwargs)

    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: providers_created.append(provider)
        or provider,
        image_provider_closer=providers_closed.append,
        measurement_function=blocking_measure,
    )
    repository.load_dataset(xml_path)
    results: list[object] = []

    extraction = Thread(
        target=lambda: results.append(
            repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")
        )
    )
    extraction.start()
    assert started.wait(timeout=5)
    shutdown = Thread(target=repository.close)
    shutdown.start()
    release.set()
    extraction.join(timeout=5)
    shutdown.join(timeout=5)

    assert not extraction.is_alive()
    assert not shutdown.is_alive()
    assert len(results) == 1
    assert providers_created == [provider]
    assert providers_closed == [provider]
    with pytest.raises(RepositoryClosedError):
        repository.statuses()


def test_manifest_covers_nonempty_frames_outside_legacy_start_bound(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    first_image = xml_path.with_name("embryo_t1.tif")
    second_image = xml_path.with_name("embryo_t2.tif")
    second_image.write_bytes(b"original second frame")

    # A legacy start bound can exclude a frame from naming/display while the
    # Measure core still processes every non-empty absolute nuclei frame.
    config = load_config(xml_path)
    config.starting_index = 2
    # Make t2 the XML's representative image. A t1 mutation must therefore be
    # caught by the full manifest rather than the inexpensive config-source
    # fingerprint.
    config.image_file = second_image
    write_config_xml(config, xml_path)
    assert first_image.is_file()

    calls: list[str] = []
    repository = ExpressionDatasetRepository(
        measurement_function=_measurement_function(calls),
    )
    repository.load_dataset(xml_path)
    repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")

    first_image.write_bytes(b"changed frame outside configured start")

    with pytest.raises(DatasetSourceChangedError, match="image file"):
        repository.status(xml_path)
    assert calls == ["global"]


def test_provider_open_fails_closed_if_metadata_load_changes_image(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    image_path = xml_path.with_name("embryo_t1.tif")
    closed: list[object] = []

    class MutatingMetadataProvider(_FakeProvider):
        def image_source_files(self, **_bounds):
            return (image_path,)

        @property
        def num_planes(self) -> int:
            image_path.write_bytes(b"changed while provider metadata loaded")
            return 5

    provider = MutatingMetadataProvider(num_channels=1)
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider,
        image_provider_closer=closed.append,
    )
    repository.load_dataset(xml_path)

    with pytest.raises(ImageSourceUnavailableError, match="inventory"):
        repository.image_channel_count(xml_path)

    assert closed == [provider]
    assert not repository.session_status(xml_path).image_provider_loaded


def test_source_change_invalidates_cache_closes_provider_and_fails_closed(
    tmp_path: Path,
):
    xml_path = _write_dataset(tmp_path / "data")
    provider = _FakeProvider()
    closed = []
    calls: list[str] = []
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider,
        image_provider_closer=closed.append,
        measurement_function=_measurement_function(calls),
    )
    repository.load_dataset(xml_path)
    repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")
    xml_path.write_text(xml_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DatasetSourceChangedError):
        repository.extract_saved_trace(xml_path, "ABa", "rweight")

    assert closed == [provider]
    repository.close()
    assert closed == [provider]


def test_explicit_reload_recovers_changed_entry_and_clears_cache(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    provider = _FakeProvider()
    providers_closed = []
    calls: list[str] = []
    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider,
        image_provider_closer=providers_closed.append,
        measurement_function=_measurement_function(calls),
    )
    original = repository.load_dataset(xml_path)
    repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")
    xml_path.write_text(xml_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DatasetSourceChangedError):
        repository.status(xml_path)

    reloaded = repository.reload_dataset(xml_path)

    assert reloaded.source_fingerprint != original.source_fingerprint
    assert reloaded.cached_corrections == ()
    assert providers_closed == [provider]
    repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")
    assert calls == ["global", "global"]


def test_source_change_during_measurement_is_not_cached(tmp_path: Path):
    xml_path = _write_dataset(tmp_path / "data")
    provider = _FakeProvider()
    closed = []
    calls: list[str] = []
    normal_measure = _measurement_function(calls)

    def changing_measure(*args, **kwargs):
        result = normal_measure(*args, **kwargs)
        xml_path.write_text(xml_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return result

    repository = ExpressionDatasetRepository(
        image_provider_factory=lambda _config: provider,
        image_provider_closer=closed.append,
        measurement_function=changing_measure,
    )
    repository.load_dataset(xml_path)

    with pytest.raises(DatasetSourceChangedError):
        repository.extract_recomputed_trace(xml_path, "ABa", 0, "global")

    assert closed == [provider]
    assert calls == ["global"]


def test_remove_and_close_release_session_resources(tmp_path: Path):
    first = _write_dataset(tmp_path / "first", name="first")
    second = _write_dataset(tmp_path / "second", name="second")
    providers = {_path.name: _FakeProvider() for _path in (first, second)}
    closed = []

    def factory(config):
        return providers[config.config_file.name]

    repository = ExpressionDatasetRepository(
        image_provider_factory=factory,
        image_provider_closer=closed.append,
        measurement_function=_measurement_function([]),
    )
    repository.load_dataset(first)
    repository.load_dataset(second)
    repository.extract_recomputed_trace(first, "ABa", 0, "global")
    repository.extract_recomputed_trace(second, "ABa", 0, "global")

    assert repository.remove_dataset(first)
    assert closed == [providers[first.name]]
    repository.close()
    assert closed == [providers[first.name], providers[second.name]]
    with pytest.raises(RepositoryClosedError):
        repository.statuses()
