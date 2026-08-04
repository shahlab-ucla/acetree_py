"""Detached, session-only datasets for multi-movie expression comparison.

The GUI may keep one :class:`ExpressionDatasetRepository` on the application
and share it between any number of comparison windows.  Every XML is loaded
into its own manager and owns its own lazily-created image provider; the active
AceTree document is never borrowed or mutated.

Saved legacy expression is intentionally treated as provenance-unverified.
Recomputed measurements are retained only in memory and are bound to both the
detached manager and a fail-closed source fingerprint.  Measure CSVs are not a
cache format and are never read or written here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Callable

from ..core.cell import Cell
from ..core.nuclei_manager import NucleiManager
from ..core.nucleus import RED_CORRECTIONS, Nucleus
from ..io.config import AceTreeConfig, load_config
from ..io.image_provider import (
    ImageProvider,
    close_worker_image_provider,
    create_image_provider_from_config,
    image_source_manifest_token,
)
from .expression_measurements import ExpressionMeasurementSet, legacy_expression_coverage
from .expression_plot import DEFAULT_EXPRESSION_CHANNELS, ExpressionChannel

logger = logging.getLogger(__name__)

_SOURCE_FINGERPRINT_SCHEMA = 1
_SAVED_CHANNELS = {channel.key: channel for channel in DEFAULT_EXPRESSION_CHANNELS}

ProgressCallback = Callable[[int, int, int, int], bool | None]
MeasurementFunction = Callable[..., ExpressionMeasurementSet]
ImageProviderFactory = Callable[[AceTreeConfig], ImageProvider | None]
ImageProviderCloser = Callable[[ImageProvider], None]


class ExpressionDatasetRepositoryError(RuntimeError):
    """Base class for actionable repository failures."""


class RepositoryClosedError(ExpressionDatasetRepositoryError):
    """The repository has released its session resources."""


class DatasetLoadError(ExpressionDatasetRepositoryError):
    """A config could not be loaded into a detached dataset."""


class DatasetNotLoadedError(ExpressionDatasetRepositoryError):
    """The requested config is not present in this repository."""


class DatasetSourceChangedError(ExpressionDatasetRepositoryError):
    """A source file changed after its detached dataset was loaded."""


class DatasetBusyError(ExpressionDatasetRepositoryError):
    """A destructive or duplicate operation was requested during measurement."""


class CanonicalCellNotFoundError(ExpressionDatasetRepositoryError):
    """No observed cell has the requested exact canonical name."""


class CanonicalCellAmbiguousError(ExpressionDatasetRepositoryError):
    """More than one observed cell has the requested canonical name."""


class ExpressionChannelUnavailableError(ExpressionDatasetRepositoryError):
    """The requested saved or measured expression channel is unavailable."""


class ExpressionDataIncompleteError(ExpressionDatasetRepositoryError):
    """A native trace contains missing, invalid, or unverifiable samples."""


class ImageSourceUnavailableError(ExpressionDatasetRepositoryError):
    """Recomputation requires an image source that could not be opened."""


class MeasurementBackendUnavailableError(ExpressionDatasetRepositoryError):
    """The pure in-memory measurement entry point is unavailable."""


class MeasurementComputationError(ExpressionDatasetRepositoryError):
    """The in-memory measurement backend failed."""


class ExpressionTraceSource(str, Enum):
    """Where a native expression trace came from."""

    SAVED_LEGACY = "saved_legacy"
    RECOMPUTED = "recomputed"


class ExpressionTraceFreshness(str, Enum):
    """Whether measurement provenance is verifiable in this session."""

    UNVERIFIED = "unverified"
    CURRENT_SESSION = "current_session"


@dataclass(frozen=True, slots=True)
class ExpressionTraceProvenance:
    """Machine-readable provenance for one native trace."""

    source: ExpressionTraceSource
    freshness: ExpressionTraceFreshness
    image_channel: int | None
    correction_method: str | None
    at_channel: int | None
    channel_verified: bool
    correction_verified: bool


@dataclass(frozen=True, slots=True)
class NativeExpressionTrace:
    """Immutable, untransformed expression values for exactly one cell."""

    dataset_path: Path
    dataset_fingerprint: str
    cell_name: str
    cell_key: str
    start_time: int
    end_time: int
    timepoints: tuple[int, ...]
    values: tuple[float, ...]
    channel_key: str
    channel_label: str
    channel_unit: str
    provenance: ExpressionTraceProvenance
    dataset_snapshot_token: str = ""
    dataset_generation: int = 0
    image_manifest_token: str | None = None

    def __post_init__(self) -> None:
        if len(self.timepoints) != len(self.values):
            raise ValueError("Native expression timepoints and values must have equal length")
        if not self.timepoints:
            raise ValueError("A native expression trace must contain at least one sample")
        if not self.dataset_snapshot_token:
            object.__setattr__(
                self,
                "dataset_snapshot_token",
                self.dataset_fingerprint,
            )


@dataclass(frozen=True, slots=True)
class ExpressionDatasetStatus:
    """Read-only status suitable for dataset lists in several windows."""

    config_path: Path
    source_fingerprint: str
    num_timepoints: int
    num_cells: int
    image_provider_loaded: bool
    cached_corrections: tuple[str, ...]
    snapshot_token: str = ""
    generation: int = 0
    image_manifest_token: str | None = None
    cell_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.snapshot_token:
            object.__setattr__(self, "snapshot_token", self.source_fingerprint)
        object.__setattr__(self, "cell_names", tuple(self.cell_names))


@dataclass(slots=True)
class _DatasetEntry:
    config_path: Path
    config: AceTreeConfig
    manager: NucleiManager
    source_fingerprint: str
    generation: int
    image_provider: ImageProvider | None = None
    image_manifest_provider: ImageProvider | None = None
    image_manifest_token: str | None = None
    measurements_by_correction: dict[str, ExpressionMeasurementSet] = field(
        default_factory=dict
    )
    lock: threading.RLock = field(default_factory=threading.RLock)
    active: bool = True
    busy_operation: str | None = None
    close_when_idle: bool = False


class ExpressionDatasetRepository:
    """Application-scoped owner of detached expression datasets.

    The object is thread-safe at the dataset level.  Calls for one dataset are
    serialized because built-in providers cache open TIFF/ZIP handles; separate
    datasets can be owned by the same repository without sharing those handles.
    """

    def __init__(
        self,
        *,
        image_provider_factory: ImageProviderFactory = create_image_provider_from_config,
        image_provider_closer: ImageProviderCloser = close_worker_image_provider,
        measurement_function: MeasurementFunction | None = None,
    ) -> None:
        self._entries: dict[str, _DatasetEntry] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._generation = 0
        self._image_provider_factory = image_provider_factory
        self._image_provider_closer = image_provider_closer
        self._measurement_function = measurement_function

    def __enter__(self) -> ExpressionDatasetRepository:
        self._require_open()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def load_dataset(self, config_path: str | Path) -> ExpressionDatasetStatus:
        """Load one XML into a detached manager, deduplicating path aliases."""

        canonical = _canonical_xml_path(config_path, require_exists=True)
        key = _path_key(canonical)
        with self._lock:
            self._require_open()
            existing = self._entries.get(key)
            if existing is not None:
                with existing.lock:
                    self._assert_source_current(existing)
                    return self._status(existing)

            try:
                xml_token_before = _xml_content_token(canonical)
                config = load_config(canonical)
                fingerprint_before = _source_fingerprint(canonical, config)
                if _xml_content_token(canonical) != xml_token_before:
                    raise DatasetSourceChangedError(
                        f"Dataset config changed while it was being parsed: {canonical}"
                    )
                manager = NucleiManager.from_config(config)
                manager.process()
                fingerprint_after = _source_fingerprint(canonical, config)
                if fingerprint_after != fingerprint_before:
                    raise DatasetSourceChangedError(
                        f"Dataset source changed while it was being loaded: {canonical}"
                    )
                if manager.lineage_tree is None:
                    raise DatasetLoadError(
                        f"Dataset has no usable lineage tree: {canonical}"
                    )
            except ExpressionDatasetRepositoryError:
                raise
            except Exception as error:
                raise DatasetLoadError(
                    f"Could not load detached expression dataset {canonical}: {error}"
                ) from error

            entry = _DatasetEntry(
                config_path=canonical,
                config=config,
                manager=manager,
                source_fingerprint=fingerprint_before,
                generation=self._next_generation(),
            )
            self._entries[key] = entry
            return self._status(entry)

    # Concise alias for non-GUI callers.
    add_dataset = load_dataset

    def reload_dataset(self, config_path: str | Path) -> ExpressionDatasetStatus:
        """Explicitly replace one entry after its on-disk sources changed.

        Reloading drops correction caches and closes provider handles. Other
        windows keep their immutable old snapshots, whose fingerprints will no
        longer match and therefore remain non-exportable until prepared again.
        """

        canonical = _canonical_xml_path(config_path, require_exists=True)
        key = _path_key(canonical)
        with self._lock:
            self._require_open()
            existing = self._entries.get(key)
            if existing is not None:
                with existing.lock:
                    if existing.busy_operation is not None:
                        raise DatasetBusyError(
                            f"Dataset is busy with {existing.busy_operation}; "
                            "wait for it to finish before reloading"
                        )
                    existing.active = False
                    self._entries.pop(key, None)
                    self._close_provider(existing)
                    existing.measurements_by_correction.clear()
        # Loading below creates a new generation even when every on-disk stat
        # happens to be identical.
        return self.load_dataset(canonical)

    def status(self, config_path: str | Path) -> ExpressionDatasetStatus:
        """Return current session state after revalidating source metadata."""

        with self._locked_entry(config_path) as entry:
            self._assert_source_current(entry)
            return self._status(entry)

    def session_status(self, config_path: str | Path) -> ExpressionDatasetStatus:
        """Return the loaded snapshot without source revalidation.

        This is intentionally limited to recovery UI: a stale row must remain
        visible so the user can select **Reload**, but it is never suitable for
        cache reuse or export.
        """

        with self._locked_entry(config_path) as entry:
            return self._status(entry)

    def statuses(self) -> tuple[ExpressionDatasetStatus, ...]:
        """Return all loaded datasets in canonical-path order."""

        with self._lock:
            self._require_open()
            entries = tuple(
                sorted(self._entries.values(), key=lambda entry: _path_key(entry.config_path))
            )
        output: list[ExpressionDatasetStatus] = []
        for entry in entries:
            with entry.lock:
                self._assert_source_current(entry)
                output.append(self._status(entry))
        return tuple(output)

    def session_statuses(self) -> tuple[ExpressionDatasetStatus, ...]:
        """List every loaded entry without letting one stale source hide all rows."""

        with self._lock:
            self._require_open()
            paths = tuple(
                entry.config_path
                for entry in sorted(
                    self._entries.values(),
                    key=lambda entry: _path_key(entry.config_path),
                )
            )
        return tuple(self.session_status(path) for path in paths)

    def cell_names(self, config_path: str | Path) -> tuple[str, ...]:
        """Return canonical names available for exact cross-dataset matching."""

        with self._locked_entry(config_path) as entry:
            self._assert_source_current(entry)
            return self._cell_names(entry)

    def image_channel_count(self, config_path: str | Path) -> int:
        """Open the configured provider lazily and report its channel count."""

        with self._locked_entry(config_path) as entry:
            self._assert_source_current(entry)
            return int(self._provider(entry).num_channels)

    def extract_saved_trace(
        self,
        config_path: str | Path,
        canonical_cell_name: str,
        channel_key: str,
    ) -> NativeExpressionTrace:
        """Extract a complete saved legacy trace with unverified provenance."""

        with self._locked_entry(config_path) as entry:
            self._assert_source_current(entry)
            cell = _canonical_cell(entry.manager, canonical_cell_name)
            try:
                channel = _SAVED_CHANNELS[channel_key]
            except KeyError as error:
                choices = ", ".join(sorted(_SAVED_CHANNELS))
                raise ExpressionChannelUnavailableError(
                    f"Saved expression channel {channel_key!r} is unavailable; "
                    f"choose one of: {choices}"
                ) from error

            populated, expected = legacy_expression_coverage([cell], channel_key)
            if expected <= 0 or populated != expected:
                raise ExpressionDataIncompleteError(
                    f"Saved {channel.label} is populated for {populated}/{expected} "
                    f"samples in canonical cell {canonical_cell_name!r}; recompute "
                    "expression before comparison"
                )
            trace = self._legacy_trace(entry, cell, channel)
            self._assert_source_current(entry)
            return trace

    def extract_recomputed_trace(
        self,
        config_path: str | Path,
        canonical_cell_name: str,
        image_channel: int,
        correction_method: str,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> NativeExpressionTrace:
        """Extract one complete trace from a cached or fresh in-memory Measure."""

        with self._locked_entry(
            config_path,
            exclusive_operation="expression recomputation",
        ) as entry:
            self._assert_source_current(entry)
            cell = _canonical_cell(entry.manager, canonical_cell_name)
            method = _validated_correction(correction_method)
            channel_index = _validated_image_channel(image_channel)
            provider = self._provider(entry)
            if channel_index >= provider.num_channels:
                raise ExpressionChannelUnavailableError(
                    f"Image channel {channel_index + 1} is outside the "
                    f"{provider.num_channels}-channel source for {entry.config_path}"
                )
            measurement = self._measurement_set(
                entry,
                method,
                progress_cb=progress_cb,
            )
            try:
                measured_channel = measurement.channel(channel_index)
            except KeyError as error:
                raise ExpressionChannelUnavailableError(
                    f"Recomputed image channel {channel_index + 1} is unavailable in "
                    f"{entry.config_path}"
                ) from error

            timepoints, nuclei = _cell_samples(cell)
            values: list[float] = []
            try:
                for time, nucleus in zip(timepoints, nuclei):
                    sample = measurement.sample(
                        entry.manager,
                        channel_index,
                        time,
                        nucleus,
                    )
                    if sample is None:
                        raise ExpressionDataIncompleteError(
                            f"Recomputed channel {channel_index + 1} is incomplete "
                            f"for canonical cell {canonical_cell_name!r} at time {time}"
                        )
                    values.append(
                        _finite_float(sample.value, canonical_cell_name, time)
                    )
            except ExpressionDataIncompleteError:
                # A transient unreadable stack must be retryable without an
                # explicit source reload.  Do not retain a known-incomplete
                # all-channel correction snapshot indefinitely.
                entry.measurements_by_correction.pop(method, None)
                raise

            self._assert_source_current(entry)
            return NativeExpressionTrace(
                dataset_path=entry.config_path,
                dataset_fingerprint=entry.source_fingerprint,
                cell_name=cell.name,
                cell_key=cell.hash_key or cell.name,
                start_time=int(cell.start_time),
                end_time=int(cell.end_time),
                timepoints=timepoints,
                values=tuple(values),
                channel_key=measured_channel.key,
                channel_label=measured_channel.label,
                channel_unit="scaled mean intensity",
                provenance=ExpressionTraceProvenance(
                    source=ExpressionTraceSource.RECOMPUTED,
                    freshness=ExpressionTraceFreshness.CURRENT_SESSION,
                    image_channel=channel_index,
                    correction_method=method,
                    at_channel=measurement.at_channel,
                    channel_verified=True,
                    correction_verified=True,
                ),
                dataset_snapshot_token=self._snapshot_token(entry),
                dataset_generation=entry.generation,
                image_manifest_token=entry.image_manifest_token,
            )

    def remove_dataset(self, config_path: str | Path) -> bool:
        """Drop one detached dataset and release its provider handles."""

        canonical = _canonical_xml_path(config_path, require_exists=False)
        with self._lock:
            self._require_open()
            key = _path_key(canonical)
            entry = self._entries.get(key)
            if entry is None:
                return False
            with entry.lock:
                if entry.busy_operation is not None:
                    raise DatasetBusyError(
                        f"Dataset is busy with {entry.busy_operation}; "
                        "wait for it to finish before removing it"
                    )
                entry.active = False
                self._entries.pop(key, None)
                self._close_provider(entry)
                entry.measurements_by_correction.clear()
        return True

    def close(self) -> None:
        """Release every provider and make this session repository unusable."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = tuple(self._entries.values())
            self._entries.clear()
        for entry in entries:
            with entry.lock:
                entry.active = False
                if entry.busy_operation is not None:
                    # QApplication.aboutToQuit can be delivered inside a
                    # progress callback on the same thread.  Defer provider
                    # release until the active extraction unwinds.
                    entry.close_when_idle = True
                else:
                    self._close_provider(entry)
                    entry.measurements_by_correction.clear()

    @contextmanager
    def _locked_entry(
        self,
        config_path: str | Path,
        *,
        exclusive_operation: str | None = None,
    ):
        """Lease an active entry without the lookup/close orphan race."""

        canonical = _canonical_xml_path(config_path, require_exists=False)
        with self._lock:
            self._require_open()
            try:
                entry = self._entries[_path_key(canonical)]
            except KeyError as error:
                raise DatasetNotLoadedError(
                    f"Expression dataset is not loaded: {canonical}"
                ) from error
            entry.lock.acquire()
            try:
                if not entry.active:
                    raise DatasetNotLoadedError(
                        f"Expression dataset is no longer active: {canonical}"
                    )
                if (
                    exclusive_operation is not None
                    and entry.busy_operation is not None
                ):
                    raise DatasetBusyError(
                        f"Dataset is busy with {entry.busy_operation}; wait for "
                        f"it to finish before starting {exclusive_operation}"
                    )
                if exclusive_operation is not None:
                    entry.busy_operation = exclusive_operation
            except BaseException:
                entry.lock.release()
                raise
        try:
            yield entry
        finally:
            if exclusive_operation is not None:
                entry.busy_operation = None
            if entry.close_when_idle:
                entry.close_when_idle = False
                self._close_provider(entry)
                entry.measurements_by_correction.clear()
            entry.lock.release()

    def _status(self, entry: _DatasetEntry) -> ExpressionDatasetStatus:
        tree = entry.manager.lineage_tree
        return ExpressionDatasetStatus(
            config_path=entry.config_path,
            source_fingerprint=entry.source_fingerprint,
            num_timepoints=entry.manager.num_timepoints,
            num_cells=tree.num_cells if tree is not None else 0,
            image_provider_loaded=entry.image_provider is not None,
            cached_corrections=tuple(sorted(entry.measurements_by_correction)),
            snapshot_token=self._snapshot_token(entry),
            generation=entry.generation,
            image_manifest_token=entry.image_manifest_token,
            cell_names=self._cell_names(entry),
        )

    @staticmethod
    def _cell_names(entry: _DatasetEntry) -> tuple[str, ...]:
        tree = entry.manager.lineage_tree
        if tree is None:
            return ()
        return tuple(
            sorted(
                {
                    cell.name
                    for cell in tree.all_cells()
                    if cell.name.strip() and cell.nuclei
                },
                key=str.casefold,
            )
        )

    @staticmethod
    def _snapshot_token(entry: _DatasetEntry) -> str:
        payload = json.dumps(
            {
                "schema": 1,
                "source_fingerprint": entry.source_fingerprint,
                "image_manifest_token": entry.image_manifest_token,
                "generation": entry.generation,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def _legacy_trace(
        self,
        entry: _DatasetEntry,
        cell: Cell,
        channel: ExpressionChannel,
    ) -> NativeExpressionTrace:
        timepoints, nuclei = _cell_samples(cell)
        values = tuple(
            _finite_float(channel.reader(cell, time, nucleus), cell.name, time)
            for time, nucleus in zip(timepoints, nuclei)
        )
        return NativeExpressionTrace(
            dataset_path=entry.config_path,
            dataset_fingerprint=entry.source_fingerprint,
            cell_name=cell.name,
            cell_key=cell.hash_key or cell.name,
            start_time=int(cell.start_time),
            end_time=int(cell.end_time),
            timepoints=timepoints,
            values=values,
            channel_key=channel.key,
            channel_label=channel.label,
            channel_unit=channel.unit,
            provenance=ExpressionTraceProvenance(
                source=ExpressionTraceSource.SAVED_LEGACY,
                freshness=ExpressionTraceFreshness.UNVERIFIED,
                image_channel=None,
                correction_method=None,
                at_channel=None,
                channel_verified=False,
                correction_verified=False,
            ),
            dataset_snapshot_token=self._snapshot_token(entry),
            dataset_generation=entry.generation,
            image_manifest_token=entry.image_manifest_token,
        )

    def _measurement_set(
        self,
        entry: _DatasetEntry,
        correction_method: str,
        *,
        progress_cb: ProgressCallback | None,
    ) -> ExpressionMeasurementSet:
        cached = entry.measurements_by_correction.get(correction_method)
        if cached is not None:
            if cached.is_current(entry.manager) and cached.dependencies_current(entry.manager):
                return cached
            entry.measurements_by_correction.pop(correction_method, None)

        provider = self._provider(entry)
        if provider.num_channels <= 0:
            raise ImageSourceUnavailableError(
                f"Image source reports no channels for {entry.config_path}"
            )
        measure = self._resolve_measurement_function()
        try:
            result = measure(
                entry.manager,
                provider,
                at_channel=0,
                correction_method=correction_method,
                progress_cb=progress_cb,
            )
        except ExpressionDatasetRepositoryError:
            raise
        except Exception as error:
            raise MeasurementComputationError(
                f"Expression measurement failed for {entry.config_path}: {error}"
            ) from error

        self._assert_source_current(entry)
        if not isinstance(result, ExpressionMeasurementSet):
            raise MeasurementComputationError(
                "measure_expression_set returned an unexpected result type"
            )
        if result.correction_method != correction_method:
            raise MeasurementComputationError(
                "measure_expression_set returned a mismatched correction method"
            )
        if result.at_channel != 0:
            raise MeasurementComputationError(
                "measure_expression_set returned a mismatched AT channel"
            )
        if not result.is_current(entry.manager) or not result.dependencies_current(
            entry.manager
        ):
            raise MeasurementComputationError(
                "measure_expression_set returned a stale measurement snapshot"
            )
        entry.measurements_by_correction[correction_method] = result
        return result

    def _provider(self, entry: _DatasetEntry) -> ImageProvider:
        if entry.image_provider is not None:
            return entry.image_provider
        try:
            provider = self._image_provider_factory(entry.config)
        except Exception as error:
            raise ImageSourceUnavailableError(
                f"Could not open image source for {entry.config_path}: {error}"
            ) from error
        if provider is None:
            raise ImageSourceUnavailableError(
                f"No readable image source is configured for {entry.config_path}"
            )
        try:
            manifest = self._image_manifest_token(entry, provider)
        except Exception as error:
            try:
                self._image_provider_closer(provider)
            except Exception:
                logger.warning(
                    "Could not close image provider after manifest failure for %s",
                    entry.config_path,
                    exc_info=True,
                )
            raise ImageSourceUnavailableError(
                f"Could not inventory the full image source for "
                f"{entry.config_path}: {error}"
            ) from error
        entry.image_provider = provider
        if manifest is not None:
            entry.image_manifest_provider = provider
            entry.image_manifest_token = manifest
        return provider

    def _resolve_measurement_function(self) -> MeasurementFunction:
        if self._measurement_function is not None:
            return self._measurement_function
        from . import measure_runner

        candidate = getattr(measure_runner, "measure_expression_set", None)
        if not callable(candidate):
            raise MeasurementBackendUnavailableError(
                "The in-memory measure_expression_set backend is unavailable"
            )
        return candidate

    def _assert_source_current(self, entry: _DatasetEntry) -> None:
        if not entry.active:
            if self._closed:
                raise RepositoryClosedError("Expression dataset repository is closed")
            raise DatasetNotLoadedError(
                f"Expression dataset is no longer active: {entry.config_path}"
            )
        try:
            current = _source_fingerprint(entry.config_path, entry.config)
        except Exception as error:
            self._invalidate_entry(entry)
            raise DatasetSourceChangedError(
                f"Could not revalidate source files for {entry.config_path}: {error}"
            ) from error
        if current != entry.source_fingerprint:
            self._invalidate_entry(entry)
            raise DatasetSourceChangedError(
                f"Dataset source changed after it was loaded: {entry.config_path}; "
                "remove and reload it before comparison"
            )
        if (
            entry.image_manifest_provider is not None
            and entry.image_manifest_token is not None
        ):
            try:
                current_manifest = self._image_manifest_token(
                    entry,
                    entry.image_manifest_provider,
                )
            except Exception as error:
                self._invalidate_entry(entry)
                raise DatasetSourceChangedError(
                    f"Could not revalidate the full image movie for "
                    f"{entry.config_path}: {error}"
                ) from error
            if current_manifest != entry.image_manifest_token:
                self._invalidate_entry(entry)
                raise DatasetSourceChangedError(
                    f"An image file in the configured movie changed, appeared, or "
                    f"disappeared after it was opened: {entry.config_path}; remove "
                    "and reload it before comparison"
                )

    @staticmethod
    def _image_manifest_token(
        entry: _DatasetEntry,
        provider: ImageProvider,
    ) -> str | None:
        # The Measure core reads every non-empty absolute nuclei frame in the
        # detached record.  Do not bound this inventory by XML ``start`` /
        # ``end``: legacy archives can contain usable nuclei outside those
        # display/naming bounds, and Measure still requests their stacks.
        timepoints = tuple(
            time
            for time, nuclei in enumerate(entry.manager.nuclei_record, start=1)
            if nuclei
        )
        # Capture paths/stats before asking provider metadata.  OME providers
        # load eagerly when ``num_planes`` is queried; comparing this token to
        # the post-query token prevents cached pixels from being paired with a
        # manifest captured only after a concurrent source replacement.
        before_load = image_source_manifest_token(
            provider,
            timepoints=timepoints,
            planes=None,
        )
        # Built-in per-plane providers read from plane 1 through their own
        # stack depth.  Include the XML end as a conservative lower bound for
        # custom providers whose reported depth is temporarily unavailable.
        try:
            provider_planes = int(provider.num_planes)
        except Exception:  # noqa: BLE001 - manifest validation remains fail closed
            provider_planes = 0
        plane_end = max(1, int(entry.config.plane_end), provider_planes)
        after_load = image_source_manifest_token(
            provider,
            timepoints=timepoints,
            planes=range(1, plane_end + 1),
        )
        if (
            before_load is not None
            and after_load is not None
            and before_load != after_load
        ):
            raise DatasetSourceChangedError(
                "Image movie changed while its provider was being opened"
            )
        return after_load

    def _invalidate_entry(self, entry: _DatasetEntry) -> None:
        entry.measurements_by_correction.clear()
        self._close_provider(entry)

    def _close_provider(self, entry: _DatasetEntry) -> None:
        provider = entry.image_provider
        entry.image_provider = None
        if provider is None:
            return
        try:
            self._image_provider_closer(provider)
        except Exception:
            logger.warning(
                "Could not close expression dataset image provider for %s",
                entry.config_path,
                exc_info=True,
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RepositoryClosedError("Expression dataset repository is closed")

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation


def source_fingerprint_for_config(config_path: str | Path) -> str:
    """Return the deterministic repository fingerprint for one existing XML."""

    canonical = _canonical_xml_path(config_path, require_exists=True)
    try:
        config = load_config(canonical)
        return _source_fingerprint(canonical, config)
    except Exception as error:
        raise DatasetLoadError(
            f"Could not fingerprint expression dataset {canonical}: {error}"
        ) from error


def _canonical_xml_path(path: str | Path, *, require_exists: bool) -> Path:
    candidate = Path(path).expanduser()
    if candidate.suffix.lower() != ".xml":
        raise DatasetLoadError(f"Expression datasets require an XML config: {candidate}")
    try:
        return candidate.resolve(strict=require_exists)
    except FileNotFoundError as error:
        raise DatasetLoadError(f"Dataset config does not exist: {candidate}") from error


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path))


def _xml_content_token(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    content = path.read_bytes()
    return stat.st_size, stat.st_mtime_ns, hashlib.sha256(content).hexdigest()


def _source_fingerprint(config_path: Path, config: AceTreeConfig) -> str:
    config_bytes = config_path.read_bytes()
    image_paths: list[tuple[str, Path]] = []
    if _is_configured_path(config.image_file):
        image_paths.append(("image", Path(config.image_file)))
    for channel, path in sorted(config.image_channels.items()):
        if _is_configured_path(path):
            image_paths.append((f"image_channel_{int(channel)}", Path(path)))

    payload = {
        "schema": _SOURCE_FINGERPRINT_SCHEMA,
        "config": {
            **_stat_record(config_path),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "nuclei_zip": _stat_record(Path(config.zip_file)),
        "images": [
            {"role": role, **_stat_record(path)} for role, path in image_paths
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _stat_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=False)
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {
            "path": _path_key(resolved),
            "exists": False,
            "size": None,
            "mtime_ns": None,
            "is_file": False,
        }
    return {
        "path": _path_key(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "is_file": resolved.is_file(),
    }


def _is_configured_path(path: Path) -> bool:
    return path != Path() and str(path) not in ("", ".")


def _canonical_cell(manager: NucleiManager, name: str) -> Cell:
    if not isinstance(name, str) or not name:
        raise CanonicalCellNotFoundError("Canonical cell name cannot be blank")
    tree = manager.lineage_tree
    if tree is None:
        raise DatasetLoadError("Detached dataset has no lineage tree")
    matches = [cell for cell in tree.all_cells() if cell.name == name and cell.nuclei]
    if not matches:
        raise CanonicalCellNotFoundError(
            f"Canonical cell {name!r} is absent from this dataset"
        )
    unique = {id(cell): cell for cell in matches}
    if len(unique) != 1:
        raise CanonicalCellAmbiguousError(
            f"Canonical cell {name!r} is duplicated in this dataset"
        )
    return next(iter(unique.values()))


def _cell_samples(cell: Cell) -> tuple[tuple[int, ...], tuple[Nucleus, ...]]:
    ordered = sorted(cell.nuclei, key=lambda item: int(item[0]))
    timepoints = tuple(int(time) for time, _nucleus in ordered)
    if not timepoints:
        raise ExpressionDataIncompleteError(
            f"Canonical cell {cell.name!r} has no observed nuclei"
        )
    if len(set(timepoints)) != len(timepoints):
        raise CanonicalCellAmbiguousError(
            f"Canonical cell {cell.name!r} has duplicate samples at one timepoint"
        )
    return timepoints, tuple(nucleus for _time, nucleus in ordered)


def _finite_float(value: object, cell_name: str, time: int) -> float:
    if not isinstance(value, Real):
        raise ExpressionDataIncompleteError(
            f"Expression is missing for canonical cell {cell_name!r} at time {time}"
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise ExpressionDataIncompleteError(
            f"Expression is non-finite for canonical cell {cell_name!r} at time {time}"
        )
    return converted


def _validated_correction(value: str) -> str:
    if value not in RED_CORRECTIONS:
        choices = ", ".join(RED_CORRECTIONS)
        raise ExpressionChannelUnavailableError(
            f"Unknown correction method {value!r}; choose one of: {choices}"
        )
    return value


def _validated_image_channel(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExpressionChannelUnavailableError(
            f"Image channel must be a non-negative 0-based integer, got {value!r}"
        )
    return value


__all__ = [
    "CanonicalCellAmbiguousError",
    "CanonicalCellNotFoundError",
    "DatasetBusyError",
    "DatasetLoadError",
    "DatasetNotLoadedError",
    "DatasetSourceChangedError",
    "ExpressionChannelUnavailableError",
    "ExpressionDataIncompleteError",
    "ExpressionDatasetRepository",
    "ExpressionDatasetRepositoryError",
    "ExpressionDatasetStatus",
    "ExpressionTraceFreshness",
    "ExpressionTraceProvenance",
    "ExpressionTraceSource",
    "ImageSourceUnavailableError",
    "MeasurementBackendUnavailableError",
    "MeasurementComputationError",
    "NativeExpressionTrace",
    "RepositoryClosedError",
    "source_fingerprint_for_config",
]
