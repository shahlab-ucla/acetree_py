"""Portable, versioned expression-comparison captures.

An ``.aceexpr`` file stores the materialised native :class:`ExpressionDataset`
inputs and the comparison specification, not aligned samples or summary rows.
Consequently an imported capture can rebuild every supported time, grid,
smoothing, and summary mode without reopening or validating the source XMLs.
Schema v2 can additionally embed correction-neutral, all-cell/all-channel
measurement caches.  Those caches allow the cell, physical channel, and
correction to be changed offline; schema-v1 files remain fixed to their
materialised trace request.

The JSON envelope is checksummed, duplicate-key rejecting, and written by an
atomic same-directory replacement.  Capture provenance is immutable across a
resave; each saved revision receives a new result id linked to its parent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from acetree_py import __version__

from .expression_comparison import (
    ComparisonSpec,
    DatasetAcquisitionStatus,
    DatasetExpressionTrace,
    DatasetProvenance,
    ExpressionComparisonData,
    ExpressionComparisonService,
    ExpressionDataset,
    GridDomain,
    GridSpec,
    SmoothingSpec,
    SummarySpec,
    TraceAvailability,
)
from .expression_plot import TimeAxisMode
from .expression_measurements import (
    EXPRESSION_MEASUREMENT_CACHE_VERSION,
    FrozenCellMeasurements,
    FrozenDatasetMeasurementCache,
    MeasuredExpressionAggregate,
    MeasuredExpressionAggregateChannel,
    NucleusGeometrySignature,
)


EXPRESSION_COMPARISON_RESULT_SCHEMA = "acetree.expression-comparison-result"
EXPRESSION_COMPARISON_RESULT_VERSION = 2
EXPRESSION_COMPARISON_RESULT_LEGACY_VERSION = 1
EXPRESSION_COMPARISON_RESULT_SUFFIX = ".aceexpr"
EXPRESSION_COMPARISON_CALCULATION_VERSION = 1
APPEARANCE_INCLUDED_DATASET_IDS = "included_dataset_ids"

_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_DATASETS = 10_000
_MAX_CHANNELS_PER_CACHE = 256
_MAX_CELLS_PER_CACHE = 2_000_000
_MAX_SAMPLES_PER_CHANNEL = 100_000_000
_MAX_GEOMETRIES_PER_CACHE = 100_000_000
_MAX_MISSING_REASONS_PER_CACHE = 100_000_000
_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UNSET = object()


class ExpressionComparisonResultFormatError(ValueError):
    """Raised when a portable comparison capture is malformed or unsupported."""


class ExpressionComparisonSourceMode(str, Enum):
    """Acquisition family represented by a portable comparison capture."""

    SAVED = "saved"
    RECOMPUTED = "recomputed"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class ExpressionComparisonResult:
    """Immutable native inputs, provenance, and default view for one capture.

    ``captured_at`` and ``producer_version`` describe the original acquisition
    boundary and are preserved by :func:`save_expression_comparison_result`.
    ``saved_at`` is ``None`` only for a newly captured or revised in-memory
    object.  JSON mappings are recursively frozen with mapping proxies/tuples.
    """

    result_id: str
    parent_result_id: str | None
    captured_at: str
    saved_at: str | None
    producer_version: str
    saved_by_version: str
    source_mode: ExpressionComparisonSourceMode
    acquisition_metadata: Mapping[str, Any]
    legacy_acknowledged: bool
    datasets: tuple[ExpressionDataset, ...]
    spec: ComparisonSpec
    appearance: Mapping[str, Any]
    calculation_version: int = EXPRESSION_COMPARISON_CALCULATION_VERSION
    measurement_caches: tuple[FrozenDatasetMeasurementCache, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", _uuid_string(self.result_id, "result_id"))
        if self.parent_result_id is not None:
            parent = _uuid_string(self.parent_result_id, "parent_result_id")
            if parent == self.result_id:
                raise ValueError("parent_result_id cannot equal result_id")
            object.__setattr__(self, "parent_result_id", parent)

        captured = _timestamp(self.captured_at, "captured_at")
        object.__setattr__(self, "captured_at", captured)
        if self.saved_at is not None:
            saved = _timestamp(self.saved_at, "saved_at")
            if _parse_timestamp(saved) < _parse_timestamp(captured):
                raise ValueError("saved_at cannot precede captured_at")
            object.__setattr__(self, "saved_at", saved)

        if not isinstance(self.producer_version, str) or not self.producer_version.strip():
            raise ValueError("producer_version cannot be blank")
        if not isinstance(self.saved_by_version, str):
            raise TypeError("saved_by_version must be a string")
        if self.saved_at is None and self.saved_by_version:
            raise ValueError("an unsaved result cannot have saved_by_version")
        if self.saved_at is not None and not self.saved_by_version.strip():
            raise ValueError("a saved result requires saved_by_version")

        mode = self.source_mode
        if not isinstance(mode, ExpressionComparisonSourceMode):
            mode = _enum_value(ExpressionComparisonSourceMode, mode, "source_mode")
            object.__setattr__(self, "source_mode", mode)
        if type(self.legacy_acknowledged) is not bool:
            raise TypeError("legacy_acknowledged must be a boolean")
        if type(self.calculation_version) is not int:
            raise TypeError("calculation_version must be an integer")
        if self.calculation_version != EXPRESSION_COMPARISON_CALCULATION_VERSION:
            raise ValueError(
                "unsupported expression-comparison calculation version "
                f"{self.calculation_version}"
            )

        metadata = _freeze_json_mapping(
            self.acquisition_metadata, "acquisition_metadata"
        )
        appearance = _freeze_json_mapping(self.appearance, "appearance")
        object.__setattr__(self, "acquisition_metadata", metadata)
        object.__setattr__(self, "appearance", appearance)

        datasets = tuple(self.datasets)
        if not datasets:
            raise ValueError("an expression comparison result requires datasets")
        if len(datasets) > _MAX_DATASETS:
            raise ValueError(f"at most {_MAX_DATASETS} datasets are supported")
        if any(not isinstance(item, ExpressionDataset) for item in datasets):
            raise TypeError("datasets must contain ExpressionDataset objects")
        object.__setattr__(self, "datasets", datasets)
        dataset_ids = {
            item.provenance.dataset_id: item
            for item in datasets
        }

        caches = tuple(self.measurement_caches)
        if len(caches) > _MAX_DATASETS:
            raise ValueError(f"at most {_MAX_DATASETS} measurement caches are supported")
        if any(not isinstance(item, FrozenDatasetMeasurementCache) for item in caches):
            raise TypeError(
                "measurement_caches must contain FrozenDatasetMeasurementCache objects"
            )
        cache_ids = tuple(item.dataset_id for item in caches)
        if len(set(cache_ids)) != len(cache_ids):
            raise ValueError("measurement_caches must have unique dataset ids")
        orphan_cache_ids = set(cache_ids) - set(dataset_ids)
        if orphan_cache_ids:
            raise ValueError(
                "measurement_caches require corresponding default-view datasets: "
                f"{sorted(orphan_cache_ids)!r}"
            )
        for index, cache in enumerate(caches):
            _timestamp(cache.measured_at, f"measurement_caches[{index}].measured_at")
            default_dataset = dataset_ids[cache.dataset_id]
            if (
                default_dataset.provenance.source_fingerprint
                != cache.source_fingerprint
            ):
                raise ValueError(
                    "measurement cache source fingerprint does not match its "
                    f"default-view dataset for {cache.dataset_id!r}"
                )
            if (
                cache.measurement_algorithm_version
                not in (1, EXPRESSION_MEASUREMENT_CACHE_VERSION)
            ):
                raise ValueError(
                    "unsupported frozen measurement algorithm version "
                    f"{cache.measurement_algorithm_version}; expected 1 or "
                    f"{EXPRESSION_MEASUREMENT_CACHE_VERSION}"
                )
        object.__setattr__(self, "measurement_caches", caches)
        _validate_appearance_dataset_inclusion(
            appearance,
            datasets,
            additional_dataset_ids=set(cache_ids),
        )

        spec = _normalise_spec(self.spec)
        object.__setattr__(self, "spec", spec)
        _validate_materialised_inputs(datasets, spec)
        _validate_cache_default_views(self)


@dataclass(frozen=True, slots=True)
class SavedExpressionComparisonResult:
    """The normalized destination and immutable revision written there."""

    path: Path
    result: ExpressionComparisonResult


def capture_expression_comparison_result(
    datasets: Iterable[ExpressionDataset],
    spec: ComparisonSpec,
    *,
    source_mode: ExpressionComparisonSourceMode | str,
    acquisition_metadata: Mapping[str, Any] | None = None,
    legacy_acknowledged: bool = False,
    appearance: Mapping[str, Any] | None = None,
    captured_at: str | None = None,
    producer_version: str = __version__,
    result_id: str | None = None,
    measurement_caches: Iterable[FrozenDatasetMeasurementCache] = (),
) -> ExpressionComparisonResult:
    """Capture validated native inputs before their repository is released."""

    return ExpressionComparisonResult(
        result_id=result_id or str(uuid.uuid4()),
        parent_result_id=None,
        captured_at=captured_at or _utc_now(),
        saved_at=None,
        producer_version=producer_version,
        saved_by_version="",
        source_mode=_enum_value(
            ExpressionComparisonSourceMode, source_mode, "source_mode"
        ),
        acquisition_metadata=acquisition_metadata or {},
        legacy_acknowledged=legacy_acknowledged,
        datasets=tuple(datasets),
        spec=spec,
        appearance=appearance or {},
        measurement_caches=tuple(measurement_caches),
    )


def materialize_expression_measurement_caches(
    result: ExpressionComparisonResult,
    *,
    cell_name: str,
    image_channel: int,
    correction_method: str,
    dataset_ids: Iterable[str] | None = None,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[ExpressionDataset, ...]:
    """Materialise one offline request from embedded full-dataset caches.

    This operation is deliberately repository-free. Missing/ambiguous cells,
    unavailable channels, and partial samples are represented by the cache
    model as explicit statuses or gaps instead of aborting other datasets.
    Schema-v1 results have no measurement caches and therefore cannot use this
    retargeting boundary.
    """

    if not isinstance(result, ExpressionComparisonResult):
        raise TypeError("result must be an ExpressionComparisonResult")
    if not isinstance(cell_name, str) or not cell_name.strip():
        raise ValueError("cell_name cannot be blank")
    if type(image_channel) is not int or image_channel < 0:
        raise ValueError("image_channel must be a non-negative integer")
    if not isinstance(correction_method, str) or not correction_method.strip():
        raise ValueError("correction_method cannot be blank")
    if not result.measurement_caches:
        raise ValueError(
            "this result contains only selected traces and cannot change its "
            "cell, channel, or correction offline"
        )

    by_id = {cache.dataset_id: cache for cache in result.measurement_caches}
    selected_ids = tuple(by_id) if dataset_ids is None else tuple(dataset_ids)
    _validate_dataset_id_selection(selected_ids, set(by_id), "dataset_ids")
    override_map = _normalise_cache_overrides(overrides, set(by_id))
    return tuple(
        by_id[dataset_id].materialize_dataset(
            cell_name,
            image_channel,
            correction_method,
            overrides=override_map.get(dataset_id),
        )
        for dataset_id in selected_ids
    )


def capture_expression_comparison_measurement_caches(
    measurement_caches: Iterable[FrozenDatasetMeasurementCache],
    spec: ComparisonSpec,
    *,
    cell_name: str,
    image_channel: int,
    correction_method: str,
    acquisition_metadata: Mapping[str, Any] | None = None,
    appearance: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
    captured_at: str | None = None,
    producer_version: str = __version__,
    result_id: str | None = None,
) -> ExpressionComparisonResult:
    """Create a portable full-cache result with one editable default view."""

    caches = _normalise_measurement_caches(measurement_caches)
    if not caches:
        raise ValueError("at least one measurement cache is required")
    override_map = _normalise_cache_overrides(
        overrides, {item.dataset_id for item in caches}
    )
    materialised = tuple(
        cache.materialize_dataset(
            cell_name,
            image_channel,
            correction_method,
            overrides=override_map.get(cache.dataset_id),
        )
        for cache in caches
    )
    selected_spec = _spec_for_cache_request(
        spec,
        materialised,
        cell_name=cell_name,
        image_channel=image_channel,
    )
    metadata = dict(acquisition_metadata or {})
    metadata.update(
        {
            "source_mode": ExpressionComparisonSourceMode.RECOMPUTED.value,
            "cell_name": cell_name,
            "image_channel_one_based": image_channel + 1,
            "correction_method": correction_method,
            "capture_scope": "all_cells_all_channels_all_corrections",
        }
    )
    return capture_expression_comparison_result(
        materialised,
        selected_spec,
        source_mode=ExpressionComparisonSourceMode.RECOMPUTED,
        acquisition_metadata=metadata,
        appearance=appearance,
        captured_at=captured_at,
        producer_version=producer_version,
        result_id=result_id,
        measurement_caches=caches,
    )


def revise_expression_comparison_measurement_caches(
    result: ExpressionComparisonResult,
    measurement_caches: Iterable[FrozenDatasetMeasurementCache],
    *,
    replace_existing: bool = False,
    cell_name: str | None = None,
    image_channel: int | None = None,
    correction_method: str | None = None,
    spec: ComparisonSpec | None = None,
    appearance: Mapping[str, Any] | object = _UNSET,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> ExpressionComparisonResult:
    """Append or explicitly replace immutable dataset caches as a child result.

    Existing cache objects and their acquisition provenance are retained
    unchanged unless their dataset id is present in ``measurement_caches`` and
    ``replace_existing`` is true. A conflicting id otherwise fails closed.
    """

    if not isinstance(result, ExpressionComparisonResult):
        raise TypeError("result must be an ExpressionComparisonResult")
    incoming = _normalise_measurement_caches(measurement_caches)
    existing_by_id = {
        cache.dataset_id: cache for cache in result.measurement_caches
    }
    incoming_by_id = {cache.dataset_id: cache for cache in incoming}
    conflicts = {
        dataset_id
        for dataset_id, cache in incoming_by_id.items()
        if dataset_id in existing_by_id and existing_by_id[dataset_id] != cache
    }
    if conflicts and not replace_existing:
        raise ValueError(
            "measurement cache dataset ids already exist with different content: "
            f"{sorted(conflicts)!r}; pass replace_existing=True to replace them"
        )

    merged: list[FrozenDatasetMeasurementCache] = []
    for cache in result.measurement_caches:
        replacement = incoming_by_id.get(cache.dataset_id)
        merged.append(replacement if replacement is not None else cache)
    existing_ids = set(existing_by_id)
    merged.extend(cache for cache in incoming if cache.dataset_id not in existing_ids)
    merged_caches = tuple(merged)
    if not merged_caches:
        raise ValueError("a cache revision requires at least one measurement cache")

    default_cell, default_channel, default_correction = _default_cache_request(result)
    selected_cell = default_cell if cell_name is None else cell_name
    selected_channel = default_channel if image_channel is None else image_channel
    selected_correction = (
        default_correction if correction_method is None else correction_method
    )
    explicit_request_change = (
        selected_cell != default_cell
        or selected_channel != default_channel
        or selected_correction != default_correction
    )
    known_ids = {cache.dataset_id for cache in merged_caches}
    uncached_datasets = tuple(
        dataset
        for dataset in result.datasets
        if dataset.provenance.dataset_id not in known_ids
    )
    if explicit_request_change and uncached_datasets:
        raise ValueError(
            "cannot retarget selected-trace datasets that have no full measurement cache"
        )

    override_map = _presentation_overrides(result.datasets)
    override_map.update(_normalise_cache_overrides(overrides, known_ids))
    materialised_by_id = {
        cache.dataset_id: cache.materialize_dataset(
            selected_cell,
            selected_channel,
            selected_correction,
            overrides=override_map.get(cache.dataset_id),
        )
        for cache in merged_caches
    }
    revised_datasets: list[ExpressionDataset] = []
    seen: set[str] = set()
    for dataset in result.datasets:
        dataset_id = dataset.provenance.dataset_id
        revised_datasets.append(materialised_by_id.get(dataset_id, dataset))
        seen.add(dataset_id)
    for cache in merged_caches:
        if cache.dataset_id not in seen:
            revised_datasets.append(materialised_by_id[cache.dataset_id])
            seen.add(cache.dataset_id)

    base_spec = result.spec if spec is None else _normalise_spec(spec)
    revised_spec = _spec_for_cache_request(
        base_spec,
        tuple(revised_datasets),
        cell_name=selected_cell,
        image_channel=selected_channel,
    )
    next_appearance = result.appearance if appearance is _UNSET else appearance
    if not isinstance(next_appearance, Mapping):
        raise TypeError("appearance must be a mapping")
    metadata = dict(result.acquisition_metadata)
    metadata.update(
        {
            "source_mode": (
                ExpressionComparisonSourceMode.RECOMPUTED.value
                if not uncached_datasets
                else ExpressionComparisonSourceMode.MIXED.value
            ),
            "cell_name": selected_cell,
            "image_channel_one_based": selected_channel + 1,
            "correction_method": selected_correction,
            "capture_scope": (
                "all_cells_all_channels_all_corrections"
                if not uncached_datasets
                else "mixed_full_cache_and_selected_traces"
            ),
        }
    )
    mode = (
        ExpressionComparisonSourceMode.RECOMPUTED
        if not uncached_datasets
        else ExpressionComparisonSourceMode.MIXED
    )
    cache_payload_unchanged = merged_caches == result.measurement_caches
    return ExpressionComparisonResult(
        result_id=str(uuid.uuid4()),
        parent_result_id=result.result_id,
        captured_at=(result.captured_at if cache_payload_unchanged else _utc_now()),
        saved_at=None,
        producer_version=(
            result.producer_version if cache_payload_unchanged else __version__
        ),
        saved_by_version="",
        source_mode=mode,
        acquisition_metadata=metadata,
        legacy_acknowledged=result.legacy_acknowledged,
        datasets=tuple(revised_datasets),
        spec=revised_spec,
        appearance=next_appearance,
        measurement_caches=merged_caches,
        calculation_version=result.calculation_version,
    )


def revise_expression_comparison_result(
    result: ExpressionComparisonResult,
    *,
    datasets: Iterable[ExpressionDataset] | None = None,
    spec: ComparisonSpec | None = None,
    appearance: Mapping[str, Any] | object = _UNSET,
) -> ExpressionComparisonResult:
    """Create an unsaved presentation child without changing captured numbers.

    Dataset label/group and trace series-label/color edits are allowed. Native
    cell/channel identities, times, values, missingness, acquisition statuses,
    and source provenance must remain byte-for-byte equivalent. Dataset Use
    state belongs in ``appearance['included_dataset_ids']`` so excluded native
    data remain available for later views and resaves.
    """

    if not isinstance(result, ExpressionComparisonResult):
        raise TypeError("result must be an ExpressionComparisonResult")
    next_appearance = result.appearance if appearance is _UNSET else appearance
    if not isinstance(next_appearance, Mapping):
        raise TypeError("appearance must be a mapping")
    next_datasets = result.datasets if datasets is None else tuple(datasets)
    _validate_presentation_only_dataset_revision(result.datasets, next_datasets)
    next_spec = result.spec if spec is None else _normalise_spec(spec)
    _validate_presentation_only_spec_revision(result.spec, next_spec)
    return replace(
        result,
        result_id=str(uuid.uuid4()),
        parent_result_id=result.result_id,
        saved_at=None,
        saved_by_version="",
        datasets=next_datasets,
        spec=next_spec,
        appearance=next_appearance,
    )


def save_expression_comparison_result(
    path: str | Path,
    result: ExpressionComparisonResult,
    *,
    saved_by_version: str = __version__,
) -> SavedExpressionComparisonResult:
    """Atomically save a new ``.aceexpr`` revision.

    Saving an already-saved result creates a child revision with a new
    ``result_id`` and the previous id as ``parent_result_id``.  The capture
    timestamp, capture producer, datasets, acquisition metadata, and legacy
    acknowledgement remain unchanged.
    """

    if not isinstance(result, ExpressionComparisonResult):
        raise TypeError("result must be an ExpressionComparisonResult")
    if not isinstance(saved_by_version, str) or not saved_by_version.strip():
        raise ValueError("saved_by_version cannot be blank")

    destination = _with_result_suffix(path)
    now = _utc_now()
    if result.saved_at is None:
        persisted = replace(
            result,
            saved_at=now,
            saved_by_version=saved_by_version,
        )
    else:
        persisted = replace(
            result,
            result_id=str(uuid.uuid4()),
            parent_result_id=result.result_id,
            saved_at=now,
            saved_by_version=saved_by_version,
        )

    text = _encode_envelope(persisted)
    encoded_size = len(text.encode("utf-8"))
    if encoded_size > _MAX_FILE_BYTES:
        raise ValueError(
            f"Expression comparison result exceeds {_MAX_FILE_BYTES} bytes"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, _replacement_mode(destination))
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return SavedExpressionComparisonResult(destination, persisted)


def load_expression_comparison_result(
    path: str | Path,
) -> ExpressionComparisonResult:
    """Load and fully validate a checksummed portable comparison capture."""

    source = Path(path)
    if source.stat().st_size > _MAX_FILE_BYTES:
        raise ExpressionComparisonResultFormatError(
            f"Expression comparison result exceeds {_MAX_FILE_BYTES} bytes"
        )
    try:
        with source.open("r", encoding="utf-8", newline=None) as stream:
            root = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
    except ExpressionComparisonResultFormatError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExpressionComparisonResultFormatError(
            f"Invalid expression comparison JSON in {source}: {error}"
        ) from error

    envelope = _mapping(root, "expression comparison envelope")
    _exact_keys(
        envelope,
        {"schema", "schema_version", "checksum", "result"},
        "expression comparison envelope",
    )
    schema = _string(envelope["schema"], "schema")
    if schema != EXPRESSION_COMPARISON_RESULT_SCHEMA:
        raise ExpressionComparisonResultFormatError(
            f"Unsupported expression comparison schema: {schema!r}"
        )
    version = _integer(envelope["schema_version"], "schema_version")
    if version not in (
        EXPRESSION_COMPARISON_RESULT_LEGACY_VERSION,
        EXPRESSION_COMPARISON_RESULT_VERSION,
    ):
        raise ExpressionComparisonResultFormatError(
            "Unsupported expression comparison schema version "
            f"{version}; expected one of "
            f"{EXPRESSION_COMPARISON_RESULT_LEGACY_VERSION}, "
            f"{EXPRESSION_COMPARISON_RESULT_VERSION}"
        )

    checksum = _mapping(envelope["checksum"], "checksum")
    _exact_keys(checksum, {"algorithm", "sha256"}, "checksum")
    if _string(checksum["algorithm"], "checksum.algorithm") != "sha256":
        raise ExpressionComparisonResultFormatError(
            "checksum.algorithm must be 'sha256'"
        )
    expected = _string(checksum["sha256"], "checksum.sha256")
    if not _SHA256_PATTERN.fullmatch(expected):
        raise ExpressionComparisonResultFormatError(
            "checksum.sha256 must be 64 lowercase hexadecimal characters"
        )
    actual = _payload_checksum(envelope["result"])
    if not hmac.compare_digest(expected, actual):
        raise ExpressionComparisonResultFormatError(
            "Expression comparison result checksum does not match its payload"
        )

    try:
        result = _result_from_payload(envelope["result"], schema_version=version)
    except ExpressionComparisonResultFormatError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise ExpressionComparisonResultFormatError(
            f"Invalid expression comparison result: {error}"
        ) from error
    if result.saved_at is None:
        raise ExpressionComparisonResultFormatError(
            "A persisted expression comparison result requires saved_at"
        )
    return result


def build_expression_comparison_data(
    result: ExpressionComparisonResult,
    *,
    spec: ComparisonSpec | None = None,
    service: ExpressionComparisonService | None = None,
    included_dataset_ids: Iterable[str] | None = None,
) -> ExpressionComparisonData:
    """Rebuild comparison data without consulting an XML/image repository."""

    if not isinstance(result, ExpressionComparisonResult):
        raise TypeError("result must be an ExpressionComparisonResult")
    selected = result.spec if spec is None else _normalise_spec(spec)
    if included_dataset_ids is None:
        configured = result.appearance.get(APPEARANCE_INCLUDED_DATASET_IDS)
        included = (
            tuple(dataset.provenance.dataset_id for dataset in result.datasets)
            if configured is None
            else tuple(configured)
        )
    else:
        included = tuple(included_dataset_ids)
    if not included:
        raise ValueError("at least one dataset must be included to build a comparison")
    if any(not isinstance(item, str) for item in included):
        raise TypeError("included_dataset_ids must contain strings")
    if len(set(included)) != len(included):
        raise ValueError("included_dataset_ids must be unique")
    by_id = {item.provenance.dataset_id: item for item in result.datasets}
    unknown = set(included) - set(by_id)
    if unknown:
        raise ValueError(f"included_dataset_ids are unknown: {sorted(unknown)!r}")
    datasets = tuple(by_id[dataset_id] for dataset_id in included)
    selected = replace(
        selected,
        channel_bindings=tuple(
            item for item in selected.channel_bindings if item[0] in included
        ),
        cell_aliases=tuple(
            item for item in selected.cell_aliases if item[0] in included
        ),
    )
    _validate_materialised_inputs(datasets, selected)
    bindings = dict(selected.channel_bindings)
    aliases = {
        (dataset_id, cell_name): source_name
        for dataset_id, cell_name, source_name in selected.cell_aliases
    }
    return (service or ExpressionComparisonService()).build(
        datasets,
        cell_names=selected.cell_names,
        channel_key=selected.channel_key,
        channel_label=selected.channel_label,
        channel_unit=selected.channel_unit,
        channel_bindings=bindings,
        cell_aliases=aliases,
        time_mode=selected.time_mode,
        grid=selected.grid,
        smoothing=selected.smoothing,
        summary=selected.summary,
    )


def build_expression_comparison_cache_data(
    result: ExpressionComparisonResult,
    *,
    cell_name: str,
    image_channel: int,
    correction_method: str,
    spec: ComparisonSpec | None = None,
    service: ExpressionComparisonService | None = None,
    included_dataset_ids: Iterable[str] | None = None,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> ExpressionComparisonData:
    """Build a comparison for any request embedded in schema-v2 caches."""

    cache_ids = tuple(cache.dataset_id for cache in result.measurement_caches)
    if included_dataset_ids is None:
        configured = result.appearance.get(APPEARANCE_INCLUDED_DATASET_IDS)
        if configured is None:
            included = cache_ids
        else:
            included = tuple(
                dataset_id for dataset_id in configured if dataset_id in cache_ids
            )
    else:
        included = tuple(included_dataset_ids)
    if not included:
        raise ValueError("at least one cached dataset must be included")
    datasets = materialize_expression_measurement_caches(
        result,
        cell_name=cell_name,
        image_channel=image_channel,
        correction_method=correction_method,
        dataset_ids=included,
        overrides=overrides,
    )
    selected = _spec_for_cache_request(
        result.spec if spec is None else _normalise_spec(spec),
        datasets,
        cell_name=cell_name,
        image_channel=image_channel,
    )
    _validate_materialised_inputs(datasets, selected)
    return (service or ExpressionComparisonService()).build(
        datasets,
        cell_names=selected.cell_names,
        channel_key=selected.channel_key,
        channel_label=selected.channel_label,
        channel_unit=selected.channel_unit,
        channel_bindings=dict(selected.channel_bindings),
        cell_aliases={
            (dataset_id, canonical_cell): source_cell
            for dataset_id, canonical_cell, source_cell in selected.cell_aliases
        },
        time_mode=selected.time_mode,
        grid=selected.grid,
        smoothing=selected.smoothing,
        summary=selected.summary,
    )


def _encode_envelope(result: ExpressionComparisonResult) -> str:
    if result.saved_at is None:
        raise ValueError("cannot encode an expression comparison before it is saved")
    payload = _result_to_payload(result)
    envelope = {
        "schema": EXPRESSION_COMPARISON_RESULT_SCHEMA,
        "schema_version": EXPRESSION_COMPARISON_RESULT_VERSION,
        "checksum": {
            "algorithm": "sha256",
            "sha256": _payload_checksum(payload),
        },
        "result": payload,
    }
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if result.measurement_caches:
        # Full caches contain large numeric arrays; whitespace can otherwise
        # multiply portable file size without improving machine readability.
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    return json.dumps(envelope, **options) + "\n"


def _result_to_payload(result: ExpressionComparisonResult) -> dict[str, Any]:
    return {
        "result_id": result.result_id,
        "parent_result_id": result.parent_result_id,
        "captured_at": result.captured_at,
        "saved_at": result.saved_at,
        "producer_version": result.producer_version,
        "saved_by_version": result.saved_by_version,
        "calculation_version": result.calculation_version,
        "source_mode": result.source_mode.value,
        "legacy_acknowledged": result.legacy_acknowledged,
        "acquisition_metadata": _thaw_json(result.acquisition_metadata),
        "appearance": _thaw_json(result.appearance),
        "spec": _spec_to_payload(result.spec),
        "datasets": [_dataset_to_payload(item) for item in result.datasets],
        "measurement_caches": [
            _measurement_cache_to_payload(item)
            for item in result.measurement_caches
        ],
    }


def _result_from_payload(
    value: Any,
    *,
    schema_version: int,
) -> ExpressionComparisonResult:
    data = _mapping(value, "result")
    fields = {
        "result_id",
        "parent_result_id",
        "captured_at",
        "saved_at",
        "producer_version",
        "saved_by_version",
        "calculation_version",
        "source_mode",
        "legacy_acknowledged",
        "acquisition_metadata",
        "appearance",
        "spec",
        "datasets",
    }
    if schema_version >= 2:
        fields.add("measurement_caches")
    _exact_keys(data, fields, "result")
    parent = data["parent_result_id"]
    saved_at = data["saved_at"]
    return ExpressionComparisonResult(
        result_id=_string(data["result_id"], "result.result_id"),
        parent_result_id=(
            None
            if parent is None
            else _string(parent, "result.parent_result_id")
        ),
        captured_at=_string(data["captured_at"], "result.captured_at"),
        saved_at=(
            None
            if saved_at is None
            else _string(saved_at, "result.saved_at")
        ),
        producer_version=_string(
            data["producer_version"], "result.producer_version"
        ),
        saved_by_version=_string(
            data["saved_by_version"], "result.saved_by_version"
        ),
        calculation_version=_integer(
            data["calculation_version"], "result.calculation_version"
        ),
        source_mode=_enum_value(
            ExpressionComparisonSourceMode,
            data["source_mode"],
            "result.source_mode",
        ),
        legacy_acknowledged=_boolean(
            data["legacy_acknowledged"], "result.legacy_acknowledged"
        ),
        acquisition_metadata=_mapping(
            data["acquisition_metadata"], "result.acquisition_metadata"
        ),
        appearance=_mapping(data["appearance"], "result.appearance"),
        spec=_spec_from_payload(data["spec"]),
        datasets=tuple(
            _dataset_from_payload(item, index)
            for index, item in enumerate(
                _bounded_sequence(
                    data["datasets"],
                    "result.datasets",
                    maximum=_MAX_DATASETS,
                )
            )
        ),
        measurement_caches=(
            ()
            if schema_version == EXPRESSION_COMPARISON_RESULT_LEGACY_VERSION
            else tuple(
                _measurement_cache_from_payload(
                    item,
                    f"result.measurement_caches[{index}]",
                )
                for index, item in enumerate(
                    _bounded_sequence(
                        data["measurement_caches"],
                        "result.measurement_caches",
                        maximum=_MAX_DATASETS,
                    )
                )
            )
        ),
    )


def _measurement_cache_to_payload(
    cache: FrozenDatasetMeasurementCache,
) -> dict[str, Any]:
    return {
        "dataset_id": cache.dataset_id,
        "source_uri": cache.source_uri,
        "source_fingerprint": cache.source_fingerprint,
        "snapshot_token": cache.snapshot_token,
        "dataset_generation": cache.dataset_generation,
        "image_manifest_token": cache.image_manifest_token,
        "measured_at": cache.measured_at,
        "measurement_algorithm_version": cache.measurement_algorithm_version,
        "source_revision": cache.source_revision,
        "source_dependency_fingerprint": cache.source_dependency_fingerprint,
        "source_calibration": list(cache.source_calibration),
        "channels": [
            {
                "image_channel": channel.image_channel,
                "label": channel.label,
                "samples": [
                    [
                        time,
                        nucleus_index,
                        sample.raw,
                        sample.annulus_background,
                        sample.blot_background,
                        sample.inner_pixel_count,
                        sample.annulus_pixel_count,
                        sample.blot_pixel_count,
                    ]
                    for (time, nucleus_index), sample in sorted(
                        channel.samples.items()
                    )
                ],
            }
            for channel in cache.channels
        ],
        "cells": [
            {
                "cell_id": cell.cell_id,
                "cell_name": cell.cell_name,
                "start_time": cell.start_time,
                "end_time": cell.end_time,
                "sample_keys": [list(key) for key in cell.sample_keys],
            }
            for cell in cache.cells
        ],
        "geometries": [
            [
                time,
                nucleus_index,
                geometry.x,
                geometry.y,
                geometry.z,
                geometry.size,
                geometry.status,
            ]
            for (time, nucleus_index), geometry in sorted(cache.geometries.items())
        ],
        "missing_reasons": [
            [image_channel, time, nucleus_index, reason]
            for (image_channel, time, nucleus_index), reason in sorted(
                cache.missing_reasons.items()
            )
        ],
    }


def _measurement_cache_from_payload(
    value: Any,
    label: str,
) -> FrozenDatasetMeasurementCache:
    data = _mapping(value, label)
    _exact_keys(
        data,
        {
            "dataset_id",
            "source_uri",
            "source_fingerprint",
            "snapshot_token",
            "dataset_generation",
            "image_manifest_token",
            "measured_at",
            "measurement_algorithm_version",
            "source_revision",
            "source_dependency_fingerprint",
            "source_calibration",
            "channels",
            "cells",
            "geometries",
            "missing_reasons",
        },
        label,
    )
    calibration = _bounded_sequence(
        data["source_calibration"],
        f"{label}.source_calibration",
        maximum=4,
    )
    if len(calibration) != 4:
        raise ExpressionComparisonResultFormatError(
            f"{label}.source_calibration must contain four values"
        )

    channels: list[MeasuredExpressionAggregateChannel] = []
    seen_channels: set[int] = set()
    for channel_index, raw_channel in enumerate(
        _bounded_sequence(
            data["channels"],
            f"{label}.channels",
            maximum=_MAX_CHANNELS_PER_CACHE,
        )
    ):
        channel_label = f"{label}.channels[{channel_index}]"
        channel_data = _mapping(raw_channel, channel_label)
        _exact_keys(
            channel_data,
            {"image_channel", "label", "samples"},
            channel_label,
        )
        image_channel = _nonnegative_integer(
            channel_data["image_channel"], f"{channel_label}.image_channel"
        )
        if image_channel in seen_channels:
            raise ExpressionComparisonResultFormatError(
                f"{label} has duplicate image channel {image_channel}"
            )
        seen_channels.add(image_channel)
        samples: dict[tuple[int, int], MeasuredExpressionAggregate] = {}
        for sample_index, raw_sample in enumerate(
            _bounded_sequence(
                channel_data["samples"],
                f"{channel_label}.samples",
                maximum=_MAX_SAMPLES_PER_CHANNEL,
            )
        ):
            sample_label = f"{channel_label}.samples[{sample_index}]"
            row = _sequence(raw_sample, sample_label)
            if len(row) != 8:
                raise ExpressionComparisonResultFormatError(
                    f"{sample_label} must contain eight values"
                )
            key = (
                _nonnegative_integer(row[0], f"{sample_label}[0]"),
                _nonnegative_integer(row[1], f"{sample_label}[1]"),
            )
            if key in samples:
                raise ExpressionComparisonResultFormatError(
                    f"{channel_label} has duplicate sample key {key!r}"
                )
            samples[key] = MeasuredExpressionAggregate(
                raw=_number(row[2], f"{sample_label}[2]"),
                annulus_background=(
                    None
                    if row[3] is None
                    else _number(row[3], f"{sample_label}[3]")
                ),
                blot_background=(
                    None
                    if row[4] is None
                    else _number(row[4], f"{sample_label}[4]")
                ),
                inner_pixel_count=_nonnegative_integer(
                    row[5], f"{sample_label}[5]"
                ),
                annulus_pixel_count=_nonnegative_integer(
                    row[6], f"{sample_label}[6]"
                ),
                blot_pixel_count=_nonnegative_integer(
                    row[7], f"{sample_label}[7]"
                ),
            )
        channels.append(
            MeasuredExpressionAggregateChannel(
                image_channel=image_channel,
                label=_string(channel_data["label"], f"{channel_label}.label"),
                samples=samples,
            )
        )

    cells: list[FrozenCellMeasurements] = []
    seen_cell_ids: set[str] = set()
    for cell_index, raw_cell in enumerate(
        _bounded_sequence(
            data["cells"],
            f"{label}.cells",
            maximum=_MAX_CELLS_PER_CACHE,
        )
    ):
        cell_label = f"{label}.cells[{cell_index}]"
        cell_data = _mapping(raw_cell, cell_label)
        _exact_keys(
            cell_data,
            {"cell_id", "cell_name", "start_time", "end_time", "sample_keys"},
            cell_label,
        )
        cell_id = _string(cell_data["cell_id"], f"{cell_label}.cell_id")
        if cell_id in seen_cell_ids:
            raise ExpressionComparisonResultFormatError(
                f"{label} has duplicate cell_id {cell_id!r}"
            )
        seen_cell_ids.add(cell_id)
        sample_keys: list[tuple[int, int]] = []
        seen_sample_keys: set[tuple[int, int]] = set()
        for key_index, raw_key in enumerate(
            _bounded_sequence(
                cell_data["sample_keys"],
                f"{cell_label}.sample_keys",
                maximum=_MAX_SAMPLES_PER_CHANNEL,
            )
        ):
            key_label = f"{cell_label}.sample_keys[{key_index}]"
            key_row = _sequence(raw_key, key_label)
            if len(key_row) != 2:
                raise ExpressionComparisonResultFormatError(
                    f"{key_label} must contain time and nucleus index"
                )
            key = (
                _nonnegative_integer(key_row[0], f"{key_label}[0]"),
                _nonnegative_integer(key_row[1], f"{key_label}[1]"),
            )
            if key in seen_sample_keys:
                raise ExpressionComparisonResultFormatError(
                    f"{cell_label} has duplicate sample key {key!r}"
                )
            seen_sample_keys.add(key)
            sample_keys.append(key)
        cells.append(
            FrozenCellMeasurements(
                cell_id=cell_id,
                cell_name=_string(
                    cell_data["cell_name"], f"{cell_label}.cell_name"
                ),
                start_time=_nonnegative_integer(
                    cell_data["start_time"], f"{cell_label}.start_time"
                ),
                end_time=_nonnegative_integer(
                    cell_data["end_time"], f"{cell_label}.end_time"
                ),
                sample_keys=tuple(sample_keys),
            )
        )

    geometries: dict[tuple[int, int], NucleusGeometrySignature] = {}
    for geometry_index, raw_geometry in enumerate(
        _bounded_sequence(
            data["geometries"],
            f"{label}.geometries",
            maximum=_MAX_GEOMETRIES_PER_CACHE,
        )
    ):
        geometry_label = f"{label}.geometries[{geometry_index}]"
        row = _sequence(raw_geometry, geometry_label)
        if len(row) != 7:
            raise ExpressionComparisonResultFormatError(
                f"{geometry_label} must contain seven values"
            )
        key = (
            _nonnegative_integer(row[0], f"{geometry_label}[0]"),
            _nonnegative_integer(row[1], f"{geometry_label}[1]"),
        )
        if key in geometries:
            raise ExpressionComparisonResultFormatError(
                f"{label} has duplicate geometry key {key!r}"
            )
        geometries[key] = NucleusGeometrySignature(
            x=_integer(row[2], f"{geometry_label}[2]"),
            y=_integer(row[3], f"{geometry_label}[3]"),
            z=_number(row[4], f"{geometry_label}[4]"),
            size=_nonnegative_integer(row[5], f"{geometry_label}[5]"),
            status=_integer(row[6], f"{geometry_label}[6]"),
        )

    missing_reasons: dict[tuple[int, int, int], str] = {}
    for reason_index, raw_reason in enumerate(
        _bounded_sequence(
            data["missing_reasons"],
            f"{label}.missing_reasons",
            maximum=_MAX_MISSING_REASONS_PER_CACHE,
        )
    ):
        reason_label = f"{label}.missing_reasons[{reason_index}]"
        row = _sequence(raw_reason, reason_label)
        if len(row) != 4:
            raise ExpressionComparisonResultFormatError(
                f"{reason_label} must contain four values"
            )
        key = (
            _nonnegative_integer(row[0], f"{reason_label}[0]"),
            _nonnegative_integer(row[1], f"{reason_label}[1]"),
            _nonnegative_integer(row[2], f"{reason_label}[2]"),
        )
        if key in missing_reasons:
            raise ExpressionComparisonResultFormatError(
                f"{label} has duplicate missing-reason key {key!r}"
            )
        reason = _string(row[3], f"{reason_label}[3]")
        if not reason.strip():
            raise ExpressionComparisonResultFormatError(
                f"{reason_label}[3] cannot be blank"
            )
        missing_reasons[key] = reason

    manifest = data["image_manifest_token"]
    return FrozenDatasetMeasurementCache(
        dataset_id=_string(data["dataset_id"], f"{label}.dataset_id"),
        source_uri=_string(data["source_uri"], f"{label}.source_uri"),
        source_fingerprint=_string(
            data["source_fingerprint"], f"{label}.source_fingerprint"
        ),
        snapshot_token=_string(data["snapshot_token"], f"{label}.snapshot_token"),
        dataset_generation=_nonnegative_integer(
            data["dataset_generation"], f"{label}.dataset_generation"
        ),
        image_manifest_token=(
            None
            if manifest is None
            else _string(manifest, f"{label}.image_manifest_token")
        ),
        measured_at=_timestamp(data["measured_at"], f"{label}.measured_at"),
        measurement_algorithm_version=_nonnegative_integer(
            data["measurement_algorithm_version"],
            f"{label}.measurement_algorithm_version",
        ),
        source_revision=_nonnegative_integer(
            data["source_revision"], f"{label}.source_revision"
        ),
        source_dependency_fingerprint=_string(
            data["source_dependency_fingerprint"],
            f"{label}.source_dependency_fingerprint",
        ),
        source_calibration=(
            _number(calibration[0], f"{label}.source_calibration[0]"),
            _number(calibration[1], f"{label}.source_calibration[1]"),
            _integer(calibration[2], f"{label}.source_calibration[2]"),
            _number(calibration[3], f"{label}.source_calibration[3]"),
        ),
        channels=tuple(channels),
        cells=tuple(cells),
        geometries=geometries,
        missing_reasons=missing_reasons,
    )


def _dataset_to_payload(dataset: ExpressionDataset) -> dict[str, Any]:
    source = dataset.provenance
    return {
        "provenance": {
            "dataset_id": source.dataset_id,
            "label": source.label,
            "group_id": source.group_id,
            "source_uri": source.source_uri,
            "source_fingerprint": source.source_fingerprint,
            "source_revision": source.source_revision,
            "metadata": [[key, value] for key, value in source.metadata],
        },
        "traces": [_trace_to_payload(trace) for trace in dataset.traces],
        "acquisition_statuses": [
            _status_to_payload(status) for status in dataset.acquisition_statuses
        ],
    }


def _dataset_from_payload(value: Any, index: int) -> ExpressionDataset:
    label = f"result.datasets[{index}]"
    data = _mapping(value, label)
    _exact_keys(data, {"provenance", "traces", "acquisition_statuses"}, label)
    raw_source = _mapping(data["provenance"], f"{label}.provenance")
    provenance_fields = {
        "dataset_id",
        "label",
        "group_id",
        "source_uri",
        "source_fingerprint",
        "source_revision",
        "metadata",
    }
    _exact_keys(raw_source, provenance_fields, f"{label}.provenance")
    metadata: list[tuple[str, str]] = []
    seen_metadata: set[str] = set()
    for item_index, raw_item in enumerate(
        _sequence(raw_source["metadata"], f"{label}.provenance.metadata")
    ):
        item = _sequence(
            raw_item, f"{label}.provenance.metadata[{item_index}]"
        )
        if len(item) != 2:
            raise ExpressionComparisonResultFormatError(
                f"{label}.provenance.metadata[{item_index}] must have two items"
            )
        key = _string(item[0], f"{label}.provenance.metadata[{item_index}][0]")
        child = _string(item[1], f"{label}.provenance.metadata[{item_index}][1]")
        if key in seen_metadata:
            raise ExpressionComparisonResultFormatError(
                f"duplicate provenance metadata key {key!r}"
            )
        seen_metadata.add(key)
        metadata.append((key, child))
    revision = raw_source["source_revision"]
    if revision is not None:
        revision = _integer(revision, f"{label}.provenance.source_revision")
    source = DatasetProvenance(
        dataset_id=_string(raw_source["dataset_id"], f"{label}.provenance.dataset_id"),
        label=_string(raw_source["label"], f"{label}.provenance.label"),
        group_id=_string(raw_source["group_id"], f"{label}.provenance.group_id"),
        source_uri=_string(raw_source["source_uri"], f"{label}.provenance.source_uri"),
        source_fingerprint=_string(
            raw_source["source_fingerprint"],
            f"{label}.provenance.source_fingerprint",
        ),
        source_revision=revision,
        metadata=tuple(metadata),
    )
    return ExpressionDataset(
        provenance=source,
        traces=tuple(
            _trace_from_payload(item, f"{label}.traces[{item_index}]")
            for item_index, item in enumerate(
                _sequence(data["traces"], f"{label}.traces")
            )
        ),
        acquisition_statuses=tuple(
            _status_from_payload(
                item, f"{label}.acquisition_statuses[{item_index}]"
            )
            for item_index, item in enumerate(
                _sequence(
                    data["acquisition_statuses"],
                    f"{label}.acquisition_statuses",
                )
            )
        ),
    )


def _trace_to_payload(trace: DatasetExpressionTrace) -> dict[str, Any]:
    return {
        "cell_name": trace.cell_name,
        "channel_key": trace.channel_key,
        "channel_label": trace.channel_label,
        "channel_unit": trace.channel_unit,
        "absolute_times": list(trace.absolute_times),
        "values": list(trace.values),
        "birth_time": trace.birth_time,
        "end_time": trace.end_time,
        "missing_reasons": list(trace.missing_reasons),
        "series_label": trace.series_label,
        "color": trace.color,
    }


def _trace_from_payload(value: Any, label: str) -> DatasetExpressionTrace:
    data = _mapping(value, label)
    fields = {
        "cell_name",
        "channel_key",
        "channel_label",
        "channel_unit",
        "absolute_times",
        "values",
        "birth_time",
        "end_time",
        "missing_reasons",
        "series_label",
        "color",
    }
    _exact_keys(data, fields, label)
    series = data["series_label"]
    color = data["color"]
    reasons = _sequence(data["missing_reasons"], f"{label}.missing_reasons")
    return DatasetExpressionTrace(
        cell_name=_string(data["cell_name"], f"{label}.cell_name"),
        channel_key=_string(data["channel_key"], f"{label}.channel_key"),
        channel_label=_string(data["channel_label"], f"{label}.channel_label"),
        channel_unit=_string(data["channel_unit"], f"{label}.channel_unit"),
        absolute_times=tuple(
            _number(item, f"{label}.absolute_times[{index}]")
            for index, item in enumerate(
                _sequence(data["absolute_times"], f"{label}.absolute_times")
            )
        ),
        values=tuple(
            None if item is None else _number(item, f"{label}.values[{index}]")
            for index, item in enumerate(
                _sequence(data["values"], f"{label}.values")
            )
        ),
        birth_time=_number(data["birth_time"], f"{label}.birth_time"),
        end_time=_number(data["end_time"], f"{label}.end_time"),
        missing_reasons=tuple(
            None
            if item is None
            else _string(item, f"{label}.missing_reasons[{index}]")
            for index, item in enumerate(reasons)
        ),
        series_label=(
            None if series is None else _string(series, f"{label}.series_label")
        ),
        color=None if color is None else _string(color, f"{label}.color"),
    )


def _status_to_payload(status: DatasetAcquisitionStatus) -> dict[str, Any]:
    return {
        "cell_name": status.cell_name,
        "channel_key": status.channel_key,
        "availability": status.availability.value,
        "message": status.message,
        "source_cell_name": status.source_cell_name,
        "source_channel_key": status.source_channel_key,
    }


def _status_from_payload(value: Any, label: str) -> DatasetAcquisitionStatus:
    data = _mapping(value, label)
    fields = {
        "cell_name",
        "channel_key",
        "availability",
        "message",
        "source_cell_name",
        "source_channel_key",
    }
    _exact_keys(data, fields, label)
    return DatasetAcquisitionStatus(
        cell_name=_string(data["cell_name"], f"{label}.cell_name"),
        channel_key=_string(data["channel_key"], f"{label}.channel_key"),
        availability=_enum_value(
            TraceAvailability, data["availability"], f"{label}.availability"
        ),
        message=_string(data["message"], f"{label}.message"),
        source_cell_name=_string(
            data["source_cell_name"], f"{label}.source_cell_name"
        ),
        source_channel_key=_string(
            data["source_channel_key"], f"{label}.source_channel_key"
        ),
    )


def _spec_to_payload(spec: ComparisonSpec) -> dict[str, Any]:
    return {
        "cell_names": list(spec.cell_names),
        "channel_key": spec.channel_key,
        "channel_label": spec.channel_label,
        "channel_unit": spec.channel_unit,
        "time_mode": spec.time_mode.value,
        "grid": {
            "domain": spec.grid.domain.value,
            "step": spec.grid.step,
            "normalized_points": spec.grid.normalized_points,
            "start": spec.grid.start,
            "end": spec.grid.end,
            "max_points": spec.grid.max_points,
        },
        "smoothing": {
            "sigma": spec.smoothing.sigma,
            "truncate": spec.smoothing.truncate,
        },
        "summary": {
            "center": spec.summary.center.value,
            "band": spec.summary.band.value,
        },
        "channel_bindings": [list(item) for item in spec.channel_bindings],
        "cell_aliases": [list(item) for item in spec.cell_aliases],
    }


def _spec_from_payload(value: Any) -> ComparisonSpec:
    data = _mapping(value, "result.spec")
    fields = {
        "cell_names",
        "channel_key",
        "channel_label",
        "channel_unit",
        "time_mode",
        "grid",
        "smoothing",
        "summary",
        "channel_bindings",
        "cell_aliases",
    }
    _exact_keys(data, fields, "result.spec")
    grid_data = _mapping(data["grid"], "result.spec.grid")
    _exact_keys(
        grid_data,
        {"domain", "step", "normalized_points", "start", "end", "max_points"},
        "result.spec.grid",
    )
    smoothing_data = _mapping(data["smoothing"], "result.spec.smoothing")
    _exact_keys(
        smoothing_data, {"sigma", "truncate"}, "result.spec.smoothing"
    )
    summary_data = _mapping(data["summary"], "result.spec.summary")
    _exact_keys(summary_data, {"center", "band"}, "result.spec.summary")

    bindings: list[tuple[str, str]] = []
    for index, raw in enumerate(
        _sequence(data["channel_bindings"], "result.spec.channel_bindings")
    ):
        item = _sequence(raw, f"result.spec.channel_bindings[{index}]")
        if len(item) != 2:
            raise ExpressionComparisonResultFormatError(
                f"result.spec.channel_bindings[{index}] must have two items"
            )
        bindings.append(
            (
                _string(item[0], f"result.spec.channel_bindings[{index}][0]"),
                _string(item[1], f"result.spec.channel_bindings[{index}][1]"),
            )
        )
    aliases: list[tuple[str, str, str]] = []
    for index, raw in enumerate(
        _sequence(data["cell_aliases"], "result.spec.cell_aliases")
    ):
        item = _sequence(raw, f"result.spec.cell_aliases[{index}]")
        if len(item) != 3:
            raise ExpressionComparisonResultFormatError(
                f"result.spec.cell_aliases[{index}] must have three items"
            )
        aliases.append(
            tuple(
                _string(child, f"result.spec.cell_aliases[{index}][{child_index}]")
                for child_index, child in enumerate(item)
            )  # type: ignore[arg-type]
        )

    return ComparisonSpec(
        cell_names=tuple(
            _string(item, f"result.spec.cell_names[{index}]")
            for index, item in enumerate(
                _sequence(data["cell_names"], "result.spec.cell_names")
            )
        ),
        channel_key=_string(data["channel_key"], "result.spec.channel_key"),
        channel_label=_string(data["channel_label"], "result.spec.channel_label"),
        channel_unit=_string(data["channel_unit"], "result.spec.channel_unit"),
        time_mode=_enum_value(TimeAxisMode, data["time_mode"], "result.spec.time_mode"),
        grid=GridSpec(
            domain=_enum_value(
                GridDomain, grid_data["domain"], "result.spec.grid.domain"
            ),
            step=(
                None
                if grid_data["step"] is None
                else _number(grid_data["step"], "result.spec.grid.step")
            ),
            normalized_points=_integer(
                grid_data["normalized_points"],
                "result.spec.grid.normalized_points",
            ),
            start=(
                None
                if grid_data["start"] is None
                else _number(grid_data["start"], "result.spec.grid.start")
            ),
            end=(
                None
                if grid_data["end"] is None
                else _number(grid_data["end"], "result.spec.grid.end")
            ),
            max_points=_integer(
                grid_data["max_points"], "result.spec.grid.max_points"
            ),
        ),
        smoothing=SmoothingSpec(
            sigma=_number(smoothing_data["sigma"], "result.spec.smoothing.sigma"),
            truncate=_number(
                smoothing_data["truncate"], "result.spec.smoothing.truncate"
            ),
        ),
        summary=SummarySpec(
            center=summary_data["center"],
            band=summary_data["band"],
        ),
        channel_bindings=tuple(bindings),
        cell_aliases=tuple(aliases),
    )


def _normalise_spec(spec: ComparisonSpec) -> ComparisonSpec:
    if not isinstance(spec, ComparisonSpec):
        raise TypeError("spec must be a ComparisonSpec")
    cells = tuple(spec.cell_names)
    if not cells or any(not isinstance(cell, str) or not cell.strip() for cell in cells):
        raise ValueError("spec.cell_names must contain non-blank strings")
    if len(set(cells)) != len(cells):
        raise ValueError("spec.cell_names must be unique")
    for name in ("channel_key", "channel_label", "channel_unit"):
        value = getattr(spec, name)
        if not isinstance(value, str):
            raise TypeError(f"spec.{name} must be a string")
    if not spec.channel_key.strip():
        raise ValueError("spec.channel_key cannot be blank")
    if not spec.channel_label.strip():
        raise ValueError("spec.channel_label cannot be blank")
    if not isinstance(spec.grid, GridSpec):
        raise TypeError("spec.grid must be a GridSpec")
    if not isinstance(spec.smoothing, SmoothingSpec):
        raise TypeError("spec.smoothing must be a SmoothingSpec")
    if not isinstance(spec.summary, SummarySpec):
        raise TypeError("spec.summary must be a SummarySpec")

    bindings = tuple(spec.channel_bindings)
    binding_ids: set[str] = set()
    for item in bindings:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError("each channel binding must contain dataset and channel")
        dataset_id, channel = item
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("channel binding dataset ids cannot be blank")
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("channel binding channel keys cannot be blank")
        if dataset_id in binding_ids:
            raise ValueError(f"duplicate channel binding for {dataset_id!r}")
        binding_ids.add(dataset_id)

    aliases = tuple(spec.cell_aliases)
    alias_keys: set[tuple[str, str]] = set()
    for item in aliases:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 3:
            raise ValueError("each cell alias must contain dataset, cell, and source cell")
        dataset_id, cell, source = item
        if any(not isinstance(value, str) or not value.strip() for value in item):
            raise ValueError("cell alias values cannot be blank")
        key = (dataset_id, cell)
        if key in alias_keys:
            raise ValueError(f"duplicate cell alias for {key!r}")
        if cell not in cells:
            raise ValueError(f"cell alias references unrequested cell {cell!r}")
        alias_keys.add(key)

    return ComparisonSpec(
        cell_names=cells,
        channel_key=spec.channel_key,
        channel_label=spec.channel_label,
        channel_unit=spec.channel_unit,
        time_mode=_enum_value(TimeAxisMode, spec.time_mode, "spec.time_mode"),
        grid=spec.grid,
        smoothing=spec.smoothing,
        summary=spec.summary,
        channel_bindings=tuple((item[0], item[1]) for item in bindings),
        cell_aliases=tuple((item[0], item[1], item[2]) for item in aliases),
    )


def _normalise_measurement_caches(
    values: Iterable[FrozenDatasetMeasurementCache],
) -> tuple[FrozenDatasetMeasurementCache, ...]:
    caches = tuple(values)
    if len(caches) > _MAX_DATASETS:
        raise ValueError(f"at most {_MAX_DATASETS} measurement caches are supported")
    if any(not isinstance(item, FrozenDatasetMeasurementCache) for item in caches):
        raise TypeError(
            "measurement_caches must contain FrozenDatasetMeasurementCache objects"
        )
    ids = tuple(item.dataset_id for item in caches)
    if len(set(ids)) != len(ids):
        raise ValueError("measurement cache dataset ids must be unique")
    return caches


def _normalise_cache_overrides(
    value: Mapping[str, Mapping[str, object]] | None,
    known_dataset_ids: set[str],
) -> dict[str, dict[str, object]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("overrides must be a mapping")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError("override dataset ids must be non-blank strings")
    unknown = set(value) - known_dataset_ids
    if unknown:
        raise ValueError(f"overrides reference unknown cache datasets: {sorted(unknown)!r}")
    output: dict[str, dict[str, object]] = {}
    allowed = {"label", "group_id", "series_label", "color"}
    for dataset_id, raw in value.items():
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("override dataset ids cannot be blank")
        if not isinstance(raw, Mapping):
            raise TypeError(f"overrides[{dataset_id!r}] must be a mapping")
        if any(not isinstance(key, str) for key in raw):
            raise TypeError(f"overrides[{dataset_id!r}] field names must be strings")
        extra = set(raw) - allowed
        if extra:
            raise ValueError(
                f"overrides[{dataset_id!r}] has unsupported fields: {sorted(extra)!r}"
            )
        output[dataset_id] = dict(raw)
    return output


def _validate_dataset_id_selection(
    selected: tuple[str, ...],
    known: set[str],
    label: str,
) -> None:
    if not selected:
        raise ValueError(f"{label} cannot be empty")
    if any(not isinstance(item, str) or not item.strip() for item in selected):
        raise ValueError(f"{label} must contain non-blank strings")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{label} must be unique")
    unknown = set(selected) - known
    if unknown:
        raise ValueError(f"{label} contains unknown datasets: {sorted(unknown)!r}")


def _spec_for_cache_request(
    base: ComparisonSpec,
    datasets: tuple[ExpressionDataset, ...],
    *,
    cell_name: str,
    image_channel: int,
) -> ComparisonSpec:
    base = _normalise_spec(base)
    if not isinstance(cell_name, str) or not cell_name.strip():
        raise ValueError("cell_name cannot be blank")
    if type(image_channel) is not int or image_channel < 0:
        raise ValueError("image_channel must be a non-negative integer")
    traces = tuple(trace for dataset in datasets for trace in dataset.traces)
    if traces:
        channel_key = traces[0].channel_key
        channel_label = traces[0].channel_label or f"Channel {image_channel + 1}"
        channel_unit = traces[0].channel_unit
        if any(trace.channel_key != channel_key for trace in traces):
            raise ValueError("materialized caches returned inconsistent channel keys")
        if any(trace.channel_unit != channel_unit for trace in traces):
            raise ValueError("materialized caches returned inconsistent channel units")
    else:
        channel_key = f"measured_channel_{image_channel + 1}"
        channel_label = f"Channel {image_channel + 1}"
        channel_unit = "scaled mean intensity"
    return replace(
        base,
        cell_names=(cell_name,),
        channel_key=channel_key,
        channel_label=channel_label,
        channel_unit=channel_unit,
        channel_bindings=(),
        cell_aliases=(),
    )


def _default_cache_request(
    result: ExpressionComparisonResult,
) -> tuple[str, int, str]:
    metadata = result.acquisition_metadata
    raw_cell = metadata.get("cell_name")
    cell_name = (
        raw_cell
        if isinstance(raw_cell, str) and raw_cell.strip()
        else result.spec.cell_names[0]
    )
    raw_channel = metadata.get("image_channel_one_based")
    if type(raw_channel) is int and raw_channel >= 1:
        image_channel = raw_channel - 1
    else:
        match = re.fullmatch(r"measured_channel_(\d+)", result.spec.channel_key)
        image_channel = max(0, int(match.group(1)) - 1) if match else 0
    raw_correction = metadata.get("correction_method")
    correction = (
        raw_correction
        if isinstance(raw_correction, str) and raw_correction.strip()
        else "global"
    )
    return cell_name, image_channel, correction


def _validate_cache_default_views(result: ExpressionComparisonResult) -> None:
    """Require the compact default view to agree with its cache authority.

    Schema v2 keeps one materialized request so legacy comparison code can
    open a useful default immediately, but the correction-neutral cache is the
    reusable numeric authority.  Rejecting divergence prevents different APIs
    from plotting different values from one checksum-valid file.
    """

    if not result.measurement_caches:
        return
    cell_name, image_channel, correction_method = _default_cache_request(result)
    datasets_by_id = {
        dataset.provenance.dataset_id: dataset for dataset in result.datasets
    }
    for cache in result.measurement_caches:
        actual = datasets_by_id[cache.dataset_id]
        expected = cache.materialize_dataset(
            cell_name,
            image_channel,
            correction_method,
        )
        if _cache_default_view_signature(actual) != _cache_default_view_signature(
            expected
        ):
            raise ValueError(
                "measurement cache does not match its materialized default view "
                f"for dataset {cache.dataset_id!r}"
            )


def _cache_default_view_signature(dataset: ExpressionDataset) -> tuple[Any, ...]:
    """Return acquisition/numeric fields, excluding presentation overrides."""

    provenance = dataset.provenance
    traces = tuple(
        (
            trace.cell_name,
            trace.channel_key,
            trace.channel_label,
            trace.channel_unit,
            trace.absolute_times,
            trace.values,
            trace.birth_time,
            trace.end_time,
            trace.missing_reasons,
        )
        for trace in dataset.traces
    )
    statuses = tuple(
        (
            status.cell_name,
            status.channel_key,
            status.availability,
            status.message,
        )
        for status in dataset.acquisition_statuses
    )
    return (
        provenance.dataset_id,
        provenance.source_uri,
        provenance.source_fingerprint,
        provenance.source_revision,
        provenance.metadata,
        traces,
        statuses,
    )


def _presentation_overrides(
    datasets: tuple[ExpressionDataset, ...],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        source = dataset.provenance
        values: dict[str, object] = {
            "label": source.label,
            "group_id": source.group_id,
        }
        color = next(
            (trace.color for trace in dataset.traces if trace.color is not None),
            None,
        )
        if color is not None:
            values["color"] = color
        output[source.dataset_id] = values
    return output


def _validate_materialised_inputs(
    datasets: tuple[ExpressionDataset, ...], spec: ComparisonSpec
) -> None:
    dataset_ids: set[str] = set()
    for dataset in datasets:
        source = dataset.provenance
        if source.dataset_id in dataset_ids:
            raise ValueError(f"duplicate dataset_id {source.dataset_id!r}")
        dataset_ids.add(source.dataset_id)
        if source.source_revision is not None and (
            type(source.source_revision) is not int or source.source_revision < 0
        ):
            raise ValueError("dataset source_revision must be a non-negative integer")

        trace_keys: set[tuple[str, str]] = set()
        for trace in dataset.traces:
            key = (trace.cell_name, trace.channel_key)
            if key in trace_keys:
                raise ValueError(
                    f"dataset {source.dataset_id!r} has duplicate trace key {key!r}"
                )
            trace_keys.add(key)
            if not isinstance(trace.channel_label, str) or not isinstance(
                trace.channel_unit, str
            ):
                raise TypeError("trace channel label/unit must be strings")
            if trace.series_label is not None and not isinstance(trace.series_label, str):
                raise TypeError("trace series_label must be a string or None")
            if trace.color is not None and (
                not isinstance(trace.color, str) or not trace.color.strip()
            ):
                raise ValueError("trace color must be a non-blank string or None")
            for value, reason in zip(trace.values, trace.missing_reasons):
                if value is None and (reason is None or not reason.strip()):
                    raise ValueError("every missing trace value requires a reason")
                if value is not None and reason is not None:
                    raise ValueError("finite trace values cannot have missing reasons")

    bindings = dict(spec.channel_bindings)
    aliases = {
        (dataset_id, cell): source
        for dataset_id, cell, source in spec.cell_aliases
    }
    unknown = (set(bindings) | {key[0] for key in aliases}) - dataset_ids
    if unknown:
        raise ValueError(f"spec mappings reference unknown datasets: {sorted(unknown)}")

    for dataset in datasets:
        dataset_id = dataset.provenance.dataset_id
        local_channel = bindings.get(dataset_id, spec.channel_key)
        expected_trace_keys: set[tuple[str, str]] = set()
        expected_status_keys = {
            (cell, spec.channel_key) for cell in spec.cell_names
        }
        actual_status_keys = {
            (status.cell_name, status.channel_key)
            for status in dataset.acquisition_statuses
        }
        extra_statuses = actual_status_keys - expected_status_keys
        if extra_statuses:
            raise ValueError(
                f"dataset {dataset_id!r} has statuses outside the request: "
                f"{sorted(extra_statuses)!r}"
            )
        for cell in spec.cell_names:
            source_cell = aliases.get((dataset_id, cell), cell)
            trace_key = (source_cell, local_channel)
            expected_trace_keys.add(trace_key)
            candidates = [
                trace
                for trace in dataset.traces
                if (trace.cell_name, trace.channel_key) == trace_key
            ]
            status = next(
                (
                    item
                    for item in dataset.acquisition_statuses
                    if (item.cell_name, item.channel_key)
                    == (cell, spec.channel_key)
                ),
                None,
            )
            if status is not None and candidates:
                raise ValueError(
                    f"dataset {dataset_id!r}, cell {cell!r} has both a trace and status"
                )
            if status is None and len(candidates) != 1:
                raise ValueError(
                    f"dataset {dataset_id!r}, cell {cell!r} requires exactly one "
                    "materialised trace or one explicit acquisition status"
                )
            if candidates and candidates[0].channel_unit != spec.channel_unit:
                raise ValueError(
                    f"dataset {dataset_id!r}, cell {cell!r} channel unit does not "
                    "match spec.channel_unit"
                )
        actual_trace_keys = {
            (trace.cell_name, trace.channel_key) for trace in dataset.traces
        }
        extras = actual_trace_keys - expected_trace_keys
        if extras:
            raise ValueError(
                f"dataset {dataset_id!r} has traces outside the request: {sorted(extras)!r}"
            )


def _validate_appearance_dataset_inclusion(
    appearance: Mapping[str, Any],
    datasets: tuple[ExpressionDataset, ...],
    *,
    additional_dataset_ids: set[str] | None = None,
) -> None:
    raw = appearance.get(APPEARANCE_INCLUDED_DATASET_IDS)
    if raw is None:
        return
    if not isinstance(raw, tuple):
        # Appearance has already been recursively frozen, so valid arrays are
        # tuples here. Treat strings/mappings/scalars as format errors.
        raise TypeError(
            f"appearance.{APPEARANCE_INCLUDED_DATASET_IDS} must be an array"
        )
    if any(not isinstance(item, str) or not item.strip() for item in raw):
        raise ValueError(
            f"appearance.{APPEARANCE_INCLUDED_DATASET_IDS} must contain "
            "non-blank dataset ids"
        )
    if len(set(raw)) != len(raw):
        raise ValueError(
            f"appearance.{APPEARANCE_INCLUDED_DATASET_IDS} must be unique"
        )
    known = {dataset.provenance.dataset_id for dataset in datasets}
    known.update(additional_dataset_ids or ())
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"appearance.{APPEARANCE_INCLUDED_DATASET_IDS} references unknown "
            f"datasets: {sorted(unknown)!r}"
        )


def _validate_presentation_only_dataset_revision(
    previous: tuple[ExpressionDataset, ...],
    revised: tuple[ExpressionDataset, ...],
) -> None:
    """Permit label/group/color changes while protecting captured native data."""

    if len(previous) != len(revised):
        raise ValueError("a presentation revision cannot add or remove datasets")
    for before, after in zip(previous, revised):
        before_source = before.provenance
        after_source = after.provenance
        if before_source.dataset_id != after_source.dataset_id:
            raise ValueError("a presentation revision cannot reorder or replace datasets")
        if (
            before_source.source_uri != after_source.source_uri
            or before_source.source_fingerprint != after_source.source_fingerprint
            or before_source.source_revision != after_source.source_revision
            or before_source.metadata != after_source.metadata
        ):
            raise ValueError(
                f"dataset {before_source.dataset_id!r} source provenance cannot "
                "change in a presentation revision"
            )
        if len(before.traces) != len(after.traces):
            raise ValueError("a presentation revision cannot add or remove traces")
        for old_trace, new_trace in zip(before.traces, after.traces):
            old_native = (
                old_trace.cell_name,
                old_trace.channel_key,
                old_trace.channel_label,
                old_trace.channel_unit,
                old_trace.absolute_times,
                old_trace.values,
                old_trace.birth_time,
                old_trace.end_time,
                old_trace.missing_reasons,
            )
            new_native = (
                new_trace.cell_name,
                new_trace.channel_key,
                new_trace.channel_label,
                new_trace.channel_unit,
                new_trace.absolute_times,
                new_trace.values,
                new_trace.birth_time,
                new_trace.end_time,
                new_trace.missing_reasons,
            )
            if old_native != new_native:
                raise ValueError(
                    f"dataset {before_source.dataset_id!r} native trace content "
                    "cannot change in a presentation revision"
                )
        if before.acquisition_statuses != after.acquisition_statuses:
            raise ValueError(
                f"dataset {before_source.dataset_id!r} acquisition statuses cannot "
                "change in a presentation revision"
            )


def _validate_presentation_only_spec_revision(
    previous: ComparisonSpec,
    revised: ComparisonSpec,
) -> None:
    immutable_before = (
        previous.cell_names,
        previous.channel_key,
        previous.channel_unit,
        previous.channel_bindings,
        previous.cell_aliases,
    )
    immutable_after = (
        revised.cell_names,
        revised.channel_key,
        revised.channel_unit,
        revised.channel_bindings,
        revised.cell_aliases,
    )
    if immutable_before != immutable_after:
        raise ValueError(
            "a presentation revision cannot change captured cell/channel identity "
            "or source mappings"
        )


def _payload_checksum(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_json_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = _freeze_json(value, label, depth=0, active=set())
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_json(
    value: Any,
    label: str,
    *,
    depth: int,
    active: set[int],
) -> Any:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{label} exceeds maximum JSON nesting depth")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{label} contains a cycle")
        active.add(identity)
        output: dict[str, Any] = {}
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{label} mapping keys must be strings")
                if key in output:
                    raise ValueError(f"{label} contains duplicate key {key!r}")
                output[key] = _freeze_json(
                    child,
                    f"{label}.{key}",
                    depth=depth + 1,
                    active=active,
                )
        finally:
            active.remove(identity)
        return MappingProxyType(output)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{label} contains a cycle")
        active.add(identity)
        try:
            return tuple(
                _freeze_json(
                    child,
                    f"{label}[{index}]",
                    depth=depth + 1,
                    active=active,
                )
                for index, child in enumerate(value)
            )
        finally:
            active.remove(identity)
    raise TypeError(
        f"{label} must contain only JSON-safe values, got {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpressionComparisonResultFormatError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ExpressionComparisonResultFormatError(f"{label} must be an array")
    return value


def _bounded_sequence(
    value: Any,
    label: str,
    *,
    maximum: int,
) -> Sequence[Any]:
    sequence = _sequence(value, label)
    if len(sequence) > maximum:
        raise ExpressionComparisonResultFormatError(
            f"{label} contains {len(sequence)} items; maximum is {maximum}"
        )
    return sequence


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise ExpressionComparisonResultFormatError(
            f"{label} is missing fields: {sorted(missing)!r}"
        )
    if unexpected:
        raise ExpressionComparisonResultFormatError(
            f"{label} has unexpected fields: {sorted(unexpected)!r}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ExpressionComparisonResultFormatError(f"{label} must be a string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ExpressionComparisonResultFormatError(f"{label} must be a boolean")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ExpressionComparisonResultFormatError(f"{label} must be an integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    parsed = _integer(value, label)
    if parsed < 0:
        raise ExpressionComparisonResultFormatError(
            f"{label} must be non-negative"
        )
    return parsed


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ExpressionComparisonResultFormatError(f"{label} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ExpressionComparisonResultFormatError(f"{label} must be finite")
    return converted


def _enum_value(enum_type, value: Any, label: str):
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ExpressionComparisonResultFormatError(f"{label} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ExpressionComparisonResultFormatError(
            f"{label} must be one of: {choices}"
        ) from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ExpressionComparisonResultFormatError(
                f"Duplicate JSON object key {key!r}"
            )
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ExpressionComparisonResultFormatError(
        f"Non-finite JSON number {value!r} is not allowed"
    )


def _uuid_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError(f"{label} must be a UUID string") from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(f"{label} must use canonical lowercase UUID spelling")
    return canonical


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _TIMESTAMP_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be an RFC 3339 UTC timestamp ending in Z")
    _parse_timestamp(value)
    return value


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"invalid UTC timestamp {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"timestamp {value!r} is not UTC")
    return parsed


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _with_result_suffix(path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != EXPRESSION_COMPARISON_RESULT_SUFFIX:
        destination = destination.with_suffix(EXPRESSION_COMPARISON_RESULT_SUFFIX)
    return destination


def _replacement_mode(destination: Path) -> int:
    try:
        return stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        previous = os.umask(0)
        os.umask(previous)
        return 0o666 & ~previous


__all__ = [
    "APPEARANCE_INCLUDED_DATASET_IDS",
    "EXPRESSION_COMPARISON_CALCULATION_VERSION",
    "EXPRESSION_COMPARISON_RESULT_SCHEMA",
    "EXPRESSION_COMPARISON_RESULT_LEGACY_VERSION",
    "EXPRESSION_COMPARISON_RESULT_SUFFIX",
    "EXPRESSION_COMPARISON_RESULT_VERSION",
    "ExpressionComparisonResult",
    "ExpressionComparisonResultFormatError",
    "ExpressionComparisonSourceMode",
    "SavedExpressionComparisonResult",
    "build_expression_comparison_data",
    "build_expression_comparison_cache_data",
    "capture_expression_comparison_result",
    "capture_expression_comparison_measurement_caches",
    "load_expression_comparison_result",
    "materialize_expression_measurement_caches",
    "revise_expression_comparison_result",
    "revise_expression_comparison_measurement_caches",
    "save_expression_comparison_result",
]
