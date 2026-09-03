"""Renderer-neutral comparison of expression traces across datasets.

The ordinary expression plot service reads live :class:`~acetree_py.core.cell.Cell`
objects.  Cross-dataset work needs a stricter boundary: every dataset is first
materialised as immutable numeric traces, then alignment, smoothing, summary
statistics, rendering, and export all consume the same snapshots.

Datasets are the replicate unit.  Missing cells and channels remain explicit
status records, missing samples remain gaps, interpolation never extrapolates
or crosses an explicit gap, and smoothing is applied to each replicate before
pointwise statistics are calculated.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TextIO

import numpy as np

from .expression_plot import TimeAxisMode
from .expression_smoothing import gaussian_smooth_missing


class GridDomain(str, Enum):
    """How the common comparison grid spans replicate lifetimes."""

    UNION = "union"
    INTERSECTION = "intersection"


class CenterStatistic(str, Enum):
    """Central line calculated across dataset replicates."""

    NONE = "none"
    MEAN = "mean"
    MEDIAN = "median"


class BandStatistic(str, Enum):
    """Pointwise shaded band around (or accompanying) the central line."""

    NONE = "none"
    SAMPLE_SD = "sample_sd"
    SEM = "sem"
    STUDENT_T_95 = "student_t_95_ci"
    IQR = "iqr"
    SCALED_MAD = "scaled_mad"


class TraceAvailability(str, Enum):
    """Resolution state for one requested dataset/cell/channel trace."""

    AVAILABLE = "available"
    MISSING_CELL = "missing_cell"
    MISSING_CHANNEL = "missing_channel"
    AMBIGUOUS = "ambiguous"
    INCOMPLETE_DATA = "incomplete_data"


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Stable identity and provenance for one dataset replicate."""

    dataset_id: str
    label: str
    group_id: str = "all"
    source_uri: str = ""
    source_fingerprint: str = ""
    source_revision: int | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be blank")
        if not self.label.strip():
            raise ValueError("dataset label cannot be blank")
        if not self.group_id.strip():
            raise ValueError("group_id cannot be blank")
        metadata = tuple((str(key), str(value)) for key, value in self.metadata)
        if len({key for key, _value in metadata}) != len(metadata):
            raise ValueError("dataset metadata keys must be unique")
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class DatasetExpressionTrace:
    """Native expected samples for one cell and one local dataset channel.

    ``absolute_times`` must be strictly increasing and should include expected
    acquisition coordinates with ``None`` at explicitly missing samples.  That
    distinction lets alignment interpolate ordinary sampling intervals while
    refusing to bridge a known gap.
    """

    cell_name: str
    channel_key: str
    absolute_times: tuple[float, ...]
    values: tuple[float | None, ...]
    birth_time: float
    end_time: float
    channel_label: str = ""
    channel_unit: str = ""
    missing_reasons: tuple[str | None, ...] = ()
    series_label: str | None = None
    color: str | None = None

    def __post_init__(self) -> None:
        if not self.cell_name.strip():
            raise ValueError("trace cell_name cannot be blank")
        if not self.channel_key.strip():
            raise ValueError("trace channel_key cannot be blank")

        times = tuple(float(value) for value in self.absolute_times)
        values = tuple(_finite_optional(value, "trace value") for value in self.values)
        if not times:
            raise ValueError("a trace must contain at least one expected sample")
        if len(times) != len(values):
            raise ValueError("absolute_times and values must have equal length")
        if any(not math.isfinite(value) for value in times):
            raise ValueError("trace times must be finite")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("trace times must be strictly increasing")

        birth = float(self.birth_time)
        end = float(self.end_time)
        if not math.isfinite(birth) or not math.isfinite(end) or end < birth:
            raise ValueError("trace birth/end times must be finite and ordered")
        tolerance = 1e-12 * max(1.0, abs(birth), abs(end))
        if times[0] < birth - tolerance or times[-1] > end + tolerance:
            raise ValueError("expected sample times must lie within birth/end bounds")

        reasons = tuple(self.missing_reasons)
        if not reasons:
            reasons = tuple("missing" if value is None else None for value in values)
        if len(reasons) != len(values):
            raise ValueError("missing_reasons and values must have equal length")
        reasons = tuple(None if reason is None else str(reason) for reason in reasons)

        object.__setattr__(self, "absolute_times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "birth_time", birth)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "missing_reasons", reasons)


@dataclass(frozen=True, slots=True)
class DatasetAcquisitionStatus:
    """Resolved non-data outcome from acquiring one requested native trace.

    Repository extraction can establish that a selected replicate is missing,
    ambiguous, channel-incompatible, or incomplete without producing a numeric
    trace.  Keeping that outcome on the immutable dataset allows the comparison
    to retain the replicate in denominators and exact exports without pretending
    that an acquisition problem is a missing canonical cell.
    """

    cell_name: str
    channel_key: str
    availability: TraceAvailability
    message: str
    source_cell_name: str = ""
    source_channel_key: str = ""

    def __post_init__(self) -> None:
        if not self.cell_name.strip():
            raise ValueError("acquisition status cell_name cannot be blank")
        if not self.channel_key.strip():
            raise ValueError("acquisition status channel_key cannot be blank")
        availability = self.availability
        if not isinstance(availability, TraceAvailability):
            availability = TraceAvailability(availability)
            object.__setattr__(self, "availability", availability)
        if availability is TraceAvailability.AVAILABLE:
            raise ValueError("available acquisitions must be represented by a trace")
        if not self.message.strip():
            raise ValueError("acquisition status message cannot be blank")
        if not self.source_cell_name:
            object.__setattr__(self, "source_cell_name", self.cell_name)
        if not self.source_channel_key:
            object.__setattr__(self, "source_channel_key", self.channel_key)


@dataclass(frozen=True, slots=True)
class ExpressionDataset:
    """One dataset replicate and its materialised native traces."""

    provenance: DatasetProvenance
    traces: tuple[DatasetExpressionTrace, ...]
    acquisition_statuses: tuple[DatasetAcquisitionStatus, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", tuple(self.traces))
        statuses = tuple(self.acquisition_statuses)
        keys = [(status.cell_name, status.channel_key) for status in statuses]
        if len(set(keys)) != len(keys):
            raise ValueError("dataset acquisition status keys must be unique")
        object.__setattr__(self, "acquisition_statuses", statuses)


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Common-grid policy used independently for each requested cell."""

    domain: GridDomain = GridDomain.UNION
    step: float | None = None
    normalized_points: int = 101
    start: float | None = None
    end: float | None = None
    max_points: int = 1_000_000

    def __post_init__(self) -> None:
        domain = self.domain
        if not isinstance(domain, GridDomain):
            domain = GridDomain(domain)
            object.__setattr__(self, "domain", domain)
        if self.step is not None:
            step = float(self.step)
            if not math.isfinite(step) or step <= 0:
                raise ValueError("grid step must be finite and positive")
            object.__setattr__(self, "step", step)
        if self.normalized_points < 2:
            raise ValueError("normalized_points must be at least 2")
        if self.max_points < 1:
            raise ValueError("max_points must be positive")
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"grid {name} must be finite")
                object.__setattr__(self, name, value)
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("grid end cannot precede grid start")


@dataclass(frozen=True, slots=True)
class SmoothingSpec:
    """Per-replicate Gaussian smoothing in displayed-axis units."""

    sigma: float = 0.0
    truncate: float = 4.0

    def __post_init__(self) -> None:
        sigma = float(self.sigma)
        truncate = float(self.truncate)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("smoothing sigma must be finite and non-negative")
        if not math.isfinite(truncate) or truncate <= 0:
            raise ValueError("smoothing truncate must be finite and positive")
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "truncate", truncate)


@dataclass(frozen=True, slots=True)
class SummarySpec:
    """Pointwise replicate summary configuration."""

    center: CenterStatistic = CenterStatistic.MEAN
    band: BandStatistic = BandStatistic.SAMPLE_SD

    def __post_init__(self) -> None:
        center = self.center
        band = self.band
        if not isinstance(center, CenterStatistic):
            center = CenterStatistic(center)
            object.__setattr__(self, "center", center)
        if not isinstance(band, BandStatistic):
            band = BandStatistic(band)
            object.__setattr__(self, "band", band)
        if center is CenterStatistic.NONE and band is not BandStatistic.NONE:
            raise ValueError("a shaded band requires a mean or median center")
        mean_bands = {
            BandStatistic.NONE,
            BandStatistic.SAMPLE_SD,
            BandStatistic.SEM,
            BandStatistic.STUDENT_T_95,
        }
        median_bands = {
            BandStatistic.NONE,
            BandStatistic.IQR,
            BandStatistic.SCALED_MAD,
        }
        if center is CenterStatistic.MEAN and band not in mean_bands:
            raise ValueError("mean center supports SD, SEM, or Student-t bands")
        if center is CenterStatistic.MEDIAN and band not in median_bands:
            raise ValueError("median center supports IQR or scaled-MAD bands")


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    """Fully resolved, exportable comparison configuration."""

    cell_names: tuple[str, ...]
    channel_key: str
    channel_label: str
    channel_unit: str
    time_mode: TimeAxisMode
    grid: GridSpec
    smoothing: SmoothingSpec
    summary: SummarySpec
    channel_bindings: tuple[tuple[str, str], ...] = ()
    cell_aliases: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonTraceStatus:
    """Resolution result for one selected dataset and canonical cell."""

    dataset_id: str
    cell_name: str
    source_cell_name: str
    channel_key: str
    source_channel_key: str
    availability: TraceAvailability
    message: str = ""


@dataclass(frozen=True, slots=True)
class NativeComparisonTrace:
    """Resolved immutable native samples on the selected time axis."""

    dataset_id: str
    dataset_label: str
    group_id: str
    cell_name: str
    source_cell_name: str
    channel_key: str
    source_channel_key: str
    channel_label: str
    channel_unit: str
    birth_time: float
    end_time: float
    absolute_times: tuple[float, ...]
    x_values: tuple[float, ...]
    values: tuple[float | None, ...]
    missing_reasons: tuple[str | None, ...]
    series_label: str
    color: str | None


@dataclass(frozen=True, slots=True)
class AlignedComparisonTrace:
    """One dataset replicate aligned to its cell's common grid."""

    dataset_id: str
    dataset_label: str
    group_id: str
    cell_name: str
    source_cell_name: str
    channel_key: str
    source_channel_key: str
    channel_label: str
    channel_unit: str
    grid_x: tuple[float, ...]
    grid_step: float
    aligned_values: tuple[float | None, ...]
    display_values: tuple[float | None, ...]
    observed_mask: tuple[bool, ...]
    interpolated_mask: tuple[bool, ...]
    segment_ids: tuple[int | None, ...]
    series_label: str
    color: str | None


@dataclass(frozen=True, slots=True)
class ComparisonSummarySeries:
    """Pointwise aggregate for one dataset group and canonical cell."""

    group_id: str
    cell_name: str
    channel_key: str
    channel_label: str
    channel_unit: str
    grid_x: tuple[float, ...]
    grid_step: float
    center: tuple[float | None, ...]
    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]
    n_selected: int
    n_available: int
    n_valid: tuple[int, ...]
    center_statistic: CenterStatistic
    band_statistic: BandStatistic


@dataclass(frozen=True, slots=True)
class ExpressionComparisonData:
    """Complete immutable comparison input for renderers and exporters."""

    datasets: tuple[DatasetProvenance, ...]
    spec: ComparisonSpec
    statuses: tuple[ComparisonTraceStatus, ...]
    native_traces: tuple[NativeComparisonTrace, ...]
    aligned_traces: tuple[AlignedComparisonTrace, ...]
    summaries: tuple[ComparisonSummarySeries, ...]
    warnings: tuple[str, ...] = ()

    @property
    def has_data(self) -> bool:
        return any(any(value is not None for value in trace.display_values)
                   for trace in self.aligned_traces)


# The injected helper receives values with gaps, sigma in grid bins, and the
# truncation radius.  It must return the same number of optional values.
TraceSmoother = Callable[
    [tuple[float | None, ...], float, float],
    tuple[float | None, ...],
]


class ExpressionComparisonService:
    """Resolve, align, smooth, and aggregate expression dataset replicates."""

    def __init__(self, smoother: TraceSmoother | None = None) -> None:
        self._smoother = smoother or _shared_gaussian_smoother

    def build(
        self,
        datasets: Iterable[ExpressionDataset],
        *,
        cell_names: Iterable[str],
        channel_key: str,
        channel_label: str | None = None,
        channel_unit: str | None = None,
        channel_bindings: Mapping[str, str] | None = None,
        cell_aliases: Mapping[tuple[str, str], str] | None = None,
        time_mode: TimeAxisMode | str = TimeAxisMode.ABSOLUTE,
        grid: GridSpec | None = None,
        smoothing: SmoothingSpec | None = None,
        summary: SummarySpec | None = None,
    ) -> ExpressionComparisonData:
        """Build one reproducible comparison from immutable dataset snapshots."""

        datasets = tuple(datasets)
        if not datasets:
            raise ValueError("at least one expression dataset is required")
        dataset_ids = [dataset.provenance.dataset_id for dataset in datasets]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("dataset_id values must be unique within a comparison")

        cells = tuple(dict.fromkeys(str(name).strip() for name in cell_names))
        if not cells or any(not name for name in cells):
            raise ValueError("at least one non-blank cell name is required")
        if not channel_key.strip():
            raise ValueError("comparison channel_key cannot be blank")

        mode = _coerce_time_mode(time_mode)
        grid = grid or GridSpec()
        smoothing = smoothing or SmoothingSpec()
        summary = summary or SummarySpec()
        bindings = dict(channel_bindings or {})
        aliases = dict(cell_aliases or {})

        unknown_bindings = set(bindings) - set(dataset_ids)
        unknown_aliases = {dataset_id for dataset_id, _cell in aliases} - set(dataset_ids)
        if unknown_bindings or unknown_aliases:
            unknown = sorted(unknown_bindings | unknown_aliases)
            raise KeyError(f"comparison mappings reference unknown datasets: {unknown}")

        statuses: list[ComparisonTraceStatus] = []
        native: list[NativeComparisonTrace] = []
        warnings: list[str] = []

        for dataset in datasets:
            provenance = dataset.provenance
            local_channel = bindings.get(provenance.dataset_id, channel_key)
            for canonical_cell in cells:
                source_cell = aliases.get(
                    (provenance.dataset_id, canonical_cell), canonical_cell
                )
                acquisition = next(
                    (
                        status
                        for status in dataset.acquisition_statuses
                        if status.cell_name == canonical_cell
                        and status.channel_key == channel_key
                    ),
                    None,
                )
                if acquisition is not None:
                    statuses.append(
                        ComparisonTraceStatus(
                            dataset_id=provenance.dataset_id,
                            cell_name=canonical_cell,
                            source_cell_name=acquisition.source_cell_name,
                            channel_key=channel_key,
                            source_channel_key=acquisition.source_channel_key,
                            availability=acquisition.availability,
                            message=acquisition.message,
                        )
                    )
                    warnings.append(f"{provenance.label}: {acquisition.message}")
                    continue
                cell_candidates = [
                    trace for trace in dataset.traces if trace.cell_name == source_cell
                ]
                candidates = [
                    trace for trace in cell_candidates if trace.channel_key == local_channel
                ]
                if not cell_candidates:
                    availability = TraceAvailability.MISSING_CELL
                    message = f"Cell {source_cell!r} is absent"
                elif not candidates:
                    availability = TraceAvailability.MISSING_CHANNEL
                    message = (
                        f"Cell {source_cell!r} has no channel {local_channel!r}"
                    )
                elif len(candidates) > 1:
                    availability = TraceAvailability.AMBIGUOUS
                    message = (
                        f"Cell {source_cell!r} has {len(candidates)} traces for "
                        f"channel {local_channel!r}"
                    )
                else:
                    availability = TraceAvailability.AVAILABLE
                    message = ""

                statuses.append(
                    ComparisonTraceStatus(
                        dataset_id=provenance.dataset_id,
                        cell_name=canonical_cell,
                        source_cell_name=source_cell,
                        channel_key=channel_key,
                        source_channel_key=local_channel,
                        availability=availability,
                        message=message,
                    )
                )
                if availability is not TraceAvailability.AVAILABLE:
                    warnings.append(f"{provenance.label}: {message}")
                    continue

                trace = candidates[0]
                x_values = _transform_times(trace, mode)
                native.append(
                    NativeComparisonTrace(
                        dataset_id=provenance.dataset_id,
                        dataset_label=provenance.label,
                        group_id=provenance.group_id,
                        cell_name=canonical_cell,
                        source_cell_name=source_cell,
                        channel_key=channel_key,
                        source_channel_key=local_channel,
                        channel_label=trace.channel_label or local_channel,
                        channel_unit=trace.channel_unit,
                        birth_time=trace.birth_time,
                        end_time=trace.end_time,
                        absolute_times=trace.absolute_times,
                        x_values=x_values,
                        values=trace.values,
                        missing_reasons=trace.missing_reasons,
                        series_label=(
                            trace.series_label
                            or f"{provenance.label}: {canonical_cell}"
                        ),
                        color=trace.color,
                    )
                )

        resolved_units = {trace.channel_unit for trace in native}
        if channel_unit is not None:
            incompatible = {unit for unit in resolved_units if unit != channel_unit}
            if incompatible:
                raise ValueError(
                    f"channel unit mismatch: requested {channel_unit!r}, found "
                    f"{sorted(incompatible)!r}"
                )
            common_unit = channel_unit
        else:
            if len(resolved_units) > 1:
                raise ValueError(
                    "cannot aggregate traces with different channel units: "
                    f"{sorted(resolved_units)!r}"
                )
            common_unit = next(iter(resolved_units), "")

        common_label = channel_label or (
            native[0].channel_label if native else channel_key
        )
        # Normalize display metadata after compatibility validation.
        native = [
            replace(
                trace,
                channel_label=common_label,
                channel_unit=common_unit,
            )
            for trace in native
        ]

        aligned: list[AlignedComparisonTrace] = []
        for cell_name in cells:
            cell_traces = [trace for trace in native if trace.cell_name == cell_name]
            if not cell_traces:
                continue
            grid_x, grid_step = _comparison_grid(cell_traces, mode, grid, cell_name)
            for trace in cell_traces:
                aligned_values, observed, interpolated, segment_ids = _align_trace(
                    trace, grid_x
                )
                if smoothing.sigma > 0 and len(grid_x) > 1:
                    sigma_bins = smoothing.sigma / grid_step
                    display_values = _smooth_aligned_segments(
                        aligned_values,
                        segment_ids,
                        sigma_bins,
                        smoothing.truncate,
                        self._smoother,
                    )
                else:
                    display_values = aligned_values
                aligned.append(
                    AlignedComparisonTrace(
                        dataset_id=trace.dataset_id,
                        dataset_label=trace.dataset_label,
                        group_id=trace.group_id,
                        cell_name=trace.cell_name,
                        source_cell_name=trace.source_cell_name,
                        channel_key=trace.channel_key,
                        source_channel_key=trace.source_channel_key,
                        channel_label=common_label,
                        channel_unit=common_unit,
                        grid_x=grid_x,
                        grid_step=grid_step,
                        aligned_values=aligned_values,
                        display_values=display_values,
                        observed_mask=observed,
                        interpolated_mask=interpolated,
                        segment_ids=segment_ids,
                        series_label=trace.series_label,
                        color=trace.color,
                    )
                )

        group_order = tuple(
            dict.fromkeys(dataset.provenance.group_id for dataset in datasets)
        )
        summaries: list[ComparisonSummarySeries] = []
        for cell_name in cells:
            cell_aligned = [trace for trace in aligned if trace.cell_name == cell_name]
            if not cell_aligned:
                continue
            grid_x = cell_aligned[0].grid_x
            grid_step = cell_aligned[0].grid_step
            for group_id in group_order:
                group_datasets = [
                    dataset for dataset in datasets
                    if dataset.provenance.group_id == group_id
                ]
                group_traces = [
                    trace for trace in cell_aligned if trace.group_id == group_id
                ]
                centers, lowers, uppers, n_valid = _aggregate_traces(
                    group_traces, len(grid_x), summary
                )
                summaries.append(
                    ComparisonSummarySeries(
                        group_id=group_id,
                        cell_name=cell_name,
                        channel_key=channel_key,
                        channel_label=common_label,
                        channel_unit=common_unit,
                        grid_x=grid_x,
                        grid_step=grid_step,
                        center=centers,
                        lower=lowers,
                        upper=uppers,
                        n_selected=len(group_datasets),
                        n_available=len(group_traces),
                        n_valid=n_valid,
                        center_statistic=summary.center,
                        band_statistic=summary.band,
                    )
                )

        spec = ComparisonSpec(
            cell_names=cells,
            channel_key=channel_key,
            channel_label=common_label,
            channel_unit=common_unit,
            time_mode=mode,
            grid=grid,
            smoothing=smoothing,
            summary=summary,
            channel_bindings=tuple(
                (dataset_id, bindings[dataset_id])
                for dataset_id in dataset_ids
                if dataset_id in bindings
            ),
            cell_aliases=tuple(
                (dataset_id, cell, aliases[(dataset_id, cell)])
                for dataset_id in dataset_ids
                for cell in cells
                if (dataset_id, cell) in aliases
            ),
        )
        return ExpressionComparisonData(
            datasets=tuple(dataset.provenance for dataset in datasets),
            spec=spec,
            statuses=tuple(statuses),
            native_traces=tuple(native),
            aligned_traces=tuple(aligned),
            summaries=tuple(summaries),
            warnings=tuple(warnings),
        )


def export_expression_comparison_tidy_csv(
    data: ExpressionComparisonData,
    output: str | Path | TextIO,
) -> None:
    """Export native, aligned/display, summary, and status records exactly."""

    should_close = False
    if isinstance(output, (str, Path)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("w", newline="", encoding="utf-8")
        should_close = True
    else:
        stream = output

    fields = _tidy_fields()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    try:
        writer.writeheader()
        provenance = {item.dataset_id: item for item in data.datasets}
        common = _common_export_values(data)
        status_counts: dict[tuple[str, str], tuple[int, int]] = {}
        for status in data.statuses:
            group_id = provenance[status.dataset_id].group_id
            key = (group_id, status.cell_name)
            selected, available = status_counts.get(key, (0, 0))
            status_counts[key] = (
                selected + 1,
                available
                + int(status.availability is TraceAvailability.AVAILABLE),
            )

        for status in data.statuses:
            group_id = provenance[status.dataset_id].group_id
            n_selected, n_available = status_counts[(group_id, status.cell_name)]
            row = _blank_row(fields)
            row.update(common)
            row.update(_provenance_export_values(provenance[status.dataset_id]))
            row.update(
                record_type="trace_status",
                cell_name=status.cell_name,
                source_cell_name=status.source_cell_name,
                channel_key=status.channel_key,
                source_channel_key=status.source_channel_key,
                trace_status=status.availability.value,
                status_message=status.message,
                n_selected=n_selected,
                n_available=n_available,
            )
            writer.writerow(row)

        for trace in data.native_traces:
            source = provenance[trace.dataset_id]
            for index, (absolute_time, x_value, value, missing_reason) in enumerate(
                zip(
                    trace.absolute_times,
                    trace.x_values,
                    trace.values,
                    trace.missing_reasons,
                )
            ):
                row = _blank_row(fields)
                row.update(common)
                row.update(_provenance_export_values(source))
                row.update(
                    record_type="native_sample",
                    trace_status=TraceAvailability.AVAILABLE.value,
                    cell_name=trace.cell_name,
                    source_cell_name=trace.source_cell_name,
                    channel_key=trace.channel_key,
                    source_channel_key=trace.source_channel_key,
                    channel_label=trace.channel_label,
                    channel_unit=trace.channel_unit,
                    sample_index=index,
                    absolute_time=_format_float(absolute_time),
                    x=_format_float(x_value),
                    raw_value=_format_optional(value),
                    missing_reason=missing_reason or "",
                    birth_time=_format_float(trace.birth_time),
                    end_time=_format_float(trace.end_time),
                    series_label=trace.series_label,
                    color=trace.color or "",
                )
                writer.writerow(row)

        for trace in data.aligned_traces:
            source = provenance[trace.dataset_id]
            for index, (
                x_value,
                aligned_value,
                display_value,
                observed,
                interpolated,
                segment_id,
            ) in enumerate(
                zip(
                    trace.grid_x,
                    trace.aligned_values,
                    trace.display_values,
                    trace.observed_mask,
                    trace.interpolated_mask,
                    trace.segment_ids,
                )
            ):
                row = _blank_row(fields)
                row.update(common)
                row.update(_provenance_export_values(source))
                row.update(
                    record_type="aligned_sample",
                    trace_status=TraceAvailability.AVAILABLE.value,
                    cell_name=trace.cell_name,
                    source_cell_name=trace.source_cell_name,
                    channel_key=trace.channel_key,
                    source_channel_key=trace.source_channel_key,
                    channel_label=trace.channel_label,
                    channel_unit=trace.channel_unit,
                    sample_index=index,
                    x=_format_float(x_value),
                    aligned_value=_format_optional(aligned_value),
                    display_value=_format_optional(display_value),
                    is_observed=_format_bool(observed),
                    is_interpolated=_format_bool(interpolated),
                    segment_id="" if segment_id is None else segment_id,
                    missing_reason=(
                        "" if display_value is not None else "missing_or_outside_support"
                    ),
                    grid_step=_format_float(trace.grid_step),
                    series_label=trace.series_label,
                    color=trace.color or "",
                )
                writer.writerow(row)

        for summary in data.summaries:
            for index, (x_value, center, lower, upper, n_valid) in enumerate(
                zip(
                    summary.grid_x,
                    summary.center,
                    summary.lower,
                    summary.upper,
                    summary.n_valid,
                )
            ):
                row = _blank_row(fields)
                row.update(common)
                row.update(
                    record_type="summary_sample",
                    group_id=summary.group_id,
                    cell_name=summary.cell_name,
                    channel_key=summary.channel_key,
                    channel_label=summary.channel_label,
                    channel_unit=summary.channel_unit,
                    sample_index=index,
                    x=_format_float(x_value),
                    grid_step=_format_float(summary.grid_step),
                    center=_format_optional(center),
                    lower=_format_optional(lower),
                    upper=_format_optional(upper),
                    n_selected=summary.n_selected,
                    n_available=summary.n_available,
                    n_valid=n_valid,
                    center_statistic=summary.center_statistic.value,
                    band_statistic=summary.band_statistic.value,
                )
                writer.writerow(row)
    finally:
        if should_close:
            stream.close()


def _transform_times(
    trace: DatasetExpressionTrace,
    mode: TimeAxisMode,
) -> tuple[float, ...]:
    if mode is TimeAxisMode.ABSOLUTE:
        return trace.absolute_times
    if mode is TimeAxisMode.RELATIVE:
        return tuple(value - trace.birth_time for value in trace.absolute_times)
    duration = trace.end_time - trace.birth_time
    if duration <= 0:
        return tuple(0.0 for _value in trace.absolute_times)
    return tuple(
        (value - trace.birth_time) / duration for value in trace.absolute_times
    )


def _comparison_grid(
    traces: Sequence[NativeComparisonTrace],
    mode: TimeAxisMode,
    spec: GridSpec,
    cell_name: str,
) -> tuple[tuple[float, ...], float]:
    bounds = [(min(trace.x_values), max(trace.x_values)) for trace in traces]
    if mode is TimeAxisMode.NORMALIZED:
        lower, upper = 0.0, 1.0
    elif spec.domain is GridDomain.UNION:
        lower = min(item[0] for item in bounds)
        upper = max(item[1] for item in bounds)
    else:
        lower = max(item[0] for item in bounds)
        upper = min(item[1] for item in bounds)
    if spec.start is not None:
        lower = spec.start
    if spec.end is not None:
        upper = spec.end
    tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
    if upper < lower - tolerance:
        raise ValueError(
            f"No {spec.domain.value} time domain exists for cell {cell_name!r}"
        )
    if abs(upper - lower) <= tolerance:
        return (float(lower),), 1.0

    if mode is TimeAxisMode.NORMALIZED and spec.step is None:
        points = spec.normalized_points
        if points > spec.max_points:
            raise ValueError("normalized comparison grid exceeds max_points")
        values = tuple(float(value) for value in np.linspace(lower, upper, points))
        return values, (upper - lower) / (points - 1)

    requested_step = spec.step or _infer_grid_step(traces)
    intervals = int(math.ceil((upper - lower) / requested_step - 1e-12))
    count = intervals + 1
    if count < 1 or count > spec.max_points:
        raise ValueError(
            f"comparison grid for {cell_name!r} would contain {count} points"
        )
    # A common grid must include both support bounds; otherwise a terminal
    # observed sample disappears whenever the requested step is not an exact
    # divisor.  Linspace retains uniform spacing and treats the requested step
    # as the maximum cadence.
    values = tuple(float(value) for value in np.linspace(lower, upper, count))
    return values, (upper - lower) / intervals


def _infer_grid_step(traces: Sequence[NativeComparisonTrace]) -> float:
    cadences: list[float] = []
    for trace in traces:
        differences = np.diff(np.asarray(trace.x_values, dtype=float))
        positive = differences[differences > 0]
        if positive.size:
            cadences.append(float(np.median(positive)))
    return max(cadences, default=1.0)


def _align_trace(
    trace: NativeComparisonTrace,
    grid_x: tuple[float, ...],
) -> tuple[
    tuple[float | None, ...],
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[int | None, ...],
]:
    grid = np.asarray(grid_x, dtype=float)
    aligned = np.full(grid.shape, np.nan, dtype=float)
    observed = np.zeros(grid.shape, dtype=bool)
    segment_ids = np.full(grid.shape, -1, dtype=int)
    native_x = np.asarray(trace.x_values, dtype=float)
    finite = np.asarray([value is not None for value in trace.values], dtype=bool)
    native_y = np.asarray(
        [math.nan if value is None else value for value in trace.values], dtype=float
    )
    tolerance = 1e-9 * max(
        1.0,
        max((abs(value) for value in grid_x), default=1.0),
        max((abs(value) for value in trace.x_values), default=1.0),
    )

    for segment_id, (start, end) in enumerate(_true_runs(finite)):
        xs = native_x[start:end]
        ys = native_y[start:end]
        if len(xs) == 1:
            matches = np.flatnonzero(np.isclose(grid, xs[0], rtol=0.0, atol=tolerance))
            if matches.size:
                aligned[matches[0]] = ys[0]
                observed[matches[0]] = True
                segment_ids[matches[0]] = segment_id
            continue
        support = (grid >= xs[0] - tolerance) & (grid <= xs[-1] + tolerance)
        if support.any():
            aligned[support] = np.interp(grid[support], xs, ys)
            segment_ids[support] = segment_id
        for x_value, y_value in zip(xs, ys):
            matches = np.flatnonzero(
                np.isclose(grid, x_value, rtol=0.0, atol=tolerance)
            )
            if matches.size:
                aligned[matches[0]] = y_value
                observed[matches[0]] = True
                segment_ids[matches[0]] = segment_id

    present = np.isfinite(aligned)
    interpolated = present & ~observed
    return (
        tuple(None if not math.isfinite(value) else float(value) for value in aligned),
        tuple(bool(value) for value in observed),
        tuple(bool(value) for value in interpolated),
        tuple(None if value < 0 else int(value) for value in segment_ids),
    )


def _smooth_aligned_segments(
    values: tuple[float | None, ...],
    segment_ids: tuple[int | None, ...],
    sigma_bins: float,
    truncate: float,
    smoother: TraceSmoother,
) -> tuple[float | None, ...]:
    """Invoke a shared smoother separately for every native finite segment."""

    output: list[float | None] = [None] * len(values)
    start = 0
    while start < len(values):
        segment_id = segment_ids[start]
        if segment_id is None:
            start += 1
            continue
        end = start + 1
        while end < len(values) and segment_ids[end] == segment_id:
            end += 1
        segment = values[start:end]
        smoothed = _validate_smoothed_values(
            smoother(segment, sigma_bins, truncate), len(segment)
        )
        output[start:end] = smoothed
        start = end
    return tuple(output)


def _shared_gaussian_smoother(
    values: tuple[float | None, ...],
    sigma_bins: float,
    truncate: float,
) -> tuple[float | None, ...]:
    """Adapt the shared gap-preserving smoother to the injected-helper API.

    The shared implementation deliberately owns the Gaussian edge policy.  The
    comparison layer has already divided a trace by native segment identifier,
    so even a coarse display grid cannot let smoothing leak across an explicit
    missing native sample.  ``truncate`` is retained in the public injection
    boundary for alternate smoothers, but is not used by the shared helper.
    """

    return gaussian_smooth_missing(values, sigma_bins, truncate=truncate)


def _aggregate_traces(
    traces: Sequence[AlignedComparisonTrace],
    grid_length: int,
    spec: SummarySpec,
) -> tuple[
    tuple[float | None, ...],
    tuple[float | None, ...],
    tuple[float | None, ...],
    tuple[int, ...],
]:
    if traces:
        matrix = np.asarray(
            [
                [math.nan if value is None else value for value in trace.display_values]
                for trace in traces
            ],
            dtype=float,
        )
    else:
        matrix = np.empty((0, grid_length), dtype=float)

    centers: list[float | None] = []
    lowers: list[float | None] = []
    uppers: list[float | None] = []
    counts: list[int] = []
    for column in range(grid_length):
        values = matrix[:, column]
        values = values[np.isfinite(values)]
        count = int(values.size)
        counts.append(count)
        if count == 0 or spec.center is CenterStatistic.NONE:
            centers.append(None)
            lowers.append(None)
            uppers.append(None)
            continue

        if spec.center is CenterStatistic.MEAN:
            center = float(np.mean(values))
        else:
            center = float(np.median(values))
        centers.append(center)

        if spec.band is BandStatistic.NONE or count < 2:
            lowers.append(None)
            uppers.append(None)
            continue

        if spec.band is BandStatistic.SAMPLE_SD:
            half_width = float(np.std(values, ddof=1))
            lower, upper = center - half_width, center + half_width
        elif spec.band is BandStatistic.SEM:
            half_width = float(np.std(values, ddof=1) / math.sqrt(count))
            lower, upper = center - half_width, center + half_width
        elif spec.band is BandStatistic.STUDENT_T_95:
            from scipy.stats import t as student_t

            sem = float(np.std(values, ddof=1) / math.sqrt(count))
            half_width = float(student_t.ppf(0.975, count - 1) * sem)
            lower, upper = center - half_width, center + half_width
        elif spec.band is BandStatistic.IQR:
            lower, upper = (
                float(value)
                for value in np.quantile(values, (0.25, 0.75), method="linear")
            )
        else:
            median = float(np.median(values))
            scaled_mad = 1.4826 * float(np.median(np.abs(values - median)))
            lower, upper = center - scaled_mad, center + scaled_mad
        lowers.append(lower)
        uppers.append(upper)

    return tuple(centers), tuple(lowers), tuple(uppers), tuple(counts)


def _true_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    start: int | None = None
    for index, present in enumerate(mask):
        if present and start is None:
            start = index
        elif not present and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def _validate_smoothed_values(
    values: Sequence[float | None],
    expected_length: int,
) -> tuple[float | None, ...]:
    values = tuple(_finite_optional(value, "smoothed value") for value in values)
    if len(values) != expected_length:
        raise ValueError("trace smoother must preserve the aligned trace length")
    return values


def _finite_optional(value: Real | None, description: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real):
        raise TypeError(f"{description} must be numeric or None, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{description} must be finite or None, got {value!r}")
    return converted


def _coerce_time_mode(value: TimeAxisMode | str) -> TimeAxisMode:
    if isinstance(value, TimeAxisMode):
        return value
    try:
        return TimeAxisMode(value)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in TimeAxisMode)
        raise ValueError(f"unknown comparison time mode {value!r}: {choices}") from error


def _tidy_fields() -> list[str]:
    return [
        "schema_version",
        "record_type",
        "dataset_id",
        "dataset_label",
        "group_id",
        "source_uri",
        "source_fingerprint",
        "source_revision",
        "provenance_metadata",
        "trace_status",
        "status_message",
        "cell_name",
        "source_cell_name",
        "channel_key",
        "source_channel_key",
        "channel_label",
        "channel_unit",
        "time_mode",
        "grid_domain",
        "grid_step",
        "normalized_points",
        "smoothing_sigma",
        "smoothing_truncate",
        "sample_index",
        "absolute_time",
        "x",
        "birth_time",
        "end_time",
        "raw_value",
        "aligned_value",
        "display_value",
        "is_observed",
        "is_interpolated",
        "segment_id",
        "missing_reason",
        "center_statistic",
        "band_statistic",
        "center",
        "lower",
        "upper",
        "n_selected",
        "n_available",
        "n_valid",
        "series_label",
        "color",
    ]


def _blank_row(fields: Sequence[str]) -> dict[str, object]:
    return {field: "" for field in fields}


def _common_export_values(data: ExpressionComparisonData) -> dict[str, object]:
    return {
        "schema_version": "1",
        "channel_key": data.spec.channel_key,
        "channel_label": data.spec.channel_label,
        "channel_unit": data.spec.channel_unit,
        "time_mode": data.spec.time_mode.value,
        "grid_domain": data.spec.grid.domain.value,
        "grid_step": (
            "" if data.spec.grid.step is None else _format_float(data.spec.grid.step)
        ),
        "normalized_points": data.spec.grid.normalized_points,
        "smoothing_sigma": _format_float(data.spec.smoothing.sigma),
        "smoothing_truncate": _format_float(data.spec.smoothing.truncate),
        "center_statistic": data.spec.summary.center.value,
        "band_statistic": data.spec.summary.band.value,
    }


def _provenance_export_values(source: DatasetProvenance) -> dict[str, object]:
    return {
        "dataset_id": source.dataset_id,
        "dataset_label": source.label,
        "group_id": source.group_id,
        "source_uri": source.source_uri,
        "source_fingerprint": source.source_fingerprint,
        "source_revision": (
            "" if source.source_revision is None else source.source_revision
        ),
        "provenance_metadata": json.dumps(
            dict(source.metadata), sort_keys=True, separators=(",", ":")
        ),
    }


def _format_optional(value: float | None) -> str:
    return "" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "AlignedComparisonTrace",
    "BandStatistic",
    "CenterStatistic",
    "ComparisonSpec",
    "ComparisonSummarySeries",
    "ComparisonTraceStatus",
    "DatasetExpressionTrace",
    "DatasetAcquisitionStatus",
    "DatasetProvenance",
    "ExpressionComparisonData",
    "ExpressionComparisonService",
    "ExpressionDataset",
    "GridDomain",
    "GridSpec",
    "NativeComparisonTrace",
    "SmoothingSpec",
    "SummarySpec",
    "TimeAxisMode",
    "TraceAvailability",
    "TraceSmoother",
    "export_expression_comparison_tidy_csv",
]
