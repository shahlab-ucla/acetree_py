"""Framework-independent data preparation for expression plots.

The GUI owns windows, widgets, and rendering.  This module owns the parts that
must remain identical for every renderer and export path:

* resolving a named expression channel at each nucleus;
* placing several cell lifetimes on absolute, birth-relative, or normalized
  time axes;
* preserving missing samples as gaps; and
* exporting exactly the data sent to a plotting backend.

There is deliberately no Qt or matplotlib dependency here.  ``plot_xy``
returns ordinary tuples (with NaN at missing samples), which can be handed to
matplotlib and rendered to SVG without another data transformation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Callable, Iterable, Mapping, TextIO

from ..core.cell import Cell
from ..core.nucleus import Nucleus


class TimeAxisMode(str, Enum):
    """Supported alignments for a cell's expression time series."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    NORMALIZED = "normalized"


# The cell and time arguments let a channel read from an external measurement
# table as well as from fields stored directly on a Nucleus.
ChannelValueReader = Callable[[Cell, int, Nucleus], Real | None]


@dataclass(frozen=True)
class ExpressionChannel:
    """A selectable expression measurement and its display metadata."""

    key: str
    label: str
    reader: ChannelValueReader
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Expression channel key cannot be blank")
        if not self.label.strip():
            raise ValueError("Expression channel label cannot be blank")
        if not callable(self.reader):
            raise TypeError("Expression channel reader must be callable")


@dataclass(frozen=True)
class ExpressionSeriesStyle:
    """Renderer-neutral per-series choices shared by plots and exports."""

    label: str | None = None
    color: str | None = None


@dataclass(frozen=True)
class ExpressionPlotSeries:
    """One cell's values, already transformed onto the selected time axis."""

    cell_key: str
    cell_name: str
    label: str
    channel_key: str
    color: str | None
    start_time: int
    end_time: int
    absolute_timepoints: tuple[int, ...]
    x_values: tuple[float, ...]
    y_values: tuple[float | None, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.absolute_timepoints),
            len(self.x_values),
            len(self.y_values),
        }
        if len(lengths) != 1:
            raise ValueError("Expression plot series arrays must have equal length")

    @property
    def has_data(self) -> bool:
        """Whether at least one finite expression measurement is present."""

        return any(value is not None for value in self.y_values)

    @property
    def plot_xy(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return matplotlib-compatible data, using NaN to preserve gaps."""

        return self.x_values, tuple(
            math.nan if value is None else value for value in self.y_values
        )

    def finite_segments(self) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
        """Return contiguous finite runs for renderers without NaN-gap support."""

        segments: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
        xs: list[float] = []
        ys: list[float] = []
        for x_value, y_value in zip(self.x_values, self.y_values):
            if y_value is None:
                if xs:
                    segments.append((tuple(xs), tuple(ys)))
                    xs, ys = [], []
                continue
            xs.append(x_value)
            ys.append(y_value)
        if xs:
            segments.append((tuple(xs), tuple(ys)))
        return tuple(segments)


@dataclass(frozen=True)
class ExpressionPlotData:
    """Complete renderer-neutral input for one expression plot."""

    channel: ExpressionChannel
    time_mode: TimeAxisMode
    series: tuple[ExpressionPlotSeries, ...]

    @property
    def x_label(self) -> str:
        if self.time_mode is TimeAxisMode.RELATIVE:
            return "Time since birth (timepoints)"
        if self.time_mode is TimeAxisMode.NORMALIZED:
            return "Normalized lifetime"
        return "Timepoint"

    @property
    def y_label(self) -> str:
        if self.channel.unit:
            return f"{self.channel.label} ({self.channel.unit})"
        return self.channel.label

    @property
    def has_data(self) -> bool:
        return any(series.has_data for series in self.series)


class ExpressionPlotService:
    """Build plot/export data from cells and registered expression channels."""

    def __init__(self, channels: Iterable[ExpressionChannel] | None = None) -> None:
        selected = tuple(DEFAULT_EXPRESSION_CHANNELS if channels is None else channels)
        by_key: dict[str, ExpressionChannel] = {}
        for channel in selected:
            if channel.key in by_key:
                raise ValueError(f"Duplicate expression channel key: {channel.key!r}")
            by_key[channel.key] = channel
        self._channels = selected
        self._channels_by_key = by_key

    @property
    def channels(self) -> tuple[ExpressionChannel, ...]:
        """Channels in the stable order supplied for a channel selector."""

        return self._channels

    def channel(self, key: str) -> ExpressionChannel:
        """Resolve a channel key or raise an actionable error."""

        try:
            return self._channels_by_key[key]
        except KeyError as exc:
            available = ", ".join(self._channels_by_key) or "none"
            raise KeyError(
                f"Unknown expression channel {key!r}; available channels: {available}"
            ) from exc

    def build(
        self,
        cells: Iterable[Cell],
        channel_key: str,
        time_mode: TimeAxisMode | str = TimeAxisMode.ABSOLUTE,
        *,
        styles: Mapping[str, ExpressionSeriesStyle] | None = None,
    ) -> ExpressionPlotData:
        """Build series for every selected cell.

        Each cell is sampled over its inclusive ``start_time``/``end_time``
        lifetime.  A missing nucleus or measurement is retained as ``None`` so
        the plotted line has a visible gap and the CSV has an empty value.

        ``styles`` may be keyed by ``Cell.hash_key`` or cell name; the hash key
        takes precedence.  This supports independently colored entries without
        baking renderer-specific styling into the analysis layer.
        """

        channel = self.channel(channel_key)
        mode = _coerce_time_mode(time_mode)
        styles = styles or {}
        output: list[ExpressionPlotSeries] = []

        for cell in cells:
            start = int(cell.start_time)
            end = int(cell.end_time)
            absolute_times = _lifetime_timepoints(cell)
            cell_key = cell.hash_key or cell.name
            style = styles.get(cell_key) or styles.get(cell.name) or ExpressionSeriesStyle()

            x_values = tuple(_time_coordinate(time, start, end, mode) for time in absolute_times)
            y_values: list[float | None] = []
            for time in absolute_times:
                nucleus = cell.get_nucleus_at(time)
                if nucleus is None:
                    y_values.append(None)
                    continue
                raw_value = channel.reader(cell, time, nucleus)
                y_values.append(_finite_value_or_none(raw_value, channel, cell, time))

            output.append(
                ExpressionPlotSeries(
                    cell_key=cell_key,
                    cell_name=cell.name,
                    label=style.label or cell.name,
                    channel_key=channel.key,
                    color=style.color,
                    start_time=start,
                    end_time=end,
                    absolute_timepoints=absolute_times,
                    x_values=x_values,
                    y_values=tuple(y_values),
                )
            )

        return ExpressionPlotData(channel=channel, time_mode=mode, series=tuple(output))


def nucleus_attribute_channel(
    key: str,
    label: str,
    attribute: str,
    *,
    unit: str = "",
) -> ExpressionChannel:
    """Create a channel backed by a numeric :class:`Nucleus` attribute."""

    def read(_cell: Cell, _time: int, nucleus: Nucleus) -> Real | None:
        return getattr(nucleus, attribute, None)

    return ExpressionChannel(key=key, label=label, reader=read, unit=unit)


def mapped_expression_channel(
    key: str,
    label: str,
    values: Mapping[tuple[str, int], Real | None],
    *,
    unit: str = "",
) -> ExpressionChannel:
    """Create a channel from AT measurements indexed by ``(cell, time)``.

    The cell's stable ``hash_key`` is tried first and its display name second.
    This is suitable for the per-channel tables produced by the Measure tool
    and keeps those values out of the GUI layer.
    """

    def read(cell: Cell, time: int, _nucleus: Nucleus) -> Real | None:
        if cell.hash_key is not None:
            stable_key = (cell.hash_key, time)
            if stable_key in values:
                return values[stable_key]
        return values.get((cell.name, time))

    return ExpressionChannel(key=key, label=label, reader=read, unit=unit)


def export_expression_plot_csv(
    data: ExpressionPlotData,
    output: str | Path | TextIO,
) -> None:
    """Export the exact plotted samples in tidy/long-form CSV.

    Missing measurements are emitted as an empty ``value`` cell.  Both the
    transformed x-coordinate and original timepoint are included, allowing a
    normalized or relative plot to be reconstructed without losing lineage
    timing.  ``output`` may be a path or an already-open text stream.
    """

    should_close = False
    if isinstance(output, (str, Path)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("w", newline="", encoding="utf-8")
        should_close = True
    else:
        stream = output

    try:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "series_label",
                "cell_name",
                "cell_key",
                "channel",
                "channel_label",
                "channel_unit",
                "time_mode",
                "x",
                "absolute_time",
                "value",
                "color",
            ]
        )
        for series in data.series:
            for absolute_time, x_value, y_value in zip(
                series.absolute_timepoints,
                series.x_values,
                series.y_values,
            ):
                writer.writerow(
                    [
                        series.label,
                        series.cell_name,
                        series.cell_key,
                        data.channel.key,
                        data.channel.label,
                        data.channel.unit,
                        data.time_mode.value,
                        _format_number(x_value),
                        absolute_time,
                        "" if y_value is None else _format_number(y_value),
                        series.color or "",
                    ]
                )
    finally:
        if should_close:
            stream.close()


def _lifetime_timepoints(cell: Cell) -> tuple[int, ...]:
    """Return the declared inclusive lifetime, or observed times if invalid."""

    start = int(cell.start_time)
    end = int(cell.end_time)
    if end >= start:
        return tuple(range(start, end + 1))
    # A corrupt/reconstructed cell can temporarily have inverted bounds.  Do
    # not allocate an invalid range or discard measurements while the lineage
    # is being repaired.
    return tuple(sorted({int(time) for time, _nucleus in cell.nuclei}))


def _time_coordinate(
    time: int,
    start: int,
    end: int,
    mode: TimeAxisMode,
) -> float:
    if mode is TimeAxisMode.ABSOLUTE:
        return float(time)
    if mode is TimeAxisMode.RELATIVE:
        return float(time - start)
    duration = end - start
    if duration <= 0:
        return 0.0
    return (time - start) / duration


def _coerce_time_mode(value: TimeAxisMode | str) -> TimeAxisMode:
    if isinstance(value, TimeAxisMode):
        return value
    try:
        return TimeAxisMode(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in TimeAxisMode)
        raise ValueError(f"Unknown time-axis mode {value!r}; choose one of: {choices}") from exc


def _finite_value_or_none(
    value: Real | None,
    channel: ExpressionChannel,
    cell: Cell,
    time: int,
) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real):
        raise TypeError(
            f"Channel {channel.key!r} returned a non-numeric value for "
            f"cell {cell.name!r} at time {time}: {value!r}"
        )
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _format_number(value: float) -> str:
    """Compact, round-trip-safe output without gratuitous trailing zeros."""

    return format(value, ".15g")


DEFAULT_EXPRESSION_CHANNELS: tuple[ExpressionChannel, ...] = (
    nucleus_attribute_channel("rweight", "AT expression", "rweight"),
    nucleus_attribute_channel("rwraw", "Raw expression", "rwraw"),
    nucleus_attribute_channel("weight", "Primary fluorescence", "weight"),
    ExpressionChannel(
        key="red_global",
        label="Expression (global corrected)",
        reader=lambda _cell, _time, nucleus: nucleus.rwraw - nucleus.rwcorr1,
    ),
    ExpressionChannel(
        key="red_local",
        label="Expression (local corrected)",
        reader=lambda _cell, _time, nucleus: nucleus.rwraw - nucleus.rwcorr2,
    ),
    ExpressionChannel(
        key="red_blot",
        label="Expression (blot corrected)",
        reader=lambda _cell, _time, nucleus: nucleus.rwraw - nucleus.rwcorr3,
    ),
    ExpressionChannel(
        key="red_cross",
        label="Expression (cross-talk corrected)",
        reader=lambda _cell, _time, nucleus: nucleus.rwraw - nucleus.rwcorr4,
    ),
)


__all__ = [
    "DEFAULT_EXPRESSION_CHANNELS",
    "ExpressionChannel",
    "ExpressionPlotData",
    "ExpressionPlotSeries",
    "ExpressionPlotService",
    "ExpressionSeriesStyle",
    "TimeAxisMode",
    "export_expression_plot_csv",
    "mapped_expression_channel",
    "nucleus_attribute_channel",
]
