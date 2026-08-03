"""Tests for renderer-independent expression plot preparation."""

from __future__ import annotations

import csv
import io
import math

import pytest

from acetree_py.analysis.expression_plot import (
    DEFAULT_EXPRESSION_CHANNELS,
    ExpressionChannel,
    ExpressionPlotService,
    ExpressionSeriesStyle,
    TimeAxisMode,
    export_expression_plot_csv,
    mapped_expression_channel,
)
from acetree_py.core.cell import Cell
from acetree_py.core.nucleus import Nucleus


def _cell(
    name: str,
    start: int,
    values: list[float | None],
    *,
    hash_key: str | None = None,
) -> Cell:
    cell = Cell(
        name=name,
        start_time=start,
        end_time=start + len(values) - 1,
        hash_key=hash_key,
    )
    for offset, value in enumerate(values):
        if value is None:
            continue
        cell.add_nucleus(
            start + offset,
            Nucleus(index=offset + 1, status=1, rweight=int(value)),
        )
    return cell


def test_builds_multiple_cells_on_absolute_axis_in_selection_order() -> None:
    service = ExpressionPlotService()
    aba = _cell("ABa", 3, [10, 20, 30])
    abp = _cell("ABp", 5, [40, 50])

    data = service.build([abp, aba], "rweight")

    assert data.time_mode is TimeAxisMode.ABSOLUTE
    assert data.x_label == "Timepoint"
    assert data.y_label == "AT expression"
    assert [series.cell_name for series in data.series] == ["ABp", "ABa"]
    assert data.series[0].x_values == (5.0, 6.0)
    assert data.series[0].y_values == (40.0, 50.0)
    assert data.has_data


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (TimeAxisMode.ABSOLUTE, (4.0, 5.0, 6.0)),
        (TimeAxisMode.RELATIVE, (0.0, 1.0, 2.0)),
        (TimeAxisMode.NORMALIZED, (0.0, 0.5, 1.0)),
    ],
)
def test_time_axis_modes(mode: TimeAxisMode, expected: tuple[float, ...]) -> None:
    data = ExpressionPlotService().build([_cell("A", 4, [1, 2, 3])], "rweight", mode)
    assert data.series[0].x_values == expected


def test_normalization_is_per_cell_lifetime() -> None:
    short = _cell("short", 1, [1, 2])
    long = _cell("long", 10, [1, 2, 3, 4, 5])

    data = ExpressionPlotService().build(
        [short, long], "rweight", TimeAxisMode.NORMALIZED
    )

    assert data.series[0].x_values == (0.0, 1.0)
    assert data.series[1].x_values == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_missing_nucleus_and_nonfinite_measurement_remain_plot_gaps() -> None:
    cell = _cell("A", 7, [10, None, 30])
    values = {("A", 7): 1.5, ("A", 9): math.inf}
    channel = mapped_expression_channel("channel_2", "Channel 2", values)

    series = ExpressionPlotService([channel]).build([cell], "channel_2").series[0]

    assert series.y_values == (1.5, None, None)
    assert math.isnan(series.plot_xy[1][1])
    assert math.isnan(series.plot_xy[1][2])
    assert series.finite_segments() == (((7.0,), (1.5,)),)


def test_missing_point_splits_finite_segments() -> None:
    series = ExpressionPlotService().build(
        [_cell("A", 1, [1, 2, None, 4, 5])], "rweight"
    ).series[0]
    assert series.finite_segments() == (
        ((1.0, 2.0), (1.0, 2.0)),
        ((4.0, 5.0), (4.0, 5.0)),
    )


def test_zero_duration_cell_normalizes_to_birth_without_dividing_by_zero() -> None:
    cell = _cell("single", 12, [77])
    series = ExpressionPlotService().build(
        [cell], "rweight", TimeAxisMode.NORMALIZED
    ).series[0]
    assert series.x_values == (0.0,)
    assert series.y_values == (77.0,)


def test_empty_zero_duration_cell_is_retained_as_missing() -> None:
    cell = Cell(name="empty", start_time=8, end_time=8)
    series = ExpressionPlotService().build([cell], "rweight", "relative").series[0]
    assert series.x_values == (0.0,)
    assert series.y_values == (None,)
    assert not series.has_data


def test_inverted_lifetime_uses_observed_points_without_crashing() -> None:
    cell = Cell(name="repairing", start_time=9, end_time=4)
    cell.add_nucleus(6, Nucleus(index=1, status=1, rweight=11))
    cell.add_nucleus(7, Nucleus(index=1, status=1, rweight=12))

    series = ExpressionPlotService().build([cell], "rweight", "absolute").series[0]

    assert series.absolute_timepoints == (6, 7)
    assert series.y_values == (11.0, 12.0)


def test_mapped_channel_prefers_stable_cell_key_over_display_name() -> None:
    cell = _cell("renamed", 2, [1], hash_key="stable-42")
    channel = mapped_expression_channel(
        "channel_3",
        "Channel 3",
        {("stable-42", 2): 99, ("renamed", 2): 50},
        unit="AU",
    )

    data = ExpressionPlotService([channel]).build([cell], "channel_3")

    assert data.series[0].y_values == (99.0,)
    assert data.y_label == "Channel 3 (AU)"


def test_styles_can_use_cell_key_and_are_carried_to_renderer() -> None:
    cell = _cell("A", 1, [10], hash_key="a-key")
    styles = {"a-key": ExpressionSeriesStyle(label="Anterior A", color="#123abc")}

    series = ExpressionPlotService().build(
        [cell], "rweight", styles=styles
    ).series[0]

    assert series.cell_key == "a-key"
    assert series.label == "Anterior A"
    assert series.color == "#123abc"


def test_export_csv_contains_transformed_and_absolute_time_and_missing_values() -> None:
    cell = _cell("A, left", 3, [10, None, 30])
    data = ExpressionPlotService().build(
        [cell],
        "rweight",
        "normalized",
        styles={"A, left": ExpressionSeriesStyle(color="#ff0000")},
    )
    output = io.StringIO()

    export_expression_plot_csv(data, output)

    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert len(rows) == 3
    assert rows[0] == {
        "series_label": "A, left",
        "cell_name": "A, left",
        "cell_key": "A, left",
        "channel": "rweight",
        "channel_label": "AT expression",
        "channel_unit": "",
        "time_mode": "normalized",
        "x": "0",
        "absolute_time": "3",
        "value": "10",
        "color": "#ff0000",
    }
    assert rows[1]["x"] == "0.5"
    assert rows[1]["absolute_time"] == "4"
    assert rows[1]["value"] == ""
    assert rows[2]["x"] == "1"
    assert rows[2]["value"] == "30"


def test_export_csv_accepts_path_and_creates_parent(tmp_path) -> None:
    destination = tmp_path / "exports" / "plot.csv"
    data = ExpressionPlotService().build([_cell("A", 1, [2])], "rweight")

    export_expression_plot_csv(data, destination)

    assert destination.exists()
    rows = list(
        csv.DictReader(io.StringIO(destination.read_text(encoding="utf-8")))
    )
    assert rows[0]["cell_name"] == "A"
    assert rows[0]["value"] == "2"


def test_service_exposes_stable_default_channel_order() -> None:
    service = ExpressionPlotService()
    assert service.channels == DEFAULT_EXPRESSION_CHANNELS
    assert [channel.key for channel in service.channels[:3]] == [
        "rweight",
        "rwraw",
        "weight",
    ]


def test_duplicate_and_unknown_channels_fail_with_context() -> None:
    channel = ExpressionChannel("same", "Same", lambda _cell, _time, _nucleus: 1)
    with pytest.raises(ValueError, match="Duplicate expression channel key"):
        ExpressionPlotService([channel, channel])

    with pytest.raises(KeyError, match="available channels: rweight"):
        ExpressionPlotService(DEFAULT_EXPRESSION_CHANNELS[:1]).build([], "nope")


def test_invalid_time_mode_and_non_numeric_values_fail_with_context() -> None:
    cell = _cell("A", 1, [1])
    service = ExpressionPlotService()
    with pytest.raises(ValueError, match="Unknown time-axis mode"):
        service.build([cell], "rweight", "embryonic")

    bad = ExpressionChannel("bad", "Bad", lambda _cell, _time, _nucleus: "bright")
    with pytest.raises(TypeError, match="cell 'A' at time 1"):
        ExpressionPlotService([bad]).build([cell], "bad")
