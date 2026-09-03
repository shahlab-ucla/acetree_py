"""Framework-independent cross-dataset expression comparison tests."""

from __future__ import annotations

import csv
import io
import math

import pytest

from acetree_py.analysis.expression_comparison import (
    BandStatistic,
    CenterStatistic,
    DatasetExpressionTrace,
    DatasetAcquisitionStatus,
    DatasetProvenance,
    ExpressionComparisonService,
    ExpressionDataset,
    GridDomain,
    GridSpec,
    SmoothingSpec,
    SummarySpec,
    TimeAxisMode,
    TraceAvailability,
    export_expression_comparison_tidy_csv,
)
from acetree_py.analysis.expression_smoothing import gaussian_smooth_missing


def _trace(
    values: list[float | None],
    *,
    times: list[float] | None = None,
    cell: str = "A",
    channel: str = "gfp",
    label: str = "GFP",
    unit: str = "AU",
    birth: float | None = None,
    end: float | None = None,
    reasons: list[str | None] | None = None,
) -> DatasetExpressionTrace:
    if times is None:
        times = [float(index) for index in range(len(values))]
    return DatasetExpressionTrace(
        cell_name=cell,
        channel_key=channel,
        channel_label=label,
        channel_unit=unit,
        absolute_times=tuple(times),
        values=tuple(values),
        birth_time=times[0] if birth is None else birth,
        end_time=times[-1] if end is None else end,
        missing_reasons=() if reasons is None else tuple(reasons),
    )


def _dataset(
    dataset_id: str,
    *traces: DatasetExpressionTrace,
    group: str = "all",
    label: str | None = None,
) -> ExpressionDataset:
    return ExpressionDataset(
        provenance=DatasetProvenance(
            dataset_id=dataset_id,
            label=label or dataset_id.upper(),
            group_id=group,
            source_uri=f"/data/{dataset_id}.xml",
            source_fingerprint=f"fingerprint-{dataset_id}",
            source_revision=3,
            metadata=(("strain", "N2"),),
        ),
        traces=tuple(traces),
    )


def _build(
    datasets: list[ExpressionDataset],
    **kwargs,
):
    options = {
        "cell_names": ["A"],
        "channel_key": "gfp",
        "grid": GridSpec(step=1.0),
        "summary": SummarySpec(
            center=CenterStatistic.MEAN,
            band=BandStatistic.NONE,
        ),
    }
    options.update(kwargs)
    return ExpressionComparisonService().build(datasets, **options)


def test_exact_cell_alias_and_per_dataset_channel_binding() -> None:
    first = _dataset("d1", _trace([1, 2], cell="ABa", channel="green"))
    second = _dataset("d2", _trace([3, 4], cell="AB_a", channel="ch2"))

    data = _build(
        [first, second],
        cell_names=["ABa"],
        channel_bindings={"d1": "green", "d2": "ch2"},
        cell_aliases={("d2", "ABa"): "AB_a"},
    )

    assert [trace.source_cell_name for trace in data.native_traces] == ["ABa", "AB_a"]
    assert [trace.source_channel_key for trace in data.native_traces] == ["green", "ch2"]
    assert all(status.availability is TraceAvailability.AVAILABLE for status in data.statuses)
    assert data.summaries[0].center == (2.0, 3.0)


def test_missing_and_ambiguous_traces_remain_explicit_statuses() -> None:
    available = _dataset("one", _trace([2]))
    missing = _dataset("two", _trace([4], cell="B"))
    ambiguous_trace = _trace([6])
    ambiguous = _dataset("three", ambiguous_trace, ambiguous_trace)

    data = _build([available, missing, ambiguous])

    assert [status.availability for status in data.statuses] == [
        TraceAvailability.AVAILABLE,
        TraceAvailability.MISSING_CELL,
        TraceAvailability.AMBIGUOUS,
    ]
    summary = data.summaries[0]
    assert summary.n_selected == 3
    assert summary.n_available == 1
    assert summary.n_valid == (1,)
    assert summary.center == (2.0,)


def test_acquisition_status_retains_selected_replicate_and_exact_reason() -> None:
    available = _dataset("one", _trace([2]), group="control")
    incomplete = ExpressionDataset(
        provenance=DatasetProvenance(
            dataset_id="two",
            label="two",
            group_id="control",
        ),
        traces=(),
        acquisition_statuses=(
            DatasetAcquisitionStatus(
                cell_name="A",
                channel_key="gfp",
                availability=TraceAvailability.INCOMPLETE_DATA,
                message="Saved expression is incomplete; recompute from images.",
            ),
        ),
    )

    data = _build([available, incomplete])

    assert data.statuses[1].availability is TraceAvailability.INCOMPLETE_DATA
    assert "recompute" in data.statuses[1].message
    assert data.summaries[0].n_selected == 2
    assert data.summaries[0].n_available == 1

    output = io.StringIO()
    export_expression_comparison_tidy_csv(data, output)
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    status = next(
        row for row in rows
        if row["record_type"] == "trace_status" and row["dataset_id"] == "two"
    )
    assert status["trace_status"] == "incomplete_data"
    assert "recompute" in status["status_message"]
    assert status["n_selected"] == "2"
    assert status["n_available"] == "1"


def test_absolute_relative_and_normalized_native_coordinates() -> None:
    first = _dataset(
        "d1",
        _trace([1, 2, 3], times=[10, 11, 12], birth=10, end=12),
    )
    second = _dataset(
        "d2",
        _trace([4, 5, 6], times=[20, 22, 24], birth=20, end=24),
    )

    absolute = _build([first], time_mode=TimeAxisMode.ABSOLUTE)
    relative = _build([first, second], time_mode=TimeAxisMode.RELATIVE)
    normalized = _build(
        [first, second],
        time_mode=TimeAxisMode.NORMALIZED,
        grid=GridSpec(normalized_points=3),
    )

    assert absolute.native_traces[0].x_values == (10.0, 11.0, 12.0)
    assert [trace.x_values for trace in relative.native_traces] == [
        (0.0, 1.0, 2.0),
        (0.0, 2.0, 4.0),
    ]
    assert all(trace.x_values == (0.0, 0.5, 1.0) for trace in normalized.native_traces)
    assert all(trace.grid_x == (0.0, 0.5, 1.0) for trace in normalized.aligned_traces)


def test_single_point_lifetime_normalizes_to_zero_without_extrapolation() -> None:
    data = _build(
        [_dataset("d1", _trace([7], times=[5], birth=5, end=5))],
        time_mode=TimeAxisMode.NORMALIZED,
        grid=GridSpec(normalized_points=3),
    )

    trace = data.aligned_traces[0]
    assert trace.grid_x == (0.0, 0.5, 1.0)
    assert trace.aligned_values == (7.0, None, None)
    assert data.summaries[0].n_valid == (1, 0, 0)


def test_union_and_intersection_grids_are_cell_local() -> None:
    first = _dataset("d1", _trace([0, 1, 2, 3], times=[0, 1, 2, 3]))
    second = _dataset("d2", _trace([2, 3, 4, 5], times=[2, 3, 4, 5]))

    union = _build([first, second], grid=GridSpec(GridDomain.UNION, step=1))
    intersection = _build(
        [first, second], grid=GridSpec(GridDomain.INTERSECTION, step=1)
    )

    assert union.aligned_traces[0].grid_x == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert intersection.aligned_traces[0].grid_x == (2.0, 3.0)
    assert union.aligned_traces[0].aligned_values[-2:] == (None, None)
    assert union.aligned_traces[1].aligned_values[:2] == (None, None)


def test_alignment_never_interpolates_across_none_gap() -> None:
    gapped = _dataset(
        "gapped",
        _trace(
            [1, None, 3],
            times=[0, 1, 2],
            reasons=[None, "stack unavailable", None],
        ),
    )
    support = _dataset("support", _trace([0, 1, 2, 3, 4], times=[-1, 0, 1, 2, 3]))

    data = _build([gapped, support], grid=GridSpec(step=1))
    aligned = data.aligned_traces[0]

    assert aligned.grid_x == (-1.0, 0.0, 1.0, 2.0, 3.0)
    assert aligned.aligned_values == (None, 1.0, None, 3.0, None)
    assert aligned.observed_mask == (False, True, False, True, False)
    assert not any(aligned.interpolated_mask)
    assert data.native_traces[0].missing_reasons[1] == "stack unavailable"


def test_adjacent_finite_native_samples_can_interpolate_to_finer_grid() -> None:
    data = _build(
        [_dataset("d1", _trace([0, 4], times=[0, 2]))],
        grid=GridSpec(step=1),
    )

    aligned = data.aligned_traces[0]
    assert aligned.aligned_values == (0.0, 2.0, 4.0)
    assert aligned.observed_mask == (True, False, True)
    assert aligned.interpolated_mask == (False, True, False)


def test_gaussian_smoothing_preserves_constants_and_does_not_bleed_across_gap() -> None:
    data = _build(
        [_dataset("d1", _trace([1, 1, None, 3, 3]))],
        smoothing=SmoothingSpec(sigma=1.0),
    )

    display = data.aligned_traces[0].display_values
    assert display[:2] == pytest.approx((1.0, 1.0))
    assert display[2] is None
    assert display[3:] == pytest.approx((3.0, 3.0))


def test_default_gaussian_policy_matches_the_shared_expression_smoother() -> None:
    data = _build(
        [_dataset("d1", _trace([0, 2, 8]))],
        smoothing=SmoothingSpec(sigma=1.0),
    )

    assert data.aligned_traces[0].display_values == pytest.approx(
        gaussian_smooth_missing((0, 2, 8), 1.0)
    )


def test_coarse_grid_cannot_hide_gap_from_smoothing_boundary() -> None:
    data = _build(
        [_dataset("d1", _trace([1, None, 3], times=[0, 1, 2]))],
        grid=GridSpec(step=2),
        smoothing=SmoothingSpec(sigma=10),
    )

    trace = data.aligned_traces[0]
    assert trace.grid_x == (0.0, 2.0)
    assert trace.segment_ids == (0, 1)
    assert trace.display_values == pytest.approx((1.0, 3.0))


def test_injected_smoother_receives_sigma_in_grid_bins() -> None:
    calls: list[tuple[float, float]] = []

    def smoother(values, sigma_bins, truncate):
        calls.append((sigma_bins, truncate))
        return tuple(None if value is None else value + 10 for value in values)

    service = ExpressionComparisonService(smoother=smoother)
    data = service.build(
        [_dataset("d1", _trace([1, 2, 3], times=[0, 2, 4]))],
        cell_names=["A"],
        channel_key="gfp",
        grid=GridSpec(step=2),
        smoothing=SmoothingSpec(sigma=4, truncate=3),
        summary=SummarySpec(CenterStatistic.MEAN, BandStatistic.NONE),
    )

    assert calls == [(2.0, 3.0)]
    assert data.aligned_traces[0].display_values == (11.0, 12.0, 13.0)
    assert data.summaries[0].center == (11.0, 12.0, 13.0)


def test_nonintegral_grid_step_retains_terminal_observation() -> None:
    data = _build(
        [_dataset("d1", _trace([1, 5], times=[0, 2.5]))],
        grid=GridSpec(step=1.0),
        summary=SummarySpec(CenterStatistic.MEAN, BandStatistic.NONE),
    )

    aligned = data.aligned_traces[0]
    assert aligned.grid_x == pytest.approx((0.0, 2.5 / 3, 5.0 / 3, 2.5))
    assert aligned.grid_step == pytest.approx(2.5 / 3)
    assert aligned.aligned_values[-1] == 5.0
    assert aligned.observed_mask[-1]


@pytest.mark.parametrize(
    ("center", "band", "expected_lower", "expected_upper"),
    [
        (
            CenterStatistic.MEAN,
            BandStatistic.SAMPLE_SD,
            2.5 - math.sqrt(5 / 3),
            2.5 + math.sqrt(5 / 3),
        ),
        (
            CenterStatistic.MEAN,
            BandStatistic.SEM,
            2.5 - math.sqrt(5 / 3) / 2,
            2.5 + math.sqrt(5 / 3) / 2,
        ),
        (CenterStatistic.MEDIAN, BandStatistic.IQR, 1.75, 3.25),
        (
            CenterStatistic.MEDIAN,
            BandStatistic.SCALED_MAD,
            2.5 - 1.4826,
            2.5 + 1.4826,
        ),
    ],
)
def test_pointwise_compatible_center_and_dispersion_bands(
    center: CenterStatistic,
    band: BandStatistic,
    expected_lower: float,
    expected_upper: float,
) -> None:
    datasets = [
        _dataset(f"d{value}", _trace([value])) for value in (1, 2, 3, 4)
    ]
    data = _build(
        datasets,
        summary=SummarySpec(center, band),
    )
    result = data.summaries[0]

    assert result.center == (2.5,)
    assert result.lower[0] == pytest.approx(expected_lower)
    assert result.upper[0] == pytest.approx(expected_upper)
    assert result.n_valid == (4,)


def test_student_t_95_confidence_interval() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    datasets = [
        _dataset(f"d{value}", _trace([value])) for value in (1, 2, 3, 4)
    ]
    data = _build(
        datasets,
        summary=SummarySpec(
            CenterStatistic.MEAN,
            BandStatistic.STUDENT_T_95,
        ),
    )
    summary = data.summaries[0]
    half_width = scipy_stats.t.ppf(0.975, 3) * math.sqrt(5 / 3) / 2

    assert summary.lower[0] == pytest.approx(2.5 - half_width)
    assert summary.upper[0] == pytest.approx(2.5 + half_width)


def test_median_center_and_no_summary_center() -> None:
    datasets = [
        _dataset("d1", _trace([1])),
        _dataset("d2", _trace([2])),
        _dataset("d3", _trace([100])),
    ]
    median = _build(
        datasets,
        summary=SummarySpec(CenterStatistic.MEDIAN, BandStatistic.NONE),
    )
    no_center = _build(
        datasets,
        summary=SummarySpec(CenterStatistic.NONE, BandStatistic.NONE),
    )

    assert median.summaries[0].center == (2.0,)
    assert no_center.summaries[0].center == (None,)
    assert no_center.summaries[0].n_valid == (3,)


def test_one_replicate_has_center_but_no_error_band() -> None:
    data = _build(
        [_dataset("d1", _trace([7]))],
        summary=SummarySpec(CenterStatistic.MEAN, BandStatistic.SAMPLE_SD),
    )
    result = data.summaries[0]

    assert result.center == (7.0,)
    assert result.lower == (None,)
    assert result.upper == (None,)


def test_datasets_are_equal_replicate_units_with_pointwise_counts() -> None:
    first = _dataset("d1", _trace([1, 2]))
    all_missing = _dataset("d2", _trace([None, None]))
    missing_cell = _dataset("d3", _trace([9, 9], cell="B"))

    data = _build([first, all_missing, missing_cell])
    result = data.summaries[0]

    assert result.n_selected == 3
    assert result.n_available == 2
    assert result.n_valid == (1, 1)
    assert result.center == (1.0, 2.0)


def test_group_summaries_do_not_pool_conditions() -> None:
    datasets = [
        _dataset("a1", _trace([1]), group="control"),
        _dataset("a2", _trace([3]), group="control"),
        _dataset("b1", _trace([10]), group="treated"),
    ]

    data = _build(datasets)

    assert [(item.group_id, item.center) for item in data.summaries] == [
        ("control", (2.0,)),
        ("treated", (10.0,)),
    ]
    assert [item.n_selected for item in data.summaries] == [2, 1]


def test_tidy_export_contains_exact_native_aligned_summary_and_status_rows() -> None:
    precise = 1.2345678901234567
    available = _dataset("d1", _trace([precise, None]))
    absent = _dataset("d2", _trace([5, 6], cell="B"))
    data = _build([available, absent])
    output = io.StringIO()

    export_expression_comparison_tidy_csv(data, output)

    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    record_types = {row["record_type"] for row in rows}
    assert record_types == {
        "trace_status",
        "native_sample",
        "aligned_sample",
        "summary_sample",
    }
    native = next(
        row for row in rows
        if row["record_type"] == "native_sample" and row["sample_index"] == "0"
    )
    assert native["raw_value"] == format(precise, ".17g")
    assert float(native["raw_value"]) == precise
    assert native["source_fingerprint"] == "fingerprint-d1"
    assert native["provenance_metadata"] == '{"strain":"N2"}'

    missing_native = next(
        row for row in rows
        if row["record_type"] == "native_sample" and row["sample_index"] == "1"
    )
    assert missing_native["raw_value"] == ""
    assert missing_native["missing_reason"] == "missing"

    aligned = next(
        row for row in rows
        if row["record_type"] == "aligned_sample" and row["sample_index"] == "0"
    )
    assert aligned["display_value"] == format(precise, ".17g")
    assert aligned["is_observed"] == "true"

    summary = next(row for row in rows if row["record_type"] == "summary_sample")
    assert summary["n_selected"] == "2"
    assert summary["n_available"] == "1"
    assert summary["n_valid"] == "1"
    absent_status = next(
        row for row in rows
        if row["record_type"] == "trace_status" and row["dataset_id"] == "d2"
    )
    assert absent_status["trace_status"] == "missing_cell"


def test_export_accepts_path_and_creates_parent(tmp_path) -> None:
    destination = tmp_path / "nested" / "comparison.csv"
    data = _build([_dataset("d1", _trace([1]))])

    export_expression_comparison_tidy_csv(data, destination)

    assert destination.exists()
    assert destination.read_text(encoding="utf-8").startswith("schema_version,")


def test_validation_rejects_duplicate_datasets_mixed_units_and_bad_smoother() -> None:
    one = _dataset("same", _trace([1], unit="AU"))
    duplicate = _dataset("same", _trace([2], unit="AU"))
    with pytest.raises(ValueError, match="dataset_id"):
        _build([one, duplicate])

    mixed = _dataset("other", _trace([2], unit="photons"))
    with pytest.raises(ValueError, match="different channel units"):
        _build([one, mixed])

    bad_service = ExpressionComparisonService(
        smoother=lambda _values, _sigma, _truncate: (1.0,)
    )
    with pytest.raises(ValueError, match="preserve.*length"):
        bad_service.build(
            [_dataset("d1", _trace([1, 2]))],
            cell_names=["A"],
            channel_key="gfp",
            grid=GridSpec(step=1),
            smoothing=SmoothingSpec(sigma=1),
        )


def test_model_input_validation_and_no_intersection() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _trace([1, 2], times=[1, 1])
    with pytest.raises(ValueError, match="finite or None"):
        _trace([math.nan])
    with pytest.raises(ValueError, match="shaded band"):
        SummarySpec(CenterStatistic.NONE, BandStatistic.SEM)
    with pytest.raises(ValueError, match="mean center"):
        SummarySpec(CenterStatistic.MEAN, BandStatistic.SCALED_MAD)
    with pytest.raises(ValueError, match="median center"):
        SummarySpec(CenterStatistic.MEDIAN, BandStatistic.STUDENT_T_95)

    first = _dataset("d1", _trace([1, 2], times=[0, 1]))
    second = _dataset("d2", _trace([3, 4], times=[2, 3]))
    with pytest.raises(ValueError, match="No intersection"):
        _build(
            [first, second],
            grid=GridSpec(GridDomain.INTERSECTION, step=1),
        )
