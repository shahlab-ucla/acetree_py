"""Portable expression-comparison result persistence tests."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import pytest

import acetree_py.analysis.expression_comparison_result as result_module
from acetree_py.analysis.expression_comparison import (
    BandStatistic,
    CenterStatistic,
    ComparisonSpec,
    DatasetAcquisitionStatus,
    DatasetExpressionTrace,
    DatasetProvenance,
    ExpressionDataset,
    GridDomain,
    GridSpec,
    SmoothingSpec,
    SummarySpec,
    TimeAxisMode,
    TraceAvailability,
    export_expression_comparison_tidy_csv,
)
from acetree_py.analysis.expression_comparison_result import (
    APPEARANCE_INCLUDED_DATASET_IDS,
    EXPRESSION_COMPARISON_RESULT_SCHEMA,
    EXPRESSION_COMPARISON_RESULT_LEGACY_VERSION,
    EXPRESSION_COMPARISON_RESULT_VERSION,
    ExpressionComparisonResultFormatError,
    ExpressionComparisonSourceMode,
    build_expression_comparison_data,
    build_expression_comparison_cache_data,
    capture_expression_comparison_result,
    capture_expression_comparison_measurement_caches,
    load_expression_comparison_result,
    materialize_expression_measurement_caches,
    revise_expression_comparison_result,
    revise_expression_comparison_measurement_caches,
    save_expression_comparison_result,
)
from acetree_py.analysis.expression_measurements import (
    FrozenCellMeasurements,
    FrozenDatasetMeasurementCache,
    MeasuredExpressionAggregate,
    MeasuredExpressionAggregateChannel,
    NucleusGeometrySignature,
)


_CAPTURED = "2026-08-04T12:34:56.123456Z"
_RESULT_ID = "12345678-1234-4234-8234-123456789abc"


def _trace(
    values: tuple[float | None, ...],
    *,
    color: str,
    series_label: str,
) -> DatasetExpressionTrace:
    return DatasetExpressionTrace(
        cell_name="ABa",
        channel_key="gfp",
        channel_label="GFP",
        channel_unit="AU",
        absolute_times=(10.0, 11.0, 12.0),
        values=values,
        birth_time=10.0,
        end_time=12.0,
        missing_reasons=tuple("unreadable stack" if value is None else None for value in values),
        series_label=series_label,
        color=color,
    )


def _datasets() -> tuple[ExpressionDataset, ...]:
    first = ExpressionDataset(
        provenance=DatasetProvenance(
            dataset_id="d1",
            label="Embryo 1",
            group_id="control",
            source_uri="C:/source/one.xml",
            source_fingerprint="sha256:one",
            source_revision=4,
            metadata=(("correction_method", "global"), ("trace_source", "recomputed")),
        ),
        traces=(_trace((1.0, None, 5.0), color="#112233", series_label="Embryo 1"),),
    )
    second = ExpressionDataset(
        provenance=DatasetProvenance(
            dataset_id="d2",
            label="Embryo 2",
            group_id="control",
            source_uri="C:/source/two.xml",
            source_fingerprint="sha256:two",
            source_revision=7,
            metadata=(("correction_method", "global"),),
        ),
        traces=(_trace((3.0, 5.0, 7.0), color="#445566", series_label="Embryo 2"),),
    )
    unavailable = ExpressionDataset(
        provenance=DatasetProvenance(
            dataset_id="d3",
            label="Embryo 3",
            group_id="treated",
            source_uri="C:/source/three.xml",
            source_fingerprint="sha256:three",
            source_revision=2,
            metadata=(("trace_source", "saved_legacy"),),
        ),
        traces=(),
        acquisition_statuses=(
            DatasetAcquisitionStatus(
                cell_name="ABa",
                channel_key="gfp",
                availability=TraceAvailability.MISSING_CELL,
                message="Canonical cell ABa is absent.",
            ),
        ),
    )
    return first, second, unavailable


def _spec(
    *,
    time_mode: TimeAxisMode = TimeAxisMode.ABSOLUTE,
    center: CenterStatistic = CenterStatistic.MEAN,
    band: BandStatistic = BandStatistic.SAMPLE_SD,
    smoothing: float = 0.0,
) -> ComparisonSpec:
    return ComparisonSpec(
        cell_names=("ABa",),
        channel_key="gfp",
        channel_label="GFP expression",
        channel_unit="AU",
        time_mode=time_mode,
        grid=GridSpec(
            domain=GridDomain.UNION,
            step=None if time_mode is TimeAxisMode.NORMALIZED else 1.0,
            normalized_points=5,
            start=None,
            end=None,
            max_points=1000,
        ),
        smoothing=SmoothingSpec(sigma=smoothing, truncate=3.5),
        summary=SummarySpec(center=center, band=band),
    )


def _capture(*, included=("d1", "d2", "d3")):
    return capture_expression_comparison_result(
        _datasets(),
        _spec(),
        source_mode=ExpressionComparisonSourceMode.MIXED,
        acquisition_metadata={
            "request": {"image_channel": 1, "correction": "global"},
            "attempts": ["cache", "measure"],
        },
        legacy_acknowledged=True,
        appearance={
            APPEARANCE_INCLUDED_DATASET_IDS: list(included),
            "trace_opacity": 0.35,
            "group_colors": {"control": "#112233", "treated": "#778899"},
        },
        captured_at=_CAPTURED,
        producer_version="0.2.0-test",
        result_id=_RESULT_ID,
    )


def _measurement_cache(
    dataset_id: str = "d1",
    *,
    raw_offset: float = 0.0,
    measured_at: str = _CAPTURED,
    omit_abp_channel_two: bool = False,
) -> FrozenDatasetMeasurementCache:
    aba_keys = ((10, 1), (11, 2))
    abp_keys = ((12, 3), (13, 4))

    def aggregate(raw: float) -> MeasuredExpressionAggregate:
        return MeasuredExpressionAggregate(
            raw=raw + raw_offset,
            annulus_background=10.0,
            blot_background=20.0,
            inner_pixel_count=8,
            annulus_pixel_count=12,
            blot_pixel_count=10,
        )

    first_samples = {
        aba_keys[0]: aggregate(100.0),
        aba_keys[1]: aggregate(110.0),
        abp_keys[0]: aggregate(200.0),
        abp_keys[1]: aggregate(210.0),
    }
    second_samples = {
        aba_keys[0]: aggregate(300.0),
        aba_keys[1]: aggregate(310.0),
        abp_keys[0]: aggregate(400.0),
        abp_keys[1]: aggregate(410.0),
    }
    missing_reasons = {}
    if omit_abp_channel_two:
        second_samples.pop(abp_keys[1])
        missing_reasons[(1, *abp_keys[1])] = "stack unreadable"
    all_keys = aba_keys + abp_keys
    return FrozenDatasetMeasurementCache(
        dataset_id=dataset_id,
        source_uri=f"C:/source/{dataset_id}.xml",
        source_fingerprint=f"sha256:{dataset_id}",
        snapshot_token=f"snapshot:{dataset_id}",
        dataset_generation=3,
        image_manifest_token=f"manifest:{dataset_id}",
        measured_at=measured_at,
        measurement_algorithm_version=1,
        source_revision=7,
        source_dependency_fingerprint=f"dependency:{dataset_id}",
        source_calibration=(1.0, 1.0, 1, 1.0),
        channels=(
            MeasuredExpressionAggregateChannel(0, "GFP", first_samples),
            MeasuredExpressionAggregateChannel(1, "RFP", second_samples),
        ),
        cells=(
            FrozenCellMeasurements("cell-aba", "ABa", 10, 11, aba_keys),
            FrozenCellMeasurements("cell-abp", "ABp", 12, 13, abp_keys),
        ),
        geometries={
            key: NucleusGeometrySignature(
                x=index,
                y=index + 1,
                z=float(index),
                size=20,
                status=1,
            )
            for _time, index in all_keys
            for key in ((_time, index),)
        },
        missing_reasons=missing_reasons,
    )


def _cache_capture(*caches: FrozenDatasetMeasurementCache):
    return capture_expression_comparison_measurement_caches(
        caches,
        _spec(),
        cell_name="ABa",
        image_channel=0,
        correction_method="global",
        appearance={
            APPEARANCE_INCLUDED_DATASET_IDS: [cache.dataset_id for cache in caches]
        },
        overrides={
            cache.dataset_id: {
                "label": f"Dataset {cache.dataset_id}",
                "group_id": "control",
                "color": "#123456",
            }
            for cache in caches
        },
        captured_at=_CAPTURED,
        producer_version="0.2.0-test",
        result_id=_RESULT_ID,
    )


def _rewrite_with_checksum(path, mutate) -> None:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    mutate(envelope["result"])
    compact = json.dumps(
        envelope["result"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    envelope["checksum"]["sha256"] = hashlib.sha256(compact).hexdigest()
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_round_trip_is_checksummed_immutable_and_repository_free(tmp_path) -> None:
    capture = _capture(included=("d1", "d2"))

    saved = save_expression_comparison_result(tmp_path / "portable", capture)
    loaded = load_expression_comparison_result(saved.path)

    assert saved.path.suffix == ".aceexpr"
    assert loaded.result_id == _RESULT_ID
    assert loaded.parent_result_id is None
    assert loaded.captured_at == _CAPTURED
    assert loaded.saved_at is not None
    assert loaded.producer_version == "0.2.0-test"
    assert loaded.source_mode is ExpressionComparisonSourceMode.MIXED
    assert loaded.legacy_acknowledged
    assert loaded.datasets == capture.datasets
    assert loaded.spec == capture.spec
    assert loaded.acquisition_metadata["attempts"] == ("cache", "measure")
    with pytest.raises(TypeError):
        loaded.appearance["new"] = True
    with pytest.raises(TypeError):
        loaded.appearance["group_colors"]["control"] = "#ffffff"

    # Inclusion is presentation state: d3 remains in the file but is absent
    # from this build, and no repository object/path is consulted.
    data = build_expression_comparison_data(loaded)
    assert [item.dataset_id for item in data.datasets] == ["d1", "d2"]
    assert [item.dataset_id for item in data.statuses] == ["d1", "d2"]
    assert data.summaries[0].n_selected == 2
    output = io.StringIO()
    export_expression_comparison_tidy_csv(data, output)
    assert "native_sample" in output.getvalue()

    envelope = json.loads(saved.path.read_text(encoding="utf-8"))
    assert envelope["schema"] == EXPRESSION_COMPARISON_RESULT_SCHEMA
    assert envelope["checksum"]["algorithm"] == "sha256"
    assert len(envelope["checksum"]["sha256"]) == 64


@pytest.mark.parametrize("time_mode", list(TimeAxisMode))
@pytest.mark.parametrize(
    ("center", "band"),
    [
        (CenterStatistic.NONE, BandStatistic.NONE),
        (CenterStatistic.MEAN, BandStatistic.NONE),
        (CenterStatistic.MEAN, BandStatistic.SAMPLE_SD),
        (CenterStatistic.MEAN, BandStatistic.SEM),
        (CenterStatistic.MEAN, BandStatistic.STUDENT_T_95),
        (CenterStatistic.MEDIAN, BandStatistic.NONE),
        (CenterStatistic.MEDIAN, BandStatistic.IQR),
        (CenterStatistic.MEDIAN, BandStatistic.SCALED_MAD),
    ],
)
@pytest.mark.parametrize("sigma", [0.0, 0.75])
def test_native_capture_rebuilds_every_time_summary_and_smoothing_family(
    time_mode,
    center,
    band,
    sigma,
) -> None:
    capture = _capture(included=("d1", "d2"))
    selected = _spec(
        time_mode=time_mode,
        center=center,
        band=band,
        smoothing=sigma,
    )

    data = build_expression_comparison_data(capture, spec=selected)

    assert len(data.native_traces) == 2
    assert len(data.aligned_traces) == 2
    assert data.spec.time_mode is time_mode
    assert data.spec.summary == selected.summary
    assert data.spec.smoothing == selected.smoothing
    assert data.native_traces[0].absolute_times == (10.0, 11.0, 12.0)
    assert data.native_traces[0].values == (1.0, None, 5.0)


def test_resave_forms_lineage_and_preserves_original_capture_provenance(tmp_path) -> None:
    first = save_expression_comparison_result(tmp_path / "first.aceexpr", _capture())
    loaded = load_expression_comparison_result(first.path)
    datasets = list(loaded.datasets)
    source = datasets[0].provenance
    recolored = replace(
        datasets[0].traces[0],
        series_label="Renamed embryo",
        color="#abcdef",
    )
    datasets[0] = replace(
        datasets[0],
        provenance=replace(source, label="Renamed embryo", group_id="new group"),
        traces=(recolored,),
    )
    revised_spec = replace(
        loaded.spec,
        time_mode=TimeAxisMode.RELATIVE,
        smoothing=SmoothingSpec(1.25),
        summary=SummarySpec(CenterStatistic.MEDIAN, BandStatistic.IQR),
    )

    revised = revise_expression_comparison_result(
        loaded,
        datasets=datasets,
        spec=revised_spec,
        appearance={APPEARANCE_INCLUDED_DATASET_IDS: ["d1"]},
    )
    second = save_expression_comparison_result(tmp_path / "second", revised)
    restored = load_expression_comparison_result(second.path)

    assert restored.parent_result_id == loaded.result_id
    assert restored.result_id != loaded.result_id
    assert restored.captured_at == loaded.captured_at
    assert restored.producer_version == loaded.producer_version
    assert restored.acquisition_metadata == loaded.acquisition_metadata
    assert restored.legacy_acknowledged == loaded.legacy_acknowledged
    assert restored.datasets[0].provenance.label == "Renamed embryo"
    assert restored.datasets[0].traces[0].color == "#abcdef"
    assert len(restored.datasets) == 3
    assert len(build_expression_comparison_data(restored).datasets) == 1

    # Saving an already-saved object directly also creates a child revision.
    third = save_expression_comparison_result(tmp_path / "third", restored)
    assert third.result.parent_result_id == restored.result_id
    assert third.result.result_id != restored.result_id
    assert third.result.captured_at == restored.captured_at


def test_presentation_revision_rejects_native_status_source_and_request_changes() -> None:
    capture = _capture()
    changed_value = replace(
        capture.datasets[0].traces[0], values=(99.0, None, 5.0)
    )
    changed_dataset = replace(capture.datasets[0], traces=(changed_value,))
    with pytest.raises(ValueError, match="native trace content"):
        revise_expression_comparison_result(
            capture,
            datasets=(changed_dataset,) + capture.datasets[1:],
        )

    changed_source = replace(
        capture.datasets[0],
        provenance=replace(capture.datasets[0].provenance, source_fingerprint="tampered"),
    )
    with pytest.raises(ValueError, match="source provenance"):
        revise_expression_comparison_result(
            capture,
            datasets=(changed_source,) + capture.datasets[1:],
        )

    changed_status = replace(
        capture.datasets[2],
        acquisition_statuses=(
            replace(
                capture.datasets[2].acquisition_statuses[0],
                message="different",
            ),
        ),
    )
    with pytest.raises(ValueError, match="acquisition statuses"):
        revise_expression_comparison_result(
            capture,
            datasets=capture.datasets[:2] + (changed_status,),
        )

    with pytest.raises(ValueError, match="captured cell/channel"):
        revise_expression_comparison_result(
            capture,
            spec=replace(capture.spec, channel_key="rfp"),
        )


def test_capture_rejects_duplicate_or_nonmaterialized_inputs() -> None:
    datasets = _datasets()
    duplicate_id = replace(
        datasets[1],
        provenance=replace(datasets[1].provenance, dataset_id="d1"),
    )
    with pytest.raises(ValueError, match="duplicate dataset_id"):
        capture_expression_comparison_result(
            (datasets[0], duplicate_id),
            _spec(),
            source_mode="recomputed",
        )

    duplicate_trace = replace(
        datasets[0], traces=(datasets[0].traces[0], datasets[0].traces[0])
    )
    with pytest.raises(ValueError, match="duplicate trace key"):
        capture_expression_comparison_result(
            (duplicate_trace,),
            _spec(),
            source_mode="recomputed",
        )

    with pytest.raises(ValueError, match="exactly one materialised trace"):
        capture_expression_comparison_result(
            (replace(datasets[0], traces=()),),
            _spec(),
            source_mode="recomputed",
        )


def test_json_safe_metadata_appearance_and_inclusion_are_strict() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        capture_expression_comparison_result(
            _datasets(),
            _spec(),
            source_mode="mixed",
            appearance={"opacity": float("nan")},
        )
    with pytest.raises(TypeError, match="keys must be strings"):
        capture_expression_comparison_result(
            _datasets(),
            _spec(),
            source_mode="mixed",
            acquisition_metadata={1: "bad"},
        )
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="cycle"):
        capture_expression_comparison_result(
            _datasets(),
            _spec(),
            source_mode="mixed",
            appearance=cyclic,
        )
    with pytest.raises(ValueError, match="unknown datasets"):
        _capture(included=("unknown",))


def test_loader_rejects_checksum_tampering_duplicates_nonfinite_and_bad_enums(
    tmp_path,
) -> None:
    saved = save_expression_comparison_result(tmp_path / "valid", _capture())
    original = saved.path.read_text(encoding="utf-8")

    # Ordinary payload edits fail before any model construction.
    saved.path.write_text(original.replace("Embryo 1", "Embryo X", 1), encoding="utf-8")
    with pytest.raises(ExpressionComparisonResultFormatError, match="checksum"):
        load_expression_comparison_result(saved.path)

    # Duplicate keys and non-standard JSON constants are rejected explicitly.
    saved.path.write_text(
        original.replace(
            '"schema": "acetree.expression-comparison-result",',
            '"schema": "acetree.expression-comparison-result",\n  "schema": "duplicate",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExpressionComparisonResultFormatError, match="Duplicate"):
        load_expression_comparison_result(saved.path)

    saved.path.write_text(
        original.replace('"trace_opacity": 0.35', '"trace_opacity": NaN', 1),
        encoding="utf-8",
    )
    with pytest.raises(ExpressionComparisonResultFormatError, match="Non-finite"):
        load_expression_comparison_result(saved.path)

    saved.path.write_text(original, encoding="utf-8")
    _rewrite_with_checksum(
        saved.path,
        lambda payload: payload.__setitem__("source_mode", "mystery"),
    )
    with pytest.raises(ExpressionComparisonResultFormatError, match="source_mode"):
        load_expression_comparison_result(saved.path)


def test_atomic_replace_failure_preserves_last_good_file(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "comparison.aceexpr"
    destination.write_text("last-good\n", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(result_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_expression_comparison_result(destination, _capture())

    assert destination.read_text(encoding="utf-8") == "last-good\n"
    assert list(tmp_path.glob(".comparison.aceexpr.*.tmp")) == []


def test_loader_rejects_unknown_fields_even_with_valid_checksum(tmp_path) -> None:
    saved = save_expression_comparison_result(tmp_path / "strict", _capture())
    _rewrite_with_checksum(
        saved.path,
        lambda payload: payload.__setitem__("future_field", True),
    )

    with pytest.raises(ExpressionComparisonResultFormatError, match="unexpected fields"):
        load_expression_comparison_result(saved.path)


def test_v2_round_trip_retargets_every_cached_cell_channel_and_correction(
    tmp_path,
) -> None:
    cache = _measurement_cache()
    capture = _cache_capture(cache)

    saved = save_expression_comparison_result(tmp_path / "full-cache", capture)
    loaded = load_expression_comparison_result(saved.path)

    envelope = json.loads(saved.path.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == EXPRESSION_COMPARISON_RESULT_VERSION
    assert len(envelope["result"]["measurement_caches"]) == 1
    assert loaded.measurement_caches == (cache,)

    expected = {
        "none": (100.0, 110.0),
        "global": (90.0, 100.0),
        "local": (90.0, 100.0),
        "blot": (80.0, 90.0),
        "cross": (90.0, 100.0),
    }
    for correction, values in expected.items():
        dataset = materialize_expression_measurement_caches(
            loaded,
            cell_name="ABa",
            image_channel=0,
            correction_method=correction,
        )[0]
        assert dataset.traces[0].values == pytest.approx(values)
        metadata = dict(dataset.provenance.metadata)
        assert metadata["correction_method"] == correction
        assert metadata["correction_exact"] == str(
            correction not in ("local", "cross")
        ).lower()

    data = build_expression_comparison_cache_data(
        loaded,
        cell_name="ABp",
        image_channel=1,
        correction_method="none",
    )
    assert data.spec.cell_names == ("ABp",)
    assert data.spec.channel_key == "measured_channel_2"
    assert data.native_traces[0].values == pytest.approx((400.0, 410.0))


def test_v1_json_remains_fixed_view_and_resaves_as_v2(tmp_path) -> None:
    saved = save_expression_comparison_result(tmp_path / "modern", _capture())
    envelope = json.loads(saved.path.read_text(encoding="utf-8"))
    envelope["schema_version"] = EXPRESSION_COMPARISON_RESULT_LEGACY_VERSION
    envelope["result"].pop("measurement_caches")
    compact = json.dumps(
        envelope["result"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    envelope["checksum"]["sha256"] = hashlib.sha256(compact).hexdigest()
    legacy_path = tmp_path / "legacy-v1.aceexpr"
    legacy_path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded = load_expression_comparison_result(legacy_path)
    assert loaded.measurement_caches == ()
    assert build_expression_comparison_data(loaded).native_traces
    with pytest.raises(ValueError, match="only selected traces"):
        materialize_expression_measurement_caches(
            loaded,
            cell_name="ABp",
            image_channel=1,
            correction_method="global",
        )

    upgraded = save_expression_comparison_result(tmp_path / "upgraded", loaded)
    upgraded_envelope = json.loads(upgraded.path.read_text(encoding="utf-8"))
    assert upgraded_envelope["schema_version"] == EXPRESSION_COMPARISON_RESULT_VERSION
    assert upgraded_envelope["result"]["measurement_caches"] == []


def test_cache_revision_appends_preserves_and_explicitly_replaces() -> None:
    first = _measurement_cache("d1", measured_at="2026-08-01T00:00:00Z")
    second = _measurement_cache("d2", measured_at="2026-08-02T00:00:00Z")
    capture = _cache_capture(first)

    appended = revise_expression_comparison_measurement_caches(capture, (second,))

    assert appended.parent_result_id == capture.result_id
    assert appended.result_id != capture.result_id
    assert appended.saved_at is None
    assert [cache.dataset_id for cache in appended.measurement_caches] == ["d1", "d2"]
    assert appended.measurement_caches[0] == first
    assert appended.measurement_caches[0].measured_at == "2026-08-01T00:00:00Z"
    assert [dataset.provenance.dataset_id for dataset in appended.datasets] == [
        "d1",
        "d2",
    ]

    changed = _measurement_cache(
        "d1",
        raw_offset=1000.0,
        measured_at="2026-08-03T00:00:00Z",
    )
    with pytest.raises(ValueError, match="replace_existing=True"):
        revise_expression_comparison_measurement_caches(appended, (changed,))

    replaced = revise_expression_comparison_measurement_caches(
        appended,
        (changed,),
        replace_existing=True,
    )
    assert replaced.measurement_caches[0] == changed
    assert replaced.measurement_caches[1] == second
    d1 = next(
        dataset
        for dataset in replaced.datasets
        if dataset.provenance.dataset_id == "d1"
    )
    assert d1.traces[0].values == pytest.approx((1090.0, 1100.0))


def test_cache_view_revision_preserves_original_capture_provenance() -> None:
    cache = _measurement_cache("d1", measured_at="2026-08-01T00:00:00Z")
    capture = _cache_capture(cache)

    retargeted = revise_expression_comparison_measurement_caches(
        capture,
        (cache,),
        cell_name="ABp",
        image_channel=1,
        correction_method="none",
    )

    assert retargeted.parent_result_id == capture.result_id
    assert retargeted.captured_at == capture.captured_at
    assert retargeted.producer_version == capture.producer_version
    assert retargeted.measurement_caches == capture.measurement_caches


def test_partial_cached_samples_remain_explicit_gaps_after_round_trip(tmp_path) -> None:
    cache = _measurement_cache(omit_abp_channel_two=True)
    saved = save_expression_comparison_result(
        tmp_path / "partial",
        _cache_capture(cache),
    )
    loaded = load_expression_comparison_result(saved.path)

    dataset = materialize_expression_measurement_caches(
        loaded,
        cell_name="ABp",
        image_channel=1,
        correction_method="none",
    )[0]
    assert dataset.traces[0].values[0] == pytest.approx(400.0)
    assert dataset.traces[0].values[1] is None
    assert dataset.traces[0].missing_reasons == (None, "stack unreadable")


def test_v2_cache_payload_is_strict_and_checksummed(tmp_path) -> None:
    saved = save_expression_comparison_result(
        tmp_path / "strict-cache",
        _cache_capture(_measurement_cache()),
    )
    _rewrite_with_checksum(
        saved.path,
        lambda payload: payload["measurement_caches"][0].__setitem__(
            "future_cache_field", True
        ),
    )

    with pytest.raises(
        ExpressionComparisonResultFormatError,
        match="unexpected fields",
    ):
        load_expression_comparison_result(saved.path)

    saved = save_expression_comparison_result(
        tmp_path / "future-algorithm",
        _cache_capture(_measurement_cache()),
    )
    _rewrite_with_checksum(
        saved.path,
        lambda payload: payload["measurement_caches"][0].__setitem__(
            "measurement_algorithm_version", 999
        ),
    )
    with pytest.raises(
        ExpressionComparisonResultFormatError,
        match="unsupported frozen measurement algorithm version",
    ):
        load_expression_comparison_result(saved.path)


def test_v2_rejects_orphan_cache_and_conflicting_default_view(tmp_path) -> None:
    capture = _cache_capture(_measurement_cache())
    orphan = replace(
        capture.measurement_caches[0],
        dataset_id="orphan-cache",
    )
    with pytest.raises(ValueError, match="corresponding default-view datasets"):
        replace(capture, measurement_caches=(orphan,))

    conflicting_dataset = replace(
        capture.datasets[0],
        provenance=replace(
            capture.datasets[0].provenance,
            source_fingerprint="conflicting-source",
        ),
    )
    with pytest.raises(ValueError, match="source fingerprint does not match"):
        replace(capture, datasets=(conflicting_dataset,))

    saved = save_expression_comparison_result(tmp_path / "orphan", capture)
    _rewrite_with_checksum(
        saved.path,
        lambda payload: payload["measurement_caches"][0].__setitem__(
            "dataset_id", "orphan-cache"
        ),
    )
    with pytest.raises(
        ExpressionComparisonResultFormatError,
        match="corresponding default-view datasets",
    ):
        load_expression_comparison_result(saved.path)


def test_v2_rejects_checksum_valid_cache_default_view_divergence(tmp_path) -> None:
    capture = _cache_capture(_measurement_cache())
    changed_trace = replace(
        capture.datasets[0].traces[0],
        values=(123456.0, capture.datasets[0].traces[0].values[1]),
    )
    with pytest.raises(ValueError, match="materialized default view"):
        replace(
            capture,
            datasets=(replace(capture.datasets[0], traces=(changed_trace,)),),
        )

    saved = save_expression_comparison_result(tmp_path / "divergent", capture)
    _rewrite_with_checksum(
        saved.path,
        lambda payload: payload["measurement_caches"][0]["channels"][0][
            "samples"
        ][0].__setitem__(2, 123456.0),
    )
    with pytest.raises(
        ExpressionComparisonResultFormatError,
        match="materialized default view",
    ):
        load_expression_comparison_result(saved.path)


def test_writer_rejects_oversized_cache_before_replacing_last_good(
    tmp_path,
    monkeypatch,
) -> None:
    destination = tmp_path / "bounded.aceexpr"
    destination.write_text("last-good\n", encoding="utf-8")
    monkeypatch.setattr(result_module, "_MAX_FILE_BYTES", 128)

    with pytest.raises(ValueError, match="exceeds 128 bytes"):
        save_expression_comparison_result(
            destination,
            _cache_capture(_measurement_cache()),
        )

    assert destination.read_text(encoding="utf-8") == "last-good\n"
    assert list(tmp_path.glob(".bounded.aceexpr.*.tmp")) == []


def test_historical_cache_round_trip_and_corrected_comparison_boundary(tmp_path):
    legacy = _measurement_cache("legacy")
    original = _cache_capture(legacy)
    destination = tmp_path / "historical.aceexpr"
    save_expression_comparison_result(destination, original)
    before = destination.read_bytes()
    reopened = load_expression_comparison_result(destination)
    assert reopened.measurement_caches[0].measurement_algorithm_version == 1
    assert reopened.measurement_caches[0].channels == legacy.channels
    historical = build_expression_comparison_data(reopened)
    assert historical.native_traces[0].values == (90.0, 100.0)
    assert destination.read_bytes() == before

    corrected = replace(_measurement_cache("corrected"), measurement_algorithm_version=2)
    mixed = _cache_capture(legacy, corrected)
    with pytest.raises(ValueError, match="Recompute.*exclude"):
        build_expression_comparison_data(mixed)
    # Keeping the historical row in a portable set does not prevent selecting
    # only the corrected dataset for a new comparison.
    data = build_expression_comparison_data(mixed, included_dataset_ids=("corrected",))
    assert data.native_traces[0].values == (90.0, 100.0)
