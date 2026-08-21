import csv
from io import StringIO
from types import MappingProxyType, SimpleNamespace

from acetree_py.analysis.expression_plot import (
    ScalarSeriesChannel,
    ScalarSeriesSample,
    TemporalSeriesService,
    TemporalSeriesSubject,
    export_scalar_series_csv,
)
from acetree_py.analysis.roi_measurements import (
    RoiMeasurementEngine,
    RoiScalarSeriesChannel,
    roi_temporal_subject,
)


def test_temporal_series_preserves_zero_and_explicit_gaps():
    subject = TemporalSeriesSubject(
        key="roi:4d",
        label="Golgi #2",
        start_time=5,
        end_time=7,
        sample_times=(5, 6, 7),
        metadata={"object_id": "4d", "instance_index": 2},
    )
    values = {
        5: ScalarSeriesSample(0.0),
        6: ScalarSeriesSample(None, "roi_absent"),
        7: ScalarSeriesSample(8.5),
    }
    channel = ScalarSeriesChannel(
        key="roi:4d:ch2:intensity.mean",
        label="Mean intensity",
        unit="a.u.",
        reader=lambda _subject, time: values[time],
        source_token=lambda: "snapshot-7",
    )

    data = TemporalSeriesService().build(
        (subject,), channel, "relative", series_kind="subcellular_objects"
    )
    series = data.series[0]

    assert series.x_values == (0.0, 1.0, 2.0)
    assert series.y_values == (0.0, None, 8.5)
    assert series.missing_reasons == (None, "roi_absent", None)
    assert data.source_token == "snapshot-7"


def test_temporal_series_export_includes_roi_provenance_and_reason():
    subject = TemporalSeriesSubject(
        key="roi:object-uuid",
        label="Membrane #1",
        start_time=2,
        end_time=2,
        sample_times=(2,),
        metadata={"class_id": "class-uuid"},
    )
    channel = ScalarSeriesChannel(
        key="roi:object-uuid:ch1:intensity.sum",
        label="Integrated intensity",
        reader=lambda _subject, _time: ScalarSeriesSample(None, "stale"),
        source_token=lambda: "roi-revision-9",
    )
    data = TemporalSeriesService().build(
        (subject,), channel, series_kind="subcellular_objects"
    )
    output = StringIO()

    export_scalar_series_csv(data, output)

    exported = output.getvalue()
    assert "roi:object-uuid:ch1:intensity.sum" in exported
    assert "stale" in exported
    assert "roi-revision-9" in exported
    assert '""class_id"":""class-uuid""' in exported


def test_roi_adapter_exports_stable_key_and_complete_channel_provenance():
    geometry = SimpleNamespace(
        kind="polygon_2d",
        z_plane=1,
        exterior_xy_px=((1, 1), (3, 1), (3, 3), (1, 3)),
    )
    frame = SimpleNamespace(
        frame_id="frame-1",
        timepoint=1,
        presence="segmented",
        geometry=geometry,
    )
    track = SimpleNamespace(
        object_id="object-1",
        class_id="class-1",
        instance_index=2,
        expected_start_time=1,
        expected_end_time=2,
        frames=MappingProxyType({1: frame}),
    )
    document = SimpleNamespace(
        document_id="document-1",
        roi_revision=3,
        coordinate_space=SimpleNamespace(xy_res=1, z_res=2, plane_start=1),
        objects=(track,),
    )
    provider = SimpleNamespace(
        num_channels=1,
        num_planes=1,
        image_shape=(5, 5),
        manifest_token="images-v1",
        get_plane=lambda _time, _plane, _channel: __import__("numpy").ones((5, 5)),
    )
    snapshot = RoiMeasurementEngine(provider).measure(document)
    subject = roi_temporal_subject(
        track,
        object_class=SimpleNamespace(name="Golgi"),
    )
    channel = RoiScalarSeriesChannel(
        snapshot,
        image_channel=0,
        metric_key="intensity.mean",
        label="Mean intensity",
        unit="a.u.",
        source=document,
        image_provider=provider,
    ).as_scalar_series_channel()
    data = TemporalSeriesService().build(
        (subject,),
        channel,
        "relative",
        series_kind="subcellular_objects",
    )
    output = StringIO()
    export_scalar_series_csv(data, output)
    row = next(csv.DictReader(StringIO(output.getvalue())))

    assert data.x_label == "Time since first segmentation (timepoints)"
    assert row["measurement_key"] == "roi:object-1:ch1:intensity.mean"
    assert row["source_image_channel"] == "1"
    assert row["metric_key"] == "intensity.mean"
    assert row["algorithm_version"] == snapshot.algorithm_version
    assert '"document_id":"document-1"' in row["channel_metadata"]

    changed = SimpleNamespace(
        **{
            **document.__dict__,
            "objects": (
                SimpleNamespace(
                    **{
                        **track.__dict__,
                        "frames": {
                            1: SimpleNamespace(
                                **{
                                    **frame.__dict__,
                                    "geometry": SimpleNamespace(
                                        kind="polygon_2d",
                                        z_plane=1,
                                        exterior_xy_px=(
                                            (2, 1),
                                            (4, 1),
                                            (4, 3),
                                            (2, 3),
                                        ),
                                    ),
                                }
                            )
                        },
                    }
                ),
            ),
        }
    )
    stale_channel = RoiScalarSeriesChannel(
        snapshot,
        image_channel=0,
        metric_key="intensity.mean",
        source=changed,
        image_provider=provider,
    )
    assert stale_channel.read_with_reason(subject, 1).reason == "stale"


def test_temporal_subject_rejects_samples_outside_declared_span():
    try:
        TemporalSeriesSubject("object", "Object", 2, 3, (1, 2))
    except ValueError as error:
        assert "within" in str(error)
    else:
        raise AssertionError("Expected invalid temporal sample range to fail")
