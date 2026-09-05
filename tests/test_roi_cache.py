from dataclasses import replace
import weakref
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from acetree_py.analysis.roi_measurements import (
    RoiMeasurementCache,
    RoiMeasurementCancelled,
    RoiMeasurementEngine,
    RoiMeasurementRequest,
    RoiScalarSeriesChannel,
    geometry_fingerprint,
    roi_temporal_subject,
)
from acetree_py.analysis.expression_plot import TemporalSeriesService
from acetree_py.core.subcellular_roi import (
    CoordinateSpaceSnapshot,
    ContourSlice,
    ContourStack3D,
    ObjectClass,
    Polygon2D,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
)


class CountingProvider:
    num_timepoints = 2
    num_planes = 2
    num_channels = 2
    image_shape = (8, 8)

    def __init__(self):
        self.plane_reads = []
        self.stack_reads = []
        self.manifest_token = "images-v1"

    def get_plane(self, time, plane, channel=0):
        self.plane_reads.append((time, plane, channel))
        return np.full(self.image_shape, time + channel, dtype=float)

    def get_stack(self, time, channel=0):
        self.stack_reads.append((time, channel))
        return np.full((self.num_planes, *self.image_shape), time + channel, dtype=float)


def polygon(x_offset=0, *, z_plane=1):
    return SimpleNamespace(
        kind="polygon_2d",
        z_plane=z_plane,
        exterior_xy_px=((1 + x_offset, 1), (3 + x_offset, 1), (3 + x_offset, 3), (1 + x_offset, 3)),
    )


def document(geometry=None, revision=1, association=None):
    geometry = polygon() if geometry is None else geometry
    frame = SimpleNamespace(
        frame_id="frame-1",
        timepoint=1,
        presence="segmented",
        geometry=geometry,
        cell_ref=association,
    )
    track = SimpleNamespace(object_id="object-1", frames=MappingProxyType({1: frame}))
    return SimpleNamespace(
        document_id="document-1",
        roi_revision=revision,
        coordinate_space=SimpleNamespace(xy_res=1, z_res=2, plane_start=1),
        objects=(track,),
    )


def test_engine_groups_plane_reads_and_reuses_raw_cache():
    provider = CountingProvider()
    engine = RoiMeasurementEngine(provider)
    request = RoiMeasurementRequest(document=document(), channels=(0, 1))
    first = engine.measure(request)
    second = engine.measure(request)

    assert provider.plane_reads == [(1, 1, 0), (1, 1, 1)]
    assert first.value("object-1", 1, 0, "intensity.mean") == 1
    assert second.value("object-1", 1, 1, "intensity.mean") == 2
    assert engine.cache.stats.aggregate_hits == 2


def test_absolute_model_plane_is_converted_for_provider_and_manifest():
    class ManifestProvider(CountingProvider):
        def __init__(self):
            super().__init__()
            self.manifest_plane_requests = []

        def image_source_files(self, *, timepoints, planes):
            self.manifest_plane_requests.append((timepoints, planes))
            return ()

    provider = ManifestProvider()
    source = document(polygon(z_plane=5))
    source.coordinate_space = SimpleNamespace(xy_res=1, z_res=2, plane_start=5)

    snapshot = RoiMeasurementEngine(provider).measure(source, channels=(0,))

    assert provider.plane_reads == [(1, 1, 0)]
    assert snapshot.selected_planes == (5,)
    assert provider.manifest_plane_requests
    assert all(item[1] == (1,) for item in provider.manifest_plane_requests)


def test_association_change_does_not_invalidate_pixel_cache():
    provider = CountingProvider()
    engine = RoiMeasurementEngine(provider)
    engine.measure(document(association="cell-a"), channels=(0,))
    engine.invalidate_association("object-1", 1)
    engine.measure(document(association="cell-b", revision=2), channels=(0,))
    assert len(provider.plane_reads) == 1


def test_frame_invalidation_is_precise_and_geometry_fingerprint_changes():
    provider = CountingProvider()
    cache = RoiMeasurementCache(max_mask_bytes=10_000)
    engine = RoiMeasurementEngine(provider, cache=cache)
    original = polygon()
    changed = polygon(1)
    engine.measure(document(original), channels=(0,))
    old_fingerprint = geometry_fingerprint(original)
    assert geometry_fingerprint(changed) != old_fingerprint
    engine.invalidate_frame("object-1", 1)
    engine.measure(document(changed, revision=2), channels=(0,))
    assert len(provider.plane_reads) == 2


def test_cancellation_never_publishes_partial_snapshot():
    provider = CountingProvider()
    engine = RoiMeasurementEngine(provider)
    with pytest.raises(RoiMeasurementCancelled):
        engine.measure(
            document(),
            channels=(0, 1),
            progress_cb=lambda completed, total: completed < 1,
        )
    assert engine.latest_snapshot is None


def test_snapshot_freshness_strict_and_per_frame():
    provider = CountingProvider()
    source = document()
    snapshot = RoiMeasurementEngine(provider).measure(source, channels=(0,))
    association_only = document(revision=2, association="new-cell")
    assert not snapshot.is_current(association_only, image_provider=provider)
    assert snapshot.sample_is_current(
        association_only, "object-1", 1, image_provider=provider
    )
    provider.manifest_token = "images-v2"
    assert not snapshot.sample_is_current(
        association_only, "object-1", 1, image_provider=provider
    )


def test_blocked_physical_normalization_and_distribution_survive_snapshot():
    provider = CountingProvider()
    source_document = document()
    manager = SimpleNamespace(
        document=source_document,
        roi_revision=source_document.roi_revision,
        physical_normalization_blocked=True,
    )
    snapshot = RoiMeasurementEngine(provider).measure(
        manager,
        channels=(0,),
        include_distributions=True,
    )
    sample = snapshot.sample("object-1", 1, 0)

    assert sample.metric("intensity.mean").value == 1
    assert sample.metric("geometry.area_um2").reason == "calibration_mismatch"
    assert sample.distribution is not None
    assert sum(sample.distribution.histogram_counts) == sample.finite_sample_count
    assert snapshot.source_physical_normalization_blocked


def test_engine_consumes_the_frozen_domain_model_directly():
    provider = CountingProvider()
    object_class = ObjectClass(
        name="Golgi",
        color_rgba=(0.2, 0.4, 0.8, 1.0),
        next_instance_index=2,
    )
    frame = RoiFrameRecord(
        timepoint=1,
        presence=Presence.SEGMENTED,
        review_state=ReviewState.REVIEWED,
        geometry=Polygon2D(
            z_plane=1,
            exterior_xy_px=((1, 1), (3, 1), (3, 3), (1, 3)),
        ),
    )
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=1,
        frames={1: frame},
    )
    source = SubcellularRoiDocument(
        coordinate_space=CoordinateSpaceSnapshot(
            xy_res=1,
            z_res=2,
            image_width_px=8,
            image_height_px=8,
            plane_count=2,
            time_end=2,
        ),
        object_classes=(object_class,),
        objects=(track,),
    )

    snapshot = RoiMeasurementEngine(provider).measure(source, channels=(0,))
    assert snapshot.value(track.object_id, 1, 0, "intensity.mean") == 1

    expected_track = replace(track, expected_start_time=1, expected_end_time=3)
    subject = roi_temporal_subject(expected_track, object_class=object_class)
    channel = RoiScalarSeriesChannel(
        snapshot=snapshot,
        image_channel=0,
        metric_key="intensity.mean",
        label="Mean intensity",
        unit="a.u.",
    ).as_scalar_series_channel()
    series = TemporalSeriesService().build((subject,), channel).series[0]
    assert subject.label == "Golgi #1"
    assert subject.sample_times == (1, 2, 3)
    assert series.y_values == (1.0, None, None)
    assert series.missing_reasons == (None, "not_measured", "not_measured")


def _mixed_movie_document():
    object_class = ObjectClass("Golgi", (0.2, 0.4, 0.8, 1.0), next_instance_index=5)
    tracks = []
    for index in range(4):
        ring = polygon(index % 2).exterior_xy_px
        geometry = (
            Polygon2D(z_plane=1, exterior_xy_px=ring)
            if index < 2
            else ContourStack3D(tuple(ContourSlice(z, ring) for z in (1, 2)))
        )
        tracks.append(RoiObjectTrack(
            class_id=object_class.class_id,
            instance_index=index + 1,
            frames={time: RoiFrameRecord(
                timepoint=time,
                presence=Presence.SEGMENTED,
                review_state=ReviewState.REVIEWED,
                geometry=geometry,
            ) for time in (1, 2, 3)},
        ))
    return SubcellularRoiDocument(
        coordinate_space=CoordinateSpaceSnapshot(
            xy_res=1, z_res=2, image_width_px=8, image_height_px=8,
            plane_count=2, time_end=3,
        ),
        object_classes=(object_class,),
        objects=tuple(tracks),
    )


def test_movie_reuses_images_within_groups_and_releases_completed_groups():
    class LifetimeProvider(CountingProvider):
        num_timepoints = 3

        def __init__(self):
            super().__init__()
            self.image_refs = []
            self.retained_groups = []

        def record(self, image, time, channel):
            self.retained_groups.append({
                group for group, reference in self.image_refs if reference() is not None
            } | {(time, channel)})
            self.image_refs.append(((time, channel), weakref.ref(image)))
            return image

        def get_plane(self, time, plane, channel=0):
            return self.record(super().get_plane(time, plane, channel), time, channel)

        def get_stack(self, time, channel=0):
            return self.record(super().get_stack(time, channel), time, channel)

    provider = LifetimeProvider()
    source = _mixed_movie_document()
    snapshot = RoiMeasurementEngine(provider).measure(source)

    assert all(len(groups) == 1 for groups in provider.retained_groups)
    assert all(reference() is None for _, reference in provider.image_refs)
    assert len(provider.plane_reads) == len(provider.stack_reads) == 6
    for track in source.objects:
        for time in (1, 2, 3):
            for channel in (0, 1):
                assert snapshot.value(track.object_id, time, channel, "intensity.mean") == time + channel


def test_plot_and_export_validate_images_once_and_reject_later_changes(tmp_path):
    from acetree_py.core.roi_manager import RoiManager
    from acetree_py.gui.roi_scalar_plot_window import RoiScalarPlotController

    image_file = tmp_path / "movie.tif"
    image_file.write_bytes(b"original source")

    class ManifestProvider(CountingProvider):
        num_timepoints = 3

        def __init__(self):
            super().__init__()
            self.manifest_reads = 0

        def image_source_files(self, *, timepoints, planes):
            self.manifest_reads += 1
            return (image_file,)

    provider = ManifestProvider()
    manager = RoiManager(_mixed_movie_document())
    engine = RoiMeasurementEngine(provider)
    snapshot = engine.measure(manager)
    controller = RoiScalarPlotController(manager, snapshot, image_provider=provider)
    provider.manifest_reads = 0

    original = controller.build()
    assert provider.manifest_reads == 1
    assert all(series.y_values == (1.0, 2.0, 3.0) for series in original.series)
    controller.export_csv(tmp_path / "current.csv")
    assert provider.manifest_reads == 2
    manager.update_class(manager.classes[0].class_id, name="Renamed Golgi")
    assert controller.build().source_token == original.source_token
    assert provider.manifest_reads == 3

    context = snapshot.prepare_read(manager, image_provider=provider)
    track = manager.objects[0]
    assert context.sample_is_current(track.object_id, 1)
    manager.update_frame_geometry(track.object_id, 1, Polygon2D(
        z_plane=1, exterior_xy_px=polygon(2).exterior_xy_px,
    ))
    assert not context.sample_is_current(track.object_id, 1)
    changed = controller.build()
    assert changed.source_token != original.source_token
    assert controller.stale_samples(changed) == 1
    controller.update_snapshot(engine.measure(manager))
    controller.export_csv(tmp_path / "remeasured.csv")

    image_file.write_bytes(b"changed external image with a different size")
    provider.manifest_reads = 0
    with pytest.raises(RuntimeError, match="stale"):
        controller.export_csv(tmp_path / "stale.csv")
    assert provider.manifest_reads == 1
    assert not (tmp_path / "stale.csv").exists()
    # Direct readers still perform their own freshness check outside a build.
    direct = RoiScalarSeriesChannel(snapshot, 0, "intensity.mean", source=manager,
                                    image_provider=provider)
    assert direct.read_with_reason(track.object_id, 2).reason == "stale"
