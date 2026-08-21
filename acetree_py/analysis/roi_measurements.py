"""Immutable ROI measurement snapshots, bounded caches, and orchestration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import threading
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from .roi_measure import (
    ROI_MEASUREMENT_ALGORITHM_VERSION,
    RoiDistribution,
    RoiIntensityReducer,
    RoiIntensityResult,
    RoiMetricValue,
    RoiProfileSampler,
    RoiSpatialProfile,
)

if TYPE_CHECKING:
    from ..io.image_provider import ImageProvider
else:
    ImageProvider = Any
from .roi_rasterization import (
    ROI_RASTERIZATION_VERSION,
    RasterizedRoi,
    RoiCalibration,
    RoiMaskRasterizer,
    RoiRasterizationError,
    geometry_kind,
)


ROI_ANALYSIS_ALGORITHM_VERSION = (
    f"raster-{ROI_RASTERIZATION_VERSION}:measure-{ROI_MEASUREMENT_ALGORITHM_VERSION}"
)
RoiProgressCallback = Callable[[int, int], bool | None]


class RoiMeasurementCancelled(RuntimeError):
    """Raised when a bulk measurement is cancelled before publication."""


@dataclass(frozen=True, slots=True)
class RoiMeasurementRequest:
    """Selection and algorithm settings for an atomic measurement run."""

    document: Any
    object_ids: tuple[str, ...] | None = None
    timepoints: tuple[int, ...] | None = None
    channels: tuple[int, ...] | None = None
    metric_keys: tuple[str, ...] | None = None
    include_profiles: bool = False
    profile_step_um: float | None = None
    profile_width_step_um: float | None = None
    include_distributions: bool = False
    histogram_bins: int | tuple[float, ...] = 32
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    frame_dependency_token: str | None = None

    def __post_init__(self) -> None:
        if self.object_ids is not None:
            object.__setattr__(
                self, "object_ids", tuple(str(value) for value in self.object_ids)
            )
        if self.timepoints is not None:
            object.__setattr__(
                self, "timepoints", tuple(sorted({int(value) for value in self.timepoints}))
            )
        if self.channels is not None:
            object.__setattr__(
                self, "channels", tuple(sorted({int(value) for value in self.channels}))
            )
        if self.metric_keys is not None:
            object.__setattr__(self, "metric_keys", tuple(self.metric_keys))
        object.__setattr__(self, "quantiles", tuple(float(value) for value in self.quantiles))


@dataclass(frozen=True, slots=True)
class RoiMeasurementSample:
    """One ROI frame/channel result embedded in a published snapshot."""

    object_id: str
    frame_id: str
    timepoint: int
    image_channel: int
    geometry_fingerprint: str | None
    status: str
    metrics: Mapping[str, RoiMetricValue]
    nominal_sample_count: int = 0
    in_bounds_sample_count: int = 0
    finite_sample_count: int = 0
    coverage_fraction: float = 0.0
    clipped: bool = False
    warnings: tuple[str, ...] = ()
    distribution: RoiDistribution | None = None
    profile: RoiSpatialProfile | None = None
    profile_missing_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", str(self.object_id))
        object.__setattr__(self, "frame_id", str(self.frame_id))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @classmethod
    def from_result(
        cls,
        *,
        object_id: Any,
        frame_id: Any,
        timepoint: int,
        image_channel: int,
        fingerprint: str,
        result: RoiIntensityResult,
        profile: RoiSpatialProfile | None = None,
        profile_missing_reason: str | None = None,
    ) -> "RoiMeasurementSample":
        return cls(
            object_id=str(object_id),
            frame_id=str(frame_id),
            timepoint=int(timepoint),
            image_channel=int(image_channel),
            geometry_fingerprint=fingerprint,
            status=result.status,
            metrics=result.metrics,
            nominal_sample_count=result.nominal_sample_count,
            in_bounds_sample_count=result.in_bounds_sample_count,
            finite_sample_count=result.finite_sample_count,
            coverage_fraction=result.coverage_fraction,
            clipped=result.clipped,
            warnings=result.warnings,
            distribution=result.distribution,
            profile=profile,
            profile_missing_reason=profile_missing_reason,
        )

    def metric(self, metric_key: str) -> RoiMetricValue:
        return self.metrics.get(metric_key, RoiMetricValue(None, "metric_unavailable"))


@dataclass(frozen=True, slots=True)
class RoiMeasurementSnapshot:
    """Atomic, immutable, provenance-bound ROI measurement publication."""

    source_document_id: str
    source_roi_revision: int
    source_image_manifest_token: str
    source_calibration: RoiCalibration
    algorithm_version: str
    parameters_token: str
    channels: tuple[int, ...]
    samples: Mapping[tuple[str, int, int], RoiMeasurementSample]
    frame_fingerprints: Mapping[tuple[str, int], str | None]
    selected_timepoints: tuple[int, ...] = ()
    selected_planes: tuple[int, ...] = ()
    source_physical_normalization_blocked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_document_id", str(self.source_document_id))
        object.__setattr__(self, "channels", tuple(int(value) for value in self.channels))
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))
        object.__setattr__(
            self,
            "frame_fingerprints",
            MappingProxyType(dict(self.frame_fingerprints)),
        )
        object.__setattr__(
            self, "selected_timepoints", tuple(int(value) for value in self.selected_timepoints)
        )
        object.__setattr__(
            self, "selected_planes", tuple(int(value) for value in self.selected_planes)
        )

    @property
    def document_id(self) -> str:
        return self.source_document_id

    @property
    def roi_revision(self) -> int:
        return self.source_roi_revision

    @property
    def image_manifest_token(self) -> str:
        return self.source_image_manifest_token

    @property
    def calibration(self) -> RoiCalibration:
        return self.source_calibration

    @property
    def source_token(self) -> str:
        payload = "|".join(
            (
                self.source_document_id,
                str(self.source_roi_revision),
                self.source_image_manifest_token,
                repr(self.source_calibration.cache_key()),
                self.algorithm_version,
                self.parameters_token,
                repr(self.channels),
                repr(self.selected_timepoints),
                repr(self.selected_planes),
                repr(self.source_physical_normalization_blocked),
                repr(tuple(sorted(self.frame_fingerprints.items()))),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def sample(
        self,
        object_id: Any,
        timepoint: int,
        image_channel: int,
    ) -> RoiMeasurementSample | None:
        return self.samples.get((str(object_id), int(timepoint), int(image_channel)))

    def metric(
        self,
        object_id: Any,
        timepoint: int,
        image_channel: int,
        metric_key: str,
    ) -> RoiMetricValue:
        sample = self.sample(object_id, timepoint, image_channel)
        if sample is None:
            return RoiMetricValue(None, "not_measured")
        return sample.metric(metric_key)

    def value(
        self,
        object_id: Any,
        timepoint: int,
        image_channel: int,
        metric_key: str,
    ) -> float | None:
        return self.metric(object_id, timepoint, image_channel, metric_key).value

    def missing_reason(
        self,
        object_id: Any,
        timepoint: int,
        image_channel: int,
        metric_key: str,
    ) -> str | None:
        return self.metric(object_id, timepoint, image_channel, metric_key).reason

    def is_current(
        self,
        source: Any,
        *,
        image_provider: ImageProvider | None = None,
        calibration: Any | None = None,
    ) -> bool:
        document = _source_document(source)
        if str(_field(document, "document_id", "")) != self.source_document_id:
            return False
        if _source_revision(source, document) != self.source_roi_revision:
            return False
        current_calibration = _source_calibration(source, document, calibration)
        if current_calibration != self.source_calibration:
            return False
        if (
            _source_physical_normalization_blocked(source)
            != self.source_physical_normalization_blocked
        ):
            return False
        if image_provider is not None:
            times = self.selected_timepoints
            planes = self.selected_planes
            if _image_token(
                image_provider,
                times,
                planes,
                plane_start=self.source_calibration.plane_start,
            ) != self.source_image_manifest_token:
                return False
        return True

    def sample_is_current(
        self,
        source: Any,
        object_id: Any,
        timepoint: int,
        *,
        image_provider: ImageProvider | None = None,
        calibration: Any | None = None,
    ) -> bool:
        """Check only dependencies of one frame, ignoring unrelated ROI edits."""

        document = _source_document(source)
        if str(_field(document, "document_id", "")) != self.source_document_id:
            return False
        if _source_calibration(source, document, calibration) != self.source_calibration:
            return False
        if (
            _source_physical_normalization_blocked(source)
            != self.source_physical_normalization_blocked
        ):
            return False
        if image_provider is not None:
            times = self.selected_timepoints
            planes = self.selected_planes
            if _image_token(
                image_provider,
                times,
                planes,
                plane_start=self.source_calibration.plane_start,
            ) != self.source_image_manifest_token:
                return False
        frame = _find_frame(document, str(object_id), int(timepoint))
        current_fingerprint = None
        if frame is not None and _field(frame, "geometry", None) is not None:
            current_fingerprint = geometry_fingerprint(_field(frame, "geometry"))
        return self.frame_fingerprints.get((str(object_id), int(timepoint))) == current_fingerprint

    def current_source_token(
        self,
        source: Any,
        *,
        image_provider: ImageProvider | None = None,
        calibration: Any | None = None,
    ) -> str:
        """Return a precise token for the dependencies covered by this snapshot.

        Association, class-name, and display-index edits intentionally do not
        affect the token. Geometry, calibration acknowledgement, selected
        image files, and algorithm inputs do.
        """

        document = _source_document(source)
        current_calibration = _source_calibration(source, document, calibration)
        current_fingerprints: list[tuple[tuple[str, int], str | None]] = []
        for frame_key in self.frame_fingerprints:
            frame = _find_frame(document, frame_key[0], frame_key[1])
            geometry = None if frame is None else _field(frame, "geometry", None)
            current_fingerprints.append(
                (
                    frame_key,
                    None if geometry is None else geometry_fingerprint(geometry),
                )
            )
        image_token = self.source_image_manifest_token
        if image_provider is not None:
            image_token = _image_token(
                image_provider,
                self.selected_timepoints,
                self.selected_planes,
                plane_start=current_calibration.plane_start,
            )
        payload = (
            str(_field(document, "document_id", "")),
            current_calibration.cache_key(),
            _source_physical_normalization_blocked(source),
            image_token,
            self.algorithm_version,
            self.parameters_token,
            self.channels,
            tuple(sorted(current_fingerprints)),
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


class RoiMeasurementStore:
    """Thread-safe holder that publishes only complete immutable snapshots."""

    def __init__(self) -> None:
        self._snapshot: RoiMeasurementSnapshot | None = None
        self._lock = threading.RLock()

    @property
    def snapshot(self) -> RoiMeasurementSnapshot | None:
        with self._lock:
            return self._snapshot

    def publish(self, snapshot: RoiMeasurementSnapshot) -> None:
        if not isinstance(snapshot, RoiMeasurementSnapshot):
            raise TypeError("Only RoiMeasurementSnapshot values may be published")
        with self._lock:
            self._snapshot = snapshot

    def clear(self) -> None:
        with self._lock:
            self._snapshot = None


@dataclass(frozen=True, slots=True)
class RoiCacheStats:
    mask_entries: int
    aggregate_entries: int
    mask_bytes: int
    aggregate_bytes: int
    mask_hits: int
    mask_misses: int
    aggregate_hits: int
    aggregate_misses: int


class RoiMeasurementCache:
    """Byte-bounded LRU caches with frame-precise invalidation indexes."""

    def __init__(
        self,
        *,
        max_mask_bytes: int = 64 * 1024 * 1024,
        max_aggregate_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if max_mask_bytes < 0 or max_aggregate_bytes < 0:
            raise ValueError("ROI cache byte limits cannot be negative")
        self.max_mask_bytes = int(max_mask_bytes)
        self.max_aggregate_bytes = int(max_aggregate_bytes)
        self._masks: OrderedDict[Any, RasterizedRoi] = OrderedDict()
        self._aggregates: OrderedDict[Any, RoiIntensityResult] = OrderedDict()
        self._mask_sizes: dict[Any, int] = {}
        self._aggregate_sizes: dict[Any, int] = {}
        self._mask_frames: dict[Any, set[tuple[str, int]]] = defaultdict(set)
        self._aggregate_frames: dict[Any, set[tuple[str, int]]] = defaultdict(set)
        self._frame_masks: dict[tuple[str, int], set[Any]] = defaultdict(set)
        self._frame_aggregates: dict[tuple[str, int], set[Any]] = defaultdict(set)
        self._mask_bytes = 0
        self._aggregate_bytes = 0
        self._mask_hits = 0
        self._mask_misses = 0
        self._aggregate_hits = 0
        self._aggregate_misses = 0
        self._lock = threading.RLock()

    def get_mask(
        self,
        key: Any,
        *,
        frame_key: tuple[str, int] | None = None,
    ) -> RasterizedRoi | None:
        with self._lock:
            value = self._masks.get(key)
            if value is None:
                self._mask_misses += 1
                return None
            self._mask_hits += 1
            self._masks.move_to_end(key)
            if frame_key is not None:
                self._link(self._mask_frames, self._frame_masks, key, frame_key)
            return value

    def put_mask(
        self,
        key: Any,
        value: RasterizedRoi,
        *,
        frame_key: tuple[str, int] | None = None,
    ) -> None:
        size = int(value.mask.nbytes)
        if value.parent_mask is not None:
            size += int(value.parent_mask.nbytes)
        with self._lock:
            self._remove_mask(key)
            if size <= self.max_mask_bytes:
                self._masks[key] = value
                self._mask_sizes[key] = size
                self._mask_bytes += size
                if frame_key is not None:
                    self._link(self._mask_frames, self._frame_masks, key, frame_key)
                while self._mask_bytes > self.max_mask_bytes and self._masks:
                    self._remove_mask(next(iter(self._masks)))

    def get_aggregate(
        self,
        key: Any,
        *,
        frame_key: tuple[str, int] | None = None,
    ) -> RoiIntensityResult | None:
        with self._lock:
            value = self._aggregates.get(key)
            if value is None:
                self._aggregate_misses += 1
                return None
            self._aggregate_hits += 1
            self._aggregates.move_to_end(key)
            if frame_key is not None:
                self._link(
                    self._aggregate_frames,
                    self._frame_aggregates,
                    key,
                    frame_key,
                )
            return value

    def put_aggregate(
        self,
        key: Any,
        value: RoiIntensityResult,
        *,
        frame_key: tuple[str, int] | None = None,
    ) -> None:
        size = _aggregate_size(value)
        with self._lock:
            self._remove_aggregate(key)
            if size <= self.max_aggregate_bytes:
                self._aggregates[key] = value
                self._aggregate_sizes[key] = size
                self._aggregate_bytes += size
                if frame_key is not None:
                    self._link(
                        self._aggregate_frames,
                        self._frame_aggregates,
                        key,
                        frame_key,
                    )
                while self._aggregate_bytes > self.max_aggregate_bytes and self._aggregates:
                    self._remove_aggregate(next(iter(self._aggregates)))

    def invalidate_frame(self, object_id: Any, timepoint: int) -> None:
        frame_key = (str(object_id), int(timepoint))
        with self._lock:
            for key in tuple(self._frame_masks.pop(frame_key, ())):
                frames = self._mask_frames.get(key)
                if frames is not None:
                    frames.discard(frame_key)
                    if not frames:
                        self._remove_mask(key)
            for key in tuple(self._frame_aggregates.pop(frame_key, ())):
                frames = self._aggregate_frames.get(key)
                if frames is not None:
                    frames.discard(frame_key)
                    if not frames:
                        self._remove_aggregate(key)

    def invalidate_geometry(self, fingerprint: str) -> None:
        with self._lock:
            for key in tuple(self._masks):
                if fingerprint in key:
                    self._remove_mask(key)
            for key in tuple(self._aggregates):
                if fingerprint in key:
                    self._remove_aggregate(key)

    def invalidate_association(self, object_id: Any, timepoint: int) -> None:
        """Association changes intentionally do not invalidate pixel work."""

    def invalidate_image_source(self) -> None:
        with self._lock:
            for key in tuple(self._aggregates):
                self._remove_aggregate(key)

    def invalidate_calibration(self) -> None:
        self.clear()

    def invalidate_algorithm(self) -> None:
        self.clear()

    def clear(self) -> None:
        with self._lock:
            self._masks.clear()
            self._aggregates.clear()
            self._mask_sizes.clear()
            self._aggregate_sizes.clear()
            self._mask_frames.clear()
            self._aggregate_frames.clear()
            self._frame_masks.clear()
            self._frame_aggregates.clear()
            self._mask_bytes = 0
            self._aggregate_bytes = 0

    @property
    def stats(self) -> RoiCacheStats:
        with self._lock:
            return RoiCacheStats(
                mask_entries=len(self._masks),
                aggregate_entries=len(self._aggregates),
                mask_bytes=self._mask_bytes,
                aggregate_bytes=self._aggregate_bytes,
                mask_hits=self._mask_hits,
                mask_misses=self._mask_misses,
                aggregate_hits=self._aggregate_hits,
                aggregate_misses=self._aggregate_misses,
            )

    @staticmethod
    def _link(
        forward: dict[Any, set[tuple[str, int]]],
        reverse: dict[tuple[str, int], set[Any]],
        key: Any,
        frame_key: tuple[str, int],
    ) -> None:
        forward[key].add(frame_key)
        reverse[frame_key].add(key)

    def _remove_mask(self, key: Any) -> None:
        if key in self._masks:
            self._masks.pop(key)
            self._mask_bytes -= self._mask_sizes.pop(key, 0)
        for frame_key in self._mask_frames.pop(key, ()):
            keys = self._frame_masks.get(frame_key)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._frame_masks.pop(frame_key, None)

    def _remove_aggregate(self, key: Any) -> None:
        if key in self._aggregates:
            self._aggregates.pop(key)
            self._aggregate_bytes -= self._aggregate_sizes.pop(key, 0)
        for frame_key in self._aggregate_frames.pop(key, ()):
            keys = self._frame_aggregates.get(frame_key)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    self._frame_aggregates.pop(frame_key, None)


class RoiMeasurementEngine:
    """Time-major, provider-efficient, atomic ROI measurement service."""

    def __init__(
        self,
        image_provider: ImageProvider | None = None,
        *,
        store: RoiMeasurementStore | None = None,
        cache: RoiMeasurementCache | None = None,
        max_mask_cache_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.image_provider = image_provider
        self.store = store if store is not None else RoiMeasurementStore()
        self.cache = cache if cache is not None else RoiMeasurementCache(
            max_mask_bytes=max_mask_cache_bytes
        )
        self.rasterizer = RoiMaskRasterizer()
        self.reducer = RoiIntensityReducer()
        self.profile_sampler = RoiProfileSampler()

    @property
    def latest_snapshot(self) -> RoiMeasurementSnapshot | None:
        return self.store.snapshot

    def measure(
        self,
        request_or_document: RoiMeasurementRequest | Any,
        image_provider: ImageProvider | None = None,
        *,
        calibration: Any | None = None,
        object_ids: Iterable[Any] | None = None,
        timepoints: Iterable[int] | None = None,
        channels: Iterable[int] | None = None,
        metric_keys: Iterable[str] | None = None,
        include_profiles: bool = False,
        profile_step_um: float | None = None,
        profile_width_step_um: float | None = None,
        include_distributions: bool = False,
        histogram_bins: int | Sequence[float] = 32,
        quantiles: Iterable[float] = (0.25, 0.5, 0.75),
        progress_cb: RoiProgressCallback | None = None,
    ) -> RoiMeasurementSnapshot:
        provider = image_provider if image_provider is not None else self.image_provider
        if provider is None:
            raise ValueError("An image provider is required")
        if isinstance(request_or_document, RoiMeasurementRequest):
            request = request_or_document
        else:
            request = RoiMeasurementRequest(
                document=request_or_document,
                object_ids=None if object_ids is None else tuple(str(v) for v in object_ids),
                timepoints=None if timepoints is None else tuple(int(v) for v in timepoints),
                channels=None if channels is None else tuple(int(v) for v in channels),
                metric_keys=None if metric_keys is None else tuple(metric_keys),
                include_profiles=include_profiles,
                profile_step_um=profile_step_um,
                profile_width_step_um=profile_width_step_um,
                include_distributions=include_distributions,
                histogram_bins=(
                    int(histogram_bins)
                    if isinstance(histogram_bins, int)
                    else tuple(float(v) for v in histogram_bins)
                ),
                quantiles=tuple(float(v) for v in quantiles),
            )
        source = request.document
        document = _source_document(source)
        cal = _source_calibration(source, document, calibration)
        normalization_blocked_before = _source_physical_normalization_blocked(source)
        revision_before = _source_revision(source, document)
        document_id = str(_field(document, "document_id", ""))
        selected_frames = _select_frames(document, request)
        selected_times = sorted({item[2] for item in selected_frames})
        selected_planes = _selected_planes(selected_frames)
        selected_channels = request.channels
        if selected_channels is None:
            selected_channels = tuple(range(int(provider.num_channels)))
        if not selected_channels:
            raise ValueError("At least one image channel must be selected")
        image_token_before = _image_token(
            provider,
            selected_times,
            selected_planes,
            plane_start=cal.plane_start,
        )
        parameters_token = _parameters_token(request)
        samples: dict[tuple[str, int, int], RoiMeasurementSample] = {}
        frame_fingerprints: dict[tuple[str, int], str | None] = {}
        plane_cache: dict[tuple[int, int, int], Any] = {}
        stack_cache: dict[tuple[int, int], Any] = {}
        tasks = [
            (track, frame, timepoint, channel)
            for track, frame, timepoint in selected_frames
            for channel in selected_channels
        ]
        tasks.sort(key=lambda item: (item[2], item[3], str(_field(item[0], "object_id"))))
        total_tasks = len(tasks)
        for completed, (track, frame, timepoint, channel) in enumerate(tasks, start=1):
            object_id = str(_field(track, "object_id"))
            frame_id = str(_field(frame, "frame_id", f"{object_id}:{timepoint}"))
            frame_key = (object_id, timepoint)
            geometry = _field(frame, "geometry", None)
            presence = str(_enum_value(_field(frame, "presence", "segmented")))
            fingerprint = None if geometry is None else geometry_fingerprint(geometry)
            frame_fingerprints[frame_key] = fingerprint
            if presence == "absent":
                sample = _missing_sample(
                    object_id, frame_id, timepoint, channel, None, "roi_absent"
                )
            elif geometry is None:
                sample = _missing_sample(
                    object_id, frame_id, timepoint, channel, None, "invalid_geometry"
                )
            elif channel < 0 or channel >= int(provider.num_channels):
                sample = _missing_sample(
                    object_id,
                    frame_id,
                    timepoint,
                    channel,
                    fingerprint,
                    "channel_unavailable",
                )
            else:
                sample = self._measure_one(
                    provider=provider,
                    geometry=geometry,
                    object_id=object_id,
                    frame_id=frame_id,
                    timepoint=timepoint,
                    channel=channel,
                    fingerprint=fingerprint,
                    calibration=cal,
                    image_token=image_token_before,
                    request=request,
                    physical_normalization_blocked=normalization_blocked_before,
                    parameters_token=parameters_token,
                    plane_cache=plane_cache,
                    stack_cache=stack_cache,
                )
            samples[(object_id, timepoint, channel)] = sample
            if progress_cb is not None and progress_cb(completed, total_tasks) is False:
                raise RoiMeasurementCancelled(
                    "ROI measurement cancelled; no partial snapshot was published"
                )

        image_token_after = _image_token(
            provider,
            selected_times,
            selected_planes,
            plane_start=cal.plane_start,
        )
        if image_token_after != image_token_before:
            raise RuntimeError(
                "Image source changed during ROI measurement; no snapshot was published"
            )
        if _source_revision(source, _source_document(source)) != revision_before:
            raise RuntimeError(
                "ROI document changed during measurement; no snapshot was published"
            )
        if _source_physical_normalization_blocked(source) != normalization_blocked_before:
            raise RuntimeError(
                "ROI calibration acknowledgement changed during measurement; "
                "no snapshot was published"
            )
        # A source can be mutated without following the manager revision
        # protocol.  Geometry fingerprints make publication fail closed too.
        for track, frame, timepoint in _select_frames(_source_document(source), request):
            frame_key = (str(_field(track, "object_id")), timepoint)
            geometry = _field(frame, "geometry", None)
            current = None if geometry is None else geometry_fingerprint(geometry)
            if frame_fingerprints.get(frame_key) != current:
                raise RuntimeError(
                    "ROI geometry changed during measurement; no snapshot was published"
                )
        snapshot = RoiMeasurementSnapshot(
            source_document_id=document_id,
            source_roi_revision=revision_before,
            source_image_manifest_token=image_token_before,
            source_calibration=cal,
            algorithm_version=ROI_ANALYSIS_ALGORITHM_VERSION,
            parameters_token=parameters_token,
            channels=tuple(selected_channels),
            samples=samples,
            frame_fingerprints=frame_fingerprints,
            selected_timepoints=tuple(selected_times),
            selected_planes=tuple(selected_planes),
            source_physical_normalization_blocked=normalization_blocked_before,
        )
        self.store.publish(snapshot)
        return snapshot

    run = measure

    def _measure_one(
        self,
        *,
        provider: ImageProvider,
        geometry: Any,
        object_id: str,
        frame_id: str,
        timepoint: int,
        channel: int,
        fingerprint: str,
        calibration: RoiCalibration,
        image_token: str,
        request: RoiMeasurementRequest,
        physical_normalization_blocked: bool,
        parameters_token: str,
        plane_cache: dict[tuple[int, int, int], Any],
        stack_cache: dict[tuple[int, int], Any],
    ) -> RoiMeasurementSample:
        frame_key = (object_id, timepoint)
        kind = geometry_kind(geometry)
        is_3d = kind == "contour_stack_3d"
        image_shape = (
            (int(provider.num_planes), *tuple(int(v) for v in provider.image_shape))
            if is_3d
            else tuple(int(v) for v in provider.image_shape)
        )
        mask_key = (
            "mask",
            ROI_RASTERIZATION_VERSION,
            fingerprint,
            image_shape,
            calibration.cache_key(),
        )
        try:
            raster = self.cache.get_mask(mask_key, frame_key=frame_key)
            if raster is None:
                raster = self.rasterizer.rasterize(
                    geometry,
                    image_shape,
                    calibration=calibration,
                )
                self.cache.put_mask(mask_key, raster, frame_key=frame_key)
        except (RoiRasterizationError, ValueError, MemoryError) as exc:
            reason = getattr(exc, "reason", "invalid_geometry")
            return _missing_sample(
                object_id, frame_id, timepoint, channel, fingerprint, reason
            )
        aggregate_key = (
            "aggregate",
            ROI_ANALYSIS_ALGORITHM_VERSION,
            fingerprint,
            image_token,
            timepoint,
            channel,
            calibration.cache_key(),
            physical_normalization_blocked,
            parameters_token,
        )
        result = self.cache.get_aggregate(aggregate_key, frame_key=frame_key)
        profile = None
        profile_reason = None
        if result is not None and not (
            request.include_profiles and kind == "thick_polyline_2d"
        ):
            return RoiMeasurementSample.from_result(
                object_id=object_id,
                frame_id=frame_id,
                timepoint=timepoint,
                image_channel=channel,
                fingerprint=fingerprint,
                result=result,
            )
        try:
            if is_3d:
                stack_key = (timepoint, channel)
                if stack_key not in stack_cache:
                    stack_cache[stack_key] = provider.get_stack(timepoint, channel)
                image = stack_cache[stack_key]
            else:
                absolute_plane = int(_field(geometry, "z_plane"))
                plane_key = (timepoint, absolute_plane, channel)
                if plane_key not in plane_cache:
                    provider_plane = absolute_plane - calibration.plane_start + 1
                    if provider_plane < 1:
                        raise IndexError(
                            f"ROI plane {absolute_plane} precedes plane_start "
                            f"{calibration.plane_start}"
                        )
                    plane_cache[plane_key] = provider.get_plane(
                        timepoint,
                        provider_plane,
                        channel,
                    )
                image = plane_cache[plane_key]
            if result is None:
                result = self.reducer.reduce(
                    image,
                    raster,
                    geometry=geometry,
                    calibration=calibration,
                    physical_normalization_blocked=physical_normalization_blocked,
                    include_distribution=request.include_distributions,
                    histogram_bins=request.histogram_bins,
                    quantiles=request.quantiles,
                )
                if request.metric_keys is not None:
                    result = _filter_metrics(result, request.metric_keys)
                self.cache.put_aggregate(
                    aggregate_key,
                    result,
                    frame_key=frame_key,
                )
            if request.include_profiles and kind == "thick_polyline_2d":
                profile = self.profile_sampler.sample(
                    image,
                    geometry,
                    calibration=calibration,
                    step_um=request.profile_step_um,
                    width_step_um=request.profile_width_step_um,
                )
        except (FileNotFoundError, OSError, IndexError, KeyError, ValueError) as exc:
            if result is None:
                return _missing_sample(
                    object_id,
                    frame_id,
                    timepoint,
                    channel,
                    fingerprint,
                    "image_unavailable",
                    warning=str(exc),
                )
            profile_reason = "image_unavailable"
        return RoiMeasurementSample.from_result(
            object_id=object_id,
            frame_id=frame_id,
            timepoint=timepoint,
            image_channel=channel,
            fingerprint=fingerprint,
            result=result,
            profile=profile,
            profile_missing_reason=profile_reason,
        )

    def invalidate_frame(self, object_id: Any, timepoint: int) -> None:
        self.cache.invalidate_frame(object_id, timepoint)

    def invalidate_geometry(self, fingerprint: str) -> None:
        self.cache.invalidate_geometry(fingerprint)

    def invalidate_association(self, object_id: Any, timepoint: int) -> None:
        self.cache.invalidate_association(object_id, timepoint)

    def invalidate_image_source(self) -> None:
        self.cache.invalidate_image_source()

    def invalidate_calibration(self) -> None:
        self.cache.invalidate_calibration()

    def invalidate_algorithm(self) -> None:
        self.cache.invalidate_algorithm()


@dataclass(frozen=True, slots=True)
class RoiScalarSeriesChannel:
    """ROI-specific scalar reader for a generalized temporal-series UI."""

    snapshot: RoiMeasurementSnapshot
    image_channel: int
    metric_key: str
    label: str | None = None
    unit: str | None = None
    source: Any | None = None
    image_provider: ImageProvider | None = None

    @property
    def key(self) -> str:
        return f"roi:ch{self.image_channel + 1}:{self.metric_key}"

    def measurement_key(self, object_id: Any) -> str:
        return roi_metric_key(object_id, self.image_channel, self.metric_key)

    def read_with_reason(self, subject: Any, timepoint: int) -> RoiMetricValue:
        object_id = _subject_object_id(subject)
        if self.source is not None and not self.snapshot.sample_is_current(
            self.source,
            object_id,
            int(timepoint),
            image_provider=self.image_provider,
        ):
            return RoiMetricValue(None, "stale")
        return self.snapshot.metric(
            object_id,
            int(timepoint),
            self.image_channel,
            self.metric_key,
        )

    def reader(self, subject: Any, timepoint: int) -> float | None:
        return self.read_with_reason(subject, timepoint).value

    read = reader

    def source_token(self) -> str:
        if self.source is None:
            return self.snapshot.source_token
        return self.snapshot.current_source_token(
            self.source,
            image_provider=self.image_provider,
        )

    def validate_coverage(self, subjects: Iterable[Any]) -> tuple[str, ...]:
        missing: list[str] = []
        for subject in subjects:
            object_id = _subject_object_id(subject)
            sample_times = tuple(_field(subject, "sample_times", ()))
            has_valid = any(
                self.read_with_reason(subject, int(timepoint)).is_valid
                for timepoint in sample_times
            )
            if not has_valid:
                missing.append(object_id)
        return tuple(missing)

    def as_scalar_series_channel(self):
        """Adapt this ROI reader to the generic temporal-series contract."""

        from .expression_plot import ScalarSeriesChannel

        return ScalarSeriesChannel(
            key=self.key,
            label=self.label or self.metric_key,
            unit=self.unit or "",
            reader=self.read_with_reason,
            source_token=self.source_token,
            validate_coverage=self.validate_coverage,
            measurement_key=lambda subject: self.measurement_key(
                _subject_object_id(subject)
            ),
            metadata={
                "source_image_channel": self.image_channel + 1,
                "metric_key": self.metric_key,
                "algorithm_version": self.snapshot.algorithm_version,
                "document_id": self.snapshot.source_document_id,
                "roi_revision": self.snapshot.source_roi_revision,
                "parameters_token": self.snapshot.parameters_token,
                "physical_normalization_blocked": (
                    self.snapshot.source_physical_normalization_blocked
                ),
                "calibration": {
                    "xy_res": self.snapshot.source_calibration.xy_res,
                    "z_res": self.snapshot.source_calibration.z_res,
                    "plane_start": self.snapshot.source_calibration.plane_start,
                },
            },
        )


def roi_temporal_subject(
    track: Any,
    *,
    object_class: Any | None = None,
):
    """Build one generic temporal subject, retaining expected-span gaps."""

    from .expression_plot import TemporalSeriesSubject

    frames = _field(track, "frames", {})
    if isinstance(frames, Mapping):
        frame_times = tuple(sorted(int(value) for value in frames))
    else:
        frame_times = tuple(
            sorted(int(_field(frame, "timepoint")) for frame in frames)
        )
    expected_start = _field(track, "expected_start_time", None)
    expected_end = _field(track, "expected_end_time", None)
    start_time = int(
        expected_start
        if expected_start is not None
        else (frame_times[0] if frame_times else 1)
    )
    end_time = int(
        expected_end
        if expected_end is not None
        else (frame_times[-1] if frame_times else start_time)
    )
    if end_time < start_time:
        raise ValueError("An ROI track's expected end cannot precede its start")
    sample_times = tuple(range(start_time, end_time + 1))
    object_id = str(_field(track, "object_id"))
    class_id = str(_field(track, "class_id"))
    instance_index = int(_field(track, "instance_index"))
    class_name = (
        str(_field(object_class, "name"))
        if object_class is not None
        else "Object"
    )
    return TemporalSeriesSubject(
        key=object_id,
        label=f"{class_name} #{instance_index}",
        start_time=start_time,
        end_time=end_time,
        sample_times=sample_times,
        metadata={
            "object_id": object_id,
            "class_id": class_id,
            "class_name": class_name,
            "instance_index": instance_index,
            "segmented_timepoints": frame_times,
            "expected_start_time": expected_start,
            "expected_end_time": expected_end,
        },
    )


def roi_temporal_subjects(
    tracks: Iterable[Any],
    *,
    object_classes: Iterable[Any] = (),
) -> tuple[Any, ...]:
    """Build temporal subjects for tracks using UUID-matched class metadata."""

    classes = {
        str(_field(object_class, "class_id")): object_class
        for object_class in object_classes
    }
    return tuple(
        roi_temporal_subject(
            track,
            object_class=classes.get(str(_field(track, "class_id"))),
        )
        for track in tracks
    )


def roi_metric_key(object_id: Any, image_channel: int, metric_key: str) -> str:
    """Return the stable UUID-based key used in plots and exports."""

    return f"roi:{object_id}:ch{int(image_channel) + 1}:{metric_key}"


def geometry_fingerprint(geometry: Any) -> str:
    """Canonical SHA-256 fingerprint of geometry-only scientific inputs."""

    payload = json.dumps(
        _canonical_value(geometry),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _missing_sample(
    object_id: str,
    frame_id: str,
    timepoint: int,
    channel: int,
    fingerprint: str | None,
    reason: str,
    *,
    warning: str | None = None,
) -> RoiMeasurementSample:
    return RoiMeasurementSample(
        object_id=object_id,
        frame_id=frame_id,
        timepoint=timepoint,
        image_channel=channel,
        geometry_fingerprint=fingerprint,
        status=reason,
        metrics={
            key: RoiMetricValue(None, reason, unit)
            for key, unit in (
                ("intensity.sum", "a.u."),
                ("intensity.mean", "a.u."),
                ("intensity.median", "a.u."),
                ("intensity.sum_per_length_um", "a.u./um"),
                ("intensity.sum_per_area_um2", "a.u./um2"),
                ("intensity.sum_per_volume_um3", "a.u./um3"),
                ("intensity.sum_per_surface_area_um2", "a.u./um2"),
                ("geometry.length_um", "um"),
                ("geometry.area_um2", "um2"),
                ("geometry.volume_um3", "um3"),
                ("geometry.surface_area_um2", "um2"),
            )
        },
        warnings=() if warning is None else (warning,),
    )


def _filter_metrics(
    result: RoiIntensityResult,
    metric_keys: Iterable[str],
) -> RoiIntensityResult:
    selected = tuple(metric_keys)
    return dataclasses.replace(
        result,
        metrics={key: result.metric(key) for key in selected},
    )


def _select_frames(
    document: Any,
    request: RoiMeasurementRequest,
) -> list[tuple[Any, Any, int]]:
    object_filter = None if request.object_ids is None else set(request.object_ids)
    time_filter = None if request.timepoints is None else set(request.timepoints)
    selected: list[tuple[Any, Any, int]] = []
    objects = _field(document, "objects", ())
    if isinstance(objects, Mapping):
        objects = objects.values()
    for track in objects:
        object_id = str(_field(track, "object_id"))
        if object_filter is not None and object_id not in object_filter:
            continue
        frames = _field(track, "frames", {})
        iterable = frames.items() if isinstance(frames, Mapping) else (
            (int(_field(frame, "timepoint")), frame) for frame in frames
        )
        for timepoint, frame in iterable:
            timepoint = int(timepoint)
            if time_filter is not None and timepoint not in time_filter:
                continue
            selected.append((track, frame, timepoint))
    selected.sort(key=lambda item: (item[2], str(_field(item[0], "object_id"))))
    return selected


def _find_frame(document: Any, object_id: str, timepoint: int) -> Any | None:
    objects = _field(document, "objects", ())
    if isinstance(objects, Mapping):
        objects = objects.values()
    for track in objects:
        if str(_field(track, "object_id")) != object_id:
            continue
        frames = _field(track, "frames", {})
        if isinstance(frames, Mapping):
            return frames.get(timepoint)
        for frame in frames:
            if int(_field(frame, "timepoint")) == timepoint:
                return frame
    return None


def _selected_planes(selected_frames: Iterable[tuple[Any, Any, int]]) -> tuple[int, ...]:
    planes: set[int] = set()
    for _track, frame, _timepoint in selected_frames:
        geometry = _field(frame, "geometry", None)
        if geometry is None:
            continue
        kind = geometry_kind(geometry)
        if kind == "contour_stack_3d":
            planes.update(int(_field(value, "z_plane")) for value in _field(geometry, "slices", ()))
        else:
            planes.add(int(_field(geometry, "z_plane")))
    return tuple(sorted(planes))


def _document_planes(document: Any, *, times: Iterable[int]) -> tuple[int, ...]:
    request = RoiMeasurementRequest(document=document, timepoints=tuple(times))
    return _selected_planes(_select_frames(document, request))


def _source_document(source: Any) -> Any:
    document = _field(source, "document", None)
    return source if document is None else document


def _source_revision(source: Any, document: Any) -> int:
    value = _field(source, "roi_revision", None)
    if value is None:
        value = _field(document, "roi_revision", 0)
    return int(value)


def _source_physical_normalization_blocked(source: Any) -> bool:
    return bool(_field(source, "physical_normalization_blocked", False))


def _source_calibration(
    source: Any,
    document: Any,
    explicit: Any | None,
) -> RoiCalibration:
    if explicit is not None:
        return RoiCalibration.from_value(explicit)
    coordinate_space = _field(document, "coordinate_space", None)
    if coordinate_space is not None:
        calibration = RoiCalibration.from_value(coordinate_space)
        if calibration.has_xy or calibration.z_res is not None:
            return calibration
    config = _field(source, "config", None)
    if config is not None:
        return RoiCalibration.from_value(config)
    return RoiCalibration()


def _image_token(
    provider: ImageProvider,
    timepoints: Iterable[int],
    planes: Iterable[int],
    *,
    plane_start: int,
) -> str:
    from ..io.image_provider import image_source_manifest_token

    provider_planes = tuple(
        sorted(
            {
                int(plane) - int(plane_start) + 1
                for plane in planes
                if int(plane) - int(plane_start) + 1 >= 1
            }
        )
    )
    token = image_source_manifest_token(
        provider,
        timepoints=timepoints,
        planes=provider_planes,
    )
    if token is not None:
        return token
    for name in ("image_manifest_token", "manifest_token", "source_token", "cache_token"):
        value = getattr(provider, name, None)
        if callable(value):
            value = value()
        if value is not None:
            return str(value)
    # In-memory/third-party providers without an explicit manifest remain
    # safely scoped to this provider instance instead of colliding globally.
    return f"provider-session:{type(provider).__qualname__}:{id(provider)}"


def _parameters_token(request: RoiMeasurementRequest) -> str:
    payload = {
        "metric_keys": request.metric_keys,
        "profiles": request.include_profiles,
        "profile_step_um": request.profile_step_um,
        "profile_width_step_um": request.profile_width_step_um,
        "distributions": request.include_distributions,
        "histogram_bins": request.histogram_bins,
        "quantiles": request.quantiles,
        "frame_dependency_token": request.frame_dependency_token,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name not in {"revision", "frame_id"}
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in {"revision", "frame_id"}
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Geometry fingerprints require finite values")
        return 0.0 if value == 0 else value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _canonical_value(vars(value))
    raise TypeError(f"Unsupported geometry fingerprint value: {type(value).__name__}")


def _aggregate_size(value: RoiIntensityResult) -> int:
    size = 256 + len(value.metrics) * 128 + len(value.warnings) * 64
    if value.distribution is not None:
        size += 8 * (
            len(value.distribution.histogram_counts)
            + len(value.distribution.histogram_edges)
            + 2 * len(value.distribution.quantiles)
        )
    return size


def _subject_object_id(subject: Any) -> str:
    if isinstance(subject, (str, uuid.UUID)):
        return str(subject)
    if isinstance(subject, Mapping):
        return str(subject.get("object_id", subject.get("key", "")))
    return str(getattr(subject, "object_id", getattr(subject, "key", "")))


def _field(value: Any, name: str, default: Any = ...):
    if isinstance(value, Mapping):
        if default is ...:
            return value[name]
        return value.get(name, default)
    if default is ...:
        return getattr(value, name)
    return getattr(value, name, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


__all__ = [
    "ROI_ANALYSIS_ALGORITHM_VERSION",
    "RoiCacheStats",
    "RoiMeasurementCache",
    "RoiMeasurementCancelled",
    "RoiMeasurementEngine",
    "RoiMeasurementRequest",
    "RoiMeasurementSample",
    "RoiMeasurementSnapshot",
    "RoiMeasurementStore",
    "RoiScalarSeriesChannel",
    "geometry_fingerprint",
    "roi_metric_key",
    "roi_temporal_subject",
    "roi_temporal_subjects",
]
