"""Capture, compute, and publish boundaries for background ROI measurement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from ..core.subcellular_roi import SubcellularRoiDocument
from ..io.image_provider import clone_image_provider_for_worker, close_worker_image_provider
from .roi_measurements import (
    RoiMeasurementCancelled, RoiMeasurementEngine, RoiMeasurementRequest,
    RoiMeasurementSnapshot, RoiMeasurementStore, _image_token, _select_frames,
    _selected_planes, _source_calibration, _source_document,
    _source_physical_normalization_blocked, _source_revision,
)
from .roi_rasterization import RoiCalibration


@dataclass(frozen=True, slots=True)
class _RoiSource:
    document: SubcellularRoiDocument
    roi_revision: int
    physical_normalization_blocked: bool


@dataclass(frozen=True, slots=True)
class PreparedRoiMeasurement:
    request: RoiMeasurementRequest
    source: Any
    image_provider: Any
    calibration: RoiCalibration
    source_image_manifest_token: str


def prepare_roi_measurement(request: RoiMeasurementRequest, image_provider: Any) -> PreparedRoiMeasurement:
    """Capture the immutable ROI document and selected image dependencies."""

    source = request.document
    document = _source_document(source)
    if not isinstance(document, SubcellularRoiDocument):
        raise TypeError("Background ROI measurement requires an immutable ROI document")
    calibration = _source_calibration(source, document, None)
    frozen_source = _RoiSource(
        document, _source_revision(source, document),
        _source_physical_normalization_blocked(source),
    )
    request = replace(request, document=frozen_source)
    frames = _select_frames(document, request)
    token = _image_token(
        image_provider, sorted({frame[2] for frame in frames}),
        _selected_planes(frames), plane_start=calibration.plane_start,
    )
    return PreparedRoiMeasurement(request, source, image_provider, calibration, token)


def compute_roi_measurement(
    prepared: PreparedRoiMeasurement,
    progress: Callable[[int, int, str], None],
    is_cancelled: Callable[[], bool],
) -> RoiMeasurementSnapshot:
    """Read private image handles and return an unpublished snapshot."""

    if is_cancelled():
        raise RoiMeasurementCancelled("ROI measurement cancelled")
    provider = clone_image_provider_for_worker(prepared.image_provider)
    if provider is None:
        raise RuntimeError(
            "This image source does not support background measurement. "
            "Synchronous ROI measurement remains available through RoiMeasurementEngine.measure."
        )
    try:
        def report(done: int, total: int) -> bool:
            progress(done, total, f"Measuring ROI channel sample {done}/{total}")
            return not is_cancelled()

        result = RoiMeasurementEngine(provider).measure(
            prepared.request, calibration=prepared.calibration, progress_cb=report,
        )
        if is_cancelled():
            raise RoiMeasurementCancelled("ROI measurement cancelled")
        # In-memory clones have different wrapper identities. Keep provenance
        # bound to the original source, which is revalidated before publication.
        return replace(result, source_image_manifest_token=prepared.source_image_manifest_token)
    finally:
        close_worker_image_provider(provider)


def publish_roi_measurement(
    prepared: PreparedRoiMeasurement,
    snapshot: RoiMeasurementSnapshot,
    source: Any,
    image_provider: Any,
    store: RoiMeasurementStore,
) -> None:
    """Revalidate and publish on the GUI thread, preserving old results on error."""

    if (
        source is not prepared.source
        or image_provider is not prepared.image_provider
        or not snapshot.is_current(source, image_provider=image_provider,
                                   calibration=prepared.calibration)
        or _source_calibration(source, _source_document(source), None) != prepared.calibration
    ):
        raise RuntimeError("ROI or image data changed during measurement; run Measure again")
    store.publish(snapshot)
