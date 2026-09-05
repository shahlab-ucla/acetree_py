"""Detached nuclear measurement inputs and GUI-only atomic publication."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.nuclei_manager import NucleiManager
from ..core.nucleus import Nucleus
from ..io.image_provider import (
    clone_image_provider_for_worker, close_worker_image_provider,
    image_source_manifest_token,
)
from .measure_runner import (
    PreparedMeasurePublication, prepare_measure_publication,
    commit_measure_publication, discard_measure_publication,
)


@dataclass(frozen=True, slots=True)
class _CsvCell:
    name: str
    start_time: int
    end_time: int
    nuclei: tuple[tuple[int, Nucleus], ...]


@dataclass(frozen=True, slots=True)
class _CsvLineage:
    cells: tuple[_CsvCell, ...]

    def all_cells(self):
        return iter(self.cells)


@dataclass(frozen=True, slots=True)
class PreparedNuclearMeasurement:
    source: NucleiManager
    manager: NucleiManager
    image_provider: Any
    output_dir: Path
    at_channel: int
    correction_method: str
    measured_timepoints: tuple[int, ...]
    source_image_manifest_token: str | None


def prepare_nuclear_measurement(
    manager: NucleiManager, image_provider: Any, output_dir: Path,
    at_channel: int, correction_method: str,
) -> PreparedNuclearMeasurement:
    """Capture copied nuclei/calibration and only the cell fields used by CSVs."""

    if manager.lineage_tree is None or not manager.nuclei_record:
        raise ValueError("Load nuclei and build their lineage before measuring")
    detached = NucleiManager()
    detached.nuclei_record = [[nucleus.copy() for nucleus in frame]
                              for frame in manager.nuclei_record]
    copied_nuclei = {
        id(original): copied
        for original_frame, copied_frame in zip(manager.nuclei_record, detached.nuclei_record)
        for original, copied in zip(original_frame, copied_frame)
    }
    detached.movie, detached.config = deepcopy((manager.movie, manager.config))
    # CSV staging only reads these flat fields; parent/daughter trees and
    # naming caches do not cross the worker boundary.
    detached.lineage_tree = _CsvLineage(tuple(
        _CsvCell(cell.name, cell.start_time, cell.end_time, tuple(
            (time, copied_nuclei[id(nucleus)] if id(nucleus) in copied_nuclei else nucleus.copy())
            for time, nucleus in cell.nuclei
        ))
        for cell in manager.lineage_tree.all_cells()
    ))
    detached._expr_corr = manager._expr_corr
    detached._data_revision = manager.data_revision
    times = tuple(time for time, nuclei in enumerate(manager.nuclei_record, 1) if nuclei)
    token = image_source_manifest_token(image_provider, timepoints=times, planes=None)
    return PreparedNuclearMeasurement(
        manager, detached, image_provider, Path(output_dir), int(at_channel),
        correction_method, times, token,
    )


def compute_nuclear_measurement(
    prepared: PreparedNuclearMeasurement,
    progress: Callable[[int, int, str], None],
    is_cancelled: Callable[[], bool],
) -> PreparedMeasurePublication:
    """Compute and stage CSVs off-thread without changing final files or fields."""

    if is_cancelled():
        raise RuntimeError("Measurement cancelled")
    provider = clone_image_provider_for_worker(prepared.image_provider)
    if provider is None:
        raise RuntimeError(
            "This image source does not support background measurement. "
            "Open the source as a supported TIFF dataset to measure it here."
        )
    result = None
    try:
        def report(channel: int, channels: int, time: int, times: int) -> bool:
            done = (time - 1) * channels + channel + 1
            total = channels * times
            progress(done, total, f"Measuring nuclear channel sample {done}/{total}")
            return not is_cancelled()

        result = prepare_measure_publication(
            prepared.manager, provider, prepared.output_dir, prepared.at_channel,
            progress_cb=report, correction_method=prepared.correction_method,
        )
        if is_cancelled():
            raise RuntimeError("Measurement cancelled")
        return result
    except BaseException:
        if result is not None:
            discard_measure_publication(result)
        raise
    finally:
        close_worker_image_provider(provider)


def publish_nuclear_measurement(
    prepared: PreparedNuclearMeasurement, result: PreparedMeasurePublication,
    manager: NucleiManager, image_provider: Any,
) -> list[Path]:
    """Reject source replacement/image edits before the atomic commit boundary."""

    if (
        manager is not prepared.source
        or image_provider is not prepared.image_provider
        or image_source_manifest_token(image_provider, timepoints=prepared.measured_timepoints,
                                       planes=None) != prepared.source_image_manifest_token
    ):
        raise RuntimeError("Nuclei or image data changed during measurement; run Measure again")
    return commit_measure_publication(manager, result)
