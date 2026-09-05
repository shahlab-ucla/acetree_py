"""Measure orchestrator — runs pixel measurement across all channels and writes CSVs.

This is the "file-level" entry point for the Measure feature.  Given
a NucleiManager (with an already-built lineage tree) and an
ImageProvider, it:

1. Iterates every channel in the image provider.
2. For each channel, iterates every timepoint and calls
   :func:`measure_timepoint` to collect per-nucleus pixel sums.
3. Writes one CSV per channel with per-cell time series, using the
   session's current correction method to derive the per-timepoint
   value (``rwraw - rwcorr1`` for global, plain ``rwraw`` otherwise).
4. For the *chosen* AT expression channel only, publishes the computed
   legacy red fields back onto each Nucleus. Samples without a valid inner
   measurement have those fields cleared so a save/reopen cannot disguise
   stale values as part of the new, intentionally partial result.

The orchestrator is deliberately separate from ``NucleiManager`` so
the pixel-measurement dependency (and numpy) can be optional for
non-image workflows (tests, CSV-only analyses, etc.).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..core.nuclei_manager import NucleiManager
from ..core.nucleus import RED_CORRECTIONS, Nucleus
from ..io.image_provider import ImageProvider, image_source_manifest_token
from .measure import measure_timepoint, measure_timepoint_with_blot
from .measure_csv import write_measure_csv
from .expression_measurements import (
    ExpressionMeasurementFamily,
    ExpressionMeasurementSet,
    MeasuredExpressionAggregate,
    MeasuredExpressionAggregateChannel,
    MeasuredExpressionChannel,
    MeasuredExpressionSample,
    NucleusGeometrySignature,
    expression_document_fingerprint,
    expression_measurement_calibration,
    expression_measurement_plane_start,
    expression_measurement_dependency_fingerprint,
)

logger = logging.getLogger(__name__)

# AceTree scaling convention from NucleiMgr.computeRWeight: rwraw and
# rwcorr* are stored as mean_intensity * SCALE to preserve precision in
# the integer-valued nuclei file format.
SCALE: int = 1000

# Progress callback signature:
#   progress_cb(channel_idx, num_channels, t_1based, num_timepoints) -> bool
# Return False to cancel; True (or None) to continue.
ProgressCallback = Callable[[int, int, int, int], bool | None]
MeasurementTuple = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _MeasurementRun:
    measurements: list[list[list[MeasurementTuple]]]
    method: str
    at_channel: int
    n_channels: int
    n_timepoints: int
    source_revision: int
    source_fingerprint: str
    source_dependency_fingerprint: str
    source_calibration: tuple[float, float, int, float]
    source_plane_start: int


def measure_expression_set(
    manager: NucleiManager,
    image_provider: ImageProvider,
    *,
    at_channel: int = 0,
    progress_cb: ProgressCallback | None = None,
    correction_method: str | None = None,
) -> ExpressionMeasurementSet:
    """Measure all image channels without mutating or writing the dataset.

    This is the reusable computation boundary for comparison/analysis tools.
    The returned immutable snapshot is revision-, calibration-, and
    geometry-bound, but it is not installed on ``manager`` and no legacy
    expression fields or CSV files are changed.
    """

    run = _collect_measurement_run(
        manager,
        image_provider,
        at_channel=at_channel,
        progress_cb=progress_cb,
        correction_method=correction_method,
    )
    result = _build_expression_measurement_set(
        manager,
        run.measurements,
        method=run.method,
        at_channel=run.at_channel,
        csv_paths=[],
        source_revision=run.source_revision,
        source_dependency_fingerprint=run.source_dependency_fingerprint,
        source_calibration=run.source_calibration,
        source_plane_start=run.source_plane_start,
    )
    if not _measurement_source_matches(manager, run):
        raise RuntimeError(
            "Dataset changed while Measure was preparing results; no "
            "measurement snapshot was returned. Run Measure again."
        )
    return result


def measure_expression_family(
    manager: NucleiManager,
    image_provider: ImageProvider,
    *,
    progress_cb: ProgressCallback | None = None,
    _validate_image_manifest: bool = True,
) -> ExpressionMeasurementFamily:
    """Measure every channel and supported correction in one image pass.

    ``_validate_image_manifest=False`` is reserved for the repository, which
    owns a stronger full-source validation lease around this call.
    """

    run = _collect_measurement_run(
        manager,
        image_provider,
        at_channel=0,
        progress_cb=progress_cb,
        correction_method="blot",
        validate_image_manifest=_validate_image_manifest,
    )
    result = _build_expression_measurement_family(
        manager,
        run.measurements,
        source_revision=run.source_revision,
        source_dependency_fingerprint=run.source_dependency_fingerprint,
        source_calibration=run.source_calibration,
        source_plane_start=run.source_plane_start,
    )
    return result


def run_measure(
    manager: NucleiManager,
    image_provider: ImageProvider,
    output_dir: Path,
    at_channel: int,
    progress_cb: ProgressCallback | None = None,
    correction_method: str | None = None,
) -> list[Path]:
    """Measure every channel, write CSVs, and update the AT channel's rweight.

    Args:
        manager: Loaded NucleiManager with a built lineage tree.
        image_provider: Image provider exposing all channels.
        output_dir: Folder to write CSVs into.  Created if missing.
        at_channel: 0-based channel whose measurements become the new
            nucleus.rwraw / rwcorr1 (/ rwcorr3 for blot) and therefore
            the lineage tree coloring.  Other channels produce CSVs
            only.
        progress_cb: Optional callback fired after each (channel, t).
            Returning ``False`` cancels the run cleanly.
        correction_method: Background-correction mode for the measured
            values.  One of:

            - ``"none"``: no subtraction; CSV value is raw mean intensity.
            - ``"global"``: subtract the annulus background (``rwcorr1``).
            - ``"blot"``: subtract a neighbor-masked annulus background
              (stored in ``rwcorr3``).  This excludes any pixels belonging
              to another nucleus's inner disk at that plane, giving a
              cleaner local-background estimate in crowded regions.

            When ``None`` (default), falls back to ``manager._expr_corr``
            so legacy callers (and the config-driven path) still work.

    Returns:
        List of Paths written (one per channel), in channel order.

    Raises:
        ValueError: On invalid inputs (no tree, no nuclei, bad channel).
        RuntimeError: If the run is cancelled via progress_cb.
    """
    prepared = prepare_measure_publication(
        manager, image_provider, output_dir, at_channel,
        progress_cb=progress_cb, correction_method=correction_method,
    )
    return commit_measure_publication(manager, prepared)


@dataclass(frozen=True, slots=True)
class PreparedMeasurePublication:
    """Privately staged measurement data, ready for one publication attempt.

    Preparing changes neither final CSVs nor the source manager. A background
    caller owns this object until the GUI publishes it or discards its files.
    The immutable measurement set is complete before publication begins.
    """

    measurement_set: ExpressionMeasurementSet
    output_dir: Path
    _run: _MeasurementRun
    _staged_csvs: tuple[tuple[Path, Path], ...]


def prepare_measure_publication(
    manager: NucleiManager,
    image_provider: ImageProvider,
    output_dir: Path,
    at_channel: int,
    progress_cb: ProgressCallback | None = None,
    correction_method: str | None = None,
) -> PreparedMeasurePublication:
    """Compute and stage privately against a caller-owned stable manager.

    Worker callers must pass a detached manager snapshot and private provider.
    The live manager is checked again by commit_measure_publication.
    """
    measurement = _collect_measurement_run(
        manager,
        image_provider,
        at_channel=at_channel,
        progress_cb=progress_cb,
        correction_method=correction_method,
    )
    valid_at_samples = sum(
        1
        for timepoint in measurement.measurements[at_channel]
        for sample in timepoint
        if sample[1] > 0
    )
    if valid_at_samples == 0:
        raise RuntimeError(
            "Measure produced no valid samples for the selected AT channel; "
            "existing expression values and files were left unchanged. Verify "
            "the image source, channel, time range, and nucleus geometry."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements = measurement.measurements
    method = measurement.method
    n_timepoints = measurement.n_timepoints
    source_revision = measurement.source_revision
    source_dependency_fingerprint = measurement.source_dependency_fingerprint
    source_calibration = measurement.source_calibration
    source_plane_start = measurement.source_plane_start

    staged_csvs = _stage_measure_csvs(
        manager,
        measurements,
        method=method,
        output_dir=output_dir,
        at_channel=at_channel,
        n_timepoints=n_timepoints,
    )
    final_csv_paths = [final_path for _staged_path, final_path in staged_csvs]

    try:
        # Non-GUI callers may mutate the manager from another thread while
        # files are being staged. Validate before constructing the result.
        if not _measurement_source_matches(manager, measurement):
            raise RuntimeError(
                "Dataset changed while Measure was writing results; no "
                "measurements were applied. Run Measure again."
            )

        pending_store = _build_expression_measurement_set(
            manager,
            measurements,
            method=method,
            at_channel=at_channel,
            csv_paths=final_csv_paths,
            source_revision=source_revision,
            source_dependency_fingerprint=source_dependency_fingerprint,
            source_calibration=source_calibration,
            source_plane_start=source_plane_start,
        )

        # Building a large store can take long enough for a background caller
        # to mutate the source. This is the last boundary before publication.
        if not _measurement_source_matches(manager, measurement):
            raise RuntimeError(
                "Dataset changed while Measure was preparing results; no "
                "measurements were applied. Run Measure again."
            )

        return PreparedMeasurePublication(
            measurement_set=pending_store,
            output_dir=output_dir,
            _run=measurement,
            _staged_csvs=tuple(staged_csvs),
        )
    except BaseException:
        _discard_staged_csvs(staged_csvs)
        raise


def discard_measure_publication(prepared: PreparedMeasurePublication) -> None:
    """Release staged files after cancellation, stale-source rejection or close."""

    _discard_staged_csvs(list(prepared._staged_csvs))


def commit_measure_publication(
    manager: NucleiManager,
    prepared: PreparedMeasurePublication,
) -> list[Path]:
    """Validate the live source and atomically publish prepared files and data.

    Call on the GUI thread for interactive use. Source-provider identity,
    external image freshness and job cancellation are the caller's boundary;
    document revision, calibration and record changes are checked here.
    Staging files are discarded on failure and final files roll back together.
    """

    measurement = prepared._run
    measurements = measurement.measurements
    method = measurement.method
    at_channel = measurement.at_channel
    use_blot = method == "blot"
    staged_csvs = list(prepared._staged_csvs)
    # CSV replacement, the one legacy AT slot, and the all-channel snapshot
    # form one publication transaction.  Old files remain as rollback copies
    # until every in-memory step succeeds.
    previous_method = manager._expr_corr
    previous_config_method = (
        manager.config.expr_corr if manager.config is not None else None
    )
    previous_config_dirty = bool(getattr(manager, "_config_dirty", False))
    previous_store = manager.expression_measurements
    previous_freshness = manager.expression_measurement_freshness_known
    previous_fields = [
        (
            nucleus,
            nucleus.rweight,
            nucleus.rwraw,
            nucleus.rwcorr1,
            nucleus.rwcorr2,
            nucleus.rwcorr3,
            nucleus.rwcorr4,
            nucleus.rsum,
            nucleus.rcount,
        )
        for nuclei in manager.nuclei_record
        for nucleus in nuclei
    ]
    written: list[Path] = []
    csv_backups: dict[Path, Path] = {}
    publication_started = False
    try:
        if not _measurement_source_matches(manager, measurement):
            raise RuntimeError(
                "Dataset changed while Measure was preparing results; no "
                "measurements were applied. Run Measure again."
            )
        written, csv_backups = _install_staged_csvs(staged_csvs)
        if not _measurement_source_matches(manager, measurement):
            raise RuntimeError(
                "Dataset changed immediately before Measure publication; no "
                "measurements were applied. Run Measure again."
            )
        publication_started = True
        if method in ("none", "global", "blot"):
            manager._expr_corr = method
        elif method in RED_CORRECTIONS:
            # Python cannot recompute local/cross fields. Their documented
            # Measure fallback is the fresh global annulus.
            manager._expr_corr = "global"
        if (
            manager.config is not None
            and manager.config.expr_corr != manager._expr_corr
        ):
            manager.config.expr_corr = manager._expr_corr
            manager._config_dirty = True
        _apply_to_at_channel(
            manager, measurements[at_channel], at_channel, use_blot=use_blot,
        )
        _set_measured_at_weights(manager, measurements[at_channel], method)
        manager.expression_measurements = prepared.measurement_set
        manager.expression_measurement_freshness_known = True
    except BaseException:
        # A source mismatch detected before publication may be the result of
        # a legitimate concurrent edit.  Do not overwrite that edit with the
        # older snapshot when Measure has not mutated the manager yet.
        if publication_started:
            manager._expr_corr = previous_method
            if manager.config is not None and previous_config_method is not None:
                manager.config.expr_corr = previous_config_method
            manager._config_dirty = previous_config_dirty
            manager.expression_measurements = previous_store
            manager.expression_measurement_freshness_known = previous_freshness
            for (
                nucleus,
                rweight,
                rwraw,
                rwcorr1,
                rwcorr2,
                rwcorr3,
                rwcorr4,
                rsum,
                rcount,
            ) in previous_fields:
                nucleus.rweight = rweight
                nucleus.rwraw = rwraw
                nucleus.rwcorr1 = rwcorr1
                nucleus.rwcorr2 = rwcorr2
                nucleus.rwcorr3 = rwcorr3
                nucleus.rwcorr4 = rwcorr4
                nucleus.rsum = rsum
                nucleus.rcount = rcount
        if written or csv_backups:
            _rollback_installed_csvs(written, csv_backups)
        _discard_staged_csvs(staged_csvs)
        raise
    else:
        _finalize_installed_csvs(csv_backups)

    logger.info("Measure complete: wrote %d CSV(s) to %s", len(written), prepared.output_dir)
    return written


def _collect_measurement_run(
    manager: NucleiManager,
    image_provider: ImageProvider,
    *,
    at_channel: int,
    progress_cb: ProgressCallback | None,
    correction_method: str | None,
    validate_image_manifest: bool = True,
) -> _MeasurementRun:
    """Compute all channel aggregates against one immutable source state."""

    if manager.lineage_tree is None:
        raise ValueError("Lineage tree not built — call manager.process() first")
    if not manager.nuclei_record:
        raise ValueError("Nuclei record is empty")
    measured_timepoints = tuple(
        time
        for time, nuclei in enumerate(manager.nuclei_record, start=1)
        if nuclei
    )
    image_manifest_before = (
        image_source_manifest_token(
            image_provider,
            timepoints=measured_timepoints,
            planes=None,
        )
        if validate_image_manifest
        else None
    )
    n_channels = image_provider.num_channels
    if not 0 <= at_channel < n_channels:
        raise ValueError(
            f"at_channel={at_channel} out of range (provider has "
            f"{n_channels} channel(s))"
        )

    n_timepoints = len(manager.nuclei_record)
    z_pix_res = manager.z_pix_res
    source_revision = int(getattr(manager, "data_revision", 0))
    source_fingerprint = expression_document_fingerprint(manager)
    source_dependency_fingerprint = (
        expression_measurement_dependency_fingerprint(manager)
    )
    source_calibration = expression_measurement_calibration(manager)
    source_plane_start = expression_measurement_plane_start(manager)

    method = manager._expr_corr if correction_method is None else correction_method
    if method not in RED_CORRECTIONS:
        choices = ", ".join(RED_CORRECTIONS)
        raise ValueError(
            f"Unknown correction_method={method!r}; choose one of: {choices}"
        )
    use_blot = method == "blot"

    logger.info(
        "Measure starting: %d channel(s), %d timepoint(s), z_pix_res=%.3f, "
        "at_channel=%d, correction=%s",
        n_channels,
        n_timepoints,
        z_pix_res,
        at_channel,
        method,
    )

    # measurements[channel][t_0based] = per-nucleus measurement tuples.
    # Stored uniformly as 6-tuples; non-blot runs use zero blot aggregates.
    measurements: list[list[list[MeasurementTuple]]] = [
        [] for _channel in range(n_channels)
    ]
    all_channel_loader = getattr(image_provider, "get_all_channel_stacks", None)
    for t0 in range(n_timepoints):
        time = t0 + 1
        nuclei = manager.nuclei_record[t0]
        shared_stacks = None
        if nuclei and callable(all_channel_loader):
            try:
                shared_stacks = tuple(all_channel_loader(time))
                if len(shared_stacks) != n_channels:
                    raise ValueError(
                        "get_all_channel_stacks returned "
                        f"{len(shared_stacks)} stacks for {n_channels} channels"
                    )
            except Exception as error:  # noqa: BLE001 — optional compatibility path
                # Bulk loading is an optional optimization.  Preserve the
                # established per-channel API when a provider cannot use it.
                logger.warning(
                    "Bulk stack load failed at t=%d (%s); falling back to "
                    "per-channel reads",
                    time,
                    error,
                )

        for channel_index in range(n_channels):
            if not nuclei:
                tuples: list[MeasurementTuple] = []
            else:
                stack = None
                load_error: Exception | None = None
                if shared_stacks is not None:
                    stack = shared_stacks[channel_index]
                else:
                    try:
                        stack = image_provider.get_stack(time, channel_index)
                    except Exception as error:  # noqa: BLE001 — partial coverage
                        load_error = error
                if load_error is not None:
                    logger.warning(
                        "Failed to load stack t=%d channel=%d: %s; "
                        "emitting missing measurements for this timepoint",
                        time,
                        channel_index,
                        load_error,
                    )
                    tuples = [(0, 0, 0, 0, 0, 0)] * len(nuclei)
                elif use_blot:
                    tuples = measure_timepoint_with_blot(
                        stack,
                        nuclei,
                        z_pix_res,
                        plane_start=source_plane_start,
                    )
                else:
                    raw = measure_timepoint(
                        stack, nuclei, z_pix_res, plane_start=source_plane_start
                    )
                    tuples = [
                        (inner_sum, inner_count, ann_sum, ann_count, 0, 0)
                        for inner_sum, inner_count, ann_sum, ann_count in raw
                    ]
            measurements[channel_index].append(tuples)

            if progress_cb is not None:
                proceed = progress_cb(
                    channel_index,
                    n_channels,
                    time,
                    n_timepoints,
                )
                if proceed is False:
                    raise RuntimeError("Measure cancelled by user")

    run = _MeasurementRun(
        measurements=measurements,
        method=method,
        at_channel=at_channel,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        source_revision=source_revision,
        source_fingerprint=source_fingerprint,
        source_dependency_fingerprint=source_dependency_fingerprint,
        source_calibration=source_calibration,
        source_plane_start=source_plane_start,
    )
    image_manifest_after = (
        image_source_manifest_token(
            image_provider,
            timepoints=measured_timepoints,
            planes=None,
        )
        if validate_image_manifest
        else None
    )
    if (
        image_manifest_before is not None
        and image_manifest_after is not None
        and image_manifest_before != image_manifest_after
    ):
        raise RuntimeError(
            "Image source changed while Measure was running; no measurements "
            "were applied. Reload the dataset and run Measure again."
        )
    if not _measurement_source_matches(manager, run):
        raise RuntimeError(
            "Dataset changed while Measure was running; no measurements were "
            "applied. Run Measure again against the current nuclei."
        )
    return run


def _measurement_source_matches(
    manager: NucleiManager,
    run: _MeasurementRun,
) -> bool:
    return (
        int(getattr(manager, "data_revision", 0)) == run.source_revision
        and expression_document_fingerprint(manager) == run.source_fingerprint
    )


def _build_expression_measurement_set(
    manager: NucleiManager,
    measurements: list[list[list[tuple[int, int, int, int, int, int]]]],
    *,
    method: str,
    at_channel: int,
    csv_paths: list[Path],
    source_revision: int,
    source_dependency_fingerprint: str,
    source_calibration: tuple[float, float, int, float],
    source_plane_start: int,
) -> ExpressionMeasurementSet:
    """Retain every measured channel in a revision- and geometry-bound store."""

    geometries = {
        (t0 + 1, int(nucleus.index)): NucleusGeometrySignature.from_nucleus(nucleus)
        for t0, nuclei in enumerate(manager.nuclei_record)
        for nucleus in nuclei
        if nucleus.status >= 1
    }
    channels: list[MeasuredExpressionChannel] = []
    for channel_index, per_timepoint in enumerate(measurements):
        samples: dict[tuple[int, int], MeasuredExpressionSample] = {}
        for t0, nuclei in enumerate(manager.nuclei_record):
            if t0 >= len(per_timepoint):
                continue
            measured_nuclei = per_timepoint[t0]
            for offset, nucleus in enumerate(nuclei):
                if offset >= len(measured_nuclei):
                    continue
                (
                    sum_in,
                    count_in,
                    sum_ann,
                    count_ann,
                    sum_blot,
                    count_blot,
                ) = measured_nuclei[offset]
                if count_in <= 0:
                    continue
                raw = sum_in * SCALE / count_in
                annulus = sum_ann * SCALE / count_ann if count_ann > 0 else None
                blot = sum_blot * SCALE / count_blot if count_blot > 0 else None
                # _combine historically substitutes the global annulus when
                # blot data is absent. Preserve that exact compatibility.
                annulus_for_combine = annulus if annulus is not None else 0.0
                blot_for_combine = blot if blot is not None else annulus_for_combine
                samples[(t0 + 1, int(nucleus.index))] = MeasuredExpressionSample(
                    value=_combine(raw, annulus_for_combine, blot_for_combine, method),
                    raw=raw,
                    annulus_background=annulus,
                    blot_background=blot,
                    inner_pixel_count=int(count_in),
                    annulus_pixel_count=int(count_ann),
                    blot_pixel_count=int(count_blot),
                )
        channels.append(
            MeasuredExpressionChannel(
                image_channel=channel_index,
                label=f"Channel {channel_index + 1}",
                samples=samples,
            )
        )

    return ExpressionMeasurementSet(
        source_revision=source_revision,
        source_dependency_fingerprint=source_dependency_fingerprint,
        source_calibration=source_calibration,
        source_plane_start=source_plane_start,
        correction_method=method,
        at_channel=at_channel,
        channels=tuple(channels),
        geometries=geometries,
        csv_paths=tuple(Path(path) for path in csv_paths),
    )


def _build_expression_measurement_family(
    manager: NucleiManager,
    measurements: list[list[list[MeasurementTuple]]],
    *,
    source_revision: int,
    source_dependency_fingerprint: str,
    source_calibration: tuple[float, float, int, float],
    source_plane_start: int,
) -> ExpressionMeasurementFamily:
    """Build one aggregate store from an all-corrections measurement pass."""

    geometries = {
        (t0 + 1, int(nucleus.index)): NucleusGeometrySignature.from_nucleus(nucleus)
        for t0, nuclei in enumerate(manager.nuclei_record)
        for nucleus in nuclei
        if nucleus.status >= 1
    }
    channels: list[MeasuredExpressionAggregateChannel] = []
    for channel_index, per_timepoint in enumerate(measurements):
        samples: dict[tuple[int, int], MeasuredExpressionAggregate] = {}
        for t0, nuclei in enumerate(manager.nuclei_record):
            if t0 >= len(per_timepoint):
                continue
            measured_nuclei = per_timepoint[t0]
            for offset, nucleus in enumerate(nuclei):
                if offset >= len(measured_nuclei):
                    continue
                (
                    sum_in,
                    count_in,
                    sum_ann,
                    count_ann,
                    sum_blot,
                    count_blot,
                ) = measured_nuclei[offset]
                if count_in <= 0:
                    continue
                samples[(t0 + 1, int(nucleus.index))] = MeasuredExpressionAggregate(
                    raw=sum_in * SCALE / count_in,
                    annulus_background=(
                        sum_ann * SCALE / count_ann if count_ann > 0 else None
                    ),
                    blot_background=(
                        sum_blot * SCALE / count_blot if count_blot > 0 else None
                    ),
                    inner_pixel_count=int(count_in),
                    annulus_pixel_count=int(count_ann),
                    blot_pixel_count=int(count_blot),
                )
        channels.append(
            MeasuredExpressionAggregateChannel(
                image_channel=channel_index,
                label=f"Channel {channel_index + 1}",
                samples=samples,
            )
        )

    return ExpressionMeasurementFamily(
        source_revision=source_revision,
        source_dependency_fingerprint=source_dependency_fingerprint,
        source_calibration=source_calibration,
        source_plane_start=source_plane_start,
        channels=tuple(channels),
        geometries=geometries,
    )


def _csv_path_for_channel(out_dir: Path, channel: int, is_at: bool) -> Path:
    """Pick a filename for a channel's CSV."""
    suffix = "_AT" if is_at else ""
    return out_dir / f"measure_channel{channel + 1}{suffix}.csv"


def _stage_measure_csvs(
    manager: NucleiManager,
    measurements: list[list[list[tuple[int, int, int, int, int, int]]]],
    *,
    method: str,
    output_dir: Path,
    at_channel: int,
    n_timepoints: int,
) -> list[tuple[Path, Path]]:
    """Write every channel to private siblings without touching old outputs."""

    staged: list[tuple[Path, Path]] = []
    try:
        for channel_index, per_timepoint in enumerate(measurements):
            final_path = _csv_path_for_channel(
                output_dir,
                channel_index,
                channel_index == at_channel,
            )
            staged_path = _unused_sibling_path(final_path, suffix=".csv.tmp")
            staged.append((staged_path, final_path))
            rows = _build_rows(manager, per_timepoint, method)
            write_measure_csv(staged_path, rows, n_timepoints)
    except BaseException:
        _discard_staged_csvs(staged)
        raise
    return staged


def _install_staged_csvs(
    staged: list[tuple[Path, Path]],
) -> tuple[list[Path], dict[Path, Path]]:
    """Install a complete channel set while retaining rollback copies."""

    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for _staged_path, final_path in staged:
            if final_path.exists():
                backup = _unused_sibling_path(final_path, suffix=".csv.bak")
                final_path.replace(backup)
                backups[final_path] = backup
        for staged_path, final_path in staged:
            staged_path.replace(final_path)
            installed.append(final_path)
    except BaseException:
        _rollback_installed_csvs(installed, backups)
        raise
    return [final_path for _staged_path, final_path in staged], backups


def _rollback_installed_csvs(
    installed: list[Path],
    backups: dict[Path, Path],
) -> None:
    """Remove a new channel set and restore every prior destination."""

    for final_path in installed:
        try:
            final_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove partial Measure CSV %s", final_path)
    for final_path, backup in backups.items():
        try:
            if backup.exists():
                backup.replace(final_path)
        except OSError:
            logger.exception("Could not restore prior Measure CSV %s", final_path)


def _finalize_installed_csvs(backups: dict[Path, Path]) -> None:
    """Discard rollback copies after all in-memory publication succeeds."""

    for backup in backups.values():
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove Measure CSV backup %s", backup)


def _discard_staged_csvs(staged: list[tuple[Path, Path]]) -> None:
    for staged_path, _final_path in staged:
        try:
            staged_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove staged Measure CSV %s", staged_path)


def _unused_sibling_path(destination: Path, *, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _apply_to_at_channel(
    manager: NucleiManager,
    per_tp: list[list[tuple[int, int, int, int, int, int]]],
    at_channel: int,
    use_blot: bool = False,
) -> None:
    """Write measured rwraw / rwcorr1 (/ rwcorr3) back onto each nucleus.

    A non-positive inner-pixel count is the explicit missing-sample marker.
    In that case all legacy red-expression fields are cleared. This prevents
    a successful partial run from combining old values with new ones after
    the nuclei archive is saved and reopened, when the in-memory measurement
    snapshot is no longer available to carry an exact validity mask.

    When ``use_blot`` is True, the neighbor-masked annulus aggregation
    is stored in ``rwcorr3`` (the historical "blot" slot).  Otherwise
    ``rwcorr3`` is left untouched.
    """
    updated = 0
    cleared = 0
    for t0, nucs in enumerate(manager.nuclei_record):
        if t0 >= len(per_tp):
            break
        tp_meas = per_tp[t0]
        for j, nuc in enumerate(nucs):
            if j >= len(tp_meas):
                continue
            sum_in, count_in, sum_ann, count_ann, sum_blot, count_blot = tp_meas[j]
            if count_in <= 0:
                _clear_legacy_at_measurement(nuc)
                cleared += 1
                continue
            nuc.rwraw = int(round(sum_in * SCALE / count_in))
            nuc.rwcorr1 = (
                int(round(sum_ann * SCALE / count_ann)) if count_ann > 0 else 0
            )
            if use_blot:
                # Match _combine(): absent blot pixels fall back to the newly
                # measured global annulus, never a correction from an old run.
                nuc.rwcorr3 = (
                    int(round(sum_blot * SCALE / count_blot))
                    if count_blot > 0
                    else nuc.rwcorr1
                )
            # rsum / rcount preserve the raw (unscaled) pixel aggregation
            # so downstream tools that expect the Java columns also
            # reflect the new measurement.
            nuc.rsum = int(sum_in)
            nuc.rcount = int(count_in)
            updated += 1

    logger.info(
        "Updated rwraw / rwcorr1%s on %d nuclei and cleared %d missing "
        "samples from channel %d",
        " / rwcorr3" if use_blot else "",
        updated,
        cleared,
        at_channel + 1,
    )


def _clear_legacy_at_measurement(nucleus: Nucleus) -> None:
    """Mark one selected-channel legacy sample as absent for persistence."""

    nucleus.rweight = 0
    nucleus.rsum = 0
    nucleus.rcount = 0
    nucleus.rwraw = 0
    nucleus.rwcorr1 = 0
    nucleus.rwcorr2 = 0
    nucleus.rwcorr3 = 0
    nucleus.rwcorr4 = 0


def _set_measured_at_weights(
    manager: NucleiManager,
    per_tp: list[list[tuple[int, int, int, int, int, int]]],
    method: str,
) -> None:
    """Make legacy rweight match the retained sample and Measure CSV exactly."""

    for t0, nuclei in enumerate(manager.nuclei_record):
        if t0 >= len(per_tp):
            break
        measured_nuclei = per_tp[t0]
        for offset, nucleus in enumerate(nuclei):
            if offset >= len(measured_nuclei) or measured_nuclei[offset][1] <= 0:
                continue
            if method == "blot":
                nucleus.rweight = nucleus.rwraw - nucleus.rwcorr3
            elif method in RED_CORRECTIONS and method != "none":
                # local/cross use the documented global fallback because this
                # port does not calculate rwcorr2/rwcorr4.
                nucleus.rweight = nucleus.rwraw - nucleus.rwcorr1
            else:
                nucleus.rweight = nucleus.rwraw


def _build_rows(
    manager: NucleiManager,
    per_tp: list[list[tuple[int, int, int, int, int, int]]],
    method: str,
) -> list[tuple[str, int, int, list[float | None]]]:
    """Assemble per-cell CSV rows from a channel's per-timepoint measurements.

    The per-timepoint value uses the same correction formula the
    lineage tree applies to its color mapping, so the CSVs match
    what the user sees on screen (for the AT channel) or would see
    if they switched to that channel (for the other channels).
    """
    assert manager.lineage_tree is not None  # checked by caller
    n_timepoints = len(manager.nuclei_record)
    rows: list[tuple[str, int, int, list[float | None]]] = []

    # Iterate cells in a stable order: by start_time, then by name.
    cells = sorted(
        manager.lineage_tree.all_cells(),
        key=lambda c: (c.start_time, c.name),
    )

    for cell in cells:
        if not cell.nuclei:
            continue
        series: list[float | None] = [None] * n_timepoints
        for t_1based, nuc in cell.nuclei:
            t0 = t_1based - 1
            if t0 < 0 or t0 >= len(per_tp):
                continue
            j = nuc.index - 1  # 1-based nucleus index → 0-based
            tp = per_tp[t0]
            if j < 0 or j >= len(tp):
                continue
            sum_in, count_in, sum_ann, count_ann, sum_blot, count_blot = tp[j]
            if count_in <= 0:
                continue
            rwraw = sum_in * SCALE / count_in
            rwcorr1 = (sum_ann * SCALE / count_ann) if count_ann else 0.0
            rwcorr3 = (sum_blot * SCALE / count_blot) if count_blot else rwcorr1
            series[t0] = _combine(rwraw, rwcorr1, rwcorr3, method)

        rows.append((cell.name, cell.start_time, cell.end_time, series))

    return rows


def _combine(
    rwraw: float,
    rwcorr1: float,
    rwcorr3: float,
    method: str,
) -> float:
    """Combine rwraw and the background terms per the correction method.

    Matches :func:`Nucleus.corrected_red` semantics but operates on
    the freshly-measured values rather than stored fields:

    - ``"none"``: return rwraw (no subtraction).
    - ``"global"``: return ``rwraw - rwcorr1`` (annulus background).
    - ``"blot"``: return ``rwraw - rwcorr3`` (annulus with neighbors
      masked out — cleaner in crowded regions).
    - Other legacy modes (``"local"``, ``"cross"``) fall back to
      ``rwraw - rwcorr1`` since rwcorr2/4 are not computed by this
      Python port (they came from external MATLAB / crosstalk solver).
    """
    if method not in RED_CORRECTIONS or method == "none":
        return rwraw
    if method == "blot":
        return rwraw - rwcorr3
    return rwraw - rwcorr1
