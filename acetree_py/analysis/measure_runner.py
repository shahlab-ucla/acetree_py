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
4. For the *chosen* AT expression channel only, writes the computed
   ``rwraw`` and ``rwcorr1`` back onto each Nucleus and re-runs
   :meth:`NucleiManager.compute_red_weights` so the lineage tree
   re-colours immediately.

The orchestrator is deliberately separate from ``NucleiManager`` so
the pixel-measurement dependency (and numpy) can be optional for
non-image workflows (tests, CSV-only analyses, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ..core.nuclei_manager import NucleiManager
from ..core.nucleus import RED_CORRECTIONS, Nucleus
from ..io.image_provider import ImageProvider
from .measure import measure_timepoint, measure_timepoint_with_blot
from .measure_csv import write_measure_csv

logger = logging.getLogger(__name__)

# AceTree scaling convention from NucleiMgr.computeRWeight: rwraw and
# rwcorr* are stored as mean_intensity * SCALE to preserve precision in
# the integer-valued nuclei file format.
SCALE: int = 1000

# Progress callback signature:
#   progress_cb(channel_idx, num_channels, t_1based, num_timepoints) -> bool
# Return False to cancel; True (or None) to continue.
ProgressCallback = Callable[[int, int, int, int], bool | None]


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
    if manager.lineage_tree is None:
        raise ValueError("Lineage tree not built — call manager.process() first")
    if not manager.nuclei_record:
        raise ValueError("Nuclei record is empty")
    n_channels = image_provider.num_channels
    if not 0 <= at_channel < n_channels:
        raise ValueError(
            f"at_channel={at_channel} out of range (provider has "
            f"{n_channels} channel(s))"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_timepoints = len(manager.nuclei_record)
    z_pix_res = manager.z_pix_res

    # Resolve correction method.  Dialog-driven callers pass it
    # explicitly; legacy callers fall back to the session default.
    if correction_method is None:
        method = manager._expr_corr  # "none", "global", "local", "blot", "cross"
    else:
        method = correction_method
        # Keep the manager's session method in sync so compute_red_weights
        # picks the matching corrected column.
        if method in RED_CORRECTIONS:
            manager._expr_corr = method

    use_blot = method == "blot"

    logger.info(
        "Measure starting: %d channel(s), %d timepoint(s), z_pix_res=%.3f, "
        "at_channel=%d, correction=%s",
        n_channels, n_timepoints, z_pix_res, at_channel, method,
    )

    # measurements[channel][t_0based] = per-nucleus measurement tuples.
    # 4-tuple for non-blot modes, 6-tuple for blot (last two fields are
    # neighbor-masked sums).  Stored uniformly as 6-tuples internally so
    # downstream code doesn't need to branch on length.
    measurements: list[list[list[tuple[int, int, int, int, int, int]]]] = []

    for c in range(n_channels):
        per_channel: list[list[tuple[int, int, int, int, int, int]]] = []
        for t0 in range(n_timepoints):
            t_1based = t0 + 1
            nucs = manager.nuclei_record[t0]
            if not nucs:
                per_channel.append([])
            else:
                try:
                    stack = image_provider.get_stack(t_1based, c)
                except Exception as e:  # noqa: BLE001 — report & skip
                    logger.warning(
                        "Failed to load stack t=%d channel=%d: %s; "
                        "emitting zeros for this timepoint",
                        t_1based, c, e,
                    )
                    per_channel.append([(0, 0, 0, 0, 0, 0)] * len(nucs))
                else:
                    if use_blot:
                        tuples = measure_timepoint_with_blot(
                            stack, nucs, z_pix_res,
                        )
                    else:
                        raw = measure_timepoint(stack, nucs, z_pix_res)
                        # Widen each 4-tuple to 6-tuple with zeros for the
                        # blot fields so storage is uniform.
                        tuples = [
                            (a, b, c_, d, 0, 0) for (a, b, c_, d) in raw
                        ]
                    per_channel.append(tuples)

            if progress_cb is not None:
                cont = progress_cb(c, n_channels, t_1based, n_timepoints)
                if cont is False:
                    raise RuntimeError("Measure cancelled by user")

        measurements.append(per_channel)

    # Write rwraw / rwcorr1 (and rwcorr3 for blot) for the AT channel
    _apply_to_at_channel(
        manager, measurements[at_channel], at_channel, use_blot=use_blot,
    )

    # Recompute rweight using the session's current correction method
    if method and method != "none":
        manager.compute_red_weights()
    else:
        # For correction "none" the tree uses rwraw directly as rweight
        for t0, nucs in enumerate(manager.nuclei_record):
            for j, nuc in enumerate(nucs):
                if nuc.rwraw > 0:
                    nuc.rweight = nuc.rwraw

    # Write one CSV per channel
    written: list[Path] = []
    for c in range(n_channels):
        csv_path = _csv_path_for_channel(output_dir, c, c == at_channel)
        rows = _build_rows(manager, measurements[c], method)
        write_measure_csv(csv_path, rows, n_timepoints)
        written.append(csv_path)

    logger.info("Measure complete: wrote %d CSV(s) to %s", len(written), output_dir)
    return written


def _csv_path_for_channel(out_dir: Path, channel: int, is_at: bool) -> Path:
    """Pick a filename for a channel's CSV."""
    suffix = "_AT" if is_at else ""
    return out_dir / f"measure_channel{channel + 1}{suffix}.csv"


def _apply_to_at_channel(
    manager: NucleiManager,
    per_tp: list[list[tuple[int, int, int, int, int, int]]],
    at_channel: int,
    use_blot: bool = False,
) -> None:
    """Write measured rwraw / rwcorr1 (/ rwcorr3) back onto each nucleus.

    Only touches nuclei with a non-zero pixel count; leaves dead or
    unmeasured nuclei untouched so prior values aren't blown away.

    When ``use_blot`` is True, the neighbor-masked annulus aggregation
    is stored in ``rwcorr3`` (the historical "blot" slot).  Otherwise
    ``rwcorr3`` is left untouched.
    """
    updated = 0
    for t0, nucs in enumerate(manager.nuclei_record):
        if t0 >= len(per_tp):
            break
        tp_meas = per_tp[t0]
        for j, nuc in enumerate(nucs):
            if j >= len(tp_meas):
                continue
            sum_in, count_in, sum_ann, count_ann, sum_blot, count_blot = tp_meas[j]
            if count_in <= 0:
                continue
            nuc.rwraw = int(round(sum_in * SCALE / count_in))
            if count_ann > 0:
                nuc.rwcorr1 = int(round(sum_ann * SCALE / count_ann))
            if use_blot and count_blot > 0:
                nuc.rwcorr3 = int(round(sum_blot * SCALE / count_blot))
            # rsum / rcount preserve the raw (unscaled) pixel aggregation
            # so downstream tools that expect the Java columns also
            # reflect the new measurement.
            nuc.rsum = int(sum_in)
            nuc.rcount = int(count_in)
            updated += 1

    logger.info(
        "Updated rwraw / rwcorr1%s on %d nuclei from channel %d",
        " / rwcorr3" if use_blot else "",
        updated, at_channel + 1,
    )


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
