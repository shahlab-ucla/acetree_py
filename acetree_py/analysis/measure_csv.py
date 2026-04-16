"""CSV writer for Measure output.

Writes one CSV per expression channel.  Each row is one cell; columns
are ``cell_name, start_time, end_time, t1, t2, ..., tN`` where N is
the last timepoint in the dataset.  Cells absent at a given timepoint
receive an empty cell in that column.

This layout (absolute-time columns) makes the files load cleanly in
pandas / Excel and preserves the sparse nature of the lineage.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def write_measure_csv(
    path: Path,
    rows: Iterable[tuple[str, int, int, list[float | None]]],
    n_timepoints: int,
) -> None:
    """Write a single measure CSV.

    Args:
        path: Destination file path.
        rows: Iterable of ``(cell_name, start_time, end_time, series)``
            tuples.  ``series`` must have length ``n_timepoints``;
            ``None`` entries become empty CSV cells.  ``start_time``
            and ``end_time`` are 1-based inclusive timepoint bounds.
        n_timepoints: Number of timepoints in the dataset.  Used to
            size the header row.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = ["cell_name", "start_time", "end_time"]
    header.extend(f"t{t}" for t in range(1, n_timepoints + 1))

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        count = 0
        for cell_name, start_t, end_t, series in rows:
            if len(series) != n_timepoints:
                raise ValueError(
                    f"Series for '{cell_name}' has length {len(series)}, "
                    f"expected {n_timepoints}"
                )
            row: list[str] = [cell_name, str(start_t), str(end_t)]
            for v in series:
                row.append("" if v is None else _fmt(v))
            writer.writerow(row)
            count += 1

    logger.info("Wrote %d rows to %s", count, path)


def _fmt(v: float) -> str:
    """Format a numeric value for CSV output.

    Integers come out as ``"1234"`` (no trailing ``.0``); floats
    are rounded to 4 decimal places to keep file sizes sane.
    """
    if isinstance(v, bool):  # bool is an int subclass
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.4f}"
