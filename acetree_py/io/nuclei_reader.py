"""Nuclei ZIP reader — reads nuclei text files from ZIP archives.

Each nuclei ZIP contains a directory structure like:
    nuclei/
        t001-nuclei
        t002-nuclei
        ...

Each file contains one line per detected nucleus, comma-separated.

Ported from: org.rhwlab.snight.ZipNuclei (ZipNuclei.java)
             org.rhwlab.tree.AncesTree (reading logic)
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

from acetree_py.core.nucleus import Nucleus

logger = logging.getLogger(__name__)

# Pattern to extract timepoint number from entry names like "t001-nuclei"
_TIME_PATTERN = re.compile(r"t(\d+)-")


def read_nuclei_zip(
    zip_path: str | Path,
    nuc_dir: str = "nuclei/",
    *,
    old_format: bool = False,
) -> list[list[Nucleus]]:
    """Read all nuclei from a ZIP archive.

    Args:
        zip_path: Path to the nuclei ZIP file.
        nuc_dir: Subdirectory within the ZIP containing nuclei files.
            Defaults to "nuclei/".
        old_format: If True, parse using the old column layout.

    Returns:
        A list of nucleus lists, indexed by timepoint (0-based).
        nuclei_record[0] contains nuclei for the first timepoint, etc.
        Each timepoint's list is ordered by the nucleus index in the file.

    Raises:
        FileNotFoundError: If the ZIP file does not exist.
        zipfile.BadZipFile: If the file is not a valid ZIP.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Nuclei ZIP not found: {zip_path}")

    logger.info("Reading nuclei from: %s", zip_path)

    # Collect all nuclei entries and their timepoints
    entries: list[tuple[int, str]] = []  # (timepoint, entry_name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.namelist():
            # Filter to nuclei directory entries that are files (not directories)
            if not entry.startswith(nuc_dir) or entry.endswith("/"):
                continue

            # Extract the filename part after the directory
            parts = entry.split("/")
            if len(parts) < 2:
                continue
            filename = parts[-1]

            # Extract timepoint number
            time = _parse_timepoint(filename)
            if time is not None:
                entries.append((time, entry))

        if not entries:
            logger.warning("No nuclei entries found in ZIP: %s", zip_path)
            return []

        # Sort by timepoint
        entries.sort(key=lambda x: x[0])

        # Build the nuclei record
        # We need to create a list indexed from 0 to max_time
        max_time = max(t for t, _ in entries)
        min_time = min(t for t, _ in entries)

        nuclei_record: list[list[Nucleus]] = []

        # Map timepoints to sequential indices
        time_to_idx: dict[int, int] = {}
        for time, entry_name in entries:
            if time not in time_to_idx:
                time_to_idx[time] = len(nuclei_record)
                nuclei_record.append([])

            idx = time_to_idx[time]
            nuclei = _read_entry(zf, entry_name, old_format=old_format)
            nuclei_record[idx] = nuclei

    logger.info(
        "Read %d timepoints (%d-%d), total nuclei: %d",
        len(nuclei_record),
        min_time,
        max_time,
        sum(len(nl) for nl in nuclei_record),
    )
    return nuclei_record


def _read_entry(
    zf: zipfile.ZipFile,
    entry_name: str,
    *,
    old_format: bool = False,
) -> list[Nucleus]:
    """Read nuclei from a single ZIP entry.

    Args:
        zf: Open ZipFile.
        entry_name: Name of the entry to read.
        old_format: If True, use old column layout.

    Returns:
        List of Nucleus objects parsed from the entry.
    """
    nuclei = []
    with zf.open(entry_name) as f:
        for line_bytes in f:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                nuc = Nucleus.from_text_line(line, old_format=old_format)
                nuclei.append(nuc)
            except (ValueError, IndexError) as e:
                logger.debug("Skipping malformed line in %s: %s", entry_name, e)

    return nuclei


def _parse_timepoint(filename: str) -> int | None:
    """Extract the timepoint number from a nuclei filename.

    Expected formats:
        "t001-nuclei" -> 1
        "t1234-nuclei" -> 1234

    Returns None if the filename doesn't match the expected pattern.
    """
    match = _TIME_PATTERN.search(filename)
    if match:
        return int(match.group(1))
    return None
