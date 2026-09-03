"""Nuclei ZIP writer — writes nuclei text files to ZIP archives.

Creates or updates a ZIP archive with nuclei data in the standard format.

Ported from: org.rhwlab.snight.NucZipper (NucZipper.java)
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile
import zipfile
from pathlib import Path

from acetree_py.core.nucleus import Nucleus

logger = logging.getLogger(__name__)


def write_nuclei_zip(
    nuclei_record: list[list[Nucleus]],
    zip_path: str | Path,
    nuc_dir: str = "nuclei/",
    start_time: int = 1,
) -> None:
    """Write nuclei data to a ZIP archive.

    Args:
        nuclei_record: List of nucleus lists, indexed by timepoint (0-based).
        zip_path: Path for the output ZIP file.
        nuc_dir: Subdirectory name within the ZIP for nuclei files.
        start_time: The 1-based timepoint corresponding to nuclei_record[0].
    """
    zip_path = Path(zip_path)
    tmp_path = stage_nuclei_zip(
        nuclei_record,
        zip_path,
        nuc_dir=nuc_dir,
        start_time=start_time,
    )
    try:
        os.replace(tmp_path, zip_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info("Wrote %d timepoints to %s", len(nuclei_record), zip_path)


def stage_nuclei_zip(
    nuclei_record: list[list[Nucleus]],
    zip_path: str | Path,
    nuc_dir: str = "nuclei/",
    start_time: int = 1,
) -> Path:
    """Build a complete sibling archive without replacing the destination.

    The returned temporary path has the mode that ``zip_path`` should retain
    after replacement.  The caller owns it and must either atomically replace
    the destination or remove it.
    """
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Staging nuclei for: %s (%d timepoints)", zip_path, len(nuclei_record))

    fd, tmp_name = tempfile.mkstemp(
        dir=zip_path.parent,
        prefix=f".{zip_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, nuclei in enumerate(nuclei_record):
                time = start_time + i
                entry_name = f"{nuc_dir}t{time:03d}-nuclei"
                lines = [nuc.to_text_line() for nuc in nuclei]
                content = "\n".join(lines) + "\n" if lines else ""
                zf.writestr(entry_name, content)
        os.chmod(tmp_path, _replacement_mode(zip_path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def _replacement_mode(destination: Path) -> int:
    """Mode for an atomic replacement, preserving target or normal defaults."""
    try:
        return stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        previous = os.umask(0)
        os.umask(previous)
        return 0o666 & ~previous
