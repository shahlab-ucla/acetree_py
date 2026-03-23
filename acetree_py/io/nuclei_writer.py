"""Nuclei ZIP writer — writes nuclei text files to ZIP archives.

Creates or updates a ZIP archive with nuclei data in the standard format.

Ported from: org.rhwlab.snight.NucZipper (NucZipper.java)
"""

from __future__ import annotations

import logging
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
    logger.info("Writing nuclei to: %s (%d timepoints)", zip_path, len(nuclei_record))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, nuclei in enumerate(nuclei_record):
            time = start_time + i
            entry_name = f"{nuc_dir}t{time:03d}-nuclei"

            lines = []
            for nuc in nuclei:
                lines.append(nuc.to_text_line())

            content = "\n".join(lines) + "\n" if lines else ""
            zf.writestr(entry_name, content)

    logger.info("Wrote %d timepoints to %s", len(nuclei_record), zip_path)
