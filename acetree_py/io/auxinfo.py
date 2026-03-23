"""AuxInfo CSV parser — embryo orientation and shape parameters.

Reads AuxInfo_v2.csv (uncompressed embryo with AP/LR orientation vectors)
and AuxInfo.csv (compressed embryo with axis string and angle) files.

These files sit alongside the XML config and provide embryo shape/orientation
parameters needed for the Sulston naming algorithm.

Ported from: org.rhwlab.snight.MeasureCSV (MeasureCSV.java)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# File extensions (appended to config base name)
V2_FILE_EXT = "AuxInfo_v2.csv"
V1_FILE_EXT = "AuxInfo.csv"

# ── V1 column names and defaults ──
V1_COLUMNS = [
    "name", "slope", "intercept", "xc", "yc", "maj", "min",
    "ang", "zc", "zslope", "time", "zpixres", "axis",
]

V1_DEFAULTS = {
    "name": "xxxx", "slope": "0.9", "intercept": "-27",
    "xc": "360", "yc": "255", "maj": "585", "min": "390",
    "ang": "0", "zc": "14", "zslope": "10.4", "time": "160",
    "zpixres": "11.1", "axis": "XXX",
}

# ── V2 column names and defaults ──
V2_COLUMNS = [
    "name", "slope", "intercept", "xc", "yc", "maj", "min",
    "zc", "zslope", "time", "zpixres", "AP_orientation", "LR_orientation",
]

V2_DEFAULTS = {
    "name": "xxxx", "slope": "0.9", "intercept": "-27",
    "xc": "360", "yc": "255", "maj": "585", "min": "390",
    "zc": "14", "zslope": "10.4", "time": "160",
    "zpixres": "11.1", "AP_orientation": "XXX", "LR_orientation": "XXX",
}


@dataclass
class AuxInfo:
    """Embryo orientation and shape parameters from AuxInfo CSV files.

    Attributes:
        version: 1 for compressed embryos, 2 for uncompressed.
        data: Dict of column_name -> string value.
        data_v1: Backup v1 data (used as fallback if v2 fails).
    """

    version: int = 1
    data: dict[str, str] = field(default_factory=dict)
    data_v1: dict[str, str] = field(default_factory=dict)

    @property
    def is_v2(self) -> bool:
        """True if using AuxInfo v2 (uncompressed embryo)."""
        return self.version == 2

    # ── Common accessors ──

    @property
    def series_name(self) -> str:
        return self.data.get("name", "xxxx")

    @property
    def embryo_major(self) -> float:
        return float(self.data.get("maj", "585"))

    @property
    def embryo_minor(self) -> float:
        return float(self.data.get("min", "390"))

    @property
    def z_slope(self) -> float:
        return float(self.data.get("zslope", "10.4"))

    @property
    def z_pix_res(self) -> float:
        return float(self.data.get("zpixres", "11.1"))

    # ── V1-specific accessors ──

    @property
    def axis(self) -> str:
        """Axis string for v1 (e.g. 'ADL', 'AVR')."""
        if self.is_v2:
            return self.data_v1.get("axis", "XXX")
        return self.data.get("axis", "XXX")

    @property
    def angle(self) -> float:
        """Rotation angle for v1 compressed embryos (in degrees)."""
        if self.is_v2:
            ang_str = self.data_v1.get("ang", "0")
        else:
            ang_str = self.data.get("ang", "0")
        return float(ang_str) if ang_str else 0.0

    # ── V2-specific accessors ──

    @property
    def ap_orientation(self) -> np.ndarray | None:
        """AP orientation vector for v2 uncompressed embryos."""
        if not self.is_v2:
            return None
        vec_str = self.data.get("AP_orientation", "")
        if not vec_str or vec_str == "XXX":
            return None
        return _parse_vector(vec_str)

    @property
    def lr_orientation(self) -> np.ndarray | None:
        """LR orientation vector for v2 uncompressed embryos."""
        if not self.is_v2:
            return None
        vec_str = self.data.get("LR_orientation", "")
        if not vec_str or vec_str == "XXX":
            return None
        return _parse_vector(vec_str)


def load_auxinfo(config_base_path: str | Path) -> AuxInfo:
    """Load AuxInfo data from CSV files.

    Tries v2 first, falls back to v1, then defaults.

    Args:
        config_base_path: Base file path without the AuxInfo extension.
            E.g. if config is "/data/081505.xml", pass "/data/081505".

    Returns:
        AuxInfo instance with data populated.
    """
    base = Path(config_base_path)
    info = AuxInfo()

    v2_path = base.parent / (base.name + V2_FILE_EXT)
    v1_path = base.parent / (base.name + V1_FILE_EXT)

    v2_loaded = False
    v1_loaded = False

    # Try v2 first
    if v2_path.exists():
        try:
            info.data = _read_csv(v2_path, V2_COLUMNS)
            info.version = 2
            v2_loaded = True
            logger.info("Loaded AuxInfo v2 from: %s", v2_path)
        except Exception as e:
            logger.warning("Failed to read AuxInfo v2: %s", e)

    # Try v1 (as primary or backup)
    if v1_path.exists():
        try:
            v1_data = _read_csv(v1_path, V1_COLUMNS)
            if v2_loaded:
                info.data_v1 = v1_data
                logger.info("Loaded AuxInfo v1 as backup from: %s", v1_path)
            else:
                info.data = v1_data
                info.version = 1
                logger.info("Loaded AuxInfo v1 from: %s", v1_path)
            v1_loaded = True
        except Exception as e:
            logger.warning("Failed to read AuxInfo v1: %s", e)

    # If neither loaded, use defaults
    if not v2_loaded and not v1_loaded:
        logger.warning("No AuxInfo file found, using defaults (v1 conventions)")
        info.data = dict(V1_DEFAULTS)
        info.version = 1

    return info


def _read_csv(path: Path, expected_columns: list[str]) -> dict[str, str]:
    """Read a two-line CSV file (header + one data row).

    Returns a dict mapping column names to values.
    """
    with open(path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"AuxInfo file too short (need header + data): {path}")

    names = [n.strip() for n in lines[0].split(",")]
    values = [v.strip() for v in lines[1].split(",")]

    result = {}
    for i, name in enumerate(names):
        if i < len(values):
            result[name] = values[i]
        else:
            result[name] = ""

    return result


def _parse_vector(s: str) -> np.ndarray:
    """Parse a space-separated 3D vector string into a numpy array.

    Args:
        s: String like "0.5 0.3 -0.8" or "-1 0 0".

    Returns:
        numpy array of shape (3,).

    Raises:
        ValueError: If the string cannot be parsed as a 3D vector.
    """
    parts = s.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3D vector, got: '{s}'")
    return np.array([float(p) for p in parts])
