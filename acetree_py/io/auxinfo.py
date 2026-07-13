"""AuxInfo CSV parser — embryo orientation and shape parameters.

Reads AuxInfo_v2.csv (uncompressed embryo with AP/LR orientation vectors)
and AuxInfo.csv (compressed embryo with axis string and angle) files.

These files sit alongside the XML config and provide embryo shape/orientation
parameters needed for the Sulston naming algorithm.

Ported from: org.rhwlab.snight.MeasureCSV (MeasureCSV.java)
"""

from __future__ import annotations

import csv
import logging
import os
import stat
import tempfile
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
    "orientation_source", "orientation_quality", "reference_time",
]

V2_DEFAULTS = {
    "name": "xxxx", "slope": "0.9", "intercept": "-27",
    "xc": "360", "yc": "255", "maj": "585", "min": "390",
    "zc": "14", "zslope": "10.4", "time": "160",
    "zpixres": "11.1", "AP_orientation": "XXX", "LR_orientation": "XXX",
    "orientation_source": "", "orientation_quality": "0", "reference_time": "0",
}

VALID_V1_AXES = frozenset({"ADL", "AVR", "PDR", "PVL"})


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

    @property
    def has_orientation(self) -> bool:
        """Whether this record contains a usable, explicit body orientation."""
        if self.is_v2:
            try:
                ap = self.ap_orientation
                lr = self.lr_orientation
            except ValueError:
                return False
            if ap is None or lr is None:
                return False
            if not np.all(np.isfinite(ap)) or not np.all(np.isfinite(lr)):
                return False
            ap_norm = float(np.linalg.norm(ap))
            lr_norm = float(np.linalg.norm(lr))
            if ap_norm < 1e-8 or lr_norm < 1e-8:
                return False
            return float(np.linalg.norm(np.cross(ap / ap_norm, lr / lr_norm))) >= 1e-3
        return self.axis.upper() in VALID_V1_AXES

    @property
    def is_manual(self) -> bool:
        """True for an orientation committed through the manual axis workflow."""
        return self.data.get("orientation_source", "").strip().lower() == "manual"

    @property
    def orientation_quality(self) -> float:
        try:
            return float(self.data.get("orientation_quality", "0"))
        except (TypeError, ValueError):
            return 0.0

    @property
    def reference_time(self) -> int:
        try:
            return int(float(self.data.get("reference_time", "0")))
        except (TypeError, ValueError):
            return 0

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
    v2_path = base.parent / (base.name + V2_FILE_EXT)
    v1_path = base.parent / (base.name + V1_FILE_EXT)

    v2_data: dict[str, str] | None = None
    v1_data: dict[str, str] | None = None

    # Try v2 first
    if v2_path.exists():
        try:
            v2_data = _read_csv(v2_path, V2_COLUMNS)
        except Exception as e:
            logger.warning("Failed to read AuxInfo v2: %s", e)

    # Try v1 (as primary or backup)
    if v1_path.exists():
        try:
            v1_data = _read_csv(v1_path, V1_COLUMNS)
        except Exception as e:
            logger.warning("Failed to read AuxInfo v1: %s", e)

    # A syntactically readable v2 file is not necessarily usable.  Reject
    # placeholders, malformed/non-finite vectors, and degenerate frames before
    # selecting it, so a valid legacy orientation can still be honored.
    v2_candidate: AuxInfo | None = None
    if v2_data is not None:
        v2_candidate = AuxInfo(version=2, data=v2_data, data_v1=v1_data or {})
        if v2_candidate.has_orientation:
            logger.info("Loaded usable AuxInfo v2 from: %s", v2_path)
            if v1_data is not None:
                logger.info("Loaded AuxInfo v1 as backup from: %s", v1_path)
            return v2_candidate
        logger.warning("Ignoring unusable AuxInfo v2 orientation: %s", v2_path)

    v1_candidate: AuxInfo | None = None
    if v1_data is not None:
        v1_candidate = AuxInfo(version=1, data=v1_data)
        if v1_candidate.has_orientation:
            logger.info("Loaded usable AuxInfo v1 from: %s", v1_path)
            return v1_candidate
        logger.warning("Ignoring unsupported AuxInfo v1 orientation: %s", v1_path)

    # Keep non-orientation measurements from a readable file even when neither
    # file establishes a body frame.  NucleiManager gates on has_orientation,
    # while direct AuxInfo consumers can still use z resolution and shape.
    if v2_candidate is not None:
        return v2_candidate
    if v1_candidate is not None:
        return v1_candidate

    logger.warning("No usable AuxInfo orientation found, using v1 defaults")
    return AuxInfo(version=1, data=dict(V1_DEFAULTS))


def auxinfo_from_axes(
    ap: np.ndarray,
    lr: np.ndarray,
    *,
    z_pix_res: float,
    reference_time: int = 0,
    quality: float = 1.0,
    series_name: str = "manual",
) -> AuxInfo:
    """Create an AuxInfo v2 record from a validated manual body frame."""
    ap = np.asarray(ap, dtype=float)
    lr = np.asarray(lr, dtype=float)
    if ap.shape != (3,) or lr.shape != (3,):
        raise ValueError("AP and LR vectors must both be 3D")
    if np.linalg.norm(ap) < 1e-8 or np.linalg.norm(lr) < 1e-8:
        raise ValueError("AP and LR vectors must be non-zero")
    if np.linalg.norm(np.cross(ap, lr)) < 1e-6:
        raise ValueError("AP and LR vectors must not be parallel")

    data = dict(V2_DEFAULTS)
    data.update({
        "name": series_name,
        "zpixres": f"{float(z_pix_res):.12g}",
        "AP_orientation": _format_vector(ap),
        "LR_orientation": _format_vector(lr),
        "orientation_source": "manual",
        "orientation_quality": f"{float(quality):.6g}",
        "reference_time": str(int(reference_time)),
    })
    return AuxInfo(version=2, data=data)


def write_auxinfo_v2(info: AuxInfo, config_base_path: str | Path) -> Path:
    """Persist an AuxInfo v2 sidecar next to a nuclei/config base path."""
    path, temp_path = stage_auxinfo_v2(info, config_base_path)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def auxinfo_v2_path(config_base_path: str | Path) -> Path:
    """Return the v2 sidecar path for a config or nuclei base path."""
    base = Path(config_base_path)
    return base.parent / (base.name + V2_FILE_EXT)


def stage_auxinfo_v2(
    info: AuxInfo,
    config_base_path: str | Path,
) -> tuple[Path, Path]:
    """Write a complete v2 sidecar beside its destination without committing.

    Returns ``(destination, staged_path)``.  The staged file already has the
    mode that the destination should retain after replacement.  Callers own
    the staged file and must either replace or remove it.
    """
    if not info.is_v2 or not info.has_orientation:
        raise ValueError("A valid AuxInfo v2 orientation is required")
    path = auxinfo_v2_path(config_base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(V2_COLUMNS)
    # Preserve future/third-party columns after the compatible core fields.
    columns.extend(k for k in info.data if k not in columns)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(columns)
            writer.writerow([info.data.get(col, "") for col in columns])
        os.chmod(temp_path, _replacement_mode(path))
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path, temp_path


def remove_manual_auxinfo_v2(config_base_path: str | Path) -> bool:
    """Remove a stale sidecar only when it was created by manual labelling.

    Acquisition-provided AuxInfo v2 files are never removed by this helper.
    It is used when an orientation edit is undone and the dataset is saved
    again, so reopening cannot silently resurrect the undone body frame.
    """
    path = auxinfo_v2_path(config_base_path)
    if not path.exists():
        return False
    try:
        data = _read_csv(path, V2_COLUMNS)
    except Exception:
        logger.warning("Not removing unreadable AuxInfo v2 sidecar: %s", path)
        return False
    if data.get("orientation_source", "").strip().lower() != "manual":
        return False
    path.unlink()
    return True


def is_manual_auxinfo_v2(config_base_path: str | Path) -> bool:
    """Whether the existing sidecar is readable and marked as user-created."""
    path = auxinfo_v2_path(config_base_path)
    if not path.exists():
        return False
    try:
        data = _read_csv(path, V2_COLUMNS)
    except Exception:
        logger.warning("Unreadable AuxInfo v2 sidecar is not safe to replace: %s", path)
        return False
    return data.get("orientation_source", "").strip().lower() == "manual"


def _read_csv(path: Path, expected_columns: list[str]) -> dict[str, str]:
    """Read a two-line CSV file (header + one data row).

    Returns a dict mapping column names to values.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if any(value.strip() for value in row)]

    if len(rows) < 2:
        raise ValueError(f"AuxInfo file too short (need header + data): {path}")

    names = [n.strip() for n in rows[0]]
    values = [v.strip() for v in rows[1]]

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


def _format_vector(vec: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in np.asarray(vec, dtype=float))


def _replacement_mode(destination: Path) -> int:
    """Mode for an atomic replacement, preserving target or normal defaults."""
    try:
        return stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        previous = os.umask(0)
        os.umask(previous)
        return 0o666 & ~previous
