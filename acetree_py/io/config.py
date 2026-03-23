"""Configuration file parser for AceTree XML config files.

Parses .xml config files that specify the nuclei ZIP, image paths,
naming method, resolution, and other dataset parameters.

Ported from: org.rhwlab.snight.Config, XMLConfig, NucleiConfig, ImageConfig
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class NamingMethod(IntEnum):
    """Cell naming algorithm to use."""

    STANDARD = 2
    MANUAL = 2       # Intentionally same as STANDARD in Java
    NEWCANONICAL = 3

    @classmethod
    def from_string(cls, s: str) -> NamingMethod:
        """Parse naming method from config string."""
        s = s.strip().upper()
        if s == "STANDARD":
            return cls.STANDARD
        elif s == "MANUAL":
            return cls.MANUAL
        elif s == "NEWCANONICAL" or s == "3":
            return cls.NEWCANONICAL
        else:
            try:
                return cls(int(s))
            except (ValueError, KeyError):
                logger.warning("Unknown naming method '%s', defaulting to NEWCANONICAL", s)
                return cls.NEWCANONICAL


# Red expression correction methods
RED_CORRECTION_METHODS = ("none", "global", "local", "blot", "cross")


@dataclass
class AceTreeConfig:
    """Complete configuration for an AceTree dataset.

    This replaces the Java Config/NucleiConfig/ImageConfig trio with
    a single clean dataclass.

    Attributes:
        config_file: Path to the XML config file.
        zip_file: Path to the nuclei ZIP archive.
        image_file: Path to a typical image (used to infer image directory, prefix, format).
        image_channels: Dict of channel_num -> image_path for multi-channel configs.
        num_channels: Number of image channels.
        naming_method: Which naming algorithm to use.
        starting_index: First timepoint to process.
        ending_index: Last timepoint to process.
        xy_res: XY pixel resolution in microns.
        z_res: Z-plane spacing in microns.
        plane_end: Last z-plane index.
        polar_size: Polar body filter size.
        axis_given: Pre-specified axis orientation string (e.g. "adl", "avr").
        expr_corr: Expression correction method.
        use_zip: Image ZIP mode (0=tif files, 2=tifs in zip).
        use_stack: Stack type (0=8-bit, 1=16-bit).
        split: Whether to split 16-bit channels.
        start_time: Starting timepoint from image name parsing.
    """

    config_file: Path = Path()
    zip_file: Path = Path()
    image_file: Path = Path()
    image_channels: dict[int, Path] = field(default_factory=dict)
    num_channels: int = 1
    naming_method: NamingMethod = NamingMethod.NEWCANONICAL
    starting_index: int = 1
    ending_index: int = 9999
    xy_res: float = 0.09
    z_res: float = 1.0
    plane_start: int = 1
    plane_end: int = 50
    polar_size: int = 45
    axis_given: str = ""
    expr_corr: str = "none"
    use_zip: int = 0
    use_stack: int = 0
    split: int = 1
    flip: int = 1
    start_time: int = 1
    angle: float = -1.0

    # Derived fields set after parsing
    tif_prefix: str = ""
    tif_directory: Path = Path()

    @property
    def z_pix_res(self) -> float:
        """Z resolution in pixel units (z_res / xy_res)."""
        if self.xy_res == 0:
            return 1.0
        return self.z_res / self.xy_res

    @property
    def config_dir(self) -> Path:
        """Directory containing the config file."""
        return self.config_file.parent


def load_config(path: str | Path) -> AceTreeConfig:
    """Load an AceTree configuration from an XML file.

    Args:
        path: Path to the .xml config file.

    Returns:
        An AceTreeConfig instance with all settings populated.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.info("Loading config from: %s", path)

    config = AceTreeConfig(config_file=path)

    if path.suffix.lower() == ".xml":
        _parse_xml_config(path, config)
    else:
        _parse_legacy_config(path, config)

    # Resolve relative paths against config file directory
    _resolve_paths(config)

    logger.info("Config loaded: zip=%s, naming=%s, times=%d-%d",
                config.zip_file, config.naming_method.name,
                config.starting_index, config.ending_index)

    return config


def _parse_xml_config(path: Path, config: AceTreeConfig) -> None:
    """Parse an XML configuration file.

    Expected format:
        <?xml version='1.0' encoding='utf-8'?>
        <embryo>
            <nuclei file="path/to/nuclei.zip"/>
            <image file="path/to/typical_image.tif"/>
            <end index="350"/>
            <naming method="NEWCANONICAL"/>
            <axis axis="adl"/>
            <resolution xyRes="0.09" zRes="1.0" planeEnd="30"/>
            <exprCorr type="blot"/>
            ...
        </embryo>
    """
    tree = ET.parse(path)
    root = tree.getroot()

    for elem in root:
        tag = elem.tag.lower()

        if tag == "nuclei":
            config.zip_file = Path(elem.get("file", ""))

        elif tag == "image":
            if "file" in elem.attrib:
                config.image_file = Path(elem.get("file", ""))
            elif "numchannels" in {k.lower(): k for k in elem.attrib}:
                # Multi-channel image definition
                nc_key = next(k for k in elem.attrib if k.lower() == "numchannels")
                config.num_channels = int(elem.get(nc_key, "1"))
                for i in range(1, config.num_channels + 1):
                    ch_key = f"channel{i}"
                    # Case-insensitive lookup
                    for k in elem.attrib:
                        if k.lower() == ch_key.lower():
                            config.image_channels[i] = Path(elem.get(k, ""))
                            break

        elif tag == "start":
            val = elem.get("index", "")
            if val:
                config.starting_index = int(val)

        elif tag == "end":
            val = elem.get("index", "")
            if val:
                config.ending_index = int(val)

        elif tag == "naming":
            method = elem.get("method", "")
            if method:
                config.naming_method = NamingMethod.from_string(method)

        elif tag == "axis":
            config.axis_given = elem.get("axis", "")

        elif tag == "polar":
            val = elem.get("size", "")
            if val:
                config.polar_size = int(val)

        elif tag == "resolution":
            xy = elem.get("xyRes", "")
            if xy:
                config.xy_res = float(xy)
            z = elem.get("zRes", "")
            if z:
                config.z_res = float(z)
            pe = elem.get("planeEnd", "")
            if pe:
                config.plane_end = int(pe)

        elif tag == "exprcorr":
            val = elem.get("type", "")
            if val:
                config.expr_corr = val

        elif tag == "usezip":
            val = elem.get("type", "")
            if val:
                config.use_zip = int(val)

        elif tag == "usestack":
            val = elem.get("type", "")
            if val:
                config.use_stack = int(val)

        elif tag == "split":
            val = elem.get("SplitMode", elem.get("splitmode", ""))
            if val:
                config.split = int(val)

        elif tag == "flip":
            val = elem.get("FlipMode", elem.get("flipmode", elem.get("type", "")))
            if val:
                config.flip = int(val)

        elif tag == "angle":
            val = elem.get("degrees", "")
            if val:
                config.angle = float(val)

        else:
            logger.debug("Ignoring unknown XML tag: %s", tag)


def _parse_legacy_config(path: Path, config: AceTreeConfig) -> None:
    """Parse a legacy (non-XML) comma-separated config file.

    Format: key, value (one per line, '#' for comments)
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ", " not in line:
                continue
            key, _, value = line.partition(", ")
            key = key.strip()
            value = value.strip()

            if key == "zipFileName":
                config.zip_file = Path(value)
            elif key == "typical image":
                config.image_file = Path(value)
            elif key == "starting index" and value:
                config.starting_index = int(value)
            elif key == "ending index" and value:
                config.ending_index = int(value)
            elif key == "namingMethod" and value:
                config.naming_method = NamingMethod.from_string(value)
            elif key == "axis" and value:
                config.axis_given = value
            elif key == "xyRes" and value:
                config.xy_res = float(value)
            elif key == "zRes" and value:
                config.z_res = float(value)
            elif key == "planeEnd" and value:
                config.plane_end = int(value)
            elif key == "exprCorr" and value:
                config.expr_corr = value
            elif key == "use zip" and value:
                config.use_zip = int(value)
            elif key == "use stack" and value:
                config.use_stack = int(value)
            elif key == "polarSize" and value:
                config.polar_size = int(value)
            elif key == "flip" and value:
                config.flip = int(value)
            elif key == "split" and value:
                config.split = int(value)


def _resolve_paths(config: AceTreeConfig) -> None:
    """Resolve relative paths against the config file's directory."""
    base = config.config_dir

    if config.zip_file and not config.zip_file.is_absolute():
        config.zip_file = (base / config.zip_file).resolve()

    if config.image_file and str(config.image_file) and not config.image_file.is_absolute():
        config.image_file = (base / config.image_file).resolve()

    for ch_num, ch_path in list(config.image_channels.items()):
        if ch_path and not ch_path.is_absolute():
            config.image_channels[ch_num] = (base / ch_path).resolve()

    # Derive tif_directory and tif_prefix from image_file
    if config.image_file and str(config.image_file):
        _derive_image_params(config)


def _derive_image_params(config: AceTreeConfig) -> None:
    """Extract tif_directory and tif_prefix from the image_file path.

    Given a typical image path like 'Y:/data/SPIMA_t1.tif', extracts:
    - tif_directory = Path('Y:/data')
    - tif_prefix = 'SPIMA_t'
    - start_time = 1

    Handles patterns like:
    - {prefix}t{N}.tif  (stack TIFFs, e.g. SPIMA_t1.tif)
    - {prefix}t{NNN}-p{NN}.tif  (per-plane TIFFs)
    - {prefix}t{NNN}-p{NN}.zip  (per-plane ZIPs)
    """
    import re

    image_path = config.image_file
    config.tif_directory = image_path.parent

    filename = image_path.name
    stem = image_path.stem  # Without extension

    # Try to find a time number in the filename
    # Pattern: something ending in 't' followed by digits, optionally followed by '-p' + digits
    match = re.match(r'^(.+?t)(\d+)(?:-p(\d+))?$', stem, re.IGNORECASE)
    if match:
        config.tif_prefix = match.group(1)  # e.g. "SPIMA_t"
        config.start_time = int(match.group(2))  # e.g. 1
        logger.info("Derived image params: dir=%s, prefix='%s', start_time=%d",
                     config.tif_directory, config.tif_prefix, config.start_time)
    else:
        # Fallback: use filename without digits at the end
        match2 = re.match(r'^(.+?)(\d+)$', stem)
        if match2:
            config.tif_prefix = match2.group(1)
            config.start_time = int(match2.group(2))
            logger.info("Derived image params (fallback): dir=%s, prefix='%s'",
                         config.tif_directory, config.tif_prefix)
        else:
            logger.warning("Could not parse image filename pattern from: %s", filename)
