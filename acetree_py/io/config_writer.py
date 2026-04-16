"""Write AceTreeConfig to XML — inverse of config.py's load_config().

Produces the same XML schema that _parse_xml_config() reads, enabling
round-trip: load_config(path) → write_config_xml(config, path2) → load_config(path2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

from .config import AceTreeConfig

logger = logging.getLogger(__name__)


def write_config_xml(config: AceTreeConfig, path: str | Path) -> None:
    """Serialize an AceTreeConfig to an XML file.

    Args:
        config: The configuration to write.
        path: Destination file path.
    """
    path = Path(path)
    config_dir = path.parent

    def _rel(p: Path) -> str:
        """Make *p* relative to the config directory if possible."""
        try:
            return str(p.relative_to(config_dir))
        except ValueError:
            return str(p)

    root = Element("embryo")

    # <nuclei file="..."/>
    if config.zip_file and str(config.zip_file) not in ("", "."):
        SubElement(root, "nuclei", file=_rel(config.zip_file))

    # <image file="..."/> or <image numChannels="..." channel1="..." .../>
    # or <image file="..." numChannels="N" channelOrder="CZ"/> (interleaved)
    if config.image_channels and len(config.image_channels) > 1:
        attrs = {"numChannels": str(len(config.image_channels))}
        for ch_num, ch_path in sorted(config.image_channels.items()):
            attrs[f"channel{ch_num}"] = _rel(ch_path)
        SubElement(root, "image", **attrs)
    elif config.image_file and str(config.image_file) not in ("", "."):
        if config.stack_interleaved and config.num_channels > 1:
            SubElement(
                root, "image",
                file=_rel(config.image_file),
                numChannels=str(config.num_channels),
                channelOrder=config.stack_channel_order or "CZ",
            )
        else:
            SubElement(root, "image", file=_rel(config.image_file))

    # <start index="..."/>
    SubElement(root, "start", index=str(config.starting_index))

    # <end index="..."/>
    SubElement(root, "end", index=str(config.ending_index))

    # <naming method="..."/>
    SubElement(root, "naming", method=config.naming_method.name)

    # <axis axis="..."/>
    if config.axis_given:
        SubElement(root, "axis", axis=config.axis_given)

    # <polar size="..."/>
    if config.polar_size != 45:  # only write non-default
        SubElement(root, "polar", size=str(config.polar_size))

    # <resolution xyRes="..." zRes="..." planeEnd="..."/>
    SubElement(
        root, "resolution",
        xyRes=str(config.xy_res),
        zRes=str(config.z_res),
        planeEnd=str(config.plane_end),
    )

    # <exprCorr type="..."/>
    if config.expr_corr and config.expr_corr != "none":
        SubElement(root, "exprCorr", type=config.expr_corr)

    # <useZip type="..."/> — CamelCase for Java AceTree compatibility.
    # Python's own parser lowercases tags, so this remains backward-compatible
    # with lowercase files already on disk.
    SubElement(root, "useZip", type=str(config.use_zip))

    # <useStack type="..."/>
    SubElement(root, "useStack", type=str(config.use_stack))

    # <Split SplitMode="..."/> — PascalCase tag required by Java's
    # case-sensitive XML reader (see org.rhwlab.snight.XMLConfig).
    SubElement(root, "Split", SplitMode=str(config.split))

    # <Flip FlipMode="..."/>
    SubElement(root, "Flip", FlipMode=str(config.flip))

    # <angle degrees="..."/>
    if config.angle != -1.0:
        SubElement(root, "angle", degrees=str(config.angle))

    indent(root)
    tree = ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(path), xml_declaration=True, encoding="utf-8")
    logger.info("Wrote config XML to %s", path)
