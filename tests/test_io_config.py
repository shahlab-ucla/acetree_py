"""Tests for acetree_py.io.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from acetree_py.io.config import AceTreeConfig, NamingMethod, load_config


class TestLoadXMLConfig:
    """Test XML config file parsing."""

    def test_load_basic_config(self, sample_xml_config: Path, sample_nuclei_zip: Path):
        config = load_config(sample_xml_config)

        assert config.config_file == sample_xml_config
        assert config.zip_file == sample_nuclei_zip
        assert config.ending_index == 350
        assert config.naming_method == NamingMethod.NEWCANONICAL
        assert config.xy_res == 0.09
        assert config.z_res == 1.0
        assert config.plane_end == 30
        assert config.expr_corr == "blot"

    def test_load_nonexistent_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.xml")

    def test_z_pix_res(self, sample_xml_config: Path):
        config = load_config(sample_xml_config)
        expected = 1.0 / 0.09
        assert abs(config.z_pix_res - expected) < 0.01

    def test_config_with_naming_manual(self, tmp_path: Path):
        config_path = tmp_path / "test.xml"
        config_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <naming method="MANUAL"/>
</embryo>
""")
        config = load_config(config_path)
        assert config.naming_method == NamingMethod.MANUAL

    def test_config_with_multichannel_images(self, tmp_path: Path):
        config_path = tmp_path / "test.xml"
        config_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image numChannels="2" channel1="images/ch1/img.tif" channel2="images/ch2/img.tif"/>
</embryo>
""")
        config = load_config(config_path)
        assert config.num_channels == 2
        assert 1 in config.image_channels
        assert 2 in config.image_channels

    def test_relative_paths_resolved(self, tmp_path: Path):
        """Relative paths should be resolved against config dir."""
        config_path = tmp_path / "dats" / "test.xml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="data.zip"/>
</embryo>
""")
        config = load_config(config_path)
        # zip_file should be resolved to the dats/ directory
        assert config.zip_file.parent == config_path.parent


class TestNamingMethod:
    """Test NamingMethod enum."""

    def test_from_string_standard(self):
        assert NamingMethod.from_string("STANDARD") == NamingMethod.STANDARD

    def test_from_string_newcanonical(self):
        assert NamingMethod.from_string("NEWCANONICAL") == NamingMethod.NEWCANONICAL

    def test_from_string_unknown_defaults(self):
        assert NamingMethod.from_string("UNKNOWN") == NamingMethod.NEWCANONICAL

    def test_from_string_numeric(self):
        assert NamingMethod.from_string("3") == NamingMethod.NEWCANONICAL


# ── Java AceTree XML compatibility ─────────────────────────────────


class TestJavaXMLCompat:
    """Python's config writer must emit element tags with the casing that
    Java AceTree's case-sensitive XML reader expects:

        <useZip>, <useStack>, <Split>, <Flip>

    Python's parser is case-insensitive (normalised via ``tag.lower()``),
    so this remains backward-compatible with existing lowercase files
    on disk.
    """

    def test_casing_in_written_xml(self, tmp_path: Path):
        from acetree_py.io.config import AceTreeConfig, NamingMethod
        from acetree_py.io.config_writer import write_config_xml

        cfg_path = tmp_path / "cfg.xml"
        src = AceTreeConfig(
            config_file=cfg_path,
            zip_file=tmp_path / "nuclei.zip",
            image_file=tmp_path / "img.tif",
            naming_method=NamingMethod.NEWCANONICAL,
            use_zip=0,
            use_stack=1,
            split=1,
            flip=0,
        )
        write_config_xml(src, cfg_path)
        text = cfg_path.read_text()

        # Each expected tag must appear with Java's exact casing.
        assert "<useZip " in text
        assert "<useStack " in text
        assert "<Split " in text
        assert "<Flip " in text

        # And the lowercase forms must NOT be emitted.
        assert "<usezip " not in text
        assert "<usestack " not in text
        # Generic <split / <flip check would catch false positives inside
        # other tokens; test on a word boundary via the full tag form.
        assert "<split " not in text
        assert "<flip " not in text

    def test_roundtrip_preserves_flags(self, tmp_path: Path):
        """Round-trip write -> load should preserve all flag values
        regardless of the new casing."""
        from acetree_py.io.config import AceTreeConfig, NamingMethod
        from acetree_py.io.config_writer import write_config_xml

        cfg_path = tmp_path / "cfg.xml"
        src = AceTreeConfig(
            config_file=cfg_path,
            zip_file=tmp_path / "nuclei.zip",
            image_file=tmp_path / "img.tif",
            naming_method=NamingMethod.NEWCANONICAL,
            use_zip=2,
            use_stack=1,
            split=1,
            flip=1,
        )
        write_config_xml(src, cfg_path)
        loaded = load_config(cfg_path)
        assert loaded.use_zip == 2
        assert loaded.use_stack == 1
        assert loaded.split == 1
        assert loaded.flip == 1

    def test_naming_method_written_as_numeric(self, tmp_path: Path):
        """Java's NucleiConfig parses the naming method with Integer.parseInt,
        so the written value must be numeric (the enum's int value), not the
        enum name.  Python's parser accepts both."""
        from acetree_py.io.config import AceTreeConfig, NamingMethod
        from acetree_py.io.config_writer import write_config_xml

        cfg_path = tmp_path / "cfg.xml"
        src = AceTreeConfig(
            config_file=cfg_path,
            zip_file=tmp_path / "nuclei.zip",
            image_file=tmp_path / "img.tif",
            naming_method=NamingMethod.NEWCANONICAL,
        )
        write_config_xml(src, cfg_path)
        text = cfg_path.read_text()
        # Must emit the enum's numeric value, not the name.
        assert '<naming method="3"' in text
        assert '<naming method="NEWCANONICAL"' not in text

        # And a round-trip must still restore the enum.
        loaded = load_config(cfg_path)
        assert loaded.naming_method == NamingMethod.NEWCANONICAL

    def test_parser_accepts_legacy_lowercase(self, tmp_path: Path):
        """Parser is case-insensitive, so old Python-generated configs
        with lowercase tags still load correctly."""
        cfg = tmp_path / "legacy.xml"
        cfg.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="n.zip"/>
    <image file="img.tif"/>
    <usezip type="2"/>
    <usestack type="1"/>
    <split SplitMode="1"/>
    <flip FlipMode="0"/>
</embryo>
""")
        config = load_config(cfg)
        assert config.use_zip == 2
        assert config.use_stack == 1
        assert config.split == 1
        assert config.flip == 0
