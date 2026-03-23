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
