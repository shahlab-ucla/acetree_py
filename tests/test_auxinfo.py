"""Tests for acetree_py.io.auxinfo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acetree_py.io.auxinfo import AuxInfo, load_auxinfo


class TestLoadAuxInfoV1:
    """Test loading AuxInfo v1 (compressed embryo)."""

    def test_load_v1(self, sample_auxinfo_v1: Path):
        # The fixture creates "testAuxInfo.csv", so base path is "test"
        base = sample_auxinfo_v1.parent / "test"
        info = load_auxinfo(base)

        assert info.version == 1
        assert not info.is_v2
        assert info.series_name == "test"
        assert info.axis == "ADL"
        assert info.angle == 0.0
        assert info.embryo_major == 585.0
        assert info.embryo_minor == 390.0
        assert info.z_pix_res == 11.1

    def test_v1_has_no_orientation_vectors(self, sample_auxinfo_v1: Path):
        base = sample_auxinfo_v1.parent / "test"
        info = load_auxinfo(base)

        assert info.ap_orientation is None
        assert info.lr_orientation is None


class TestLoadAuxInfoV2:
    """Test loading AuxInfo v2 (uncompressed embryo)."""

    def test_load_v2(self, sample_auxinfo_v2: Path):
        base = sample_auxinfo_v2.parent / "test"
        info = load_auxinfo(base)

        assert info.version == 2
        assert info.is_v2
        assert info.series_name == "test"

    def test_v2_orientation_vectors(self, sample_auxinfo_v2: Path):
        base = sample_auxinfo_v2.parent / "test"
        info = load_auxinfo(base)

        ap = info.ap_orientation
        lr = info.lr_orientation

        assert ap is not None
        assert lr is not None
        np.testing.assert_array_almost_equal(ap, [-1, 0, 0])
        np.testing.assert_array_almost_equal(lr, [0, 0, 1])


class TestLoadAuxInfoDefaults:
    """Test fallback when no AuxInfo files exist."""

    def test_defaults_used(self, tmp_path: Path):
        base = tmp_path / "nonexistent"
        info = load_auxinfo(base)

        assert info.version == 1
        assert info.series_name == "xxxx"
        assert info.embryo_major == 585.0


class TestLoadAuxInfoV2WithV1Backup:
    """Test loading v2 with v1 as backup."""

    def test_v2_with_v1_backup(self, sample_auxinfo_v1: Path, sample_auxinfo_v2: Path):
        # Both files exist in the same directory with matching base name
        base = sample_auxinfo_v1.parent / "test"
        info = load_auxinfo(base)

        # Should load v2 as primary
        assert info.version == 2
        assert info.is_v2

        # v1 data should be available as backup
        assert len(info.data_v1) > 0
        assert info.axis == "ADL"  # from v1 backup
