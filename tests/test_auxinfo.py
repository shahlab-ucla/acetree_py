"""Tests for acetree_py.io.auxinfo."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest

from acetree_py.io.auxinfo import (
    AuxInfo,
    auxinfo_from_axes,
    load_auxinfo,
    write_auxinfo_v2,
)


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

    def test_default_placeholder_axis_is_not_an_orientation(self, tmp_path: Path):
        info = load_auxinfo(tmp_path / "nonexistent")
        assert info.axis == "XXX"
        assert not info.has_orientation


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

    @pytest.mark.parametrize(
        ("ap", "lr"),
        [
            ("1 0 0", "2 0 0"),
            ("not-a-vector", "0 1 0"),
            ("nan 0 0", "0 1 0"),
            ("XXX", "XXX"),
        ],
    )
    def test_unusable_v2_falls_back_to_valid_v1(
        self,
        tmp_path: Path,
        ap: str,
        lr: str,
    ):
        base = tmp_path / "embryo"
        (tmp_path / "embryoAuxInfo_v2.csv").write_text(
            "name,AP_orientation,LR_orientation\n"
            f"bad-v2,{ap},{lr}\n",
            encoding="utf-8",
        )
        (tmp_path / "embryoAuxInfo.csv").write_text(
            "name,ang,axis\nlegacy,12,PDR\n",
            encoding="utf-8",
        )

        info = load_auxinfo(base)

        assert not info.is_v2
        assert info.has_orientation
        assert info.axis == "PDR"
        assert info.angle == 12.0

    def test_unusable_files_keep_measurements_without_claiming_orientation(
        self,
        tmp_path: Path,
    ):
        base = tmp_path / "embryo"
        (tmp_path / "embryoAuxInfo_v2.csv").write_text(
            "name,zpixres,AP_orientation,LR_orientation\n"
            "measured,2.75,1 0 0,2 0 0\n",
            encoding="utf-8",
        )
        (tmp_path / "embryoAuxInfo.csv").write_text(
            "axis\nPDL\n",
            encoding="utf-8",
        )

        info = load_auxinfo(base)

        assert info.is_v2
        assert info.series_name == "measured"
        assert info.z_pix_res == 2.75
        assert not info.has_orientation


class TestManualAuxInfo:
    def test_manual_orientation_round_trips_with_provenance(self, tmp_path: Path):
        info = auxinfo_from_axes(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            z_pix_res=2.5,
            reference_time=17,
            quality=0.8,
            series_name="embryo, corrected",
        )

        path = write_auxinfo_v2(info, tmp_path / "sample")
        loaded = load_auxinfo(tmp_path / "sample")

        assert path.name == "sampleAuxInfo_v2.csv"
        assert loaded.has_orientation
        assert loaded.is_manual
        assert loaded.reference_time == 17
        assert loaded.orientation_quality == pytest.approx(0.8)
        assert loaded.series_name == "embryo, corrected"
        np.testing.assert_allclose(loaded.ap_orientation, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(loaded.lr_orientation, [0.0, 1.0, 0.0])

    def test_v1_orientation_requires_a_supported_code(self):
        assert AuxInfo(version=1, data={"axis": "AVR"}).has_orientation
        assert not AuxInfo(version=1, data={"axis": "garbage"}).has_orientation

    def test_parallel_v2_vectors_are_not_usable(self):
        info = AuxInfo(
            version=2,
            data={"AP_orientation": "1 0 0", "LR_orientation": "2 0 0"},
        )
        assert not info.has_orientation

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics")
    def test_atomic_replace_preserves_existing_file_mode(self, tmp_path: Path):
        path = tmp_path / "sampleAuxInfo_v2.csv"
        path.write_text("old", encoding="utf-8")
        path.chmod(0o640)
        info = auxinfo_from_axes(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            z_pix_res=1.0,
        )

        write_auxinfo_v2(info, tmp_path / "sample")

        assert stat.S_IMODE(path.stat().st_mode) == 0o640

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics")
    def test_new_sidecar_uses_normal_umask_mode(self, tmp_path: Path):
        info = auxinfo_from_axes(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            z_pix_res=1.0,
        )
        previous = os.umask(0o027)
        try:
            path = write_auxinfo_v2(info, tmp_path / "sample")
        finally:
            os.umask(previous)

        assert stat.S_IMODE(path.stat().st_mode) == 0o640
