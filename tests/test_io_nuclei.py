"""Tests for acetree_py.io.nuclei_reader and nuclei_writer."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from acetree_py.core.nucleus import Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.io.nuclei_writer import write_nuclei_zip


class TestReadNucleiZip:
    """Test reading nuclei from ZIP archives."""

    def test_read_sample_zip(self, sample_nuclei_zip: Path):
        nuclei_record = read_nuclei_zip(sample_nuclei_zip)

        assert len(nuclei_record) == 3  # 3 timepoints

        # Timepoint 1: 1 nucleus (P0)
        assert len(nuclei_record[0]) == 1
        assert nuclei_record[0][0].identity == "P0"

        # Timepoint 2: 2 nuclei (AB, P1)
        assert len(nuclei_record[1]) == 2
        names_t2 = {n.identity for n in nuclei_record[1]}
        assert names_t2 == {"AB", "P1"}

        # Timepoint 3: 3 nuclei (ABa, ABp, P1)
        assert len(nuclei_record[2]) == 3
        names_t3 = {n.identity for n in nuclei_record[2]}
        assert names_t3 == {"ABa", "ABp", "P1"}

    def test_read_preserves_fields(self, sample_nuclei_zip: Path):
        nuclei_record = read_nuclei_zip(sample_nuclei_zip)
        p0 = nuclei_record[0][0]

        assert p0.index == 1
        assert p0.status == 1
        assert p0.is_alive
        assert p0.x == 300
        assert p0.y == 250
        assert p0.z == 15.0
        assert p0.size == 20
        assert p0.weight == 5000
        assert p0.rwraw == 120
        assert p0.rwcorr1 == 10

    def test_read_nonexistent_zip(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_nuclei_zip(tmp_path / "nonexistent.zip")


class TestWriteNucleiZip:
    """Test writing nuclei to ZIP archives."""

    def test_write_and_read_back(self, tmp_path: Path):
        """Write nuclei, read them back, verify they match."""
        nuclei_record = [
            [
                Nucleus(index=1, x=100, y=200, z=10.0, size=15, identity="P0",
                        status=1, predecessor=-1, successor1=1, successor2=-1,
                        weight=5000, rwraw=120, rwcorr1=10),
            ],
            [
                Nucleus(index=1, x=110, y=210, z=11.0, size=16, identity="AB",
                        status=1, predecessor=1, successor1=1, successor2=-1,
                        weight=4800, rwraw=110, rwcorr1=8),
                Nucleus(index=2, x=130, y=230, z=13.0, size=18, identity="P1",
                        status=1, predecessor=1, successor1=-1, successor2=-1,
                        weight=5200, rwraw=130, rwcorr1=12),
            ],
        ]

        zip_path = tmp_path / "output.zip"
        write_nuclei_zip(nuclei_record, zip_path, start_time=1)

        # Read back
        read_back = read_nuclei_zip(zip_path)
        assert len(read_back) == 2

        # Verify timepoint 1
        assert len(read_back[0]) == 1
        p0 = read_back[0][0]
        assert p0.identity == "P0"
        assert p0.x == 100
        assert p0.rwraw == 120

        # Verify timepoint 2
        assert len(read_back[1]) == 2
        names = {n.identity for n in read_back[1]}
        assert names == {"AB", "P1"}

    def test_failed_write_preserves_last_good_archive(self, tmp_path: Path, monkeypatch):
        zip_path = tmp_path / "output.zip"
        old_contents = b"last known good archive"
        zip_path.write_bytes(old_contents)
        nucleus = Nucleus(index=1, status=1)

        def fail_serialization():
            raise RuntimeError("simulated serialization failure")

        monkeypatch.setattr(nucleus, "to_text_line", fail_serialization)

        with pytest.raises(RuntimeError, match="simulated"):
            write_nuclei_zip([[nucleus]], zip_path)

        assert zip_path.read_bytes() == old_contents
        assert list(tmp_path.glob(".output.zip.*.tmp")) == []

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics")
    def test_atomic_replace_preserves_existing_file_mode(self, tmp_path: Path):
        zip_path = tmp_path / "output.zip"
        zip_path.write_bytes(b"old")
        zip_path.chmod(0o640)

        write_nuclei_zip([[Nucleus(index=1, status=1)]], zip_path)

        assert stat.S_IMODE(zip_path.stat().st_mode) == 0o640

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file mode semantics")
    def test_new_archive_uses_normal_umask_mode(self, tmp_path: Path):
        zip_path = tmp_path / "output.zip"
        previous = os.umask(0o027)
        try:
            write_nuclei_zip([[Nucleus(index=1, status=1)]], zip_path)
        finally:
            os.umask(previous)

        assert stat.S_IMODE(zip_path.stat().st_mode) == 0o640


class TestRoundTrip:
    """Test full read -> write -> read round trip."""

    def test_round_trip_preserves_data(self, sample_nuclei_zip: Path, tmp_path: Path):
        # Read original
        original = read_nuclei_zip(sample_nuclei_zip)

        # Write to new file
        output_path = tmp_path / "round_trip.zip"
        write_nuclei_zip(original, output_path, start_time=1)

        # Read back
        round_tripped = read_nuclei_zip(output_path)

        # Compare
        assert len(round_tripped) == len(original)
        for t in range(len(original)):
            assert len(round_tripped[t]) == len(original[t])
            for i in range(len(original[t])):
                orig = original[t][i]
                rt = round_tripped[t][i]
                assert rt.index == orig.index
                assert rt.x == orig.x
                assert rt.y == orig.y
                assert rt.z == orig.z
                assert rt.size == orig.size
                assert rt.identity == orig.identity
                assert rt.status == orig.status
                assert rt.predecessor == orig.predecessor
                assert rt.successor1 == orig.successor1
                assert rt.successor2 == orig.successor2
                assert rt.rwraw == orig.rwraw
                assert rt.rwcorr1 == orig.rwcorr1
                assert rt.rwcorr2 == orig.rwcorr2
                assert rt.rwcorr3 == orig.rwcorr3
                assert rt.rwcorr4 == orig.rwcorr4
