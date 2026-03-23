"""Tests for acetree_py.core.nucleus."""

from __future__ import annotations

import pytest

from acetree_py.core.nucleus import NILLI, Nucleus


class TestNucleusCreation:
    """Test Nucleus construction and basic properties."""

    def test_default_nucleus(self):
        nuc = Nucleus()
        assert nuc.index == 0
        assert nuc.status == -1
        assert nuc.predecessor == NILLI
        assert nuc.successor1 == NILLI
        assert nuc.successor2 == NILLI
        assert nuc.identity == ""
        assert nuc.assigned_id == ""
        assert not nuc.is_alive
        assert not nuc.is_dividing

    def test_alive_nucleus(self):
        nuc = Nucleus(status=1)
        assert nuc.is_alive

    def test_dividing_nucleus(self):
        nuc = Nucleus(successor1=1, successor2=2)
        assert nuc.is_dividing

    def test_non_dividing_nucleus(self):
        nuc = Nucleus(successor1=1, successor2=NILLI)
        assert not nuc.is_dividing

    def test_effective_name_uses_assigned_id(self):
        nuc = Nucleus(identity="ABa", assigned_id="ForcedName")
        assert nuc.effective_name == "ForcedName"

    def test_effective_name_falls_back_to_identity(self):
        nuc = Nucleus(identity="ABa", assigned_id="")
        assert nuc.effective_name == "ABa"


class TestNucleusFromTextLine:
    """Test parsing from comma-separated text lines."""

    def test_parse_full_line(self):
        line = "1, 1, -1, 1, -1, 300, 250, 15.0, 20, ABa, 5000, 100, 50, 25, , 120, 10, 15, 12, 8"
        nuc = Nucleus.from_text_line(line)

        assert nuc.index == 1
        assert nuc.status == 1
        assert nuc.predecessor == NILLI
        assert nuc.successor1 == 1
        assert nuc.successor2 == NILLI
        assert nuc.x == 300
        assert nuc.y == 250
        assert nuc.z == 15.0
        assert nuc.size == 20
        assert nuc.identity == "ABa"
        assert nuc.weight == 5000
        assert nuc.rweight == 100
        assert nuc.rsum == 50
        assert nuc.rcount == 25
        assert nuc.assigned_id == ""
        assert nuc.rwraw == 120
        assert nuc.rwcorr1 == 10
        assert nuc.rwcorr2 == 15
        assert nuc.rwcorr3 == 12
        assert nuc.rwcorr4 == 8

    def test_parse_dividing_nucleus(self):
        line = "2, 1, -1, 2, 3, 400, 300, 16.5, 22, P1, 6000, 200, 60, 30, , 250, 20, 25, 18, 12"
        nuc = Nucleus.from_text_line(line)

        assert nuc.successor1 == 2
        assert nuc.successor2 == 3
        assert nuc.is_dividing
        assert nuc.is_alive

    def test_parse_dead_nucleus(self):
        line = "3, -1, -1, -1, -1, 100, 100, 5.0, 10, , 1000, 0, 0, 0, , 0, 0, 0, 0, 0"
        nuc = Nucleus.from_text_line(line)

        assert nuc.status == -1
        assert not nuc.is_alive
        assert nuc.identity == ""

    def test_parse_with_assigned_id(self):
        line = "4, 1, 1, 4, -1, 350, 280, 14.0, 18, ABa, 4500, 90, 45, 20, ForcedName, 110, 8, 12, 10, 6"
        nuc = Nucleus.from_text_line(line)

        assert nuc.assigned_id == "ForcedName"
        assert nuc.effective_name == "ForcedName"

    def test_parse_nill_links(self):
        line = "5, 1, nill, nill, -1, 200, 200, 10.0, 15, P0, 3000, 50, 20, 10, , 60, 5, 8, 6, 4"
        nuc = Nucleus.from_text_line(line)

        assert nuc.predecessor == NILLI
        assert nuc.successor1 == NILLI

    def test_parse_minimal_line(self):
        """Lines without red channel data should still parse."""
        line = "1, 1, -1, 1, -1, 300, 250, 15.0, 20, ABa, 5000"
        nuc = Nucleus.from_text_line(line)

        assert nuc.index == 1
        assert nuc.identity == "ABa"
        assert nuc.rweight == 0  # defaults
        assert nuc.rwraw == 0

    def test_parse_status_zero_treated_as_dead(self):
        """Status = 0 should be treated as dead (status > 0 for alive)."""
        line = "1, 0, -1, -1, -1, 100, 100, 5.0, 10, test, 1000"
        nuc = Nucleus.from_text_line(line)
        assert nuc.status == -1
        assert not nuc.is_alive


class TestNucleusToTextLine:
    """Test serialization back to text."""

    def test_round_trip(self):
        """Parse a line, serialize it, parse again — should match."""
        original = "1, 1, -1, 1, -1, 300, 250, 15.0, 20, ABa, , 5000, 100, 120, 10, 15, 12, 8"
        nuc = Nucleus.from_text_line(
            "1, 1, -1, 1, -1, 300, 250, 15.0, 20, ABa, 5000, 100, 50, 25, , 120, 10, 15, 12, 8"
        )
        text = nuc.to_text_line()

        # Parse the output
        nuc2 = Nucleus.from_text_line(text)

        assert nuc2.index == nuc.index
        assert nuc2.x == nuc.x
        assert nuc2.y == nuc.y
        assert nuc2.z == nuc.z
        assert nuc2.size == nuc.size
        assert nuc2.identity == nuc.identity
        assert nuc2.status == nuc.status
        assert nuc2.predecessor == nuc.predecessor
        assert nuc2.successor1 == nuc.successor1
        assert nuc2.successor2 == nuc.successor2
        assert nuc2.rwraw == nuc.rwraw
        assert nuc2.rwcorr1 == nuc.rwcorr1
        assert nuc2.rwcorr2 == nuc.rwcorr2
        assert nuc2.rwcorr3 == nuc.rwcorr3
        assert nuc2.rwcorr4 == nuc.rwcorr4


class TestNucleusCorrectedRed:
    """Test red channel correction methods."""

    def test_no_correction(self):
        nuc = Nucleus(rwraw=100, rwcorr1=10, rwcorr2=15, rwcorr3=12, rwcorr4=8)
        assert nuc.corrected_red("none") == 100

    def test_global_correction(self):
        nuc = Nucleus(rwraw=100, rwcorr1=10)
        assert nuc.corrected_red("global") == 90

    def test_local_correction(self):
        nuc = Nucleus(rwraw=100, rwcorr2=15)
        assert nuc.corrected_red("local") == 85

    def test_blot_correction(self):
        nuc = Nucleus(rwraw=100, rwcorr3=12)
        assert nuc.corrected_red("blot") == 88

    def test_cross_correction(self):
        nuc = Nucleus(rwraw=100, rwcorr4=8)
        assert nuc.corrected_red("cross") == 92


class TestNucleusCopy:
    """Test deep copy."""

    def test_copy_is_independent(self):
        nuc = Nucleus(index=1, x=100, identity="ABa", status=1)
        copy = nuc.copy()

        assert copy.index == nuc.index
        assert copy.identity == nuc.identity

        # Modify copy, original should be unchanged
        copy.identity = "Modified"
        assert nuc.identity == "ABa"
