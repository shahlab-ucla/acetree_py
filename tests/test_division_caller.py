"""Tests for acetree_py.naming.division_caller and rules."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.core.nucleus import Nucleus
from acetree_py.naming.canonical_transform import CanonicalTransform
from acetree_py.naming.division_caller import DivisionCaller
from acetree_py.naming.rules import Rule, RuleManager
from acetree_py.naming.sulston_names import complement, daughter_names


class TestSulstonNames:
    """Test complement function and naming helpers."""

    def test_complement_a_p(self):
        assert complement("a") == "p"
        assert complement("p") == "a"

    def test_complement_l_r(self):
        assert complement("l") == "r"
        assert complement("r") == "l"

    def test_complement_d_v(self):
        assert complement("d") == "v"
        assert complement("v") == "d"

    def test_complement_unknown(self):
        assert complement("x") == "g"
        assert complement("") == "g"

    def test_daughter_names(self):
        d1, d2 = daughter_names("ABa", "l")
        assert d1 == "ABal"
        assert d2 == "ABar"

    def test_daughter_names_ap(self):
        d1, d2 = daughter_names("P1", "a")
        assert d1 == "P1a"
        assert d2 == "P1p"


class TestRuleManager:
    """Test rule loading and generation."""

    def test_loads_precomputed_rules(self):
        rm = RuleManager()
        assert rm.num_precomputed > 600  # NewRules.txt has ~621 entries

    def test_loads_names_hash(self):
        rm = RuleManager()
        assert rm.num_hash_entries > 50  # namesHash.txt has ~61 entries

    def test_get_precomputed_rule(self):
        rm = RuleManager()
        rule = rm.get_rule("AB")
        assert rule.parent == "AB"
        assert rule.daughter1 == "ABa"
        assert rule.daughter2 == "ABp"
        assert rule.axis_vector is not None
        assert len(rule.axis_vector) == 3

    def test_get_rule_ems(self):
        rm = RuleManager()
        rule = rm.get_rule("EMS")
        assert rule.daughter1 == "E"
        assert rule.daughter2 == "MS"

    def test_get_rule_p1(self):
        rm = RuleManager()
        rule = rm.get_rule("P1")
        assert rule.daughter1 == "EMS"
        assert rule.daughter2 == "P2"

    def test_generates_default_rule_for_unknown(self):
        rm = RuleManager()
        rule = rm.get_rule("UnknownCell")
        assert rule.parent == "UnknownCell"
        assert rule.daughter1 == "UnknownCella"
        assert rule.daughter2 == "UnknownCellp"

    def test_generated_rule_is_cached(self):
        rm = RuleManager()
        rule1 = rm.get_rule("MyCell")
        rule2 = rm.get_rule("MyCell")
        assert rule1 is rule2

    def test_rule_axis_vector_is_numpy_array(self):
        rm = RuleManager()
        rule = rm.get_rule("AB")
        assert isinstance(rule.axis_vector, np.ndarray)
        assert rule.axis_vector.dtype == np.float64


class TestDivisionCaller:
    """Test DivisionCaller name assignment."""

    @pytest.fixture
    def rule_manager(self):
        return RuleManager()

    def _make_nucleus(self, x: int, y: int, z: float, identity: str = "") -> Nucleus:
        return Nucleus(
            index=1, x=x, y=y, z=z, size=20,
            identity=identity, status=1
        )

    def test_basic_ap_division(self, rule_manager):
        """Test a simple AP division where daughter1 is more anterior."""
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
            canonical_transform=CanonicalTransform(
                ap_vec=np.array([-1.0, 0.0, 0.0]),
                lr_vec=np.array([0.0, 0.0, 1.0]),
            ),
        )

        parent = self._make_nucleus(300, 250, 15.0, identity="AB")
        # daughter1 more anterior (smaller x in raw coords -> more negative x after transform)
        dau1 = self._make_nucleus(280, 250, 15.0)
        dau2 = self._make_nucleus(320, 250, 15.0)

        name1, name2 = dc.assign_names(parent, dau1, dau2)
        # Both should be named (one ABa, one ABp)
        assert "AB" in name1
        assert "AB" in name2
        assert name1 != name2
        assert set([name1, name2]) == {"ABa", "ABp"}

    def test_v1_mode_basic(self, rule_manager):
        """Test v1 mode with axis string and angle."""
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
            axis_string="ADL",
            angle=0.0,
        )

        parent = self._make_nucleus(300, 250, 15.0, identity="AB")
        dau1 = self._make_nucleus(280, 250, 15.0)
        dau2 = self._make_nucleus(320, 250, 15.0)

        name1, name2 = dc.assign_names(parent, dau1, dau2)
        assert set([name1, name2]) == {"ABa", "ABp"}

    def test_unnamed_parent_returns_empty(self, rule_manager):
        """Parent with no name should return empty strings."""
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
        )

        parent = self._make_nucleus(300, 250, 15.0, identity="")
        dau1 = self._make_nucleus(280, 250, 15.0)
        dau2 = self._make_nucleus(320, 250, 15.0)

        name1, name2 = dc.assign_names(parent, dau1, dau2)
        assert name1 == ""
        assert name2 == ""

    def test_is_v2_property(self, rule_manager):
        """Test is_v2 property."""
        dc_v1 = DivisionCaller(rule_manager=rule_manager, axis_string="ADL")
        assert not dc_v1.is_v2

        ct = CanonicalTransform(
            ap_vec=np.array([-1.0, 0.0, 0.0]),
            lr_vec=np.array([0.0, 0.0, 1.0]),
        )
        dc_v2 = DivisionCaller(rule_manager=rule_manager, canonical_transform=ct)
        assert dc_v2.is_v2

    def test_z_scaling_matters(self, rule_manager):
        """Division along z axis should be scaled by z_pix_res."""
        ct = CanonicalTransform(
            ap_vec=np.array([-1.0, 0.0, 0.0]),
            lr_vec=np.array([0.0, 0.0, 1.0]),
        )
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
            canonical_transform=ct,
        )

        parent = self._make_nucleus(300, 250, 15.0, identity="AB")
        # Division along z only (1 plane difference -> 11.1 pixels after scaling)
        dau1 = self._make_nucleus(300, 250, 14.0)
        dau2 = self._make_nucleus(300, 250, 16.0)

        name1, name2 = dc.assign_names(parent, dau1, dau2)
        # Should still produce valid names
        assert name1 != ""
        assert name2 != ""
