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

    @pytest.mark.parametrize(
        ("parent", "expected"),
        [
            ("P0", ("AB", "P1")),
            ("P1", ("EMS", "P2")),
            ("EMS", ("E", "MS")),
            ("P2", ("C", "P3")),
            ("P3", ("D", "P4")),
            ("P4", ("Z2", "Z3")),
        ],
    )
    def test_special_founder_daughter_families_are_exact(
        self,
        parent: str,
        expected: tuple[str, str],
    ):
        rm = RuleManager()
        rule = rm.get_rule(parent)

        assert (rule.daughter1, rule.daughter2) == expected

    @pytest.mark.parametrize(
        ("parent", "expected"),
        [
            ("P0", ("AB", "P1")),
            ("P1", ("EMS", "P2")),
            ("EMS", ("E", "MS")),
            ("P2", ("C", "P3")),
            ("P3", ("D", "P4")),
            ("P4", ("Z2", "Z3")),
        ],
    )
    def test_special_founder_families_survive_missing_rule_resource(
        self,
        parent: str,
        expected: tuple[str, str],
    ):
        rm = RuleManager()
        rm._new_rules.clear()

        rule = rm.get_rule(parent)

        assert (rule.daughter1, rule.daughter2) == expected

    def test_loads_precomputed_rules(self):
        rm = RuleManager()
        assert rm.num_precomputed > 600  # NewRules.txt has ~621 entries

    def test_loads_names_hash(self):
        rm = RuleManager()
        assert rm.num_hash_entries > 50  # namesHash.txt has ~61 entries

    def test_precomputed_special_rule_remains_authoritative(self):
        rm = RuleManager()

        assert rm.get_rule("P2") is rm._new_rules["P2"]

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

    def test_lineage_mode_uses_complete_seed_after_landmark_dropout(
        self,
        rule_manager,
    ):
        """A later missing lineage must not discard the four-cell frame."""
        record = [[
            self._make_nucleus(0, 0, 0.0),
            self._make_nucleus(10, 0, 0.0),
            self._make_nucleus(0, 10, 0.0),
        ]]
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=1.0,
            lineage_map=[["ABa", "ABp", "EMS"]],
            nuclei_record=record,
            seed_ap=np.array([-1.0, 0.0, 0.0]),
            seed_lr=np.array([0.0, 0.0, 1.0]),
            seed_dv=np.array([0.0, 1.0, 0.0]),
        )
        parent = self._make_nucleus(10, 0, 0.0, identity="AB")
        daughter1 = self._make_nucleus(0, 0, 0.0)
        daughter2 = self._make_nucleus(20, 0, 0.0)

        assert dc._get_local_axes(0) is None
        assert dc.has_complete_body_frame(0)
        assert dc.assign_names(
            parent,
            daughter1,
            daughter2,
            timepoint=0,
        ) == ("ABa", "ABp")

    def test_lineage_mode_uses_complete_seed_after_local_axis_degeneracy(
        self,
        rule_manager,
    ):
        """Collinear landmarks fall back to the retained seed frame."""
        record = [[
            self._make_nucleus(30, 0, 0.0),
            self._make_nucleus(20, 0, 0.0),
            self._make_nucleus(10, 0, 0.0),
            self._make_nucleus(0, 0, 0.0),
        ]]
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=1.0,
            lineage_map=[["ABa", "ABp", "EMS", "P2"]],
            nuclei_record=record,
            seed_ap=np.array([-1.0, 0.0, 0.0]),
            seed_lr=np.array([0.0, 0.0, 1.0]),
            seed_dv=np.array([0.0, 1.0, 0.0]),
        )
        parent = self._make_nucleus(10, 0, 0.0, identity="AB")
        daughter1 = self._make_nucleus(0, 0, 0.0)
        daughter2 = self._make_nucleus(20, 0, 0.0)

        assert dc._get_local_axes(0) is None
        assert dc.has_complete_body_frame(0)
        assert dc.assign_names(
            parent,
            daughter1,
            daughter2,
            timepoint=0,
        ) == ("ABa", "ABp")

    def test_lineage_mode_does_not_mix_incomplete_static_frames(
        self,
        rule_manager,
    ):
        """Partial seed and founder inputs cannot form a synthetic frame."""
        record = [[self._make_nucleus(0, 0, 0.0)]]
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=1.0,
            founder_dv=np.array([0.0, 1.0, 0.0]),
            lineage_map=[[""]],
            nuclei_record=record,
            seed_ap=np.array([-1.0, 0.0, 0.0]),
            seed_lr=np.array([0.0, 0.0, 1.0]),
        )
        parent = self._make_nucleus(10, 0, 0.0, identity="AB")

        assert not dc.has_complete_body_frame(0)
        assert dc.assign_names(
            parent,
            self._make_nucleus(0, 0, 0.0),
            self._make_nucleus(20, 0, 0.0),
            timepoint=0,
        ) == ("ABa", "ABp")
        assert dc.classifications[-1].confidence == 0.0

    def test_lineage_mode_prefers_available_dynamic_axes_over_seed(
        self,
        rule_manager,
    ):
        """The seed is a fallback, not a replacement for per-frame geometry."""
        record = [[
            self._make_nucleus(0, 10, 0.0),
            self._make_nucleus(10, 0, 0.0),
            self._make_nucleus(0, 0, 0.0),
            self._make_nucleus(0, 0, 0.0),
        ]]
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=1.0,
            lineage_map=[["ABa", "ABp", "EMS", "P2"]],
            nuclei_record=record,
            seed_ap=np.array([-1.0, 0.0, 0.0]),
            seed_lr=np.array([0.0, 0.0, 1.0]),
            seed_dv=np.array([0.0, 1.0, 0.0]),
        )

        corrected = dc._measurement_correction(
            np.array([10.0, 0.0, 0.0]),
            timepoint=0,
        )

        np.testing.assert_allclose(corrected, [0.0, 10.0, 0.0])

    def test_lineage_mode_uses_seed_when_fresh_axes_are_low_quality(
        self,
        rule_manager,
        monkeypatch,
    ):
        """Weak near-collinear geometry cannot replace a trusted quartet frame."""
        import acetree_py.naming.division_caller as division_caller_module

        seed = (
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        )

        def low_quality_axes(*_args, **_kwargs):
            return (
                np.array([0.0, -1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                0.01,
            )

        monkeypatch.setattr(
            division_caller_module,
            "compute_local_axes",
            low_quality_axes,
        )
        caller = DivisionCaller(
            rule_manager=rule_manager,
            lineage_map=[["ABa"]],
            nuclei_record=[[self._make_nucleus(0, 0, 0.0)]],
            seed_ap=seed[0],
            seed_lr=seed[1],
            seed_dv=seed[2],
        )

        axes = caller._get_local_axes(0)

        assert axes is not None
        for actual, expected in zip(axes, seed):
            np.testing.assert_allclose(actual, expected)
