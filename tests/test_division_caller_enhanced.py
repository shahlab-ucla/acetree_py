"""Tests for enhanced DivisionCaller — multi-frame averaging and confidence scoring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.naming.canonical_transform import CanonicalTransform
from acetree_py.naming.division_caller import (
    DEFAULT_AVG_FRAMES,
    DivisionCaller,
    DivisionClassification,
    _angle_to_confidence,
    _follow_successor,
)
from acetree_py.naming.rules import RuleManager


def _make_nuc(
    index: int,
    x: int,
    y: int,
    z: float,
    identity: str = "",
    status: int = 1,
    pred: int = NILLI,
    succ1: int = NILLI,
    succ2: int = NILLI,
) -> Nucleus:
    return Nucleus(
        index=index, x=x, y=y, z=z, size=20,
        identity=identity, status=status,
        predecessor=pred, successor1=succ1, successor2=succ2,
        weight=5000,
    )


@pytest.fixture
def rule_manager():
    return RuleManager()


@pytest.fixture
def canonical_transform():
    return CanonicalTransform(
        ap_vec=np.array([-1.0, 0.0, 0.0]),
        lr_vec=np.array([0.0, 0.0, 1.0]),
    )


class TestAngleToConfidence:
    def test_small_angle_high_confidence(self):
        assert _angle_to_confidence(5.0) == 1.0
        assert _angle_to_confidence(15.0) == 1.0

    def test_moderate_angle_medium_confidence(self):
        c = _angle_to_confidence(30.0)
        assert 0.5 < c < 1.0

    def test_large_angle_low_confidence(self):
        c = _angle_to_confidence(50.0)
        assert 0.1 < c < 0.5

    def test_right_angle_very_low_confidence(self):
        c = _angle_to_confidence(89.0)
        assert c < 0.2

    def test_monotonically_decreasing(self):
        angles = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        confidences = [_angle_to_confidence(a) for a in angles]
        for i in range(1, len(confidences)):
            assert confidences[i] <= confidences[i - 1]


class TestDivisionClassification:
    def test_produces_classification(self, rule_manager, canonical_transform):
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
            canonical_transform=canonical_transform,
        )

        parent = _make_nuc(1, 300, 250, 15.0, identity="AB")
        dau1 = _make_nuc(1, 280, 250, 15.0)
        dau2 = _make_nuc(2, 320, 250, 15.0)

        name1, name2 = dc.assign_names(parent, dau1, dau2)

        assert len(dc.classifications) == 1
        c = dc.classifications[0]
        assert c.parent_name == "AB"
        assert c.confidence > 0
        assert c.daughter1_name in ("ABa", "ABp")
        assert c.daughter2_name in ("ABa", "ABp")
        assert c.daughter1_name != c.daughter2_name

    def test_high_confidence_for_clear_division(self, rule_manager, canonical_transform):
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
            canonical_transform=canonical_transform,
        )

        parent = _make_nuc(1, 300, 250, 15.0, identity="AB")
        # Large separation along x (AP axis) — very clear division
        dau1 = _make_nuc(1, 200, 250, 15.0)
        dau2 = _make_nuc(2, 400, 250, 15.0)

        dc.assign_names(parent, dau1, dau2)
        assert dc.classifications[0].confidence >= 0.8


class TestFollowSuccessor:
    def test_follows_single_successor(self):
        nuc = _make_nuc(1, 100, 100, 10.0, succ1=2)
        next_nuclei = [
            _make_nuc(1, 90, 90, 9.0),
            _make_nuc(2, 105, 105, 10.5),
        ]
        result = _follow_successor(nuc, next_nuclei)
        assert result is not None
        assert result.x == 105

    def test_returns_none_for_dividing(self):
        nuc = _make_nuc(1, 100, 100, 10.0, succ1=1, succ2=2)
        next_nuclei = [_make_nuc(1, 90, 90, 9.0), _make_nuc(2, 110, 110, 11.0)]
        result = _follow_successor(nuc, next_nuclei)
        assert result is None

    def test_returns_none_for_dead(self):
        nuc = _make_nuc(1, 100, 100, 10.0)  # No successor
        result = _follow_successor(nuc, [])
        assert result is None


class TestMultiFrameAveraging:
    def test_multi_frame_basic(self, rule_manager, canonical_transform):
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
            canonical_transform=canonical_transform,
        )

        # Build a 4-frame record where AB divides at T1
        # T0: AB
        ab = _make_nuc(1, 300, 250, 15.0, identity="AB", succ1=1, succ2=2)
        # T1: ABa, ABp (just born)
        aba1 = _make_nuc(1, 290, 250, 15.0, pred=1, succ1=1)
        abp1 = _make_nuc(2, 310, 250, 15.0, pred=1, succ1=2)
        # T2: daughters moving apart
        aba2 = _make_nuc(1, 280, 250, 15.0, pred=1, succ1=1)
        abp2 = _make_nuc(2, 320, 250, 15.0, pred=2, succ1=2)
        # T3: daughters further apart
        aba3 = _make_nuc(1, 270, 250, 15.0, pred=1)
        abp3 = _make_nuc(2, 330, 250, 15.0, pred=2)

        record = [[ab], [aba1, abp1], [aba2, abp2], [aba3, abp3]]

        name1, name2 = dc.assign_names_multi_frame(
            ab, aba1, abp1, record, division_time=1, n_frames=3,
        )

        assert {name1, name2} == {"ABa", "ABp"}
        assert len(dc.classifications) == 1
        assert dc.classifications[0].confidence > 0

    def test_falls_back_to_single_frame(self, rule_manager, canonical_transform):
        """If multi-frame tracking fails, should fall back to single-frame."""
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
            canonical_transform=canonical_transform,
        )

        parent = _make_nuc(1, 300, 250, 15.0, identity="AB", succ1=1, succ2=2)
        dau1 = _make_nuc(1, 280, 250, 15.0, pred=1)
        dau2 = _make_nuc(2, 320, 250, 15.0, pred=1)

        # Only 1 timepoint in record after division — can't do multi-frame
        record = [[parent], [dau1, dau2]]

        name1, name2 = dc.assign_names_multi_frame(
            parent, dau1, dau2, record, division_time=1, n_frames=3,
        )
        assert {name1, name2} == {"ABa", "ABp"}


class TestFounderMode:
    def test_founder_axes_mode(self, rule_manager):
        """Test that founder-derived axes can be used for division calling."""
        # AP points in -x direction, LR in +z, DV in +y
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=1.0,
            founder_ap=np.array([-1.0, 0.0, 0.0]),
            founder_lr=np.array([0.0, 0.0, 1.0]),
            founder_dv=np.array([0.0, 1.0, 0.0]),
        )

        assert dc.is_founder_mode

        parent = _make_nuc(1, 300, 250, 15.0, identity="AB")
        dau1 = _make_nuc(1, 280, 250, 15.0)  # More anterior
        dau2 = _make_nuc(2, 320, 250, 15.0)  # More posterior

        name1, name2 = dc.assign_names(parent, dau1, dau2)
        assert {name1, name2} == {"ABa", "ABp"}

    def test_founder_mode_not_v2(self, rule_manager):
        dc = DivisionCaller(
            rule_manager=rule_manager,
            founder_ap=np.array([-1.0, 0.0, 0.0]),
            founder_lr=np.array([0.0, 0.0, 1.0]),
            founder_dv=np.array([0.0, 1.0, 0.0]),
        )
        assert dc.is_founder_mode
        assert not dc.is_v2

    def test_v2_overrides_founder(self, rule_manager, canonical_transform):
        dc = DivisionCaller(
            rule_manager=rule_manager,
            canonical_transform=canonical_transform,
            founder_ap=np.array([-1.0, 0.0, 0.0]),
            founder_lr=np.array([0.0, 0.0, 1.0]),
            founder_dv=np.array([0.0, 1.0, 0.0]),
        )
        # v2 should take precedence
        assert dc.is_v2
        assert not dc.is_founder_mode


class TestBackwardCompatibility:
    """Ensure existing test_division_caller.py patterns still work."""

    def test_basic_ap_division(self, rule_manager, canonical_transform):
        dc = DivisionCaller(
            rule_manager=rule_manager,
            z_pix_res=11.1,
            canonical_transform=canonical_transform,
        )

        parent = _make_nuc(1, 300, 250, 15.0, identity="AB")
        dau1 = _make_nuc(1, 280, 250, 15.0)
        dau2 = _make_nuc(2, 320, 250, 15.0)

        name1, name2 = dc.assign_names(parent, dau1, dau2)
        assert {name1, name2} == {"ABa", "ABp"}

    def test_unnamed_parent_returns_empty(self, rule_manager):
        dc = DivisionCaller(rule_manager=rule_manager, z_pix_res=11.1)
        parent = _make_nuc(1, 300, 250, 15.0, identity="")
        dau1 = _make_nuc(1, 280, 250, 15.0)
        dau2 = _make_nuc(2, 320, 250, 15.0)

        name1, name2 = dc.assign_names(parent, dau1, dau2)
        assert name1 == ""
        assert name2 == ""
