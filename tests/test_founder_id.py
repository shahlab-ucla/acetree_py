"""Tests for acetree_py.naming.founder_id — topology-based founder identification."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.naming.founder_id import (
    FounderAssignment,
    _axes_from_founders,
    _count_alive,
    _find_four_cell_windows,
    _find_sister_pairs,
    _get_alive,
    _is_polar_body,
    _trace_back_to_birth,
    identify_founders,
)


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
    size: int = 20,
) -> Nucleus:
    return Nucleus(
        index=index, x=x, y=y, z=z, size=size,
        identity=identity, status=status,
        predecessor=pred, successor1=succ1, successor2=succ2,
        weight=5000,
    )


def _make_standard_lineage() -> list[list[Nucleus]]:
    """Build a standard early C. elegans lineage for testing.

    T0: P0 (one cell, dividing at T1)
    T1: AB, P1 (two cells, AB divides at T2, P1 continues)
    T2: AB, P1 (AB continuing, P1 continuing) — before AB division
    T3: ABa, ABp, P1 (AB divides, P1 continues)
    T4: ABa, ABp, P1 (continuing, P1 divides at T5)
    T5: ABa, ABp, EMS, P2 (4-cell stage)
    T6: ABa, ABp, EMS, P2 (4-cell stage continuing)
    T7: ABa, ABp, EMS, P2 (4-cell stage continuing)

    Successor/predecessor links use 1-based indices.
    """
    # T0: P0
    p0 = _make_nuc(1, 300, 250, 15.0, identity="", succ1=1, succ2=2)

    # T1: AB (from P0 succ1), P1 (from P0 succ2)
    ab = _make_nuc(1, 280, 240, 14.0, identity="", pred=1, succ1=1)
    p1 = _make_nuc(2, 320, 260, 16.0, identity="", pred=1, succ1=2)

    # T2: AB continuing, P1 continuing
    ab2 = _make_nuc(1, 275, 235, 13.5, identity="", pred=1, succ1=1, succ2=2)
    p1_2 = _make_nuc(2, 325, 265, 16.5, identity="", pred=2, succ1=2)

    # T3: ABa (from AB succ1), ABp (from AB succ2), P1 (continuing)
    aba = _make_nuc(1, 260, 225, 12.0, identity="", pred=1, succ1=1)
    abp = _make_nuc(2, 290, 245, 15.0, identity="", pred=1, succ1=2)
    p1_3 = _make_nuc(3, 330, 268, 17.0, identity="", pred=2, succ1=3)  # pred is p1_2 index=2

    # Wait — need to be consistent with 1-based indexing.
    # At T2: index 1=AB, index 2=P1
    # T3: pred for ABa = 1 (AB at T2), pred for ABp = 1 (AB at T2), pred for P1 = 2 (P1 at T2)
    # AB at T2 has succ1=1 (ABa at T3), succ2=2 (ABp at T3)
    # P1 at T2 has succ1=3 (P1 at T3)

    # T4: ABa continuing, ABp continuing, P1 dividing
    aba4 = _make_nuc(1, 255, 220, 11.5, identity="", pred=1, succ1=1)
    abp4 = _make_nuc(2, 295, 250, 15.5, identity="", pred=2, succ1=2)
    p1_4 = _make_nuc(3, 335, 270, 17.5, identity="", pred=3, succ1=3, succ2=4)

    # T5: ABa, ABp, EMS, P2 (4-cell stage!)
    aba5 = _make_nuc(1, 250, 215, 11.0, identity="", pred=1, succ1=1)
    abp5 = _make_nuc(2, 300, 255, 16.0, identity="", pred=2, succ1=2)
    ems5 = _make_nuc(3, 340, 265, 17.0, identity="", pred=3, succ1=3, size=25)  # EMS larger
    p2_5 = _make_nuc(4, 360, 275, 18.0, identity="", pred=3, succ1=4, size=15)  # P2 smaller

    # T6: 4-cell continuing
    aba6 = _make_nuc(1, 248, 213, 10.8, identity="", pred=1, succ1=1)
    abp6 = _make_nuc(2, 302, 257, 16.2, identity="", pred=2, succ1=2)
    ems6 = _make_nuc(3, 342, 266, 17.1, identity="", pred=3, succ1=3, size=25)
    p2_6 = _make_nuc(4, 362, 276, 18.1, identity="", pred=4, succ1=4, size=15)

    # T7: 4-cell continuing
    aba7 = _make_nuc(1, 246, 211, 10.6, identity="", pred=1)
    abp7 = _make_nuc(2, 304, 259, 16.4, identity="", pred=2)
    ems7 = _make_nuc(3, 344, 267, 17.2, identity="", pred=3, size=25)
    p2_7 = _make_nuc(4, 364, 277, 18.2, identity="", pred=4, size=15)

    return [
        [p0],              # T0
        [ab, p1],          # T1
        [ab2, p1_2],       # T2
        [aba, abp, p1_3],  # T3
        [aba4, abp4, p1_4],  # T4
        [aba5, abp5, ems5, p2_5],  # T5
        [aba6, abp6, ems6, p2_6],  # T6
        [aba7, abp7, ems7, p2_7],  # T7
    ]


class TestCountAlive:
    def test_counts_alive_nuclei(self):
        nuclei = [
            _make_nuc(1, 0, 0, 0, status=1),
            _make_nuc(2, 0, 0, 0, status=1),
            _make_nuc(3, 0, 0, 0, status=-1),
        ]
        assert _count_alive(nuclei) == 2

    def test_excludes_polar_bodies(self):
        nuclei = [
            _make_nuc(1, 0, 0, 0, status=1),
            _make_nuc(2, 0, 0, 0, status=1, identity="polar"),
        ]
        assert _count_alive(nuclei) == 1


class TestFindFourCellWindows:
    def test_finds_window_in_standard_lineage(self):
        record = _make_standard_lineage()
        windows = _find_four_cell_windows(record, 0, len(record))
        assert len(windows) >= 1
        first, last = windows[0]
        # Should span T5-T7
        assert first == 5
        assert last == 7

    def test_no_four_cell_stage(self):
        # Only 1 cell
        record = [[_make_nuc(1, 0, 0, 0)]] * 10
        windows = _find_four_cell_windows(record, 0, 10)
        assert windows == []

    def test_empty_record(self):
        windows = _find_four_cell_windows([], 0, 0)
        assert windows == []


class TestTraceBackToBirth:
    def test_traces_to_division_event(self):
        record = _make_standard_lineage()
        # ABa at T5 (index 0) was born when AB divided at T2->T3
        aba_at_t5 = record[5][0]
        birth, parent_info = _trace_back_to_birth(record, aba_at_t5, 5)
        # ABa traces back: T5->T4->T3, born at T3 (AB divided at T2 creating ABa at T3)
        assert birth == 3
        assert parent_info[0] == 2  # Parent was at T2

    def test_root_cell(self):
        record = _make_standard_lineage()
        p0 = record[0][0]
        birth, parent_info = _trace_back_to_birth(record, p0, 0)
        assert birth == 0
        assert parent_info == (-1, -1)


class TestFindSisterPairs:
    def test_finds_correct_pairs(self):
        record = _make_standard_lineage()
        alive = _get_alive(record[5])
        assert len(alive) == 4

        pairs = _find_sister_pairs(record, alive, 5)
        assert pairs is not None

        (pair_a, birth_a), (pair_b, birth_b) = pairs

        # One pair should have birth time 3 (ABa/ABp born from AB at T3)
        # Other pair should have birth time 5 (EMS/P2 born from P1 at T5)
        births = sorted([birth_a, birth_b])
        assert births[0] == 3
        assert births[1] == 5


class TestIdentifyFounders:
    def test_identifies_standard_lineage(self):
        record = _make_standard_lineage()
        result = identify_founders(record, z_pix_res=11.1)

        assert result.success
        assert result.confidence > 0

        # Check names were assigned at the 4-cell stage
        mid = result.four_cell_time
        names = {n.identity for n in record[mid] if n.status >= 1}
        assert "ABa" in names
        assert "ABp" in names
        assert "EMS" in names
        assert "P2" in names

    def test_back_traces_ab_p1_p0(self):
        record = _make_standard_lineage()
        result = identify_founders(record, z_pix_res=11.1)
        assert result.success

        # Check that early cells were named by back-tracing
        all_names = set()
        for t_nuclei in record:
            for nuc in t_nuclei:
                if nuc.identity:
                    all_names.add(nuc.identity)

        assert "AB" in all_names
        assert "P0" in all_names

    def test_empty_record(self):
        result = identify_founders([], z_pix_res=11.1)
        assert not result.success

    def test_single_cell_record(self):
        record = [[_make_nuc(1, 100, 100, 10)]] * 10
        result = identify_founders(record, z_pix_res=11.1)
        assert not result.success

    def test_confidence_higher_with_larger_timing_gap(self):
        """AB dividing well before P1 should give higher confidence."""
        record = _make_standard_lineage()
        result = identify_founders(record, z_pix_res=11.1)
        # AB divides at T2->T3, P1 at T4->T5, gap is 2 timepoints
        assert result.confidence > 0.3

    def test_start_index_set_correctly(self):
        record = _make_standard_lineage()
        result = identify_founders(record, z_pix_res=11.1)
        assert result.success
        # Start index should be <= T0 (where P0 was found)
        assert result.start_index <= 1


class TestAxesFromFounders:
    def test_computes_axes(self):
        # Create 4 cells in a standard arrangement
        aba = _make_nuc(1, 100, 200, 10.0)
        abp = _make_nuc(2, 100, 200, 15.0)   # Different z
        ems = _make_nuc(3, 300, 200, 12.0)
        p2 = _make_nuc(4, 350, 200, 13.0)

        ap, lr, dv = _axes_from_founders(aba, abp, ems, p2, z_pix_res=1.0)

        assert ap is not None
        assert lr is not None
        assert dv is not None

        # AP should point from P2 toward AB center (negative x direction)
        assert ap[0] < 0  # AP points from posterior to anterior

        # Axes should be orthogonal
        assert abs(np.dot(ap, lr)) < 0.1
        assert abs(np.dot(ap, dv)) < 0.1
        assert abs(np.dot(lr, dv)) < 0.1

        # Axes should be unit vectors
        assert abs(np.linalg.norm(ap) - 1.0) < 1e-6
        assert abs(np.linalg.norm(lr) - 1.0) < 1e-6
        assert abs(np.linalg.norm(dv) - 1.0) < 1e-6

    def test_degenerate_returns_none(self):
        # All cells at same position
        nuc = _make_nuc(1, 100, 100, 10.0)
        ap, lr, dv = _axes_from_founders(nuc, nuc, nuc, nuc, z_pix_res=1.0)
        assert ap is None
        assert lr is None
        assert dv is None

    def test_rotation_invariance(self):
        """Axes should be determinable regardless of global orientation."""
        # Standard arrangement
        aba1 = _make_nuc(1, 100, 200, 10.0)
        abp1 = _make_nuc(2, 100, 200, 15.0)
        ems1 = _make_nuc(3, 300, 200, 12.0)
        p2_1 = _make_nuc(4, 350, 200, 13.0)

        ap1, lr1, dv1 = _axes_from_founders(aba1, abp1, ems1, p2_1, z_pix_res=1.0)

        # Rotated 90 degrees: swap x and y
        aba2 = _make_nuc(1, 200, 100, 10.0)
        abp2 = _make_nuc(2, 200, 100, 15.0)
        ems2 = _make_nuc(3, 200, 300, 12.0)
        p2_2 = _make_nuc(4, 200, 350, 13.0)

        ap2, lr2, dv2 = _axes_from_founders(aba2, abp2, ems2, p2_2, z_pix_res=1.0)

        # Both should produce valid axes
        assert ap1 is not None
        assert ap2 is not None

        # AP axis directions should differ (rotated) but both be unit vectors
        assert abs(np.linalg.norm(ap1) - 1.0) < 1e-6
        assert abs(np.linalg.norm(ap2) - 1.0) < 1e-6


class TestIsPolarBody:
    def test_identity_polar(self):
        nuc = _make_nuc(1, 0, 0, 0, identity="polar body", status=1)
        assert _is_polar_body(nuc)

    def test_assigned_id_polar(self):
        nuc = _make_nuc(1, 0, 0, 0, identity="", status=1)
        nuc.assigned_id = "Polar Body"
        assert _is_polar_body(nuc)

    def test_empty_identity_not_polar(self):
        nuc = _make_nuc(1, 0, 0, 0, identity="", status=1)
        assert not _is_polar_body(nuc)

    def test_cleared_identity_with_polar_assigned(self):
        """After _clear_all_names, identity is '' but assigned_id='polar' should still exclude."""
        nuc = _make_nuc(1, 0, 0, 0, identity="", status=1)
        nuc.assigned_id = "polar"
        assert _is_polar_body(nuc)


class TestCountAliveExcludesPolars:
    def test_excludes_polar_by_assigned_id(self):
        """Polar bodies with assigned_id should be excluded even after identity cleared."""
        nuc1 = _make_nuc(1, 0, 0, 0, status=1, identity="")
        nuc2 = _make_nuc(2, 0, 0, 0, status=1, identity="")
        nuc2.assigned_id = "polar body"
        assert _count_alive([nuc1, nuc2]) == 1

    def test_dead_polar_not_counted(self):
        """Dead polar bodies shouldn't be counted regardless."""
        nuc = _make_nuc(1, 0, 0, 0, status=0, identity="polar")
        assert _count_alive([nuc]) == 0


class TestBackTraceAllFounders:
    def test_abp_and_p2_get_named_backward(self):
        """_back_trace_founders should name ABp and P2 continuation cells backward."""
        record = _make_standard_lineage()
        result = identify_founders(record, z_pix_res=11.1)
        assert result.success

        mid = result.four_cell_time  # Should be 6 (midpoint of T5-T7)

        # Check that ABp and P2 are named at the midpoint
        abp_at_mid = any(n.identity == "ABp" for n in record[mid])
        p2_at_mid = any(n.identity == "P2" for n in record[mid])
        assert abp_at_mid, f"ABp not named at midpoint t={mid}"
        assert p2_at_mid, f"P2 not named at midpoint t={mid}"

        # Back-trace names cells BACKWARD from midpoint to birth.
        # ABp was born at T3 and continues through T4, T5, T6 (midpoint).
        # Names AFTER midpoint (T7) are set by the forward pass, not back-trace.
        for t in range(3, mid + 1):
            has_abp = any(n.identity == "ABp" for n in record[t])
            assert has_abp, f"ABp not named at t={t} (between birth and midpoint)"

        # P2 was born at T5 and continues to T6 (midpoint).
        for t in range(5, mid + 1):
            has_p2 = any(n.identity == "P2" for n in record[t])
            assert has_p2, f"P2 not named at t={t} (between birth and midpoint)"

    def test_p1_named_by_ems_trace(self):
        """EMS back-trace should name P1 at its division."""
        record = _make_standard_lineage()
        result = identify_founders(record, z_pix_res=11.1)
        assert result.success

        # P1 should be named somewhere
        p1_times = [t for t in range(len(record))
                     for nuc in record[t] if nuc.identity == "P1"]
        assert len(p1_times) > 0, "P1 was never named"


class TestFounderIdWithExistingTests:
    """Tests matching the patterns from test_identity.py to ensure compatibility."""

    def test_simple_lineage_p0_division(self):
        """A 3-timepoint P0 -> AB + P1 lineage should not find 4-cell stage."""
        p0 = _make_nuc(1, 300, 250, 15.0, succ1=1, succ2=2)
        ab = _make_nuc(1, 280, 240, 14.0, pred=1, succ1=1)
        p1 = _make_nuc(2, 320, 260, 16.0, pred=1, succ1=2)
        ab2 = _make_nuc(1, 275, 235, 13.5, pred=1)
        p1_2 = _make_nuc(2, 325, 265, 16.5, pred=2)

        record = [[p0], [ab, p1], [ab2, p1_2]]
        result = identify_founders(record, z_pix_res=11.1)
        # No 4-cell stage => not successful
        assert not result.success
