"""Tests for lineage-derived anatomical body axes."""

from __future__ import annotations

import numpy as np

from acetree_py.core.nucleus import Nucleus
from acetree_py.naming.lineage_axes import build_lineage_map, compute_local_axes


def _nucleus(index: int, x: int, y: int, z: float) -> Nucleus:
    return Nucleus(index=index, x=x, y=y, z=z, size=20, status=1)


def test_four_lineage_landmarks_define_anatomical_frame():
    # P2 -> ABa establishes +X as anterior.  EMS -> ABp has a noisy AP
    # component but its perpendicular part establishes +Z as dorsal.
    record = [[
        _nucleus(1, 10, 0, 0),  # ABa
        _nucleus(2, 3, 0, 1),   # ABp
        _nucleus(3, 0, 0, 0),   # EMS
        _nucleus(4, 0, 0, 0),   # P2
    ]]
    lineage_map = [["ABa", "ABp", "EMS", "P2"]]

    ap, lr, dv, quality = compute_local_axes(record, lineage_map, 0, z_pix_res=2.0)

    np.testing.assert_allclose(ap, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(dv, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(lr, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(np.cross(ap, lr), dv)
    assert 0.0 < quality < 1.0


def test_axes_require_all_four_landmark_lineages():
    record = [[_nucleus(1, 0, 0, 0), _nucleus(2, 1, 0, 0)]]
    axes = compute_local_axes(record, [["ABa", "P2"]], 0, z_pix_res=1.0)
    assert axes == (None, None, None, 0.0)


def test_axis_geometry_scales_z_into_physical_space():
    record = [[
        _nucleus(1, 10, 0, 0),
        _nucleus(2, 0, 1, 1),
        _nucleus(3, 0, 0, 0),
        _nucleus(4, 0, 0, 0),
    ]]
    labels = [["ABa", "ABp", "EMS", "P2"]]

    _, _, dv, _ = compute_local_axes(record, labels, 0, z_pix_res=3.0)

    expected = np.array([0.0, 1.0, 3.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(dv, expected)


def test_lineage_map_requires_unique_reciprocal_successor_links():
    founders = [
        _nucleus(1, 0, 0, 0),
        _nucleus(2, 1, 0, 0),
        _nucleus(3, 2, 0, 0),
        _nucleus(4, 3, 0, 0),
    ]
    founders[0].successor1 = 1
    child = _nucleus(1, 0, 0, 0)
    child.predecessor = 1

    valid = build_lineage_map(
        [founders, [child]],
        four_cell_time=0,
        aba_idx=0,
        abp_idx=1,
        ems_idx=2,
        p2_idx=3,
    )
    assert valid[1][0] == "ABa"

    founders[1].successor1 = 1
    malformed = build_lineage_map(
        [founders, [child]],
        four_cell_time=0,
        aba_idx=0,
        abp_idx=1,
        ems_idx=2,
        p2_idx=3,
    )
    assert malformed[1][0] == ""


def test_lineage_map_rejects_duplicate_successor_slots():
    founders = [
        _nucleus(1, 0, 0, 0),
        _nucleus(2, 1, 0, 0),
        _nucleus(3, 2, 0, 0),
        _nucleus(4, 3, 0, 0),
    ]
    founders[0].successor1 = 1
    founders[0].successor2 = 1
    child = _nucleus(1, 0, 0, 0)
    child.predecessor = 1

    lineage_map = build_lineage_map(
        [founders, [child]],
        four_cell_time=0,
        aba_idx=0,
        abp_idx=1,
        ems_idx=2,
        p2_idx=3,
    )

    assert lineage_map[1][0] == ""


def test_lineage_map_rejects_undeclared_live_reverse_claimers():
    founders = [
        _nucleus(1, 0, 0, 0),
        _nucleus(2, 1, 0, 0),
        _nucleus(3, 2, 0, 0),
        _nucleus(4, 3, 0, 0),
    ]
    founders[0].successor1 = 1
    children = [_nucleus(index, index, 0, 0) for index in (1, 2, 3)]
    for child in children:
        child.predecessor = 1

    lineage_map = build_lineage_map(
        [founders, children],
        four_cell_time=0,
        aba_idx=0,
        abp_idx=1,
        ems_idx=2,
        p2_idx=3,
    )

    assert lineage_map[1] == ["", "", ""]
