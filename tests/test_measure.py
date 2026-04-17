"""Tests for acetree_py.analysis.measure* — pixel measurement, CSV writer, orchestrator.

Covers:
- Algorithm unit tests (synthetic bright-sphere stack, background stack, dead nucleus)
- CSV round-trip
- Integration: run_measure end-to-end with a 2-channel NumpyProvider
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from acetree_py.analysis.measure import (
    DEFAULT_ANNULUS_SCALE,
    measure_nucleus,
    measure_timepoint,
    measure_timepoint_with_blot,
    project_radius,
)
from acetree_py.analysis.measure_csv import write_measure_csv
from acetree_py.analysis.measure_runner import SCALE, run_measure
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import Nucleus
from acetree_py.io.image_provider import NumpyProvider


# ── Helpers ────────────────────────────────────────────────────────


def _make_sphere_stack(
    shape: tuple[int, int, int],
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    z_pix_res: float,
    value_in: int,
    value_out: int = 0,
) -> np.ndarray:
    """Return a uint16 stack filled with ``value_out`` and a sphere at
    (cz, cy, cx) of radius ``radius`` (XY pixels) set to ``value_in``.

    Uses the same isotropic-distance metric AceTree uses (z offsets are
    multiplied by ``z_pix_res`` before comparison).
    """
    nz, ny, nx = shape
    zs = np.arange(nz).reshape(-1, 1, 1)
    ys = np.arange(ny).reshape(1, -1, 1)
    xs = np.arange(nx).reshape(1, 1, -1)
    d2 = ((xs - cx) ** 2) + ((ys - cy) ** 2) + (((zs - cz) * z_pix_res) ** 2)
    stack = np.full(shape, value_out, dtype=np.uint16)
    stack[d2 <= radius * radius] = value_in
    return stack


def _make_nucleus(
    *,
    index: int = 1,
    x: int = 32,
    y: int = 32,
    z: float = 5.0,
    size: int = 16,
    status: int = 1,
) -> Nucleus:
    """Build a minimal live Nucleus for measurement tests."""
    return Nucleus(
        index=index,
        x=x,
        y=y,
        z=z,
        size=size,
        status=status,
    )


# ── project_radius ─────────────────────────────────────────────────


class TestProjectRadius:
    def test_center_plane_full_radius(self):
        # At the nucleus's own Z plane, projected radius is full.
        r = project_radius(nuc_z=5.0, z_plane=5, radius=8.0, z_pix_res=1.0)
        assert r == pytest.approx(8.0)

    def test_outside_sphere_zero(self):
        # Plane far outside sphere → zero projection.
        r = project_radius(nuc_z=5.0, z_plane=20, radius=8.0, z_pix_res=1.0)
        assert r == 0.0

    def test_projection_shrinks_off_center(self):
        # One pixel off-center (in isotropic units): r = sqrt(1 - (1/8)²) * 8.
        r = project_radius(nuc_z=5.0, z_plane=6, radius=8.0, z_pix_res=1.0)
        assert r == pytest.approx(np.sqrt(1 - (1 / 8) ** 2) * 8.0)

    def test_zero_radius(self):
        assert project_radius(0, 0, 0, 1.0) == 0.0


# ── measure_nucleus ────────────────────────────────────────────────


class TestMeasureNucleus:
    def test_bright_sphere(self):
        """Sphere of value 1000 → sum_in ≈ 1000 * voxel_count, annulus ≈ 0."""
        shape = (11, 64, 64)
        cx, cy, cz = 32, 32, 5
        radius = 8.0
        z_pix_res = 1.0
        stack = _make_sphere_stack(
            shape, cx, cy, cz, radius, z_pix_res, value_in=1000, value_out=0,
        )
        nuc = _make_nucleus(
            x=cx, y=cy, z=float(cz), size=int(radius * 2),
        )

        sum_in, count_in, sum_ann, count_ann = measure_nucleus(
            stack, nuc, z_pix_res,
        )

        # Every pixel in the disk should be ~1000; annulus should be background.
        assert count_in > 0
        mean_in = sum_in / count_in
        assert mean_in == pytest.approx(1000.0, rel=0.01)
        # Annulus can overlap the sphere slightly because our annulus_scale
        # (1.5x) extends past the disk at the central plane; still the
        # *mean* annulus intensity should be much lower than the interior.
        if count_ann > 0:
            mean_ann = sum_ann / count_ann
            assert mean_ann < mean_in

    def test_uniform_background(self):
        """Uniform stack value=200 → rwcorr1 mean ≈ 200."""
        shape = (11, 64, 64)
        stack = np.full(shape, 200, dtype=np.uint16)
        nuc = _make_nucleus(x=32, y=32, z=5.0, size=16)

        sum_in, count_in, sum_ann, count_ann = measure_nucleus(
            stack, nuc, 1.0,
        )
        assert count_in > 0
        assert count_ann > 0
        assert sum_in / count_in == pytest.approx(200.0)
        assert sum_ann / count_ann == pytest.approx(200.0)

    def test_dead_nucleus_skipped(self):
        """status<1 ⇒ all zeros returned."""
        stack = np.ones((5, 32, 32), dtype=np.uint16) * 500
        nuc = _make_nucleus(status=-1)
        assert measure_nucleus(stack, nuc, 1.0) == (0, 0, 0, 0)

    def test_zero_size_returns_zero(self):
        stack = np.ones((5, 32, 32), dtype=np.uint16) * 500
        nuc = _make_nucleus(size=0)
        assert measure_nucleus(stack, nuc, 1.0) == (0, 0, 0, 0)

    def test_requires_3d_stack(self):
        stack = np.zeros((32, 32), dtype=np.uint16)
        nuc = _make_nucleus()
        with pytest.raises(ValueError):
            measure_nucleus(stack, nuc, 1.0)

    def test_offscreen_nucleus(self):
        """Nucleus centered outside image returns mostly empty measurement."""
        stack = np.ones((5, 32, 32), dtype=np.uint16) * 100
        nuc = _make_nucleus(x=500, y=500, z=2.0, size=16)
        sum_in, count_in, _, _ = measure_nucleus(stack, nuc, 1.0)
        # The disk bounding box is entirely outside the image bounds, so
        # count should be zero.
        assert count_in == 0
        assert sum_in == 0


# ── measure_timepoint ──────────────────────────────────────────────


class TestMeasureTimepoint:
    def test_aligns_to_input_length(self):
        stack = np.ones((5, 32, 32), dtype=np.uint16) * 100
        nuclei = [
            _make_nucleus(index=1, x=16, y=16, z=2.0, size=8),
            _make_nucleus(index=2, x=20, y=20, z=2.0, size=8, status=-1),
            _make_nucleus(index=3, x=10, y=10, z=2.0, size=8),
        ]
        out = measure_timepoint(stack, nuclei, 1.0)
        assert len(out) == 3
        # Dead nucleus returns zeros
        assert out[1] == (0, 0, 0, 0)
        # Live nuclei non-zero
        assert out[0][1] > 0
        assert out[2][1] > 0

    def test_empty_list(self):
        stack = np.ones((5, 32, 32), dtype=np.uint16)
        assert measure_timepoint(stack, [], 1.0) == []


# ── measure_timepoint_with_blot ────────────────────────────────────


class TestMeasureTimepointWithBlot:
    """Blot correction = annulus with every nucleus's projected disk
    masked out.  With no neighbors, blot == global annulus; with a
    bright neighbor overlapping the annulus, blot drops (neighbor
    bright pixels are excluded, leaving only true background)."""

    def test_shape_and_alignment(self):
        stack = np.ones((5, 32, 32), dtype=np.uint16) * 100
        nuclei = [
            _make_nucleus(index=1, x=16, y=16, z=2.0, size=8),
            _make_nucleus(index=2, x=20, y=20, z=2.0, size=8, status=-1),
        ]
        out = measure_timepoint_with_blot(stack, nuclei, 1.0)
        assert len(out) == 2
        # Each tuple has 6 fields
        assert all(len(t) == 6 for t in out)
        # Dead nucleus: all zeros
        assert out[1] == (0, 0, 0, 0, 0, 0)

    def test_blot_matches_global_when_isolated(self):
        """With only one nucleus (no neighbors), the blot annulus equals
        the global annulus — the union-of-disks mask covers only self's
        own disk, which is already excluded from the annulus."""
        # Single isolated nucleus, bright interior, uniform background 50
        stack = _make_sphere_stack(
            shape=(11, 64, 64),
            cx=32, cy=32, cz=5.0,
            radius=8.0, z_pix_res=1.0,
            value_in=1000, value_out=50,
        )
        nuc = _make_nucleus(index=1, x=32, y=32, z=5.0, size=16)
        out = measure_timepoint_with_blot(stack, [nuc], 1.0)
        sum_in, count_in, sum_ann, count_ann, sum_blot, count_blot = out[0]
        # With one nucleus, blot annulus == global annulus
        assert sum_ann == sum_blot
        assert count_ann == count_blot
        # Background of 50 dominates the annulus
        assert count_ann > 0
        assert sum_ann / count_ann == pytest.approx(50, rel=1e-6)

    def test_blot_excludes_bright_neighbor_from_annulus(self):
        """A bright neighbor overlapping the target's annulus must be
        masked out: blot mean should be closer to true background than
        the raw annulus mean (which is contaminated by the neighbor)."""
        # Canvas: background 50.
        nz, ny, nx = 11, 80, 80
        stack = np.full((nz, ny, nx), 50, dtype=np.uint16)

        # Target nucleus at (32, 40), radius 8
        # Neighbor nucleus at (44, 40), radius 8 — 12 pixels apart.
        # annulus_scale=1.5 → outer radius 12; target's annulus reaches
        # x=44, and the neighbor's inner disk spans x=36..52, so the
        # neighbor's disk overlaps the right half of the target's annulus.
        for nuc_cx in (32, 44):
            r = 8.0
            for z in range(nz):
                y_off = (z - 5.0) * 1.0
                r2 = r * r - y_off * y_off
                if r2 <= 0:
                    continue
                import math
                rz = math.sqrt(r2)
                y0 = max(0, int(40 - rz))
                y1 = min(ny, int(40 + rz) + 1)
                x0 = max(0, int(nuc_cx - rz))
                x1 = min(nx, int(nuc_cx + rz) + 1)
                ys = np.arange(y0, y1).reshape(-1, 1)
                xs = np.arange(x0, x1).reshape(1, -1)
                dx = xs - nuc_cx
                dy = ys - 40
                stack[z, y0:y1, x0:x1][(dx * dx + dy * dy) <= rz * rz] = 1000

        target = _make_nucleus(index=1, x=32, y=40, z=5.0, size=16)
        neighbor = _make_nucleus(index=2, x=44, y=40, z=5.0, size=16)

        out = measure_timepoint_with_blot(stack, [target, neighbor], 1.0)
        sum_in, count_in, sum_ann, count_ann, sum_blot, count_blot = out[0]

        # Sanity: interior is bright
        assert count_in > 0
        assert sum_in / count_in == pytest.approx(1000, rel=1e-6)

        # Global annulus is contaminated by the neighbor → mean > 50
        assert count_ann > 0
        ann_mean = sum_ann / count_ann
        assert ann_mean > 100  # well above background thanks to neighbor

        # Blot annulus masks the neighbor out → mean much closer to 50
        assert count_blot > 0
        blot_mean = sum_blot / count_blot
        # Blot mean strictly lower than global annulus mean
        assert blot_mean < ann_mean
        # And within a few % of the true background
        assert blot_mean == pytest.approx(50, rel=0.05)

    def test_blot_count_le_annulus_count(self):
        """The blot mask is always a subset of the annulus (we only
        remove pixels).  So count_blot <= count_ann always."""
        stack = _make_sphere_stack(
            shape=(7, 48, 48),
            cx=24, cy=24, cz=3.0,
            radius=8.0, z_pix_res=1.0,
            value_in=500, value_out=10,
        )
        # Two nuclei, same Z plane, annuli overlapping
        n1 = _make_nucleus(index=1, x=20, y=24, z=3.0, size=16)
        n2 = _make_nucleus(index=2, x=32, y=24, z=3.0, size=16)
        out = measure_timepoint_with_blot(stack, [n1, n2], 1.0)
        for row in out:
            _, _, _, count_ann, _, count_blot = row
            assert count_blot <= count_ann


# ── CSV writer ─────────────────────────────────────────────────────


class TestWriteMeasureCsv:
    def test_header_and_rows(self, tmp_path: Path):
        out = tmp_path / "m.csv"
        rows = [
            ("ABa", 1, 3, [1000.0, 2000.0, None, None, None]),
            ("ABp", 2, 5, [None, 1500.0, 1600.0, 1700.0, 1800.0]),
        ]
        write_measure_csv(out, rows, n_timepoints=5)
        assert out.exists()

        with out.open() as f:
            reader = list(csv.reader(f))

        header = reader[0]
        assert header == [
            "cell_name", "start_time", "end_time", "t1", "t2", "t3", "t4", "t5",
        ]
        # ABa row: 1000 and 2000 are integers → formatted without decimals
        assert reader[1][:5] == ["ABa", "1", "3", "1000", "2000"]
        assert reader[1][5:] == ["", "", ""]
        # ABp row
        assert reader[2][0] == "ABp"
        assert reader[2][3] == ""
        assert reader[2][4] == "1500"

    def test_wrong_series_length_raises(self, tmp_path: Path):
        with pytest.raises(ValueError):
            write_measure_csv(
                tmp_path / "bad.csv",
                [("A", 1, 1, [1.0, 2.0])],
                n_timepoints=3,
            )

    def test_fractional_formatting(self, tmp_path: Path):
        out = tmp_path / "frac.csv"
        write_measure_csv(
            out,
            [("C", 1, 1, [123.456789])],
            n_timepoints=1,
        )
        reader = list(csv.reader(out.open()))
        assert reader[1][3] == "123.4568"

    def test_empty_rows(self, tmp_path: Path):
        out = tmp_path / "empty.csv"
        write_measure_csv(out, [], n_timepoints=3)
        reader = list(csv.reader(out.open()))
        assert len(reader) == 1
        assert reader[0] == ["cell_name", "start_time", "end_time", "t1", "t2", "t3"]


# ── run_measure integration ────────────────────────────────────────


@pytest.fixture
def manager_2tp_1nuc() -> NucleiManager:
    """Minimal NucleiManager with 2 timepoints, 1 nucleus each, lineage built."""
    mgr = NucleiManager()
    mgr._expr_corr = "global"
    mgr.movie.xy_res = 1.0
    mgr.movie.z_res = 1.0

    # Two timepoints, one nucleus at each — both named "ABa" via pred/succ link
    n1 = Nucleus(
        index=1, x=32, y=32, z=5.0, size=16, status=1,
        identity="ABa", predecessor=-1, successor1=1, successor2=-1,
    )
    n2 = Nucleus(
        index=1, x=32, y=32, z=5.0, size=16, status=1,
        identity="ABa", predecessor=1, successor1=-1, successor2=-1,
    )
    mgr.nuclei_record = [[n1], [n2]]
    mgr.set_all_successors()
    mgr.process(do_identity=False)  # builds lineage_tree
    assert mgr.lineage_tree is not None
    return mgr


def test_run_measure_writes_csvs_and_updates_rwraw(
    manager_2tp_1nuc: NucleiManager, tmp_path: Path,
):
    """Two-channel run_measure should create two CSVs and update AT channel."""
    # 5D stack: (T=2, C=2, Z=11, Y=64, X=64)
    # Channel 0 bright (value 1000), channel 1 dim (value 100)
    shape_3d = (11, 64, 64)
    c0 = np.stack([
        _make_sphere_stack(shape_3d, 32, 32, 5, 8.0, 1.0, 1000, 0),
        _make_sphere_stack(shape_3d, 32, 32, 5, 8.0, 1.0, 1000, 0),
    ])
    c1 = np.stack([
        _make_sphere_stack(shape_3d, 32, 32, 5, 8.0, 1.0, 100, 0),
        _make_sphere_stack(shape_3d, 32, 32, 5, 8.0, 1.0, 100, 0),
    ])
    # Stack to shape (T, C, Z, Y, X)
    data = np.stack([c0, c1], axis=1)
    assert data.shape == (2, 2, 11, 64, 64)
    provider = NumpyProvider(data)
    assert provider.num_channels == 2

    written = run_measure(
        manager_2tp_1nuc,
        provider,
        tmp_path,
        at_channel=0,
        progress_cb=None,
    )
    assert len(written) == 2
    for p in written:
        assert p.exists()

    # Channel 0 is the AT channel → filename contains "_AT"
    assert any("_AT" in p.name for p in written)

    # rwraw on each nucleus should reflect channel-0 measurement: mean ~1000,
    # scaled by SCALE=1000 → ~1,000,000.
    for nucs in manager_2tp_1nuc.nuclei_record:
        for n in nucs:
            assert n.rwraw > 0
            # Within 1% of 1000 * SCALE
            assert abs(n.rwraw - 1000 * SCALE) / (1000 * SCALE) < 0.05
            # rweight = rwraw - rwcorr1 under "global"
            assert n.rweight == n.rwraw - n.rwcorr1


def test_run_measure_blot_correction_writes_rwcorr3(tmp_path: Path):
    """Blot correction should populate rwcorr3 on the AT channel and
    switch the session expr_corr to 'blot' so compute_red_weights picks
    the rwcorr3 term for rweight."""
    mgr = NucleiManager()
    mgr._expr_corr = "none"  # will be flipped to blot by run_measure
    mgr.movie.xy_res = 1.0
    mgr.movie.z_res = 1.0

    # Two neighboring nuclei sharing a plane so their annuli overlap.
    n1 = Nucleus(
        index=1, x=32, y=40, z=5.0, size=16, status=1,
        identity="A", predecessor=-1, successor1=-1, successor2=-1,
    )
    n2 = Nucleus(
        index=2, x=44, y=40, z=5.0, size=16, status=1,
        identity="B", predecessor=-1, successor1=-1, successor2=-1,
    )
    mgr.nuclei_record = [[n1, n2]]
    mgr.set_all_successors()
    mgr.process(do_identity=False)

    # One-channel stack: bright nuclei, uniform background 50.
    shape_3d = (11, 80, 80)
    stack = np.full(shape_3d, 50, dtype=np.uint16)
    # Burn bright spheres at each nucleus position.
    for cx in (32, 44):
        sphere = _make_sphere_stack(shape_3d, cx, 40, 5.0, 8.0, 1.0, 1000, 0)
        stack = np.where(sphere > 0, sphere, stack).astype(np.uint16)
    data = stack[np.newaxis, ...]  # (T=1, Z, Y, X)
    provider = NumpyProvider(data)
    assert provider.num_channels == 1

    run_measure(
        mgr,
        provider,
        tmp_path,
        at_channel=0,
        progress_cb=None,
        correction_method="blot",
    )

    # Session expr_corr should now be "blot"
    assert mgr._expr_corr == "blot"

    # Both nuclei should have rwcorr3 populated (non-zero), and it should
    # be lower than rwcorr1 because blot masks out the bright neighbor.
    for n in (n1, n2):
        assert n.rwraw > 0
        assert n.rwcorr1 > 0
        assert n.rwcorr3 > 0
        # Blot < global because the bright neighbor inflates the raw annulus
        assert n.rwcorr3 < n.rwcorr1

    # rweight = rwraw - rwcorr3 in blot mode
    for n in (n1, n2):
        assert n.rweight == n.rwraw - n.rwcorr3


def test_run_measure_blot_matches_global_when_isolated(tmp_path: Path):
    """With a single isolated nucleus (no neighbors), blot == global.
    rwcorr3 should equal rwcorr1, and rweight differs only by choice
    of correction field."""
    mgr = NucleiManager()
    mgr.movie.xy_res = 1.0
    mgr.movie.z_res = 1.0
    n1 = Nucleus(
        index=1, x=32, y=32, z=5.0, size=16, status=1,
        identity="A", predecessor=-1, successor1=-1, successor2=-1,
    )
    mgr.nuclei_record = [[n1]]
    mgr.set_all_successors()
    mgr.process(do_identity=False)

    shape_3d = (11, 64, 64)
    stack = _make_sphere_stack(shape_3d, 32, 32, 5, 8.0, 1.0, 1000, 50)
    data = stack[np.newaxis, ...]
    provider = NumpyProvider(data)

    run_measure(
        mgr, provider, tmp_path,
        at_channel=0, progress_cb=None,
        correction_method="blot",
    )
    # With one nucleus, neighbor-masked annulus == full annulus
    assert n1.rwcorr1 == n1.rwcorr3


def test_run_measure_respects_correction_none(
    manager_2tp_1nuc: NucleiManager, tmp_path: Path,
):
    """Under expr_corr='none', rweight should equal rwraw directly."""
    manager_2tp_1nuc._expr_corr = "none"

    shape_3d = (11, 64, 64)
    per_tp = _make_sphere_stack(shape_3d, 32, 32, 5, 8.0, 1.0, 800, 0)
    data = np.stack([per_tp, per_tp])  # 4D (T, Z, Y, X)
    provider = NumpyProvider(data)
    assert provider.num_channels == 1

    run_measure(
        manager_2tp_1nuc,
        provider,
        tmp_path,
        at_channel=0,
        progress_cb=None,
    )
    for nucs in manager_2tp_1nuc.nuclei_record:
        for n in nucs:
            assert n.rwraw > 0
            assert n.rweight == n.rwraw


def test_run_measure_invalid_channel_raises(
    manager_2tp_1nuc: NucleiManager, tmp_path: Path,
):
    shape_3d = (11, 64, 64)
    data = np.zeros((2, 11, 64, 64), dtype=np.uint16)
    provider = NumpyProvider(data)
    with pytest.raises(ValueError):
        run_measure(
            manager_2tp_1nuc, provider, tmp_path,
            at_channel=5, progress_cb=None,
        )


def test_run_measure_empty_nuclei_raises(tmp_path: Path):
    mgr = NucleiManager()
    mgr._expr_corr = "none"
    # Build a provider so the at_channel check doesn't fire first
    provider = NumpyProvider(np.zeros((1, 3, 8, 8), dtype=np.uint16))
    with pytest.raises(ValueError):
        run_measure(mgr, provider, tmp_path, at_channel=0, progress_cb=None)


def test_run_measure_progress_callback_cancels(
    manager_2tp_1nuc: NucleiManager, tmp_path: Path,
):
    """Progress callback returning False should raise RuntimeError."""
    data = np.ones((2, 11, 64, 64), dtype=np.uint16) * 10
    provider = NumpyProvider(data)

    calls: list[tuple[int, int, int, int]] = []

    def cb(c, nc, t, nt):
        calls.append((c, nc, t, nt))
        return False  # cancel on first call

    with pytest.raises(RuntimeError, match="cancelled"):
        run_measure(
            manager_2tp_1nuc, provider, tmp_path,
            at_channel=0, progress_cb=cb,
        )
    assert calls  # callback was invoked at least once


def test_run_measure_csv_has_cell_row(
    manager_2tp_1nuc: NucleiManager, tmp_path: Path,
):
    """The produced CSV should contain a row for the 'ABa' cell across both timepoints."""
    shape_3d = (11, 64, 64)
    per_tp = _make_sphere_stack(shape_3d, 32, 32, 5, 8.0, 1.0, 500, 0)
    data = np.stack([per_tp, per_tp])
    provider = NumpyProvider(data)

    written = run_measure(
        manager_2tp_1nuc, provider, tmp_path,
        at_channel=0, progress_cb=None,
    )
    reader = list(csv.reader(written[0].open()))
    # Header + >=1 cell row
    assert reader[0][0] == "cell_name"
    data_rows = [r for r in reader[1:] if r]
    assert data_rows
    # The AB/ABa cell should be present
    names = [r[0] for r in data_rows]
    assert any("AB" in n for n in names)
