"""Tests for analysis/export.py — CSV and Newick export functions.

Tests cover:
- Cell table CSV: correct headers, row count, field values
- Nucleus table CSV: correct headers, alive-only filtering
- Newick export: format correctness, branch lengths, roundtrip
- Expression CSV: expression-specific columns
- File and StringIO output modes
- Edge cases: empty trees, single cell, no expression
"""

from __future__ import annotations

import csv
import io

import pytest

from acetree_py.core.cell import Cell, CellFate
from acetree_py.core.lineage import LineageTree
from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.analysis.export import (
    export_cell_table_csv,
    export_expression_csv,
    export_newick,
    export_nucleus_table_csv,
)


# ── Helpers ──────────────────────────────────────────────────────


def _nuc(index=1, x=100, y=200, z=10.0, size=20, identity="ABa",
         status=1, rweight=50, rwraw=60, predecessor=NILLI):
    return Nucleus(
        index=index, x=x, y=y, z=z, size=size, identity=identity,
        status=status, predecessor=predecessor,
        rweight=rweight, rwraw=rwraw,
    )


def _make_simple_tree() -> LineageTree:
    """
    Tree structure:
        P0 (t=1-1, DIVIDED)
        ├── AB (t=2-4, DIVIDED, rweights=[10, 20, 30])
        │   ├── ABa (t=5-6, ALIVE, rweights=[100, 200])
        │   └── ABp (t=5-6, ALIVE, rweights=[50, 60])
        └── P1 (t=2-4, ALIVE, rweights=[5, 10, 15])
    """
    p0 = Cell(name="P0", start_time=1, end_time=1, end_fate=CellFate.DIVIDED)
    p0.add_nucleus(1, _nuc(rweight=0, identity="P0"))

    ab = Cell(name="AB", start_time=2, end_time=4, end_fate=CellFate.DIVIDED)
    for i, rw in enumerate([10, 20, 30]):
        ab.add_nucleus(2 + i, _nuc(rweight=rw, identity="AB"))

    p1 = Cell(name="P1", start_time=2, end_time=4, end_fate=CellFate.ALIVE)
    for i, rw in enumerate([5, 10, 15]):
        p1.add_nucleus(2 + i, _nuc(rweight=rw, identity="P1"))

    aba = Cell(name="ABa", start_time=5, end_time=6)
    for i, rw in enumerate([100, 200]):
        aba.add_nucleus(5 + i, _nuc(rweight=rw, identity="ABa"))

    abp = Cell(name="ABp", start_time=5, end_time=6)
    for i, rw in enumerate([50, 60]):
        abp.add_nucleus(5 + i, _nuc(rweight=rw, identity="ABp"))

    p0.add_child(ab)
    p0.add_child(p1)
    ab.add_child(aba)
    ab.add_child(abp)

    tree = LineageTree(
        root=p0,
        cells_by_name={
            "P0": p0, "AB": ab, "P1": p1, "ABa": aba, "ABp": abp,
        },
    )
    return tree


def _make_nuclei_record():
    """Create a simple nuclei record: 3 timepoints, a few nuclei each."""
    return [
        [_nuc(1, 300, 250, 15.0, 20, "P0", 1, 50, 60)],
        [
            _nuc(1, 280, 240, 14.0, 18, "AB", 1, 10, 15, predecessor=1),
            _nuc(2, 320, 260, 16.0, 22, "P1", 1, 5, 8, predecessor=1),
            _nuc(3, 100, 100, 5.0, 10, "", -1, 0, 0),  # Dead nucleus
        ],
        [
            _nuc(1, 260, 230, 13.0, 16, "ABa", 1, 100, 120, predecessor=1),
            _nuc(2, 300, 250, 15.0, 20, "ABp", 1, 50, 65, predecessor=1),
        ],
    ]


# ── Cell table CSV tests ────────────────────────────────────────


class TestExportCellTableCSV:

    def test_correct_headers(self):
        tree = _make_simple_tree()
        buf = io.StringIO()
        export_cell_table_csv(tree, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        expected = {
            "name", "start_time", "end_time", "lifetime", "fate",
            "parent", "num_children", "depth",
            "mean_expression", "max_expression", "onset_time",
        }
        assert set(reader.fieldnames) == expected

    def test_row_count(self):
        tree = _make_simple_tree()
        buf = io.StringIO()
        export_cell_table_csv(tree, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = list(reader)
        assert len(rows) == 5  # P0, AB, P1, ABa, ABp

    def test_cell_values(self):
        tree = _make_simple_tree()
        buf = io.StringIO()
        export_cell_table_csv(tree, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = {r["name"]: r for r in reader}

        assert rows["AB"]["fate"] == "DIVIDED"
        assert rows["AB"]["parent"] == "P0"
        assert rows["AB"]["num_children"] == "2"
        assert rows["P0"]["parent"] == ""

    def test_sorted_by_start_time(self):
        tree = _make_simple_tree()
        buf = io.StringIO()
        export_cell_table_csv(tree, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = list(reader)
        times = [int(r["start_time"]) for r in rows]
        assert times == sorted(times)

    def test_write_to_file(self, tmp_path):
        tree = _make_simple_tree()
        path = tmp_path / "cells.csv"
        export_cell_table_csv(tree, path)

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5


# ── Nucleus table CSV tests ─────────────────────────────────────


class TestExportNucleusTableCSV:

    def test_correct_headers(self):
        record = _make_nuclei_record()
        buf = io.StringIO()
        export_nucleus_table_csv(record, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        assert "timepoint" in reader.fieldnames
        assert "x" in reader.fieldnames
        assert "rweight" in reader.fieldnames

    def test_alive_only(self):
        record = _make_nuclei_record()
        buf = io.StringIO()
        export_nucleus_table_csv(record, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = list(reader)
        # t1: 1 alive, t2: 2 alive (1 dead excluded), t3: 2 alive = 5 total
        assert len(rows) == 5

    def test_timepoints_correct(self):
        record = _make_nuclei_record()
        buf = io.StringIO()
        export_nucleus_table_csv(record, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        timepoints = [int(r["timepoint"]) for r in reader]
        assert 1 in timepoints
        assert 2 in timepoints
        assert 3 in timepoints

    def test_write_to_file(self, tmp_path):
        record = _make_nuclei_record()
        path = tmp_path / "nuclei.csv"
        export_nucleus_table_csv(record, path)

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5


# ── Newick export tests ─────────────────────────────────────────


class TestExportNewick:

    def test_basic_format(self):
        tree = _make_simple_tree()
        newick = export_newick(tree)

        assert newick.endswith(";")
        assert "P0" in newick
        assert "ABa" in newick
        assert "ABp" in newick

    def test_branch_lengths(self):
        tree = _make_simple_tree()
        newick = export_newick(tree, include_branch_lengths=True)

        # ABa has lifetime 2 (t=5-6)
        assert "ABa:2" in newick
        # P0 has lifetime 1 (t=1-1)
        assert newick.endswith("P0:1;")

    def test_no_branch_lengths(self):
        tree = _make_simple_tree()
        newick = export_newick(tree, include_branch_lengths=False)

        assert ":" not in newick.replace(";", "")
        assert "ABa" in newick

    def test_parentheses_balanced(self):
        tree = _make_simple_tree()
        newick = export_newick(tree)

        assert newick.count("(") == newick.count(")")

    def test_leaf_format(self):
        # Single cell tree (leaf)
        cell = Cell(name="X", start_time=1, end_time=5)
        tree = LineageTree(root=cell, cells_by_name={"X": cell})
        newick = export_newick(tree)
        assert newick == "X:5;"

    def test_empty_tree(self):
        tree = LineageTree()
        newick = export_newick(tree)
        assert newick == ";"

    def test_write_to_file(self, tmp_path):
        tree = _make_simple_tree()
        path = tmp_path / "tree.nwk"
        newick = export_newick(tree, output=path)

        assert path.exists()
        content = path.read_text().strip()
        assert content == newick

    def test_write_to_stringio(self):
        tree = _make_simple_tree()
        buf = io.StringIO()
        newick = export_newick(tree, output=buf)

        buf.seek(0)
        assert newick in buf.read()


# ── Expression CSV tests ────────────────────────────────────────


class TestExportExpressionCSV:

    def test_correct_headers(self):
        tree = _make_simple_tree()
        buf = io.StringIO()
        export_expression_csv(tree, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        assert "expr_mean" in reader.fieldnames
        assert "expr_max" in reader.fieldnames
        assert "onset_time" in reader.fieldnames

    def test_expression_values(self):
        tree = _make_simple_tree()
        buf = io.StringIO()
        export_expression_csv(tree, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = {r["name"]: r for r in reader}

        # ABa: mean = (100+200)/2 = 150
        assert float(rows["ABa"]["expr_mean"]) == 150.0
        assert float(rows["ABa"]["expr_max"]) == 200.0

    def test_row_count(self):
        tree = _make_simple_tree()
        buf = io.StringIO()
        export_expression_csv(tree, buf)

        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = list(reader)
        assert len(rows) == 5
