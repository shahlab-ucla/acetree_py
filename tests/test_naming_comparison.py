"""Naming comparison test framework.

Loads real AT ZIP files containing validated cell names from older AT
versions, performs de novo naming with the new unified pipeline, and
compares results cell-by-cell.

Usage as pytest:
    pytest tests/test_naming_comparison.py --zip-path /path/to/validated.zip -v

Usage as standalone script:
    python tests/test_naming_comparison.py /path/to/validated.zip --report naming_report.json
    python tests/test_naming_comparison.py /path/to/zip_dir/ --report naming_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acetree_py.core.nucleus import Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.naming.identity import IdentityAssigner, NEWCANONICAL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MismatchDetail:
    """A single cell-level naming mismatch."""

    timepoint: int
    index: int
    ref_name: str
    new_name: str
    parent_ref_name: str = ""
    is_sister_swap: bool = False
    is_cascade: bool = False
    root_cause_time: int = -1
    root_cause_name: str = ""


@dataclass
class ComparisonReport:
    """Full comparison report between reference and de novo naming."""

    zip_path: str = ""
    total_cells: int = 0
    matched: int = 0
    mismatched: int = 0
    unnamed_in_reference: int = 0
    unnamed_in_new: int = 0
    sister_swaps: int = 0
    cascade_roots: int = 0
    cascade_descendants: int = 0
    first_mismatch_time: int = -1
    match_rate: float = 0.0
    mismatch_details: list[MismatchDetail] = field(default_factory=list)
    mismatch_by_lineage: dict[str, int] = field(default_factory=dict)
    founder_confidence: float = 0.0
    timing_confidence: float = 0.0
    size_confidence: float = 0.0
    axis_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def load_reference_names(
    nuclei_record: list[list[Nucleus]],
) -> dict[tuple[int, int], str]:
    """Extract reference names from a loaded nuclei record.

    Returns:
        Dict mapping (timepoint, nucleus_index) -> name.
        Only includes cells with non-empty names and status >= 1.
    """
    ref = {}
    for t, nuclei in enumerate(nuclei_record):
        for j, nuc in enumerate(nuclei):
            if nuc.status >= 1 and nuc.identity:
                ref[(t, j)] = nuc.identity
    return ref


def clear_all_names(nuclei_record: list[list[Nucleus]]) -> None:
    """Clear all non-forced names (preserve assigned_id)."""
    for t_nuclei in nuclei_record:
        for nuc in t_nuclei:
            if nuc.assigned_id:
                continue
            nuc.identity = ""


def run_denovo_naming(
    nuclei_record: list[list[Nucleus]],
    z_pix_res: float = 11.1,
    legacy_mode: bool = False,
) -> IdentityAssigner:
    """Run de novo naming on a nuclei record.

    Returns the IdentityAssigner for confidence inspection.
    """
    assigner = IdentityAssigner(
        nuclei_record,
        naming_method=NEWCANONICAL,
        z_pix_res=z_pix_res,
        legacy_mode=legacy_mode,
    )
    assigner.assign_identities()
    return assigner


def get_new_names(
    nuclei_record: list[list[Nucleus]],
) -> dict[tuple[int, int], str]:
    """Extract names from the nuclei record after de novo naming."""
    names = {}
    for t, nuclei in enumerate(nuclei_record):
        for j, nuc in enumerate(nuclei):
            if nuc.status >= 1 and nuc.identity:
                names[(t, j)] = nuc.identity
    return names


def _lineage_root(name: str) -> str:
    """Get the founder lineage root of a Sulston name."""
    if not name:
        return ""
    # Standard lineage prefixes
    for prefix in ("ABa", "ABp", "EMS", "P2", "P3", "P4", "C", "D", "E", "MS",
                   "AB", "P1", "P0"):
        if name.startswith(prefix):
            return prefix
    return name


def _detect_sister_swap(
    ref_names: dict[tuple[int, int], str],
    new_names: dict[tuple[int, int], str],
    nuclei_record: list[list[Nucleus]],
    t: int,
    j: int,
) -> bool:
    """Check if a mismatch is a sister swap (daughters got each other's names)."""
    nuc = nuclei_record[t][j]

    # Find the sister nucleus (shares the same predecessor)
    if t == 0 or nuc.predecessor < 1:
        return False

    pred_idx = nuc.predecessor - 1
    if pred_idx < 0 or pred_idx >= len(nuclei_record[t - 1]):
        return False

    pred = nuclei_record[t - 1][pred_idx]
    if pred.successor2 < 1:
        return False  # not a division

    # Find which successor we are and who the sister is
    s1_idx = pred.successor1 - 1
    s2_idx = pred.successor2 - 1
    if j == s1_idx:
        sister_idx = s2_idx
    elif j == s2_idx:
        sister_idx = s1_idx
    else:
        return False

    if sister_idx < 0 or sister_idx >= len(nuclei_record[t]):
        return False

    # Check if we got the sister's reference name and vice versa
    ref_me = ref_names.get((t, j), "")
    ref_sister = ref_names.get((t, sister_idx), "")
    new_me = new_names.get((t, j), "")
    new_sister = new_names.get((t, sister_idx), "")

    return (new_me == ref_sister and new_sister == ref_me
            and ref_me and ref_sister and ref_me != ref_sister)


def compare_naming(
    nuclei_record: list[list[Nucleus]],
    ref_names: dict[tuple[int, int], str],
    new_names: dict[tuple[int, int], str],
    zip_path: str = "",
    assigner: IdentityAssigner | None = None,
) -> ComparisonReport:
    """Compare reference names against de novo names cell-by-cell.

    Performs:
    - Exact match counting
    - Sister swap detection
    - Cascade analysis (errors propagating from a single root cause)
    - Per-lineage mismatch breakdown

    Returns:
        ComparisonReport with full comparison details.
    """
    report = ComparisonReport(zip_path=zip_path)

    # Gather confidence from the assigner
    if assigner is not None and assigner.founder_assignment is not None:
        fa = assigner.founder_assignment
        report.founder_confidence = fa.confidence
        report.timing_confidence = fa.timing_confidence
        report.size_confidence = fa.size_confidence
        report.axis_confidence = fa.axis_confidence

    # All cells that have a reference name OR a new name
    all_keys = set(ref_names.keys()) | set(new_names.keys())
    report.total_cells = len(all_keys)

    mismatches: list[MismatchDetail] = []
    mismatch_set: set[tuple[int, int]] = set()
    lineage_counts: dict[str, int] = defaultdict(int)

    for (t, j) in sorted(all_keys):
        ref = ref_names.get((t, j), "")
        new = new_names.get((t, j), "")

        if not ref:
            report.unnamed_in_reference += 1
            continue
        if not new:
            report.unnamed_in_new += 1
            continue

        if ref == new:
            report.matched += 1
        else:
            report.mismatched += 1
            mismatch_set.add((t, j))
            lineage_counts[_lineage_root(ref)] += 1

            if report.first_mismatch_time < 0 or t < report.first_mismatch_time:
                report.first_mismatch_time = t

            # Get parent reference name
            parent_ref = ""
            nuc = nuclei_record[t][j]
            if t > 0 and nuc.predecessor >= 1:
                pred_idx = nuc.predecessor - 1
                if 0 <= pred_idx < len(nuclei_record[t - 1]):
                    parent_ref = ref_names.get((t - 1, pred_idx), "")

            is_swap = _detect_sister_swap(
                ref_names, new_names, nuclei_record, t, j,
            )

            mismatches.append(MismatchDetail(
                timepoint=t,
                index=j,
                ref_name=ref,
                new_name=new,
                parent_ref_name=parent_ref,
                is_sister_swap=is_swap,
            ))

    # Cascade analysis: find root-cause mismatches
    # A mismatch is a "cascade" if its parent was also mismatched
    cascade_roots_set: set[tuple[int, int]] = set()
    for detail in mismatches:
        t, j = detail.timepoint, detail.index
        nuc = nuclei_record[t][j]
        if t > 0 and nuc.predecessor >= 1:
            pred_idx = nuc.predecessor - 1
            if (t - 1, pred_idx) in mismatch_set:
                detail.is_cascade = True
                # Trace up to find root
                root_t, root_j = t - 1, pred_idx
                while root_t > 0:
                    rnuc = nuclei_record[root_t][root_j]
                    if rnuc.predecessor < 1:
                        break
                    rpred_idx = rnuc.predecessor - 1
                    if (root_t - 1, rpred_idx) not in mismatch_set:
                        break
                    root_t -= 1
                    root_j = rpred_idx
                detail.root_cause_time = root_t
                detail.root_cause_name = ref_names.get((root_t, root_j), "")
                cascade_roots_set.add((root_t, root_j))

    report.sister_swaps = sum(1 for d in mismatches if d.is_sister_swap)
    report.cascade_roots = len(cascade_roots_set)
    report.cascade_descendants = sum(1 for d in mismatches if d.is_cascade)
    report.mismatch_details = mismatches
    report.mismatch_by_lineage = dict(lineage_counts)

    named_cells = report.matched + report.mismatched
    report.match_rate = report.matched / named_cells if named_cells > 0 else 0.0

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(report: ComparisonReport) -> None:
    """Print a concise console summary of the comparison."""
    print(f"\n{'='*60}")
    print(f"Naming Comparison: {report.zip_path}")
    print(f"{'='*60}")
    print(f"Total cells:       {report.total_cells}")
    print(f"Matched:           {report.matched}")
    print(f"Mismatched:        {report.mismatched}")
    print(f"Match rate:        {report.match_rate:.1%}")
    print(f"Unnamed (ref):     {report.unnamed_in_reference}")
    print(f"Unnamed (new):     {report.unnamed_in_new}")
    print(f"Sister swaps:      {report.sister_swaps}")
    print(f"Cascade roots:     {report.cascade_roots}")
    print(f"Cascade descendants: {report.cascade_descendants}")
    print(f"First mismatch at: t={report.first_mismatch_time}")
    print(f"\nFounder confidence: {report.founder_confidence:.2f} "
          f"(timing={report.timing_confidence:.2f}, "
          f"size={report.size_confidence:.2f}, "
          f"axis={report.axis_confidence:.2f})")

    if report.mismatch_by_lineage:
        print(f"\nMismatches by lineage:")
        for lineage, count in sorted(
            report.mismatch_by_lineage.items(), key=lambda x: -x[1],
        ):
            print(f"  {lineage:10s} {count}")

    # Top 10 mismatched cells (non-cascade roots first)
    roots = [d for d in report.mismatch_details if not d.is_cascade]
    roots.sort(key=lambda d: d.timepoint)
    print(f"\nTop non-cascade mismatches (first 10):")
    for d in roots[:10]:
        swap_tag = " [SWAP]" if d.is_sister_swap else ""
        print(f"  t={d.timepoint:4d} idx={d.index:3d}: "
              f"ref={d.ref_name:15s} new={d.new_name:15s} "
              f"parent={d.parent_ref_name}{swap_tag}")
    print()


def save_json_report(report: ComparisonReport, path: Path) -> None:
    """Save the full report as JSON."""
    data = {
        "zip_path": report.zip_path,
        "total_cells": report.total_cells,
        "matched": report.matched,
        "mismatched": report.mismatched,
        "match_rate": report.match_rate,
        "unnamed_in_reference": report.unnamed_in_reference,
        "unnamed_in_new": report.unnamed_in_new,
        "sister_swaps": report.sister_swaps,
        "cascade_roots": report.cascade_roots,
        "cascade_descendants": report.cascade_descendants,
        "first_mismatch_time": report.first_mismatch_time,
        "founder_confidence": report.founder_confidence,
        "timing_confidence": report.timing_confidence,
        "size_confidence": report.size_confidence,
        "axis_confidence": report.axis_confidence,
        "mismatch_by_lineage": report.mismatch_by_lineage,
        "mismatch_details": [asdict(d) for d in report.mismatch_details],
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"JSON report saved to: {path}")


def save_csv_mismatches(report: ComparisonReport, path: Path) -> None:
    """Save mismatch details as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timepoint", "index", "ref_name", "new_name",
            "parent_ref_name", "is_sister_swap", "is_cascade",
            "root_cause_time", "root_cause_name",
        ])
        for d in report.mismatch_details:
            writer.writerow([
                d.timepoint, d.index, d.ref_name, d.new_name,
                d.parent_ref_name, d.is_sister_swap, d.is_cascade,
                d.root_cause_time, d.root_cause_name,
            ])
    print(f"CSV mismatches saved to: {path}")


# ---------------------------------------------------------------------------
# Run a full comparison on a single ZIP
# ---------------------------------------------------------------------------

def run_comparison(
    zip_path: Path,
    z_pix_res: float = 11.1,
    legacy_mode: bool = False,
) -> ComparisonReport:
    """Run the full comparison pipeline on a single ZIP file.

    1. Load ZIP and extract reference names
    2. Clear names, run de novo naming
    3. Compare cell-by-cell

    Returns:
        ComparisonReport
    """
    # Load
    nuclei_record = read_nuclei_zip(zip_path)
    ref_names = load_reference_names(nuclei_record)

    # Clear and re-name
    clear_all_names(nuclei_record)
    assigner = run_denovo_naming(
        nuclei_record, z_pix_res=z_pix_res, legacy_mode=legacy_mode,
    )

    # Collect new names
    new_names = get_new_names(nuclei_record)

    # Compare
    report = compare_naming(
        nuclei_record, ref_names, new_names,
        zip_path=str(zip_path),
        assigner=assigner,
    )

    return report


# ---------------------------------------------------------------------------
# Pytest integration
# ---------------------------------------------------------------------------

@pytest.fixture
def zip_paths(request: pytest.FixtureRequest) -> list[Path]:
    """Collect ZIP paths from --zip-path option or NAMING_TEST_ZIP env var."""
    import os

    raw = os.environ.get("NAMING_TEST_ZIP")
    if raw is None:
        pytest.skip(
            "No test ZIP provided. Set NAMING_TEST_ZIP=/path/to/file.zip "
            "or NAMING_TEST_ZIP=/path/to/zip_dir/"
        )
    p = Path(raw)
    if p.is_dir():
        zips = sorted(p.glob("*.zip"))
        if not zips:
            pytest.skip(f"No ZIP files found in {p}")
        return zips
    elif p.is_file():
        return [p]
    else:
        pytest.skip(f"Path does not exist: {p}")


@pytest.fixture
def z_pix_res() -> float:
    import os
    return float(os.environ.get("NAMING_TEST_ZPIXRES", "11.1"))


class TestNamingComparison:
    """Test class for naming comparison against validated ZIPs."""

    def test_denovo_naming_matches_reference(
        self, zip_paths: list[Path], z_pix_res: float,
    ) -> None:
        """De novo naming should closely match validated reference names."""
        for zip_path in zip_paths:
            report = run_comparison(zip_path, z_pix_res=z_pix_res)
            print_summary(report)

            # Basic sanity — we should name something
            assert report.total_cells > 0, f"No cells found in {zip_path}"
            assert report.matched > 0, f"No matching names in {zip_path}"

            # Log the match rate (don't fail on specific threshold yet —
            # we need real data to calibrate)
            logger.info(
                "%s: match_rate=%.1f%% (%d/%d)",
                zip_path.name, report.match_rate * 100,
                report.matched, report.matched + report.mismatched,
            )

    def test_legacy_pipeline_comparison(
        self, zip_paths: list[Path], z_pix_res: float,
    ) -> None:
        """Compare new unified pipeline against legacy pipeline."""
        for zip_path in zip_paths:
            new_report = run_comparison(
                zip_path, z_pix_res=z_pix_res, legacy_mode=False,
            )
            legacy_report = run_comparison(
                zip_path, z_pix_res=z_pix_res, legacy_mode=True,
            )

            print(f"\n--- {zip_path.name} ---")
            print(f"New pipeline:    {new_report.match_rate:.1%}")
            print(f"Legacy pipeline: {legacy_report.match_rate:.1%}")

            # New pipeline should be at least as good as legacy
            # (or very close — allow 1% margin for edge cases)
            margin = 0.01
            assert new_report.match_rate >= legacy_report.match_rate - margin, (
                f"New pipeline ({new_report.match_rate:.1%}) significantly worse "
                f"than legacy ({legacy_report.match_rate:.1%}) on {zip_path.name}"
            )


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare de novo naming against validated AT ZIP files",
    )
    parser.add_argument(
        "path",
        help="Path to a ZIP file or directory of ZIPs",
    )
    parser.add_argument(
        "--report", "-r",
        default=None,
        help="Path for JSON report output",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path for CSV mismatch output",
    )
    parser.add_argument(
        "--z-pix-res",
        type=float,
        default=11.1,
        help="Z pixel resolution (default: 11.1)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run legacy pipeline instead of unified",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both pipelines and compare",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    p = Path(args.path)
    if p.is_dir():
        zip_files = sorted(p.glob("*.zip"))
    elif p.is_file():
        zip_files = [p]
    else:
        print(f"Error: {p} does not exist")
        sys.exit(1)

    if not zip_files:
        print(f"No ZIP files found in {p}")
        sys.exit(1)

    all_reports = []
    for zf in zip_files:
        report = run_comparison(
            zf, z_pix_res=args.z_pix_res, legacy_mode=args.legacy,
        )
        print_summary(report)
        all_reports.append(report)

        if args.both:
            legacy_report = run_comparison(
                zf, z_pix_res=args.z_pix_res, legacy_mode=True,
            )
            print(f"  Legacy pipeline: {legacy_report.match_rate:.1%}")
            print(f"  New pipeline:    {report.match_rate:.1%}")

    if args.report:
        report_path = Path(args.report)
        if len(all_reports) == 1:
            save_json_report(all_reports[0], report_path)
        else:
            # Aggregate
            data = {
                "summary": {
                    "num_zips": len(all_reports),
                    "avg_match_rate": sum(r.match_rate for r in all_reports) / len(all_reports),
                    "total_matched": sum(r.matched for r in all_reports),
                    "total_mismatched": sum(r.mismatched for r in all_reports),
                },
                "per_zip": [
                    {
                        "zip_path": r.zip_path,
                        "match_rate": r.match_rate,
                        "matched": r.matched,
                        "mismatched": r.mismatched,
                        "founder_confidence": r.founder_confidence,
                    }
                    for r in all_reports
                ],
            }
            report_path.write_text(json.dumps(data, indent=2))
            print(f"Aggregate JSON report saved to: {report_path}")

    if args.csv and all_reports:
        csv_path = Path(args.csv)
        # Concatenate all mismatch details
        combined = ComparisonReport()
        for r in all_reports:
            combined.mismatch_details.extend(r.mismatch_details)
        save_csv_mismatches(combined, csv_path)


if __name__ == "__main__":
    main()
