"""Command-line entry point for the local MATLAB differential oracle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .experiment import (
    default_experiment_config,
    resolution_experiment_config,
    run_parity_experiment,
    write_experiment_outputs,
)
from .matlab_backend import MatlabOracleConfig, MatlabOracleError, MatlabStarryNiteOracle


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Python StarryNite rewrite with a local MATLAB checkout "
            "using identical deterministic 3-D simulations."
        )
    )
    parser.add_argument(
        "--starrynite-root",
        default=os.environ.get("STARRYNITE_ROOT"),
        help="Path to a zhirongbaolab/StarryNite checkout (or STARRYNITE_ROOT).",
    )
    parser.add_argument(
        "--matlab",
        default=os.environ.get("MATLAB_EXECUTABLE"),
        help="Path to matlab.exe; auto-detected when omitted.",
    )
    parser.add_argument(
        "--parameter-file",
        help="Optional legacy StarryNite parameter file used as the staged baseline.",
    )
    parser.add_argument(
        "--cell-count",
        type=int,
        help="Cell count used to select staged legacy values.",
    )
    parser.add_argument(
        "--radius-um",
        type=float,
        default=2.0,
        help="Canonical simulated nucleus radius in microns (default: 2).",
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "full", "resolution"),
        default="smoke",
        help=(
            "Smoke is fast; full covers more scenes and sweep points; "
            "resolution runs a multiseed separation/noise matrix."
        ),
    )
    parser.add_argument("--seed", type=int, default=1731)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=3,
        help="Number of independent noise seeds for --suite resolution.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("starrynite-parity-report"),
        help="Directory for JSON, CSV, and Markdown reports.",
    )
    arguments = parser.parse_args()
    if not arguments.starrynite_root:
        parser.error("--starrynite-root or STARRYNITE_ROOT is required")

    try:
        synthetic_movies = None
        if arguments.suite == "resolution":
            config, legacy, synthetic_movies = resolution_experiment_config(
                parameter_file=arguments.parameter_file,
                cell_count=arguments.cell_count,
                radius_um=arguments.radius_um,
                seed=arguments.seed,
                seed_count=arguments.seed_count,
            )
        else:
            config, legacy = default_experiment_config(
                suite=arguments.suite,
                parameter_file=arguments.parameter_file,
                cell_count=arguments.cell_count,
                radius_um=arguments.radius_um,
                seed=arguments.seed,
            )
        matlab_config = MatlabOracleConfig.discover(
            arguments.starrynite_root,
            matlab_executable=arguments.matlab,
            timeout_seconds=arguments.timeout,
        )
        oracle = MatlabStarryNiteOracle(matlab_config)
        report = run_parity_experiment(
            oracle,
            config,
            legacy_parameters=legacy,
            synthetic_movies=synthetic_movies,
        )
        write_experiment_outputs(arguments.output, report)
    except (OSError, ValueError, MatlabOracleError) as exc:
        parser.exit(2, f"StarryNite parity experiment failed: {exc}\n")
    summary = report["summary"]
    print(f"Completed {summary['trial_count']} detector parity trials.")
    print(f"Report: {(arguments.output / 'summary.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
