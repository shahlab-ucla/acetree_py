"""End-to-end synthetic parity experiments and machine-readable reports."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ...api import Calibration, Detection
from ..detector import StarryNiteDetector, legacy_dog_response
from ..parameters import ParameterValue, read_parameter_file
from ..presets import tuning_profile_from_parameters
from .matlab_backend import (
    MatlabStarryNiteOracle,
    full_detection_request,
    separable_dog_request,
    write_run_manifest,
)
from .metrics import (
    compare_detections,
    compare_sensitivity_curves,
    compare_volumes,
    matched_scalar_errors,
)
from .synthetic import (
    SyntheticMovie,
    default_synthetic_suite,
    resolution_noise_suite,
)


@dataclass(frozen=True, slots=True)
class OracleBaseline:
    radius_um: float = 2.0
    sigma_factor: float = 1.0
    intensity_threshold: float = 18.0
    boundary_percent: float = 0.35
    cell_count: int = 4
    parameter_file: str = ""
    parameter_sha256: str = ""


@dataclass(frozen=True, slots=True)
class ParameterSweep:
    name: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.name not in {"sigma", "intensitythreshold", "boundary_percent"}:
            raise ValueError(f"Unsupported detector sweep: {self.name}")
        if len(self.values) < 2 or any(not np.isfinite(value) for value in self.values):
            raise ValueError("Parameter sweeps need at least two finite values")
        if any(next_value <= value for value, next_value in zip(self.values, self.values[1:])):
            raise ValueError("Parameter sweep values must be strictly increasing")


@dataclass(frozen=True, slots=True)
class ParityExperimentConfig:
    baseline: OracleBaseline
    sweeps: tuple[ParameterSweep, ...]
    scenario_frames: tuple[tuple[str, int], ...]
    seed: int = 1731
    matching_tolerance_um: float = 2.0


@dataclass(frozen=True, slots=True)
class _Trial:
    trial_id: str
    scenario: str
    frame: int
    sweep: str
    parameter_value: float
    sigma_factor: float
    intensity_threshold: float
    boundary_percent: float


def default_experiment_config(
    *,
    suite: str = "smoke",
    parameter_file: str | Path | None = None,
    cell_count: int | None = None,
    radius_um: float = 2.0,
    seed: int = 1731,
) -> tuple[ParityExperimentConfig, dict[str, Any]]:
    """Build a canonical experiment, optionally seeded by a legacy file."""

    baseline, legacy = baseline_from_parameter_file(
        parameter_file,
        cell_count=cell_count,
        radius_um=radius_um,
    )
    if suite == "smoke":
        factors = (0.75, 1.0, 1.25)
        threshold_factors = (0.4, 1.0, 2.5)
        boundary_values = (0.2, baseline.boundary_percent, 0.65)
        scenario_frames = (("isolated", 1), ("close_pair", 1), ("division", 4))
    elif suite == "full":
        factors = (0.6, 0.8, 1.0, 1.2, 1.4)
        threshold_factors = (0.25, 0.5, 1.0, 2.0, 4.0)
        boundary_values = (0.15, 0.25, baseline.boundary_percent, 0.5, 0.7, 0.85)
        scenario_frames = (
            ("isolated", 1),
            ("close_pair", 1),
            ("unequal_pair", 1),
            ("boundary", 1),
            ("division", 3),
            ("division", 4),
            ("division", 6),
        )
    else:
        raise ValueError("suite must be 'smoke' or 'full'")
    sigma_values = _unique_sorted(baseline.sigma_factor * item for item in factors)
    threshold_values = _unique_sorted(
        baseline.intensity_threshold * item for item in threshold_factors
    )
    boundary_values = _unique_sorted(boundary_values)
    config = ParityExperimentConfig(
        baseline=baseline,
        sweeps=(
            ParameterSweep("sigma", sigma_values),
            ParameterSweep("intensitythreshold", threshold_values),
            ParameterSweep("boundary_percent", boundary_values),
        ),
        scenario_frames=scenario_frames,
        seed=seed,
        matching_tolerance_um=radius_um,
    )
    return config, legacy


def resolution_experiment_config(
    *,
    parameter_file: str | Path | None = None,
    cell_count: int | None = None,
    radius_um: float = 2.0,
    seed: int = 1731,
    seed_count: int = 3,
) -> tuple[ParityExperimentConfig, dict[str, Any], tuple[SyntheticMovie, ...]]:
    """Build the multiseed separation/noise regression matrix."""

    baseline, legacy = baseline_from_parameter_file(
        parameter_file,
        cell_count=cell_count,
        radius_um=radius_um,
    )
    movies = resolution_noise_suite(seed=seed, seed_count=seed_count)
    config = ParityExperimentConfig(
        baseline=baseline,
        sweeps=(),
        scenario_frames=tuple((movie.name, 1) for movie in movies),
        seed=seed,
        matching_tolerance_um=radius_um,
    )
    return config, legacy, movies


def baseline_from_parameter_file(
    parameter_file: str | Path | None,
    *,
    cell_count: int | None,
    radius_um: float,
) -> tuple[OracleBaseline, dict[str, Any]]:
    """Resolve staged legacy values but keep canonical synthetic geometry."""

    if radius_um <= 0 or not np.isfinite(radius_um):
        raise ValueError("radius_um must be positive and finite")
    if parameter_file is None:
        return OracleBaseline(radius_um=float(radius_um)), {}
    path = Path(parameter_file).resolve(strict=True)
    parameters = read_parameter_file(path)
    profile = tuning_profile_from_parameters(
        parameters,
        cell_count=cell_count,
        fallback_radius_um=radius_um,
    )
    detector = profile.detector_settings
    baseline = OracleBaseline(
        radius_um=float(radius_um),
        sigma_factor=float(detector.get("SIGMA", 1.0)),
        intensity_threshold=float(detector.get("INTENSITY_THRESHOLD", 18.0)),
        boundary_percent=float(detector.get("BOUNDARY_PERCENT", 0.35)),
        cell_count=profile.cell_count,
        parameter_file=path.name,
        parameter_sha256=profile.parameter_sha256 or "",
    )
    legacy = _legacy_detector_parameters(parameters.normalized_settings)
    return baseline, legacy


def run_parity_experiment(
    oracle: MatlabStarryNiteOracle,
    config: ParityExperimentConfig,
    *,
    legacy_parameters: Mapping[str, Any] | None = None,
    synthetic_movies: Sequence[SyntheticMovie] | None = None,
) -> dict[str, Any]:
    """Run identical simulated images through MATLAB and the Python rewrite."""

    suite = {
        item.name: item
        for item in (
            default_synthetic_suite(config.seed)
            if synthetic_movies is None
            else synthetic_movies
        )
    }
    snapshots = _select_snapshots(suite, config.scenario_frames)
    trials = _build_trials(config)
    baseline = config.baseline
    legacy_parameters = dict(legacy_parameters or {})

    filter_keys = sorted(
        {
            (trial.scenario, trial.frame, trial.sigma_factor)
            for trial in trials
        },
        key=lambda item: (item[0], item[1], item[2]),
    )
    requests: list[dict[str, Any]] = []
    request_metadata: list[tuple[str, Any]] = []
    for scenario, frame, sigma_factor in filter_keys:
        movie = snapshots[(scenario, frame)]
        requests.append(
            separable_dog_request(
                movie.frames_tzyx[frame - 1],
                radius_um=baseline.radius_um,
                sigma_factor=sigma_factor,
                calibration=movie.calibration,
            )
        )
        request_metadata.append(("filter", (scenario, frame, sigma_factor)))
    for trial in trials:
        movie = snapshots[(trial.scenario, trial.frame)]
        requests.append(
            full_detection_request(
                movie.frames_tzyx[trial.frame - 1],
                radius_um=baseline.radius_um,
                sigma_factor=trial.sigma_factor,
                intensity_threshold=trial.intensity_threshold,
                boundary_percent=trial.boundary_percent,
                calibration=movie.calibration,
                num_cells=baseline.cell_count,
                legacy_parameters=legacy_parameters,
            )
        )
        request_metadata.append(("detection", trial.trial_id))

    matlab_runs = oracle.run_many(requests)
    filter_runs = {
        key: run
        for (kind, key), run in zip(request_metadata, matlab_runs, strict=True)
        if kind == "filter"
    }
    detection_runs = {
        key: run
        for (kind, key), run in zip(request_metadata, matlab_runs, strict=True)
        if kind == "detection"
    }

    python_detector = StarryNiteDetector()
    filter_reports: dict[tuple[str, int, float], dict[str, Any]] = {}
    for key, matlab_run in filter_runs.items():
        scenario, frame, sigma_factor = key
        movie = snapshots[(scenario, frame)]
        image = movie.frames_tzyx[frame - 1]
        python_response = legacy_dog_response(
            image,
            baseline.radius_um,
            sigma_factor,
            movie.calibration,
        )
        matlab_response = matlab_run.volume_zyx()
        filter_reports[key] = {
            "similarity": compare_volumes(matlab_response, python_response).to_dict(),
            "matlab_peak": float(np.max(matlab_response)),
            "python_peak": float(np.max(python_response)),
            "matlab_energy": float(np.linalg.norm(matlab_response.ravel())),
            "python_energy": float(np.linalg.norm(python_response.ravel())),
        }

    trial_reports: list[dict[str, Any]] = []
    for trial in trials:
        movie = snapshots[(trial.scenario, trial.frame)]
        image = movie.frames_tzyx[trial.frame - 1]
        detections = python_detector.detect(
            image,
            trial.frame,
            movie.calibration,
            {
                "RADIUS": baseline.radius_um,
                "SIGMA": trial.sigma_factor,
                "THRESHOLD": 0.0,
                "INTENSITY_THRESHOLD": trial.intensity_threshold,
                "BOUNDARY_PERCENT": trial.boundary_percent,
                "DO_SUBPIXEL_LOCALIZATION": False,
            },
        )
        matlab_run = detection_runs[trial.trial_id]
        matlab_zyx = matlab_run.points_zyx0("final_points_zyx_0based")
        matlab_xyz = matlab_zyx[:, ::-1]
        python_xyz = _python_detection_xyz_px(detections, movie.calibration)
        truth_xyz = movie.truth_positions_xyz_px(trial.frame)
        spacing_xyz = (
            movie.calibration.xy_um,
            movie.calibration.xy_um,
            movie.calibration.z_um,
        )
        engine_tolerance_um = min(
            config.matching_tolerance_um,
            0.5 * movie.calibration.xy_um,
        )
        engine_similarity = compare_detections(
            matlab_xyz,
            python_xyz,
            tolerance=engine_tolerance_um,
            spacing=spacing_xyz,
        )
        matlab_truth = compare_detections(
            truth_xyz,
            matlab_xyz,
            tolerance=config.matching_tolerance_um,
            spacing=spacing_xyz,
        )
        python_truth = compare_detections(
            truth_xyz,
            python_xyz,
            tolerance=config.matching_tolerance_um,
            spacing=spacing_xyz,
        )
        matlab_diameters = np.asarray(
            matlab_run.result.get("final_diameters_xy", ()), dtype=float
        ).reshape(-1)
        python_diameters = np.asarray(
            [2.0 * item.radius_um / movie.calibration.xy_um for item in detections]
        )
        diameter_similarity = matched_scalar_errors(
            matlab_diameters,
            python_diameters,
            engine_similarity.matching,
        )
        matlab_quality = np.asarray(
            matlab_run.result.get("final_maxima", ()), dtype=float
        ).reshape(-1)
        python_quality = np.asarray([item.quality for item in detections], dtype=float)
        quality_similarity = matched_scalar_errors(
            matlab_quality,
            python_quality,
            engine_similarity.matching,
        )
        filter_key = (trial.scenario, trial.frame, trial.sigma_factor)
        trial_reports.append(
            {
                "trial_id": trial.trial_id,
                "scenario": trial.scenario,
                "frame": trial.frame,
                "sweep": trial.sweep,
                "parameter_value": trial.parameter_value,
                "effective_settings": {
                    "radius_um": baseline.radius_um,
                    "sigma": trial.sigma_factor,
                    "intensitythreshold": trial.intensity_threshold,
                    "boundary_percent": trial.boundary_percent,
                    "cell_count": baseline.cell_count,
                },
                "engine_matching_tolerance_um": engine_tolerance_um,
                "image_sha256": _array_sha256(image),
                "filter_similarity": filter_reports[filter_key]["similarity"],
                "filter_reference_energy": filter_reports[filter_key]["matlab_energy"],
                "engine_detection_similarity": engine_similarity.to_dict(),
                "matlab_truth_similarity": matlab_truth.to_dict(),
                "python_truth_similarity": python_truth.to_dict(),
                "diameter_similarity": diameter_similarity,
                "quality_similarity": quality_similarity,
                "matlab_points_xyz_px": matlab_xyz.tolist(),
                "python_points_xyz_px": python_xyz.tolist(),
                "matlab_diameters_xy_px": matlab_diameters.tolist(),
                "python_diameters_xy_px": python_diameters.tolist(),
                "matlab_quality": matlab_quality.tolist(),
                "python_quality": python_quality.tolist(),
                "matlab_detection_count": len(matlab_xyz),
                "python_detection_count": len(python_xyz),
                "truth_count": len(truth_xyz),
                "matlab_elapsed_seconds": matlab_run.duration_seconds,
            }
        )

    sensitivity = _sensitivity_reports(config, trial_reports)
    summary = _summarize(trial_reports, sensitivity)
    report = {
        "schema_version": 1,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": asdict(config.baseline),
        "configuration": {
            "seed": config.seed,
            "matching_tolerance_um": config.matching_tolerance_um,
            "scenario_frames": [list(item) for item in config.scenario_frames],
            "sweeps": [asdict(item) for item in config.sweeps],
        },
        "provenance": {
            **oracle.installation_provenance(),
            "matlab_version": matlab_runs[0].matlab_version if matlab_runs else "",
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
        },
        "synthetic_cases": _synthetic_manifest(snapshots.values()),
        "filter_trials": [
            {
                "scenario": scenario,
                "frame": frame,
                "sigma": sigma,
                **values,
            }
            for (scenario, frame, sigma), values in sorted(filter_reports.items())
        ],
        "detection_trials": trial_reports,
        "sensitivity": sensitivity,
        "summary": summary,
        "compatibility_gates": _compatibility_gates(summary),
        "known_gaps": [
            (
                "Ray geometry, contiguous slice claims, and the distance/aspect-ratio "
                "conflict predicates are native; distribution-backed log-odds "
                "merge/split predicates remain a separate compatibility tier."
            ),
            (
                "parameters.rangethreshold is exercised by MATLAB but is not yet "
                "mapped to the native detector."
            ),
            (
                "Tracking classifier/model sensitivity is a separate oracle tier "
                "and is not scored by this detector report."
            ),
        ],
    }
    return report


def write_experiment_outputs(
    output_directory: str | Path,
    report: Mapping[str, Any],
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_run_manifest(output / "parity_report.json", report)
    _write_trial_csv(output / "detection_trials.csv", report["detection_trials"])
    (output / "summary.md").write_text(_render_summary(report), encoding="utf-8")


def _build_trials(config: ParityExperimentConfig) -> tuple[_Trial, ...]:
    baseline = config.baseline
    trials: list[_Trial] = []
    for scenario, frame in config.scenario_frames:
        trials.append(
            _Trial(
                f"{scenario}-t{frame:03d}-baseline",
                scenario,
                frame,
                "baseline",
                0.0,
                baseline.sigma_factor,
                baseline.intensity_threshold,
                baseline.boundary_percent,
            )
        )
        for sweep in config.sweeps:
            for index, value in enumerate(sweep.values):
                sigma = value if sweep.name == "sigma" else baseline.sigma_factor
                threshold = (
                    value
                    if sweep.name == "intensitythreshold"
                    else baseline.intensity_threshold
                )
                boundary = (
                    value if sweep.name == "boundary_percent" else baseline.boundary_percent
                )
                if (
                    math.isclose(float(sigma), baseline.sigma_factor)
                    and math.isclose(float(threshold), baseline.intensity_threshold)
                    and math.isclose(float(boundary), baseline.boundary_percent)
                ):
                    continue
                trials.append(
                    _Trial(
                        f"{scenario}-t{frame:03d}-{sweep.name}-{index:02d}",
                        scenario,
                        frame,
                        sweep.name,
                        float(value),
                        float(sigma),
                        float(threshold),
                        float(boundary),
                    )
                )
    return tuple(trials)


def _select_snapshots(
    suite: Mapping[str, SyntheticMovie],
    selections: Sequence[tuple[str, int]],
) -> dict[tuple[str, int], SyntheticMovie]:
    result = {}
    for name, frame in selections:
        if name not in suite:
            raise KeyError(f"Unknown synthetic scenario: {name}")
        movie = suite[name]
        movie.truth_for_frame(frame)
        result[(name, frame)] = movie
    return result


def _python_detection_xyz_px(
    detections: Sequence[Detection], calibration: Calibration
) -> np.ndarray:
    if not detections:
        return np.empty((0, 3), dtype=float)
    return np.asarray(
        [
            (
                item.x_um / calibration.xy_um,
                item.y_um / calibration.xy_um,
                item.z_um / calibration.z_um,
            )
            for item in detections
        ],
        dtype=float,
    )


def _sensitivity_reports(
    config: ParityExperimentConfig,
    trials: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for scenario, frame in config.scenario_frames:
        baseline_trial = next(
            item
            for item in trials
            if item["scenario"] == scenario
            and item["frame"] == frame
            and item["sweep"] == "baseline"
        )
        for sweep in config.sweeps:
            baseline_value = {
                "sigma": config.baseline.sigma_factor,
                "intensitythreshold": config.baseline.intensity_threshold,
                "boundary_percent": config.baseline.boundary_percent,
            }[sweep.name]
            selected = [
                (float(item["parameter_value"]), item)
                for item in trials
                if item["scenario"] == scenario
                and item["frame"] == frame
                and item["sweep"] == sweep.name
            ]
            selected.append((float(baseline_value), baseline_trial))
            selected.sort(key=lambda item: item[0])
            values = [value for value, _item in selected]
            matlab_counts = [
                item["matlab_detection_count"] for _value, item in selected
            ]
            python_counts = [
                item["python_detection_count"] for _value, item in selected
            ]
            matlab_truth = [
                item["matlab_truth_similarity"]["f1"] for _value, item in selected
            ]
            python_truth = [
                item["python_truth_similarity"]["f1"] for _value, item in selected
            ]
            reports.append(
                {
                    "scenario": scenario,
                    "frame": frame,
                    "parameter": sweep.name,
                    "detection_count": compare_sensitivity_curves(
                        values, matlab_counts, python_counts
                    ).to_dict(),
                    "truth_f1": compare_sensitivity_curves(
                        values, matlab_truth, python_truth
                    ).to_dict(),
                }
            )
    return reports


def _summarize(
    trials: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    relative_l2 = [item["filter_similarity"]["relative_l2"] for item in trials]
    filter_correlations = [
        item["filter_similarity"]["pearson_correlation"]
        for item in trials
        if item["filter_similarity"]["pearson_correlation"] is not None
    ]
    filter_peak_displacements = [
        item["filter_similarity"]["peak_displacement_px"]
        for item in trials
        if float(item.get("filter_reference_energy", 1.0)) > 0.01
    ]
    filter_maximum_absolute_errors = [
        item["filter_similarity"]["max_absolute_error"] for item in trials
    ]
    engine_f1 = [item["engine_detection_similarity"]["f1"] for item in trials]
    matlab_truth = [item["matlab_truth_similarity"]["f1"] for item in trials]
    python_truth = [item["python_truth_similarity"]["f1"] for item in trials]
    slope = [item["detection_count"]["slope_sign_agreement"] for item in sensitivity]
    area = [
        item["detection_count"]["normalized_area_between_curves"]
        for item in sensitivity
    ]
    exact_counts = [
        item["matlab_detection_count"] == item["python_detection_count"]
        for item in trials
    ]
    exact_detections = [
        math.isclose(item["engine_detection_similarity"]["f1"], 1.0)
        for item in trials
    ]
    matched_count = sum(
        int(item["engine_detection_similarity"]["matched_count"])
        for item in trials
    )
    possible_matches = sum(
        max(
            int(item["engine_detection_similarity"]["reference_count"]),
            int(item["engine_detection_similarity"]["candidate_count"]),
        )
        for item in trials
    )
    centroid_distances = [
        float(match["distance"])
        for item in trials
        for match in item["engine_detection_similarity"]["matching"]["matches"]
    ]
    diameter_errors = [
        float(error)
        for item in trials
        for error in item["diameter_similarity"]["signed_errors"]
    ]
    diameter_references = [
        float(value)
        for item in trials
        for value in item["diameter_similarity"]["reference_values"]
    ]
    diameter_relative = [
        abs(error) / max(abs(reference), np.finfo(float).eps)
        for error, reference in zip(
            diameter_errors,
            diameter_references,
            strict=True,
        )
    ]
    quality_errors = [
        float(error)
        for item in trials
        for error in item["quality_similarity"]["signed_errors"]
    ]
    quality_references = [
        float(value)
        for item in trials
        for value in item["quality_similarity"]["reference_values"]
    ]
    quality_relative = [
        abs(error) / max(abs(reference), np.finfo(float).eps)
        for error, reference in zip(
            quality_errors,
            quality_references,
            strict=True,
        )
    ]
    return {
        "trial_count": len(trials),
        "max_filter_relative_l2": max(relative_l2, default=None),
        "median_filter_relative_l2": _median(relative_l2),
        "minimum_filter_correlation": min(filter_correlations, default=None),
        "maximum_filter_peak_displacement_px": max(
            filter_peak_displacements, default=None
        ),
        "maximum_filter_absolute_error": max(
            filter_maximum_absolute_errors, default=None
        ),
        "mean_engine_detection_f1": _mean(engine_f1),
        "minimum_engine_detection_f1": min(engine_f1, default=None),
        "mean_matlab_truth_f1": _mean(matlab_truth),
        "mean_python_truth_f1": _mean(python_truth),
        "mean_detection_count_slope_sign_agreement": _mean(slope),
        "mean_detection_count_area_between_curves": _mean(area),
        "exact_count_trial_fraction": _mean(exact_counts),
        "exact_detection_trial_fraction": _mean(exact_detections),
        "matched_detection_coverage": (
            1.0 if possible_matches == 0 else matched_count / possible_matches
        ),
        "centroid_p95_um": _quantile(centroid_distances, 0.95),
        "diameter_mae_px": _mean([abs(item) for item in diameter_errors]),
        "diameter_signed_bias_px": _mean(diameter_errors),
        "diameter_median_relative_error": _median(diameter_relative),
        "diameter_p95_relative_error": _quantile(diameter_relative, 0.95),
        "quality_signed_bias": _mean(quality_errors),
        "quality_p95_relative_error": _quantile(quality_relative, 0.95),
    }


def _compatibility_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    filter_error = summary["max_filter_relative_l2"]
    filter_correlation = summary["minimum_filter_correlation"]
    filter_peak_displacement = summary["maximum_filter_peak_displacement_px"]
    filter_absolute_error = summary["maximum_filter_absolute_error"]
    mean_detection_f1 = summary["mean_engine_detection_f1"]
    minimum_detection_f1 = summary["minimum_engine_detection_f1"]
    slope_agreement = summary["mean_detection_count_slope_sign_agreement"]
    curve_area = summary["mean_detection_count_area_between_curves"]
    matched_coverage = summary["matched_detection_coverage"]
    centroid_p95 = summary["centroid_p95_um"]
    diameter_median = summary["diameter_median_relative_error"]
    diameter_p95 = summary["diameter_p95_relative_error"]
    diameter_bias = summary["diameter_signed_bias_px"]
    quality_p95 = summary["quality_p95_relative_error"]
    filter_pass = (
        filter_error is not None
        and filter_correlation is not None
        and filter_peak_displacement is not None
        and filter_absolute_error is not None
        and filter_absolute_error <= 1e-4
        and (
            filter_error <= 1.1e-5
            or filter_correlation >= 0.999999999
        )
        and filter_peak_displacement == 0
    )
    detection_pass = (
        mean_detection_f1 is not None
        and minimum_detection_f1 is not None
        and mean_detection_f1 >= 0.95
        and minimum_detection_f1 >= 0.8
    )
    sensitivity_available = slope_agreement is not None and curve_area is not None
    sensitivity_pass = (
        sensitivity_available
        and slope_agreement >= 0.9
        and curve_area <= 0.1
    )
    geometry_pass = (
        matched_coverage is not None
        and centroid_p95 is not None
        and diameter_median is not None
        and diameter_p95 is not None
        and diameter_bias is not None
        and matched_coverage >= 0.99
        and centroid_p95 <= 0.25
        and diameter_median <= 0.05
        and diameter_p95 <= 0.1
        and abs(diameter_bias) <= 0.25
    )
    quality_pass = (
        matched_coverage is not None
        and quality_p95 is not None
        and matched_coverage >= 0.99
        and quality_p95 <= 1e-5
    )
    return {
        "separable_dog": {
            "status": "pass" if filter_pass else "needs_work",
            "metric": "relative_l2_or_correlation_with_absolute_bound",
            "value": filter_error,
            "maximum": 1.1e-5,
            "minimum_correlation": 0.999999999,
            "correlation": filter_correlation,
            "maximum_absolute_error": 1e-4,
            "absolute_error": filter_absolute_error,
            "maximum_peak_displacement_px": 0.0,
            "peak_displacement_px": filter_peak_displacement,
        },
        "final_detections": {
            "status": "pass" if detection_pass else "needs_work",
            "mean_f1": mean_detection_f1,
            "minimum_f1": minimum_detection_f1,
            "required_mean_f1": 0.95,
            "required_minimum_f1": 0.8,
        },
        "parameter_sensitivity": {
            "status": (
                "pass"
                if sensitivity_pass
                else "diagnostic"
                if not sensitivity_available
                else "needs_work"
            ),
            "slope_sign_agreement": slope_agreement,
            "area_between_curves": curve_area,
            "required_slope_sign_agreement": 0.9,
            "maximum_area_between_curves": 0.1,
        },
        "candidate_geometry": {
            "status": "pass" if geometry_pass else "needs_work",
            "matched_coverage": matched_coverage,
            "centroid_p95_um": centroid_p95,
            "diameter_median_relative_error": diameter_median,
            "diameter_p95_relative_error": diameter_p95,
            "diameter_signed_bias_px": diameter_bias,
        },
        "final_quality": {
            "status": "pass" if quality_pass else "needs_work",
            "matched_coverage": matched_coverage,
            "p95_relative_error": quality_p95,
            "maximum_p95_relative_error": 1e-5,
        },
        "tracking_classifier": {
            "status": "diagnostic",
            "reason": (
                "The unchanged MATLAB 2019 model has a live smoke gate; native "
                "classifier-score parity is not yet claimed."
            ),
        },
    }


def _legacy_detector_parameters(
    settings: Mapping[str, ParameterValue],
) -> dict[str, np.ndarray | float]:
    result: dict[str, np.ndarray | float] = {}
    for name, value in settings.items():
        if not name.startswith("parameters."):
            continue
        short_name = name.removeprefix("parameters.")
        if isinstance(value, bool) or isinstance(value, str):
            continue
        if isinstance(value, tuple):
            if any(isinstance(item, (tuple, str, bool)) for item in value):
                continue
            result[short_name] = np.asarray(value, dtype=float)
        else:
            result[short_name] = float(value)
    return result


def _synthetic_manifest(movies: Iterable[SyntheticMovie]) -> list[dict[str, Any]]:
    unique: dict[str, SyntheticMovie] = {movie.name: movie for movie in movies}
    return [
        {
            "name": movie.name,
            "description": movie.description,
            "shape_tzyx": list(movie.frames_tzyx.shape),
            "image_sha256": _array_sha256(movie.frames_tzyx),
            "calibration": movie.calibration.to_dict(),
            "object_count": len(movie.objects),
        }
        for movie in sorted(unique.values(), key=lambda item: item.name)
    ]


def _write_trial_csv(path: Path, trials: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "trial_id",
        "scenario",
        "frame",
        "sweep",
        "parameter_value",
        "matlab_detection_count",
        "python_detection_count",
        "truth_count",
        "engine_matching_tolerance_um",
        "matched_count",
        "engine_f1",
        "matlab_truth_f1",
        "python_truth_f1",
        "centroid_rmse_um",
        "diameter_mae_px",
        "diameter_signed_bias_px",
        "diameter_p95_relative_error",
        "diameter_reference_coverage",
        "quality_p95_relative_error",
        "filter_relative_l2",
        "filter_max_absolute_error",
        "filter_correlation",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in trials:
            engine = item["engine_detection_similarity"]
            filtered = item["filter_similarity"]
            diameter = item["diameter_similarity"]
            quality = item["quality_similarity"]
            writer.writerow(
                {
                    "trial_id": item["trial_id"],
                    "scenario": item["scenario"],
                    "frame": item["frame"],
                    "sweep": item["sweep"],
                    "parameter_value": item["parameter_value"],
                    "matlab_detection_count": item["matlab_detection_count"],
                    "python_detection_count": item["python_detection_count"],
                    "truth_count": item["truth_count"],
                    "engine_matching_tolerance_um": item[
                        "engine_matching_tolerance_um"
                    ],
                    "matched_count": engine["matched_count"],
                    "engine_f1": engine["f1"],
                    "matlab_truth_f1": item["matlab_truth_similarity"]["f1"],
                    "python_truth_f1": item["python_truth_similarity"]["f1"],
                    "centroid_rmse_um": engine["centroid_rmse"],
                    "diameter_mae_px": diameter["mae"],
                    "diameter_signed_bias_px": diameter["signed_bias"],
                    "diameter_p95_relative_error": diameter[
                        "p95_relative_error"
                    ],
                    "diameter_reference_coverage": diameter[
                        "reference_coverage"
                    ],
                    "quality_p95_relative_error": quality[
                        "p95_relative_error"
                    ],
                    "filter_relative_l2": filtered["relative_l2"],
                    "filter_max_absolute_error": filtered[
                        "max_absolute_error"
                    ],
                    "filter_correlation": filtered["pearson_correlation"],
                }
            )


def _render_summary(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    gates = report.get("compatibility_gates", {})
    tracking_gate = (
        gates.get("tracking_classifier", {})
        if isinstance(gates, Mapping)
        else {}
    )
    tracking_status = (
        tracking_gate.get("status", "not scored")
        if isinstance(tracking_gate, Mapping)
        else "not scored"
    )
    lines = [
        "# StarryNite MATLAB/Python detector parity report",
        "",
        f"Detector suite status: **{report['status']}**",
        "",
        f"Tracking classifier status: **{tracking_status}** (reported separately; "
        "this matrix does not claim end-to-end tracking parity).",
        "",
        "| Measure | Result |",
        "|---|---:|",
    ]
    for label, key in (
        ("Detection trials", "trial_count"),
        ("Maximum filter relative L2 error", "max_filter_relative_l2"),
        ("Maximum filter absolute error", "maximum_filter_absolute_error"),
        ("Minimum filter correlation", "minimum_filter_correlation"),
        ("Mean engine detection F1", "mean_engine_detection_f1"),
        ("Minimum engine detection F1", "minimum_engine_detection_f1"),
        ("Exact-count trial fraction", "exact_count_trial_fraction"),
        ("Matched detection coverage", "matched_detection_coverage"),
        ("Centroid p95 (um)", "centroid_p95_um"),
        ("Diameter MAE (px)", "diameter_mae_px"),
        (
            "Diameter median relative error",
            "diameter_median_relative_error",
        ),
        ("Final quality p95 relative error", "quality_p95_relative_error"),
        ("Mean MATLAB-vs-truth F1", "mean_matlab_truth_f1"),
        ("Mean Python-vs-truth F1", "mean_python_truth_f1"),
        (
            "Mean count-curve slope agreement",
            "mean_detection_count_slope_sign_agreement",
        ),
    ):
        value = summary.get(key)
        rendered = (
            "n/a"
            if value is None
            else f"{value:.6g}"
            if isinstance(value, float)
            else str(value)
        )
        lines.append(f"| {label} | {rendered} |")
    gates = report.get("compatibility_gates", {})
    if gates:
        lines.extend(("", "## Compatibility gates", "", "| Gate | Status |", "|---|---|"))
        for name, gate in gates.items():
            lines.append(f"| {name} | {gate['status']} |")
    if (
        summary.get("max_filter_relative_l2", 0.0) > 1.1e-5
        and gates.get("separable_dog", {}).get("status") == "pass"
    ):
        lines.extend(
            (
                "",
                (
                    "Relative L2 is ill-conditioned for near-zero DoG controls. "
                    "The filter gate therefore also requires absolute error at most "
                    "1e-4, correlation at least 0.999999999 when defined, and zero "
                    "peak displacement on non-negligible responses."
                ),
            )
        )
    lines.extend(
        (
            "",
            (
                "The JSON report contains every matched point, per-engine truth "
                "score, parameter response curve, provenance hash, and known "
                "compatibility gap."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _unique_sorted(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(sorted({float(value) for value in values}))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _quantile(values: Sequence[float], probability: float) -> float | None:
    return None if not values else float(np.quantile(values, probability))


__all__ = [
    "OracleBaseline",
    "ParameterSweep",
    "ParityExperimentConfig",
    "baseline_from_parameter_file",
    "default_experiment_config",
    "resolution_experiment_config",
    "run_parity_experiment",
    "write_experiment_outputs",
]
