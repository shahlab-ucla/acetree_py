"""End-to-end lineage parity runner using a live MATLAB StarryNite oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ...api import TrackerGraphResult
from ..detector import StarryNiteDetector
from ..tracker import StarryNiteDivisionTracker
from .lineage import (
    LineageSnapshot,
    compare_lineage_snapshots,
    matlab_lineage_snapshot,
    python_lineage_snapshot,
)
from .event_trace import TrackingEventTrace
from .metrics import LineageGraphSimilarity
from .synthetic import SyntheticMovie
from .matlab_backend import (
    MatlabOracleRun,
    MatlabStarryNiteOracle,
    full_tracking_request,
)


@dataclass(frozen=True, slots=True)
class LineageParityCaseResult:
    scenario: str
    similarity: LineageGraphSimilarity
    matlab_snapshot: LineageSnapshot
    python_snapshot: LineageSnapshot
    matlab_provenance: Mapping[str, Any]
    matlab_classifier_diagnostics: Mapping[str, Any]
    matlab_event_trace: TrackingEventTrace | None

    def __post_init__(self) -> None:
        if self.matlab_event_trace is not None and not isinstance(
            self.matlab_event_trace, TrackingEventTrace
        ):
            raise TypeError(
                "matlab_event_trace must be a TrackingEventTrace or None"
            )
        object.__setattr__(
            self,
            "matlab_provenance",
            MappingProxyType(dict(self.matlab_provenance)),
        )
        object.__setattr__(
            self,
            "matlab_classifier_diagnostics",
            MappingProxyType(dict(self.matlab_classifier_diagnostics)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "similarity": self.similarity.to_dict(),
            "matlab_snapshot": self.matlab_snapshot.to_dict(),
            "python_snapshot": self.python_snapshot.to_dict(),
            "matlab_provenance": dict(self.matlab_provenance),
            "matlab_classifier_diagnostics": dict(
                self.matlab_classifier_diagnostics
            ),
            "matlab_event_trace": (
                None
                if self.matlab_event_trace is None
                else self.matlab_event_trace.to_dict()
            ),
        }


def run_lineage_parity_case(
    movie: SyntheticMovie,
    oracle: MatlabStarryNiteOracle,
    *,
    radius_um: float,
    sigma_factor: float,
    intensity_threshold: float,
    boundary_percent: float,
    tracker_settings: Mapping[str, Any] | None = None,
    tracking_overrides: Mapping[str, float] | None = None,
    legacy_parameters: Mapping[str, Any] | None = None,
    distribution_file: str | Path | None = None,
    model_file: str | Path | None = None,
    num_cells: int = 0,
    tolerance_um: float | None = None,
) -> LineageParityCaseResult:
    """Run both engines on one movie and score complete normalized lineages."""

    matlab_run = oracle.full_tracking(
        movie.frames_tzyx,
        radius_um=radius_um,
        sigma_factor=sigma_factor,
        intensity_threshold=intensity_threshold,
        boundary_percent=boundary_percent,
        calibration=movie.calibration,
        num_cells=num_cells,
        legacy_parameters=legacy_parameters,
        tracking_overrides=tracking_overrides,
        distribution_file=distribution_file,
        model_file=model_file,
    )
    return _compare_lineage_run(
        movie,
        matlab_run,
        oracle,
        radius_um=radius_um,
        sigma_factor=sigma_factor,
        intensity_threshold=intensity_threshold,
        boundary_percent=boundary_percent,
        tracker_settings=tracker_settings,
        tolerance_um=tolerance_um,
    )


def run_lineage_parity_suite(
    movies: Sequence[SyntheticMovie],
    oracle: MatlabStarryNiteOracle,
    *,
    radius_um: float,
    sigma_factor: float,
    intensity_threshold: float,
    boundary_percent: float,
    tracker_settings: Mapping[str, Any] | None = None,
    tracking_overrides: Mapping[str, float] | None = None,
    legacy_parameters: Mapping[str, Any] | None = None,
    distribution_file: str | Path | None = None,
    model_file: str | Path | None = None,
    num_cells: int = 0,
    tolerance_um: float | None = None,
) -> tuple[LineageParityCaseResult, ...]:
    """Score several movies with one MATLAB startup.

    MATLAB startup and toolbox initialization dominate small synthetic trials.
    Batching makes a lineage regression matrix practical while keeping every
    movie's calibration and normalized graph independent.
    """

    ordered_movies = tuple(movies)
    if not ordered_movies:
        return ()
    requests = tuple(
        full_tracking_request(
            movie.frames_tzyx,
            radius_um=radius_um,
            sigma_factor=sigma_factor,
            intensity_threshold=intensity_threshold,
            boundary_percent=boundary_percent,
            calibration=movie.calibration,
            num_cells=num_cells,
            legacy_parameters=legacy_parameters,
            tracking_overrides=tracking_overrides,
            distribution_file=distribution_file,
            model_file=model_file,
        )
        for movie in ordered_movies
    )
    matlab_runs = oracle.run_many(requests)
    return tuple(
        _compare_lineage_run(
            movie,
            matlab_run,
            oracle,
            radius_um=radius_um,
            sigma_factor=sigma_factor,
            intensity_threshold=intensity_threshold,
            boundary_percent=boundary_percent,
            tracker_settings=tracker_settings,
            tolerance_um=tolerance_um,
        )
        for movie, matlab_run in zip(ordered_movies, matlab_runs, strict=True)
    )


def _compare_lineage_run(
    movie: SyntheticMovie,
    matlab_run: MatlabOracleRun,
    oracle: MatlabStarryNiteOracle,
    *,
    radius_um: float,
    sigma_factor: float,
    intensity_threshold: float,
    boundary_percent: float,
    tracker_settings: Mapping[str, Any] | None,
    tolerance_um: float | None,
) -> LineageParityCaseResult:
    detector_settings = {
        "RADIUS": float(radius_um),
        "SIGMA": float(sigma_factor),
        "INTENSITY_THRESHOLD": float(intensity_threshold),
        "BOUNDARY_PERCENT": float(boundary_percent),
        "DO_SUBPIXEL_LOCALIZATION": False,
    }
    detector = StarryNiteDetector()
    raw_detections = tuple(
        detection
        for frame, image in enumerate(movie.frames_tzyx, start=1)
        for detection in detector.detect(
            image,
            frame,
            movie.calibration,
            detector_settings,
        )
    )
    tracker = StarryNiteDivisionTracker()
    settings = dict(tracker_settings or {})
    refine_graph = getattr(tracker, "refine_graph", None)
    if callable(refine_graph):
        graph = refine_graph(raw_detections, settings)
        if not isinstance(graph, TrackerGraphResult):
            raise TypeError("StarryNite refine_graph returned an invalid result")
        retained_ids = {item.detection_id for item in graph.detections}
        python_edges = graph.edges
        graph_provenance = graph.provenance
    else:
        retained_ids = {item.detection_id for item in raw_detections}
        python_edges = tracker.track(raw_detections, settings)
        graph_provenance = {"mode": "edge-only"}

    matlab_snapshot = matlab_lineage_snapshot(
        matlab_run,
        movie.calibration,
        provenance=oracle.installation_provenance(),
    )
    python_snapshot = python_lineage_snapshot(
        raw_detections,
        retained_ids,
        python_edges,
        provenance=graph_provenance,
    )
    tolerance = (
        movie.calibration.xy_um / 2.0
        if tolerance_um is None
        else float(tolerance_um)
    )
    similarity = compare_lineage_snapshots(
        matlab_snapshot,
        python_snapshot,
        tolerance_um=tolerance,
    )
    return LineageParityCaseResult(
        scenario=movie.name,
        similarity=similarity,
        matlab_snapshot=matlab_snapshot,
        python_snapshot=python_snapshot,
        matlab_provenance=matlab_run.to_provenance(),
        matlab_classifier_diagnostics=_classifier_diagnostics(matlab_run),
        matlab_event_trace=(
            None
            if "tracking_event_trace" not in matlab_run.result
            else matlab_run.tracking_event_trace()
        ),
    )


def _classifier_diagnostics(run: MatlabOracleRun) -> dict[str, Any]:
    fields = (
        "classifier_computed_classes",
        "classifier_reference_classes",
        "classifier_removed_diagnostics",
        "classifier_simple_fn_correct",
        "classifier_fn_type",
        "classifier_rounds",
    )
    diagnostics: dict[str, Any] = {}
    for name in fields:
        if name not in run.result:
            continue
        value = run.result[name]
        diagnostics[name] = value.tolist() if hasattr(value, "tolist") else value
    return diagnostics


__all__ = [
    "LineageParityCaseResult",
    "run_lineage_parity_case",
    "run_lineage_parity_suite",
]
