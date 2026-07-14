"""A deterministic, dependency-light Simple LAP spot tracker."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .api import Detection, TrackEdge


_DEFAULT_SETTINGS = MappingProxyType(
    {
        "LINKING_MAX_DISTANCE": 15.0,
        "LINKING_FEATURE_PENALTIES": {},
        "ALLOW_GAP_CLOSING": True,
        "GAP_CLOSING_MAX_DISTANCE": 15.0,
        "GAP_CLOSING_FEATURE_PENALTIES": {},
        # Match TrackMate semantics: 2 links t to t+2 (one missed frame).
        "MAX_FRAME_GAP": 2,
        "ALLOW_TRACK_SPLITTING": False,
        "ALLOW_TRACK_MERGING": False,
        "ALTERNATIVE_LINKING_COST_FACTOR": 1.05,
    }
)


def _penalty_map(name: str, value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping of feature names to weights")
    result: dict[str, float] = {}
    for key, weight in value.items():
        try:
            number = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Penalty weight for {key!r} must be numeric") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"Penalty weight for {key!r} must be finite and non-negative")
        result[str(key)] = number
    return result


def _settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(settings) - set(_DEFAULT_SETTINGS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported Simple LAP setting(s): {names}")
    values = dict(_DEFAULT_SETTINGS)
    values.update(settings)

    for key in ("LINKING_MAX_DISTANCE", "GAP_CLOSING_MAX_DISTANCE"):
        try:
            values[key] = float(values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")

    try:
        values["ALTERNATIVE_LINKING_COST_FACTOR"] = float(
            values["ALTERNATIVE_LINKING_COST_FACTOR"]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("ALTERNATIVE_LINKING_COST_FACTOR must be numeric") from exc
    if (
        not math.isfinite(values["ALTERNATIVE_LINKING_COST_FACTOR"])
        or values["ALTERNATIVE_LINKING_COST_FACTOR"] <= 1
    ):
        raise ValueError("ALTERNATIVE_LINKING_COST_FACTOR must be greater than 1")

    gap = values["MAX_FRAME_GAP"]
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 1:
        raise ValueError("MAX_FRAME_GAP must be a positive integer")
    values["MAX_FRAME_GAP"] = int(gap)

    for key in ("ALLOW_GAP_CLOSING", "ALLOW_TRACK_SPLITTING", "ALLOW_TRACK_MERGING"):
        if not isinstance(values[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be boolean")
        values[key] = bool(values[key])
    if values["ALLOW_TRACK_SPLITTING"]:
        raise ValueError("Simple LAP does not support track splitting")
    if values["ALLOW_TRACK_MERGING"]:
        raise ValueError("Simple LAP does not support track merging")

    values["LINKING_FEATURE_PENALTIES"] = _penalty_map(
        "LINKING_FEATURE_PENALTIES", values["LINKING_FEATURE_PENALTIES"]
    )
    values["GAP_CLOSING_FEATURE_PENALTIES"] = _penalty_map(
        "GAP_CLOSING_FEATURE_PENALTIES", values["GAP_CLOSING_FEATURE_PENALTIES"]
    )
    return values


def _distance(first: Detection, second: Detection) -> float:
    return math.dist(first.position_um, second.position_um)


def _feature_value(detection: Detection, key: str) -> float:
    value = detection.feature(key)
    if value is None:
        raise ValueError(
            f"Detection {detection.detection_id!r} does not provide feature {key!r}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature {key!r} must be numeric for LAP penalties") from exc
    if not math.isfinite(number):
        raise ValueError(f"Feature {key!r} must be finite for LAP penalties")
    return number


def _link_cost(
    first: Detection,
    second: Detection,
    max_distance: float,
    penalties: Mapping[str, float],
) -> float:
    distance = _distance(first, second)
    if distance > max_distance:
        return math.inf
    penalty = 1.0
    for key, weight in penalties.items():
        first_value = _feature_value(first, key)
        second_value = _feature_value(second, key)
        denominator = abs(first_value) + abs(second_value)
        if denominator > np.finfo(float).eps:
            penalty += 3.0 * weight * abs(first_value - second_value) / denominator
    return (distance * penalty) ** 2


def _solve_cost_matrix(
    pair_costs: np.ndarray,
    max_distance: float,
    alternative_factor: float,
) -> list[tuple[int, int, float]]:
    """Solve an assignment with explicit birth/death alternatives."""
    n_sources, n_targets = pair_costs.shape
    if n_sources == 0 or n_targets == 0:
        return []

    pair_alternative = alternative_factor * max_distance**2
    unassigned = pair_alternative / 2.0
    finite = pair_costs[np.isfinite(pair_costs)]
    largest = max(pair_alternative, float(finite.max()) if finite.size else 0.0, 1.0)
    blocking = largest * 1.0e12

    size = n_sources + n_targets
    matrix = np.full((size, size), blocking, dtype=float)
    matrix[:n_sources, :n_targets] = np.where(
        np.isfinite(pair_costs), pair_costs, blocking
    )
    for source in range(n_sources):
        matrix[source, n_targets + source] = unassigned
    for target in range(n_targets):
        matrix[n_sources + target, target] = unassigned
    matrix[n_sources:, n_targets:] = 0.0

    row_indices, column_indices = linear_sum_assignment(matrix)
    matches: list[tuple[int, int, float]] = []
    for row, column in zip(row_indices, column_indices, strict=True):
        if row >= n_sources or column >= n_targets:
            continue
        cost = float(pair_costs[row, column])
        if math.isfinite(cost) and cost < pair_alternative:
            matches.append((int(row), int(column), cost))
    return matches


def _assign(
    sources: Sequence[Detection],
    targets: Sequence[Detection],
    max_distance: float,
    penalties: Mapping[str, float],
    alternative_factor: float,
) -> list[tuple[int, int, float]]:
    pair_costs = np.empty((len(sources), len(targets)), dtype=float)
    for row, source in enumerate(sources):
        for column, target in enumerate(targets):
            pair_costs[row, column] = _link_cost(
                source, target, max_distance, penalties
            )
    return _solve_cost_matrix(pair_costs, max_distance, alternative_factor)


class SimpleLAPTracker:
    """TrackMate-inspired one-to-one LAP linking with optional gap closing.

    This intentionally cannot create splits or merges. Division-capable tracking
    belongs in a separate full Sparse-LAP implementation so the simple tracker's
    topology guarantees stay explicit.
    """

    plugin_id = "acetree.simple_lap"
    display_name = "Simple LAP tracker"
    default_settings = _DEFAULT_SETTINGS

    def track(
        self,
        detections: Sequence[Detection],
        settings: Mapping[str, Any],
    ) -> tuple[TrackEdge, ...]:
        values = _settings(settings)
        ordered = sorted(detections, key=lambda item: (item.frame, item.detection_id))
        ids = [item.detection_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("Detection IDs must be unique")
        if not ordered:
            return ()

        by_frame: dict[int, list[Detection]] = {}
        for detection in ordered:
            by_frame.setdefault(detection.frame, []).append(detection)

        edges: list[TrackEdge] = []
        incoming: set[str] = set()
        outgoing: set[str] = set()
        frames = range(min(by_frame), max(by_frame))
        for frame in frames:
            sources = by_frame.get(frame, [])
            targets = by_frame.get(frame + 1, [])
            for source_idx, target_idx, cost in _assign(
                sources,
                targets,
                values["LINKING_MAX_DISTANCE"],
                values["LINKING_FEATURE_PENALTIES"],
                values["ALTERNATIVE_LINKING_COST_FACTOR"],
            ):
                source = sources[source_idx]
                target = targets[target_idx]
                edges.append(TrackEdge(source.detection_id, target.detection_id, cost))
                outgoing.add(source.detection_id)
                incoming.add(target.detection_id)

        if values["ALLOW_GAP_CLOSING"] and values["MAX_FRAME_GAP"] >= 2:
            ends = [item for item in ordered if item.detection_id not in outgoing]
            starts = [item for item in ordered if item.detection_id not in incoming]
            costs = np.full((len(ends), len(starts)), math.inf, dtype=float)
            for row, source in enumerate(ends):
                for column, target in enumerate(starts):
                    frame_delta = target.frame - source.frame
                    if 2 <= frame_delta <= values["MAX_FRAME_GAP"]:
                        costs[row, column] = _link_cost(
                            source,
                            target,
                            values["GAP_CLOSING_MAX_DISTANCE"],
                            values["GAP_CLOSING_FEATURE_PENALTIES"],
                        )
            for source_idx, target_idx, cost in _solve_cost_matrix(
                costs,
                values["GAP_CLOSING_MAX_DISTANCE"],
                values["ALTERNATIVE_LINKING_COST_FACTOR"],
            ):
                source = ends[source_idx]
                target = starts[target_idx]
                frame_delta = target.frame - source.frame
                edges.append(
                    TrackEdge(
                        source.detection_id,
                        target.detection_id,
                        cost,
                        kind="gap",
                        features={
                            "FRAME_DELTA": frame_delta,
                            "MISSED_FRAMES": frame_delta - 1,
                        },
                    )
                )

        edges.sort(
            key=lambda edge: (
                next(item.frame for item in ordered if item.detection_id == edge.source_id),
                edge.source_id,
                edge.target_id,
            )
        )
        return tuple(edges)


LAPTracker = SimpleLAPTracker
