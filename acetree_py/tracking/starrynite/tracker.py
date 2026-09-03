"""Deterministic, division-aware tracking for the native StarryNite pipeline.

This module is an independent Python implementation of the public StarryNite
workflow concepts: conservative easy links, a bounded candidate list,
tentative bifurcations, and gap closure.  It does not translate the GPL MATLAB
source.  The optimizer below remains an AceTree-Py implementation.  Exported
legacy models are evaluated only through the separate exact-feature
classifier/lineage boundary; this native geometry tracker never feeds them an
approximate substitute feature vector.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..api import Detection, TrackEdge
from ..lap import _solve_cost_matrix


_DEFAULT_SETTINGS = MappingProxyType(
    {
        "LINKING_MAX_DISTANCE": 15.0,
        "LINKING_FEATURE_PENALTIES": {},
        "ALLOW_GAP_CLOSING": True,
        "GAP_CLOSING_MAX_DISTANCE": 15.0,
        "GAP_CLOSING_FEATURE_PENALTIES": {},
        "MAX_FRAME_GAP": 2,
        "ALLOW_TRACK_SPLITTING": True,
        "ALLOW_TRACK_MERGING": False,
        "ALTERNATIVE_LINKING_COST_FACTOR": 1.05,
        # Public StarryNite tracking controls.
        "CANDIDATE_CUTOFF": 1.2,
        "NN_NUMBER": 2,
        "FORWARD_NN_NUMBER": 4,
        "SAFE_FACTOR": 2.0,
        # Native bifurcation scorer.  Values are deliberately interpretable so
        # sparse forward tracking can expose them in a tuning UI.
        "DIVISION_COST_THRESHOLD": 0.85,
        "DIVISION_MAX_DAUGHTER_DISTANCE": 15.0,
        "DIVISION_MAX_DAUGHTER_SEPARATION": 12.0,
        "DIVISION_MAX_MIDPOINT_ERROR": 6.0,
        "DIVISION_MIN_QUALITY_RATIO": 0.25,
        "DIVISION_MIN_VOLUME_RATIO": 0.15,
        "DIVISION_MAX_VOLUME_RATIO": 2.5,
        "DIVISION_REQUIRE_EXCESS_TARGET": True,
        "MAX_ACTIVE_BRANCHES": 8,
        "RADIUS_PENALTY_WEIGHT": 0.25,
        "QUALITY_PENALTY_WEIGHT": 0.0,
        # The paths are retained in settings/provenance. Opaque MATLAB
        # classifier objects are never executed in this process; the neutral
        # classifier and exact legacy feature bridge remain intentionally
        # separate from this faster, explicitly native geometry scorer.
        "STARRYNITE_PARAMETER_FILE": "",
        "STARRYNITE_PARAMETER_SHA256": "",
        "STARRYNITE_MODEL_FILE": "",
        "STARRYNITE_MODEL_SHA256": "",
        # Persist the behavioral boundary in every request/proposal.  Exact
        # refinement is a separate validated backend and is never activated by
        # putting a model path into native settings.
        "STARRYNITE_COMPATIBILITY_MODE": "native_fast",
    }
)


def _finite_number(name: str, value: Any, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _penalty_map(name: str, value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must map feature names to weights")
    penalties: dict[str, float] = {}
    for feature, weight in value.items():
        penalties[str(feature)] = _finite_number(
            f"{name}.{feature}", weight, minimum=0.0
        )
    return penalties


def _settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(settings) - set(_DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(
            "Unsupported StarryNite tracker setting(s): " + ", ".join(sorted(unknown))
        )
    values = dict(_DEFAULT_SETTINGS)
    values.update(settings)

    for key in (
        "LINKING_MAX_DISTANCE",
        "GAP_CLOSING_MAX_DISTANCE",
        "CANDIDATE_CUTOFF",
        "SAFE_FACTOR",
        "DIVISION_COST_THRESHOLD",
        "DIVISION_MAX_DAUGHTER_DISTANCE",
        "DIVISION_MAX_DAUGHTER_SEPARATION",
        "DIVISION_MAX_MIDPOINT_ERROR",
        "DIVISION_MIN_QUALITY_RATIO",
        "DIVISION_MIN_VOLUME_RATIO",
        "DIVISION_MAX_VOLUME_RATIO",
        "RADIUS_PENALTY_WEIGHT",
        "QUALITY_PENALTY_WEIGHT",
    ):
        values[key] = _finite_number(key, values[key], minimum=0.0)
    if values["LINKING_MAX_DISTANCE"] <= 0:
        raise ValueError("LINKING_MAX_DISTANCE must be positive")
    if values["GAP_CLOSING_MAX_DISTANCE"] <= 0:
        raise ValueError("GAP_CLOSING_MAX_DISTANCE must be positive")
    if values["CANDIDATE_CUTOFF"] <= 0:
        raise ValueError("CANDIDATE_CUTOFF must be positive")
    if values["DIVISION_MAX_DAUGHTER_DISTANCE"] <= 0:
        raise ValueError("DIVISION_MAX_DAUGHTER_DISTANCE must be positive")
    if values["DIVISION_MAX_DAUGHTER_SEPARATION"] <= 0:
        raise ValueError("DIVISION_MAX_DAUGHTER_SEPARATION must be positive")
    if values["DIVISION_MAX_MIDPOINT_ERROR"] <= 0:
        raise ValueError("DIVISION_MAX_MIDPOINT_ERROR must be positive")
    if not 0 <= values["DIVISION_MIN_QUALITY_RATIO"] <= 1:
        raise ValueError("DIVISION_MIN_QUALITY_RATIO must be between 0 and 1")
    if values["DIVISION_MIN_VOLUME_RATIO"] > values["DIVISION_MAX_VOLUME_RATIO"]:
        raise ValueError("Division volume-ratio bounds are reversed")

    for key in (
        "NN_NUMBER",
        "FORWARD_NN_NUMBER",
        "MAX_FRAME_GAP",
        "MAX_ACTIVE_BRANCHES",
    ):
        values[key] = _positive_integer(key, values[key])
    if values["MAX_ACTIVE_BRANCHES"] < 2:
        raise ValueError("MAX_ACTIVE_BRANCHES must be at least 2")
    for key in (
        "ALLOW_GAP_CLOSING",
        "ALLOW_TRACK_SPLITTING",
        "ALLOW_TRACK_MERGING",
        "DIVISION_REQUIRE_EXCESS_TARGET",
    ):
        values[key] = _boolean(key, values[key])
    if values["ALLOW_TRACK_MERGING"]:
        raise ValueError("StarryNite native tracking does not support merges")

    values["ALTERNATIVE_LINKING_COST_FACTOR"] = _finite_number(
        "ALTERNATIVE_LINKING_COST_FACTOR",
        values["ALTERNATIVE_LINKING_COST_FACTOR"],
    )
    if values["ALTERNATIVE_LINKING_COST_FACTOR"] <= 1:
        raise ValueError("ALTERNATIVE_LINKING_COST_FACTOR must be greater than 1")
    for key in (
        "STARRYNITE_PARAMETER_FILE",
        "STARRYNITE_PARAMETER_SHA256",
        "STARRYNITE_MODEL_FILE",
        "STARRYNITE_MODEL_SHA256",
    ):
        value = values[key]
        if value is None:
            value = ""
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{key} must be a path string")
        values[key] = str(value)
    mode = values["STARRYNITE_COMPATIBILITY_MODE"]
    if mode != "native_fast":
        raise ValueError(
            "STARRYNITE_COMPATIBILITY_MODE must be 'native_fast' for the native "
            "division tracker; legacy refinement requires its separately "
            "validated backend"
        )
    values["LINKING_FEATURE_PENALTIES"] = _penalty_map(
        "LINKING_FEATURE_PENALTIES", values["LINKING_FEATURE_PENALTIES"]
    )
    values["GAP_CLOSING_FEATURE_PENALTIES"] = _penalty_map(
        "GAP_CLOSING_FEATURE_PENALTIES",
        values["GAP_CLOSING_FEATURE_PENALTIES"],
    )
    return values


def _distance(first: Detection, second: Detection) -> float:
    return math.dist(first.position_um, second.position_um)


def _relative_difference(first: float, second: float) -> float:
    denominator = abs(first) + abs(second)
    if denominator <= np.finfo(float).eps:
        return 0.0
    return 2.0 * abs(first - second) / denominator


def _pair_cost(
    source: Detection,
    target: Detection,
    max_distance: float,
    values: Mapping[str, Any],
    *,
    gap: bool = False,
) -> float:
    distance = _distance(source, target)
    if distance > max_distance:
        return math.inf
    factor = 1.0
    factor += values["RADIUS_PENALTY_WEIGHT"] * _relative_difference(
        source.radius_um, target.radius_um
    )
    factor += values["QUALITY_PENALTY_WEIGHT"] * _relative_difference(
        source.quality, target.quality
    )
    penalty_key = (
        "GAP_CLOSING_FEATURE_PENALTIES" if gap else "LINKING_FEATURE_PENALTIES"
    )
    for feature, weight in values[penalty_key].items():
        first = source.feature(feature)
        second = target.feature(feature)
        if first is None or second is None:
            raise ValueError(
                f"Feature penalty {feature!r} is unavailable on a detection"
            )
        first_number = _finite_number(f"feature {feature!r}", first)
        second_number = _finite_number(f"feature {feature!r}", second)
        factor += weight * _relative_difference(first_number, second_number)
    return (distance * factor) ** 2


def _nearest_neighbor_scale(targets: Sequence[Detection]) -> float | None:
    if len(targets) < 2:
        return None
    positions = np.asarray([target.position_um for target in targets], dtype=float)
    distances, _indices = cKDTree(positions).query(positions, k=2)
    nearest = np.asarray(distances[:, 1], dtype=float)
    # A median is robust to isolated detector artifacts while remaining
    # deterministic and inexpensive for a single frame.
    return float(np.median(nearest))


def _candidate_gate(
    targets: Sequence[Detection],
    absolute_gate: float,
    candidate_cutoff: float,
) -> float:
    scale = _nearest_neighbor_scale(targets)
    if scale is None or scale <= np.finfo(float).eps:
        return absolute_gate
    return min(absolute_gate, candidate_cutoff * scale)


@dataclass(frozen=True, slots=True)
class _DivisionHypothesis:
    source: Detection
    first: Detection
    second: Detection
    cost: float
    midpoint_error_um: float
    separation_um: float
    volume_ratio: float
    quality_ratio: float

    @property
    def sort_key(self) -> tuple[float, str, str, str]:
        daughters = tuple(sorted((self.first.detection_id, self.second.detection_id)))
        return (self.cost, self.source.detection_id, daughters[0], daughters[1])


def _division_hypothesis(
    source: Detection,
    first: Detection,
    second: Detection,
    values: Mapping[str, Any],
) -> _DivisionHypothesis | None:
    if first.frame != source.frame + 1 or second.frame != source.frame + 1:
        return None
    first_distance = _distance(source, first)
    second_distance = _distance(source, second)
    max_distance = values["DIVISION_MAX_DAUGHTER_DISTANCE"]
    if first_distance > max_distance or second_distance > max_distance:
        return None

    separation = _distance(first, second)
    if separation <= np.finfo(float).eps:
        return None
    if separation > values["DIVISION_MAX_DAUGHTER_SEPARATION"]:
        return None
    midpoint = tuple(
        (a + b) / 2.0 for a, b in zip(first.position_um, second.position_um)
    )
    midpoint_error = math.dist(source.position_um, midpoint)
    if midpoint_error > values["DIVISION_MAX_MIDPOINT_ERROR"]:
        return None

    quality_denominator = max(abs(first.quality), abs(second.quality), 1e-12)
    quality_ratio = min(abs(first.quality), abs(second.quality)) / quality_denominator
    if quality_ratio < values["DIVISION_MIN_QUALITY_RATIO"]:
        return None

    parent_volume = max(source.radius_um**3, 1e-12)
    volume_ratio = (first.radius_um**3 + second.radius_um**3) / parent_volume
    if not (
        values["DIVISION_MIN_VOLUME_RATIO"]
        <= volume_ratio
        <= values["DIVISION_MAX_VOLUME_RATIO"]
    ):
        return None

    # Dimensionless, symmetric features.  The score has no hidden learned
    # constants, making it suitable for a small-cell-count tuning workflow.
    motion = (first_distance + second_distance) / (2.0 * max_distance)
    midpoint_term = midpoint_error / values["DIVISION_MAX_MIDPOINT_ERROR"]
    separation_target = max(source.radius_um, 1e-12)
    separation_term = abs(math.log(max(separation / separation_target, 1e-12)))
    balance_term = -math.log(max(quality_ratio, 1e-12))
    volume_term = abs(math.log(max(volume_ratio, 1e-12)))
    cost = (
        0.30 * motion**2
        + 0.35 * midpoint_term**2
        + 0.10 * separation_term**2
        + 0.10 * balance_term**2
        + 0.15 * volume_term**2
    )
    if cost > values["DIVISION_COST_THRESHOLD"]:
        return None
    return _DivisionHypothesis(
        source=source,
        first=first,
        second=second,
        cost=float(cost),
        midpoint_error_um=float(midpoint_error),
        separation_um=float(separation),
        volume_ratio=float(volume_ratio),
        quality_ratio=float(quality_ratio),
    )


def _division_hypotheses(
    sources: Sequence[Detection],
    targets: Sequence[Detection],
    values: Mapping[str, Any],
) -> list[_DivisionHypothesis]:
    if not values["ALLOW_TRACK_SPLITTING"] or len(targets) < 2:
        return []
    if values["DIVISION_REQUIRE_EXCESS_TARGET"] and len(targets) <= len(sources):
        return []

    hypotheses: list[_DivisionHypothesis] = []
    limit = values["FORWARD_NN_NUMBER"]
    gate = values["DIVISION_MAX_DAUGHTER_DISTANCE"]
    target_tree = cKDTree(
        np.asarray([target.position_um for target in targets], dtype=float)
    )
    for source in sources:
        target_indices = target_tree.query_ball_point(source.position_um, r=gate)
        nearby = sorted(
            (targets[index] for index in target_indices),
            key=lambda target: (_distance(source, target), target.detection_id),
        )[:limit]
        for first, second in itertools.combinations(nearby, 2):
            hypothesis = _division_hypothesis(source, first, second, values)
            if hypothesis is not None:
                hypotheses.append(hypothesis)
    hypotheses.sort(key=lambda hypothesis: hypothesis.sort_key)
    return hypotheses


def _select_divisions(
    sources: Sequence[Detection],
    targets: Sequence[Detection],
    values: Mapping[str, Any],
) -> list[_DivisionHypothesis]:
    available_count = (
        max(0, len(targets) - len(sources))
        if values["DIVISION_REQUIRE_EXCESS_TARGET"]
        else len(sources)
    )
    selected: list[_DivisionHypothesis] = []
    used_sources: set[str] = set()
    used_targets: set[str] = set()
    for hypothesis in _division_hypotheses(sources, targets, values):
        if len(selected) >= available_count:
            break
        if hypothesis.source.detection_id in used_sources:
            continue
        daughter_ids = {hypothesis.first.detection_id, hypothesis.second.detection_id}
        if daughter_ids & used_targets:
            continue
        selected.append(hypothesis)
        used_sources.add(hypothesis.source.detection_id)
        used_targets.update(daughter_ids)
    return selected


def _assign_links(
    sources: Sequence[Detection],
    targets: Sequence[Detection],
    max_distance: float,
    values: Mapping[str, Any],
) -> list[tuple[int, int, float]]:
    if not sources or not targets:
        return []
    costs = np.empty((len(sources), len(targets)), dtype=float)
    for row, source in enumerate(sources):
        for column, target in enumerate(targets):
            costs[row, column] = _pair_cost(source, target, max_distance, values)
    return _assign_cost_matrix(costs, sources, targets, max_distance, values)


def _assign_cost_matrix(
    costs: np.ndarray,
    sources: Sequence[Detection],
    targets: Sequence[Detection],
    max_distance: float,
    values: Mapping[str, Any],
) -> list[tuple[int, int, float]]:
    """Apply legacy nearest-neighbor limits, LAP, and the safe-link margin."""

    if not sources or not targets:
        return []
    candidate_costs = np.full(costs.shape, math.inf, dtype=float)
    forward_limit = min(values["FORWARD_NN_NUMBER"], len(targets))
    backward_limit = min(values["NN_NUMBER"], len(sources))
    forward_pairs: set[tuple[int, int]] = set()
    backward_pairs: set[tuple[int, int]] = set()
    for row in range(len(sources)):
        ranked_columns = sorted(
            range(len(targets)),
            key=lambda column: (costs[row, column], targets[column].detection_id),
        )[:forward_limit]
        forward_pairs.update((row, column) for column in ranked_columns)
    for column in range(len(targets)):
        ranked_rows = sorted(
            range(len(sources)),
            key=lambda row: (costs[row, column], sources[row].detection_id),
        )[:backward_limit]
        backward_pairs.update((row, column) for row in ranked_rows)
    for row, column in forward_pairs & backward_pairs:
        candidate_costs[row, column] = costs[row, column]

    assignments = _solve_cost_matrix(
        candidate_costs,
        max_distance,
        values["ALTERNATIVE_LINKING_COST_FACTOR"],
    )
    safe_factor = values["SAFE_FACTOR"]
    if safe_factor <= 1:
        return assignments
    safe_assignments: list[tuple[int, int, float]] = []
    for row, column, cost in assignments:
        alternatives = np.concatenate(
            (
                candidate_costs[row, :column],
                candidate_costs[row, column + 1 :],
                candidate_costs[:row, column],
                candidate_costs[row + 1 :, column],
            )
        )
        finite_alternatives = alternatives[np.isfinite(alternatives)]
        if (
            not finite_alternatives.size
            or cost <= np.finfo(float).eps
            or float(np.min(finite_alternatives)) >= cost * safe_factor**2
        ):
            safe_assignments.append((row, column, cost))
    return safe_assignments


def _division_features(hypothesis: _DivisionHypothesis) -> dict[str, Any]:
    return {
        "STARRYNITE_HYPOTHESIS": "division",
        "DIVISION_SCORE": hypothesis.cost,
        "MIDPOINT_ERROR_UM": hypothesis.midpoint_error_um,
        "DAUGHTER_SEPARATION_UM": hypothesis.separation_um,
        "DAUGHTER_VOLUME_RATIO": hypothesis.volume_ratio,
        "DAUGHTER_QUALITY_RATIO": hypothesis.quality_ratio,
        "SCORER": "acetree.native.starrynite_geometry/v1",
    }


class StarryNiteDivisionTracker:
    """Candidate-limited LAP tracker with deterministic two-daughter splits."""

    plugin_id = "acetree.starrynite_division"
    display_name = "StarryNite native division tracker"
    default_settings = _DEFAULT_SETTINGS

    def track_frontier(
        self,
        sources: Sequence[Detection],
        targets: Sequence[Detection],
        settings: Mapping[str, Any],
    ) -> tuple[TrackEdge, ...]:
        """Solve an explicit sparse lineage frontier, including mixed-age gaps.

        Separating active sources from current-frame targets avoids inferring
        source/target roles from frame membership when one branch is behind
        after a missed observation.
        """

        values = _settings(settings)
        ordered_sources = tuple(
            sorted(sources, key=lambda item: (item.frame, item.detection_id))
        )
        ordered_targets = tuple(
            sorted(targets, key=lambda item: (item.frame, item.detection_id))
        )
        identifiers = [
            item.detection_id for item in (*ordered_sources, *ordered_targets)
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Frontier source and target IDs must be unique")
        if not ordered_sources or not ordered_targets:
            return ()
        if max(item.frame for item in ordered_sources) >= min(
            item.frame for item in ordered_targets
        ):
            raise ValueError("Frontier targets must follow every active source")

        divisions = _select_divisions(ordered_sources, ordered_targets, values)
        divided_sources = {item.source.detection_id for item in divisions}
        daughter_targets = {
            daughter.detection_id
            for item in divisions
            for daughter in (item.first, item.second)
        }
        edges: list[TrackEdge] = []
        for hypothesis in divisions:
            features = _division_features(hypothesis)
            for daughter in sorted(
                (hypothesis.first, hypothesis.second),
                key=lambda item: item.detection_id,
            ):
                edges.append(
                    TrackEdge(
                        hypothesis.source.detection_id,
                        daughter.detection_id,
                        hypothesis.cost,
                        kind="split",
                        features=features,
                    )
                )

        remaining_sources = tuple(
            source
            for source in ordered_sources
            if source.detection_id not in divided_sources
        )
        remaining_targets = tuple(
            target
            for target in ordered_targets
            if target.detection_id not in daughter_targets
        )
        costs = np.full(
            (len(remaining_sources), len(remaining_targets)),
            math.inf,
            dtype=float,
        )
        neighbor_scale = _nearest_neighbor_scale(remaining_targets)
        density_gate = (
            math.inf
            if neighbor_scale is None or neighbor_scale <= np.finfo(float).eps
            else values["CANDIDATE_CUTOFF"] * neighbor_scale
        )
        for row, source in enumerate(remaining_sources):
            for column, target in enumerate(remaining_targets):
                frame_delta = target.frame - source.frame
                if frame_delta == 1:
                    maximum = values["LINKING_MAX_DISTANCE"]
                    gap = False
                elif (
                    values["ALLOW_GAP_CLOSING"]
                    and 2 <= frame_delta <= values["MAX_FRAME_GAP"]
                ):
                    maximum = values["GAP_CLOSING_MAX_DISTANCE"]
                    gap = True
                else:
                    continue
                costs[row, column] = _pair_cost(
                    source,
                    target,
                    min(maximum, density_gate),
                    values,
                    gap=gap,
                )

        assignment_gate = max(
            values["LINKING_MAX_DISTANCE"],
            values["GAP_CLOSING_MAX_DISTANCE"],
        )
        for source_index, target_index, cost in _assign_cost_matrix(
            costs,
            remaining_sources,
            remaining_targets,
            assignment_gate,
            values,
        ):
            source = remaining_sources[source_index]
            target = remaining_targets[target_index]
            frame_delta = target.frame - source.frame
            kind = "gap" if frame_delta > 1 else "link"
            edges.append(
                TrackEdge(
                    source.detection_id,
                    target.detection_id,
                    cost,
                    kind=kind,
                    features={
                        "STARRYNITE_HYPOTHESIS": kind,
                        "FRAME_DELTA": frame_delta,
                        "CANDIDATE_GATE_UM": min(assignment_gate, density_gate),
                    },
                )
            )
        edges.sort(
            key=lambda edge: (edge.source_id, edge.target_id, edge.kind, edge.cost)
        )
        return tuple(edges)

    def track(
        self,
        detections: Sequence[Detection],
        settings: Mapping[str, Any],
    ) -> tuple[TrackEdge, ...]:
        values = _settings(settings)
        ordered = sorted(detections, key=lambda item: (item.frame, item.detection_id))
        identifiers = [item.detection_id for item in ordered]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Detection IDs must be unique")
        if not ordered:
            return ()

        by_frame: dict[int, list[Detection]] = {}
        for detection in ordered:
            by_frame.setdefault(detection.frame, []).append(detection)

        edges: list[TrackEdge] = []
        incoming: set[str] = set()
        outgoing: set[str] = set()
        for frame in range(min(by_frame), max(by_frame)):
            sources = by_frame.get(frame, [])
            targets = by_frame.get(frame + 1, [])
            if not sources or not targets:
                continue

            divisions = _select_divisions(sources, targets, values)
            divided_sources = {item.source.detection_id for item in divisions}
            daughter_targets = {
                daughter.detection_id
                for item in divisions
                for daughter in (item.first, item.second)
            }
            for hypothesis in divisions:
                features = _division_features(hypothesis)
                # A division is one event, but both explicit edges carry the
                # same event score and diagnostics for standalone interchange.
                for daughter in sorted(
                    (hypothesis.first, hypothesis.second),
                    key=lambda item: item.detection_id,
                ):
                    edges.append(
                        TrackEdge(
                            hypothesis.source.detection_id,
                            daughter.detection_id,
                            hypothesis.cost,
                            kind="split",
                            features=features,
                        )
                    )
                    incoming.add(daughter.detection_id)
                outgoing.add(hypothesis.source.detection_id)

            remaining_sources = [
                source for source in sources if source.detection_id not in divided_sources
            ]
            remaining_targets = [
                target for target in targets if target.detection_id not in daughter_targets
            ]
            gate = _candidate_gate(
                remaining_targets,
                values["LINKING_MAX_DISTANCE"],
                values["CANDIDATE_CUTOFF"],
            )
            for source_index, target_index, cost in _assign_links(
                remaining_sources,
                remaining_targets,
                gate,
                values,
            ):
                source = remaining_sources[source_index]
                target = remaining_targets[target_index]
                edges.append(
                    TrackEdge(
                        source.detection_id,
                        target.detection_id,
                        cost,
                        features={
                            "STARRYNITE_HYPOTHESIS": "continuation",
                            "CANDIDATE_GATE_UM": gate,
                        },
                    )
                )
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
                        costs[row, column] = _pair_cost(
                            source,
                            target,
                            values["GAP_CLOSING_MAX_DISTANCE"],
                            values,
                            gap=True,
                        )
            for source_index, target_index, cost in _solve_cost_matrix(
                costs,
                values["GAP_CLOSING_MAX_DISTANCE"],
                values["ALTERNATIVE_LINKING_COST_FACTOR"],
            ):
                source = ends[source_index]
                target = starts[target_index]
                if source.detection_id in outgoing or target.detection_id in incoming:
                    continue
                edges.append(
                    TrackEdge(
                        source.detection_id,
                        target.detection_id,
                        cost,
                        kind="gap",
                        features={
                            "STARRYNITE_HYPOTHESIS": "gap",
                            "FRAME_DELTA": target.frame - source.frame,
                        },
                    )
                )
                outgoing.add(source.detection_id)
                incoming.add(target.detection_id)

        frame_by_id = {item.detection_id: item.frame for item in ordered}
        edges.sort(
            key=lambda edge: (
                frame_by_id[edge.source_id],
                edge.source_id,
                edge.target_id,
                edge.kind,
            )
        )
        return tuple(edges)


StarryNiteTracker = StarryNiteDivisionTracker


__all__ = ["StarryNiteDivisionTracker", "StarryNiteTracker"]
