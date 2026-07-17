"""Numerical, detection, sensitivity, and lineage parity metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class VolumeSimilarity:
    rmse: float
    relative_l2: float
    nrmse_dynamic_range: float | None
    max_absolute_error: float
    p99_absolute_error: float
    pearson_correlation: float | None
    cosine_similarity: float | None
    reference_peak_zyx: tuple[int, ...]
    candidate_peak_zyx: tuple[int, ...]
    peak_displacement_px: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PointMatch:
    reference_index: int
    candidate_index: int
    distance: float


@dataclass(frozen=True, slots=True)
class PointMatching:
    matches: tuple[PointMatch, ...]
    unmatched_reference: tuple[int, ...]
    unmatched_candidate: tuple[int, ...]
    tolerance: float

    @property
    def matched_count(self) -> int:
        return len(self.matches)

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": [asdict(item) for item in self.matches],
            "unmatched_reference": list(self.unmatched_reference),
            "unmatched_candidate": list(self.unmatched_candidate),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class DetectionSimilarity:
    reference_count: int
    candidate_count: int
    matched_count: int
    count_delta: int
    precision: float
    recall: float
    f1: float
    centroid_rmse: float | None
    centroid_p95: float | None
    centroid_bias: tuple[float, ...] | None
    matching: PointMatching

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["matching"] = self.matching.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class SensitivitySimilarity:
    parameter_values: tuple[float, ...]
    reference_curve: tuple[float, ...]
    candidate_curve: tuple[float, ...]
    pearson_correlation: float | None
    spearman_correlation: float | None
    normalized_curve_rmse: float
    normalized_area_between_curves: float
    slope_sign_agreement: float
    transition_distance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LineageSimilarity:
    reference_edge_count: int
    candidate_edge_count: int
    matched_edge_count: int
    precision: float
    recall: float
    f1: float
    reference_division_count: int
    candidate_division_count: int
    matched_division_count: int
    division_f1: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LineageNodeState:
    """One ID-bearing node used for spatially aligned lineage comparison."""

    node_id: Hashable
    frame: int
    position: tuple[float, ...]
    retained: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or int(self.frame) != self.frame:
            raise ValueError("Lineage node frames must be integers")
        coordinates = tuple(float(item) for item in self.position)
        if not coordinates or not all(math.isfinite(item) for item in coordinates):
            raise ValueError("Lineage node positions must contain finite coordinates")
        object.__setattr__(self, "frame", int(self.frame))
        object.__setattr__(self, "position", coordinates)
        object.__setattr__(self, "retained", bool(self.retained))


@dataclass(frozen=True, slots=True)
class LineageGraphSimilarity:
    """Node-, event-, and ancestry-aware comparison of two lineage graphs."""

    reference_node_count: int
    candidate_node_count: int
    matched_node_count: int
    node_precision: float
    node_recall: float
    node_f1: float
    reference_retained_count: int
    candidate_retained_count: int
    matched_retained_count: int
    retained_precision: float
    retained_recall: float
    retained_f1: float
    matched_state_count: int
    state_accuracy: float
    edge_similarity: LineageSimilarity
    edge_by_kind: Mapping[str, Mapping[str, float | int]]
    reference_root_count: int
    candidate_root_count: int
    reference_termination_count: int
    candidate_termination_count: int
    reference_component_count: int
    candidate_component_count: int
    ancestry_pair_count: int
    ancestry_agreement: float
    candidate_to_reference: Mapping[Hashable, Hashable]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "edge_by_kind",
            MappingProxyType(
                {
                    str(kind): MappingProxyType(dict(values))
                    for kind, values in self.edge_by_kind.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "candidate_to_reference",
            MappingProxyType(dict(self.candidate_to_reference)),
        )

    def to_dict(self) -> dict[str, object]:
        result = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in {"edge_similarity", "edge_by_kind", "candidate_to_reference"}
        }
        result["edge_similarity"] = self.edge_similarity.to_dict()
        result["edge_by_kind"] = {
            kind: dict(values) for kind, values in self.edge_by_kind.items()
        }
        result["candidate_to_reference"] = {
            str(key): str(value) for key, value in self.candidate_to_reference.items()
        }
        return result


def compare_volumes(reference: np.ndarray, candidate: np.ndarray) -> VolumeSimilarity:
    """Compare same-shaped numeric volumes without hiding constant-array cases."""

    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"Volume shapes differ: {first.shape} != {second.shape}")
    if first.size == 0:
        raise ValueError("Volumes cannot be empty")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Volumes must contain only finite values")
    difference = second - first
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(difference**2)))
    reference_l2 = float(np.linalg.norm(first.ravel()))
    difference_l2 = float(np.linalg.norm(difference.ravel()))
    relative_l2 = difference_l2 / max(reference_l2, np.finfo(float).eps)
    dynamic_range = float(np.ptp(first))
    nrmse = None if dynamic_range == 0 else rmse / dynamic_range

    first_flat = first.ravel()
    second_flat = second.ravel()
    pearson = _correlation(first_flat, second_flat, method="pearson")
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    if denominator <= np.finfo(float).eps:
        cosine = 1.0 if np.array_equal(first_flat, second_flat) else None
    else:
        cosine = float(np.dot(first_flat, second_flat) / denominator)
    reference_peak = tuple(int(item) for item in np.unravel_index(np.argmax(first), first.shape))
    candidate_peak = tuple(
        int(item) for item in np.unravel_index(np.argmax(second), second.shape)
    )
    peak_displacement = float(
        np.linalg.norm(np.asarray(candidate_peak, dtype=float) - reference_peak)
    )
    return VolumeSimilarity(
        rmse=rmse,
        relative_l2=relative_l2,
        nrmse_dynamic_range=nrmse,
        max_absolute_error=float(np.max(absolute)),
        p99_absolute_error=float(np.quantile(absolute, 0.99)),
        pearson_correlation=pearson,
        cosine_similarity=cosine,
        reference_peak_zyx=reference_peak,
        candidate_peak_zyx=candidate_peak,
        peak_displacement_px=peak_displacement,
    )


def match_point_sets(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    tolerance: float,
    spacing: Sequence[float] | None = None,
) -> PointMatching:
    """Maximum-cardinality, minimum-distance matching under a hard gate.

    Inputs may be XYZ pixels, XYZ microns, or any common coordinate system.
    ``spacing`` scales each coordinate axis before Euclidean distances are
    calculated.  IDs and row order are deliberately ignored.
    """

    if not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    first = _point_array("reference", reference)
    second = _point_array("candidate", candidate, dimensions=first.shape[1])
    dimensions = first.shape[1]
    scale = np.ones(dimensions, dtype=np.float64)
    if spacing is not None:
        scale = np.asarray(tuple(spacing), dtype=np.float64)
        if scale.shape != (dimensions,) or not np.all(np.isfinite(scale)):
            raise ValueError("spacing must contain one finite value per coordinate axis")
        if np.any(scale <= 0):
            raise ValueError("spacing values must be positive")
    reference_count, candidate_count = len(first), len(second)
    if reference_count == 0 or candidate_count == 0:
        return PointMatching(
            (),
            tuple(range(reference_count)),
            tuple(range(candidate_count)),
            float(tolerance),
        )

    distances = np.linalg.norm(
        (first[:, None, :] - second[None, :, :]) * scale,
        axis=2,
    )
    # The augmented assignment lets every real point choose an explicit dummy.
    # A valid match costs less than leaving both endpoints unmatched, which
    # maximizes cardinality before minimizing total distance.
    size = reference_count + candidate_count
    unmatched_cost = float(tolerance) + max(1.0, float(tolerance)) * 1e-6
    invalid_cost = unmatched_cost * (size + 2) * 4.0
    costs = np.full((size, size), invalid_cost, dtype=np.float64)
    costs[:reference_count, :candidate_count] = np.where(
        distances <= tolerance,
        distances,
        invalid_cost,
    )
    for index in range(reference_count):
        costs[index, candidate_count + index] = unmatched_cost
    for index in range(candidate_count):
        costs[reference_count + index, index] = unmatched_cost
    costs[reference_count:, candidate_count:] = 0.0
    rows, columns = linear_sum_assignment(costs)
    matches = tuple(
        PointMatch(int(row), int(column), float(distances[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if row < reference_count
        and column < candidate_count
        and distances[row, column] <= tolerance
    )
    matched_reference = {item.reference_index for item in matches}
    matched_candidate = {item.candidate_index for item in matches}
    return PointMatching(
        matches=tuple(sorted(matches, key=lambda item: item.reference_index)),
        unmatched_reference=tuple(
            index for index in range(reference_count) if index not in matched_reference
        ),
        unmatched_candidate=tuple(
            index for index in range(candidate_count) if index not in matched_candidate
        ),
        tolerance=float(tolerance),
    )


def compare_detections(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    tolerance: float,
    spacing: Sequence[float] | None = None,
) -> DetectionSimilarity:
    """Compare coordinate sets after ID-invariant gated bipartite matching."""

    first = _point_array("reference", reference)
    second = _point_array("candidate", candidate, dimensions=first.shape[1])
    matching = match_point_sets(
        first,
        second,
        tolerance=tolerance,
        spacing=spacing,
    )
    matched = matching.matched_count
    precision, recall, f1 = _prf(matched, len(second), len(first))
    if not matching.matches:
        rmse = None
        p95 = None
        bias = None
    else:
        scale = np.ones(first.shape[1]) if spacing is None else np.asarray(spacing)
        vectors = np.asarray(
            [
                (second[item.candidate_index] - first[item.reference_index]) * scale
                for item in matching.matches
            ],
            dtype=np.float64,
        )
        distances = np.linalg.norm(vectors, axis=1)
        rmse = float(np.sqrt(np.mean(distances**2)))
        p95 = float(np.quantile(distances, 0.95))
        bias = tuple(float(item) for item in np.mean(vectors, axis=0))
    return DetectionSimilarity(
        reference_count=len(first),
        candidate_count=len(second),
        matched_count=matched,
        count_delta=len(second) - len(first),
        precision=precision,
        recall=recall,
        f1=f1,
        centroid_rmse=rmse,
        centroid_p95=p95,
        centroid_bias=bias,
        matching=matching,
    )


def matched_scalar_errors(
    reference_values: Sequence[float],
    candidate_values: Sequence[float],
    matching: PointMatching,
) -> dict[str, object]:
    """Compare radius, intensity, or weight values on already matched points."""

    first = np.asarray(reference_values, dtype=np.float64)
    second = np.asarray(candidate_values, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Scalar values must be one-dimensional")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Scalar values must be finite")
    if matching.matches and (
        max(item.reference_index for item in matching.matches) >= len(first)
        or max(item.candidate_index for item in matching.matches) >= len(second)
    ):
        raise IndexError("Point matching references a missing scalar value")
    if not matching.matches:
        return {
            "matched_count": 0,
            "reference_count": int(len(first)),
            "candidate_count": int(len(second)),
            "reference_coverage": 1.0 if len(first) == 0 else 0.0,
            "candidate_coverage": 1.0 if len(second) == 0 else 0.0,
            "mae": None,
            "rmse": None,
            "signed_bias": None,
            "p95_absolute_error": None,
            "median_relative_error": None,
            "p95_relative_error": None,
            "reference_values": [],
            "candidate_values": [],
            "signed_errors": [],
        }
    reference_matched = np.asarray(
        [first[item.reference_index] for item in matching.matches]
    )
    candidate_matched = np.asarray(
        [second[item.candidate_index] for item in matching.matches]
    )
    difference = candidate_matched - reference_matched
    relative = np.abs(difference) / np.maximum(
        np.abs(reference_matched), np.finfo(float).eps
    )
    return {
        "matched_count": int(len(difference)),
        "reference_count": int(len(first)),
        "candidate_count": int(len(second)),
        "reference_coverage": float(len(difference) / max(1, len(first))),
        "candidate_coverage": float(len(difference) / max(1, len(second))),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "signed_bias": float(np.mean(difference)),
        "p95_absolute_error": float(np.quantile(np.abs(difference), 0.95)),
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
        "reference_values": [float(item) for item in reference_matched],
        "candidate_values": [float(item) for item in candidate_matched],
        "signed_errors": [float(item) for item in difference],
    }


def compare_sensitivity_curves(
    parameter_values: Sequence[float],
    reference_curve: Sequence[float],
    candidate_curve: Sequence[float],
    *,
    dead_band: float = 1e-9,
) -> SensitivitySimilarity:
    """Compare response curves, including slope directions and change points."""

    x = _finite_vector("parameter_values", parameter_values)
    first = _finite_vector("reference_curve", reference_curve)
    second = _finite_vector("candidate_curve", candidate_curve)
    if not (len(x) == len(first) == len(second)) or len(x) < 2:
        raise ValueError("Sensitivity curves need equal lengths of at least two")
    if np.any(np.diff(x) <= 0):
        raise ValueError("parameter_values must be strictly increasing")
    if dead_band < 0 or not math.isfinite(dead_band):
        raise ValueError("dead_band must be finite and non-negative")

    combined_range = float(max(np.ptp(first), np.ptp(second)))
    combined_scale = max(
        combined_range,
        float(np.max(np.abs(np.concatenate((first, second))))),
        np.finfo(float).eps,
    )
    normalized_rmse = float(np.sqrt(np.mean((second - first) ** 2)) / combined_scale)
    x_range = float(x[-1] - x[0])
    absolute_curve_error = np.abs(second - first)
    area_numerator = float(
        np.sum(
            (absolute_curve_error[:-1] + absolute_curve_error[1:])
            * np.diff(x)
            / 2.0
        )
    )
    area = area_numerator / (x_range * combined_scale)
    first_slopes = np.diff(first) / np.diff(x)
    second_slopes = np.diff(second) / np.diff(x)
    first_signs = _dead_band_sign(first_slopes, dead_band)
    second_signs = _dead_band_sign(second_slopes, dead_band)
    slope_agreement = float(np.mean(first_signs == second_signs))
    transition_distance = _transition_distance(
        x,
        first,
        second,
        dead_band=dead_band,
    )
    return SensitivitySimilarity(
        parameter_values=tuple(float(item) for item in x),
        reference_curve=tuple(float(item) for item in first),
        candidate_curve=tuple(float(item) for item in second),
        pearson_correlation=_correlation(first, second, method="pearson"),
        spearman_correlation=_correlation(first, second, method="spearman"),
        normalized_curve_rmse=normalized_rmse,
        normalized_area_between_curves=area,
        slope_sign_agreement=slope_agreement,
        transition_distance=transition_distance,
    )


def compare_lineages(
    reference_edges: Iterable[tuple[Hashable, Hashable, str]],
    candidate_edges: Iterable[tuple[Hashable, Hashable, str]],
    *,
    candidate_to_reference: Mapping[Hashable, Hashable] | None = None,
) -> LineageSimilarity:
    """Compare normalized edges and order-independent two-daughter divisions."""

    reference = {_edge_tuple(item) for item in reference_edges}
    mapping = dict(candidate_to_reference or {})
    candidate = {
        (
            mapping.get(source, source),
            mapping.get(target, target),
            str(kind),
        )
        for source, target, kind in (_edge_tuple(item) for item in candidate_edges)
    }
    matched = len(reference & candidate)
    precision, recall, f1 = _prf(matched, len(candidate), len(reference))
    reference_divisions = _division_events(reference)
    candidate_divisions = _division_events(candidate)
    matched_divisions = len(reference_divisions & candidate_divisions)
    _division_precision, _division_recall, division_f1 = _prf(
        matched_divisions,
        len(candidate_divisions),
        len(reference_divisions),
    )
    return LineageSimilarity(
        reference_edge_count=len(reference),
        candidate_edge_count=len(candidate),
        matched_edge_count=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        reference_division_count=len(reference_divisions),
        candidate_division_count=len(candidate_divisions),
        matched_division_count=matched_divisions,
        division_f1=division_f1,
    )


def compare_lineage_graphs(
    reference_nodes: Iterable[LineageNodeState],
    reference_edges: Iterable[tuple[Hashable, Hashable, str]],
    candidate_nodes: Iterable[LineageNodeState],
    candidate_edges: Iterable[tuple[Hashable, Hashable, str]],
    *,
    tolerance: float,
    spacing: Sequence[float] | None = None,
) -> LineageGraphSimilarity:
    """Compare complete lineage state after frame-local spatial ID alignment.

    MATLAB and Python assign unrelated node identifiers.  Nodes are therefore
    matched independently in each frame under a hard physical gate before any
    topology is scored.  Retained/deleted state is kept separate from raw node
    detection so a tracker cannot hide a false-positive cleanup regression by
    simply omitting an edge.
    """

    first_nodes = _lineage_node_mapping("reference", reference_nodes)
    second_nodes = _lineage_node_mapping("candidate", candidate_nodes)
    dimensions = _lineage_dimensions(first_nodes, second_nodes)
    if spacing is not None and len(tuple(spacing)) != dimensions:
        raise ValueError("spacing must contain one value per lineage coordinate axis")

    mapping: dict[Hashable, Hashable] = {}
    matched_pairs: list[tuple[LineageNodeState, LineageNodeState]] = []
    frames = sorted(
        {item.frame for item in first_nodes.values()}
        | {item.frame for item in second_nodes.values()}
    )
    for frame in frames:
        first = sorted(
            (item for item in first_nodes.values() if item.frame == frame),
            key=lambda item: str(item.node_id),
        )
        second = sorted(
            (item for item in second_nodes.values() if item.frame == frame),
            key=lambda item: str(item.node_id),
        )
        first_positions = np.asarray(
            [item.position for item in first], dtype=float
        ).reshape((-1, dimensions))
        second_positions = np.asarray(
            [item.position for item in second], dtype=float
        ).reshape((-1, dimensions))
        matching = match_point_sets(
            first_positions,
            second_positions,
            tolerance=tolerance,
            spacing=spacing,
        )
        for match in matching.matches:
            reference = first[match.reference_index]
            candidate = second[match.candidate_index]
            mapping[candidate.node_id] = reference.node_id
            matched_pairs.append((reference, candidate))

    node_precision, node_recall, node_f1 = _prf(
        len(matched_pairs), len(second_nodes), len(first_nodes)
    )
    first_retained = {key for key, item in first_nodes.items() if item.retained}
    second_retained = {key for key, item in second_nodes.items() if item.retained}
    matched_retained = sum(
        reference.retained and candidate.retained
        for reference, candidate in matched_pairs
    )
    retained_precision, retained_recall, retained_f1 = _prf(
        matched_retained, len(second_retained), len(first_retained)
    )
    matched_state = sum(
        reference.retained == candidate.retained
        for reference, candidate in matched_pairs
    )
    state_accuracy = (
        matched_state / len(matched_pairs) if matched_pairs else 1.0
    )

    first_edges = _validated_lineage_edges(
        "reference", reference_edges, first_nodes
    )
    second_edges = _validated_lineage_edges(
        "candidate", candidate_edges, second_nodes
    )
    active_first_edges = {
        edge
        for edge in first_edges
        if edge[0] in first_retained and edge[1] in first_retained
    }
    active_second_edges = {
        edge
        for edge in second_edges
        if edge[0] in second_retained and edge[1] in second_retained
    }
    edge_similarity = compare_lineages(
        active_first_edges,
        active_second_edges,
        candidate_to_reference=mapping,
    )
    mapped_second_edges = {
        (mapping.get(source, source), mapping.get(target, target), kind)
        for source, target, kind in active_second_edges
    }
    kinds = sorted(
        {kind for _source, _target, kind in active_first_edges | mapped_second_edges}
        | {"link", "gap", "split"}
    )
    edge_by_kind: dict[str, dict[str, float | int]] = {}
    for kind in kinds:
        reference_kind = {edge for edge in active_first_edges if edge[2] == kind}
        candidate_kind = {edge for edge in mapped_second_edges if edge[2] == kind}
        matched = len(reference_kind & candidate_kind)
        precision, recall, f1 = _prf(
            matched, len(candidate_kind), len(reference_kind)
        )
        edge_by_kind[kind] = {
            "reference_count": len(reference_kind),
            "candidate_count": len(candidate_kind),
            "matched_count": matched,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    first_roots, first_terminations = _roots_and_terminations(
        first_retained, active_first_edges
    )
    second_roots, second_terminations = _roots_and_terminations(
        second_retained, active_second_edges
    )
    first_components = _component_count(first_retained, active_first_edges)
    second_components = _component_count(second_retained, active_second_edges)
    ancestry_pairs, ancestry_matches = _ancestry_agreement(
        first_retained,
        active_first_edges,
        second_retained,
        active_second_edges,
        mapping,
    )
    return LineageGraphSimilarity(
        reference_node_count=len(first_nodes),
        candidate_node_count=len(second_nodes),
        matched_node_count=len(matched_pairs),
        node_precision=node_precision,
        node_recall=node_recall,
        node_f1=node_f1,
        reference_retained_count=len(first_retained),
        candidate_retained_count=len(second_retained),
        matched_retained_count=matched_retained,
        retained_precision=retained_precision,
        retained_recall=retained_recall,
        retained_f1=retained_f1,
        matched_state_count=matched_state,
        state_accuracy=float(state_accuracy),
        edge_similarity=edge_similarity,
        edge_by_kind=edge_by_kind,
        reference_root_count=len(first_roots),
        candidate_root_count=len(second_roots),
        reference_termination_count=len(first_terminations),
        candidate_termination_count=len(second_terminations),
        reference_component_count=first_components,
        candidate_component_count=second_components,
        ancestry_pair_count=ancestry_pairs,
        ancestry_agreement=(
            float(ancestry_matches / ancestry_pairs) if ancestry_pairs else 1.0
        ),
        candidate_to_reference=mapping,
    )


def _point_array(name: str, values: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim == 1 and points.size == 0:
        points = np.empty((0, 3 if dimensions is None else dimensions), dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"{name} points must be a two-dimensional array")
    if dimensions is not None and points.shape[1] != dimensions:
        raise ValueError(f"{name} points have {points.shape[1]} rather than {dimensions} axes")
    if dimensions is None and points.shape[1] < 1:
        raise ValueError(f"{name} points must have at least one coordinate axis")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} points must be finite")
    return points


def _finite_vector(name: str, values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    return vector


def _correlation(first: np.ndarray, second: np.ndarray, *, method: str) -> float | None:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.std(first) == 0 or np.std(second) == 0:
        return 1.0 if np.array_equal(first, second) else None
    if method == "pearson":
        value = stats.pearsonr(first, second).statistic
    elif method == "spearman":
        value = stats.spearmanr(first, second).statistic
    else:  # pragma: no cover - internal call contract
        raise ValueError(f"Unknown correlation method: {method}")
    return None if not np.isfinite(value) else float(value)


def _prf(matched: int, candidate_count: int, reference_count: int) -> tuple[float, float, float]:
    if reference_count == 0 and candidate_count == 0:
        return 1.0, 1.0, 1.0
    precision = matched / candidate_count if candidate_count else 0.0
    recall = matched / reference_count if reference_count else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def _dead_band_sign(values: np.ndarray, dead_band: float) -> np.ndarray:
    return np.where(values > dead_band, 1, np.where(values < -dead_band, -1, 0))


def _transition_distance(
    x: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    dead_band: float,
) -> float:
    midpoints = (x[:-1] + x[1:]) / 2.0
    first_points = midpoints[np.abs(np.diff(first)) > dead_band]
    second_points = midpoints[np.abs(np.diff(second)) > dead_band]
    if not len(first_points) and not len(second_points):
        return 0.0
    if not len(first_points) or not len(second_points):
        return 1.0
    scale = float(x[-1] - x[0])
    first_to_second = np.max(
        [np.min(np.abs(second_points - point)) for point in first_points]
    )
    second_to_first = np.max(
        [np.min(np.abs(first_points - point)) for point in second_points]
    )
    return float(max(first_to_second, second_to_first) / scale)


def _edge_tuple(edge: tuple[Hashable, Hashable, str]) -> tuple[Hashable, Hashable, str]:
    if len(edge) != 3:
        raise ValueError("Edges must contain source, target, and kind")
    source, target, kind = edge
    if source == target:
        raise ValueError("An edge cannot connect a node to itself")
    return source, target, str(kind)


def _division_events(
    edges: set[tuple[Hashable, Hashable, str]],
) -> set[tuple[Hashable, frozenset[Hashable]]]:
    children: dict[Hashable, set[Hashable]] = {}
    for source, target, kind in edges:
        if kind in {"split", "division"}:
            children.setdefault(source, set()).add(target)
    return {
        (source, frozenset(targets))
        for source, targets in children.items()
        if len(targets) == 2
    }


def _lineage_node_mapping(
    name: str,
    nodes: Iterable[LineageNodeState],
) -> dict[Hashable, LineageNodeState]:
    result: dict[Hashable, LineageNodeState] = {}
    for item in nodes:
        if not isinstance(item, LineageNodeState):
            raise TypeError(f"{name} nodes must be LineageNodeState values")
        if item.node_id in result:
            raise ValueError(f"{name} lineage node IDs must be unique")
        result[item.node_id] = item
    return result


def _lineage_dimensions(
    first: Mapping[Hashable, LineageNodeState],
    second: Mapping[Hashable, LineageNodeState],
) -> int:
    dimensions = {
        len(item.position) for item in (*first.values(), *second.values())
    }
    if len(dimensions) > 1:
        raise ValueError("All lineage node positions must use the same dimensions")
    return dimensions.pop() if dimensions else 3


def _validated_lineage_edges(
    name: str,
    edges: Iterable[tuple[Hashable, Hashable, str]],
    nodes: Mapping[Hashable, LineageNodeState],
) -> set[tuple[Hashable, Hashable, str]]:
    result = {_edge_tuple(edge) for edge in edges}
    for source, target, _kind in result:
        if source not in nodes or target not in nodes:
            raise ValueError(f"Every {name} lineage edge must reference known nodes")
        if nodes[target].frame <= nodes[source].frame:
            raise ValueError(f"Every {name} lineage edge must move forward in time")
    return result


def _roots_and_terminations(
    nodes: set[Hashable],
    edges: set[tuple[Hashable, Hashable, str]],
) -> tuple[set[Hashable], set[Hashable]]:
    incoming = {target for _source, target, _kind in edges}
    outgoing = {source for source, _target, _kind in edges}
    return nodes - incoming, nodes - outgoing


def _component_count(
    nodes: set[Hashable],
    edges: set[tuple[Hashable, Hashable, str]],
) -> int:
    neighbors = {node: set() for node in nodes}
    for source, target, _kind in edges:
        neighbors[source].add(target)
        neighbors[target].add(source)
    remaining = set(nodes)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            unseen = neighbors[current] & remaining
            remaining.difference_update(unseen)
            stack.extend(unseen)
    return components


def _ancestor_pairs(
    nodes: set[Hashable],
    edges: set[tuple[Hashable, Hashable, str]],
) -> set[tuple[Hashable, Hashable]]:
    children: dict[Hashable, set[Hashable]] = {node: set() for node in nodes}
    for source, target, _kind in edges:
        children[source].add(target)
    result: set[tuple[Hashable, Hashable]] = set()
    for ancestor in nodes:
        stack = list(children[ancestor])
        seen: set[Hashable] = set()
        while stack:
            descendant = stack.pop()
            if descendant in seen:
                continue
            seen.add(descendant)
            result.add((ancestor, descendant))
            stack.extend(children[descendant])
    return result


def _ancestry_agreement(
    first_nodes: set[Hashable],
    first_edges: set[tuple[Hashable, Hashable, str]],
    second_nodes: set[Hashable],
    second_edges: set[tuple[Hashable, Hashable, str]],
    candidate_to_reference: Mapping[Hashable, Hashable],
) -> tuple[int, int]:
    reverse = {
        reference: candidate
        for candidate, reference in candidate_to_reference.items()
        if candidate in second_nodes and reference in first_nodes
    }
    comparable = set(reverse)
    first_ancestry = _ancestor_pairs(first_nodes, first_edges)
    second_ancestry = _ancestor_pairs(second_nodes, second_edges)
    pair_count = 0
    matches = 0
    ordered = sorted(comparable, key=str)
    for first_index, left in enumerate(ordered):
        for right in ordered[first_index + 1 :]:
            for ancestor, descendant in ((left, right), (right, left)):
                pair_count += 1
                expected = (ancestor, descendant) in first_ancestry
                observed = (
                    reverse[ancestor], reverse[descendant]
                ) in second_ancestry
                matches += expected == observed
    return pair_count, matches


__all__ = [
    "DetectionSimilarity",
    "LineageGraphSimilarity",
    "LineageNodeState",
    "LineageSimilarity",
    "PointMatch",
    "PointMatching",
    "SensitivitySimilarity",
    "VolumeSimilarity",
    "compare_detections",
    "compare_lineage_graphs",
    "compare_lineages",
    "compare_sensitivity_curves",
    "compare_volumes",
    "match_point_sets",
    "matched_scalar_errors",
]
