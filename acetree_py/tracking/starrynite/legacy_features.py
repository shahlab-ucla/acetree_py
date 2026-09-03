"""Exact single-model feature extraction from immutable legacy state.

The implementation in this module is a direct, named translation of the
feature path rooted at ``assembleBifurcationData.m`` in StarryNite.  The odd
sentinels and asymmetries are intentional compatibility behaviour.  In
particular, the first and second forward branches do not use identical units
for every field in the MATLAB source.

Coordinates are the legacy detector coordinates stored by
:class:`LegacyNucleus`.  ``anisotropy_xyz`` converts them to the same metric
used by MATLAB's ``distance_anisotropic``; no TrackMate-style substitute is
accepted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .classifier import SingleModelFeatureInput
from .legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyStateError,
    LegacyTrackingContext,
    legacy_gram_distance,
    legacy_single_dot,
    legacy_single_mean,
    legacy_single_round,
    legacy_single_sum,
)
from .repair_candidates import (
    BackwardRepairCandidates,
    build_false_negative_rewire_plan,
    enumerate_forward_candidates,
    extract_backward_repair_candidates,
)
from .lineage import FalseNegativeRewirePlan


class LegacyFeatureExtractionError(ValueError):
    """Raised when exact MATLAB feature state cannot be reconstructed."""


DAUGHTER_FEATURE_NAMES: tuple[str, ...] = (
    "daughter1_log_total_gfp_ratio_plus_one",
    "daughter1_average_gfp_ratio",
    "daughter2_log_total_gfp_ratio_plus_one",
    "daughter2_average_gfp_ratio",
    "parent_perpendicular_drift_over_spacing_interval",
    "daughter_midpoint_parallel_drift_over_spacing_interval",
    "daughter_xy_separation_over_spacing_interval",
    "daughter_z_separation_over_spacing_interval",
    "daughter1_over_daughter2_total_gfp",
    "daughter1_over_daughter2_average_gfp",
    "parent_xy_principal_variance_ratio",
    "daughter1_over_parent_xy_principal_variance",
    "daughter2_over_parent_xy_principal_variance",
    "daughter1_over_daughter2_xy_principal_variance",
    "mean_daughter_relative_total_gfp",
    "mean_daughter_log_nearest_z",
    "mean_daughter_log_nearest_xy",
    "mean_daughter_relative_aspect_ratio",
    "mean_daughter_relative_log_odds_density",
    "mean_daughter_relative_average_gfp",
    "minimum_daughter_branch_length",
    "mean_spacing_over_mean_diameter",
)

BACKWARD_FEATURE_NAMES: tuple[str, ...] = (
    "backward_gap_offset_over_interval",
    "best_backward_gap_score",
    "best_backward_branch_length_over_interval",
    "best_backward_mean_relative_total_gfp",
    "best_backward_mean_log_nearest_z",
    "best_backward_mean_log_nearest_xy",
    "best_backward_mean_relative_aspect_ratio",
    "best_backward_mean_relative_log_odds_density",
    "best_backward_mean_relative_average_gfp",
    "best_backward_xy_distance_over_spacing",
    "best_backward_z_distance_over_spacing",
)

FORWARD_FEATURE_NAMES: tuple[str, ...] = (
    "forward_gap_offset_legacy_units",
    "best_forward_gap_score",
    "best_forward_target_branch_length_legacy_units",
    "best_forward_mean_relative_total_gfp",
    "best_forward_mean_log_nearest_z",
    "best_forward_mean_log_nearest_xy",
    "best_forward_mean_relative_aspect_ratio",
    "best_forward_mean_relative_log_odds_density",
    "best_forward_mean_relative_average_gfp",
    "best_forward_xy_distance_over_spacing",
    "best_forward_z_distance_over_spacing",
    "best_forward_recursive_solid_length",
    "best_forward_recursive_time_length",
)

if len(DAUGHTER_FEATURE_NAMES) != 22:  # pragma: no cover - import invariant
    raise AssertionError("The legacy daughter block must contain 22 features")
if len(BACKWARD_FEATURE_NAMES) != 11:  # pragma: no cover - import invariant
    raise AssertionError("The legacy backward block must contain 11 features")
if len(FORWARD_FEATURE_NAMES) != 13:  # pragma: no cover - import invariant
    raise AssertionError("The legacy forward block must contain 13 features")


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _number_tuple(
    values: Sequence[Real],
    expected: int,
    label: str,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected:
        raise LegacyFeatureExtractionError(
            f"{label} must contain {expected} values; received {len(result)}"
        )
    if any(not math.isfinite(value) for value in result):
        raise LegacyFeatureExtractionError(f"{label} must be finite")
    return result


def _matrix(
    values: Sequence[Sequence[Real]],
    expected: int,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    result = tuple(tuple(float(item) for item in row) for row in values)
    if len(result) != expected or any(len(row) != expected for row in result):
        raise LegacyFeatureExtractionError(
            f"{label} must have shape ({expected}, {expected})"
        )
    array = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(array)):
        raise LegacyFeatureExtractionError(f"{label} must be finite")
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
        raise LegacyFeatureExtractionError(f"{label} must be symmetric")
    sign, _ = np.linalg.slogdet(array)
    if sign <= 0:
        raise LegacyFeatureExtractionError(f"{label} must be positive definite")
    return result


@dataclass(frozen=True, slots=True)
class LegacyTrackingStatistics:
    """Gaussian tracking statistics stored in ``trackingparameters.model``."""

    division_pair_mean: tuple[float, ...]
    division_pair_covariance: tuple[tuple[float, ...], ...]
    division_triple_mean: tuple[float, ...]
    division_triple_covariance: tuple[tuple[float, ...], ...]
    nondivision_mean: tuple[float, ...]
    nondivision_covariance: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "division_pair_mean",
            _number_tuple(self.division_pair_mean, 2, "division_pair_mean"),
        )
        object.__setattr__(
            self,
            "division_pair_covariance",
            _matrix(
                self.division_pair_covariance,
                2,
                "division_pair_covariance",
            ),
        )
        object.__setattr__(
            self,
            "division_triple_mean",
            _number_tuple(self.division_triple_mean, 10, "division_triple_mean"),
        )
        object.__setattr__(
            self,
            "division_triple_covariance",
            _matrix(
                self.division_triple_covariance,
                10,
                "division_triple_covariance",
            ),
        )
        object.__setattr__(
            self,
            "nondivision_mean",
            _number_tuple(self.nondivision_mean, 4, "nondivision_mean"),
        )
        object.__setattr__(
            self,
            "nondivision_covariance",
            _matrix(
                self.nondivision_covariance,
                4,
                "nondivision_covariance",
            ),
        )

    @classmethod
    def from_model(cls, model: Mapping[str, Any] | Any) -> LegacyTrackingStatistics:
        """Load the six exact arrays from ``trackingparameters.model``.

        Both decoded mapping objects and scipy/MATLAB-style objects with
        attributes are supported.  Missing fields fail closed; the many
        unrelated tracking-statistics fields in the source struct are ignored.
        """

        source: Any = model
        fields = getattr(source, "fields", None)
        if isinstance(fields, Mapping):
            source = fields
        if isinstance(source, Mapping) and "trackingparameters" in source:
            source = source["trackingparameters"]
        if isinstance(source, Mapping) and "model" in source:
            source = source["model"]
        elif not isinstance(source, Mapping) and hasattr(source, "model"):
            source = getattr(source, "model")

        matlab_names = {
            "division_pair_mean": "div_mean",
            "division_pair_covariance": "div_std",
            "division_triple_mean": "div_triple_mean",
            "division_triple_covariance": "div_triple_std",
            "nondivision_mean": "nodiv_mean",
            "nondivision_covariance": "nodiv_std",
        }

        def read(name: str) -> Any:
            source_name = matlab_names[name]
            if isinstance(source, Mapping):
                if source_name not in source:
                    raise LegacyFeatureExtractionError(
                        f"trackingparameters.model is missing {source_name!r}"
                    )
                return source[source_name]
            try:
                return getattr(source, source_name)
            except AttributeError as exc:
                raise LegacyFeatureExtractionError(
                    f"trackingparameters.model is missing {source_name!r}"
                ) from exc

        try:
            pair_mean = np.asarray(read("division_pair_mean"), dtype=float).reshape(-1)
            pair_covariance = np.asarray(
                read("division_pair_covariance"),
                dtype=float,
            )
            triple_mean = np.asarray(
                read("division_triple_mean"),
                dtype=float,
            ).reshape(-1)
            triple_covariance = np.asarray(
                read("division_triple_covariance"),
                dtype=float,
            )
            nondivision_mean = np.asarray(
                read("nondivision_mean"),
                dtype=float,
            ).reshape(-1)
            nondivision_covariance = np.asarray(
                read("nondivision_covariance"),
                dtype=float,
            )
        except LegacyFeatureExtractionError:
            raise
        except (TypeError, ValueError) as exc:
            raise LegacyFeatureExtractionError(
                "trackingparameters.model statistics must be numeric arrays"
            ) from exc
        return cls(
            division_pair_mean=tuple(pair_mean.tolist()),
            division_pair_covariance=tuple(
                tuple(float(item) for item in row)
                for row in pair_covariance.tolist()
            ),
            division_triple_mean=tuple(triple_mean.tolist()),
            division_triple_covariance=tuple(
                tuple(float(item) for item in row)
                for row in triple_covariance.tolist()
            ),
            nondivision_mean=tuple(nondivision_mean.tolist()),
            nondivision_covariance=tuple(
                tuple(float(item) for item in row)
                for row in nondivision_covariance.tolist()
            ),
        )


@dataclass(frozen=True, slots=True)
class LegacyBifurcationExtraction:
    """Named raw blocks and classifier-ready evidence for one split."""

    parent_id: str
    daughter_ids: tuple[str, str]
    feature_input: SingleModelFeatureInput
    daughter_feature_names: tuple[str, ...]
    daughter_features: tuple[float, ...]
    backward_feature_names: tuple[str, ...]
    backward_features: tuple[float, ...]
    forward_feature_names: tuple[str, ...]
    forward_features: tuple[float, ...]
    daughter_lengths: tuple[float, float]
    backward_candidate_present: tuple[bool, bool]
    best_forward_lengths: tuple[float, float]
    nondivision_scores: tuple[float, float]
    repair_result: BackwardRepairCandidates
    false_negative_plan: FalseNegativeRewirePlan | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.parent_id) is not str or not self.parent_id:
            raise TypeError("parent_id must be a non-empty string")
        if len(self.daughter_ids) != 2 or len(set(self.daughter_ids)) != 2:
            raise LegacyFeatureExtractionError(
                "daughter_ids must contain two distinct IDs"
            )
        if not isinstance(self.feature_input, SingleModelFeatureInput):
            raise TypeError("feature_input must be a SingleModelFeatureInput")
        if not isinstance(self.repair_result, BackwardRepairCandidates):
            raise TypeError("repair_result must be BackwardRepairCandidates")
        if self.false_negative_plan is not None and not isinstance(
            self.false_negative_plan,
            FalseNegativeRewirePlan,
        ):
            raise TypeError(
                "false_negative_plan must be a FalseNegativeRewirePlan or None"
            )
        expected = (
            (self.daughter_feature_names, self.daughter_features, 22, "daughter"),
            (self.backward_feature_names, self.backward_features, 11, "backward"),
            (self.forward_feature_names, self.forward_features, 13, "forward"),
        )
        for names, values, size, label in expected:
            if len(names) != size or len(values) != size:
                raise LegacyFeatureExtractionError(
                    f"{label} feature names and values must both have length {size}"
                )
        object.__setattr__(self, "diagnostics", _immutable_mapping(self.diagnostics))

    @property
    def features(self) -> SingleModelFeatureInput:
        """Compatibility alias used by classifier-lineage orchestration."""

        return self.feature_input

    @property
    def named_daughter_features(self) -> Mapping[str, float]:
        return MappingProxyType(dict(zip(self.daughter_feature_names, self.daughter_features)))

    @property
    def named_backward_features(self) -> Mapping[str, float]:
        return MappingProxyType(dict(zip(self.backward_feature_names, self.backward_features)))

    @property
    def named_forward_features(self) -> Mapping[str, float]:
        return MappingProxyType(dict(zip(self.forward_feature_names, self.forward_features)))


@dataclass(frozen=True, slots=True)
class _ForwardChoice:
    daughter_index: int
    endpoint_id: str
    candidate_id: str | None
    score: float
    target_length: float
    confidence: tuple[float, ...]
    gap_offset: float
    xy_distance: float
    z_distance: float
    recursive_solid_length: float
    recursive_time_length: float


def _mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    # MATLAB mean includes NaN by default.
    if any(math.isnan(value) for value in values):
        return math.nan
    result = 0.0
    for value in values:
        result += float(value)
    return result / len(values)


def _column_mean(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not rows:
        return ()
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise LegacyFeatureExtractionError("Feature rows must have equal lengths")
    return tuple(_mean([float(row[index]) for row in rows]) for index in range(width))


def _matlab_divide(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if denominator == 0:
        if numerator == 0:
            return math.nan
        return math.copysign(
            math.inf,
            numerator * math.copysign(1.0, denominator),
        )
    return numerator / denominator


def _matlab_single_divide(numerator: float, denominator: float) -> float:
    return legacy_single_round(_matlab_divide(numerator, denominator))


def _matlab_single_multiply(first: float, second: float) -> float:
    return legacy_single_round(float(first) * float(second))


def _matlab_single_vector(values: Sequence[float]) -> tuple[float, ...]:
    """Apply MATLAB mixed-concatenation's whole-vector ``single`` cast."""

    return tuple(legacy_single_round(value) for value in values)


def _finite_or_zero(value: float) -> float:
    return 0.0 if not math.isfinite(value) else value


def _matlab_distance(
    first: Sequence[float],
    second: Sequence[float],
    *,
    zero_based: bool = True,
) -> float:
    return legacy_gram_distance(first, second, zero_based=zero_based)


def _normalizer(context: LegacyTrackingContext, frame: int) -> float:
    value = context.mean_self_distance(frame)
    if math.isinf(value):
        value = legacy_single_mean(
            [
                context.nucleus(nucleus_id).diameter
                for nucleus_id in context.frame_ids(frame, include_deleted=True)
            ]
        )
    return value


def calculate_legacy_division_pair(
    parent: LegacyNucleus,
    daughter: LegacyNucleus,
) -> tuple[float, float]:
    """Translate ``calculateCellPairVector.m`` exactly."""

    total_ratio = _matlab_divide(daughter.total_gfp, parent.total_gfp)
    try:
        total_feature = math.log(total_ratio + 1.0)
    except ValueError:
        total_feature = math.nan
    average_feature = _matlab_divide(daughter.avg_gfp, parent.avg_gfp)
    return (_finite_or_zero(total_feature), _finite_or_zero(average_feature))


def calculate_legacy_nondivision_pair(
    context: LegacyTrackingContext,
    source_id: str,
    target_id: str,
) -> tuple[float, float, float, float]:
    """Translate ``calculateCellPairVectorNondivision.m`` exactly."""

    source = context.nucleus(source_id)
    target = context.nucleus(target_id)
    normalizer = context.mean_self_distance(source.frame)
    values = (
        _matlab_divide(target.total_gfp, source.total_gfp),
        _matlab_divide(target.avg_gfp, source.avg_gfp),
        _matlab_single_divide(
            _matlab_distance((source.x, source.y), (target.x, target.y)),
            normalizer,
        ),
        _matlab_single_divide(
            _matlab_single_multiply(
                _matlab_distance((source.z,), (target.z,)),
                context.parameters.anisotropy_xyz[2],
            ),
            normalizer,
        ),
    )
    rounded = _matlab_single_vector(values)
    return tuple(  # type: ignore[return-value]
        _finite_or_zero(value) for value in rounded
    )


def calculate_legacy_division_triple(
    context: LegacyTrackingContext,
    parent_id: str,
    daughter1_id: str,
    daughter2_id: str,
) -> tuple[float, ...]:
    """Translate the ten active values in ``calculateCellTripleVector.m``.

    The first perpendicular distance intentionally compares the unscaled
    parent Z coordinate to a projection whose Z coordinate has been scaled.
    That coordinate-system mismatch is present in the MATLAB implementation
    and affects compatibility when Z is not zero.
    """

    parent = context.nucleus(parent_id)
    daughter1 = context.nucleus(daughter1_id)
    daughter2 = context.nucleus(daughter2_id)
    parameters = context.parameters
    z_scale = parameters.anisotropy_xyz[2]

    # Interop/native coordinates are zero based. MATLAB evaluates this block
    # on one-based finalpoints; because the perpendicular-Z calculation mixes
    # scaled and unscaled coordinates, that indexing translation does not
    # cancel and must be restored locally.
    parent_position = tuple(
        legacy_single_round(value + 1.0) for value in parent.position_xyz
    )
    daughter1_position = tuple(
        legacy_single_round(value + 1.0) for value in daughter1.position_xyz
    )
    daughter2_position = tuple(
        legacy_single_round(value + 1.0) for value in daughter2.position_xyz
    )

    midpoint = tuple(
        _matlab_single_divide(legacy_single_round(left + right), 2.0)
        for left, right in zip(
            daughter2_position,
            daughter1_position,
            strict=True,
        )
    )
    midpoint = (
        midpoint[0],
        midpoint[1],
        _matlab_single_multiply(midpoint[2], z_scale),
    )
    division_line = tuple(
        legacy_single_round(left - right)
        for left, right in zip(
            daughter1_position,
            daughter2_position,
            strict=True,
        )
    )
    division_line = (
        division_line[0],
        division_line[1],
        _matlab_single_multiply(division_line[2], z_scale),
    )
    other_line = tuple(
        legacy_single_round(left - right)
        for left, right in zip(
            parent_position,
            daughter2_position,
            strict=True,
        )
    )
    other_line = (
        other_line[0],
        other_line[1],
        _matlab_single_multiply(other_line[2], z_scale),
    )
    daughter2_scaled = (
        daughter2_position[0],
        daughter2_position[1],
        _matlab_single_multiply(daughter2_position[2], z_scale),
    )
    denominator = legacy_single_dot(division_line, division_line)
    projection_scale = _matlab_single_divide(
        legacy_single_dot(other_line, division_line),
        denominator,
    )
    parent_projection = tuple(
        legacy_single_round(
            origin + _matlab_single_multiply(projection_scale, direction)
        )
        for origin, direction in zip(
            daughter2_scaled,
            division_line,
            strict=True,
        )
    )

    perpendicular_drift = _matlab_distance(
        parent_position,
        parent_projection,
        zero_based=False,
    )
    parallel_drift = _matlab_distance(
        midpoint,
        parent_projection,
        zero_based=False,
    )
    xy_squared = legacy_single_round(
        _matlab_single_multiply(division_line[0], division_line[0])
        + _matlab_single_multiply(division_line[1], division_line[1])
    )
    xy_component = legacy_single_round(math.sqrt(xy_squared))
    z_component = legacy_single_round(abs(division_line[2]))
    normalizer = _normalizer(context, parent.frame)

    def normalize_geometry(value: float) -> float:
        return _matlab_single_divide(
            _matlab_single_divide(value, normalizer),
            parameters.interval,
        )

    parent_aspect = _matlab_divide(
        parent.xy_principal_variance,
        parent.xy_secondary_variance,
    )
    result = (
        normalize_geometry(perpendicular_drift),
        normalize_geometry(parallel_drift),
        normalize_geometry(xy_component),
        normalize_geometry(z_component),
        _matlab_divide(daughter1.total_gfp, daughter2.total_gfp),
        _matlab_divide(daughter1.avg_gfp, daughter2.avg_gfp),
        parent_aspect,
        _matlab_divide(
            daughter1.xy_principal_variance,
            parent.xy_principal_variance,
        ),
        _matlab_divide(
            daughter2.xy_principal_variance,
            parent.xy_principal_variance,
        ),
        _matlab_divide(
            daughter1.xy_principal_variance,
            daughter2.xy_principal_variance,
        ),
    )
    return _matlab_single_vector(result)


def _mvn_pdf(
    values: Sequence[float],
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
) -> float:
    vector = np.asarray(values, dtype=float)
    center = np.asarray(mean, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if np.any(np.isnan(vector)):
        return math.nan
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise LegacyFeatureExtractionError("Tracking covariance is not positive definite")
    delta = vector - center
    quadratic = float(delta @ np.linalg.solve(matrix, delta))
    log_pdf = -0.5 * (
        len(vector) * math.log(2.0 * math.pi) + float(logdet) + quadratic
    )
    try:
        return math.exp(log_pdf)
    except OverflowError:  # pragma: no cover - invalid covariance guard above
        return math.inf


def _inverse_pdf(pdf: float) -> float:
    return _matlab_divide(1.0, pdf)


def _negative_log_pdf_score(pdf: float) -> float:
    inverse = _inverse_pdf(pdf)
    if inverse == 0:
        return -math.inf
    if inverse < 0 or math.isnan(inverse):
        return math.nan
    return math.log(inverse)


def calculate_legacy_nondivision_scores(
    context: LegacyTrackingContext,
    parent_id: str,
    statistics: LegacyTrackingStatistics,
) -> tuple[float, float]:
    """Recompute MATLAB's two daughter non-division costs for one split."""

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(statistics, LegacyTrackingStatistics):
        raise TypeError("statistics must be LegacyTrackingStatistics")
    first, second = context.successor_slots(parent_id)
    if first is None or second is None:
        raise LegacyFeatureExtractionError(
            f"Nucleus {parent_id!r} is not a tentative bifurcation"
        )
    vectors = (
        calculate_legacy_nondivision_pair(context, parent_id, first),
        calculate_legacy_nondivision_pair(context, parent_id, second),
    )
    scores = tuple(
        _negative_log_pdf_score(
            _mvn_pdf(
                vector,
                statistics.nondivision_mean,
                statistics.nondivision_covariance,
            )
        )
        for vector in vectors
    )
    return (scores[0], scores[1])


def calculate_legacy_nondivision_cost(
    context: LegacyTrackingContext,
    source_id: str,
    target_id: str,
    statistics: LegacyTrackingStatistics,
) -> float:
    """Return ``nondivScoreModelCostFunction`` for one candidate link.

    This is the same probability-domain ``log(1 / mvnpdf(...))`` path used
    while StarryNite constructs the tentative pre-classifier lineage.  It is
    intentionally separate from :func:`calculate_legacy_nondivision_scores`,
    which requires an already attached two-daughter bifurcation.
    """

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(statistics, LegacyTrackingStatistics):
        raise TypeError("statistics must be LegacyTrackingStatistics")
    vector = calculate_legacy_nondivision_pair(context, source_id, target_id)
    return _negative_log_pdf_score(
        _mvn_pdf(
            vector,
            statistics.nondivision_mean,
            statistics.nondivision_covariance,
        )
    )


def calculate_legacy_division_cost(
    context: LegacyTrackingContext,
    parent_id: str,
    daughter1_id: str,
    daughter2_id: str,
    statistics: LegacyTrackingStatistics,
) -> float:
    """Return ``divScoreModelCostFunction`` for one ordered daughter pair."""

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(statistics, LegacyTrackingStatistics):
        raise TypeError("statistics must be LegacyTrackingStatistics")
    parent = context.nucleus(parent_id)
    daughter1 = context.nucleus(daughter1_id)
    daughter2 = context.nucleus(daughter2_id)
    pair1 = calculate_legacy_division_pair(parent, daughter1)
    pair2 = calculate_legacy_division_pair(parent, daughter2)
    triple = calculate_legacy_division_triple(
        context,
        parent_id,
        daughter1_id,
        daughter2_id,
    )
    components = (
        _inverse_pdf(
            _mvn_pdf(
                triple,
                statistics.division_triple_mean,
                statistics.division_triple_covariance,
            )
        ),
        _inverse_pdf(
            _mvn_pdf(
                pair1,
                statistics.division_pair_mean,
                statistics.division_pair_covariance,
            )
        ),
        _inverse_pdf(
            _mvn_pdf(
                pair2,
                statistics.division_pair_mean,
                statistics.division_pair_covariance,
            )
        ),
    )
    return _matlab_log(components[0] * components[1] * components[2])


def _matlab_log(value: float) -> float:
    """Return MATLAB's real-scalar ``log`` result for legacy score paths."""

    value = float(value)
    if math.isnan(value) or value < 0:
        return math.nan
    if value == 0:
        return -math.inf
    return math.log(value)


def _trace_forward_for_confidence(
    context: LegacyTrackingContext,
    nucleus_id: str,
    count: float,
) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    if count < 1 or math.isnan(count):
        return rows
    current: str | None = nucleus_id
    for _ in range(int(math.floor(count))):
        if current is None:
            break
        rows.append(context.confidence_vector(current))
        current = context.successor_slots(current)[0]
    return rows


def _trace_backward_for_confidence(
    context: LegacyTrackingContext,
    nucleus_id: str,
    count: float,
) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    if count < 1 or math.isnan(count):
        return rows
    current: str | None = nucleus_id
    for _ in range(int(math.floor(count))):
        if current is None:
            break
        rows.append(context.confidence_vector(current))
        current = context.predecessor(current)
    return rows


def _recursive_forward_lengths(
    context: LegacyTrackingContext,
    nucleus_id: str,
) -> tuple[float, float]:
    branch = context.traverse_forward(nucleus_id)
    solid = len(branch) / context.parameters.interval
    first = context.nucleus(branch[0])
    last = context.nucleus(branch[-1])
    temporal = (last.frame - first.frame + 1) / context.parameters.interval
    if solid >= 10:
        return temporal, solid
    candidates = enumerate_forward_candidates(
        context,
        branch[-1],
        context.parameters,
    )
    if candidates.selected is None:
        return temporal, solid
    best_id = candidates.selected.target_id
    child_temporal, child_solid = _recursive_forward_lengths(context, best_id)
    temporal += (
        child_temporal
        + (context.nucleus(best_id).frame - last.frame)
        / context.parameters.interval
    )
    solid += child_solid
    return temporal, solid


def _forward_choice(
    context: LegacyTrackingContext,
    daughter_id: str,
    daughter_index: int,
    normalizer: float,
) -> _ForwardChoice:
    endpoint_id = context.traverse_forward(daughter_id)[-1]
    endpoint = context.nucleus(endpoint_id)
    candidates = enumerate_forward_candidates(
        context,
        endpoint_id,
        context.parameters,
    )
    if candidates.selected is None:
        recursive_default = -1.0 if daughter_index == 0 else 0.0
        return _ForwardChoice(
            daughter_index=daughter_index,
            endpoint_id=endpoint_id,
            candidate_id=None,
            score=-1.0,
            target_length=-1.0,
            confidence=(0.0,) * 6,
            gap_offset=-1.0,
            xy_distance=-1.0,
            z_distance=-1.0,
            recursive_solid_length=recursive_default,
            recursive_time_length=recursive_default,
        )

    selected = candidates.selected
    score = selected.score
    candidate_id = selected.target_id
    candidate = context.nucleus(candidate_id)
    target_length = candidates.selected_feature_length
    predecessor = context.predecessor(candidate_id)
    if predecessor is not None:
        first, second = context.successor_slots(predecessor)
        if first is None or second is None:
            raise LegacyFeatureExtractionError(
                "A forward candidate with a predecessor must originate in a division"
            )
        raw_length = min(
            context.forward_depth(first),
            context.forward_depth(second),
        )
        rows = _trace_forward_for_confidence(context, first, raw_length)
        rows.extend(_trace_forward_for_confidence(context, second, raw_length))
    else:
        rows = _trace_forward_for_confidence(context, candidate_id, target_length)
    confidence = _column_mean(rows)
    if len(confidence) != 6:
        raise LegacyFeatureExtractionError(
            "The selected forward candidate has no confidence samples"
        )
    recursive_time, recursive_solid = _recursive_forward_lengths(
        context,
        candidate_id,
    )
    gap_offset = float(candidate.frame - endpoint.frame)
    if daughter_index == 1:
        gap_offset /= context.parameters.interval
    return _ForwardChoice(
        daughter_index=daughter_index,
        endpoint_id=endpoint_id,
        candidate_id=candidate_id,
        score=score,
        target_length=target_length,
        confidence=confidence,
        gap_offset=gap_offset,
        xy_distance=_matlab_single_divide(
            _matlab_distance(
                (candidate.x, candidate.y),
                (endpoint.x, endpoint.y),
            ),
            normalizer,
        ),
        z_distance=_matlab_single_divide(
            _matlab_single_multiply(
                _matlab_distance((candidate.z,), (endpoint.z,)),
                context.parameters.anisotropy_xyz[2],
            ),
            normalizer,
        ),
        recursive_solid_length=recursive_solid,
        recursive_time_length=recursive_time,
    )


def _forward_block(choice: _ForwardChoice) -> tuple[float, ...]:
    return (
        choice.gap_offset,
        choice.score,
        choice.target_length,
        *choice.confidence,
        choice.xy_distance,
        choice.z_distance,
        choice.recursive_solid_length,
        choice.recursive_time_length,
    )


def _ensure_classifier_values(values: Sequence[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) not in {11, 13, 22}:
        raise LegacyFeatureExtractionError(
            f"{label} has an invalid legacy feature count"
        )
    return result


def extract_legacy_bifurcation_features(
    context: LegacyTrackingContext,
    parent_id: str,
    statistics: LegacyTrackingStatistics,
    *,
    repair_result: BackwardRepairCandidates | None = None,
    record_answers: bool = False,
) -> LegacyBifurcationExtraction:
    """Extract all 22/11/13 raw features for one tentative bifurcation.

    ``repair_result`` can inject a previously frozen candidate snapshot.  If
    omitted, it is extracted exactly once and retained in the result together
    with the atomic class-2 rewire plan selected from it.
    """

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(statistics, LegacyTrackingStatistics):
        raise TypeError("statistics must be LegacyTrackingStatistics")
    if type(record_answers) is not bool:
        raise TypeError("record_answers must be a boolean")
    parent = context.nucleus(parent_id)
    daughter_slots = context.successor_slots(parent_id)
    if daughter_slots[0] is None or daughter_slots[1] is None:
        raise LegacyFeatureExtractionError(
            f"Nucleus {parent_id!r} is not a tentative bifurcation"
        )
    daughter_ids = (daughter_slots[0], daughter_slots[1])
    daughter1 = context.nucleus(daughter_ids[0])
    daughter2 = context.nucleus(daughter_ids[1])
    raw_lengths = (
        float(context.forward_depth(daughter_ids[0])),
        float(context.forward_depth(daughter_ids[1])),
    )
    minimum_raw_length = min(raw_lengths)
    lengths = (
        tuple(
            value / context.parameters.interval for value in raw_lengths
        )
        if record_answers
        else raw_lengths
    )
    minimum_length = min(lengths)
    normalizer = _normalizer(context, parent.frame)

    daughter_confidence_rows = _trace_forward_for_confidence(
        context,
        daughter_ids[0],
        minimum_raw_length,
    )
    daughter_confidence_rows.extend(
        _trace_forward_for_confidence(
            context,
            daughter_ids[1],
            minimum_raw_length,
        )
    )
    daughter_confidence = _column_mean(daughter_confidence_rows)
    if len(daughter_confidence) != 6:
        raise LegacyFeatureExtractionError("Daughter confidence block is incomplete")

    pair1 = calculate_legacy_division_pair(parent, daughter1)
    pair2 = calculate_legacy_division_pair(parent, daughter2)
    triple = calculate_legacy_division_triple(
        context,
        parent_id,
        daughter_ids[0],
        daughter_ids[1],
    )
    density = _matlab_single_divide(
        context.mean_self_distance(parent.frame),
        legacy_single_mean(
            [
                context.nucleus(item).diameter
                for item in context.frame_ids(parent.frame, include_deleted=True)
            ]
        ),
    )
    daughter_block = _ensure_classifier_values(
        _matlab_single_vector(
            (
                *pair1,
                *pair2,
                *triple,
                *daughter_confidence,
                minimum_length,
                density,
            )
        ),
        "daughter feature block",
    )

    if repair_result is not None:
        if not isinstance(repair_result, BackwardRepairCandidates):
            raise TypeError("repair_result must be BackwardRepairCandidates or None")
        backward_candidates = repair_result
    else:
        backward_candidates = extract_backward_repair_candidates(
            context,
            daughter_ids[0],
            daughter_ids[1],
            context.parameters,
        )
    backward_groups = (
        backward_candidates.daughter1,
        backward_candidates.daughter2,
    )
    best_backward = backward_candidates.selected
    false_negative_plan = (
        None
        if best_backward is None
        else build_false_negative_rewire_plan(
            context,
            parent_id,
            daughter_ids[0],
            daughter_ids[1],
            best_backward,
        )
    )
    if best_backward is None:
        backward_block = (-1.0, -1.0, -1.0, *(0.0,) * 6, -1.0, -1.0)
    else:
        daughter = context.nucleus(best_backward.target_id)
        candidate = context.nucleus(best_backward.source_id)
        confidence_rows = _trace_backward_for_confidence(
            context,
            best_backward.source_id,
            best_backward.source_branch_length,
        )
        backward_confidence = _column_mean(confidence_rows)
        if len(backward_confidence) != 6:
            raise LegacyFeatureExtractionError(
                "The selected backward candidate has no confidence samples"
            )
        backward_block = (
            (daughter.frame - candidate.frame) / context.parameters.interval,
            best_backward.score,
            best_backward.source_branch_length,
            *backward_confidence,
            _matlab_single_divide(
                _matlab_distance(
                    (daughter.x, daughter.y),
                    (candidate.x, candidate.y),
                ),
                normalizer,
            ),
            _matlab_single_divide(
                _matlab_single_multiply(
                    _matlab_distance((daughter.z,), (candidate.z,)),
                    context.parameters.anisotropy_xyz[2],
                ),
                normalizer,
            ),
        )
    backward_block = _ensure_classifier_values(
        _matlab_single_vector(backward_block),
        "backward feature block",
    )

    forward_choices = (
        _forward_choice(context, daughter_ids[0], 0, normalizer),
        _forward_choice(context, daughter_ids[1], 1, normalizer),
    )
    # MATLAB's equality branch selects d1 on an exact length tie.
    selected_forward = (
        forward_choices[0]
        if minimum_length == lengths[0]
        else forward_choices[1]
    )
    forward_block = _ensure_classifier_values(
        _matlab_single_vector(_forward_block(selected_forward)),
        "forward feature block",
    )

    nondivision_vectors = (
        calculate_legacy_nondivision_pair(context, parent_id, daughter_ids[0]),
        calculate_legacy_nondivision_pair(context, parent_id, daughter_ids[1]),
    )
    nondivision_scores = calculate_legacy_nondivision_scores(
        context,
        parent_id,
        statistics,
    )

    pair_pdfs = (
        _mvn_pdf(
            pair1,
            statistics.division_pair_mean,
            statistics.division_pair_covariance,
        ),
        _mvn_pdf(
            pair2,
            statistics.division_pair_mean,
            statistics.division_pair_covariance,
        ),
    )
    triple_pdf = _mvn_pdf(
        triple,
        statistics.division_triple_mean,
        statistics.division_triple_covariance,
    )
    division_components = (
        _inverse_pdf(triple_pdf),
        _inverse_pdf(pair_pdfs[0]),
        _inverse_pdf(pair_pdfs[1]),
    )
    product = division_components[0] * division_components[1] * division_components[2]
    division_score = _matlab_log(product)

    best_forward_lengths = (
        forward_choices[0].target_length,
        forward_choices[1].target_length,
    )
    backward_present = (bool(backward_groups[0]), bool(backward_groups[1]))
    feature_input = SingleModelFeatureInput(
        daughter_features=daughter_block,
        backward_features=backward_block,
        forward_features=forward_block,
        daughter_lengths=lengths,
        backward_candidate_present=backward_present,
        best_forward_lengths=best_forward_lengths,
        small_cutoff=context.parameters.small_cutoff,
    )
    diagnostics = {
        "daughter_branch_ids": (
            context.traverse_forward(daughter_ids[0]),
            context.traverse_forward(daughter_ids[1]),
        ),
        "backward_candidate_ids": (
            tuple(item.source_id for item in backward_groups[0]),
            tuple(item.source_id for item in backward_groups[1]),
        ),
        "best_backward_candidate_id": (
            None if best_backward is None else best_backward.source_id
        ),
        "best_backward_daughter_index": (
            None if best_backward is None else best_backward.daughter_index
        ),
        "forward_candidate_ids": (
            tuple(
                item.target_id
                for item in enumerate_forward_candidates(
                    context,
                    forward_choices[0].endpoint_id,
                    context.parameters,
                ).candidates
            ),
            tuple(
                item.target_id
                for item in enumerate_forward_candidates(
                    context,
                    forward_choices[1].endpoint_id,
                    context.parameters,
                ).candidates
            ),
        ),
        "best_forward_candidate_ids": tuple(
            item.candidate_id for item in forward_choices
        ),
        "selected_forward_daughter_index": selected_forward.daughter_index,
        "division_score_components": division_components,
        "division_score": division_score,
        "nondivision_vectors": nondivision_vectors,
        "legacy_quirks": (
            "d2_backward_scoring_uses_d1_time",
            "d1_forward_gap_offset_is_not_interval_normalized",
            "no_forward_recursive_sentinel_differs_by_daughter",
            "division_target_length_is_not_interval_normalized",
            "parent_perpendicular_z_uses_mixed_coordinate_units",
        ),
        "record_answers": record_answers,
        "raw_daughter_lengths": raw_lengths,
    }
    return LegacyBifurcationExtraction(
        parent_id=parent_id,
        daughter_ids=daughter_ids,
        feature_input=feature_input,
        daughter_feature_names=DAUGHTER_FEATURE_NAMES,
        daughter_features=daughter_block,
        backward_feature_names=BACKWARD_FEATURE_NAMES,
        backward_features=backward_block,
        forward_feature_names=FORWARD_FEATURE_NAMES,
        forward_features=forward_block,
        daughter_lengths=lengths,
        backward_candidate_present=backward_present,
        best_forward_lengths=best_forward_lengths,
        nondivision_scores=(nondivision_scores[0], nondivision_scores[1]),
        repair_result=backward_candidates,
        false_negative_plan=false_negative_plan,
        diagnostics=diagnostics,
    )


def extract_retained_legacy_bifurcations(
    context: LegacyTrackingContext,
    statistics: LegacyTrackingStatistics,
    *,
    record_answers: bool = False,
) -> tuple[LegacyBifurcationExtraction, ...]:
    """Extract every active two-successor event in MATLAB frame/row order.

    This is the final-snapshot counterpart of the event-level extractor and is
    the Python side of the oracle's ``extracted_bifurcations`` table.  It does
    not claim to reproduce the tracking driver's dynamic mutation order.
    """

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(statistics, LegacyTrackingStatistics):
        raise TypeError("statistics must be LegacyTrackingStatistics")
    if type(record_answers) is not bool:
        raise TypeError("record_answers must be a boolean")

    results: list[LegacyBifurcationExtraction] = []
    for frame in range(1, context.parameters.end_frame + 1):
        for nucleus_id in context.frame_ids(frame, include_deleted=False):
            first, second = context.successor_slots(nucleus_id)
            if first is None or second is None:
                continue
            results.append(
                extract_legacy_bifurcation_features(
                    context,
                    nucleus_id,
                    statistics,
                    record_answers=record_answers,
                )
            )
    return tuple(results)


__all__ = [
    "BACKWARD_FEATURE_NAMES",
    "DAUGHTER_FEATURE_NAMES",
    "FORWARD_FEATURE_NAMES",
    "LegacyBifurcationExtraction",
    "LegacyFeatureExtractionError",
    "LegacyTrackingStatistics",
    "calculate_legacy_division_cost",
    "calculate_legacy_division_pair",
    "calculate_legacy_division_triple",
    "calculate_legacy_nondivision_cost",
    "calculate_legacy_nondivision_pair",
    "calculate_legacy_nondivision_scores",
    "extract_legacy_bifurcation_features",
    "extract_retained_legacy_bifurcations",
]
