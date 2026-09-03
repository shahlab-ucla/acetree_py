"""Exact pre-classifier stages of the legacy StarryNite movie tracker.

The original tracker first creates conservative mutual-nearest-neighbour
links, optionally removes polar bodies, gathers asymmetric end candidates,
and then sweeps non-division and division score thresholds.  This module
retains MATLAB row order, strict cutoff comparisons, ordered successor slots,
and the candidate-list mutation quirks that are observable by later stages.

``run_legacy_early_tracking`` deliberately stops immediately before
``greedydeleteFPbranches``.  The post-greedy movie driver owns that function's
isolated-fragment prepass and classifier scan, preventing the deletion pass
from being applied twice at the integration seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ..api import TrackEdge
from .legacy_features import (
    LegacyFeatureExtractionError,
    LegacyTrackingStatistics,
    calculate_legacy_division_cost,
    calculate_legacy_nondivision_cost,
)
from .legacy_state import (
    LegacyStateError,
    LegacyTrackingContext,
    legacy_gram_distance,
    legacy_single_mean,
    legacy_single_round,
)


class LegacyEarlyTrackingError(ValueError):
    """Raised when an exact early-stage run cannot be represented safely."""


_NONDIVISION_COSTS = frozenset({"distance", "model"})
_DIVISION_COSTS = frozenset({"distance", "model"})


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LegacyEarlyTrackingError(f"{label} must be finite")
    if positive and result <= 0:
        raise LegacyEarlyTrackingError(f"{label} must be positive")
    return result


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise LegacyEarlyTrackingError(f"{label} must be at least {minimum}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _cost_name(value: Any, label: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must name a supported cost function")
    normalized = value.strip().lstrip("@").lower()
    aliases = {
        "distancecostfunction": "distance",
        "nondivscoremodelcostfunction": "model",
        "divdistancecostfunction": "distance",
        "divscoremodelcostfunction": "model",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in allowed:
        raise LegacyEarlyTrackingError(
            f"Unsupported {label} {value!r}; exact support is limited to "
            + ", ".join(sorted(allowed))
        )
    return normalized


def _tracking_parameters_source(model: Mapping[str, Any] | Any) -> Any:
    source: Any = model
    fields = getattr(source, "fields", None)
    if isinstance(fields, Mapping):
        source = fields
    if isinstance(source, Mapping) and "trackingparameters" in source:
        source = source["trackingparameters"]
    return source


def _read(source: Any, name: str, *, default: Any = None, required: bool = True) -> Any:
    if isinstance(source, Mapping):
        if name in source:
            return source[name]
    elif hasattr(source, name):
        return getattr(source, name)
    if required:
        raise LegacyEarlyTrackingError(
            f"trackingparameters.{name} is required for exact early tracking"
        )
    return default


@dataclass(frozen=True, slots=True)
class LegacyEarlyTrackingParameters:
    """Validated controls used before the bifurcation classifier scan."""

    start_frame: int
    end_frame: int
    safe_filter: bool
    safe_factor: float
    conflict_filter: bool
    backward_nn_count: int
    forward_nn_count: int
    min_nondivision_score: float
    nondivision_score_step: float
    max_nondivision_score: float
    min_division_score: float
    division_score_step: float
    max_division_score: float
    nondivision_cost: str = "distance"
    division_cost: str = "model"
    complete_divisions: bool = True
    polar_body_filter: bool = False
    polar_end_frame: int | None = None
    polar_threshold: float | None = None
    polar_threshold_high: float | None = None
    polar_threshold2_time: int | None = None
    polar_threshold2: float | None = None
    polar_threshold2_high: float | None = None
    hysteresis: bool = False
    hysteresis_intensity_high: float | None = None
    hysteresis_max_steps: int = 1
    delete_isolated: bool = False
    fp_size_threshold: int = 2
    early_cell_threshold: int = 250
    fp_size_threshold_small: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_frame", _integer(self.start_frame, "start_frame"))
        object.__setattr__(self, "end_frame", _integer(self.end_frame, "end_frame"))
        if self.end_frame < self.start_frame:
            raise LegacyEarlyTrackingError("end_frame cannot precede start_frame")
        for name in (
            "safe_filter",
            "conflict_filter",
            "complete_divisions",
            "polar_body_filter",
            "hysteresis",
            "delete_isolated",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        object.__setattr__(self, "safe_factor", _finite(self.safe_factor, "safe_factor", positive=True))
        object.__setattr__(
            self,
            "backward_nn_count",
            _integer(self.backward_nn_count, "backward_nn_count"),
        )
        object.__setattr__(
            self,
            "forward_nn_count",
            _integer(self.forward_nn_count, "forward_nn_count"),
        )
        for name in (
            "min_nondivision_score",
            "max_nondivision_score",
            "min_division_score",
            "max_division_score",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in ("nondivision_score_step", "division_score_step"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))
        if self.max_nondivision_score < self.min_nondivision_score:
            raise LegacyEarlyTrackingError("non-division threshold range is reversed")
        if self.max_division_score < self.min_division_score:
            raise LegacyEarlyTrackingError("division threshold range is reversed")
        object.__setattr__(
            self,
            "nondivision_cost",
            _cost_name(self.nondivision_cost, "nondivision_cost", _NONDIVISION_COSTS),
        )
        object.__setattr__(
            self,
            "division_cost",
            _cost_name(self.division_cost, "division_cost", _DIVISION_COSTS),
        )
        object.__setattr__(
            self,
            "hysteresis_max_steps",
            _integer(self.hysteresis_max_steps, "hysteresis_max_steps"),
        )
        for name in (
            "fp_size_threshold",
            "early_cell_threshold",
            "fp_size_threshold_small",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))

        if self.polar_body_filter:
            required = (
                "polar_end_frame",
                "polar_threshold",
                "polar_threshold_high",
                "polar_threshold2_time",
                "polar_threshold2",
                "polar_threshold2_high",
            )
            missing = [name for name in required if getattr(self, name) is None]
            if missing:
                raise LegacyEarlyTrackingError(
                    "Polar filtering requires " + ", ".join(missing)
                )
            object.__setattr__(
                self,
                "polar_end_frame",
                _integer(self.polar_end_frame, "polar_end_frame"),
            )
            object.__setattr__(
                self,
                "polar_threshold2_time",
                _integer(self.polar_threshold2_time, "polar_threshold2_time", minimum=0),
            )
            for name in (
                "polar_threshold",
                "polar_threshold_high",
                "polar_threshold2",
                "polar_threshold2_high",
            ):
                object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.hysteresis:
            if self.hysteresis_intensity_high is None:
                raise LegacyEarlyTrackingError(
                    "hysteresis_intensity_high is required when hysteresis is enabled"
                )
            object.__setattr__(
                self,
                "hysteresis_intensity_high",
                _finite(self.hysteresis_intensity_high, "hysteresis_intensity_high"),
            )

    @classmethod
    def from_model(
        cls,
        model: Mapping[str, Any] | Any,
        *,
        end_frame: int | None = None,
        nondivision_cost: str | None = None,
        division_cost: str | None = None,
        complete_divisions: bool = True,
        delete_isolated: bool | None = None,
    ) -> LegacyEarlyTrackingParameters:
        """Load scalar controls from a decoded ``trackingparameters`` struct.

        MATLAB function handles often become opaque when a MAT file is read
        outside MATLAB.  Callers must then provide their names from the legacy
        parameter file through ``nondivision_cost``/``division_cost``; this
        method never guesses an opaque function's identity.
        """

        source = _tracking_parameters_source(model)
        raw_nondivision = (
            nondivision_cost
            if nondivision_cost is not None
            else _read(source, "nonDivCostFunction")
        )
        raw_division = (
            division_cost
            if division_cost is not None
            else _read(source, "DivCostFunction")
        )
        polar = bool(_read(source, "polarbodyfilter", default=False, required=False))
        hysteresis = bool(_read(source, "hysteresis", default=False, required=False))
        delete = (
            bool(_read(source, "deleteisolated", default=False, required=False))
            if delete_isolated is None
            else delete_isolated
        )
        resolved_end = end_frame if end_frame is not None else _read(source, "endtime")
        return cls(
            start_frame=_read(source, "starttime"),
            end_frame=resolved_end,
            safe_filter=bool(_read(source, "safefilter")),
            safe_factor=_read(source, "safefactor"),
            conflict_filter=bool(_read(source, "conflictfilter")),
            backward_nn_count=_read(source, "nnnumber"),
            forward_nn_count=_read(source, "forwardnnnumber"),
            min_nondivision_score=_read(source, "minnondivscore"),
            nondivision_score_step=_read(source, "nondivscorestep"),
            max_nondivision_score=_read(source, "maxnondivscore"),
            min_division_score=_read(source, "mindivscore"),
            division_score_step=_read(source, "divscorestep"),
            max_division_score=_read(source, "maxdivscore"),
            nondivision_cost=raw_nondivision,
            division_cost=raw_division,
            complete_divisions=complete_divisions,
            polar_body_filter=polar,
            polar_end_frame=_read(source, "polarendtime", default=None, required=polar),
            polar_threshold=_read(source, "PolarThreshold", default=None, required=polar),
            polar_threshold_high=_read(source, "PolarThresholdHigh", default=None, required=polar),
            polar_threshold2_time=_read(source, "PolarThreshold2time", default=None, required=polar),
            polar_threshold2=_read(source, "PolarThreshold2", default=None, required=polar),
            polar_threshold2_high=_read(source, "PolarThreshold2High", default=None, required=polar),
            hysteresis=hysteresis,
            hysteresis_intensity_high=_read(
                source,
                "hysteresis_intensityhigh",
                default=None,
                required=hysteresis,
            ),
            hysteresis_max_steps=_read(
                source,
                "hysteresisMaxsteps",
                default=1,
                required=False,
            ),
            delete_isolated=delete,
            fp_size_threshold=_read(source, "FPsizethresh", default=2, required=delete),
            early_cell_threshold=_read(source, "earlythresh", default=250, required=delete),
            fp_size_threshold_small=_read(
                source,
                "FPsizethreshsmall",
                default=1,
                required=delete,
            ),
        )


@dataclass(frozen=True, slots=True)
class LegacyCandidateState:
    """MATLAB's ordered, deliberately asymmetric candidate cell arrays."""

    forward_by_id: Mapping[str, tuple[str, ...]]
    backward_by_id: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        forward = {
            str(key): tuple(str(item) for item in values)
            for key, values in self.forward_by_id.items()
        }
        backward = {
            str(key): tuple(str(item) for item in values)
            for key, values in self.backward_by_id.items()
        }
        object.__setattr__(self, "forward_by_id", MappingProxyType(forward))
        object.__setattr__(self, "backward_by_id", MappingProxyType(backward))

    def forward(self, nucleus_id: str) -> tuple[str, ...]:
        return self.forward_by_id.get(nucleus_id, ())

    def backward(self, nucleus_id: str) -> tuple[str, ...]:
        return self.backward_by_id.get(nucleus_id, ())


@dataclass(frozen=True, slots=True)
class LegacyEarlyStageSnapshot:
    """Full raw-pointer/candidate checkpoint for differential parity tests."""

    label: str
    threshold: float | None
    pointers: tuple[tuple[str, int, str], ...]
    deleted_ids: tuple[str, ...]
    forward_candidates: tuple[tuple[str, str], ...]
    backward_candidates: tuple[tuple[str, str], ...]
    link_count: int
    division_count: int


@dataclass(frozen=True, slots=True)
class LegacyEarlyStageSummary:
    """Bounded provenance for one early stage without retaining its full state."""

    label: str
    threshold: float | None
    link_count: int
    division_count: int
    deleted_count: int
    forward_candidate_count: int
    backward_candidate_count: int
    added_pointer_count: int
    removed_pointer_count: int
    newly_deleted_count: int
    restored_row_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a non-empty string")
        if self.threshold is not None:
            if isinstance(self.threshold, bool) or not isinstance(self.threshold, Real):
                raise TypeError("threshold must be a real number or None")
            object.__setattr__(self, "threshold", float(self.threshold))
        for name in (
            "link_count",
            "division_count",
            "deleted_count",
            "forward_candidate_count",
            "backward_candidate_count",
            "added_pointer_count",
            "removed_pointer_count",
            "newly_deleted_count",
            "restored_row_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, int(value))

    def as_provenance(self) -> dict[str, Any]:
        """Return the stable dictionary shape used by the exact backend."""

        threshold: float | str | None = self.threshold
        if isinstance(threshold, float) and not math.isfinite(threshold):
            threshold = "inf" if threshold > 0 else "-inf"
        return {
            "name": self.label,
            "threshold": threshold,
            "link_count": self.link_count,
            "division_count": self.division_count,
            "deleted_count": self.deleted_count,
            "forward_candidate_count": self.forward_candidate_count,
            "backward_candidate_count": self.backward_candidate_count,
            "added_pointer_count": self.added_pointer_count,
            "removed_pointer_count": self.removed_pointer_count,
            "newly_deleted_count": self.newly_deleted_count,
            "restored_row_count": self.restored_row_count,
        }


@dataclass(frozen=True, slots=True)
class LegacyEarlyTrackingResult:
    """Tentative geometry lineage at ``greedydeleteFPbranches`` entry."""

    context: LegacyTrackingContext
    candidates: LegacyCandidateState
    stages: tuple[LegacyEarlyStageSnapshot, ...] = field(default_factory=tuple)
    stage_summaries: tuple[LegacyEarlyStageSummary, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.context, LegacyTrackingContext):
            raise TypeError("context must be LegacyTrackingContext")
        if not isinstance(self.candidates, LegacyCandidateState):
            raise TypeError("candidates must be LegacyCandidateState")
        if any(not isinstance(item, LegacyEarlyStageSnapshot) for item in self.stages):
            raise TypeError("stages must contain LegacyEarlyStageSnapshot values")
        if any(
            not isinstance(item, LegacyEarlyStageSummary)
            for item in self.stage_summaries
        ):
            raise TypeError(
                "stage_summaries must contain LegacyEarlyStageSummary values"
            )
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "stage_summaries", tuple(self.stage_summaries))


def summarize_legacy_early_stages(
    stages: Sequence[LegacyEarlyStageSnapshot],
) -> tuple[LegacyEarlyStageSummary, ...]:
    """Reduce full parity checkpoints to bounded exact-backend provenance."""

    summaries: list[LegacyEarlyStageSummary] = []
    previous_pointers: frozenset[tuple[str, int, str]] = frozenset()
    previous_deleted: frozenset[str] = frozenset()
    for stage in stages:
        if not isinstance(stage, LegacyEarlyStageSnapshot):
            raise TypeError("stages must contain LegacyEarlyStageSnapshot values")
        pointers = frozenset(stage.pointers)
        deleted = frozenset(stage.deleted_ids)
        summaries.append(
            LegacyEarlyStageSummary(
                label=stage.label,
                threshold=stage.threshold,
                link_count=stage.link_count,
                division_count=stage.division_count,
                deleted_count=len(deleted),
                forward_candidate_count=len(stage.forward_candidates),
                backward_candidate_count=len(stage.backward_candidates),
                added_pointer_count=len(pointers - previous_pointers),
                removed_pointer_count=len(previous_pointers - pointers),
                newly_deleted_count=len(deleted - previous_deleted),
                restored_row_count=len(previous_deleted - deleted),
            )
        )
        previous_pointers = pointers
        previous_deleted = deleted
    return tuple(summaries)


def _thresholds(start: float, step: float, stop: float) -> tuple[float, ...]:
    """Return the finite values of the positive-step MATLAB colon expression."""

    tolerance = 4.0 * math.ulp(max(abs(start), abs(stop), abs(step), 1.0))
    count = int(math.floor((stop - start + tolerance) / step))
    return tuple(start + index * step for index in range(count + 1))


class _MutableEarlyState:
    def __init__(self, context: LegacyTrackingContext) -> None:
        self.base = context
        self.ids = tuple(
            nucleus_id
            for frame in range(1, context.parameters.end_frame + 1)
            for nucleus_id in context.frame_ids(frame, include_deleted=True)
        )
        self.slots = {item: [None, None] for item in self.ids}
        self.predecessor = {item: None for item in self.ids}
        self.deleted = set(context.deleted_ids)
        self.forward = {item: [] for item in self.ids}
        self.backward = {item: [] for item in self.ids}
        self.edge_details: dict[tuple[str, str], tuple[str, float]] = {}

    def link(
        self,
        source_id: str,
        target_id: str,
        slot: int,
        *,
        stage: str,
        score: float,
    ) -> None:
        if slot not in {0, 1}:
            raise LegacyEarlyTrackingError("Legacy successor slot must be zero or one")
        if self.slots[source_id][slot] is not None:
            raise LegacyEarlyTrackingError(
                f"Successor slot {slot} on {source_id!r} is already occupied"
            )
        if self.predecessor[target_id] is not None:
            raise LegacyEarlyTrackingError(
                f"Candidate {target_id!r} already has a predecessor"
            )
        self.slots[source_id][slot] = target_id
        self.predecessor[target_id] = source_id
        self.edge_details[(source_id, target_id)] = (stage, float(score))

    def unlink_slot(self, source_id: str, slot: int) -> str | None:
        target_id = self.slots[source_id][slot]
        if target_id is None:
            return None
        self.slots[source_id][slot] = None
        if self.predecessor[target_id] == source_id:
            self.predecessor[target_id] = None
        self.edge_details.pop((source_id, target_id), None)
        return target_id

    def context(self) -> LegacyTrackingContext:
        edges: list[TrackEdge] = []
        for source_id in self.ids:
            source_slots = self.slots[source_id]
            split = source_slots[1] is not None
            for slot, target_id in enumerate(source_slots):
                if target_id is None:
                    continue
                stage, score = self.edge_details.get(
                    (source_id, target_id),
                    ("legacy_early", math.nan),
                )
                source = self.base.nucleus(source_id)
                target = self.base.nucleus(target_id)
                kind = "split" if split else ("gap" if target.frame - source.frame > 1 else "link")
                edges.append(
                    TrackEdge(
                        source_id,
                        target_id,
                        0.0,
                        kind,
                        {
                            "LEGACY_SUCCESSOR_SLOT": slot,
                            "LEGACY_EARLY_STAGE": stage,
                            "LEGACY_LINK_SCORE": score,
                        },
                    )
                )
        return LegacyTrackingContext.from_nuclei_and_edges(
            self.base.nuclei,
            tuple(edges),
            self.base.parameters,
            deleted_ids=self.deleted,
        )

    def candidates(self) -> LegacyCandidateState:
        return LegacyCandidateState(
            {key: tuple(values) for key, values in self.forward.items()},
            {key: tuple(values) for key, values in self.backward.items()},
        )

    def snapshot(self, label: str, threshold: float | None) -> LegacyEarlyStageSnapshot:
        pointers = tuple(
            (source_id, slot, target_id)
            for source_id in self.ids
            for slot, target_id in enumerate(self.slots[source_id])
            if target_id is not None
        )
        forward = tuple(
            (source_id, target_id)
            for source_id in self.ids
            for target_id in self.forward[source_id]
        )
        backward = tuple(
            (source_id, target_id)
            for target_id in self.ids
            for source_id in self.backward[target_id]
        )
        return LegacyEarlyStageSnapshot(
            label=label,
            threshold=threshold,
            pointers=pointers,
            deleted_ids=tuple(item for item in self.ids if item in self.deleted),
            forward_candidates=forward,
            backward_candidates=backward,
            link_count=len(pointers),
            division_count=sum(1 for item in self.ids if self.slots[item][1] is not None),
        )

    def stage_summary(
        self,
        label: str,
        threshold: float | None,
        *,
        previous_pointers: frozenset[tuple[str, int, str]],
        previous_deleted: frozenset[str],
    ) -> tuple[
        LegacyEarlyStageSummary,
        frozenset[tuple[str, int, str]],
        frozenset[str],
    ]:
        """Capture counts/deltas while retaining only the preceding projection."""

        pointers = frozenset(
            (source_id, slot, target_id)
            for source_id in self.ids
            for slot, target_id in enumerate(self.slots[source_id])
            if target_id is not None
        )
        deleted = frozenset(item for item in self.ids if item in self.deleted)
        summary = LegacyEarlyStageSummary(
            label=label,
            threshold=threshold,
            link_count=len(pointers),
            division_count=sum(
                1 for item in self.ids if self.slots[item][1] is not None
            ),
            deleted_count=len(deleted),
            forward_candidate_count=sum(len(items) for items in self.forward.values()),
            backward_candidate_count=sum(
                len(items) for items in self.backward.values()
            ),
            added_pointer_count=len(pointers - previous_pointers),
            removed_pointer_count=len(previous_pointers - pointers),
            newly_deleted_count=len(deleted - previous_deleted),
            restored_row_count=len(previous_deleted - deleted),
        )
        return summary, pointers, deleted


def _require_clean_initializer_input(context: LegacyTrackingContext) -> None:
    if context.edges:
        raise LegacyEarlyTrackingError(
            "initializeTrackingStructures starts from an unlinked context"
        )
    if context.deleted_ids:
        raise LegacyEarlyTrackingError(
            "initializeTrackingStructures clears all delete flags"
        )


def _position_distance(
    context: LegacyTrackingContext,
    first_id: str,
    second_id: str,
) -> float:
    return legacy_gram_distance(
        context.nucleus(first_id).position_xyz,
        context.nucleus(second_id).position_xyz,
        context.parameters.anisotropy_xyz,
    )


def _link_easy_cases(
    state: _MutableEarlyState,
    parameters: LegacyEarlyTrackingParameters,
    maxima: Mapping[str, float] | None,
) -> None:
    context = state.base
    for frame in range(parameters.start_frame, parameters.end_frame):
        for source_id in context.frame_ids(frame, include_deleted=True):
            mutual_target: str | None = None
            for target_id in context.successor_suitors(source_id):
                if source_id in context.predecessor_suitors(target_id):
                    mutual_target = target_id
            if mutual_target is None:
                continue
            if parameters.conflict_filter and (
                len(context.predecessor_suitors(mutual_target)) != 1
                or len(context.successor_suitors(source_id)) != 1
            ):
                continue

            distance = _position_distance(context, source_id, mutual_target)
            if parameters.safe_filter:
                if context.parameters.absolute_cutoff:
                    measured_distance = distance * context.mean_self_distance(frame)
                else:
                    measured_distance = (
                        distance / context.mean_self_distance(frame)
                    ) * context.mean_self_distance(frame)
                safe_distance = min(
                    context.self_distance(source_id) / parameters.safe_factor,
                    context.self_distance(mutual_target) / parameters.safe_factor,
                )
                if measured_distance > safe_distance:
                    continue

            if parameters.hysteresis:
                assert maxima is not None
                threshold = float(parameters.hysteresis_intensity_high)
                if (
                    maxima[source_id] < threshold
                    and state.predecessor[source_id] is None
                    and maxima[mutual_target] < threshold
                    and state.slots[mutual_target][0] is None
                ):
                    continue
            if distance < context.forward_cutoff(frame):
                state.link(
                    source_id,
                    mutual_target,
                    0,
                    stage="easy_link",
                    score=distance,
                )


def _polar_body_pass(
    state: _MutableEarlyState,
    start: int,
    end: int,
    size_threshold: float,
    bright_threshold: float,
    bright_threshold_high: float,
    disk_max: Mapping[str, float],
) -> None:
    context = state.base
    for frame in range(start, end + 1):
        for nucleus_id in context.frame_ids(frame, include_deleted=True):
            nucleus = context.nucleus(nucleus_id)
            if nucleus.diameter < size_threshold and disk_max[nucleus_id] > bright_threshold:
                state.deleted.add(nucleus_id)

    for frame in range(start, end + 1):
        for nucleus_id in context.frame_ids(frame, include_deleted=True):
            if nucleus_id not in state.deleted:
                continue
            successor = state.slots[nucleus_id][0]
            predecessor = state.predecessor[nucleus_id]
            forward_deleted = successor is not None and successor in state.deleted
            backward_deleted = predecessor is not None and predecessor in state.deleted
            if (
                disk_max[nucleus_id] < bright_threshold_high
                and not forward_deleted
                and not backward_deleted
            ):
                state.deleted.remove(nucleus_id)
                continue
            if successor is not None:
                state.unlink_slot(nucleus_id, 0)
            if predecessor is not None:
                predecessor_slots = state.slots[predecessor]
                if predecessor_slots[0] == nucleus_id:
                    state.unlink_slot(predecessor, 0)
                elif predecessor_slots[1] == nucleus_id:
                    state.unlink_slot(predecessor, 1)


def _polar_body_filter(
    state: _MutableEarlyState,
    parameters: LegacyEarlyTrackingParameters,
    disk_max: Mapping[str, float],
) -> None:
    window_size = 10
    maximum = min(int(parameters.polar_end_frame), parameters.end_frame)
    starts = tuple(range(1, maximum - window_size + 1, window_size))
    if not starts:
        raise LegacyEarlyTrackingError(
            "The upstream polar-body driver references an undefined loop index "
            "when polarendtime/endtime is at most 10"
        )
    context = state.base
    last_start = starts[-1]
    for start in starts:
        diameters = [
            context.nucleus(nucleus_id).diameter
            for frame in range(start, start + window_size + 1)
            for nucleus_id in context.frame_ids(frame, include_deleted=True)
        ]
        size_threshold = legacy_single_mean(diameters)
        if start > int(parameters.polar_threshold2_time):
            bright = float(parameters.polar_threshold2)
            bright_high = float(parameters.polar_threshold2_high)
        else:
            bright = float(parameters.polar_threshold)
            bright_high = float(parameters.polar_threshold_high)
        _polar_body_pass(
            state,
            start,
            start + window_size,
            size_threshold,
            bright,
            bright_high,
            disk_max,
        )
    # MATLAB repeats the last start through maxtime, even when the final
    # regular window already covered part or all of that interval.
    if last_start > int(parameters.polar_threshold2_time):
        bright = float(parameters.polar_threshold2)
        bright_high = float(parameters.polar_threshold2_high)
    else:
        bright = float(parameters.polar_threshold)
        bright_high = float(parameters.polar_threshold_high)
    diameters = [
        context.nucleus(nucleus_id).diameter
        for frame in range(last_start, last_start + window_size + 1)
        for nucleus_id in context.frame_ids(frame, include_deleted=True)
    ]
    size_threshold = legacy_single_mean(diameters)
    _polar_body_pass(
        state,
        last_start,
        maximum,
        size_threshold,
        bright,
        bright_high,
        disk_max,
    )


def _gather_end_candidates(
    state: _MutableEarlyState,
    parameters: LegacyEarlyTrackingParameters,
) -> None:
    context = state.base
    # gatherEndCandidates hard-codes temporalcutoff=1 and nnnumber_gap=2.
    for frame in range(parameters.start_frame, parameters.end_frame + 1):
        for target_id in context.frame_ids(frame, include_deleted=True):
            if state.predecessor[target_id] is not None or target_id in state.deleted:
                continue
            source_frame = frame - 1
            if source_frame < 1:
                continue
            ranked = sorted(
                context.frame_ids(source_frame, include_deleted=True),
                key=lambda source_id: (
                    _position_distance(context, target_id, source_id),
                    context.nucleus(source_id).matlab_row,
                ),
            )
            for source_id in ranked[: parameters.backward_nn_count]:
                distance = _position_distance(context, target_id, source_id)
                if (
                    source_id not in state.deleted
                    and distance < context.forward_cutoff(source_frame)
                    and state.slots[source_id][1] is None
                ):
                    state.backward[target_id].append(source_id)
                    state.forward[source_id].append(target_id)

    for frame in range(parameters.start_frame, parameters.end_frame):
        for source_id in context.frame_ids(frame, include_deleted=True):
            candidates = state.forward[source_id]
            if len(candidates) <= parameters.forward_nn_count:
                continue
            ranked = sorted(
                enumerate(candidates),
                key=lambda item: (
                    _position_distance(context, source_id, item[1]),
                    item[0],
                ),
            )
            keep = {index for index, _target in ranked[: parameters.forward_nn_count]}
            state.forward[source_id] = [
                target_id
                for index, target_id in enumerate(candidates)
                if index in keep
            ]
            # The source MATLAB attempts to remove these entries from each
            # target's backcandidates using the target's own row/time instead
            # of this source row/time.  With temporalcutoff=1 that predicate is
            # always false, so the backward lists intentionally stay stale.


def _normalized_distance_cost(
    context: LegacyTrackingContext,
    source_id: str,
    target_id: str,
) -> float:
    distance = _position_distance(context, source_id, target_id)
    if context.parameters.absolute_cutoff:
        return distance
    return distance / context.mean_self_distance(context.nucleus(source_id).frame)


def _division_distance_cost(
    context: LegacyTrackingContext,
    parent_id: str,
    first_id: str,
    second_id: str,
) -> float:
    first = context.nucleus(first_id)
    second = context.nucleus(second_id)
    midpoint_one_based = tuple(
        legacy_single_round(
            legacy_single_round(left + 1.0 + right + 1.0) / 2.0
        )
        for left, right in zip(first.position_xyz, second.position_xyz, strict=True)
    )
    midpoint = tuple(value - 1.0 for value in midpoint_one_based)
    parent = context.nucleus(parent_id)
    distance = legacy_gram_distance(
        parent.position_xyz,
        midpoint,
        context.parameters.anisotropy_xyz,
    )
    if context.parameters.absolute_cutoff:
        return distance
    return distance / context.mean_self_distance(parent.frame)


def _remove_linked_target_from_forward_candidates(
    state: _MutableEarlyState,
    target_id: str,
) -> None:
    for source_id in state.backward[target_id]:
        state.forward[source_id] = [
            candidate for candidate in state.forward[source_id] if candidate != target_id
        ]


def _valid_greedy_candidates(
    state: _MutableEarlyState,
    source_id: str,
    parameters: LegacyEarlyTrackingParameters,
    maxima: Mapping[str, float] | None,
) -> list[str]:
    result: list[str] = []
    source = state.base.nucleus(source_id)
    for target_id in state.forward[source_id]:
        target = state.base.nucleus(target_id)
        # gapthresh is hard-coded to zero in both production sweeps.
        if target.frame != source.frame + 1:
            continue
        hysteresis_bad = False
        if parameters.hysteresis:
            assert maxima is not None
            threshold = float(parameters.hysteresis_intensity_high)
            hysteresis_bad = (
                maxima[source_id] < threshold
                and state.predecessor[source_id] is None
                and maxima[target_id] < threshold
                and state.slots[target_id][0] is None
            )
        if (
            not hysteresis_bad
            and state.predecessor[target_id] is None
            and target_id not in state.deleted
        ):
            result.append(target_id)
    return result


def _nondivision_cost(
    state: _MutableEarlyState,
    parameters: LegacyEarlyTrackingParameters,
    statistics: LegacyTrackingStatistics | None,
    source_id: str,
    target_id: str,
) -> float:
    if parameters.nondivision_cost == "distance":
        return _normalized_distance_cost(state.base, source_id, target_id)
    if statistics is None:
        raise LegacyEarlyTrackingError(
            "nondivScoreModelCostFunction requires LegacyTrackingStatistics"
        )
    return calculate_legacy_nondivision_cost(
        state.base,
        source_id,
        target_id,
        statistics,
    )


def _division_cost(
    state: _MutableEarlyState,
    parameters: LegacyEarlyTrackingParameters,
    statistics: LegacyTrackingStatistics | None,
    source_id: str,
    first_id: str,
    second_id: str,
) -> float:
    if parameters.division_cost == "distance":
        return _division_distance_cost(state.base, source_id, first_id, second_id)
    if statistics is None:
        raise LegacyEarlyTrackingError(
            "divScoreModelCostFunction requires LegacyTrackingStatistics"
        )
    return calculate_legacy_division_cost(
        state.base,
        source_id,
        first_id,
        second_id,
        statistics,
    )


def _matlab_min(items: Sequence[tuple[float, Any]]) -> tuple[float, Any] | None:
    if not items:
        return None
    best = items[0]
    for candidate in items[1:]:
        if candidate[0] < best[0]:
            best = candidate
    return best


def _greedy_end_score(
    state: _MutableEarlyState,
    parameters: LegacyEarlyTrackingParameters,
    statistics: LegacyTrackingStatistics | None,
    *,
    threshold: float,
    track_nondivision: bool,
    track_division: bool,
    maxima: Mapping[str, float] | None,
) -> None:
    context = state.base
    iterations = (
        parameters.hysteresis_max_steps
        if track_nondivision and not track_division and parameters.hysteresis
        else 1
    )
    for frame in range(parameters.start_frame, parameters.end_frame):
        for _iteration in range(iterations):
            changed = False
            for source_id in context.frame_ids(frame, include_deleted=True):
                if not state.forward[source_id] or source_id in state.deleted:
                    continue
                candidates = _valid_greedy_candidates(
                    state,
                    source_id,
                    parameters,
                    maxima,
                )
                nondivision_items: list[tuple[float, str]] = []
                if track_nondivision and state.slots[source_id][0] is None:
                    nondivision_items = [
                        (
                            _nondivision_cost(
                                state,
                                parameters,
                                statistics,
                                source_id,
                                target_id,
                            ),
                            target_id,
                        )
                        for target_id in candidates
                    ]

                division_items: list[tuple[float, tuple[str, str]]] = []
                if track_division:
                    existing = state.slots[source_id][0]
                    if existing is None:
                        for first_index in range(len(candidates) - 1):
                            for second_index in range(first_index + 1, len(candidates)):
                                first_id = candidates[first_index]
                                second_id = candidates[second_index]
                                division_items.append(
                                    (
                                        _division_cost(
                                            state,
                                            parameters,
                                            statistics,
                                            source_id,
                                            first_id,
                                            second_id,
                                        ),
                                        (first_id, second_id),
                                    )
                                )
                    else:
                        for target_id in candidates:
                            division_items.append(
                                (
                                    _division_cost(
                                        state,
                                        parameters,
                                        statistics,
                                        source_id,
                                        target_id,
                                        existing,
                                    ),
                                    (target_id, existing),
                                )
                            )

                best_nondivision = _matlab_min(nondivision_items)
                best_division = _matlab_min(division_items)
                nondivision_allowed = (
                    best_nondivision is not None and not (best_nondivision[0] > threshold)
                )
                division_allowed = (
                    best_division is not None and not (best_division[0] > threshold)
                )
                choose_division = division_allowed
                choose_nondivision = nondivision_allowed
                if choose_division and choose_nondivision:
                    if best_division[0] < best_nondivision[0]:
                        choose_nondivision = False
                    else:
                        choose_division = False
                if choose_division:
                    assert best_division is not None
                    score, (first_id, second_id) = best_division
                    if state.slots[source_id][0] is None:
                        state.link(
                            source_id,
                            first_id,
                            0,
                            stage="greedy_division",
                            score=score,
                        )
                        state.link(
                            source_id,
                            second_id,
                            1,
                            stage="greedy_division",
                            score=score,
                        )
                        _remove_linked_target_from_forward_candidates(state, first_id)
                        _remove_linked_target_from_forward_candidates(state, second_id)
                    else:
                        # MATLAB orders the new daughter first, overwriting the
                        # pair as [candidate, existing].  Recreate that slot
                        # ordering while retaining the original link details.
                        existing = state.slots[source_id][0]
                        assert existing == second_id
                        existing_details = state.edge_details[(source_id, existing)]
                        state.unlink_slot(source_id, 0)
                        state.link(
                            source_id,
                            first_id,
                            0,
                            stage="greedy_division",
                            score=score,
                        )
                        state.link(
                            source_id,
                            existing,
                            1,
                            stage=existing_details[0],
                            score=existing_details[1],
                        )
                        _remove_linked_target_from_forward_candidates(state, first_id)
                    state.forward[source_id] = []
                    changed = True
                elif choose_nondivision:
                    assert best_nondivision is not None
                    score, target_id = best_nondivision
                    state.link(
                        source_id,
                        target_id,
                        0,
                        stage="greedy_nondivision",
                        score=score,
                    )
                    _remove_linked_target_from_forward_candidates(state, target_id)
                    changed = True
            if not changed:
                break


def _clean_unlinked_hysteresis(
    state: _MutableEarlyState,
    parameters: LegacyEarlyTrackingParameters,
    maxima: Mapping[str, float],
) -> None:
    threshold = float(parameters.hysteresis_intensity_high)
    for frame in range(parameters.start_frame, parameters.end_frame + 1):
        for nucleus_id in state.base.frame_ids(frame, include_deleted=True):
            if (
                maxima[nucleus_id] < threshold
                and state.predecessor[nucleus_id] is None
                and state.slots[nucleus_id][0] is None
            ):
                state.deleted.add(nucleus_id)


def _validated_measurements(
    context: LegacyTrackingContext,
    values: Mapping[str, Real] | None,
    label: str,
) -> Mapping[str, float] | None:
    if values is None:
        return None
    result: dict[str, float] = {}
    expected = {item.nucleus_id for item in context.nuclei}
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise LegacyEarlyTrackingError(f"{label} IDs do not match nuclei: " + "; ".join(details))
    for nucleus_id, value in values.items():
        result[nucleus_id] = _finite(value, f"{label}[{nucleus_id!r}]")
    return MappingProxyType(result)


def run_legacy_early_tracking(
    initial_context: LegacyTrackingContext,
    parameters: LegacyEarlyTrackingParameters,
    statistics: LegacyTrackingStatistics | None,
    *,
    disk_max_by_id: Mapping[str, Real] | None = None,
    local_maximum_by_id: Mapping[str, Real] | None = None,
    capture_snapshots: bool = True,
    capture_summaries: bool = False,
) -> LegacyEarlyTrackingResult:
    """Run initialization through greedy tentative bifurcation creation.

    The returned context is the raw state at ``greedydeleteFPbranches`` entry,
    before its optional isolated-fragment deletion pass.
    """

    if not isinstance(initial_context, LegacyTrackingContext):
        raise TypeError("initial_context must be LegacyTrackingContext")
    if not isinstance(parameters, LegacyEarlyTrackingParameters):
        raise TypeError("parameters must be LegacyEarlyTrackingParameters")
    if statistics is not None and not isinstance(statistics, LegacyTrackingStatistics):
        raise TypeError("statistics must be LegacyTrackingStatistics or None")
    if type(capture_snapshots) is not bool:
        raise TypeError("capture_snapshots must be a boolean")
    if type(capture_summaries) is not bool:
        raise TypeError("capture_summaries must be a boolean")
    _require_clean_initializer_input(initial_context)
    if parameters.end_frame != initial_context.parameters.end_frame:
        raise LegacyEarlyTrackingError(
            "Early end_frame must equal LegacyFeatureParameters.end_frame"
        )
    maxima = _validated_measurements(
        initial_context,
        local_maximum_by_id,
        "local_maximum_by_id",
    )
    disk_max = _validated_measurements(
        initial_context,
        disk_max_by_id,
        "disk_max_by_id",
    )
    if parameters.hysteresis and maxima is None:
        raise LegacyEarlyTrackingError(
            "Exact hysteresis requires local_maximum_by_id for every nucleus"
        )
    if parameters.polar_body_filter and disk_max is None:
        raise LegacyEarlyTrackingError(
            "Exact polar-body filtering requires disk_max_by_id for every nucleus"
        )
    if (
        (parameters.nondivision_cost == "model" or parameters.division_cost == "model")
        and statistics is None
    ):
        raise LegacyEarlyTrackingError(
            "Model-based early cost functions require LegacyTrackingStatistics"
        )

    state = _MutableEarlyState(initial_context)
    snapshots: list[LegacyEarlyStageSnapshot] = []
    summaries: list[LegacyEarlyStageSummary] = []
    previous_pointers: frozenset[tuple[str, int, str]] = frozenset()
    previous_deleted: frozenset[str] = frozenset()

    def capture(label: str, threshold: float | None = None) -> None:
        nonlocal previous_pointers, previous_deleted
        if capture_snapshots:
            snapshots.append(state.snapshot(label, threshold))
        if capture_summaries:
            summary, previous_pointers, previous_deleted = state.stage_summary(
                label,
                threshold,
                previous_pointers=previous_pointers,
                previous_deleted=previous_deleted,
            )
            summaries.append(summary)

    capture("initialized")
    _link_easy_cases(state, parameters, maxima)
    capture("easy_links")
    if parameters.polar_body_filter:
        assert disk_max is not None
        _polar_body_filter(state, parameters, disk_max)
    capture("post_polar_filter")
    _gather_end_candidates(state, parameters)
    capture("candidates")

    for threshold in _thresholds(
        parameters.min_nondivision_score,
        parameters.nondivision_score_step,
        parameters.max_nondivision_score,
    ):
        _greedy_end_score(
            state,
            parameters,
            statistics,
            threshold=threshold,
            track_nondivision=True,
            track_division=False,
            maxima=maxima,
        )
        capture("nondivision", threshold)

    if parameters.hysteresis:
        assert maxima is not None
        _clean_unlinked_hysteresis(state, parameters, maxima)
        capture("hysteresis_cleanup")

    for threshold in _thresholds(
        parameters.min_division_score,
        parameters.division_score_step,
        parameters.max_division_score,
    ):
        _greedy_end_score(
            state,
            parameters,
            statistics,
            threshold=threshold,
            track_nondivision=False,
            track_division=True,
            maxima=maxima,
        )
        capture("division", threshold)
    if parameters.complete_divisions:
        _greedy_end_score(
            state,
            parameters,
            statistics,
            threshold=math.inf,
            track_nondivision=False,
            track_division=True,
            maxima=maxima,
        )
        capture("division", math.inf)
    capture("geometry_final")

    try:
        final_context = state.context()
    except (LegacyStateError, LegacyFeatureExtractionError, TypeError, ValueError) as exc:
        raise LegacyEarlyTrackingError(str(exc)) from exc
    return LegacyEarlyTrackingResult(
        context=final_context,
        candidates=state.candidates(),
        stages=tuple(snapshots),
        stage_summaries=tuple(summaries),
    )


__all__ = [
    "LegacyCandidateState",
    "LegacyEarlyStageSnapshot",
    "LegacyEarlyStageSummary",
    "LegacyEarlyTrackingError",
    "LegacyEarlyTrackingParameters",
    "LegacyEarlyTrackingResult",
    "run_legacy_early_tracking",
    "summarize_legacy_early_stages",
]
