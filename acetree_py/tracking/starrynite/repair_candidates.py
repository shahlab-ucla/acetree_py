"""Exact legacy StarryNite gap and reattachment candidate extraction.

The MATLAB tracker uses two distinct candidate procedures while resolving a
tentative bifurcation:

* backward/forward gap candidates are spatially gated and can induce a clean
  two- or three-player conflict rewire; and
* class-0 reattachment examines the four nearest *raw* nuclei, allowing
  ineligible nuclei to consume an attempt.

This module keeps both procedures deterministic and side-effect free.  In
particular, false-negative rewires are produced by simulating the legacy
ordered successor slots and then diffing the result into an atomic
``FalseNegativeRewirePlan``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Literal, Sequence

from ..api import TrackEdge
from .legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
    legacy_gram_distance,
    legacy_single_round,
    legacy_single_sum,
)
from .lineage import FalseNegativeRewirePlan


GapTopology = Literal["dirty", "clean2", "clean3"]
ClassZeroEligibility = Literal[
    "original_parent",
    "already_has_two_successors",
    "direct_attach",
    "tentative_bifurcation",
]


class LegacyRepairCandidateError(ValueError):
    """Raised when legacy candidate extraction encounters invalid topology."""


@dataclass(frozen=True, slots=True)
class FNPlayerTrace:
    """Conflict players and the optional third-player traces for one gap.

    A trace can be absent because it folds back into the two conflict players,
    or invalid because a gap/division interrupts it.  MATLAB represents those
    states as ``[]`` and ``-1`` respectively; explicit booleans avoid conflating
    them in Python.
    """

    start_players: tuple[str, ...]
    end_players: tuple[str, ...]
    start_backtrace: str | None
    end_forward_trace: str | None
    start_trace_valid: bool
    end_trace_valid: bool

    @property
    def is_clean(self) -> bool:
        extras_match = (self.start_backtrace is None) == (
            self.end_forward_trace is None
        )
        return (
            len(self.start_players) == 2
            and len(self.end_players) == 2
            and self.start_trace_valid
            and self.end_trace_valid
            and extras_match
        )

    @property
    def topology(self) -> GapTopology:
        if not self.is_clean:
            return "dirty"
        return "clean2" if self.start_backtrace is None else "clean3"


@dataclass(frozen=True, slots=True)
class GapScoreResult:
    """MATLAB ``gapScore`` result for one source/target pair."""

    score: float
    unnormalized_score: float
    normalization: float
    topology: GapTopology
    matching: tuple[int, int, int]
    start_players: tuple[str | None, str | None, str | None]
    end_players: tuple[str | None, str | None, str | None]


@dataclass(frozen=True, slots=True)
class BackwardRepairCandidate:
    """One backward FN option in legacy enumeration order."""

    daughter_index: Literal[1, 2]
    local_index: int
    enumeration_index: int
    source_id: str
    target_id: str
    source_frame: int
    target_frame: int
    offset: int
    source_branch_length: float
    endpoint_distance: float
    gap: GapScoreResult
    anisotropy_xyz: tuple[float, float, float]

    @property
    def score(self) -> float:
        return self.gap.score


@dataclass(frozen=True, slots=True)
class BackwardRepairCandidates:
    """Both daughter lists and MATLAB's first-min selected repair."""

    daughter1: tuple[BackwardRepairCandidate, ...]
    daughter2: tuple[BackwardRepairCandidate, ...]
    selected: BackwardRepairCandidate | None
    max_daughter1_branch_length: float
    max_daughter2_branch_length: float

    @property
    def available(self) -> bool:
        return self.selected is not None


@dataclass(frozen=True, slots=True)
class ForwardRepairCandidate:
    """One future gap option for a daughter branch endpoint."""

    local_index: int
    source_id: str
    target_id: str
    source_frame: int
    target_frame: int
    offset: int
    target_branch_length: float
    endpoint_distance: float
    gap: GapScoreResult

    @property
    def score(self) -> float:
        return self.gap.score


@dataclass(frozen=True, slots=True)
class ForwardRepairCandidates:
    """Forward options plus the selected classifier feature length."""

    candidates: tuple[ForwardRepairCandidate, ...]
    selected: ForwardRepairCandidate | None
    selected_feature_length: float

    @property
    def available(self) -> bool:
        return self.selected is not None


@dataclass(frozen=True, slots=True)
class ClassZeroRepairCandidate:
    """One of the four raw nearest attempts made by ``processOther``."""

    raw_rank: int
    source_id: str
    distance: float
    successor_count: int
    deleted: bool
    eligibility: ClassZeroEligibility

    @property
    def can_attach(self) -> bool:
        return self.eligibility in {"direct_attach", "tentative_bifurcation"}


def _node_id(node: LegacyNucleus) -> str:
    return node.nucleus_id


def _node_order(context: LegacyTrackingContext, node_id: str) -> tuple[int, int]:
    node = context.nucleus(node_id)
    return (node.frame, node.matlab_row)


def _position(node: LegacyNucleus) -> tuple[float, float, float]:
    return node.position_xyz


def anisotropic_distance(
    first: LegacyNucleus,
    second: LegacyNucleus,
    anisotropy: Sequence[float],
) -> float:
    """Reproduce ``distance_anisotropic`` for one pair of 3-D points."""

    scales = tuple(float(value) for value in anisotropy)
    if len(scales) != 3 or any(not math.isfinite(value) for value in scales):
        raise LegacyRepairCandidateError("anisotropy must contain three finite values")
    return legacy_gram_distance(
        _position(first),
        _position(second),
        scales,
    )


def _parameters_anisotropy(
    parameters: LegacyFeatureParameters,
) -> tuple[float, float, float]:
    return tuple(float(value) for value in parameters.anisotropy_xyz)  # type: ignore[return-value]


def _nodes_in_frame(
    context: LegacyTrackingContext,
    frame: int,
    *,
    include_deleted: bool = False,
) -> tuple[LegacyNucleus, ...]:
    return tuple(
        context.nucleus(node_id)
        for node_id in context.frame_ids(frame, include_deleted=include_deleted)
    )


def _row_sorted(
    context: LegacyTrackingContext, node_ids: Sequence[str]
) -> tuple[str, ...]:
    return tuple(sorted(set(node_ids), key=lambda item: _node_order(context, item)))


def _node_at_matlab_row(
    context: LegacyTrackingContext,
    frame: int,
    matlab_row: int,
    *,
    label: str,
) -> str:
    result = next(
        (
            node_id
            for node_id in context.frame_ids(frame, include_deleted=True)
            if context.nucleus(node_id).matlab_row == matlab_row
        ),
        None,
    )
    if result is None:
        raise LegacyRepairCandidateError(
            f"{label} cannot index MATLAB row {matlab_row} in frame {frame}"
        )
    return result


def _first_successor_depth(context: LegacyTrackingContext, node_id: str) -> int:
    depth = 1
    current = node_id
    seen = {current}
    while True:
        first, _second = context.successor_slots(current)
        if first is None:
            return depth
        if first in seen:
            raise LegacyRepairCandidateError("successor traversal contains a cycle")
        seen.add(first)
        current = first
        depth += 1


def _backward_division_stop_depth(
    context: LegacyTrackingContext, node_id: str
) -> int:
    depth = 1
    current = node_id
    seen = {current}
    while context.nucleus(current).frame != 1:
        predecessor = context.predecessor(current)
        if predecessor is None:
            return depth
        if context.successor_slots(predecessor)[1] is not None:
            return depth
        if predecessor in seen:
            raise LegacyRepairCandidateError("predecessor traversal contains a cycle")
        seen.add(predecessor)
        current = predecessor
        depth += 1
    return depth


def find_fn_players(
    context: LegacyTrackingContext,
    source_id: str,
    target_id: str,
) -> FNPlayerTrace:
    """Reproduce ``FindFNplayers`` using the frozen nearest-neighbor topology."""

    source = context.nucleus(source_id)
    target = context.nucleus(target_id)
    if target.frame <= source.frame:
        raise LegacyRepairCandidateError("FN target must be later than its source")

    forward_nn = context.f_nn(source_id)
    if forward_nn is None:
        start_players: tuple[str, ...] = ()
    else:
        start_players = tuple(
            item
            for item in context.predecessor_suitors(forward_nn)
            if item not in context.deleted_ids
        )
        actual_predecessor = context.predecessor(forward_nn)
        if actual_predecessor is not None:
            if context.nucleus(actual_predecessor).frame != source.frame:
                # An existing gap predecessor makes this a dirty case.
                start_players = ()
            else:
                start_players = _row_sorted(
                    context, (*start_players, actual_predecessor)
                )

    backward_nn = context.b_nn(target_id)
    if backward_nn is None:
        end_players: tuple[str, ...] = ()
    else:
        end_players = tuple(
            item
            for item in context.successor_suitors(backward_nn)
            if item not in context.deleted_ids
        )

    # Trace forward from the source's frame-to-frame NN.
    end_forward_trace = forward_nn
    end_trace_valid = (
        end_forward_trace is not None
        and end_forward_trace not in context.deleted_ids
    )
    if end_trace_valid:
        for frame in range(source.frame + 1, target.frame):
            assert end_forward_trace is not None
            current = context.nucleus(end_forward_trace)
            first, second = context.successor_slots(end_forward_trace)
            is_current_division_parent = (
                frame == target.frame - 1
                and end_forward_trace == context.predecessor(target_id)
            )
            if (
                current.frame != frame
                or first is None
                or context.nucleus(first).frame != frame + 1
                or (second is not None and not is_current_division_parent)
            ):
                end_forward_trace = None
                end_trace_valid = False
                break
            end_forward_trace = first
    if end_trace_valid and end_forward_trace in end_players:
        end_forward_trace = None

    # Trace backward from the target's frame-to-frame NN.
    start_backtrace = backward_nn
    start_trace_valid = (
        start_backtrace is not None and start_backtrace not in context.deleted_ids
    )
    if start_trace_valid:
        for frame in range(target.frame - 1, source.frame, -1):
            assert start_backtrace is not None
            current = context.nucleus(start_backtrace)
            predecessor = context.predecessor(start_backtrace)
            _first, second = context.successor_slots(start_backtrace)
            if (
                current.frame != frame
                or predecessor is None
                or context.nucleus(predecessor).frame != frame - 1
                or (frame != target.frame - 1 and second is not None)
            ):
                start_backtrace = None
                start_trace_valid = False
                break
            start_backtrace = predecessor
    if start_trace_valid and start_backtrace in start_players:
        start_backtrace = None

    return FNPlayerTrace(
        start_players=_row_sorted(context, start_players),
        end_players=_row_sorted(context, end_players),
        start_backtrace=start_backtrace,
        end_forward_trace=end_forward_trace,
        start_trace_valid=start_trace_valid,
        end_trace_valid=end_trace_valid,
    )


def _padded(players: Sequence[str]) -> tuple[str | None, str | None, str | None]:
    values = tuple(players)
    if len(values) > 3:
        raise LegacyRepairCandidateError("legacy FN scoring supports at most 3 players")
    return (values + (None,) * (3 - len(values)))  # type: ignore[return-value]


def _normalize_gap_score(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        if numerator == 0.0:
            return math.nan
        result = math.copysign(math.inf, numerator)
    else:
        result = numerator / denominator
    return legacy_single_round(result)


def _mean_pair(first: float, second: float) -> float:
    return legacy_single_round(legacy_single_sum((first, second)) / 2.0)


def score_gap_candidate(
    context: LegacyTrackingContext,
    source_id: str,
    target_id: str,
    parameters: LegacyFeatureParameters,
) -> GapScoreResult:
    """Score a gap exactly like MATLAB ``gapScore`` for one candidate."""

    trace = find_fn_players(context, source_id, target_id)
    anisotropy = _parameters_anisotropy(parameters)
    topology = trace.topology

    if topology == "dirty":
        numerator = anisotropic_distance(
            context.nucleus(source_id), context.nucleus(target_id), anisotropy
        )
        matching = (1, 0, 0)
        starts = _padded((source_id,))
        ends = _padded((target_id,))
    elif topology == "clean2":
        start1, start2 = trace.start_players
        end1, end2 = trace.end_players
        identity = _mean_pair(
            anisotropic_distance(
                context.nucleus(start1),
                context.nucleus(end1),
                anisotropy,
            ),
            anisotropic_distance(
                context.nucleus(start2), context.nucleus(end2), anisotropy
            ),
        )
        swapped = _mean_pair(
            anisotropic_distance(
                context.nucleus(start1),
                context.nucleus(end2),
                anisotropy,
            ),
            anisotropic_distance(
                context.nucleus(start2), context.nucleus(end1), anisotropy
            ),
        )
        # Strict comparison: an exact tie selects the swapped assignment.
        if identity < swapped:
            numerator = identity
            matching = (1, 2, 0)
        else:
            numerator = swapped
            matching = (2, 1, 0)
        starts = _padded(trace.start_players)
        ends = _padded(trace.end_players)
    else:
        assert trace.start_backtrace is not None
        assert trace.end_forward_trace is not None
        start_ids = (*trace.start_players, trace.start_backtrace)
        end_ids = (*trace.end_players, trace.end_forward_trace)
        # scoreTriplePosition scales z only, even if the configured x/y factors
        # are not one.  permutations() has the same first-min order as the
        # MATLAB i/j/k nested loops.
        triple_scale = (1.0, 1.0, anisotropy[2])
        best_sum = math.inf
        best_matching = (1, 2, 3)
        for permutation in permutations((0, 1, 2)):
            total = legacy_single_sum(
                anisotropic_distance(
                    context.nucleus(start_ids[index]),
                    context.nucleus(end_ids[target_index]),
                    triple_scale,
                )
                for index, target_index in enumerate(permutation)
            )
            if total < best_sum:
                best_sum = total
                best_matching = tuple(  # type: ignore[assignment]
                    index + 1 for index in permutation
                )
        numerator = legacy_single_round(best_sum / 3.0)
        matching = best_matching
        starts = _padded(start_ids)
        ends = _padded(end_ids)

    normalization = float(context.mean_self_distance(context.nucleus(source_id).frame))
    return GapScoreResult(
        score=_normalize_gap_score(numerator, normalization),
        unnormalized_score=numerator,
        normalization=normalization,
        topology=topology,
        matching=matching,
        start_players=starts,
        end_players=ends,
    )


def enumerate_backward_candidates(
    context: LegacyTrackingContext,
    target_id: str,
    parameters: LegacyFeatureParameters,
    *,
    daughter_index: Literal[1, 2],
    _score_target_frame: int | None = None,
) -> tuple[BackwardRepairCandidate, ...]:
    """Enumerate every strict-gated backward candidate in MATLAB order."""

    target = context.nucleus(target_id)
    if target_id in context.deleted_ids:
        return ()
    anisotropy = _parameters_anisotropy(parameters)
    candidates: list[BackwardRepairCandidate] = []
    for offset in range(
        parameters.temporal_cutoff_start, parameters.temporal_cutoff + 1
    ):
        source_frame = target.frame - offset
        # MATLAB checks frame >= 1, not trackingparameters.starttime.
        if source_frame < 1:
            continue
        cutoff = float(context.forward_cutoff(source_frame))
        for source in _nodes_in_frame(context, source_frame):
            source_id = _node_id(source)
            distance = anisotropic_distance(target, source, anisotropy)
            if (
                source_id in context.deleted_ids
                or not distance < cutoff
                or context.successor_slots(source_id)[0] is not None
            ):
                continue
            local_index = len(candidates) + 1
            score_target_id = target_id
            if _score_target_frame is not None:
                score_target_id = _node_at_matlab_row(
                    context,
                    _score_target_frame,
                    target.matlab_row,
                    label="legacy d2 scoring",
                )
            gap = score_gap_candidate(
                context,
                source_id,
                score_target_id,
                parameters,
            )
            candidates.append(
                BackwardRepairCandidate(
                    daughter_index=daughter_index,
                    local_index=local_index,
                    enumeration_index=0,
                    source_id=source_id,
                    target_id=target_id,
                    source_frame=source.frame,
                    target_frame=target.frame,
                    offset=offset,
                    source_branch_length=(
                        _backward_division_stop_depth(context, source_id)
                        / parameters.interval
                    ),
                    endpoint_distance=distance,
                    gap=gap,
                    anisotropy_xyz=anisotropy,
                )
            )
    return tuple(candidates)


def extract_backward_repair_candidates(
    context: LegacyTrackingContext,
    daughter1_id: str,
    daughter2_id: str,
    parameters: LegacyFeatureParameters,
) -> BackwardRepairCandidates:
    """Extract and select backward repair candidates for both daughters."""

    daughter1 = enumerate_backward_candidates(
        context, daughter1_id, parameters, daughter_index=1
    )
    daughter1_nucleus = context.nucleus(daughter1_id)
    context.nucleus(daughter2_id)
    # Intentional legacy bug: computeBestFNBackOption passes d2's numeric row
    # together with tcur1 to gapScore.  Discovery and rewiring still use the
    # actual d2 node at tcur2, but scoring indexes the node occupying d2's row
    # in d1's frame.
    daughter2 = enumerate_backward_candidates(
        context,
        daughter2_id,
        parameters,
        daughter_index=2,
        _score_target_frame=daughter1_nucleus.frame,
    )
    combined: list[BackwardRepairCandidate] = []
    for enumeration_index, candidate in enumerate((*daughter1, *daughter2), start=1):
        combined.append(
            BackwardRepairCandidate(
                daughter_index=candidate.daughter_index,
                local_index=candidate.local_index,
                enumeration_index=enumeration_index,
                source_id=candidate.source_id,
                target_id=candidate.target_id,
                source_frame=candidate.source_frame,
                target_frame=candidate.target_frame,
                offset=candidate.offset,
                source_branch_length=candidate.source_branch_length,
                endpoint_distance=candidate.endpoint_distance,
                gap=candidate.gap,
                anisotropy_xyz=candidate.anisotropy_xyz,
            )
        )
    split = len(daughter1)
    daughter1 = tuple(combined[:split])
    daughter2 = tuple(combined[split:])
    # Python's min is stable, matching MATLAB's first-index tie break.  NaN is
    # left unmodified for exact diagnostics; ordinary fixtures have positive
    # finite frame means.
    selected = min(combined, key=lambda candidate: candidate.score) if combined else None
    return BackwardRepairCandidates(
        daughter1=daughter1,
        daughter2=daughter2,
        selected=selected,
        max_daughter1_branch_length=max(
            (candidate.source_branch_length for candidate in daughter1), default=-1.0
        ),
        max_daughter2_branch_length=max(
            (candidate.source_branch_length for candidate in daughter2), default=-1.0
        ),
    )


def enumerate_forward_candidates(
    context: LegacyTrackingContext,
    source_id: str,
    parameters: LegacyFeatureParameters,
) -> ForwardRepairCandidates:
    """Enumerate future candidate starts and select the first minimum score."""

    source = context.nucleus(source_id)
    anisotropy = _parameters_anisotropy(parameters)
    candidates: list[ForwardRepairCandidate] = []
    maximum_offset = min(
        parameters.temporal_cutoff, parameters.end_frame - source.frame
    )
    for offset in range(2, maximum_offset + 1):
        target_frame = source.frame + offset
        cutoff = float(context.forward_cutoff(target_frame))
        for target in _nodes_in_frame(context, target_frame):
            target_id = _node_id(target)
            if target_id in context.deleted_ids:
                continue
            predecessor = context.predecessor(target_id)
            starts_at_root = predecessor is None
            starts_at_division = False
            if predecessor is not None:
                starts_at_division = (
                    context.nucleus(predecessor).frame == target_frame - 1
                    and context.successor_slots(predecessor)[1] is not None
                )
            distance = anisotropic_distance(source, target, anisotropy)
            if not distance < cutoff or not (starts_at_root or starts_at_division):
                continue
            gap = score_gap_candidate(context, source_id, target_id, parameters)
            candidates.append(
                ForwardRepairCandidate(
                    local_index=len(candidates) + 1,
                    source_id=source_id,
                    target_id=target_id,
                    source_frame=source.frame,
                    target_frame=target.frame,
                    offset=offset,
                    target_branch_length=(
                        _first_successor_depth(context, target_id) / parameters.interval
                    ),
                    endpoint_distance=distance,
                    gap=gap,
                )
            )

    selected = min(candidates, key=lambda candidate: candidate.score) if candidates else None
    if selected is None:
        selected_feature_length = -1.0
    else:
        selected_feature_length = selected.target_branch_length
        predecessor = context.predecessor(selected.target_id)
        if predecessor is not None:
            first, second = context.successor_slots(predecessor)
            if first is None or second is None:
                raise LegacyRepairCandidateError(
                    "division-origin forward candidate has incomplete daughter slots"
                )
            # Intentional legacy unit quirk: this override is in raw frames and
            # is not divided by interval.
            selected_feature_length = float(
                min(
                    _first_successor_depth(context, first),
                    _first_successor_depth(context, second),
                )
            )
    return ForwardRepairCandidates(
        candidates=tuple(candidates),
        selected=selected,
        selected_feature_length=selected_feature_length,
    )


def enumerate_class_zero_candidates(
    context: LegacyTrackingContext,
    parent_id: str,
    detached_id: str,
    parameters: LegacyFeatureParameters,
    *,
    attempt_limit: int = 4,
) -> tuple[ClassZeroRepairCandidate, ...]:
    """Return the four raw nearest class-0 attempts, including skipped rows."""

    if attempt_limit < 1:
        raise LegacyRepairCandidateError("attempt_limit must be positive")
    parent = context.nucleus(parent_id)
    detached = context.nucleus(detached_id)
    if detached.frame != parent.frame + 1:
        raise LegacyRepairCandidateError(
            "legacy class-0 reattachment requires an immediate-frame daughter"
        )
    # MATLAB does not filter deleted nuclei here; a deleted row can consume an
    # attempt or even be selected for attachment.
    nodes = _nodes_in_frame(context, parent.frame, include_deleted=True)
    if not nodes:
        return ()
    anisotropy = _parameters_anisotropy(parameters)
    distances = [anisotropic_distance(node, detached, anisotropy) for node in nodes]
    result: list[ClassZeroRepairCandidate] = []
    for raw_rank in range(1, attempt_limit + 1):
        # First row wins an exact tie, including the all-Inf case after fewer
        # than four nuclei have been exhausted.
        index = min(range(len(nodes)), key=lambda item: (distances[item], item))
        source = nodes[index]
        source_id = _node_id(source)
        distance = distances[index]
        first, second = context.successor_slots(source_id)
        if source_id == parent_id:
            eligibility: ClassZeroEligibility = "original_parent"
        elif second is not None:
            eligibility = "already_has_two_successors"
        elif first is None:
            eligibility = "direct_attach"
        else:
            eligibility = "tentative_bifurcation"
        result.append(
            ClassZeroRepairCandidate(
                raw_rank=raw_rank,
                source_id=source_id,
                distance=distance,
                successor_count=int(first is not None) + int(second is not None),
                deleted=source_id in context.deleted_ids,
                eligibility=eligibility,
            )
        )
        distances[index] = math.inf
    return tuple(result)


def _slot_edge_pairs(
    slots: dict[str, list[str | None]],
) -> set[tuple[str, str]]:
    return {
        (source, target)
        for source, values in slots.items()
        for target in values
        if target is not None
    }


def _ordered_pairs(
    context: LegacyTrackingContext, pairs: set[tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (
                *_node_order(context, pair[0]),
                *_node_order(context, pair[1]),
            ),
        )
    )


def _edge_for_addition(
    context: LegacyTrackingContext,
    source_id: str,
    target_id: str,
    *,
    gap_pair: tuple[str, str],
    gap_score: float,
    topology: GapTopology,
    successor_slot: int,
) -> TrackEdge:
    source_frame = context.nucleus(source_id).frame
    target_frame = context.nucleus(target_id).frame
    cost = gap_score if (source_id, target_id) == gap_pair else 0.0
    if not math.isfinite(cost) or cost < 0:
        raise LegacyRepairCandidateError(
            "selected gap score must be finite and non-negative for a rewire plan"
        )
    return TrackEdge(
        source_id,
        target_id,
        cost,
        "gap" if target_frame - source_frame > 1 else "link",
        {
            "LEGACY_SUCCESSOR_SLOT": successor_slot,
            "STARRYNITE_FN_REPAIR": True,
            "STARRYNITE_FN_TOPOLOGY": topology,
        },
    )


def build_false_negative_rewire_plan(
    context: LegacyTrackingContext,
    parent_id: str,
    daughter1_id: str,
    daughter2_id: str,
    candidate: BackwardRepairCandidate,
) -> FalseNegativeRewirePlan:
    """Simulate ``processFNBifurcation`` slots and return their graph diff."""

    parent_slots = context.successor_slots(parent_id)
    if parent_slots != (daughter1_id, daughter2_id):
        raise LegacyRepairCandidateError(
            "parent successor slots must exactly match daughter1/daughter2"
        )
    all_nodes = tuple(
        node
        for frame in range(1, max(context.nucleus(node_id).frame for node_id in (
            parent_id,
            daughter1_id,
            daughter2_id,
            candidate.source_id,
        )) + 1)
        for node in _nodes_in_frame(context, frame)
    )
    # Include later frames too when the context exposes them only through the
    # selected player IDs.  Every source touched below is inserted lazily.
    slots: dict[str, list[str | None]] = {
        _node_id(node): list(context.successor_slots(_node_id(node)))
        for node in all_nodes
    }
    touched_ids = {
        parent_id,
        candidate.source_id,
        *(item for item in candidate.gap.start_players if item is not None),
        *(item for item in candidate.gap.end_players if item is not None),
    }
    for node_id in touched_ids:
        slots.setdefault(node_id, list(context.successor_slots(node_id)))
    old_pairs = _slot_edge_pairs(slots)
    scored_topology = candidate.gap.topology
    end_players = tuple(
        item for item in candidate.gap.end_players if item is not None
    )
    first_two_end_players = set(end_players[:2])
    end_players_same_as_division = (
        daughter1_id in first_two_end_players
        and daughter2_id in first_two_end_players
    )
    first_two_start_players = tuple(
        item for item in candidate.gap.start_players[:2] if item is not None
    )
    start_link_present = any(
        context.successor_slots(item)[0] is not None
        for item in first_two_start_players
    )
    # A clean score is processed as a simple one-to-one repair unless its
    # conflict players overlap this exact division and at least one start
    # claimant is attached to the intervening chain.
    topology: GapTopology = scored_topology
    if (
        len(end_players) == 1
        or not end_players_same_as_division
        or not start_link_present
    ):
        topology = "dirty"
    gap_pair: tuple[str, str]

    if topology == "dirty":
        gap_pair = (candidate.source_id, candidate.target_id)
        slots[candidate.source_id][0] = candidate.target_id
        if candidate.daughter_index == 1:
            slots[parent_id][0] = slots[parent_id][1]
        slots[parent_id][1] = None
    elif topology == "clean2":
        starts = tuple(item for item in candidate.gap.start_players[:2] if item)
        ends = tuple(item for item in candidate.gap.end_players[:2] if item)
        if len(starts) != 2 or len(ends) != 2:
            raise LegacyRepairCandidateError("clean2 candidate needs two players per side")
        oneback = context.predecessor(ends[0])
        if oneback is None:
            raise LegacyRepairCandidateError("clean2 end player lacks predecessor")
        successor_options = [context.successor_slots(item)[0] for item in starts]
        linked_options = [item for item in successor_options if item is not None]
        if not linked_options:
            raise LegacyRepairCandidateError("clean2 starts lack a middle successor")
        # MATLAB takes max of the numeric successor row, then indexes that row
        # in starttime+1.  Resolve the same immediate-frame row even if a
        # malformed claimant's stored successor time points somewhere else.
        oneforward_row = max(
            context.nucleus(item).matlab_row for item in linked_options
        )
        oneforward = _node_at_matlab_row(
            context,
            candidate.source_frame + 1,
            oneforward_row,
            label="legacy clean2 middle successor",
        )
        matching = candidate.gap.matching
        first_pair = legacy_single_sum(
            (
                anisotropic_distance(
                    context.nucleus(starts[0]),
                    context.nucleus(oneforward),
                    candidate.anisotropy_xyz,
                ),
                anisotropic_distance(
                    context.nucleus(ends[matching[0] - 1]),
                    context.nucleus(oneback),
                    candidate.anisotropy_xyz,
                ),
            )
        )
        second_pair = legacy_single_sum(
            (
                anisotropic_distance(
                    context.nucleus(starts[1]),
                    context.nucleus(oneforward),
                    candidate.anisotropy_xyz,
                ),
                anisotropic_distance(
                    context.nucleus(ends[matching[1] - 1]),
                    context.nucleus(oneback),
                    candidate.anisotropy_xyz,
                ),
            )
        )
        if first_pair < second_pair:
            start_middle = starts[0]
            end_middle = ends[matching[0] - 1]
            gap_start = starts[1]
            gap_end = ends[matching[1] - 1]
        else:
            start_middle = starts[1]
            end_middle = ends[matching[1] - 1]
            gap_start = starts[0]
            gap_end = ends[matching[0] - 1]
        slots[start_middle][0] = oneforward
        slots[oneback][0] = end_middle
        slots[gap_start][0] = gap_end
        slots[oneback][1] = None
        gap_pair = (gap_start, gap_end)
    else:
        starts = tuple(item for item in candidate.gap.start_players if item)
        ends = tuple(item for item in candidate.gap.end_players if item)
        if len(starts) != 3 or len(ends) != 3:
            raise LegacyRepairCandidateError("clean3 candidate needs three players per side")
        matching = candidate.gap.matching
        match_of_start3 = ends[matching[2] - 1]
        if match_of_start3 == daughter1_id:
            gap_end = daughter2_id
            slots[parent_id][1] = None
        else:
            gap_end = daughter1_id
            slots[parent_id][0] = slots[parent_id][1]
            slots[parent_id][1] = None
        try:
            start_matching_end3_index = matching.index(3)
        except ValueError as exc:
            raise LegacyRepairCandidateError(
                "clean3 matching must contain end player 3"
            ) from exc
        other_match = starts[1] if start_matching_end3_index == 0 else starts[0]
        start_matching_end3 = starts[start_matching_end3_index]
        if start_matching_end3 == candidate.source_id:
            gap_start = other_match
            slots[candidate.source_id][0] = slots[other_match][0]
            slots[other_match][0] = None
        else:
            gap_start = candidate.source_id
        slots[gap_start][0] = gap_end
        gap_pair = (gap_start, gap_end)

    new_pairs = _slot_edge_pairs(slots)
    removals = _ordered_pairs(context, old_pairs - new_pairs)
    additions = _ordered_pairs(context, new_pairs - old_pairs)
    add_edges = tuple(
        _edge_for_addition(
            context,
            source,
            target,
            gap_pair=gap_pair,
            gap_score=candidate.score,
            topology=topology,
            successor_slot=slots[source].index(target),
        )
        for source, target in additions
    )
    return FalseNegativeRewirePlan(
        remove_edges=removals,
        add_edges=add_edges,
        label=(
            f"legacy-{topology}-fn-rewire"
            if topology == scored_topology
            else f"legacy-{scored_topology}-demoted-to-simple-fn-rewire"
        ),
    )


__all__ = [
    "BackwardRepairCandidate",
    "BackwardRepairCandidates",
    "ClassZeroRepairCandidate",
    "FNPlayerTrace",
    "ForwardRepairCandidate",
    "ForwardRepairCandidates",
    "GapScoreResult",
    "LegacyRepairCandidateError",
    "anisotropic_distance",
    "build_false_negative_rewire_plan",
    "enumerate_backward_candidates",
    "enumerate_class_zero_candidates",
    "enumerate_forward_candidates",
    "extract_backward_repair_candidates",
    "find_fn_players",
    "score_gap_candidate",
]
