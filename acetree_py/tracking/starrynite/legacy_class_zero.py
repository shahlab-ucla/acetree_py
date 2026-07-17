"""Exact legacy class-0 (``other``) bifurcation repair orchestration.

StarryNite's ``processOtherBifurcation`` is more than a nearest-neighbour
reattachment.  A loose source commits immediately, while a source with one
successor is turned into a provisional bifurcation, re-extracted, classified,
and either committed, rejected, or processed recursively.  This module keeps
that state machine separate from the generic class-label graph mutations so
the four raw MATLAB attempts and every provisional decision remain auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Callable, Literal

from ..api import TrackEdge
from .classifier import (
    AmbigiousClassifierPrediction,
    ClassifierPredictionError,
    NeutralAmbigiousClassifierFamily,
    NeutralNaiveBayesClassifier,
    SingleModelPrediction,
    classify_ambigious_family,
    classify_single_model,
)
from .legacy_features import (
    LegacyBifurcationExtraction,
    LegacyFeatureExtractionError,
    LegacyTrackingStatistics,
    calculate_legacy_nondivision_scores,
    extract_legacy_bifurcation_features,
)
from .legacy_mutations import (
    LegacyMutationError,
    resolve_legacy_false_positive_bifurcation,
)
from .legacy_state import LegacyStateError, LegacyTrackingContext
from .lineage import (
    BifurcationDecision,
    LineageGraphState,
    LineageResolutionAction,
    LineageResolutionDiagnostics,
    LineageResolutionError,
    LineageResolutionResult,
    resolve_bifurcation,
)
from .repair_candidates import (
    ClassZeroRepairCandidate,
    LegacyRepairCandidateError,
    enumerate_class_zero_candidates,
)


class LegacyClassZeroRepairError(ValueError):
    """Raised when a class-0 request cannot describe one exact split."""


LegacyClassifierModel = (
    NeutralNaiveBayesClassifier | NeutralAmbigiousClassifierFamily
)
LegacyClassifierPrediction = SingleModelPrediction | AmbigiousClassifierPrediction


LegacyClassZeroAttemptOutcome = Literal[
    "not_reached",
    "skipped_original_parent",
    "skipped_two_successors",
    "unsupported_provisional_topology",
    "direct_attached",
    "direct_attached_deleted_source",
    "tentative_other_rejected",
    "tentative_other_recursed",
    "tentative_division_committed",
    "tentative_false_negative_committed",
    "tentative_false_positive_committed",
]

LegacyClassZeroRepairStatus = Literal[
    "direct_attached",
    "recursive_other_committed",
    "division_committed",
    "false_negative_committed",
    "false_positive_committed",
    "attempts_exhausted",
    "unsupported",
]


@dataclass(frozen=True, slots=True)
class LegacyClassZeroAttemptDiagnostic:
    """Outcome of one member of the frozen four-attempt candidate snapshot."""

    candidate: ClassZeroRepairCandidate
    outcome: LegacyClassZeroAttemptOutcome = "not_reached"
    prediction: LegacyClassifierPrediction | None = None
    lineage_diagnostics: LineageResolutionDiagnostics | None = None
    recursive_diagnostics: LegacyClassZeroRepairDiagnostics | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ClassZeroRepairCandidate):
            raise TypeError("candidate must be a ClassZeroRepairCandidate")
        if self.prediction is not None and not isinstance(
            self.prediction,
            (SingleModelPrediction, AmbigiousClassifierPrediction),
        ):
            raise TypeError("prediction must be a legacy classifier prediction or None")
        if self.lineage_diagnostics is not None and not isinstance(
            self.lineage_diagnostics, LineageResolutionDiagnostics
        ):
            raise TypeError(
                "lineage_diagnostics must be LineageResolutionDiagnostics or None"
            )
        if self.recursive_diagnostics is not None and not isinstance(
            self.recursive_diagnostics, LegacyClassZeroRepairDiagnostics
        ):
            raise TypeError(
                "recursive_diagnostics must be LegacyClassZeroRepairDiagnostics "
                "or None"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))


@dataclass(frozen=True, slots=True)
class LegacyClassZeroRepairDiagnostics:
    """Complete, immutable audit trail for one class-0 invocation."""

    parent_id: str
    daughter_ids: tuple[str, str]
    detached_daughter_id: str
    retained_daughter_id: str
    nondivision_scores: tuple[float, float]
    raw_attempts: tuple[ClassZeroRepairCandidate, ...]
    attempts: tuple[LegacyClassZeroAttemptDiagnostic, ...]
    recursion_depth: int
    reattached_to_id: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.raw_attempts) != 4:
            raise LegacyClassZeroRepairError(
                "Legacy class-0 diagnostics require exactly four raw attempts"
            )
        if len(self.attempts) != len(self.raw_attempts):
            raise LegacyClassZeroRepairError(
                "Attempt diagnostics must retain the complete raw snapshot"
            )
        if tuple(item.candidate for item in self.attempts) != self.raw_attempts:
            raise LegacyClassZeroRepairError(
                "Attempt diagnostics must preserve raw candidate order and values"
            )
        if self.recursion_depth < 0:
            raise LegacyClassZeroRepairError("recursion_depth cannot be negative")


@dataclass(frozen=True, slots=True)
class LegacyClassZeroRepairResult:
    """Atomic legacy context and active lineage result for class-0 repair."""

    context: LegacyTrackingContext
    state: LineageGraphState
    status: LegacyClassZeroRepairStatus
    actions: tuple[LineageResolutionAction, ...]
    diagnostics: LegacyClassZeroRepairDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.context, LegacyTrackingContext):
            raise TypeError("context must be a LegacyTrackingContext")
        if not isinstance(self.state, LineageGraphState):
            raise TypeError("state must be a LineageGraphState")
        if any(not isinstance(item, LineageResolutionAction) for item in self.actions):
            raise TypeError("actions must contain LineageResolutionAction values")
        if not isinstance(self.diagnostics, LegacyClassZeroRepairDiagnostics):
            raise TypeError(
                "diagnostics must be a LegacyClassZeroRepairDiagnostics"
            )
        object.__setattr__(self, "actions", tuple(self.actions))

    @property
    def supported(self) -> bool:
        """Whether the requested MATLAB transition was represented safely."""

        return self.status != "unsupported"


@dataclass(frozen=True, slots=True)
class LegacyClassZeroClassificationObservation:
    """One inline round-2 classification with both raw attachment states.

    ``before_attachment_context`` is the detached state used to enumerate the
    frozen candidate. ``classification_context`` is the provisional raw state
    after that candidate has been attached and immediately before its feature
    extraction/classification result is acted on.  Keeping the two contexts in
    an optional observer rather than every repair result avoids quadratic movie
    retention while still allowing an oracle to snapshot every pointer.
    """

    before_attachment_context: LegacyTrackingContext
    classification_context: LegacyTrackingContext
    extraction: LegacyBifurcationExtraction
    prediction: LegacyClassifierPrediction
    candidate: ClassZeroRepairCandidate
    recursion_depth: int
    force_mode: bool
    classifier_round: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.before_attachment_context, LegacyTrackingContext):
            raise TypeError(
                "before_attachment_context must be a LegacyTrackingContext"
            )
        if not isinstance(self.classification_context, LegacyTrackingContext):
            raise TypeError("classification_context must be a LegacyTrackingContext")
        if not isinstance(self.extraction, LegacyBifurcationExtraction):
            raise TypeError("extraction must be a LegacyBifurcationExtraction")
        if not isinstance(
            self.prediction,
            (SingleModelPrediction, AmbigiousClassifierPrediction),
        ):
            raise TypeError("prediction must be a legacy classifier prediction")
        if not isinstance(self.candidate, ClassZeroRepairCandidate):
            raise TypeError("candidate must be a ClassZeroRepairCandidate")
        if (
            isinstance(self.recursion_depth, bool)
            or not isinstance(self.recursion_depth, Integral)
            or self.recursion_depth < 0
        ):
            raise TypeError("recursion_depth must be a non-negative integer")
        if type(self.force_mode) is not bool:
            raise TypeError("force_mode must be a boolean")
        if self.classifier_round != 2:
            raise LegacyClassZeroRepairError(
                "Nested legacy class-0 classifications must use round 2"
            )


LegacyClassZeroClassificationObserver = Callable[
    [LegacyClassZeroClassificationObservation], None
]


_EXPECTED_ATTEMPTS = 4
_SLOT_FEATURE = "LEGACY_SUCCESSOR_SLOT"


def _active_context_pairs(context: LegacyTrackingContext) -> set[tuple[str, str]]:
    return {
        (edge.source_id, edge.target_id)
        for edge in context.edges
        if edge.source_id not in context.deleted_ids
        and edge.target_id not in context.deleted_ids
    }


def _validate_state_matches_context(
    context: LegacyTrackingContext,
    state: LineageGraphState,
) -> None:
    frames = {item.nucleus_id: item.frame for item in context.nuclei}
    if dict(state.frames) != frames:
        raise LegacyClassZeroRepairError(
            "LineageGraphState frames do not match the legacy nuclei"
        )
    if state.deleted_ids != context.deleted_ids:
        raise LegacyClassZeroRepairError(
            "LineageGraphState and legacy context deletion records disagree"
        )
    state_pairs = {(edge.source_id, edge.target_id) for edge in state.edges}
    if state_pairs != _active_context_pairs(context):
        raise LegacyClassZeroRepairError(
            "LineageGraphState active edges do not match the legacy context"
        )


def _edge_kind(
    context: LegacyTrackingContext,
    source_id: str,
    target_id: str,
    successor_count: int,
) -> str:
    if successor_count == 2:
        return "split"
    if context.nucleus(target_id).frame - context.nucleus(source_id).frame > 1:
        return "gap"
    return "link"


def _rebuild_context_from_slots(
    context: LegacyTrackingContext,
    slots: dict[str, list[str | None]],
    *,
    added_edges: dict[tuple[str, str], TrackEdge] | None = None,
) -> LegacyTrackingContext:
    """Serialize contiguous MATLAB successor slots without mutating context."""

    templates = {(edge.source_id, edge.target_id): edge for edge in context.edges}
    if added_edges:
        templates.update(added_edges)
    rebuilt: list[TrackEdge] = []
    for nucleus in context.nuclei:
        source_id = nucleus.nucleus_id
        values = slots[source_id]
        if len(values) != 2 or (values[0] is None and values[1] is not None):
            raise LegacyClassZeroRepairError(
                f"Unsupported successor slots for {source_id!r}"
            )
        targets = tuple(item for item in values if item is not None)
        if len(set(targets)) != len(targets):
            raise LegacyClassZeroRepairError(
                f"Duplicate successor in slots for {source_id!r}"
            )
        for slot, target_id in enumerate(targets):
            template = templates.get((source_id, target_id))
            if template is None:
                raise LegacyClassZeroRepairError(
                    f"No edge evidence for {source_id!r} -> {target_id!r}"
                )
            features = dict(template.features)
            features[_SLOT_FEATURE] = slot
            rebuilt.append(
                TrackEdge(
                    source_id,
                    target_id,
                    template.cost,
                    _edge_kind(context, source_id, target_id, len(targets)),
                    features,
                )
            )
    try:
        return LegacyTrackingContext.from_nuclei_and_edges(
            context.nuclei,
            tuple(rebuilt),
            context.parameters,
            deleted_ids=context.deleted_ids,
            stale_predecessor_by_id=context.stale_predecessor_by_id,
        )
    except (LegacyStateError, TypeError, ValueError) as exc:
        raise LegacyClassZeroRepairError(str(exc)) from exc


def _slots(context: LegacyTrackingContext) -> dict[str, list[str | None]]:
    return {
        nucleus.nucleus_id: list(context.successor_slots(nucleus.nucleus_id))
        for nucleus in context.nuclei
    }


def _detach(
    context: LegacyTrackingContext,
    parent_id: str,
    detached_id: str,
) -> LegacyTrackingContext:
    values = _slots(context)
    first, second = values[parent_id]
    if first is None or second is None or detached_id not in {first, second}:
        raise LegacyClassZeroRepairError(
            f"Parent {parent_id!r} does not contain detached daughter {detached_id!r}"
        )
    values[parent_id] = [second, None] if detached_id == first else [first, None]
    return _rebuild_context_from_slots(context, values)


def _attach(
    context: LegacyTrackingContext,
    candidate: ClassZeroRepairCandidate,
    detached_id: str,
) -> LegacyTrackingContext:
    values = _slots(context)
    source_slots = values[candidate.source_id]
    expected = (
        [None, None]
        if candidate.eligibility == "direct_attach"
        else [source_slots[0], None]
    )
    if source_slots != expected:
        raise LegacyClassZeroRepairError(
            "Frozen class-0 candidate no longer matches its successor topology"
        )
    if any(detached_id in item for item in values.values()):
        raise LegacyClassZeroRepairError(
            f"Detached daughter {detached_id!r} still has a predecessor"
        )
    target_slot = 0 if candidate.eligibility == "direct_attach" else 1
    source_slots[target_slot] = detached_id
    cost = candidate.distance if math.isfinite(candidate.distance) else 0.0
    added = TrackEdge(
        candidate.source_id,
        detached_id,
        cost,
        "link" if target_slot == 0 else "split",
        {
            _SLOT_FEATURE: target_slot,
            "STARRYNITE_RESOLUTION_CLASS": 0,
            "STARRYNITE_REATTACHMENT": True,
            "LEGACY_CLASS_ZERO_RAW_RANK": candidate.raw_rank,
            "LEGACY_CLASS_ZERO_DISTANCE": candidate.distance,
        },
    )
    return _rebuild_context_from_slots(
        context,
        values,
        added_edges={(candidate.source_id, detached_id): added},
    )


def _state_from_context(
    context: LegacyTrackingContext,
    template: LineageGraphState,
) -> LineageGraphState:
    """Project raw legacy slots onto an active immutable lineage graph."""

    try:
        projected = context.to_lineage_graph_state()
    except (LineageResolutionError, TypeError, ValueError) as exc:
        raise LegacyClassZeroRepairError(str(exc)) from exc
    if projected.frames != template.frames:
        raise LegacyClassZeroRepairError(
            "Projected legacy frames changed during class-0 repair"
        )
    return projected


def _state_with_temporarily_active_source(
    context: LegacyTrackingContext,
    source_id: str,
) -> LineageGraphState:
    """Expose one deleted MATLAB row only while resolving its raw split.

    ``processOtherBifurcation`` never checks or clears ``delete(mind)`` before
    writing successor/predecessor slots.  A one-child deleted row must therefore
    participate in provisional classification even though it must remain absent
    from the executable lineage returned to callers.
    """

    if source_id not in context.deleted_ids:
        return context.to_lineage_graph_state()
    temporarily_deleted = context.deleted_ids - {source_id}
    try:
        return LineageGraphState(
            {item.nucleus_id: item.frame for item in context.nuclei},
            tuple(
                edge
                for edge in context.edges
                if edge.source_id not in temporarily_deleted
                and edge.target_id not in temporarily_deleted
            ),
            temporarily_deleted,
        )
    except (LineageResolutionError, TypeError, ValueError) as exc:
        raise LegacyClassZeroRepairError(
            f"Deleted source {source_id!r} cannot be provisionally activated: {exc}"
        ) from exc


def _ordered_active_edges(
    context: LegacyTrackingContext,
    state: LineageGraphState,
    source_id: str,
) -> tuple[TrackEdge, ...]:
    edges = tuple(edge for edge in state.edges if edge.source_id == source_id)
    old_slots = {
        target: slot
        for slot, target in enumerate(context.successor_slots(source_id))
        if target is not None
    }

    def order(edge: TrackEdge) -> tuple[int, int, int, str]:
        raw_slot = edge.features.get(_SLOT_FEATURE)
        explicit = (
            int(raw_slot)
            if isinstance(raw_slot, int)
            and not isinstance(raw_slot, bool)
            and raw_slot in {0, 1}
            else 2
        )
        target = context.nucleus(edge.target_id)
        return (
            explicit,
            old_slots.get(edge.target_id, 2),
            target.matlab_row,
            edge.target_id,
        )

    return tuple(sorted(edges, key=order))


def synchronize_legacy_context_after_resolution(
    context: LegacyTrackingContext,
    state: LineageGraphState,
    *,
    restore_deleted_ids: frozenset[str] = frozenset(),
) -> LegacyTrackingContext:
    """Synchronize a class-1/2/3 result back to raw legacy slot order.

    ``restore_deleted_ids`` identifies deleted MATLAB rows that were exposed
    temporarily so a provisional bifurcation could be resolved.  Their resolved
    raw slots are retained, but their delete flags are restored and all incident
    edges remain absent from the projected active lineage.

    The active :class:`LineageGraphState` cannot represent MATLAB's stale
    pointers on deleted rows.  This bridge retains unrelated pre-existing raw
    cross-deletion pointers and outgoing slots on a newly deleted branch.  It
    removes only an incoming link from a retained row to a newly condemned
    row, matching ``processFPBifurcation``'s explicit unlink and slot shift.
    """

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(state, LineageGraphState):
        raise TypeError("state must be a LineageGraphState")
    if not isinstance(restore_deleted_ids, frozenset) or any(
        type(item) is not str or not item for item in restore_deleted_ids
    ):
        raise TypeError(
            "restore_deleted_ids must be a frozenset of non-empty strings"
        )

    unknown_restored = restore_deleted_ids - context.deleted_ids
    if unknown_restored:
        raise LegacyClassZeroRepairError(
            "Only pre-existing deleted rows may be restored after provisional "
            "activation: " + ", ".join(sorted(unknown_restored))
        )
    effective_context_deleted = context.deleted_ids - restore_deleted_ids

    newly_deleted = state.deleted_ids - effective_context_deleted
    final_deleted = state.deleted_ids | restore_deleted_ids

    edges: list[TrackEdge] = []
    # Preserve raw edges that the active graph deliberately cannot expose.
    # Newly deleted sources keep all successor slots, including a slot to an
    # active side branch below a nested split.  Their incoming edge from a
    # retained parent is the pointer processFPBifurcation explicitly clears.
    # Pre-existing deleted rows are unrelated to this resolution and must not
    # be normalized as a side effect.
    for edge in context.edges:
        source_effectively_deleted = edge.source_id in effective_context_deleted
        target_effectively_deleted = edge.target_id in effective_context_deleted
        if edge.source_id in newly_deleted:
            edges.append(edge)
        elif edge.target_id in newly_deleted:
            continue
        elif source_effectively_deleted or target_effectively_deleted:
            edges.append(edge)
    for nucleus in context.nuclei:
        source_id = nucleus.nucleus_id
        if source_id in state.deleted_ids:
            continue
        ordered = _ordered_active_edges(context, state, source_id)
        for slot, edge in enumerate(ordered):
            features = dict(edge.features)
            features[_SLOT_FEATURE] = slot
            edges.append(
                TrackEdge(
                    edge.source_id,
                    edge.target_id,
                    edge.cost,
                    _edge_kind(context, edge.source_id, edge.target_id, len(ordered)),
                    features,
                )
            )
    try:
        rebuilt_pairs = {(edge.source_id, edge.target_id) for edge in edges}
        stale_predecessors = {
            target_id: source_id
            for target_id, source_id in context.stale_predecessor_by_id.items()
            if target_id in final_deleted
            and (source_id, target_id) not in rebuilt_pairs
        }
        return LegacyTrackingContext.from_nuclei_and_edges(
            context.nuclei,
            tuple(edges),
            context.parameters,
            deleted_ids=final_deleted,
            stale_predecessor_by_id=stale_predecessors,
        )
    except (LegacyStateError, TypeError, ValueError) as exc:
        raise LegacyClassZeroRepairError(str(exc)) from exc


def _diagnostics(
    *,
    extraction: LegacyBifurcationExtraction,
    detached_id: str,
    retained_id: str,
    raw_attempts: tuple[ClassZeroRepairCandidate, ...],
    attempt_results: list[LegacyClassZeroAttemptDiagnostic],
    depth: int,
    reattached_to_id: str | None = None,
    failure_reason: str | None = None,
) -> LegacyClassZeroRepairDiagnostics:
    return LegacyClassZeroRepairDiagnostics(
        parent_id=extraction.parent_id,
        daughter_ids=extraction.daughter_ids,
        detached_daughter_id=detached_id,
        retained_daughter_id=retained_id,
        nondivision_scores=extraction.nondivision_scores,
        raw_attempts=raw_attempts,
        attempts=tuple(attempt_results),
        recursion_depth=depth,
        reattached_to_id=reattached_to_id,
        failure_reason=failure_reason,
    )


def _unsupported_result(
    context: LegacyTrackingContext,
    state: LineageGraphState,
    *,
    extraction: LegacyBifurcationExtraction,
    detached_id: str,
    retained_id: str,
    raw_attempts: tuple[ClassZeroRepairCandidate, ...],
    attempt_results: list[LegacyClassZeroAttemptDiagnostic],
    depth: int,
    reason: str,
) -> LegacyClassZeroRepairResult:
    return LegacyClassZeroRepairResult(
        context=context,
        state=state,
        status="unsupported",
        actions=(),
        diagnostics=_diagnostics(
            extraction=extraction,
            detached_id=detached_id,
            retained_id=retained_id,
            raw_attempts=raw_attempts,
            attempt_results=attempt_results,
            depth=depth,
            failure_reason=reason,
        ),
    )


def _replace_attempt(
    attempts: list[LegacyClassZeroAttemptDiagnostic],
    index: int,
    *,
    outcome: LegacyClassZeroAttemptOutcome,
    prediction: LegacyClassifierPrediction | None = None,
    lineage_diagnostics: LineageResolutionDiagnostics | None = None,
    recursive_diagnostics: LegacyClassZeroRepairDiagnostics | None = None,
    notes: tuple[str, ...] = (),
) -> None:
    attempts[index] = LegacyClassZeroAttemptDiagnostic(
        attempts[index].candidate,
        outcome,
        prediction,
        lineage_diagnostics,
        recursive_diagnostics,
        notes,
    )


def _resolved_predecessor(
    context: LegacyTrackingContext,
    state: LineageGraphState,
    node_id: str,
) -> str | None:
    raw_predecessor = context.predecessor(node_id)
    if raw_predecessor is not None:
        return raw_predecessor
    if node_id in state.deleted_ids:
        return None
    return state.predecessor(node_id)


def _repair(
    context: LegacyTrackingContext,
    state: LineageGraphState,
    extraction: LegacyBifurcationExtraction,
    model: LegacyClassifierModel,
    statistics: LegacyTrackingStatistics,
    *,
    force_mode: bool,
    record_answers: bool,
    depth: int,
    visited_parents: frozenset[str],
    classification_observer: LegacyClassZeroClassificationObserver | None,
) -> LegacyClassZeroRepairResult:
    parent_id = extraction.parent_id
    extraction = replace(
        extraction,
        nondivision_scores=calculate_legacy_nondivision_scores(
            context,
            parent_id,
            statistics,
        ),
    )
    first_id, second_id = extraction.daughter_ids
    first_score, second_score = extraction.nondivision_scores
    # Ordinary MATLAB comparison: NaN falls through and detaches daughter 2.
    detached_id = first_id if first_score > second_score else second_id
    retained_id = second_id if detached_id == first_id else first_id

    base_context = _detach(context, parent_id, detached_id)
    base_state = _state_from_context(base_context, state)
    raw_attempts = enumerate_class_zero_candidates(
        base_context,
        parent_id,
        detached_id,
        base_context.parameters,
        attempt_limit=_EXPECTED_ATTEMPTS,
    )
    if len(raw_attempts) != _EXPECTED_ATTEMPTS:
        raise LegacyClassZeroRepairError(
            "Class-0 candidate extraction did not return four raw attempts"
        )
    attempts = [
        LegacyClassZeroAttemptDiagnostic(candidate) for candidate in raw_attempts
    ]
    detach_action = LineageResolutionAction(
        "detach_weaker_nondivision_branch",
        source_id=parent_id,
        target_id=detached_id,
        details={
            "daughter1_score": first_score,
            "daughter2_score": second_score,
            "tie_break": "daughter2",
            "legacy_nested_class_zero": True,
        },
    )

    for index, candidate in enumerate(raw_attempts):
        if candidate.eligibility == "original_parent":
            _replace_attempt(attempts, index, outcome="skipped_original_parent")
            continue
        if candidate.eligibility == "already_has_two_successors":
            _replace_attempt(attempts, index, outcome="skipped_two_successors")
            continue
        try:
            provisional_context = _attach(base_context, candidate, detached_id)
            provisional_state = _state_from_context(
                provisional_context,
                base_state,
            )
        except LegacyClassZeroRepairError as exc:
            reason = f"Unsupported provisional class-0 topology: {exc}"
            _replace_attempt(
                attempts,
                index,
                outcome="unsupported_provisional_topology",
                notes=(reason,),
            )
            return _unsupported_result(
                context,
                state,
                extraction=extraction,
                detached_id=detached_id,
                retained_id=retained_id,
                raw_attempts=raw_attempts,
                attempt_results=attempts,
                depth=depth,
                reason=reason,
            )

        if candidate.eligibility == "direct_attach":
            direct_outcome: LegacyClassZeroAttemptOutcome = (
                "direct_attached_deleted_source"
                if candidate.deleted
                else "direct_attached"
            )
            direct_notes = (
                (
                    "MATLAB raw slots were attached without clearing the source "
                    "delete flag; the edge is intentionally absent from the active "
                    "lineage",
                )
                if candidate.deleted
                else ()
            )
            _replace_attempt(
                attempts,
                index,
                outcome=direct_outcome,
                notes=direct_notes,
            )
            attach_action = LineageResolutionAction(
                "reattach_detached_branch",
                source_id=candidate.source_id,
                target_id=detached_id,
                details={
                    "cost": candidate.distance,
                    "raw_rank": candidate.raw_rank,
                    "legacy_direct_attach": True,
                    "source_deleted": candidate.deleted,
                    "active_lineage_edge": not candidate.deleted,
                    "matlab_delete_flag_preserved": candidate.deleted,
                },
            )
            return LegacyClassZeroRepairResult(
                provisional_context,
                provisional_state,
                "direct_attached",
                (detach_action, attach_action),
                _diagnostics(
                    extraction=extraction,
                    detached_id=detached_id,
                    retained_id=retained_id,
                    raw_attempts=raw_attempts,
                    attempt_results=attempts,
                    depth=depth,
                    reattached_to_id=candidate.source_id,
                ),
            )

        try:
            nested_extraction = extract_legacy_bifurcation_features(
                provisional_context,
                candidate.source_id,
                statistics,
                record_answers=record_answers,
            )
            classify = (
                classify_ambigious_family
                if isinstance(model, NeutralAmbigiousClassifierFamily)
                else classify_single_model
            )
            prediction = classify(
                model,
                nested_extraction.feature_input,
                force_mode=force_mode,
                backward_repair_available=(
                    nested_extraction.false_negative_plan is not None
                ),
            )
            if classification_observer is not None:
                classification_observer(
                    LegacyClassZeroClassificationObservation(
                        before_attachment_context=base_context,
                        classification_context=provisional_context,
                        extraction=nested_extraction,
                        prediction=prediction,
                        candidate=candidate,
                        recursion_depth=depth,
                        force_mode=force_mode,
                    )
                )
        except (
            ClassifierPredictionError,
            LegacyFeatureExtractionError,
            LegacyRepairCandidateError,
            LegacyStateError,
            LineageResolutionError,
            ValueError,
        ) as exc:
            reason = (
                f"Exact provisional extraction/classification failed for "
                f"{candidate.source_id!r}: {exc}"
            )
            _replace_attempt(
                attempts,
                index,
                outcome="unsupported_provisional_topology",
                notes=(reason,),
            )
            return _unsupported_result(
                context,
                state,
                extraction=extraction,
                detached_id=detached_id,
                retained_id=retained_id,
                raw_attempts=raw_attempts,
                attempt_results=attempts,
                depth=depth,
                reason=reason,
            )

        if prediction.predicted_class == 0:
            nested_first, nested_second = nested_extraction.nondivision_scores
            # MATLAB only recurses when the existing first successor is worse.
            # If the newly attached slot-2 daughter is worse (including ties),
            # it removes the provisional link and continues the outer attempts.
            if not nested_second < nested_first:
                _replace_attempt(
                    attempts,
                    index,
                    outcome="tentative_other_rejected",
                    prediction=prediction,
                    notes=(
                        "Nested class 0 kept daughter 1; provisional daughter 2 "
                        "was rejected",
                    ),
                )
                continue
            if candidate.source_id in visited_parents:
                reason = (
                    "Recursive legacy class-0 repair revisited parent "
                    f"{candidate.source_id!r}; refusing a topology cycle"
                )
                _replace_attempt(
                    attempts,
                    index,
                    outcome="unsupported_provisional_topology",
                    prediction=prediction,
                    notes=(reason,),
                )
                return _unsupported_result(
                    context,
                    state,
                    extraction=extraction,
                    detached_id=detached_id,
                    retained_id=retained_id,
                    raw_attempts=raw_attempts,
                    attempt_results=attempts,
                    depth=depth,
                    reason=reason,
                )
            recursive = _repair(
                provisional_context,
                provisional_state,
                nested_extraction,
                model,
                statistics,
                force_mode=force_mode,
                record_answers=record_answers,
                depth=depth + 1,
                visited_parents=visited_parents | {candidate.source_id},
                classification_observer=classification_observer,
            )
            _replace_attempt(
                attempts,
                index,
                outcome="tentative_other_recursed",
                prediction=prediction,
                recursive_diagnostics=recursive.diagnostics,
            )
            if not recursive.supported:
                reason = (
                    "Recursive legacy class-0 repair failed closed: "
                    f"{recursive.diagnostics.failure_reason}"
                )
                return _unsupported_result(
                    context,
                    state,
                    extraction=extraction,
                    detached_id=detached_id,
                    retained_id=retained_id,
                    raw_attempts=raw_attempts,
                    attempt_results=attempts,
                    depth=depth,
                    reason=reason,
                )
            attach_action = LineageResolutionAction(
                "provisionally_reattach_for_nested_classification",
                source_id=candidate.source_id,
                target_id=detached_id,
                details={"raw_rank": candidate.raw_rank, "nested_class": 0},
            )
            return LegacyClassZeroRepairResult(
                recursive.context,
                recursive.state,
                "recursive_other_committed",
                (detach_action, attach_action, *recursive.actions),
                _diagnostics(
                    extraction=extraction,
                    detached_id=detached_id,
                    retained_id=retained_id,
                    raw_attempts=raw_attempts,
                    attempt_results=attempts,
                    depth=depth,
                    reattached_to_id=_resolved_predecessor(
                        recursive.context,
                        recursive.state,
                        detached_id,
                    ),
                ),
            )

        decision = BifurcationDecision(
            classification=prediction.predicted_class,
            parent_id=nested_extraction.parent_id,
            daughter1_id=nested_extraction.daughter_ids[0],
            daughter2_id=nested_extraction.daughter_ids[1],
            daughter1_nondivision_score=nested_extraction.nondivision_scores[0],
            daughter2_nondivision_score=nested_extraction.nondivision_scores[1],
            false_negative_plan=nested_extraction.false_negative_plan,
        )
        restored_deleted = (
            frozenset({candidate.source_id})
            if candidate.deleted
            else frozenset()
        )
        try:
            if prediction.predicted_class == 3:
                raw_resolution = resolve_legacy_false_positive_bifurcation(
                    provisional_context,
                    nested_extraction.parent_id,
                    nested_extraction.daughter_ids,
                )
                lineage_result = LineageResolutionResult(
                    raw_resolution.state,
                    raw_resolution.actions,
                    raw_resolution.diagnostics,
                )
                resolved_context = raw_resolution.context
                resolved_state = raw_resolution.state
            else:
                resolution_state = (
                    _state_with_temporarily_active_source(
                        provisional_context,
                        candidate.source_id,
                    )
                    if candidate.deleted
                    else provisional_state
                )
                lineage_result = resolve_bifurcation(resolution_state, decision)
                resolved_context = (
                    provisional_context
                    if prediction.predicted_class == 1 and not candidate.deleted
                    else synchronize_legacy_context_after_resolution(
                        provisional_context,
                        lineage_result.state,
                        restore_deleted_ids=restored_deleted,
                    )
                )
                resolved_state = _state_from_context(
                    resolved_context,
                    provisional_state,
                )
        except (
            LegacyClassZeroRepairError,
            LegacyMutationError,
            LineageResolutionError,
            ValueError,
        ) as exc:
            reason = (
                f"Nested class {prediction.predicted_class} could not be applied "
                f"exactly for {candidate.source_id!r}: {exc}"
            )
            _replace_attempt(
                attempts,
                index,
                outcome="unsupported_provisional_topology",
                prediction=prediction,
                notes=(reason,),
            )
            return _unsupported_result(
                context,
                state,
                extraction=extraction,
                detached_id=detached_id,
                retained_id=retained_id,
                raw_attempts=raw_attempts,
                attempt_results=attempts,
                depth=depth,
                reason=reason,
            )

        outcome_by_class: dict[int, LegacyClassZeroAttemptOutcome] = {
            1: "tentative_division_committed",
            2: "tentative_false_negative_committed",
            3: "tentative_false_positive_committed",
        }
        status_by_class: dict[int, LegacyClassZeroRepairStatus] = {
            1: "division_committed",
            2: "false_negative_committed",
            3: "false_positive_committed",
        }
        _replace_attempt(
            attempts,
            index,
            outcome=outcome_by_class[prediction.predicted_class],
            prediction=prediction,
            lineage_diagnostics=lineage_result.diagnostics,
            notes=(
                (
                    "Deleted MATLAB source was provisionally activated for nested "
                    "resolution and its delete flag was restored atomically",
                )
                if candidate.deleted and prediction.predicted_class != 3
                else (
                    "Deleted MATLAB source remained raw-only while class 3 "
                    "updated its successor slots and preserved its delete flag",
                )
                if candidate.deleted
                else ()
            ),
        )
        attach_action = LineageResolutionAction(
            "provisionally_reattach_for_nested_classification",
            source_id=candidate.source_id,
            target_id=detached_id,
            details={
                "raw_rank": candidate.raw_rank,
                "nested_class": prediction.predicted_class,
            },
        )
        return LegacyClassZeroRepairResult(
            resolved_context,
            resolved_state,
            status_by_class[prediction.predicted_class],
            (detach_action, attach_action, *lineage_result.actions),
            _diagnostics(
                extraction=extraction,
                detached_id=detached_id,
                retained_id=retained_id,
                raw_attempts=raw_attempts,
                attempt_results=attempts,
                depth=depth,
                reattached_to_id=_resolved_predecessor(
                    resolved_context,
                    resolved_state,
                    detached_id,
                ),
            ),
        )

    return LegacyClassZeroRepairResult(
        base_context,
        base_state,
        "attempts_exhausted",
        (detach_action,),
        _diagnostics(
            extraction=extraction,
            detached_id=detached_id,
            retained_id=retained_id,
            raw_attempts=raw_attempts,
            attempt_results=attempts,
            depth=depth,
        ),
    )


def repair_legacy_class_zero_bifurcation(
    context: LegacyTrackingContext,
    state: LineageGraphState,
    extraction: LegacyBifurcationExtraction,
    model: LegacyClassifierModel,
    statistics: LegacyTrackingStatistics,
    *,
    force_mode: bool = False,
    record_answers: bool = False,
    classification_observer: LegacyClassZeroClassificationObserver | None = None,
) -> LegacyClassZeroRepairResult:
    """Apply exact ``processOtherBifurcation`` semantics without mutation.

    ``extraction`` is the already classified class-0 split.  Every one-child
    candidate is provisionally attached, re-extracted with the same legacy
    statistics, and classified again.  MATLAB attachments to deleted candidate
    rows are retained in the raw context without clearing their delete flags;
    those edges remain absent from the active lineage projection.  Otherwise
    unrepresentable topology returns ``status == \"unsupported\"`` with the
    original context and state, making the operation atomic and fail closed.
    """

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(state, LineageGraphState):
        raise TypeError("state must be a LineageGraphState")
    if not isinstance(extraction, LegacyBifurcationExtraction):
        raise TypeError("extraction must be a LegacyBifurcationExtraction")
    if not isinstance(
        model,
        (NeutralNaiveBayesClassifier, NeutralAmbigiousClassifierFamily),
    ):
        raise TypeError("model must be a supported neutral legacy classifier")
    if not isinstance(statistics, LegacyTrackingStatistics):
        raise TypeError("statistics must be LegacyTrackingStatistics")
    if type(force_mode) is not bool:
        raise TypeError("force_mode must be a boolean")
    if type(record_answers) is not bool:
        raise TypeError("record_answers must be a boolean")
    if classification_observer is not None and not callable(
        classification_observer
    ):
        raise TypeError("classification_observer must be callable or None")
    _validate_state_matches_context(context, state)

    if extraction.parent_id in context.deleted_ids:
        raise LegacyClassZeroRepairError("Class-0 parent cannot be deleted")
    if any(item in context.deleted_ids for item in extraction.daughter_ids):
        raise LegacyClassZeroRepairError("Class-0 daughters cannot be deleted")
    if context.successor_slots(extraction.parent_id) != extraction.daughter_ids:
        raise LegacyClassZeroRepairError(
            "Extraction daughters do not match the legacy successor slots"
        )
    parent_frame = context.nucleus(extraction.parent_id).frame
    if any(
        context.nucleus(item).frame != parent_frame + 1
        for item in extraction.daughter_ids
    ):
        raise LegacyClassZeroRepairError(
            "Legacy class-0 repair supports immediate-frame daughters only"
        )
    return _repair(
        context,
        state,
        extraction,
        model,
        statistics,
        force_mode=force_mode,
        record_answers=record_answers,
        depth=0,
        visited_parents=frozenset({extraction.parent_id}),
        classification_observer=classification_observer,
    )


__all__ = [
    "LegacyClassZeroAttemptDiagnostic",
    "LegacyClassZeroAttemptOutcome",
    "LegacyClassZeroClassificationObservation",
    "LegacyClassZeroClassificationObserver",
    "LegacyClassZeroRepairDiagnostics",
    "LegacyClassZeroRepairError",
    "LegacyClassZeroRepairResult",
    "LegacyClassZeroRepairStatus",
    "repair_legacy_class_zero_bifurcation",
    "synchronize_legacy_context_after_resolution",
]
