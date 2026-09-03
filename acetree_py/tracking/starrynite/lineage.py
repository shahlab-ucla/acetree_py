"""Validated, immutable StarryNite bifurcation-resolution primitives.

The legacy tracker first builds a deliberately over-divided lineage and then
classifies every tentative bifurcation as one of four observable classes:

``0`` other, ``1`` division, ``2`` false-negative gap, or ``3`` false-positive
branch.  This module implements only the graph mutation associated with those
class labels.  Candidate discovery and classifier evaluation remain separate
so a false-negative repair can never be guessed from incomplete information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..api import Detection, TrackEdge


class LineageResolutionError(ValueError):
    """Raised when a lineage state or requested repair is not well formed."""


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _edge_key(edge: TrackEdge) -> tuple[str, str, str, float]:
    return (edge.source_id, edge.target_id, edge.kind, edge.cost)


@dataclass(frozen=True, slots=True)
class LineageGraphState:
    """An immutable active lineage graph plus its retained deletion record.

    ``frames`` contains every known node, including deleted nodes.  ``edges``
    contains active edges only.  This makes branch deletion explicit without
    losing the identities needed by MATLAB/Python parity diagnostics.
    """

    frames: Mapping[str, int]
    edges: tuple[TrackEdge, ...] = ()
    deleted_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        frames: dict[str, int] = {}
        for raw_id, raw_frame in self.frames.items():
            node_id = str(raw_id)
            if not node_id:
                raise LineageResolutionError("Lineage node IDs cannot be empty")
            if isinstance(raw_frame, bool):
                raise LineageResolutionError("Lineage frames must be integers")
            frame = int(raw_frame)
            if frame != raw_frame or frame < 1:
                raise LineageResolutionError(
                    f"Lineage frame for {node_id!r} must be a positive integer"
                )
            frames[node_id] = frame
        if len(frames) != len(self.frames):
            raise LineageResolutionError("Lineage node IDs must be unique strings")

        deleted = frozenset(str(item) for item in self.deleted_ids)
        unknown_deleted = deleted - frames.keys()
        if unknown_deleted:
            raise LineageResolutionError(
                "Deleted IDs are unknown: " + ", ".join(sorted(unknown_deleted))
            )

        edges = tuple(self.edges)
        pairs: set[tuple[str, str]] = set()
        predecessor_count: dict[str, int] = {}
        successor_count: dict[str, int] = {}
        for edge in edges:
            if not isinstance(edge, TrackEdge):
                raise LineageResolutionError("Lineage edges must be TrackEdge values")
            pair = (edge.source_id, edge.target_id)
            if pair in pairs:
                raise LineageResolutionError(
                    f"Duplicate lineage edge {edge.source_id!r} -> {edge.target_id!r}"
                )
            pairs.add(pair)
            if edge.source_id not in frames or edge.target_id not in frames:
                raise LineageResolutionError("Lineage edges must reference known nodes")
            if edge.source_id in deleted or edge.target_id in deleted:
                raise LineageResolutionError(
                    "Active lineage edges cannot reference deleted nodes"
                )
            if frames[edge.target_id] <= frames[edge.source_id]:
                raise LineageResolutionError(
                    "Lineage edges must point strictly forward in time"
                )
            predecessor_count[edge.target_id] = (
                predecessor_count.get(edge.target_id, 0) + 1
            )
            successor_count[edge.source_id] = successor_count.get(edge.source_id, 0) + 1
            if predecessor_count[edge.target_id] > 1:
                raise LineageResolutionError(
                    f"Lineage merge at {edge.target_id!r} is not supported"
                )
            if successor_count[edge.source_id] > 2:
                raise LineageResolutionError(
                    f"Node {edge.source_id!r} has more than two successors"
                )

        object.__setattr__(self, "frames", MappingProxyType(frames))
        object.__setattr__(self, "edges", tuple(sorted(edges, key=_edge_key)))
        object.__setattr__(self, "deleted_ids", deleted)

    @classmethod
    def from_detections(
        cls,
        detections: Sequence[Detection],
        edges: Sequence[TrackEdge] = (),
        *,
        deleted_ids: Sequence[str] = (),
    ) -> LineageGraphState:
        frames: dict[str, int] = {}
        for detection in detections:
            if detection.detection_id in frames:
                raise LineageResolutionError(
                    f"Duplicate detection ID {detection.detection_id!r}"
                )
            frames[detection.detection_id] = detection.frame
        return cls(frames, tuple(edges), frozenset(deleted_ids))

    @property
    def active_ids(self) -> frozenset[str]:
        return frozenset(self.frames.keys() - self.deleted_ids)

    def successors(self, node_id: str) -> tuple[str, ...]:
        self._require_active(node_id)
        return tuple(
            edge.target_id for edge in self.edges if edge.source_id == node_id
        )

    def predecessor(self, node_id: str) -> str | None:
        self._require_active(node_id)
        matches = tuple(
            edge.source_id for edge in self.edges if edge.target_id == node_id
        )
        return matches[0] if matches else None

    def edge(self, source_id: str, target_id: str) -> TrackEdge | None:
        return next(
            (
                edge
                for edge in self.edges
                if edge.source_id == source_id and edge.target_id == target_id
            ),
            None,
        )

    def _require_active(self, node_id: str) -> None:
        if node_id not in self.frames:
            raise LineageResolutionError(f"Unknown lineage node {node_id!r}")
        if node_id in self.deleted_ids:
            raise LineageResolutionError(f"Lineage node {node_id!r} is deleted")


@dataclass(frozen=True, slots=True)
class LineageReattachmentCandidate:
    """One explicit class-0 source candidate, ranked by ascending cost.

    Equal-cost candidates retain caller order, matching MATLAB's first-index
    ``min`` tie break when the caller supplies candidates in nucleus-row order.
    """

    source_id: str
    cost: float
    features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise LineageResolutionError("Reattachment source ID cannot be empty")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise LineageResolutionError(
                "Reattachment candidate cost must be finite and non-negative"
            )
        object.__setattr__(self, "features", _immutable_mapping(self.features))


@dataclass(frozen=True, slots=True)
class FalseNegativeRewirePlan:
    """An atomic, caller-selected class-2 graph repair.

    At least one parent-to-daughter edge must be removed and replaced by a
    genuine multi-frame ``gap`` edge to that daughter.  Extra remove/add edges
    permit the legacy clean two- and three-player conflict rewires while all
    graph invariants are validated atomically.
    """

    remove_edges: tuple[tuple[str, str], ...]
    add_edges: tuple[TrackEdge, ...]
    label: str = ""

    def __post_init__(self) -> None:
        removals = tuple((str(source), str(target)) for source, target in self.remove_edges)
        if len(set(removals)) != len(removals):
            raise LineageResolutionError("False-negative removals must be unique")
        additions = tuple(self.add_edges)
        if any(not isinstance(edge, TrackEdge) for edge in additions):
            raise LineageResolutionError(
                "False-negative additions must be TrackEdge values"
            )
        if len({(edge.source_id, edge.target_id) for edge in additions}) != len(
            additions
        ):
            raise LineageResolutionError("False-negative additions must be unique")
        object.__setattr__(self, "remove_edges", removals)
        object.__setattr__(self, "add_edges", additions)


@dataclass(frozen=True, slots=True)
class BifurcationDecision:
    """Classifier output and the explicit evidence needed to apply it."""

    classification: int
    parent_id: str
    daughter1_id: str
    daughter2_id: str
    daughter1_nondivision_score: float | None = None
    daughter2_nondivision_score: float | None = None
    reattachment_candidates: tuple[LineageReattachmentCandidate, ...] = ()
    false_negative_plan: FalseNegativeRewirePlan | None = None

    def __post_init__(self) -> None:
        if self.classification not in {0, 1, 2, 3}:
            raise LineageResolutionError(
                "StarryNite bifurcation class must be 0, 1, 2, or 3"
            )
        identifiers = (self.parent_id, self.daughter1_id, self.daughter2_id)
        if any(not item for item in identifiers) or len(set(identifiers)) != 3:
            raise LineageResolutionError(
                "Bifurcation parent and daughter IDs must be non-empty and distinct"
            )
        if self.classification == 0:
            for label, score in (
                ("daughter1", self.daughter1_nondivision_score),
                ("daughter2", self.daughter2_nondivision_score),
            ):
                if score is None or isinstance(score, bool) or not isinstance(score, Real):
                    raise LineageResolutionError(
                        f"Class 0 requires a numeric {label} nondivision score"
                    )
        object.__setattr__(
            self, "reattachment_candidates", tuple(self.reattachment_candidates)
        )
        if any(
            not isinstance(item, LineageReattachmentCandidate)
            for item in self.reattachment_candidates
        ):
            raise LineageResolutionError(
                "Reattachment candidates must be LineageReattachmentCandidate values"
            )


@dataclass(frozen=True, slots=True)
class LineageResolutionAction:
    kind: str
    source_id: str | None = None
    target_id: str | None = None
    node_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(self, "details", _immutable_mapping(self.details))


@dataclass(frozen=True, slots=True)
class LineageResolutionDiagnostics:
    classification: int
    parent_id: str
    selected_daughter_id: str | None = None
    detached_daughter_id: str | None = None
    reattached_to_id: str | None = None
    deleted_ids: tuple[str, ...] = ()
    applied_gap_edges: tuple[tuple[str, str], ...] = ()
    skipped_reattachment_candidates: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LineageResolutionResult:
    state: LineageGraphState
    actions: tuple[LineageResolutionAction, ...]
    diagnostics: LineageResolutionDiagnostics


def _validate_bifurcation(
    state: LineageGraphState, decision: BifurcationDecision
) -> None:
    for node_id in (
        decision.parent_id,
        decision.daughter1_id,
        decision.daughter2_id,
    ):
        state._require_active(node_id)
    successors = set(state.successors(decision.parent_id))
    expected = {decision.daughter1_id, decision.daughter2_id}
    if successors != expected:
        raise LineageResolutionError(
            f"Parent {decision.parent_id!r} does not have the requested two daughters"
        )


def _linear_branch(state: LineageGraphState, daughter_id: str) -> tuple[str, ...]:
    """Follow MATLAB's first successor, including through nested splits."""

    branch: list[str] = []
    current = daughter_id
    while True:
        branch.append(current)
        successor_edges = tuple(
            edge for edge in state.edges if edge.source_id == current
        )
        if not successor_edges:
            return tuple(branch)
        if len(successor_edges) == 1:
            current = successor_edges[0].target_id
            continue

        explicit_slot_zero = tuple(
            edge
            for edge in successor_edges
            if edge.features.get("LEGACY_SUCCESSOR_SLOT") == 0
            and type(edge.features.get("LEGACY_SUCCESSOR_SLOT")) is int
        )
        if len(explicit_slot_zero) == 1:
            current = explicit_slot_zero[0].target_id
        else:
            # Generic graphs may predate the slot feature. LineageGraphState
            # retains a deterministic edge order, which is the best available
            # compatibility fallback outside the exact legacy-state bridge.
            current = successor_edges[0].target_id


def _normalize_sources(
    frames: Mapping[str, int],
    edges: Sequence[TrackEdge],
    source_ids: set[str],
) -> tuple[TrackEdge, ...]:
    counts: dict[str, int] = {}
    for edge in edges:
        counts[edge.source_id] = counts.get(edge.source_id, 0) + 1

    successor_slots: dict[tuple[str, str], int] = {}
    for source_id in source_ids:
        source_edges = [edge for edge in edges if edge.source_id == source_id]
        raw_slots = [edge.features.get("LEGACY_SUCCESSOR_SLOT") for edge in source_edges]
        if (
            source_edges
            and all(
                type(value) is int and value in {0, 1}
                for value in raw_slots
            )
            and len(set(raw_slots)) == len(raw_slots)
        ):
            source_edges.sort(
                key=lambda edge: int(edge.features["LEGACY_SUCCESSOR_SLOT"])
            )
        for slot, edge in enumerate(source_edges):
            successor_slots[(edge.source_id, edge.target_id)] = slot

    normalized: list[TrackEdge] = []
    for edge in edges:
        if edge.source_id not in source_ids:
            normalized.append(edge)
            continue
        if counts[edge.source_id] == 2:
            kind = "split"
        elif frames[edge.target_id] - frames[edge.source_id] > 1:
            kind = "gap"
        else:
            kind = "link"
        normalized.append(
            TrackEdge(
                edge.source_id,
                edge.target_id,
                edge.cost,
                kind,
                {
                    **dict(edge.features),
                    "LEGACY_SUCCESSOR_SLOT": successor_slots[
                        (edge.source_id, edge.target_id)
                    ],
                },
            )
        )
    return tuple(normalized)


def _new_state(
    state: LineageGraphState,
    edges: Sequence[TrackEdge],
    *,
    deleted_ids: frozenset[str] | None = None,
) -> LineageGraphState:
    try:
        return LineageGraphState(
            state.frames,
            tuple(edges),
            state.deleted_ids if deleted_ids is None else deleted_ids,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, LineageResolutionError):
            raise
        raise LineageResolutionError(str(exc)) from exc


def _preserve_division(
    state: LineageGraphState, decision: BifurcationDecision
) -> LineageResolutionResult:
    return LineageResolutionResult(
        state,
        (),
        LineageResolutionDiagnostics(
            classification=1,
            parent_id=decision.parent_id,
            notes=("Legacy class 1 preserves both daughter links",),
        ),
    )


def _delete_false_positive_branch(
    state: LineageGraphState, decision: BifurcationDecision
) -> LineageResolutionResult:
    first_branch = _linear_branch(state, decision.daughter1_id)
    second_branch = _linear_branch(state, decision.daughter2_id)
    # MATLAB uses ``d1length == minsize``; an exact tie therefore deletes d1.
    if len(first_branch) <= len(second_branch):
        selected = decision.daughter1_id
        branch = first_branch
    else:
        selected = decision.daughter2_id
        branch = second_branch
    deleted = state.deleted_ids | frozenset(branch)
    edges = tuple(
        edge
        for edge in state.edges
        if edge.source_id not in deleted and edge.target_id not in deleted
    )
    edges = _normalize_sources(state.frames, edges, {decision.parent_id})
    resolved = _new_state(state, edges, deleted_ids=deleted)
    action = LineageResolutionAction(
        "delete_false_positive_branch",
        source_id=decision.parent_id,
        target_id=selected,
        node_ids=branch,
        details={
            "daughter1_length": len(first_branch),
            "daughter2_length": len(second_branch),
            "tie_break": "daughter1",
        },
    )
    return LineageResolutionResult(
        resolved,
        (action,),
        LineageResolutionDiagnostics(
            classification=3,
            parent_id=decision.parent_id,
            selected_daughter_id=selected,
            deleted_ids=tuple(sorted(branch)),
        ),
    )


def _reattach_class_zero(
    state: LineageGraphState,
    decision: BifurcationDecision,
    detached_id: str,
) -> tuple[
    LineageGraphState,
    LineageResolutionAction | None,
    str | None,
    tuple[tuple[str, str], ...],
]:
    skipped: list[tuple[str, str]] = []
    ranked = sorted(decision.reattachment_candidates, key=lambda item: item.cost)
    for candidate in ranked:
        source_id = candidate.source_id
        if source_id not in state.frames:
            skipped.append((source_id, "unknown source"))
            continue
        if source_id in state.deleted_ids:
            skipped.append((source_id, "deleted source"))
            continue
        if source_id == decision.parent_id:
            skipped.append((source_id, "original parent"))
            continue
        if state.frames[source_id] != state.frames[decision.parent_id]:
            skipped.append((source_id, "source is not in the parent frame"))
            continue
        if len(state.successors(source_id)) >= 2:
            skipped.append((source_id, "source already has two successors"))
            continue
        kind = (
            "gap"
            if state.frames[detached_id] - state.frames[source_id] > 1
            else "link"
        )
        new_edge = TrackEdge(
            source_id,
            detached_id,
            candidate.cost,
            kind,
            {
                **dict(candidate.features),
                "STARRYNITE_RESOLUTION_CLASS": 0,
                "STARRYNITE_REATTACHMENT": True,
            },
        )
        edges = (*state.edges, new_edge)
        edges = _normalize_sources(state.frames, edges, {source_id})
        try:
            resolved = _new_state(state, edges)
        except LineageResolutionError as exc:
            skipped.append((source_id, str(exc)))
            continue
        return (
            resolved,
            LineageResolutionAction(
                "reattach_detached_branch",
                source_id=source_id,
                target_id=detached_id,
                details={"cost": candidate.cost},
            ),
            source_id,
            tuple(skipped),
        )
    return state, None, None, tuple(skipped)


def _resolve_other(
    state: LineageGraphState, decision: BifurcationDecision
) -> LineageResolutionResult:
    first_score = float(decision.daughter1_nondivision_score)
    second_score = float(decision.daughter2_nondivision_score)
    # MATLAB detaches d1 only for a strict ``score1 > score2``; ties detach d2.
    detached = (
        decision.daughter1_id if first_score > second_score else decision.daughter2_id
    )
    remaining_edges = tuple(
        edge
        for edge in state.edges
        if not (edge.source_id == decision.parent_id and edge.target_id == detached)
    )
    remaining_edges = _normalize_sources(
        state.frames, remaining_edges, {decision.parent_id}
    )
    detached_state = _new_state(state, remaining_edges)
    actions: list[LineageResolutionAction] = [
        LineageResolutionAction(
            "detach_weaker_nondivision_branch",
            source_id=decision.parent_id,
            target_id=detached,
            details={
                "daughter1_score": first_score,
                "daughter2_score": second_score,
                "tie_break": "daughter2",
            },
        )
    ]
    resolved, reattach_action, reattached_to, skipped = _reattach_class_zero(
        detached_state, decision, detached
    )
    if reattach_action is not None:
        actions.append(reattach_action)
    return LineageResolutionResult(
        resolved,
        tuple(actions),
        LineageResolutionDiagnostics(
            classification=0,
            parent_id=decision.parent_id,
            detached_daughter_id=detached,
            reattached_to_id=reattached_to,
            skipped_reattachment_candidates=skipped,
        ),
    )


def _resolve_false_negative(
    state: LineageGraphState, decision: BifurcationDecision
) -> LineageResolutionResult:
    plan = decision.false_negative_plan
    if plan is None:
        raise LineageResolutionError(
            "Class 2 requires an explicit FalseNegativeRewirePlan; "
            "candidate selection is intentionally not inferred"
        )
    edge_map = {(edge.source_id, edge.target_id): edge for edge in state.edges}
    for removal in plan.remove_edges:
        if removal not in edge_map:
            raise LineageResolutionError(
                f"False-negative removal {removal[0]!r} -> {removal[1]!r} "
                "does not exist"
            )
    daughter_ids = {decision.daughter1_id, decision.daughter2_id}
    removed_daughters = {
        target
        for source, target in plan.remove_edges
        if source == decision.parent_id and target in daughter_ids
    }
    if not removed_daughters:
        raise LineageResolutionError(
            "Class-2 plan must remove at least one parent-to-daughter edge"
        )
    genuine_gaps = tuple(
        edge
        for edge in plan.add_edges
        if edge.kind == "gap"
        and edge.target_id in removed_daughters
        and edge.source_id in state.frames
        and state.frames[edge.target_id] - state.frames[edge.source_id] > 1
    )
    if not genuine_gaps:
        raise LineageResolutionError(
            "Class-2 plan must add a multi-frame gap to a removed daughter"
        )

    removals = set(plan.remove_edges)
    edges = [
        edge
        for edge in state.edges
        if (edge.source_id, edge.target_id) not in removals
    ]
    existing_pairs = {(edge.source_id, edge.target_id) for edge in edges}
    for edge in plan.add_edges:
        if edge.source_id not in state.frames or edge.target_id not in state.frames:
            raise LineageResolutionError(
                "False-negative additions must reference known nodes"
            )
        pair = (edge.source_id, edge.target_id)
        if pair in existing_pairs:
            raise LineageResolutionError(
                f"False-negative addition {pair[0]!r} -> {pair[1]!r} already exists"
            )
        existing_pairs.add(pair)
        edges.append(edge)
    affected_sources = {source for source, _target in plan.remove_edges}
    affected_sources.update(edge.source_id for edge in plan.add_edges)
    edges = list(_normalize_sources(state.frames, edges, affected_sources))
    resolved = _new_state(state, edges)

    actions = tuple(
        [
            LineageResolutionAction(
                "remove_edge_for_false_negative_rewire",
                source_id=source,
                target_id=target,
            )
            for source, target in plan.remove_edges
        ]
        + [
            LineageResolutionAction(
                "add_edge_for_false_negative_rewire",
                source_id=edge.source_id,
                target_id=edge.target_id,
                details={"kind": edge.kind, "cost": edge.cost},
            )
            for edge in plan.add_edges
        ]
    )
    return LineageResolutionResult(
        resolved,
        actions,
        LineageResolutionDiagnostics(
            classification=2,
            parent_id=decision.parent_id,
            applied_gap_edges=tuple(
                (edge.source_id, edge.target_id) for edge in genuine_gaps
            ),
            notes=((plan.label,) if plan.label else ()),
        ),
    )


def resolve_bifurcation(
    state: LineageGraphState, decision: BifurcationDecision
) -> LineageResolutionResult:
    """Apply one classifier decision without mutating ``state``.

    The input must describe an active two-daughter bifurcation.  Every returned
    graph is revalidated for forward-only edges, one predecessor, at most two
    successors, and absence of merges.
    """

    _validate_bifurcation(state, decision)
    if decision.classification == 1:
        return _preserve_division(state, decision)
    if decision.classification == 3:
        return _delete_false_positive_branch(state, decision)
    if decision.classification == 0:
        return _resolve_other(state, decision)
    return _resolve_false_negative(state, decision)


__all__ = [
    "BifurcationDecision",
    "FalseNegativeRewirePlan",
    "LineageGraphState",
    "LineageReattachmentCandidate",
    "LineageResolutionAction",
    "LineageResolutionDiagnostics",
    "LineageResolutionError",
    "LineageResolutionResult",
    "resolve_bifurcation",
]
