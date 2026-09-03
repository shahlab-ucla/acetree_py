"""Raw-pointer graph mutations shared by legacy StarryNite orchestrators."""

from __future__ import annotations

from dataclasses import dataclass

from ..api import TrackEdge
from .legacy_state import LegacyStateError, LegacyTrackingContext
from .lineage import (
    LineageGraphState,
    LineageResolutionAction,
    LineageResolutionDiagnostics,
    LineageResolutionError,
)


class LegacyMutationError(ValueError):
    """Raised when a raw MATLAB mutation cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class LegacyFalsePositiveResolution:
    """Raw context and active projection after ``processFPBifurcation``."""

    context: LegacyTrackingContext
    state: LineageGraphState
    actions: tuple[LineageResolutionAction, ...]
    diagnostics: LineageResolutionDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.context, LegacyTrackingContext):
            raise TypeError("context must be a LegacyTrackingContext")
        if not isinstance(self.state, LineageGraphState):
            raise TypeError("state must be a LineageGraphState")
        if self.context.to_lineage_graph_state() != self.state:
            raise LegacyMutationError(
                "Raw false-positive context and active projection disagree"
            )
        if any(not isinstance(item, LineageResolutionAction) for item in self.actions):
            raise TypeError("actions must contain LineageResolutionAction values")
        if not isinstance(self.diagnostics, LineageResolutionDiagnostics):
            raise TypeError("diagnostics must be LineageResolutionDiagnostics")
        object.__setattr__(self, "actions", tuple(self.actions))


def resolve_legacy_false_positive_bifurcation(
    context: LegacyTrackingContext,
    parent_id: str,
    daughter_ids: tuple[str, str],
) -> LegacyFalsePositiveResolution:
    """Apply class 3 by following raw successor slot zero exactly.

    The active graph hides links incident to deleted rows. MATLAB does not:
    ``traverse_forward`` and ``processFPBifurcation`` follow stored successor
    slot 1 even through rows whose delete flag was already set.  Every outgoing
    pointer on the condemned path is retained; only the path root's incoming
    parent link is cleared and the surviving daughter shifts into slot 1.
    """

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if type(parent_id) is not str or not parent_id:
        raise TypeError("parent_id must be a non-empty string")
    daughters = tuple(daughter_ids)
    if len(daughters) != 2 or len(set(daughters)) != 2:
        raise LegacyMutationError("daughter_ids must contain two distinct IDs")
    first_id, second_id = daughters
    if context.successor_slots(parent_id) != daughters:
        raise LegacyMutationError(
            "False-positive daughters no longer match raw successor slots"
        )

    first_branch = context.traverse_forward(first_id)
    second_branch = context.traverse_forward(second_id)
    # MATLAB: ``d1length == minsize`` deletes daughter 1 on an exact tie.
    if len(first_branch) <= len(second_branch):
        selected_id = first_id
        retained_id = second_id
        selected_branch = first_branch
    else:
        selected_id = second_id
        retained_id = first_id
        selected_branch = second_branch

    rebuilt: list[TrackEdge] = []
    for edge in context.edges:
        if edge.source_id != parent_id:
            rebuilt.append(edge)
            continue
        if edge.target_id == selected_id:
            continue
        if edge.target_id != retained_id:
            raise LegacyMutationError(
                "False-positive parent has an unexpected raw successor"
            )
        features = dict(edge.features)
        features["LEGACY_SUCCESSOR_SLOT"] = 0
        frame_gap = (
            context.nucleus(retained_id).frame
            - context.nucleus(parent_id).frame
        )
        rebuilt.append(
            TrackEdge(
                parent_id,
                retained_id,
                edge.cost,
                "gap" if frame_gap > 1 else "link",
                features,
            )
        )
    try:
        resolved_context = LegacyTrackingContext.from_nuclei_and_edges(
            context.nuclei,
            tuple(rebuilt),
            context.parameters,
            deleted_ids=context.deleted_ids | frozenset(selected_branch),
            stale_predecessor_by_id=context.stale_predecessor_by_id,
        )
        resolved_state = resolved_context.to_lineage_graph_state()
    except (LegacyStateError, LineageResolutionError, TypeError, ValueError) as exc:
        raise LegacyMutationError(str(exc)) from exc

    action = LineageResolutionAction(
        "delete_false_positive_branch",
        source_id=parent_id,
        target_id=selected_id,
        node_ids=selected_branch,
        details={
            "daughter1_length": len(first_branch),
            "daughter2_length": len(second_branch),
            "tie_break": "daughter1",
            "legacy_raw_successor_path": True,
        },
    )
    return LegacyFalsePositiveResolution(
        context=resolved_context,
        state=resolved_state,
        actions=(action,),
        diagnostics=LineageResolutionDiagnostics(
            classification=3,
            parent_id=parent_id,
            selected_daughter_id=selected_id,
            deleted_ids=tuple(selected_branch),
            notes=("Deletion followed raw legacy successor slot zero",),
        ),
    )


__all__ = [
    "LegacyFalsePositiveResolution",
    "LegacyMutationError",
    "resolve_legacy_false_positive_bifurcation",
]
