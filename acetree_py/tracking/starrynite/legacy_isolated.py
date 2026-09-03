"""Exact isolated-fragment prepass at ``greedydeleteFPbranches`` entry."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from ..api import TrackEdge
from .legacy_state import LegacyStateError, LegacyTrackingContext


class LegacyIsolatedFragmentError(ValueError):
    """Raised when the MATLAB isolated-fragment mutation is not representable."""


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise LegacyIsolatedFragmentError(f"{label} must be at least {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class LegacyIsolatedFragmentParameters:
    """The four controls read by the legacy isolated-fragment loop."""

    enabled: bool = False
    start_frame: int = 1
    end_frame: int | None = None
    fp_size_threshold: int = 2
    early_cell_threshold: int = 250
    fp_size_threshold_small: int = 1

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        object.__setattr__(
            self,
            "start_frame",
            _integer(self.start_frame, "start_frame", minimum=1),
        )
        if self.end_frame is not None:
            object.__setattr__(
                self,
                "end_frame",
                _integer(self.end_frame, "end_frame", minimum=1),
            )
            if self.end_frame < self.start_frame:
                raise LegacyIsolatedFragmentError(
                    "end_frame cannot precede start_frame"
                )
        for name in (
            "fp_size_threshold",
            "early_cell_threshold",
            "fp_size_threshold_small",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name),
            )

    def resolved_end_frame(self, context: LegacyTrackingContext) -> int:
        result = (
            context.parameters.end_frame
            if self.end_frame is None
            else self.end_frame
        )
        if result > context.parameters.end_frame:
            raise LegacyIsolatedFragmentError(
                "end_frame cannot exceed the legacy context end frame"
            )
        return result


@dataclass(frozen=True, slots=True)
class LegacyIsolatedFragmentDeletion:
    """One deletion decision in MATLAB frame/row scan order."""

    source_id: str
    branch_ids: tuple[str, ...]
    shifted_successor_id: str | None
    totally_isolated: bool

    def __post_init__(self) -> None:
        if type(self.source_id) is not str or not self.source_id:
            raise TypeError("source_id must be a non-empty string")
        branch = tuple(self.branch_ids)
        if not branch or len(branch) != len(set(branch)):
            raise LegacyIsolatedFragmentError(
                "branch_ids must contain unique nucleus IDs"
            )
        if self.shifted_successor_id is not None and (
            type(self.shifted_successor_id) is not str
            or not self.shifted_successor_id
        ):
            raise TypeError("shifted_successor_id must be non-empty text or None")
        if type(self.totally_isolated) is not bool:
            raise TypeError("totally_isolated must be a boolean")
        object.__setattr__(self, "branch_ids", branch)


@dataclass(frozen=True, slots=True)
class LegacyIsolatedFragmentResult:
    """Raw context plus ordered decisions produced by the prepass."""

    context: LegacyTrackingContext
    deletions: tuple[LegacyIsolatedFragmentDeletion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, LegacyTrackingContext):
            raise TypeError("context must be a LegacyTrackingContext")
        values = tuple(self.deletions)
        if any(not isinstance(item, LegacyIsolatedFragmentDeletion) for item in values):
            raise TypeError(
                "deletions must contain LegacyIsolatedFragmentDeletion values"
            )
        object.__setattr__(self, "deletions", values)


def _edge_kind(
    context: LegacyTrackingContext,
    source_id: str,
    target_id: str,
    successor_count: int,
) -> str:
    if successor_count == 2:
        return "split"
    return (
        "gap"
        if context.nucleus(target_id).frame - context.nucleus(source_id).frame > 1
        else "link"
    )


def _traverse_slot_zero(
    slots: dict[str, list[str | None]],
    start_id: str,
) -> tuple[str, ...]:
    result: list[str] = []
    visited: set[str] = set()
    current: str | None = start_id
    while current is not None:
        if current in visited:
            raise LegacyIsolatedFragmentError(
                "Cycle encountered in an isolated-fragment successor path"
            )
        visited.add(current)
        result.append(current)
        current = slots[current][0]
    return tuple(result)


def _rebuild_context(
    context: LegacyTrackingContext,
    slots: dict[str, list[str | None]],
    deleted: set[str],
    stale_predecessors: dict[str, str],
) -> LegacyTrackingContext:
    templates = {(edge.source_id, edge.target_id): edge for edge in context.edges}
    edges: list[TrackEdge] = []
    for nucleus in context.nuclei:
        source_id = nucleus.nucleus_id
        targets = tuple(item for item in slots[source_id] if item is not None)
        if slots[source_id][0] is None and slots[source_id][1] is not None:
            raise LegacyIsolatedFragmentError(
                f"Nucleus {source_id!r} has successor slot 1 without slot 0"
            )
        for slot, target_id in enumerate(targets):
            template = templates.get((source_id, target_id))
            if template is None:
                raise LegacyIsolatedFragmentError(
                    f"No raw edge template for {source_id!r} -> {target_id!r}"
                )
            features = dict(template.features)
            features["LEGACY_SUCCESSOR_SLOT"] = slot
            edges.append(
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
            tuple(edges),
            context.parameters,
            deleted_ids=deleted,
            stale_predecessor_by_id=stale_predecessors,
        )
    except (LegacyStateError, TypeError, ValueError) as exc:
        raise LegacyIsolatedFragmentError(str(exc)) from exc


def apply_legacy_isolated_fragment_prepass(
    context: LegacyTrackingContext,
    parameters: LegacyIsolatedFragmentParameters,
) -> LegacyIsolatedFragmentResult:
    """Apply the first loop of MATLAB ``greedydeleteFPbranches`` exactly.

    MATLAB clears or shifts the source successor slot but leaves the deleted
    branch root's predecessor pointer untouched.  The returned context keeps
    that one-sided pointer in ``stale_predecessor_by_id`` so raw checkpoints
    remain byte-for-byte topologically comparable while the active graph
    projection continues to use reciprocal :class:`TrackEdge` values only.
    """

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(parameters, LegacyIsolatedFragmentParameters):
        raise TypeError("parameters must be LegacyIsolatedFragmentParameters")
    if not parameters.enabled:
        return LegacyIsolatedFragmentResult(context, ())

    end_frame = parameters.resolved_end_frame(context)
    ids = tuple(item.nucleus_id for item in context.nuclei)
    slots = {
        nucleus_id: list(context.successor_slots(nucleus_id))
        for nucleus_id in ids
    }
    predecessor = {
        nucleus_id: context.predecessor(nucleus_id) for nucleus_id in ids
    }
    deleted = set(context.deleted_ids)
    stale = dict(context.stale_predecessor_by_id)
    decisions: list[LegacyIsolatedFragmentDeletion] = []

    for frame in range(parameters.start_frame, end_frame):
        frame_count = len(context.frame_ids(frame, include_deleted=True))
        for source_id in context.frame_ids(frame, include_deleted=True):
            if source_id in deleted or predecessor[source_id] is not None:
                continue
            first_id, second_id = slots[source_id]
            if first_id is None:
                deleted.add(source_id)
                decisions.append(
                    LegacyIsolatedFragmentDeletion(
                        source_id,
                        (source_id,),
                        None,
                        True,
                    )
                )
                continue

            branch = _traverse_slot_zero(slots, first_id)
            should_delete = (
                len(branch) < parameters.fp_size_threshold
                and frame_count <= parameters.early_cell_threshold
            ) or len(branch) <= parameters.fp_size_threshold_small
            if not should_delete:
                continue
            deleted.update(branch)
            # Deliberately leave predecessor[first_id] untouched, reproducing
            # the raw one-sided pointer retained by the MATLAB implementation.
            stale[first_id] = source_id
            shifted: str | None = None
            if second_id is None:
                slots[source_id] = [None, None]
                deleted.add(source_id)
            else:
                shifted = second_id
                slots[source_id] = [second_id, None]
            decisions.append(
                LegacyIsolatedFragmentDeletion(
                    source_id,
                    branch,
                    shifted,
                    False,
                )
            )

    rebuilt = _rebuild_context(context, slots, deleted, stale)
    return LegacyIsolatedFragmentResult(rebuilt, tuple(decisions))


__all__ = [
    "LegacyIsolatedFragmentDeletion",
    "LegacyIsolatedFragmentError",
    "LegacyIsolatedFragmentParameters",
    "LegacyIsolatedFragmentResult",
    "apply_legacy_isolated_fragment_prepass",
]
