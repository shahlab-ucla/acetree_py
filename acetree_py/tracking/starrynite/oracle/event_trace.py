"""Ordered full-movie StarryNite classification and mutation parity.

Final lineage snapshots can agree even when the two trackers reached them in a
different order.  That distinction matters for StarryNite: class-0 handling
can mutate the graph, classify a newly created bifurcation in round two, and
then resume the outer repair.  This module represents those classifier
checkpoints and derives deterministic graph-mutation batches between them.

MATLAB node references are normalized as ``matlab:<frame0>:<row0>``.  Python
traces may retain their native detection IDs; ``compare_event_traces`` accepts
the spatial candidate-to-reference mapping produced by the lineage comparator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..legacy_class_zero import LegacyClassZeroRepairResult
    from ..legacy_driver import (
        LegacyMovieClassificationRecord,
        LegacyMovieClassificationObservation,
        LegacyMovieDecisionResult,
    )
    from ..classifier import AmbigiousClassifierPrediction
    from ..legacy_state import LegacyTrackingContext
    from ..lineage import LineageGraphState


EVENT_TRACE_SCHEMA = "acetree.starrynite.event-trace/v1"

_CHECKPOINT_PHASES = {"initial", "pre_classification", "final"}
_MUTATION_KINDS = {
    "node_deleted",
    "node_restored",
    "predecessor_removed",
    "predecessor_set",
    "edge_removed",
    "edge_added",
}
_CLASSIFIER_FAMILIES = {"single_model", "ambigious_multi_model"}


class EventTraceFormatError(ValueError):
    """Raised when an event trace is incomplete or internally inconsistent."""


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _node_id(value: Any, label: str) -> str:
    result = str(value)
    if not result:
        raise EventTraceFormatError(f"{label} cannot be empty")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise EventTraceFormatError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise EventTraceFormatError(f"{label} must be >= {minimum}")
    return result


def _class_value(value: Any, label: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    result = _integer(value, label)
    if result not in {0, 1, 2, 3}:
        raise EventTraceFormatError(f"{label} must be 0, 1, 2, or 3")
    return result


@dataclass(frozen=True, slots=True)
class LegacyNodePointerState:
    """One row of MATLAB's mutable ``pred``/``suc``/``delete`` state."""

    node_id: str
    frame_0based: int
    row_0based: int
    deleted: bool
    predecessor_id: str | None
    successor_slots: tuple[str | None, str | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _node_id(self.node_id, "node_id"))
        object.__setattr__(
            self,
            "frame_0based",
            _integer(self.frame_0based, "frame_0based"),
        )
        object.__setattr__(
            self,
            "row_0based",
            _integer(self.row_0based, "row_0based"),
        )
        if type(self.deleted) is not bool:
            raise EventTraceFormatError("deleted must be a boolean")
        if self.predecessor_id is not None:
            object.__setattr__(
                self,
                "predecessor_id",
                _node_id(self.predecessor_id, "predecessor_id"),
            )
        slots = tuple(self.successor_slots)
        if len(slots) != 2:
            raise EventTraceFormatError("successor_slots must contain two values")
        object.__setattr__(
            self,
            "successor_slots",
            tuple(
                None if value is None else _node_id(value, "successor_id")
                for value in slots
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "frame_0based": self.frame_0based,
            "row_0based": self.row_0based,
            "deleted": self.deleted,
            "predecessor": self.predecessor_id,
            "successor_slots": list(self.successor_slots),
        }


@dataclass(frozen=True, slots=True)
class TrackingCheckpoint:
    """Complete low-level graph state at one classifier boundary."""

    index: int
    phase: str
    nodes: tuple[LegacyNodePointerState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _integer(self.index, "checkpoint index"))
        if self.phase not in _CHECKPOINT_PHASES:
            raise EventTraceFormatError(
                "checkpoint phase must be initial, pre_classification, or final"
            )
        nodes = tuple(self.nodes)
        if any(not isinstance(item, LegacyNodePointerState) for item in nodes):
            raise EventTraceFormatError(
                "checkpoint nodes must be LegacyNodePointerState values"
            )
        identifiers = [item.node_id for item in nodes]
        if len(identifiers) != len(set(identifiers)):
            raise EventTraceFormatError("checkpoint node IDs must be unique")
        order = [(item.frame_0based, item.row_0based) for item in nodes]
        if len(order) != len(set(order)):
            raise EventTraceFormatError(
                "checkpoint frame/row references must be unique"
            )
        object.__setattr__(self, "nodes", nodes)

    @property
    def by_id(self) -> Mapping[str, LegacyNodePointerState]:
        return MappingProxyType({item.node_id: item for item in self.nodes})

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "phase": self.phase,
            "nodes": [item.to_dict() for item in self.nodes],
        }


@dataclass(frozen=True, slots=True)
class TrackingMutation:
    """One normalized low-level change inside a checkpoint interval."""

    kind: str
    node_id: str
    related_id: str | None = None
    slot: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _MUTATION_KINDS:
            raise EventTraceFormatError(f"Unsupported mutation kind: {self.kind!r}")
        object.__setattr__(self, "node_id", _node_id(self.node_id, "mutation node_id"))
        if self.kind in {"node_deleted", "node_restored"}:
            if self.related_id is not None or self.slot is not None:
                raise EventTraceFormatError(
                    f"{self.kind} cannot carry related_id or slot"
                )
            return
        if self.related_id is None:
            raise EventTraceFormatError(f"{self.kind} requires related_id")
        object.__setattr__(
            self,
            "related_id",
            _node_id(self.related_id, "mutation related_id"),
        )
        if self.kind in {"edge_removed", "edge_added"}:
            if type(self.slot) is not int or self.slot not in {0, 1}:
                raise EventTraceFormatError(f"{self.kind} requires slot 0 or 1")
        elif self.slot is not None:
            raise EventTraceFormatError(f"{self.kind} cannot carry a slot")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "node": self.node_id}
        if self.related_id is not None:
            result["related"] = self.related_id
        if self.slot is not None:
            result["slot"] = self.slot
        return result


@dataclass(frozen=True, slots=True)
class MutationBatch:
    """All state changes since the preceding classifier checkpoint."""

    checkpoint_index: int
    next_classification_index: int | None
    mutations: tuple[TrackingMutation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_index",
            _integer(self.checkpoint_index, "mutation checkpoint index", minimum=1),
        )
        if self.next_classification_index is not None:
            object.__setattr__(
                self,
                "next_classification_index",
                _integer(
                    self.next_classification_index,
                    "next classification index",
                ),
            )
        mutations = tuple(self.mutations)
        if any(not isinstance(item, TrackingMutation) for item in mutations):
            raise EventTraceFormatError(
                "mutation batches must contain TrackingMutation values"
            )
        object.__setattr__(self, "mutations", mutations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "mutation_batch",
            "checkpoint_index": self.checkpoint_index,
            "next_classification_index": self.next_classification_index,
            "mutations": [item.to_dict() for item in self.mutations],
        }


@dataclass(frozen=True, slots=True)
class ClassificationEvent:
    """One classifier invocation in actual movie execution order."""

    event_index: int
    checkpoint_index: int
    parent_id: str
    daughter_ids: tuple[str, str]
    classifier_round: int
    computed_class: int | None
    effective_class: int
    force_mode: bool
    classifier_family: str = "single_model"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_index",
            _integer(self.event_index, "classification event index"),
        )
        object.__setattr__(
            self,
            "checkpoint_index",
            _integer(self.checkpoint_index, "classification checkpoint", minimum=1),
        )
        object.__setattr__(
            self,
            "parent_id",
            _node_id(self.parent_id, "classification parent_id"),
        )
        daughters = tuple(self.daughter_ids)
        if len(daughters) != 2:
            raise EventTraceFormatError("classification requires two daughter IDs")
        daughters = tuple(
            _node_id(value, "classification daughter_id") for value in daughters
        )
        if self.parent_id in daughters or daughters[0] == daughters[1]:
            raise EventTraceFormatError(
                "classification parent and daughters must be distinct"
            )
        object.__setattr__(self, "daughter_ids", daughters)
        round_value = _integer(self.classifier_round, "classifier_round", minimum=1)
        if round_value not in {1, 2}:
            raise EventTraceFormatError("classifier_round must be 1 or 2")
        object.__setattr__(self, "classifier_round", round_value)
        object.__setattr__(
            self,
            "computed_class",
            _class_value(self.computed_class, "computed_class", optional=True),
        )
        effective = _class_value(self.effective_class, "effective_class")
        if effective is None:  # pragma: no cover - guarded by optional=False
            raise AssertionError("effective class unexpectedly missing")
        object.__setattr__(self, "effective_class", effective)
        if type(self.force_mode) is not bool:
            raise EventTraceFormatError("force_mode must be a boolean")
        if self.classifier_family not in _CLASSIFIER_FAMILIES:
            raise EventTraceFormatError(
                "classifier_family must be single_model or ambigious_multi_model"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "classification",
            "event_index": self.event_index,
            "checkpoint_index": self.checkpoint_index,
            "parent": self.parent_id,
            "daughters": list(self.daughter_ids),
            "classifier_round": self.classifier_round,
            "computed_class": self.computed_class,
            "effective_class": self.effective_class,
            "force_mode": self.force_mode,
            "classifier_family": self.classifier_family,
        }


TrackingEvent = MutationBatch | ClassificationEvent


@dataclass(frozen=True, slots=True)
class TrackingEventTrace:
    """A full movie's ordered classifier checkpoints and pointer snapshots."""

    checkpoints: tuple[TrackingCheckpoint, ...]
    classifications: tuple[ClassificationEvent, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        checkpoints = tuple(self.checkpoints)
        classifications = tuple(self.classifications)
        if len(checkpoints) != len(classifications) + 2:
            raise EventTraceFormatError(
                "event traces require initial, one pre-classification checkpoint "
                "per event, and final"
            )
        if any(not isinstance(item, TrackingCheckpoint) for item in checkpoints):
            raise EventTraceFormatError(
                "checkpoints must contain TrackingCheckpoint values"
            )
        if any(
            not isinstance(item, ClassificationEvent) for item in classifications
        ):
            raise EventTraceFormatError(
                "classifications must contain ClassificationEvent values"
            )
        if tuple(item.index for item in checkpoints) != tuple(range(len(checkpoints))):
            raise EventTraceFormatError("checkpoint indices must be contiguous")
        expected_phases = (
            "initial",
            *("pre_classification" for _ in classifications),
            "final",
        )
        if tuple(item.phase for item in checkpoints) != expected_phases:
            raise EventTraceFormatError("checkpoint phases are out of order")
        if tuple(item.event_index for item in classifications) != tuple(
            range(len(classifications))
        ):
            raise EventTraceFormatError(
                "classification event indices must be contiguous"
            )
        if tuple(item.checkpoint_index for item in classifications) != tuple(
            range(1, len(classifications) + 1)
        ):
            raise EventTraceFormatError(
                "each classification must reference its pre-classification checkpoint"
            )

        first_identity = {
            item.node_id: (item.frame_0based, item.row_0based)
            for item in checkpoints[0].nodes
        }
        for checkpoint in checkpoints[1:]:
            identity = {
                item.node_id: (item.frame_0based, item.row_0based)
                for item in checkpoint.nodes
            }
            if identity != first_identity:
                raise EventTraceFormatError(
                    "node identity/frame/row must remain fixed across checkpoints"
                )
        for event, checkpoint in zip(
            classifications,
            checkpoints[1:-1],
            strict=True,
        ):
            by_id = checkpoint.by_id
            if event.parent_id not in by_id or any(
                daughter not in by_id for daughter in event.daughter_ids
            ):
                raise EventTraceFormatError(
                    "classification references a node absent from its checkpoint"
                )
            parent = by_id[event.parent_id]
            if parent.successor_slots != event.daughter_ids or (
                event.classifier_round == 1 and parent.deleted
            ):
                raise EventTraceFormatError(
                    "classification parent is not the permitted raw ordered "
                    "bifurcation recorded at its checkpoint"
                )

        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "classifications", classifications)
        object.__setattr__(self, "provenance", _immutable_mapping(self.provenance))

    @property
    def ordered_events(self) -> tuple[TrackingEvent, ...]:
        events: list[TrackingEvent] = []
        for event_index, classification in enumerate(self.classifications):
            events.append(
                MutationBatch(
                    checkpoint_index=event_index + 1,
                    next_classification_index=event_index,
                    mutations=_checkpoint_mutations(
                        self.checkpoints[event_index],
                        self.checkpoints[event_index + 1],
                    ),
                )
            )
            events.append(classification)
        events.append(
            MutationBatch(
                checkpoint_index=len(self.checkpoints) - 1,
                next_classification_index=None,
                mutations=_checkpoint_mutations(
                    self.checkpoints[-2],
                    self.checkpoints[-1],
                ),
            )
        )
        return tuple(events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVENT_TRACE_SCHEMA,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "classifications": [item.to_dict() for item in self.classifications],
            "ordered_events": [item.to_dict() for item in self.ordered_events],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrackingEventTrace:
        if value.get("schema") != EVENT_TRACE_SCHEMA:
            raise EventTraceFormatError("Unsupported StarryNite event-trace schema")
        raw_checkpoints = value.get("checkpoints", ())
        raw_classifications = value.get("classifications", ())
        if not isinstance(raw_checkpoints, Sequence) or isinstance(
            raw_checkpoints, (str, bytes)
        ):
            raise EventTraceFormatError("checkpoints must be a sequence")
        if not isinstance(raw_classifications, Sequence) or isinstance(
            raw_classifications, (str, bytes)
        ):
            raise EventTraceFormatError("classifications must be a sequence")
        checkpoints = tuple(
            _checkpoint_from_dict(item) for item in raw_checkpoints
        )
        classifications = tuple(
            _classification_from_dict(item) for item in raw_classifications
        )
        provenance = value.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise EventTraceFormatError("event-trace provenance must be a mapping")
        return cls(checkpoints, classifications, provenance)


@dataclass(frozen=True, slots=True)
class EventTraceComparison:
    """Classifier-order and checkpoint-delta parity diagnostics."""

    exact_match: bool
    reference_event_count: int
    candidate_event_count: int
    matching_prefix_count: int
    first_divergence_index: int | None
    classification_order_match: bool
    mutation_batch_match: bool
    reference_event: Mapping[str, Any] | None = None
    candidate_event: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "reference_event_count",
            "candidate_event_count",
            "matching_prefix_count",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.first_divergence_index is not None:
            object.__setattr__(
                self,
                "first_divergence_index",
                _integer(self.first_divergence_index, "first_divergence_index"),
            )
        if type(self.exact_match) is not bool:
            raise EventTraceFormatError("exact_match must be a boolean")
        if type(self.classification_order_match) is not bool:
            raise EventTraceFormatError(
                "classification_order_match must be a boolean"
            )
        if type(self.mutation_batch_match) is not bool:
            raise EventTraceFormatError("mutation_batch_match must be a boolean")
        if self.reference_event is not None:
            object.__setattr__(
                self,
                "reference_event",
                _immutable_mapping(self.reference_event),
            )
        if self.candidate_event is not None:
            object.__setattr__(
                self,
                "candidate_event",
                _immutable_mapping(self.candidate_event),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "reference_event_count": self.reference_event_count,
            "candidate_event_count": self.candidate_event_count,
            "matching_prefix_count": self.matching_prefix_count,
            "first_divergence_index": self.first_divergence_index,
            "classification_order_match": self.classification_order_match,
            "mutation_batch_match": self.mutation_batch_match,
            "reference_event": (
                None if self.reference_event is None else dict(self.reference_event)
            ),
            "candidate_event": (
                None if self.candidate_event is None else dict(self.candidate_event)
            ),
        }

    @property
    def mutation_order_match(self) -> bool:
        """Compatibility alias for the formerly over-strong field name.

        This value has always compared normalized mutation *batches* derived
        from checkpoint snapshots; it does not establish intra-batch write
        order.  New callers should use :attr:`mutation_batch_match`.
        """

        return self.mutation_batch_match


def compare_event_traces(
    reference: TrackingEventTrace,
    candidate: TrackingEventTrace,
    *,
    candidate_to_reference: Mapping[str, str] | None = None,
) -> EventTraceComparison:
    """Compare classifier order and normalized checkpoint-delta batches.

    A checkpoint records only the state before each classifier call and at the
    end of the pass.  Consequently, mutation comparison is exact for the net
    pointer delta in each interval, but cannot certify the execution order of
    individual writes inside one interval.
    """

    if not isinstance(reference, TrackingEventTrace):
        raise TypeError("reference must be a TrackingEventTrace")
    if not isinstance(candidate, TrackingEventTrace):
        raise TypeError("candidate must be a TrackingEventTrace")
    mapping = None
    if candidate_to_reference is not None:
        mapping = {
            _node_id(source, "candidate mapping key"): _node_id(
                target, "candidate mapping value"
            )
            for source, target in candidate_to_reference.items()
        }
    reference_events = reference.ordered_events
    candidate_events = candidate.ordered_events
    reference_keys = tuple(_event_key(item, None) for item in reference_events)
    candidate_keys = tuple(_event_key(item, mapping) for item in candidate_events)
    prefix = 0
    for first, second in zip(reference_keys, candidate_keys):
        if first != second:
            break
        prefix += 1
    exact = reference_keys == candidate_keys
    divergence = None if exact else prefix
    reference_event = (
        None
        if divergence is None or divergence >= len(reference_events)
        else _event_dict(reference_events[divergence], None)
    )
    candidate_event = (
        None
        if divergence is None or divergence >= len(candidate_events)
        else _event_dict(candidate_events[divergence], mapping)
    )
    reference_classifications = tuple(
        _event_key(item, None)
        for item in reference_events
        if isinstance(item, ClassificationEvent)
    )
    candidate_classifications = tuple(
        _event_key(item, mapping)
        for item in candidate_events
        if isinstance(item, ClassificationEvent)
    )
    reference_mutations = tuple(
        _event_key(item, None)
        for item in reference_events
        if isinstance(item, MutationBatch)
    )
    candidate_mutations = tuple(
        _event_key(item, mapping)
        for item in candidate_events
        if isinstance(item, MutationBatch)
    )
    return EventTraceComparison(
        exact_match=exact,
        reference_event_count=len(reference_events),
        candidate_event_count=len(candidate_events),
        matching_prefix_count=prefix,
        first_divergence_index=divergence,
        classification_order_match=(
            reference_classifications == candidate_classifications
        ),
        mutation_batch_match=(reference_mutations == candidate_mutations),
        reference_event=reference_event,
        candidate_event=candidate_event,
    )


def assert_event_trace_parity(
    reference: TrackingEventTrace,
    candidate: TrackingEventTrace,
    *,
    candidate_to_reference: Mapping[str, str] | None = None,
) -> None:
    """Raise at the first classifier or checkpoint-delta divergence."""

    comparison = compare_event_traces(
        reference,
        candidate,
        candidate_to_reference=candidate_to_reference,
    )
    if comparison.exact_match:
        return
    raise AssertionError(
        "StarryNite event traces diverge at ordered checkpoint event "
        f"{comparison.first_divergence_index}: "
        f"MATLAB={comparison.reference_event!r}, "
        f"Python={comparison.candidate_event!r}"
    )


def matlab_event_trace(value: Mapping[str, Any] | Any) -> TrackingEventTrace:
    """Decode the inert trace emitted by the MATLAB full-tracking oracle."""

    outer = value.result if hasattr(value, "result") else value
    if not isinstance(outer, Mapping):
        raise TypeError("MATLAB event trace source must be a mapping or oracle run")
    raw = outer.get("tracking_event_trace", outer)
    if not isinstance(raw, Mapping):
        raise EventTraceFormatError("tracking_event_trace must be a structure")
    version = _numeric_integer(raw.get("schema_version"), "schema_version")
    if version != 1:
        raise EventTraceFormatError(
            f"Unsupported MATLAB tracking event trace version: {version}"
        )
    classification_table = _numeric_table(
        raw.get("classification_table", ()),
        14,
        "classification_table",
    )
    snapshot_count = _numeric_integer(raw.get("snapshot_count"), "snapshot_count")
    snapshots = _snapshot_tables(raw.get("snapshots", ()), snapshot_count)
    if snapshot_count != len(classification_table) + 2:
        raise EventTraceFormatError(
            "MATLAB snapshot count must equal classification count plus two"
        )
    checkpoints: list[TrackingCheckpoint] = []
    for index, table in enumerate(snapshots):
        if index == 0:
            phase = "initial"
        elif index == snapshot_count - 1:
            phase = "final"
        else:
            phase = "pre_classification"
        checkpoints.append(
            TrackingCheckpoint(
                index=index,
                phase=phase,
                nodes=tuple(_matlab_snapshot_node(row) for row in table),
            )
        )

    families = {0: "single_model", 1: "ambigious_multi_model"}
    classifications: list[ClassificationEvent] = []
    for row_index, row in enumerate(classification_table):
        event_index = _float_integer(row[0], "classification event index")
        checkpoint_index = _float_integer(row[1], "classification checkpoint")
        if event_index != row_index or checkpoint_index != row_index + 1:
            raise EventTraceFormatError(
                "MATLAB classification indices are not contiguous"
            )
        family_code = _float_integer(row[13], "classifier family code")
        if family_code not in families:
            raise EventTraceFormatError(
                f"Unknown MATLAB classifier family code: {family_code}"
            )
        predicted = _float_class(row[11], "predicted class")
        effective = _float_class(row[10], "effective class")
        if predicted != effective:
            raise EventTraceFormatError(
                "MATLAB trace predicted and effective classes disagree"
            )
        computed = (
            None
            if math.isnan(float(row[9]))
            else _float_class(row[9], "computed class")
        )
        force_raw = _float_integer(row[12], "force mode")
        if force_raw not in {0, 1}:
            raise EventTraceFormatError("MATLAB force mode must be 0 or 1")
        classifications.append(
            ClassificationEvent(
                event_index=event_index,
                checkpoint_index=checkpoint_index,
                parent_id=_matlab_reference(row[2], row[3], "parent"),
                daughter_ids=(
                    _matlab_reference(row[4], row[5], "daughter 1"),
                    _matlab_reference(row[6], row[7], "daughter 2"),
                ),
                classifier_round=_float_integer(row[8], "classifier round"),
                computed_class=computed,
                effective_class=effective,
                force_mode=bool(force_raw),
                classifier_family=families[family_code],
            )
        )
    return TrackingEventTrace(
        checkpoints=tuple(checkpoints),
        classifications=tuple(classifications),
        provenance={
            "source": "matlab_oracle",
            "trace_schema_version": version,
        },
    )


def trace_from_lineage_results(
    initial_state: LineageGraphState,
    results: Sequence[Any],
    *,
    classifier_rounds: Sequence[int] | None = None,
    computed_classes: Sequence[int | None] | None = None,
    force_modes: Sequence[bool] | None = None,
    classifier_families: Sequence[str] | None = None,
) -> TrackingEventTrace:
    """Build the Python trace from results yielded in actual movie order.

    A full-movie orchestrator should append results immediately after each
    classifier/resolver call, including recursive round-two calls.  This
    function deliberately does not sort them; caller order is the behavior
    under test.
    """

    from ..classifier import (
        AmbigiousClassifierPrediction,
        SingleModelPrediction,
    )
    from ..lineage import (
        BifurcationDecision,
        LineageGraphState,
        LineageResolutionResult,
    )

    if not isinstance(initial_state, LineageGraphState):
        raise TypeError("initial_state must be a LineageGraphState")
    ordered = tuple(results)
    for item in ordered:
        if not isinstance(getattr(item, "prediction", None), (
            SingleModelPrediction,
            AmbigiousClassifierPrediction,
        )):
            raise TypeError(
                "results must carry a single- or ambigious-model prediction"
            )
        if not isinstance(getattr(item, "decision", None), BifurcationDecision):
            raise TypeError("results must carry a BifurcationDecision")
        if not isinstance(
            getattr(item, "resolution", None), LineageResolutionResult
        ):
            raise TypeError("results must carry a LineageResolutionResult")
    rounds = _per_event_values(
        classifier_rounds,
        len(ordered),
        1,
        "classifier_rounds",
    )
    computed = _per_event_values(
        computed_classes,
        len(ordered),
        None,
        "computed_classes",
    )
    forces = _per_event_values(force_modes, len(ordered), False, "force_modes")
    families = _per_event_values(
        classifier_families,
        len(ordered),
        None,
        "classifier_families",
    )
    computed_was_overridden = computed_classes is not None
    families_were_overridden = classifier_families is not None

    checkpoints: list[TrackingCheckpoint] = [
        _lineage_checkpoint(initial_state, 0, "initial")
    ]
    classifications: list[ClassificationEvent] = []
    state = initial_state
    for index, result in enumerate(ordered):
        checkpoint = _lineage_checkpoint(
            state,
            index + 1,
            "pre_classification",
        )
        checkpoints.append(checkpoint)
        decision = result.decision
        try:
            parent = checkpoint.by_id[decision.parent_id]
        except KeyError as exc:
            raise EventTraceFormatError(
                f"Python event {index} parent is not active in the prior state"
            ) from exc
        if parent.deleted:
            raise EventTraceFormatError(
                f"Python event {index} parent is not active in the prior state"
            )
        actual_daughters = parent.successor_slots
        if actual_daughters != (decision.daughter1_id, decision.daughter2_id):
            if set(actual_daughters) != {
                decision.daughter1_id,
                decision.daughter2_id,
            }:
                raise EventTraceFormatError(
                    f"Python event {index} does not classify the prior-state split"
                )
            # LineageGraphState preserves explicit legacy slots when available.
            # A remaining reversal here is behaviorally significant.
            raise EventTraceFormatError(
                f"Python event {index} daughter slot order differs from its state"
            )
        effective = result.prediction.predicted_class
        if computed_was_overridden:
            event_computed = computed[index]
        elif isinstance(result.prediction, AmbigiousClassifierPrediction):
            event_computed = result.prediction.computed_class
        else:
            event_computed = effective
        if families_were_overridden:
            classifier_family = families[index]
        elif isinstance(result.prediction, AmbigiousClassifierPrediction):
            classifier_family = "ambigious_multi_model"
        else:
            classifier_family = "single_model"
        classifications.append(
            ClassificationEvent(
                event_index=index,
                checkpoint_index=index + 1,
                parent_id=decision.parent_id,
                daughter_ids=(decision.daughter1_id, decision.daughter2_id),
                classifier_round=rounds[index],
                computed_class=event_computed,
                effective_class=effective,
                force_mode=forces[index],
                classifier_family=classifier_family,
            )
        )
        state = result.resolution.state
    checkpoints.append(_lineage_checkpoint(state, len(ordered) + 1, "final"))
    return TrackingEventTrace(
        tuple(checkpoints),
        tuple(classifications),
        {"source": "python_lineage_resolution"},
    )


def checkpoint_from_legacy_context(
    context: LegacyTrackingContext,
    index: int,
    phase: str,
) -> TrackingCheckpoint:
    """Snapshot raw MATLAB-style pointers without filtering deleted rows.

    ``LegacyTrackingContext`` deliberately retains predecessor and ordered
    successor slots incident to a delete-marked row.  Those pointers are part
    of the MATLAB mutation state even though the executable
    :class:`LineageGraphState` projection omits the corresponding edges.
    """

    from ..legacy_state import LegacyTrackingContext

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    ordered = sorted(
        context.nuclei,
        key=lambda item: (item.frame, item.matlab_row, item.nucleus_id),
    )
    return TrackingCheckpoint(
        index=index,
        phase=phase,
        nodes=tuple(
            LegacyNodePointerState(
                node_id=nucleus.nucleus_id,
                frame_0based=nucleus.frame - 1,
                row_0based=nucleus.matlab_row,
                deleted=nucleus.nucleus_id in context.deleted_ids,
                predecessor_id=context.predecessor(nucleus.nucleus_id),
                successor_slots=context.successor_slots(nucleus.nucleus_id),
            )
            for nucleus in ordered
        ),
    )


def legacy_context_from_checkpoint(
    template: LegacyTrackingContext,
    checkpoint: TrackingCheckpoint,
) -> LegacyTrackingContext:
    """Rehydrate a driver input from one complete MATLAB pointer checkpoint.

    ``template`` supplies immutable nucleus measurements and feature
    parameters; none of its edges or delete flags are reused.  The checkpoint
    must describe exactly the same frame/row identities. Reciprocal raw edges
    are reconstructed normally; the one asymmetric form produced by
    ``greedydeleteFPbranches`` is retained explicitly when a deleted branch
    root still names the parent whose successor slot was cleared. Asymmetry on
    an active row continues to fail closed.

    This is principally useful with the initial checkpoint emitted by the live
    full-tracking oracle: the final normalized node table retains all feature
    measurements, while the trace retains the graph immediately before
    ``greedydeleteFPbranches``.
    """

    from ...api import TrackEdge
    from ..legacy_state import LegacyStateError, LegacyTrackingContext

    if not isinstance(template, LegacyTrackingContext):
        raise TypeError("template must be a LegacyTrackingContext")
    if not isinstance(checkpoint, TrackingCheckpoint):
        raise TypeError("checkpoint must be a TrackingCheckpoint")

    expected_identity = {
        nucleus.nucleus_id: (nucleus.frame - 1, nucleus.matlab_row)
        for nucleus in template.nuclei
    }
    actual_identity = {
        node.node_id: (node.frame_0based, node.row_0based)
        for node in checkpoint.nodes
    }
    if actual_identity != expected_identity:
        missing = sorted(expected_identity.keys() - actual_identity.keys())
        unexpected = sorted(actual_identity.keys() - expected_identity.keys())
        moved = sorted(
            node_id
            for node_id in expected_identity.keys() & actual_identity.keys()
            if expected_identity[node_id] != actual_identity[node_id]
        )
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        if moved:
            details.append("frame/row drift=" + ",".join(moved))
        raise EventTraceFormatError(
            "Checkpoint identities do not match the measurement template"
            + (": " + "; ".join(details) if details else "")
        )

    by_id = checkpoint.by_id
    edges = []
    outgoing_pairs: set[tuple[str, str]] = set()
    for node in checkpoint.nodes:
        is_division = all(target is not None for target in node.successor_slots)
        for slot, target_id in enumerate(node.successor_slots):
            if target_id is None:
                continue
            target = by_id.get(target_id)
            if target is None:
                raise EventTraceFormatError(
                    f"Checkpoint successor {target_id!r} is not a known node"
                )
            if target.predecessor_id != node.node_id:
                raise EventTraceFormatError(
                    "Checkpoint raw pointers are asymmetric: "
                    f"{node.node_id!r} successor slot {slot} names {target_id!r}, "
                    f"but that row's predecessor is {target.predecessor_id!r}"
                )
            outgoing_pairs.add((node.node_id, target_id))
            source_frame = node.frame_0based
            target_frame = target.frame_0based
            kind = (
                "split"
                if is_division
                else ("gap" if target_frame - source_frame > 1 else "link")
            )
            edges.append(
                TrackEdge(
                    node.node_id,
                    target_id,
                    0.0,
                    kind,
                    {"LEGACY_SUCCESSOR_SLOT": slot},
                )
            )

    stale_predecessors: dict[str, str] = {}
    for node in checkpoint.nodes:
        if node.predecessor_id is None:
            continue
        if node.predecessor_id not in by_id:
            raise EventTraceFormatError(
                f"Checkpoint predecessor {node.predecessor_id!r} is not a known node"
            )
        if (node.predecessor_id, node.node_id) not in outgoing_pairs:
            if not node.deleted:
                raise EventTraceFormatError(
                    "Checkpoint raw pointers are asymmetric on an active row: "
                    f"{node.node_id!r} names predecessor {node.predecessor_id!r}, "
                    "but the predecessor has no matching successor slot"
                )
            stale_predecessors[node.node_id] = node.predecessor_id

    try:
        context = LegacyTrackingContext.from_nuclei_and_edges(
            template.nuclei,
            tuple(edges),
            template.parameters,
            deleted_ids=(node.node_id for node in checkpoint.nodes if node.deleted),
            stale_predecessor_by_id=stale_predecessors,
        )
    except LegacyStateError as exc:
        raise EventTraceFormatError(
            f"Checkpoint cannot be represented as exact legacy state: {exc}"
        ) from exc

    reconstructed = checkpoint_from_legacy_context(
        context,
        checkpoint.index,
        checkpoint.phase,
    )
    if reconstructed != checkpoint:
        raise EventTraceFormatError(
            "Checkpoint reconstruction changed raw pointer or delete state"
        )
    return context


def trace_from_legacy_class_zero_results(
    initial_context: LegacyTrackingContext,
    results: Sequence[LegacyClassZeroRepairResult],
    *,
    classifier_rounds: Sequence[int] | None = None,
    computed_classes: Sequence[int | None] | None = None,
    force_modes: Sequence[bool] | None = None,
    classifier_families: Sequence[str] | None = None,
) -> TrackingEventTrace:
    """Build an ordered raw-pointer trace for completed class-0 repairs.

    Each result represents the mutations following one already-completed
    effective class-0 decision.  A result that contains a nested classifier
    prediction is rejected: its intermediate raw context is not present in the
    aggregate result, so collapsing that invocation would make the event order
    look complete when it is not.  Full drivers can instead record each raw
    context with :func:`checkpoint_from_legacy_context` as it executes.
    """

    from ..legacy_class_zero import LegacyClassZeroRepairResult
    from ..legacy_state import LegacyTrackingContext

    if not isinstance(initial_context, LegacyTrackingContext):
        raise TypeError("initial_context must be a LegacyTrackingContext")
    ordered = tuple(results)
    if any(not isinstance(item, LegacyClassZeroRepairResult) for item in ordered):
        raise TypeError(
            "results must contain only LegacyClassZeroRepairResult values"
        )
    rounds = _per_event_values(
        classifier_rounds,
        len(ordered),
        1,
        "classifier_rounds",
    )
    computed = _per_event_values(
        computed_classes,
        len(ordered),
        0,
        "computed_classes",
    )
    forces = _per_event_values(force_modes, len(ordered), False, "force_modes")
    families = _per_event_values(
        classifier_families,
        len(ordered),
        "single_model",
        "classifier_families",
    )

    checkpoints = [checkpoint_from_legacy_context(initial_context, 0, "initial")]
    classifications: list[ClassificationEvent] = []
    context = initial_context
    for event_index, result in enumerate(ordered):
        if any(
            attempt.prediction is not None
            for attempt in result.diagnostics.attempts
        ):
            raise EventTraceFormatError(
                f"Class-0 result {event_index} contains nested classifier events; "
                "record their intermediate raw contexts explicitly"
            )
        projected = result.context.to_lineage_graph_state()
        if result.state != projected:
            raise EventTraceFormatError(
                f"Class-0 result {event_index} active state does not match its "
                "raw legacy context projection"
            )
        checkpoint_index = event_index + 1
        checkpoints.append(
            checkpoint_from_legacy_context(
                context,
                checkpoint_index,
                "pre_classification",
            )
        )
        diagnostics = result.diagnostics
        classifications.append(
            ClassificationEvent(
                event_index=event_index,
                checkpoint_index=checkpoint_index,
                parent_id=diagnostics.parent_id,
                daughter_ids=diagnostics.daughter_ids,
                classifier_round=rounds[event_index],
                computed_class=computed[event_index],
                effective_class=0,
                force_mode=forces[event_index],
                classifier_family=families[event_index],
            )
        )
        context = result.context
    checkpoints.append(
        checkpoint_from_legacy_context(context, len(ordered) + 1, "final")
    )
    return TrackingEventTrace(
        tuple(checkpoints),
        tuple(classifications),
        {"source": "python_legacy_class_zero_raw_context"},
    )


def trace_from_legacy_movie_result(
    initial_context: LegacyTrackingContext,
    result: LegacyMovieDecisionResult,
    observations: Sequence[LegacyMovieClassificationObservation],
) -> TrackingEventTrace:
    """Build the complete raw ordered trace captured by the movie driver.

    Pass ``list.append`` as ``run_legacy_movie_decisions``' classification
    observer, then supply that list here with the completed result.  Round-2
    observations retain the provisional raw attachment context.  The adapter
    validates every observable relationship before publishing the trace,
    including the exact two-pointer provisional attachment delta.  Derived
    mutation batches describe net deltas between checkpoints; they do not
    claim to recover the order of individual writes inside an interval.
    """

    from ..legacy_driver import (
        LegacyMovieClassificationObservation,
        LegacyMovieDecisionResult,
    )
    from ..classifier import AmbigiousClassifierPrediction
    from ..legacy_state import LegacyTrackingContext

    if not isinstance(initial_context, LegacyTrackingContext):
        raise TypeError("initial_context must be a LegacyTrackingContext")
    if not isinstance(result, LegacyMovieDecisionResult):
        raise TypeError("result must be a LegacyMovieDecisionResult")
    if not result.supported:
        raise EventTraceFormatError(
            "Cannot publish an exact event trace for an unsupported rolled-back movie"
        )
    captured = tuple(observations)
    if any(
        not isinstance(item, LegacyMovieClassificationObservation)
        for item in captured
    ):
        raise TypeError(
            "observations must contain LegacyMovieClassificationObservation values"
        )
    records = tuple(item.record for item in captured)
    if records != result.classifications:
        raise EventTraceFormatError(
            "Movie observations do not exactly match the result classification stream"
        )
    if tuple(item.sequence_index for item in records) != tuple(range(len(records))):
        raise EventTraceFormatError(
            "Movie classification sequence indices must be contiguous"
        )

    event_indices = tuple(item.event_index for item in result.events)
    if event_indices != tuple(range(len(result.events))):
        raise EventTraceFormatError("Movie top-level event indices must be contiguous")

    classifier_entry_context = (
        initial_context
        if result.classifier_entry_context is None
        else result.classifier_entry_context
    )
    named_contexts = [
        ("result", result.context),
        ("classifier entry", classifier_entry_context),
        *(
            (f"observation {index} detached", item.before_attachment_context)
            for index, item in enumerate(captured)
        ),
        *(
            (f"observation {index} classification", item.classification_context)
            for index, item in enumerate(captured)
        ),
    ]
    for label, context in named_contexts:
        if context.nuclei != initial_context.nuclei:
            raise EventTraceFormatError(
                f"{label} context changed immutable nucleus measurements"
            )
        if context.parameters != initial_context.parameters:
            raise EventTraceFormatError(
                f"{label} context changed immutable feature parameters"
            )

    if not captured:
        if result.events:
            raise EventTraceFormatError(
                "A movie with top-level events must contain classifier observations"
            )
        if result.context != classifier_entry_context:
            raise EventTraceFormatError(
                "An unclassified movie changed context after its isolated prepass"
            )
    else:
        first = captured[0]
        if first.record.classifier_round != 1:
            raise EventTraceFormatError(
                "The first movie observation must be a round-1 classification"
            )
        if first.before_attachment_context != classifier_entry_context:
            raise EventTraceFormatError(
                "The first movie observation is not anchored to the initial "
                "classifier-entry context"
            )

    round_one_records: list[LegacyMovieClassificationRecord] = []
    for observation_index, observation in enumerate(captured):
        record = observation.record
        extraction = observation.extraction
        classification_context = observation.classification_context
        if record.top_level_event_index >= len(result.events):
            raise EventTraceFormatError(
                f"Observation {observation_index} references an absent top-level event"
            )
        event = result.events[record.top_level_event_index]
        if extraction.parent_id != record.parent_id:
            raise EventTraceFormatError(
                f"Observation {observation_index} extraction parent disagrees with its record"
            )
        if extraction.daughter_ids != record.daughter_ids:
            raise EventTraceFormatError(
                f"Observation {observation_index} extraction daughters disagree with its record"
            )
        try:
            parent = classification_context.nucleus(record.parent_id)
        except (KeyError, ValueError) as exc:
            raise EventTraceFormatError(
                f"Observation {observation_index} classifier parent is absent"
            ) from exc
        if parent.frame != record.frame or parent.matlab_row != record.matlab_row:
            raise EventTraceFormatError(
                f"Observation {observation_index} frame/row disagrees with its record"
            )
        if record.force_mode != event.force_mode:
            raise EventTraceFormatError(
                f"Observation {observation_index} force mode disagrees with its event"
            )

        if record.classifier_round == 1:
            round_one_records.append(record)
            if record.parent_id in classification_context.deleted_ids or (
                classification_context.successor_slots(record.parent_id)
                != record.daughter_ids
            ):
                raise EventTraceFormatError(
                    f"Observation {observation_index} is not an active ordered bifurcation"
                )
            if observation.before_attachment_context != classification_context:
                raise EventTraceFormatError(
                    f"Round-1 observation {observation_index} has two different contexts"
                )
            if event.extraction != extraction:
                raise EventTraceFormatError(
                    f"Round-1 observation {observation_index} extraction disagrees with its event"
                )
            if event.classification_sequence_index != record.sequence_index:
                raise EventTraceFormatError(
                    f"Round-1 observation {observation_index} sequence disagrees with its event"
                )
            if event.prediction != record.prediction:
                raise EventTraceFormatError(
                    f"Round-1 observation {observation_index} prediction disagrees with its event"
                )
            continue

        candidate = observation.candidate
        if candidate is None:  # guarded by the observation dataclass
            raise EventTraceFormatError(
                f"Round-2 observation {observation_index} lacks its repair candidate"
            )
        if candidate.eligibility != "tentative_bifurcation":
            raise EventTraceFormatError(
                f"Round-2 observation {observation_index} is not a tentative bifurcation"
            )
        if (
            candidate.raw_rank != record.raw_attempt_rank
            or candidate.source_id != record.attachment_source_id
            or candidate.source_id != record.parent_id
        ):
            raise EventTraceFormatError(
                f"Round-2 observation {observation_index} attachment metadata disagrees"
            )
        target_id = record.attachment_target_id
        if target_id is None or target_id != record.daughter_ids[1]:
            raise EventTraceFormatError(
                f"Round-2 observation {observation_index} target is not ordered daughter 2"
            )
        top_first, top_second = event.extraction.daughter_ids
        first_score, second_score = event.extraction.nondivision_scores
        detached_id = top_first if first_score > second_score else top_second
        if target_id != detached_id:
            raise EventTraceFormatError(
                f"Round-2 observation {observation_index} target is not the detached daughter"
            )
        before_checkpoint = checkpoint_from_legacy_context(
            observation.before_attachment_context,
            0,
            "initial",
        )
        classification_checkpoint = checkpoint_from_legacy_context(
            classification_context,
            0,
            "initial",
        )
        actual_attachment = frozenset(
            _checkpoint_mutations(before_checkpoint, classification_checkpoint)
        )
        expected_attachment = frozenset(
            {
                TrackingMutation(
                    "predecessor_set",
                    target_id,
                    candidate.source_id,
                ),
                TrackingMutation(
                    "edge_added",
                    candidate.source_id,
                    target_id,
                    1,
                ),
            }
        )
        if actual_attachment != expected_attachment:
            raise EventTraceFormatError(
                f"Round-2 observation {observation_index} does not contain exactly "
                "the provisional attachment pointer delta"
            )
        if classification_context.successor_slots(record.parent_id) != (
            record.daughter_ids
        ):
            raise EventTraceFormatError(
                f"Observation {observation_index} is not a raw ordered bifurcation"
            )

    if tuple(item.top_level_event_index for item in round_one_records) != tuple(
        range(len(result.events))
    ):
        raise EventTraceFormatError(
            "Round-1 observations do not map one-to-one onto top-level events"
        )
    top_sequences = {
        item.top_level_event_index: item.sequence_index for item in round_one_records
    }
    for event in result.events:
        nested = tuple(
            item
            for item in records
            if item.top_level_event_index == event.event_index
            and item.classifier_round == 2
        )
        if len(nested) != event.nested_classification_count:
            raise EventTraceFormatError(
                f"Top-level event {event.event_index} nested-call count disagrees"
            )
        next_top_sequence = top_sequences.get(event.event_index + 1, len(records))
        if any(
            item.sequence_index <= top_sequences[event.event_index]
            or item.sequence_index >= next_top_sequence
            for item in nested
        ):
            raise EventTraceFormatError(
                f"Top-level event {event.event_index} nested calls are out of order"
            )

    checkpoints = [checkpoint_from_legacy_context(initial_context, 0, "initial")]
    classifications: list[ClassificationEvent] = []
    for event_index, observation in enumerate(captured):
        record = observation.record
        checkpoint_index = event_index + 1
        checkpoints.append(
            checkpoint_from_legacy_context(
                observation.classification_context,
                checkpoint_index,
                "pre_classification",
            )
        )
        classifier_family = (
            "ambigious_multi_model"
            if isinstance(record.prediction, AmbigiousClassifierPrediction)
            else "single_model"
        )
        classifications.append(
            ClassificationEvent(
                event_index=event_index,
                checkpoint_index=checkpoint_index,
                parent_id=record.parent_id,
                daughter_ids=record.daughter_ids,
                classifier_round=record.classifier_round,
                computed_class=record.computed_class,
                effective_class=record.effective_class,
                force_mode=record.force_mode,
                classifier_family=classifier_family,
            )
        )
    checkpoints.append(
        checkpoint_from_legacy_context(
            result.context,
            len(captured) + 1,
            "final",
        )
    )
    return TrackingEventTrace(
        tuple(checkpoints),
        tuple(classifications),
        {
            "source": "python_legacy_movie_decision_driver",
            "source_model_sha256": result.source_model_sha256,
            "classifier_family": result.classifier_family,
        },
    )


def write_event_trace(path: str | Path, trace: TrackingEventTrace) -> None:
    if not isinstance(trace, TrackingEventTrace):
        raise TypeError("trace must be a TrackingEventTrace")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(trace.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_event_trace(path: str | Path) -> TrackingEventTrace:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise EventTraceFormatError("event-trace root must be an object")
    return TrackingEventTrace.from_dict(value)


def _checkpoint_mutations(
    before: TrackingCheckpoint,
    after: TrackingCheckpoint,
) -> tuple[TrackingMutation, ...]:
    first = before.by_id
    second = after.by_id
    if first.keys() != second.keys():
        raise EventTraceFormatError("checkpoint node identities changed")
    ordered_ids = tuple(
        item.node_id
        for item in sorted(
            before.nodes,
            key=lambda item: (item.frame_0based, item.row_0based, item.node_id),
        )
    )
    mutations: list[TrackingMutation] = []
    for node_id in ordered_ids:
        old = first[node_id]
        new = second[node_id]
        if old.deleted != new.deleted:
            mutations.append(
                TrackingMutation(
                    "node_deleted" if new.deleted else "node_restored",
                    node_id,
                )
            )
        if old.predecessor_id != new.predecessor_id:
            if old.predecessor_id is not None:
                mutations.append(
                    TrackingMutation(
                        "predecessor_removed",
                        node_id,
                        old.predecessor_id,
                    )
                )
            if new.predecessor_id is not None:
                mutations.append(
                    TrackingMutation(
                        "predecessor_set",
                        node_id,
                        new.predecessor_id,
                    )
                )
        for slot, (old_target, new_target) in enumerate(
            zip(old.successor_slots, new.successor_slots, strict=True)
        ):
            if old_target == new_target:
                continue
            if old_target is not None:
                mutations.append(
                    TrackingMutation("edge_removed", node_id, old_target, slot)
                )
            if new_target is not None:
                mutations.append(
                    TrackingMutation("edge_added", node_id, new_target, slot)
                )
    return tuple(mutations)


def _event_key(
    event: TrackingEvent,
    mapping: Mapping[str, str] | None,
) -> tuple[Any, ...]:
    if isinstance(event, ClassificationEvent):
        return (
            "classification",
            _mapped(event.parent_id, mapping),
            tuple(_mapped(item, mapping) for item in event.daughter_ids),
            event.classifier_round,
            event.computed_class,
            event.effective_class,
            event.force_mode,
            event.classifier_family,
        )
    # Snapshots expose the exact net changes in an interval, not the order in
    # which MATLAB wrote individual fields.  Canonicalize the set deliberately
    # so comparison certifies checkpoint-delta parity without overclaiming
    # intra-batch causal order.
    mutation_keys = tuple(
        sorted(_mutation_key(item, mapping) for item in event.mutations)
    )
    return (
        "mutation_batch",
        event.next_classification_index,
        mutation_keys,
    )


def _mutation_key(
    mutation: TrackingMutation,
    mapping: Mapping[str, str] | None,
) -> tuple[Any, ...]:
    return (
        mutation.kind,
        _mapped(mutation.node_id, mapping),
        (
            None
            if mutation.related_id is None
            else _mapped(mutation.related_id, mapping)
        ),
        mutation.slot,
    )


def _mapped(node_id: str, mapping: Mapping[str, str] | None) -> str:
    if mapping is None:
        return node_id
    if node_id in mapping:
        return mapping[node_id]
    return f"<unmapped:{node_id}>"


def _event_dict(
    event: TrackingEvent,
    mapping: Mapping[str, str] | None,
) -> dict[str, Any]:
    if mapping is None:
        return event.to_dict()
    if isinstance(event, ClassificationEvent):
        result = event.to_dict()
        result["parent"] = _mapped(event.parent_id, mapping)
        result["daughters"] = [
            _mapped(item, mapping) for item in event.daughter_ids
        ]
        return result
    result = event.to_dict()
    result["mutations"] = [
        {
            **item.to_dict(),
            "node": _mapped(item.node_id, mapping),
            **(
                {}
                if item.related_id is None
                else {"related": _mapped(item.related_id, mapping)}
            ),
        }
        for item in event.mutations
    ]
    return result


def _checkpoint_from_dict(value: Any) -> TrackingCheckpoint:
    if not isinstance(value, Mapping):
        raise EventTraceFormatError("checkpoint must be an object")
    raw_nodes = value.get("nodes", ())
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
        raise EventTraceFormatError("checkpoint nodes must be a sequence")
    nodes: list[LegacyNodePointerState] = []
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise EventTraceFormatError("checkpoint node must be an object")
        slots = raw.get("successor_slots", ())
        if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
            raise EventTraceFormatError("successor_slots must be a sequence")
        nodes.append(
            LegacyNodePointerState(
                node_id=raw.get("id", ""),
                frame_0based=raw.get("frame_0based"),
                row_0based=raw.get("row_0based"),
                deleted=raw.get("deleted"),
                predecessor_id=raw.get("predecessor"),
                successor_slots=tuple(slots),
            )
        )
    return TrackingCheckpoint(
        index=value.get("index"),
        phase=str(value.get("phase", "")),
        nodes=tuple(nodes),
    )


def _classification_from_dict(value: Any) -> ClassificationEvent:
    if not isinstance(value, Mapping):
        raise EventTraceFormatError("classification must be an object")
    daughters = value.get("daughters", ())
    if not isinstance(daughters, Sequence) or isinstance(daughters, (str, bytes)):
        raise EventTraceFormatError("classification daughters must be a sequence")
    return ClassificationEvent(
        event_index=value.get("event_index"),
        checkpoint_index=value.get("checkpoint_index"),
        parent_id=value.get("parent", ""),
        daughter_ids=tuple(daughters),
        classifier_round=value.get("classifier_round"),
        computed_class=value.get("computed_class"),
        effective_class=value.get("effective_class"),
        force_mode=value.get("force_mode"),
        classifier_family=str(value.get("classifier_family", "")),
    )


def _numeric_integer(value: Any, label: str) -> int:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 1:
        raise EventTraceFormatError(f"{label} must be a scalar")
    return _float_integer(array[0], label)


def _float_integer(value: Any, label: str) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise EventTraceFormatError(f"{label} must be a finite integer")
    return int(numeric)


def _float_class(value: Any, label: str) -> int:
    result = _float_integer(value, label)
    if result not in {0, 1, 2, 3}:
        raise EventTraceFormatError(f"{label} must be 0, 1, 2, or 3")
    return result


def _numeric_table(value: Any, columns: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EventTraceFormatError(f"{label} must be numeric") from exc
    if array.size == 0:
        return np.empty((0, columns), dtype=float)
    if array.ndim == 1:
        if array.shape[0] != columns:
            raise EventTraceFormatError(
                f"{label} has shape {array.shape}; expected N-by-{columns}"
            )
        array = array.reshape(1, columns)
    if array.ndim != 2 or array.shape[1] != columns:
        raise EventTraceFormatError(
            f"{label} has shape {array.shape}; expected N-by-{columns}"
        )
    return array


def _snapshot_tables(value: Any, count: int) -> tuple[np.ndarray, ...]:
    if count < 2:
        raise EventTraceFormatError("snapshot_count must be at least two")
    if isinstance(value, np.ndarray) and value.dtype == object:
        raw_items = tuple(value.reshape(-1))
    elif isinstance(value, (list, tuple)):
        raw_items = tuple(value)
    else:
        raw_items = (value,)
    if len(raw_items) != count:
        raise EventTraceFormatError(
            f"MATLAB emitted {len(raw_items)} snapshots; expected {count}"
        )
    return tuple(
        _numeric_table(item, 9, f"snapshots[{index}]")
        for index, item in enumerate(raw_items)
    )


def _matlab_snapshot_node(row: np.ndarray) -> LegacyNodePointerState:
    frame = _float_integer(row[0], "snapshot frame")
    node = _float_integer(row[1], "snapshot node")
    deleted = _float_integer(row[2], "snapshot deleted")
    if deleted not in {0, 1}:
        raise EventTraceFormatError("snapshot deleted flag must be 0 or 1")
    return LegacyNodePointerState(
        node_id=f"matlab:{frame}:{node}",
        frame_0based=frame,
        row_0based=node,
        deleted=bool(deleted),
        predecessor_id=_optional_matlab_reference(
            row[3], row[4], "snapshot predecessor"
        ),
        successor_slots=(
            _optional_matlab_reference(row[5], row[6], "snapshot successor 1"),
            _optional_matlab_reference(row[7], row[8], "snapshot successor 2"),
        ),
    )


def _matlab_reference(frame: Any, node: Any, label: str) -> str:
    result = _optional_matlab_reference(frame, node, label)
    if result is None:
        raise EventTraceFormatError(f"{label} cannot be absent")
    return result


def _optional_matlab_reference(frame: Any, node: Any, label: str) -> str | None:
    frame_value = _float_integer(frame, f"{label} frame")
    node_value = _float_integer(node, f"{label} node")
    if frame_value == -1 and node_value == -1:
        return None
    if frame_value < 0 or node_value < 0:
        raise EventTraceFormatError(
            f"{label} must use two non-negative indices or the -1/-1 sentinel"
        )
    return f"matlab:{frame_value}:{node_value}"


def _lineage_checkpoint(
    state: LineageGraphState,
    index: int,
    phase: str,
) -> TrackingCheckpoint:
    from ..lineage import LineageGraphState

    if not isinstance(state, LineageGraphState):
        raise TypeError("lineage checkpoint state must be a LineageGraphState")
    ordered_ids = sorted(state.frames, key=lambda item: (state.frames[item], item))
    row_by_id: dict[str, int] = {}
    next_row: dict[int, int] = {}
    for node_id in ordered_ids:
        frame = state.frames[node_id]
        row_by_id[node_id] = next_row.get(frame, 0)
        next_row[frame] = row_by_id[node_id] + 1
    outgoing: dict[str, list[Any]] = {}
    incoming: dict[str, str] = {}
    for edge in state.edges:
        outgoing.setdefault(edge.source_id, []).append(edge)
        incoming[edge.target_id] = edge.source_id
    nodes: list[LegacyNodePointerState] = []
    for node_id in ordered_ids:
        edges = outgoing.get(node_id, [])
        slots: list[str | None] = [None, None]
        unresolved: list[Any] = []
        for edge in edges:
            raw_slot = edge.features.get("LEGACY_SUCCESSOR_SLOT")
            if isinstance(raw_slot, Integral) and not isinstance(raw_slot, bool):
                slot = int(raw_slot)
                if slot not in {0, 1} or slots[slot] is not None:
                    raise EventTraceFormatError(
                        f"Invalid legacy successor slot on {node_id!r}"
                    )
                slots[slot] = edge.target_id
            else:
                unresolved.append(edge)
        for edge in sorted(unresolved, key=lambda item: item.target_id):
            try:
                slot = slots.index(None)
            except ValueError as exc:
                raise EventTraceFormatError(
                    f"Node {node_id!r} has more than two successor slots"
                ) from exc
            slots[slot] = edge.target_id
        deleted = node_id in state.deleted_ids
        nodes.append(
            LegacyNodePointerState(
                node_id=node_id,
                frame_0based=state.frames[node_id] - 1,
                row_0based=row_by_id[node_id],
                deleted=deleted,
                predecessor_id=None if deleted else incoming.get(node_id),
                successor_slots=(None, None) if deleted else tuple(slots),
            )
        )
    return TrackingCheckpoint(index, phase, tuple(nodes))


def _per_event_values(
    values: Sequence[Any] | None,
    count: int,
    default: Any,
    label: str,
) -> tuple[Any, ...]:
    if values is None:
        return (default,) * count
    result = tuple(values)
    if len(result) != count:
        raise EventTraceFormatError(f"{label} must contain {count} values")
    return result


__all__ = [
    "EVENT_TRACE_SCHEMA",
    "ClassificationEvent",
    "EventTraceComparison",
    "EventTraceFormatError",
    "LegacyNodePointerState",
    "MutationBatch",
    "TrackingCheckpoint",
    "TrackingEventTrace",
    "TrackingMutation",
    "assert_event_trace_parity",
    "checkpoint_from_legacy_context",
    "legacy_context_from_checkpoint",
    "compare_event_traces",
    "matlab_event_trace",
    "read_event_trace",
    "trace_from_legacy_class_zero_results",
    "trace_from_legacy_movie_result",
    "trace_from_lineage_results",
    "write_event_trace",
]
