"""Deterministic full-movie StarryNite event-order validation tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from acetree_py.tracking.api import TrackEdge
from acetree_py.tracking.starrynite.bifurcation import SingleModelLineageResult
from acetree_py.tracking.starrynite.classifier import (
    AmbigiousClassifierPrediction,
    SingleModelPrediction,
)
from acetree_py.tracking.starrynite.lineage import (
    BifurcationDecision,
    LineageGraphState,
    resolve_bifurcation,
)
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
)
from acetree_py.tracking.starrynite.oracle.event_trace import (
    ClassificationEvent,
    EventTraceFormatError,
    LegacyNodePointerState,
    TrackingCheckpoint,
    TrackingEventTrace,
    assert_event_trace_parity,
    compare_event_traces,
    legacy_context_from_checkpoint,
    matlab_event_trace,
    read_event_trace,
    trace_from_lineage_results,
    write_event_trace,
)
from acetree_py.tracking.starrynite.oracle.matlab_backend import MatlabOracleRun


def _matlab_payload() -> dict[str, object]:
    initial = np.asarray(
        [
            [0, 0, 0, -1, -1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, -1, -1, -1, -1],
            [1, 1, 0, 0, 0, -1, -1, -1, -1],
        ],
        dtype=float,
    )
    final = np.asarray(
        [
            [0, 0, 0, -1, -1, 1, 1, -1, -1],
            [1, 0, 1, -1, -1, -1, -1, -1, -1],
            [1, 1, 0, 0, 0, -1, -1, -1, -1],
        ],
        dtype=float,
    )
    snapshots = np.empty(3, dtype=object)
    snapshots[:] = [initial, initial.copy(), final]
    return {
        "tracking_event_trace": {
            "schema_version": 1,
            "classification_table": np.asarray(
                [[0, 1, 0, 0, 1, 0, 1, 1, 1, 3, 3, 3, 0, 0]],
                dtype=float,
            ),
            "snapshot_count": 3,
            "snapshots": snapshots,
        }
    }


def _remap_trace(
    trace: TrackingEventTrace,
    mapping: dict[str, str],
) -> TrackingEventTrace:
    checkpoints = tuple(
        TrackingCheckpoint(
            checkpoint.index,
            checkpoint.phase,
            tuple(
                LegacyNodePointerState(
                    node_id=mapping[node.node_id],
                    frame_0based=node.frame_0based,
                    row_0based=node.row_0based,
                    deleted=node.deleted,
                    predecessor_id=(
                        None
                        if node.predecessor_id is None
                        else mapping[node.predecessor_id]
                    ),
                    successor_slots=tuple(
                        None if item is None else mapping[item]
                        for item in node.successor_slots
                    ),
                )
                for node in checkpoint.nodes
            ),
        )
        for checkpoint in trace.checkpoints
    )
    classifications = tuple(
        replace(
            event,
            parent_id=mapping[event.parent_id],
            daughter_ids=tuple(mapping[item] for item in event.daughter_ids),
        )
        for event in trace.classifications
    )
    return TrackingEventTrace(checkpoints, classifications, {"engine": "python"})


def test_matlab_trace_interleaves_classifier_and_pointer_mutations() -> None:
    trace = matlab_event_trace(_matlab_payload())

    assert len(trace.checkpoints) == 3
    assert trace.classifications == (
        ClassificationEvent(
            event_index=0,
            checkpoint_index=1,
            parent_id="matlab:0:0",
            daughter_ids=("matlab:1:0", "matlab:1:1"),
            classifier_round=1,
            computed_class=3,
            effective_class=3,
            force_mode=False,
        ),
    )
    before, classification, after = trace.ordered_events
    assert before.mutations == ()
    assert classification.effective_class == 3
    assert {item.kind for item in after.mutations} == {
        "node_deleted",
        "predecessor_removed",
        "edge_removed",
        "edge_added",
    }
    assert any(
        item.kind == "edge_added"
        and item.node_id == "matlab:0:0"
        and item.related_id == "matlab:1:1"
        and item.slot == 0
        for item in after.mutations
    )


def test_raw_checkpoint_rehydrates_exact_driver_input_and_rejects_asymmetry() -> None:
    parameters = LegacyFeatureParameters(
        interval=1.0,
        candidate_cutoff=5.0,
        temporal_cutoff=3,
        temporal_cutoff_start=1,
        small_cutoff=1.0,
        anisotropy_xyz=(1.0, 1.0, 1.0),
        end_frame=2,
    )

    def nucleus(node_id: str, frame: int, row: int, x: float) -> LegacyNucleus:
        return LegacyNucleus(
            node_id,
            frame,
            row,
            (x, 0.0, 0.0),
            4.0,
            10.0,
            2.0,
            1.0,
            0.0,
            2,
            1.0,
            0.5,
        )

    nuclei = (
        nucleus("parent", 1, 0, 0.0),
        nucleus("daughter-1", 2, 0, -1.0),
        nucleus("daughter-2", 2, 1, 1.0),
    )
    # The template deliberately represents a different, final topology.  Only
    # its immutable measurements and parameters may survive reconstruction.
    template = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        (
            TrackEdge(
                "parent",
                "daughter-2",
                0.0,
                "link",
                {"LEGACY_SUCCESSOR_SLOT": 0},
            ),
        ),
        parameters,
        deleted_ids=("daughter-1",),
    )
    checkpoint = TrackingCheckpoint(
        0,
        "initial",
        (
            LegacyNodePointerState(
                "parent", 0, 0, False, None, ("daughter-1", "daughter-2")
            ),
            LegacyNodePointerState(
                "daughter-1", 1, 0, False, "parent", (None, None)
            ),
            LegacyNodePointerState(
                "daughter-2", 1, 1, False, "parent", (None, None)
            ),
        ),
    )

    reconstructed = legacy_context_from_checkpoint(template, checkpoint)
    assert reconstructed.deleted_ids == frozenset()
    assert reconstructed.successor_slots("parent") == (
        "daughter-1",
        "daughter-2",
    )
    assert reconstructed.predecessor("daughter-1") == "parent"
    assert {edge.kind for edge in reconstructed.edges} == {"split"}

    asymmetric_nodes = tuple(
        replace(node, predecessor_id=None)
        if node.node_id == "daughter-1"
        else node
        for node in checkpoint.nodes
    )
    with pytest.raises(EventTraceFormatError, match="asymmetric"):
        legacy_context_from_checkpoint(
            template,
            replace(checkpoint, nodes=asymmetric_nodes),
        )


def test_event_comparison_aligns_ids_but_preserves_daughter_slot_order() -> None:
    reference = matlab_event_trace(_matlab_payload())
    matlab_to_python = {
        "matlab:0:0": "py:parent",
        "matlab:1:0": "py:daughter1",
        "matlab:1:1": "py:daughter2",
    }
    candidate = _remap_trace(reference, matlab_to_python)
    python_to_matlab = {value: key for key, value in matlab_to_python.items()}

    comparison = compare_event_traces(
        reference,
        candidate,
        candidate_to_reference=python_to_matlab,
    )
    assert comparison.exact_match
    assert comparison.matching_prefix_count == 3
    assert_event_trace_parity(
        reference,
        candidate,
        candidate_to_reference=python_to_matlab,
    )

    reversed_classification = replace(
        candidate.classifications[0],
        daughter_ids=tuple(reversed(candidate.classifications[0].daughter_ids)),
    )
    reversed_checkpoints = list(candidate.checkpoints)
    parent = reversed_checkpoints[1].by_id["py:parent"]
    nodes = tuple(
        replace(node, successor_slots=tuple(reversed(parent.successor_slots)))
        if node.node_id == "py:parent"
        else node
        for node in reversed_checkpoints[1].nodes
    )
    reversed_checkpoints[1] = replace(reversed_checkpoints[1], nodes=nodes)
    reversed_trace = TrackingEventTrace(
        tuple(reversed_checkpoints),
        (reversed_classification,),
    )
    mismatch = compare_event_traces(
        reference,
        reversed_trace,
        candidate_to_reference=python_to_matlab,
    )
    assert not mismatch.exact_match
    assert not mismatch.classification_order_match


def test_dynamic_classification_divergence_is_visible_when_final_state_matches() -> None:
    reference = matlab_event_trace(_matlab_payload())
    candidate = TrackingEventTrace(
        reference.checkpoints,
        (replace(reference.classifications[0], computed_class=1, effective_class=1),),
    )

    comparison = compare_event_traces(reference, candidate)
    assert not comparison.exact_match
    assert comparison.first_divergence_index == 1
    assert not comparison.classification_order_match
    assert comparison.mutation_batch_match
    with pytest.raises(AssertionError, match="ordered checkpoint event 1"):
        assert_event_trace_parity(reference, candidate)


def test_trace_json_round_trip_and_oracle_run_accessor(tmp_path) -> None:
    run = MatlabOracleRun(
        operation="full_tracking",
        result=_matlab_payload(),
        stdout="",
        stderr="",
        duration_seconds=0.0,
    )
    trace = run.tracking_event_trace()
    destination = tmp_path / "event-trace.json"
    write_event_trace(destination, trace)
    restored = read_event_trace(destination)

    assert restored.to_dict() == trace.to_dict()


def test_python_resolution_results_retain_caller_order() -> None:
    state = LineageGraphState(
        {"parent": 1, "daughter1": 2, "daughter2": 2},
        (
            TrackEdge(
                "parent",
                "daughter1",
                0.0,
                "split",
                {"LEGACY_SUCCESSOR_SLOT": 0},
            ),
            TrackEdge(
                "parent",
                "daughter2",
                0.0,
                "split",
                {"LEGACY_SUCCESSOR_SLOT": 1},
            ),
        ),
    )
    decision = BifurcationDecision(3, "parent", "daughter1", "daughter2")
    resolution = resolve_bifurcation(state, decision)
    prediction = SingleModelPrediction(
        predicted_class=3,
        topology_class=4,
        topology_case="fully_false_positive_looking",
        features=(4.0,),
        log_scores=(0.0, 0.0, 0.0, 1.0),
        posterior=(0.1, 0.1, 0.1, 0.7),
    )
    result = SingleModelLineageResult(prediction, decision, resolution)

    trace = trace_from_lineage_results(state, (result,), classifier_rounds=(1,))
    assert trace.classifications[0].parent_id == "parent"
    assert trace.classifications[0].daughter_ids == ("daughter1", "daughter2")
    assert trace.ordered_events[-1].mutations


def test_python_trace_infers_ambigious_family_and_computed_class() -> None:
    state = LineageGraphState(
        {"parent": 1, "daughter1": 2, "daughter2": 2},
        (
            TrackEdge(
                "parent",
                "daughter1",
                0.0,
                "split",
                {"LEGACY_SUCCESSOR_SLOT": 0},
            ),
            TrackEdge(
                "parent",
                "daughter2",
                0.0,
                "split",
                {"LEGACY_SUCCESSOR_SLOT": 1},
            ),
        ),
    )
    decision = BifurcationDecision(3, "parent", "daughter1", "daughter2")
    resolution = resolve_bifurcation(state, decision)
    prediction = AmbigiousClassifierPrediction(
        predicted_class=3,
        computed_class=2,
        topology_class=4,
        topology_case="fully_false_positive_looking",
        submodel_name="fp_div",
        features=(1.0,),
        log_scores=(0.0, 1.0),
        posterior=(0.25, 0.75),
    )
    result = SimpleNamespace(
        prediction=prediction,
        decision=decision,
        resolution=resolution,
    )

    inferred = trace_from_lineage_results(state, (result,))
    event = inferred.classifications[0]
    assert event.classifier_family == "ambigious_multi_model"
    assert event.computed_class == 2
    assert event.effective_class == 3

    overridden = trace_from_lineage_results(
        state,
        (result,),
        computed_classes=(1,),
        classifier_families=("single_model",),
    )
    assert overridden.classifications[0].computed_class == 1
    assert overridden.classifications[0].classifier_family == "single_model"


def test_matlab_trace_rejects_checkpoint_index_drift() -> None:
    payload = _matlab_payload()
    raw = payload["tracking_event_trace"]
    assert isinstance(raw, dict)
    table = np.asarray(raw["classification_table"]).copy()
    table[0, 1] = 2
    raw["classification_table"] = table

    with pytest.raises(EventTraceFormatError, match="indices"):
        matlab_event_trace(payload)
