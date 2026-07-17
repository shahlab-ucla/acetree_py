"""Focused tests for the recursive legacy class-0 repair state machine."""

from __future__ import annotations

from dataclasses import replace

import pytest

import acetree_py.tracking.starrynite.legacy_class_zero as class_zero_module
from acetree_py.tracking.api import TrackEdge
from acetree_py.tracking.starrynite.classifier import (
    AmbigiousClassifierPrediction,
    CategoricalFeatureDistribution,
    NeutralAmbigiousClassifierFamily,
    NeutralNaiveBayesClassifier,
    NeutralNaiveBayesSubmodel,
    SingleModelFeatureInput,
    SingleModelFeatureLayout,
    SingleModelPrediction,
)
from acetree_py.tracking.starrynite.legacy_class_zero import (
    repair_legacy_class_zero_bifurcation,
)
from acetree_py.tracking.starrynite.legacy_features import (
    BACKWARD_FEATURE_NAMES,
    DAUGHTER_FEATURE_NAMES,
    FORWARD_FEATURE_NAMES,
    LegacyBifurcationExtraction,
    LegacyTrackingStatistics,
)
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
)
from acetree_py.tracking.starrynite.repair_candidates import BackwardRepairCandidates
from acetree_py.tracking.starrynite.oracle.event_trace import (
    EventTraceFormatError,
    TrackingEventTrace,
    compare_event_traces,
    trace_from_legacy_class_zero_results,
)


@pytest.fixture(autouse=True)
def _stable_nondivision_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        class_zero_module,
        "calculate_legacy_nondivision_scores",
        lambda *_args, **_kwargs: (2.0, 1.0),
    )


def _nucleus(identifier: str, frame: int, row: int, x: float) -> LegacyNucleus:
    return LegacyNucleus(
        identifier,
        frame,
        row,
        (x, 0.0, 0.0),
        4.0,
        10.0,
        2.0,
        1.0,
        1.0,
        2,
        2.0,
        1.0,
    )


def _edge(source: str, target: str, slot: int) -> TrackEdge:
    return TrackEdge(
        source,
        target,
        0.0,
        "split" if slot == 1 else "link",
        {"LEGACY_SUCCESSOR_SLOT": slot},
    )


def _context(
    *,
    include_loose: bool,
    include_one_child: bool,
    deleted_loose: bool = False,
    deleted_one_child: bool = False,
) -> LegacyTrackingContext:
    nuclei = [
        _nucleus("parent", 1, 0, 100.0),
        _nucleus("d1", 2, 0, 0.0),
        _nucleus("d2", 2, 1, 100.0),
    ]
    edges = [_edge("parent", "d1", 0), _edge("parent", "d2", 1)]
    if include_one_child:
        nuclei.extend(
            (
                _nucleus("one", 1, 1, 0.0),
                _nucleus("existing", 2, 2, 30.0),
            )
        )
        edges.append(_edge("one", "existing", 0))
    if include_loose:
        row = 2 if include_one_child else 1
        nuclei.append(_nucleus("loose", 1, row, 10.0 if include_one_child else 0.0))
    parameters = LegacyFeatureParameters(
        interval=1.0,
        candidate_cutoff=2.0,
        temporal_cutoff=2,
        temporal_cutoff_start=2,
        small_cutoff=2.0,
        anisotropy_xyz=(1.0, 1.0, 1.0),
        end_frame=2,
    )
    deleted_ids: list[str] = []
    if deleted_loose:
        deleted_ids.append("loose")
    if deleted_one_child:
        deleted_ids.append("one")
    return LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        edges,
        parameters,
        deleted_ids=deleted_ids,
    )


def _extraction(
    parent: str,
    daughter1: str,
    daughter2: str,
    scores: tuple[float, float],
) -> LegacyBifurcationExtraction:
    feature_input = SingleModelFeatureInput(
        daughter_features=(0.0,) * 22,
        backward_features=(0.0,) * 11,
        forward_features=(0.0,) * 13,
        daughter_lengths=(1.0, 1.0),
        backward_candidate_present=(False, False),
        best_forward_lengths=(-1.0, -1.0),
        small_cutoff=2.0,
    )
    repairs = BackwardRepairCandidates((), (), None, -1.0, -1.0)
    return LegacyBifurcationExtraction(
        parent,
        (daughter1, daughter2),
        feature_input,
        DAUGHTER_FEATURE_NAMES,
        feature_input.daughter_features,
        BACKWARD_FEATURE_NAMES,
        feature_input.backward_features,
        FORWARD_FEATURE_NAMES,
        feature_input.forward_features,
        feature_input.daughter_lengths,
        feature_input.backward_candidate_present,
        feature_input.best_forward_lengths,
        scores,
        repairs,
        None,
    )


def _statistics() -> LegacyTrackingStatistics:
    def identity(size: int) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(row == column) for column in range(size))
            for row in range(size)
        )

    return LegacyTrackingStatistics(
        (0.0, 0.0),
        identity(2),
        (0.0,) * 10,
        identity(10),
        (0.0,) * 4,
        identity(4),
    )


def _model() -> NeutralNaiveBayesClassifier:
    return NeutralNaiveBayesClassifier(
        source_model_sha256="c" * 64,
        classifier_family="new_classifier",
        feature_layout=SingleModelFeatureLayout(
            (False,) * 22,
            (False,) * 11,
            (False,) * 13,
        ),
        feature_names=("topology_class",),
        class_labels=(0, 1, 2, 3),
        class_priors=(0.25,) * 4,
        misclassification_costs=tuple(
            tuple(float(row != column) for column in range(4))
            for row in range(4)
        ),
        distributions=(
            CategoricalFeatureDistribution(
                categories=(1.0, 2.0, 3.0, 4.0, 5.0),
                probabilities=((0.2,) * 5,) * 4,
            ),
        ),
    )


def _ambigious_model() -> NeutralAmbigiousClassifierFamily:
    def submodel(
        labels: tuple[int, ...],
        predicted: int,
    ) -> NeutralNaiveBayesSubmodel:
        other_prior = 0.2 / (len(labels) - 1)
        priors = tuple(0.8 if label == predicted else other_prior for label in labels)
        return NeutralNaiveBayesSubmodel(
            classifier_family="new_classifier",
            feature_names=("daughter_feature_1",),
            class_labels=labels,
            class_priors=priors,
            misclassification_costs=tuple(
                tuple(float(row != column) for column in range(len(labels)))
                for row in range(len(labels))
            ),
            distributions=(
                CategoricalFeatureDistribution(
                    categories=(0.0,),
                    probabilities=tuple((1.0,) for _label in labels),
                ),
            ),
        )

    return NeutralAmbigiousClassifierFamily(
        source_model_sha256="d" * 64,
        feature_layout=SingleModelFeatureLayout(
            (True,) + (False,) * 21,
            (False,) * 11,
            (False,) * 13,
        ),
        ambigious=submodel((0, 2, 3), 0),
        fp_div=submodel((0, 1, 3), 1),
        dirtyfp_fn=submodel((0, 1, 2, 3), 1),
        divfp=submodel((0, 1, 3), 1),
    )


def _prediction(classification: int) -> SingleModelPrediction:
    posterior = tuple(1.0 if index == classification else 0.0 for index in range(4))
    return SingleModelPrediction(
        classification,
        1,
        "test",
        (1.0,),
        tuple(0.0 for _ in range(4)),
        posterior,
    )


def test_direct_class_zero_candidate_commits_first_loose_source() -> None:
    context = _context(include_loose=True, include_one_child=False)

    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
    )

    assert result.status == "direct_attached"
    assert len(result.diagnostics.raw_attempts) == 4
    assert result.diagnostics.reattached_to_id == "loose"
    assert result.context.successor_slots("parent") == ("d2", None)
    assert result.context.successor_slots("loose") == ("d1", None)


def test_entry_recomputes_nondivision_scores_instead_of_trusting_stale_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(include_loose=True, include_one_child=False)
    monkeypatch.setattr(
        class_zero_module,
        "calculate_legacy_nondivision_scores",
        lambda *_args, **_kwargs: (1.0, 2.0),
    )

    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (100.0, -100.0)),
        _model(),
        _statistics(),
    )

    assert result.context.successor_slots("parent") == ("d1", None)
    assert result.context.successor_slots("loose") == ("d2", None)
    assert result.diagnostics.nondivision_scores == (1.0, 2.0)


def test_nested_other_rejection_continues_to_next_raw_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(include_loose=True, include_one_child=True)
    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        lambda _context, parent, _statistics, **_kwargs: _extraction(
            parent, "existing", "d1", (1.0, 2.0)
        ),
    )
    monkeypatch.setattr(
        class_zero_module,
        "classify_single_model",
        lambda *_args, **_kwargs: _prediction(0),
    )

    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
    )

    assert result.status == "direct_attached"
    assert result.diagnostics.attempts[0].outcome == "tentative_other_rejected"
    assert result.diagnostics.attempts[1].outcome == "direct_attached"
    assert result.diagnostics.reattached_to_id == "loose"


def test_nested_division_prediction_commits_provisional_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(include_loose=False, include_one_child=True)
    record_answer_values: list[bool] = []

    def extract_nested(
        _context: LegacyTrackingContext,
        parent: str,
        _statistics_value: LegacyTrackingStatistics,
        *,
        record_answers: bool,
    ) -> LegacyBifurcationExtraction:
        record_answer_values.append(record_answers)
        return _extraction(parent, "existing", "d1", (1.0, 1.0))

    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        extract_nested,
    )
    monkeypatch.setattr(
        class_zero_module,
        "classify_single_model",
        lambda *_args, **_kwargs: _prediction(1),
    )

    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
        record_answers=True,
    )

    assert result.status == "division_committed"
    assert result.context.successor_slots("one") == ("existing", "d1")
    assert result.diagnostics.attempts[0].outcome == "tentative_division_committed"
    assert record_answer_values == [True]
    with pytest.raises(EventTraceFormatError, match="nested classifier events"):
        trace_from_legacy_class_zero_results(context, (result,))


def test_nested_repair_accepts_ambigious_classifier_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(include_loose=False, include_one_child=True)
    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        lambda _context, parent, _statistics, **_kwargs: _extraction(
            parent,
            "existing",
            "d1",
            (1.0, 1.0),
        ),
    )

    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _ambigious_model(),
        _statistics(),
    )

    prediction = result.diagnostics.attempts[0].prediction
    assert result.status == "division_committed"
    assert isinstance(prediction, AmbigiousClassifierPrediction)
    assert prediction.predicted_class == 1


def test_nested_false_positive_prediction_deletes_shorter_provisional_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(include_loose=False, include_one_child=True)
    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        lambda _context, parent, _statistics, **_kwargs: _extraction(
            parent,
            "existing",
            "d1",
            (1.0, 1.0),
        ),
    )
    monkeypatch.setattr(
        class_zero_module,
        "classify_single_model",
        lambda *_args, **_kwargs: _prediction(3),
    )

    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
    )

    assert result.status == "false_positive_committed"
    assert "existing" in result.context.deleted_ids
    assert result.context.successor_slots("one") == ("d1", None)
    assert result.diagnostics.attempts[0].outcome == (
        "tentative_false_positive_committed"
    )


def test_nested_other_recurses_when_existing_first_daughter_is_worse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(include_loose=True, include_one_child=True)
    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        lambda _context, parent, _statistics, **_kwargs: _extraction(
            parent, "existing", "d1", (2.0, 1.0)
        ),
    )
    monkeypatch.setattr(
        class_zero_module,
        "classify_single_model",
        lambda *_args, **_kwargs: _prediction(0),
    )

    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
    )

    assert result.status == "recursive_other_committed"
    assert result.diagnostics.attempts[0].outcome == "tentative_other_recursed"
    assert result.context.successor_slots("one") == ("d1", None)
    assert result.context.successor_slots("loose") == ("existing", None)


def test_deleted_loose_candidate_retains_raw_attachment_without_resurrection() -> None:
    context = _context(
        include_loose=True,
        include_one_child=False,
        deleted_loose=True,
    )
    state = context.to_lineage_graph_state()

    result = repair_legacy_class_zero_bifurcation(
        context,
        state,
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
    )

    assert result.status == "direct_attached"
    assert result.context.deleted_ids == frozenset({"loose"})
    assert result.context.successor_slots("parent") == ("d2", None)
    assert result.context.successor_slots("loose") == ("d1", None)
    assert result.context.predecessor("d1") == "loose"
    assert result.state.predecessor("d1") is None
    assert not any(edge.source_id == "loose" for edge in result.state.edges)
    assert result.diagnostics.reattached_to_id == "loose"
    assert result.diagnostics.attempts[0].outcome == (
        "direct_attached_deleted_source"
    )
    assert result.actions[1].details["matlab_delete_flag_preserved"] is True
    assert context.successor_slots("loose") == (None, None)
    assert state.predecessor("d1") == "parent"


def test_deleted_loose_attachment_is_preserved_in_raw_event_trace() -> None:
    context = _context(
        include_loose=True,
        include_one_child=False,
        deleted_loose=True,
    )
    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
    )

    trace = trace_from_legacy_class_zero_results(context, (result,))
    final = trace.checkpoints[-1]
    deleted_source = final.by_id["loose"]
    detached_daughter = final.by_id["d1"]

    assert deleted_source.deleted
    assert deleted_source.successor_slots == ("d1", None)
    assert detached_daughter.predecessor_id == "loose"
    assert not any(edge.source_id == "loose" for edge in result.state.edges)
    assert trace.classifications[0].effective_class == 0

    drifted_nodes = tuple(
        replace(node, successor_slots=(None, None))
        if node.node_id == "loose"
        else replace(node, predecessor_id=None)
        if node.node_id == "d1"
        else node
        for node in final.nodes
    )
    drifted_checkpoints = (
        *trace.checkpoints[:-1],
        replace(final, nodes=drifted_nodes),
    )
    drifted = TrackingEventTrace(
        drifted_checkpoints,
        trace.classifications,
        trace.provenance,
    )
    comparison = compare_event_traces(trace, drifted)

    assert not comparison.exact_match
    assert comparison.classification_order_match
    assert not comparison.mutation_batch_match


def test_deleted_one_child_source_is_temporarily_active_for_division_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        include_loose=False,
        include_one_child=True,
        deleted_one_child=True,
    )
    state = context.to_lineage_graph_state()
    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        lambda _context, parent, _statistics, **_kwargs: _extraction(
            parent,
            "existing",
            "d1",
            (1.0, 1.0),
        ),
    )
    monkeypatch.setattr(
        class_zero_module,
        "classify_single_model",
        lambda *_args, **_kwargs: _prediction(1),
    )

    result = repair_legacy_class_zero_bifurcation(
        context,
        state,
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
    )

    assert result.status == "division_committed"
    assert result.context.deleted_ids == frozenset({"one"})
    assert result.context.successor_slots("one") == ("existing", "d1")
    assert result.context.predecessor("d1") == "one"
    assert result.state.predecessor("d1") is None
    assert result.state.predecessor("existing") is None
    assert not any(edge.source_id == "one" for edge in result.state.edges)
    assert result.diagnostics.reattached_to_id == "one"
    assert "restored atomically" in result.diagnostics.attempts[0].notes[0]
    assert context.successor_slots("one") == ("existing", None)
    assert state.deleted_ids == frozenset({"one"})


def test_deleted_one_child_false_positive_preserves_matlab_slot_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        include_loose=False,
        include_one_child=True,
        deleted_one_child=True,
    )
    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        lambda _context, parent, _statistics, **_kwargs: _extraction(
            parent,
            "existing",
            "d1",
            (1.0, 1.0),
        ),
    )
    monkeypatch.setattr(
        class_zero_module,
        "classify_single_model",
        lambda *_args, **_kwargs: _prediction(3),
    )

    result = repair_legacy_class_zero_bifurcation(
        context,
        context.to_lineage_graph_state(),
        _extraction("parent", "d1", "d2", (2.0, 1.0)),
        _model(),
        _statistics(),
    )

    assert result.status == "false_positive_committed"
    assert result.context.deleted_ids == frozenset({"one", "existing"})
    assert result.context.successor_slots("one") == ("d1", None)
    assert result.context.predecessor("existing") is None
    assert result.context.predecessor("d1") == "one"
    assert result.state.predecessor("d1") is None
    assert result.diagnostics.attempts[0].outcome == (
        "tentative_false_positive_committed"
    )
