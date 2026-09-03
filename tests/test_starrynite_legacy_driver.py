"""Focused tests for the dynamic whole-movie legacy decision pass."""

from __future__ import annotations

from dataclasses import replace

import pytest

import acetree_py.tracking.starrynite.legacy_class_zero as class_zero_module
import acetree_py.tracking.starrynite.legacy_driver as driver_module
from acetree_py.tracking.api import TrackEdge
from acetree_py.tracking.starrynite.classifier import (
    AmbigiousClassifierPrediction,
    CategoricalFeatureDistribution,
    GaussianFeatureDistribution,
    NeutralAmbigiousClassifierFamily,
    NeutralNaiveBayesClassifier,
    NeutralNaiveBayesSubmodel,
    SingleModelFeatureInput,
    SingleModelFeatureLayout,
    SingleModelPrediction,
)
from acetree_py.tracking.starrynite.legacy_driver import (
    LegacyMovieDecisionConfig,
    run_legacy_movie_decisions,
)
from acetree_py.tracking.starrynite.legacy_features import (
    BACKWARD_FEATURE_NAMES,
    DAUGHTER_FEATURE_NAMES,
    FORWARD_FEATURE_NAMES,
    LegacyBifurcationExtraction,
    LegacyFeatureExtractionError,
    LegacyTrackingStatistics,
)
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
)
from acetree_py.tracking.starrynite.lineage import FalseNegativeRewirePlan
from acetree_py.tracking.starrynite.repair_candidates import BackwardRepairCandidates
from acetree_py.tracking.starrynite.oracle.event_trace import (
    EventTraceFormatError,
    trace_from_legacy_movie_result,
)


def _nucleus(
    identifier: str,
    frame: int,
    row: int,
    x: float,
    *,
    total: float = 10.0,
    average: float = 2.0,
) -> LegacyNucleus:
    return LegacyNucleus(
        identifier,
        frame,
        row,
        (x, 0.0, 1.0),
        4.0,
        total,
        average,
        1.0,
        1.0,
        2,
        2.0,
        1.0,
    )


def _edge(source: str, target: str, slot: int = 0, *, kind: str | None = None) -> TrackEdge:
    return TrackEdge(
        source,
        target,
        0.0,
        kind or ("split" if slot == 1 else "link"),
        {"LEGACY_SUCCESSOR_SLOT": slot},
    )


def _parameters(end_frame: int, *, small_cutoff: float = 2.0) -> LegacyFeatureParameters:
    return LegacyFeatureParameters(
        interval=1.0,
        candidate_cutoff=2.0,
        temporal_cutoff=max(2, end_frame),
        temporal_cutoff_start=2,
        small_cutoff=small_cutoff,
        anisotropy_xyz=(1.0, 1.0, 1.0),
        end_frame=end_frame,
    )


def _context(
    nuclei: list[LegacyNucleus],
    edges: list[TrackEdge],
    end_frame: int,
) -> LegacyTrackingContext:
    return LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        edges,
        _parameters(end_frame),
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


def _model_for_class(classification: int = 1) -> NeutralNaiveBayesClassifier:
    priors = [0.01] * 4
    priors[classification] = 0.97
    return NeutralNaiveBayesClassifier(
        source_model_sha256="e" * 64,
        classifier_family="new_classifier",
        feature_layout=SingleModelFeatureLayout(
            (False,) * 22,
            (False,) * 11,
            (False,) * 13,
        ),
        feature_names=("topology_class",),
        class_labels=(0, 1, 2, 3),
        class_priors=tuple(priors),
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


def _ambigious_family_for_division() -> NeutralAmbigiousClassifierFamily:
    def submodel(
        labels: tuple[int, ...],
        predicted: int,
    ) -> NeutralNaiveBayesSubmodel:
        remainder = 0.2 / (len(labels) - 1)
        priors = tuple(0.8 if label == predicted else remainder for label in labels)
        return NeutralNaiveBayesSubmodel(
            classifier_family="legacy_classifier",
            feature_names=("daughter_feature_1",),
            class_labels=labels,
            class_priors=priors,
            misclassification_costs=tuple(
                tuple(float(row != column) for column in range(len(labels)))
                for row in range(len(labels))
            ),
            distributions=(
                GaussianFeatureDistribution(
                    means=(0.0,) * len(labels),
                    standard_deviations=(1.0,) * len(labels),
                ),
            ),
        )

    return NeutralAmbigiousClassifierFamily(
        source_model_sha256="f" * 64,
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
    return SingleModelPrediction(
        predicted_class=classification,
        topology_class=2,
        topology_case="test",
        features=(2.0,),
        log_scores=(0.0,) * 4,
        posterior=tuple(
            1.0 if index == classification else 0.0 for index in range(4)
        ),
    )


def _extraction(
    context: LegacyTrackingContext,
    parent_id: str,
    *,
    scores: tuple[float, float] = (2.0, 1.0),
    false_negative_plan: FalseNegativeRewirePlan | None = None,
) -> LegacyBifurcationExtraction:
    daughters = context.successor_slots(parent_id)
    assert daughters[0] is not None and daughters[1] is not None
    feature_input = SingleModelFeatureInput(
        daughter_features=(0.0,) * 22,
        backward_features=(0.0,) * 11,
        forward_features=(0.0,) * 13,
        daughter_lengths=(1.0, 1.0),
        backward_candidate_present=(false_negative_plan is not None, False),
        best_forward_lengths=(-1.0, -1.0),
        small_cutoff=2.0,
    )
    repairs = BackwardRepairCandidates((), (), None, -1.0, -1.0)
    return LegacyBifurcationExtraction(
        parent_id,
        (daughters[0], daughters[1]),
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
        false_negative_plan,
    )


def _real_feature_context() -> LegacyTrackingContext:
    nuclei = [
        _nucleus("a", 1, 0, 0.0),
        _nucleus("n1", 1, 1, 10.0),
        _nucleus("p", 2, 0, 0.0, total=100.0, average=10.0),
        _nucleus("n2", 2, 1, 10.0),
        _nucleus("d1", 3, 0, 1.0, total=60.0, average=6.0),
        _nucleus("d2", 3, 1, -1.0, total=50.0, average=5.0),
        _nucleus("n3", 3, 2, 10.0),
        _nucleus("d14", 4, 0, 1.0, total=60.0, average=6.0),
        _nucleus("d24", 4, 1, -1.0, total=50.0, average=5.0),
        _nucleus("n4", 4, 2, 10.0),
    ]
    edges = [
        _edge("a", "p"),
        _edge("n1", "n2"),
        _edge("p", "d1", 0),
        _edge("p", "d2", 1),
        _edge("n2", "n3"),
        _edge("d1", "d14"),
        _edge("d2", "d24"),
        _edge("n3", "n4"),
    ]
    return _context(nuclei, edges, 4)


def test_real_feature_extractor_and_classifier_complete_movie_pass() -> None:
    context = _real_feature_context()

    result = run_legacy_movie_decisions(
        context,
        _model_for_class(1),
        _statistics(),
    )

    assert result.status == "completed"
    assert result.context is context
    assert result.state == context.to_lineage_graph_state()
    assert [event.parent_id for event in result.events] == ["p"]
    assert [item.effective_class for item in result.classifications] == [1]


def test_whole_movie_pass_accepts_strict_legacy_ambigious_family() -> None:
    result = run_legacy_movie_decisions(
        _real_feature_context(),
        _ambigious_family_for_division(),
        _statistics(),
    )

    assert result.supported
    assert result.classifier_family == "legacy_classifier"
    assert result.classifier_mode == "ambigious_four_model"
    assert len(result.classifications) == 1
    prediction = result.classifications[0].prediction
    assert isinstance(prediction, AmbigiousClassifierPrediction)
    assert prediction.submodel_name == "fp_div"
    assert prediction.predicted_class == 1


def test_scan_uses_frame_then_original_matlab_row_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        [
            _nucleus("p0", 1, 0, 0.0),
            _nucleus("p1", 1, 1, 10.0),
            _nucleus("a0", 2, 0, 0.0),
            _nucleus("b0", 2, 1, 1.0),
            _nucleus("a1", 2, 2, 10.0),
            _nucleus("b1", 2, 3, 11.0),
        ],
        [
            _edge("p0", "a0", 0),
            _edge("p0", "b0", 1),
            _edge("p1", "a1", 0),
            _edge("p1", "b1", 1),
        ],
        2,
    )
    extracted: list[str] = []

    def extract(current, parent, *_args, **_kwargs):
        extracted.append(parent)
        return _extraction(current, parent)

    monkeypatch.setattr(driver_module, "extract_legacy_bifurcation_features", extract)
    monkeypatch.setattr(driver_module, "_predict", lambda *_args, **_kwargs: _prediction(1))

    result = run_legacy_movie_decisions(context, _model_for_class(), _statistics())

    assert result.supported
    assert extracted == ["p0", "p1"]
    assert [event.parent_id for event in result.events] == ["p0", "p1"]


def test_earlier_false_positive_deletion_skips_later_parent_and_keeps_raw_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        [
            _nucleus("p0", 1, 0, 0.0),
            _nucleus("neighbor", 1, 1, 20.0),
            _nucleus("p1", 2, 0, 0.0),
            _nucleus("long1", 2, 1, 10.0),
            _nucleus("n2", 2, 2, 20.0),
            _nucleus("c1", 3, 0, 0.0),
            _nucleus("c2", 3, 1, 1.0),
            _nucleus("long2", 3, 2, 10.0),
            _nucleus("long3", 4, 0, 10.0),
            _nucleus("n4", 4, 1, 20.0),
        ],
        [
            _edge("p0", "p1", 0),
            _edge("p0", "long1", 1),
            _edge("p1", "c1", 0),
            _edge("p1", "c2", 1),
            _edge("long1", "long2"),
            _edge("long2", "long3"),
        ],
        4,
    )
    extracted: list[str] = []

    def extract(current, parent, *_args, **_kwargs):
        extracted.append(parent)
        return _extraction(current, parent)

    monkeypatch.setattr(driver_module, "extract_legacy_bifurcation_features", extract)
    monkeypatch.setattr(driver_module, "_predict", lambda *_args, **_kwargs: _prediction(3))

    result = run_legacy_movie_decisions(context, _model_for_class(), _statistics())

    assert result.supported
    assert extracted == ["p0"]
    assert result.context.deleted_ids == frozenset({"p1", "c1"})
    assert result.context.successor_slots("p0") == ("long1", None)
    # processFP marks the first-successor path deleted but does not clear the
    # condemned row's own successor slots, including its active side branch.
    assert result.context.successor_slots("p1") == ("c1", "c2")
    assert result.context.predecessor("c2") == "p1"
    assert result.state.predecessor("c2") is None


def test_false_positive_choice_uses_raw_path_through_predeleted_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nuclei = [
        _nucleus("parent", 1, 0, 0.0),
        _nucleus("neighbor", 1, 1, 20.0),
        _nucleus("d1", 2, 0, 0.0),
        _nucleus("d2", 2, 1, 10.0),
        _nucleus("old_deleted", 3, 0, 0.0),
        _nucleus("n3", 3, 1, 20.0),
    ]
    context = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        (
            _edge("parent", "d1", 0),
            _edge("parent", "d2", 1),
            _edge("d1", "old_deleted", 0),
        ),
        _parameters(3),
        deleted_ids=("old_deleted",),
    )
    monkeypatch.setattr(
        driver_module,
        "extract_legacy_bifurcation_features",
        lambda current, parent, *_args, **_kwargs: _extraction(current, parent),
    )
    monkeypatch.setattr(
        driver_module,
        "_predict",
        lambda *_args, **_kwargs: _prediction(3),
    )

    result = run_legacy_movie_decisions(context, _model_for_class(), _statistics())

    assert result.supported
    # The active graph sees a 1-vs-1 tie; raw MATLAB slots see d1 length 2, so
    # the true shorter branch is d2.
    assert result.context.deleted_ids == frozenset({"old_deleted", "d2"})
    assert result.context.successor_slots("parent") == ("d1", None)
    assert result.context.successor_slots("d1") == ("old_deleted", None)
    assert result.events[0].lineage_diagnostics is not None
    assert result.events[0].lineage_diagnostics.selected_daughter_id == "d2"


def test_class_two_rewire_is_synchronized_back_to_ordered_raw_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        [
            _nucleus("gap", 1, 0, 0.0),
            _nucleus("n1", 1, 1, 20.0),
            _nucleus("parent", 2, 0, 10.0),
            _nucleus("n2", 2, 1, 20.0),
            _nucleus("d1", 3, 0, 0.0),
            _nucleus("d2", 3, 1, 10.0),
        ],
        [
            _edge("parent", "d1", 0),
            _edge("parent", "d2", 1),
        ],
        3,
    )
    plan = FalseNegativeRewirePlan(
        remove_edges=(("parent", "d1"),),
        add_edges=(_edge("gap", "d1", 0, kind="gap"),),
        label="test-gap",
    )
    monkeypatch.setattr(
        driver_module,
        "extract_legacy_bifurcation_features",
        lambda current, parent, *_args, **_kwargs: _extraction(
            current,
            parent,
            false_negative_plan=plan,
        ),
    )
    monkeypatch.setattr(driver_module, "_predict", lambda *_args, **_kwargs: _prediction(2))

    result = run_legacy_movie_decisions(context, _model_for_class(), _statistics())

    assert result.supported
    assert result.context.successor_slots("parent") == ("d2", None)
    assert result.context.successor_slots("gap") == ("d1", None)
    assert result.context.predecessor("d1") == "gap"
    assert result.events[0].lineage_diagnostics is not None
    assert result.events[0].lineage_diagnostics.applied_gap_edges == (
        ("gap", "d1"),
    )


def test_class_zero_round_two_is_inline_and_observer_sees_raw_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        [
            _nucleus("parent", 1, 0, 100.0),
            _nucleus("one", 1, 1, 0.0),
            _nucleus("d1", 2, 0, 0.0),
            _nucleus("d2", 2, 1, 100.0),
            _nucleus("existing", 2, 2, 30.0),
        ],
        [
            _edge("parent", "d1", 0),
            _edge("parent", "d2", 1),
            _edge("one", "existing", 0),
        ],
        2,
    )

    monkeypatch.setattr(
        driver_module,
        "extract_legacy_bifurcation_features",
        lambda current, parent, *_args, **_kwargs: _extraction(current, parent),
    )
    monkeypatch.setattr(
        driver_module,
        "_predict",
        lambda _model, extraction, **_kwargs: _prediction(
            0 if extraction.parent_id == "parent" else 1
        ),
    )
    monkeypatch.setattr(
        class_zero_module,
        "calculate_legacy_nondivision_scores",
        lambda *_args, **_kwargs: (2.0, 1.0),
    )
    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        lambda current, parent, *_args, **_kwargs: _extraction(current, parent),
    )
    monkeypatch.setattr(
        class_zero_module,
        "classify_single_model",
        lambda *_args, **_kwargs: _prediction(1),
    )
    observations = []

    result = run_legacy_movie_decisions(
        context,
        _model_for_class(),
        _statistics(),
        classification_observer=observations.append,
    )

    assert result.supported
    # MATLAB classifies the provisional split inline (round 2), then reaches
    # the later source row and classifies the committed split again in round 1.
    assert [
        (item.parent_id, item.classifier_round)
        for item in result.classifications
    ] == [("parent", 1), ("one", 2), ("one", 1)]
    assert [event.parent_id for event in result.events] == ["parent", "one"]
    nested = observations[1]
    assert nested.before_attachment_context.successor_slots("one") == (
        "existing",
        None,
    )
    assert nested.before_attachment_context.predecessor("d1") is None
    assert nested.classification_context.successor_slots("one") == (
        "existing",
        "d1",
    )
    assert nested.classification_context.predecessor("d1") == "one"
    assert nested.record.raw_attempt_rank == 1

    trace = trace_from_legacy_movie_result(context, result, observations)
    assert [
        (item.parent_id, item.classifier_round, item.effective_class)
        for item in trace.classifications
    ] == [("parent", 1, 0), ("one", 2, 1), ("one", 1, 1)]
    assert any(
        mutation.kind == "edge_added"
        and mutation.node_id == "one"
        and mutation.related_id == "d1"
        and mutation.slot == 1
        for event in trace.ordered_events
        if hasattr(event, "mutations")
        for mutation in event.mutations
    )

    fabricated_first = replace(
        observations[0],
        before_attachment_context=nested.before_attachment_context,
        classification_context=nested.before_attachment_context,
    )
    with pytest.raises(EventTraceFormatError, match="anchored to the initial"):
        trace_from_legacy_movie_result(
            context,
            result,
            (fabricated_first, *observations[1:]),
        )

    missing_attachment = replace(
        nested,
        classification_context=nested.before_attachment_context,
    )
    with pytest.raises(EventTraceFormatError, match="provisional attachment"):
        trace_from_legacy_movie_result(
            context,
            result,
            (observations[0], missing_attachment, *observations[2:]),
        )

    deleted_context = replace(context, deleted_ids=frozenset({"one"}))
    deleted_observations = []
    deleted_result = run_legacy_movie_decisions(
        deleted_context,
        _model_for_class(),
        _statistics(),
        classification_observer=deleted_observations.append,
    )

    assert deleted_result.supported
    assert [
        (item.parent_id, item.classifier_round)
        for item in deleted_result.classifications
    ] == [("parent", 1), ("one", 2)]
    assert deleted_result.context.deleted_ids == frozenset({"one"})
    deleted_trace = trace_from_legacy_movie_result(
        deleted_context,
        deleted_result,
        deleted_observations,
    )
    assert deleted_trace.checkpoints[2].by_id["one"].deleted
    assert deleted_trace.checkpoints[2].by_id["one"].successor_slots == (
        "existing",
        "d1",
    )


def test_nested_class_zero_false_positive_uses_raw_predeleted_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nuclei = [
        _nucleus("parent", 1, 0, 100.0),
        _nucleus("one", 1, 1, 0.0),
        _nucleus("d1", 2, 0, 0.0),
        _nucleus("d2", 2, 1, 100.0),
        _nucleus("existing", 2, 2, 30.0),
        _nucleus("old_deleted", 3, 0, 30.0),
        _nucleus("n3", 3, 1, 100.0),
    ]
    context = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        (
            _edge("parent", "d1", 0),
            _edge("parent", "d2", 1),
            _edge("one", "existing", 0),
            _edge("existing", "old_deleted", 0),
        ),
        _parameters(3),
        deleted_ids=("old_deleted",),
    )
    monkeypatch.setattr(
        driver_module,
        "extract_legacy_bifurcation_features",
        lambda current, parent, *_args, **_kwargs: _extraction(current, parent),
    )
    monkeypatch.setattr(
        driver_module,
        "_predict",
        lambda *_args, **_kwargs: _prediction(0),
    )
    monkeypatch.setattr(
        class_zero_module,
        "calculate_legacy_nondivision_scores",
        lambda *_args, **_kwargs: (2.0, 1.0),
    )
    monkeypatch.setattr(
        class_zero_module,
        "extract_legacy_bifurcation_features",
        lambda current, parent, *_args, **_kwargs: _extraction(current, parent),
    )
    monkeypatch.setattr(
        class_zero_module,
        "classify_single_model",
        lambda *_args, **_kwargs: _prediction(3),
    )

    result = run_legacy_movie_decisions(context, _model_for_class(), _statistics())

    assert result.supported
    assert result.context.deleted_ids == frozenset({"old_deleted", "d1"})
    assert result.context.successor_slots("parent") == ("d2", None)
    assert result.context.successor_slots("one") == ("existing", None)
    assert result.context.successor_slots("existing") == ("old_deleted", None)
    assert result.events[0].class_zero_status == "false_positive_committed"


def test_failure_after_mutating_prefix_rolls_back_entire_movie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        [
            _nucleus("p0", 1, 0, 0.0),
            _nucleus("p1", 1, 1, 10.0),
            _nucleus("a0", 2, 0, 0.0),
            _nucleus("b0", 2, 1, 1.0),
            _nucleus("a1", 2, 2, 10.0),
            _nucleus("b1", 2, 3, 11.0),
        ],
        [
            _edge("p0", "a0", 0),
            _edge("p0", "b0", 1),
            _edge("p1", "a1", 0),
            _edge("p1", "b1", 1),
        ],
        2,
    )

    def extract(current, parent, *_args, **_kwargs):
        if parent == "p1":
            raise LegacyFeatureExtractionError("deliberate unsupported boundary")
        return _extraction(current, parent)

    monkeypatch.setattr(driver_module, "extract_legacy_bifurcation_features", extract)
    monkeypatch.setattr(driver_module, "_predict", lambda *_args, **_kwargs: _prediction(3))

    result = run_legacy_movie_decisions(context, _model_for_class(), _statistics())

    assert not result.supported
    assert result.context is context
    assert result.state == context.to_lineage_graph_state()
    assert result.context.deleted_ids == frozenset()
    assert len(result.events) == 1
    assert result.events[0].deleted_ids_after
    assert result.failure is not None
    assert result.failure.stage == "feature_extraction"
    assert result.failure.parent_id == "p1"


def test_force_end_frame_uses_matlab_strict_less_than_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        [
            _nucleus("p1", 1, 0, 0.0),
            _nucleus("n1", 1, 1, 20.0),
            _nucleus("a1", 2, 0, 0.0),
            _nucleus("b1", 2, 1, 1.0),
            _nucleus("p2", 2, 2, 10.0),
            _nucleus("a2", 3, 0, 10.0),
            _nucleus("b2", 3, 1, 11.0),
        ],
        [
            _edge("p1", "a1", 0),
            _edge("p1", "b1", 1),
            _edge("p2", "a2", 0),
            _edge("p2", "b2", 1),
        ],
        3,
    )
    forced: list[bool] = []
    monkeypatch.setattr(
        driver_module,
        "extract_legacy_bifurcation_features",
        lambda current, parent, *_args, **_kwargs: _extraction(current, parent),
    )

    def predict(*_args, force_mode, **_kwargs):
        forced.append(force_mode)
        return _prediction(1)

    monkeypatch.setattr(driver_module, "_predict", predict)

    result = run_legacy_movie_decisions(
        context,
        _model_for_class(),
        _statistics(),
        config=LegacyMovieDecisionConfig(
            force_mode=True,
            force_end_frame=2,
        ),
    )

    assert result.supported
    assert forced == [True, False]


def test_driver_applies_isolated_fragment_prepass_before_classifier_scan() -> None:
    context = _context(
        [
            _nucleus("p", 1, 0, 0.0),
            _nucleus("d", 2, 0, 1.0),
            _nucleus("tail", 3, 0, 2.0),
        ],
        [_edge("p", "d"), _edge("d", "tail")],
        3,
    )

    result = run_legacy_movie_decisions(
        context,
        _model_for_class(),
        _statistics(),
        config=LegacyMovieDecisionConfig(
            end_frame=3,
            delete_isolated=True,
            fp_size_threshold=3,
            early_cell_threshold=250,
            fp_size_threshold_small=0,
        ),
    )

    assert result.status == "completed"
    assert result.events == ()
    assert result.classifications == ()
    assert result.context.deleted_ids == {"p", "d", "tail"}
    assert result.context.stale_predecessor_by_id == {"d": "p"}
    trace = trace_from_legacy_movie_result(context, result, ())
    assert len(trace.checkpoints) == 2
    assert not any(node.deleted for node in trace.checkpoints[0].nodes)
    assert {
        node.node_id for node in trace.checkpoints[-1].nodes if node.deleted
    } == {"p", "d", "tail"}
