"""Tests for the exact-feature classifier-to-lineage boundary."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from acetree_py.tracking.api import Detection, TrackEdge
from acetree_py.tracking.starrynite.bifurcation import (
    BifurcationOrchestrationError,
    SingleModelLineageRequest,
    classify_and_resolve_bifurcation,
)
from acetree_py.tracking.starrynite.classifier import (
    CategoricalFeatureDistribution,
    GaussianFeatureDistribution,
    NeutralAmbigiousClassifierFamily,
    NeutralNaiveBayesClassifier,
    NeutralNaiveBayesSubmodel,
    SingleModelFeatureInput,
    SingleModelFeatureLayout,
)
from acetree_py.tracking.starrynite.lineage import (
    FalseNegativeRewirePlan,
    LineageGraphState,
    LineageReattachmentCandidate,
)


_STANDARD_COSTS = (
    (0.0, 1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 0.0),
)


def _detection(identifier: str, frame: int) -> Detection:
    return Detection(identifier, frame, float(frame), 0.0, 0.0, 1.0, 1.0)


def _edge(
    source: str,
    target: str,
    *,
    kind: str = "link",
    cost: float = 1.0,
) -> TrackEdge:
    return TrackEdge(source, target, cost, kind)


def _state() -> LineageGraphState:
    return LineageGraphState.from_detections(
        (
            _detection("gap_source", 1),
            _detection("parent", 2),
            _detection("alternative", 2),
            _detection("daughter1", 3),
            _detection("daughter2", 3),
        ),
        (
            _edge("parent", "daughter1", kind="split"),
            _edge("parent", "daughter2", kind="split"),
        ),
    )


def _features() -> SingleModelFeatureInput:
    return SingleModelFeatureInput(
        daughter_features=(0.0,) * 22,
        backward_features=(0.0,) * 11,
        forward_features=(0.0,) * 13,
        daughter_lengths=(6.0, 6.0),
        backward_candidate_present=(False, False),
        best_forward_lengths=(-1.0, -1.0),
        small_cutoff=5.0,
    )


def _model_for_class(classification: int) -> NeutralNaiveBayesClassifier:
    priors = [0.01, 0.01, 0.01, 0.01]
    priors[classification] = 0.97
    layout = SingleModelFeatureLayout(
        daughter_keep=(False,) * 22,
        backward_keep=(False,) * 11,
        forward_keep=(False,) * 13,
    )
    topology = CategoricalFeatureDistribution(
        categories=(1.0, 2.0, 3.0, 4.0, 5.0),
        probabilities=((0.2,) * 5,) * 4,
    )
    return NeutralNaiveBayesClassifier(
        source_model_sha256="b" * 64,
        classifier_family="new_classifier",
        feature_layout=layout,
        feature_names=("topology_class",),
        class_labels=(0, 1, 2, 3),
        class_priors=tuple(priors),
        misclassification_costs=_STANDARD_COSTS,
        distributions=(topology,),
    )


def _ambigious_submodel(
    predictor_count: int,
    labels: tuple[int, ...],
    priors: tuple[float, ...],
) -> NeutralNaiveBayesSubmodel:
    class_count = len(labels)
    return NeutralNaiveBayesSubmodel(
        classifier_family="legacy_classifier",
        feature_names=tuple(f"feature_{index}" for index in range(predictor_count)),
        class_labels=labels,
        class_priors=priors,
        misclassification_costs=tuple(
            tuple(0.0 if row == column else 1.0 for column in range(class_count))
            for row in range(class_count)
        ),
        distributions=tuple(
            GaussianFeatureDistribution(
                means=(0.0,) * class_count,
                standard_deviations=(1.0,) * class_count,
            )
            for _ in range(predictor_count)
        ),
    )


def _ambigious_family() -> NeutralAmbigiousClassifierFamily:
    layout = SingleModelFeatureLayout(
        daughter_keep=(True,) + (False,) * 21,
        backward_keep=(True,) + (False,) * 10,
        forward_keep=(True,) + (False,) * 12,
    )
    return NeutralAmbigiousClassifierFamily(
        source_model_sha256="c" * 64,
        feature_layout=layout,
        ambigious=_ambigious_submodel(3, (0, 2, 3), (0.1, 0.8, 0.1)),
        fp_div=_ambigious_submodel(1, (0, 1, 3), (0.1, 0.8, 0.1)),
        dirtyfp_fn=_ambigious_submodel(
            2,
            (0, 1, 2, 3),
            (0.1, 0.1, 0.7, 0.1),
        ),
        divfp=_ambigious_submodel(2, (0, 1, 3), (0.1, 0.1, 0.8)),
    )


def _request() -> SingleModelLineageRequest:
    gap_plan = FalseNegativeRewirePlan(
        remove_edges=(("parent", "daughter1"),),
        add_edges=(_edge("gap_source", "daughter1", kind="gap"),),
        label="exact-candidate-plan",
    )
    return SingleModelLineageRequest(
        state=_state(),
        features=_features(),
        parent_id="parent",
        daughter1_id="daughter1",
        daughter2_id="daughter2",
        daughter1_nondivision_score=8.0,
        daughter2_nondivision_score=2.0,
        reattachment_candidates=(
            LineageReattachmentCandidate("alternative", 0.25),
        ),
        false_negative_plan=gap_plan,
    )


@pytest.mark.parametrize("classification", (0, 1, 2, 3))
def test_all_four_classifier_classes_flow_through_lineage_resolution(classification):
    result = classify_and_resolve_bifurcation(
        _model_for_class(classification),
        _request(),
    )

    assert result.prediction.predicted_class == classification
    assert result.decision.classification == classification
    assert result.resolution.diagnostics.classification == classification
    assert result.prediction_diagnostics is result.prediction
    assert result.resolution_diagnostics is result.resolution.diagnostics
    pairs = {
        (edge.source_id, edge.target_id, edge.kind)
        for edge in result.resolution.state.edges
    }
    if classification == 0:
        assert result.resolution_diagnostics.detached_daughter_id == "daughter1"
        assert result.resolution_diagnostics.reattached_to_id == "alternative"
        assert ("alternative", "daughter1", "link") in pairs
        remaining_parent_edge = next(
            edge
            for edge in result.resolution.state.edges
            if edge.source_id == "parent"
        )
        assert remaining_parent_edge.features["LEGACY_SUCCESSOR_SLOT"] == 0
    elif classification == 1:
        assert pairs == {
            ("parent", "daughter1", "split"),
            ("parent", "daughter2", "split"),
        }
    elif classification == 2:
        assert ("gap_source", "daughter1", "gap") in pairs
        assert result.resolution_diagnostics.applied_gap_edges == (
            ("gap_source", "daughter1"),
        )
    else:
        assert result.resolution.state.deleted_ids == frozenset({"daughter1"})
        assert pairs == {("parent", "daughter2", "link")}


def test_legacy_ambigious_family_flows_through_lineage_resolution():
    result = classify_and_resolve_bifurcation(_ambigious_family(), _request())

    assert result.prediction.submodel_name == "fp_div"
    assert result.prediction.predicted_class == 1
    assert result.decision.classification == 1
    assert result.resolution.diagnostics.classification == 1


def test_false_negative_prediction_is_demoted_without_an_explicit_rewire_plan():
    request = SingleModelLineageRequest(
        state=_state(),
        features=_features(),
        parent_id="parent",
        daughter1_id="daughter1",
        daughter2_id="daughter2",
        daughter1_nondivision_score=4.0,
        daughter2_nondivision_score=1.0,
    )

    result = classify_and_resolve_bifurcation(_model_for_class(2), request)

    assert result.prediction.posterior[2] > 0.9
    assert result.prediction.predicted_class == 0
    assert result.resolution_diagnostics.classification == 0


def test_class_zero_requires_both_exact_nondivision_scores():
    request = SingleModelLineageRequest(
        state=_state(),
        features=_features(),
        parent_id="parent",
        daughter1_id="daughter1",
        daughter2_id="daughter2",
    )

    with pytest.raises(BifurcationOrchestrationError, match="class 0"):
        classify_and_resolve_bifurcation(_model_for_class(0), request)


def test_class_zero_uses_matlab_strict_comparison_for_nan_tail_score():
    request = SingleModelLineageRequest(
        state=_state(),
        features=_features(),
        parent_id="parent",
        daughter1_id="daughter1",
        daughter2_id="daughter2",
        daughter1_nondivision_score=math.nan,
        daughter2_nondivision_score=1.0,
    )

    result = classify_and_resolve_bifurcation(_model_for_class(0), request)

    # MATLAB's ``scores(1) > scores(2)`` is false when d1 is NaN, so d2 is
    # the detached branch.
    assert result.resolution_diagnostics.detached_daughter_id == "daughter2"


def test_request_fails_closed_on_inexact_types_and_invalid_graph_context():
    with pytest.raises(TypeError, match="exact SingleModelFeatureInput"):
        SingleModelLineageRequest(
            state=_state(),
            features=(0.0,) * 21,  # type: ignore[arg-type]
            parent_id="parent",
            daughter1_id="daughter1",
            daughter2_id="daughter2",
        )

    with pytest.raises(BifurcationOrchestrationError, match="both be supplied"):
        SingleModelLineageRequest(
            state=_state(),
            features=_features(),
            parent_id="parent",
            daughter1_id="daughter1",
            daughter2_id="daughter2",
            daughter1_nondivision_score=1.0,
        )

    state_without_split = LineageGraphState(_state().frames)
    with pytest.raises(BifurcationOrchestrationError, match="exactly the two"):
        SingleModelLineageRequest(
            state=state_without_split,
            features=_features(),
            parent_id="parent",
            daughter1_id="daughter1",
            daughter2_id="daughter2",
        )


def test_request_and_result_are_immutable():
    request = _request()
    result = classify_and_resolve_bifurcation(_model_for_class(1), request)

    with pytest.raises(FrozenInstanceError):
        request.parent_id = "different"
    with pytest.raises(FrozenInstanceError):
        result.prediction = result.prediction


def test_orchestrator_validates_public_argument_types():
    with pytest.raises(TypeError, match="model"):
        classify_and_resolve_bifurcation(object(), _request())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="request"):
        classify_and_resolve_bifurcation(
            _model_for_class(1),
            object(),  # type: ignore[arg-type]
        )
