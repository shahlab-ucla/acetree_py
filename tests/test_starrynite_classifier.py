"""Neutral StarryNite classifier and single-model feature conformance tests."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from acetree_py.tracking.starrynite.classifier import (
    AmbigiousClassifierPrediction,
    AssembledAmbigiousClassifierFeatures,
    CategoricalFeatureDistribution,
    ClassifierPredictionError,
    GaussianFeatureDistribution,
    GaussianKernelFeatureDistribution,
    LEGACY_SINGLE_MODEL_FEATURE_COUNT,
    NEUTRAL_CLASSIFIER_SCHEMA,
    NEUTRAL_CLASSIFIER_VERSION,
    NeutralAmbigiousClassifierFamily,
    NeutralClassifierFormatError,
    NeutralNaiveBayesClassifier,
    NeutralNaiveBayesSubmodel,
    SingleModelFeatureInput,
    SingleModelFeatureLayout,
    assemble_ambigious_classifier_features,
    assemble_single_model_features,
    classify_ambigious_family,
    classify_single_model,
    load_neutral_ambigious_classifier,
    load_neutral_classifier,
    neutral_ambigious_classifier_from_matlab_export,
    neutral_classifier_from_matlab_export,
    save_neutral_ambigious_classifier,
    save_neutral_classifier,
)


_SOURCE_SHA = "a" * 64
_STANDARD_COST = (
    (0.0, 1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 0.0),
)
_CATEGORY_PROBABILITIES = (
    (0.60, 0.10, 0.10, 0.10, 0.10),
    (0.30, 0.40, 0.10, 0.10, 0.10),
    (0.20, 0.10, 0.40, 0.20, 0.10),
    (0.10, 0.10, 0.10, 0.60, 0.10),
)


def _legacy_21_layout() -> SingleModelFeatureLayout:
    return SingleModelFeatureLayout(
        daughter_keep=(True,) * 12 + (False,) * 10,
        backward_keep=(False,) * 11,
        forward_keep=(True,) * 8 + (False,) * 5,
    )


def _small_layout() -> SingleModelFeatureLayout:
    return SingleModelFeatureLayout(
        daughter_keep=(True,) + (False,) * 21,
        backward_keep=(True,) + (False,) * 10,
        forward_keep=(True,) + (False,) * 12,
    )


def _categorical(
    probabilities: tuple[tuple[float, ...], ...] = _CATEGORY_PROBABILITIES,
) -> CategoricalFeatureDistribution:
    return CategoricalFeatureDistribution(
        categories=(1.0, 2.0, 3.0, 4.0, 5.0),
        probabilities=probabilities,
    )


def _gaussian() -> GaussianFeatureDistribution:
    return GaussianFeatureDistribution(
        means=(0.0, 1.0, 2.0, 3.0),
        standard_deviations=(1.0, 1.0, 1.0, 1.0),
    )


def _fake_model(
    *,
    layout: SingleModelFeatureLayout | None = None,
    distributions=None,
    costs=_STANDARD_COST,
) -> NeutralNaiveBayesClassifier:
    layout = layout or _legacy_21_layout()
    if distributions is None:
        distributions = (_categorical(),) + tuple(
            _gaussian() for _ in range(layout.selected_feature_count)
        )
    return NeutralNaiveBayesClassifier(
        source_model_sha256=_SOURCE_SHA,
        classifier_family="new_classifier",
        feature_layout=layout,
        feature_names=("topology_class",)
        + tuple(f"feature_{index}" for index in range(layout.selected_feature_count)),
        class_labels=(0, 1, 2, 3),
        class_priors=(0.25, 0.25, 0.25, 0.25),
        misclassification_costs=costs,
        distributions=tuple(distributions),
    )


def _fake_matlab_export() -> dict[str, object]:
    """Synthetic shape emitted by the MATLAB classifier export operation."""

    predictor_count = 4
    distribution_names = np.asarray(
        ["mvmn", "normal", "kernel", "normal"],
        dtype=object,
    )
    categorical_levels = np.empty(predictor_count, dtype=object)
    categorical_levels[0] = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    for predictor in range(1, predictor_count):
        categorical_levels[predictor] = np.asarray([], dtype=float)
    kernel_names = np.asarray(["", "", "normal", ""], dtype=object)
    support_names = np.asarray(["", "", "unbounded", ""], dtype=object)
    widths = np.full((4, predictor_count), np.nan)
    distributions = np.empty((4, predictor_count), dtype=object)

    for class_index in range(4):
        for predictor_index, distribution_name in enumerate(distribution_names):
            item: dict[str, object] = {
                "class_index_1based": np.uint32(class_index + 1),
                "predictor_index_1based": np.uint32(predictor_index + 1),
                "distribution_name": distribution_name,
                "numeric_parameters": np.asarray([], dtype=float),
                "categorical_levels": categorical_levels[predictor_index],
                "kernel_name": kernel_names[predictor_index],
                "support": support_names[predictor_index],
                "bandwidth": np.asarray([], dtype=float),
                "input_data": np.asarray([], dtype=float),
                "input_frequency": np.asarray([], dtype=float),
                "input_censored": np.asarray([], dtype=float),
                "truncation": np.asarray([], dtype=float),
                "is_truncated": np.bool_(False),
            }
            if distribution_name == "mvmn":
                item["numeric_parameters"] = np.asarray(
                    _CATEGORY_PROBABILITIES[class_index]
                )
            elif distribution_name == "normal":
                item["numeric_parameters"] = np.asarray(
                    [class_index + 10.0 * predictor_index, 1.0 + predictor_index]
                )
            else:
                bandwidth = 0.5 + class_index * 0.1
                widths[class_index, predictor_index] = bandwidth
                item["bandwidth"] = bandwidth
                item["input_data"] = np.asarray(
                    [float(class_index), class_index + 0.5]
                )
                item["input_frequency"] = np.asarray([1.0, 2.0])
                item["input_censored"] = np.asarray([0.0, 0.0])
                item["truncation"] = np.asarray([-np.inf, np.inf])
            distributions[class_index, predictor_index] = item

    return {
        "matlab_class": "ClassificationNaiveBayes",
        "score_transform": "none",
        "standardization_state": "none",
        "mu": np.asarray([], dtype=float),
        "sigma": np.asarray([], dtype=float),
        "class_names": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "prior": np.asarray([0.25, 0.25, 0.25, 0.25]),
        "cost": np.asarray(_STANDARD_COST),
        "predictor_names": np.asarray(
            ["topology_class", "daughter_0", "back_0", "forward_0"],
            dtype=object,
        ),
        "categorical_predictors_1based": np.uint32(1),
        "num_observations": np.uint32(100),
        "distribution_names": distribution_names,
        "categorical_levels": categorical_levels,
        "kernel_names": kernel_names,
        "support_names": support_names,
        "width": widths,
        "distributions": distributions,
        "daughter_keep": np.asarray([True] + [False] * 21),
        "back_keep": np.asarray([True] + [False] * 10),
        "forward_keep": np.asarray([True] + [False] * 12),
        "selected_feature_count": np.uint32(4),
    }


def _feature_input(
    *,
    lengths=(6.0, 6.0),
    backward=(False, False),
    forward=(-1.0, -1.0),
    cutoff=5.0,
) -> SingleModelFeatureInput:
    return SingleModelFeatureInput(
        daughter_features=tuple(float(index) for index in range(22)),
        backward_features=tuple(100.0 + index for index in range(11)),
        forward_features=tuple(200.0 + index for index in range(13)),
        daughter_lengths=lengths,
        backward_candidate_present=backward,
        best_forward_lengths=forward,
        small_cutoff=cutoff,
    )


def _submodel(
    predictor_count: int,
    labels: tuple[int, ...],
    priors: tuple[float, ...],
) -> NeutralNaiveBayesSubmodel:
    class_count = len(labels)
    costs = tuple(
        tuple(0.0 if row == column else 1.0 for column in range(class_count))
        for row in range(class_count)
    )
    return NeutralNaiveBayesSubmodel(
        classifier_family="legacy_classifier",
        feature_names=tuple(f"feature_{index}" for index in range(predictor_count)),
        class_labels=labels,
        class_priors=priors,
        misclassification_costs=costs,
        distributions=tuple(
            GaussianFeatureDistribution(
                means=(0.0,) * class_count,
                standard_deviations=(1.0,) * class_count,
            )
            for _ in range(predictor_count)
        ),
    )


def _ambigious_family() -> NeutralAmbigiousClassifierFamily:
    return NeutralAmbigiousClassifierFamily(
        source_model_sha256=_SOURCE_SHA,
        feature_layout=_small_layout(),
        ambigious=_submodel(3, (0, 2, 3), (0.1, 0.8, 0.1)),
        fp_div=_submodel(1, (0, 1, 3), (0.1, 0.8, 0.1)),
        dirtyfp_fn=_submodel(2, (0, 1, 2, 3), (0.1, 0.1, 0.7, 0.1)),
        divfp=_submodel(2, (0, 1, 3), (0.1, 0.1, 0.8)),
    )


def test_legacy_layout_assembles_exact_21_features_in_matlab_block_order():
    layout = _legacy_21_layout()
    inputs = _feature_input(
        lengths=(3.0, 6.0),
        backward=(True, False),
        forward=(2.0, -1.0),
    )

    assembled = assemble_single_model_features(inputs, layout)

    assert layout.predictor_count == LEGACY_SINGLE_MODEL_FEATURE_COUNT
    assert assembled.topology_case == "truly_ambiguous"
    assert assembled.topology_class == 1
    assert assembled.values == (
        (1.0,)
        + tuple(float(index) for index in range(12))
        + tuple(200.0 + index for index in range(8))
    )


@pytest.mark.parametrize(
    (
        "lengths",
        "backward",
        "forward",
        "topology_class",
        "topology_case",
        "backward_is_missing",
        "forward_is_missing",
    ),
    (
        (
            (6.0, 6.0),
            (False, False),
            (-1.0, -1.0),
            2,
            "fully_division_looking",
            True,
            True,
        ),
        (
            (6.0, 6.0),
            (True, False),
            (-1.0, -1.0),
            3,
            "false_negative_division_looking",
            False,
            True,
        ),
        (
            (3.0, 6.0),
            (False, False),
            (-1.0, -1.0),
            4,
            "fully_false_positive_looking",
            True,
            True,
        ),
        (
            (3.0, 6.0),
            (True, False),
            (-1.0, -1.0),
            5,
            "dirty_false_positive_looking",
            False,
            True,
        ),
        (
            (3.0, 6.0),
            (False, False),
            (2.0, -1.0),
            5,
            "division_false_positive_looking",
            True,
            True,
        ),
        (
            (3.0, 6.0),
            (True, False),
            (2.0, -1.0),
            1,
            "truly_ambiguous",
            False,
            False,
        ),
    ),
)
def test_all_single_model_topology_branches_apply_exact_missing_blocks(
    lengths,
    backward,
    forward,
    topology_class,
    topology_case,
    backward_is_missing,
    forward_is_missing,
):
    result = assemble_single_model_features(
        _feature_input(lengths=lengths, backward=backward, forward=forward),
        _small_layout(),
    )

    assert result.topology_class == topology_class
    assert result.topology_case == topology_case
    assert result.values[1] == 0.0
    assert math.isnan(result.values[2]) is backward_is_missing
    assert math.isnan(result.values[3]) is forward_is_missing
    if not backward_is_missing:
        assert result.values[2] == 100.0
    if not forward_is_missing:
        assert result.values[3] == 200.0


@pytest.mark.parametrize(
    ("lengths", "backward", "forward", "submodel", "expected"),
    (
        ((6.0, 6.0), (False, False), (-1.0, -1.0), "fp_div", (0.0,)),
        (
            (6.0, 6.0),
            (True, False),
            (-1.0, -1.0),
            "dirtyfp_fn",
            (0.0, 100.0),
        ),
        ((3.0, 6.0), (False, False), (-1.0, -1.0), "fp_div", (0.0,)),
        (
            (3.0, 6.0),
            (True, False),
            (-1.0, -1.0),
            "dirtyfp_fn",
            (0.0, 100.0),
        ),
        (
            (3.0, 6.0),
            (False, False),
            (2.0, -1.0),
            "divfp",
            (0.0, 200.0),
        ),
        (
            (3.0, 6.0),
            (True, False),
            (2.0, -1.0),
            "ambigious",
            (0.0, 100.0, 200.0),
        ),
    ),
)
def test_ambigious_family_routes_exact_matlab_blocks(
    lengths,
    backward,
    forward,
    submodel,
    expected,
):
    assembled = assemble_ambigious_classifier_features(
        _feature_input(lengths=lengths, backward=backward, forward=forward),
        _small_layout(),
    )

    assert isinstance(assembled, AssembledAmbigiousClassifierFeatures)
    assert assembled.submodel_name == submodel
    assert assembled.values == expected
    assert assembled.values[0] == 0.0


@pytest.mark.parametrize(
    ("lengths", "backward", "forward", "submodel", "classification"),
    (
        ((6.0, 6.0), (False, False), (-1.0, -1.0), "fp_div", 1),
        ((6.0, 6.0), (True, False), (-1.0, -1.0), "dirtyfp_fn", 2),
        ((3.0, 6.0), (False, False), (2.0, -1.0), "divfp", 3),
        ((3.0, 6.0), (True, False), (2.0, -1.0), "ambigious", 2),
    ),
)
def test_ambigious_family_scores_the_selected_submodel_only(
    lengths,
    backward,
    forward,
    submodel,
    classification,
):
    prediction = classify_ambigious_family(
        _ambigious_family(),
        _feature_input(lengths=lengths, backward=backward, forward=forward),
    )

    assert isinstance(prediction, AmbigiousClassifierPrediction)
    assert prediction.submodel_name == submodel
    assert prediction.predicted_class == classification
    assert len(prediction.log_scores) == len(prediction.posterior)


def test_ambigious_force_mode_preserves_hard_coded_class_positions_and_fallback():
    family = _ambigious_family()
    other_ambigious = _submodel(3, (0, 2, 3), (0.8, 0.15, 0.05))
    family = NeutralAmbigiousClassifierFamily(
        source_model_sha256=_SOURCE_SHA,
        feature_layout=family.feature_layout,
        ambigious=other_ambigious,
        fp_div=family.fp_div,
        dirtyfp_fn=family.dirtyfp_fn,
        divfp=family.divfp,
    )
    inputs = _feature_input(
        lengths=(3.0, 6.0),
        backward=(True, False),
        forward=(2.0, -1.0),
    )

    ordinary = classify_ambigious_family(family, inputs)
    forced = classify_ambigious_family(family, inputs, force_mode=True)
    unavailable = classify_ambigious_family(
        family,
        inputs,
        force_mode=True,
        backward_repair_available=False,
    )

    assert ordinary.predicted_class == 0
    assert forced.computed_class == 2
    assert forced.predicted_class == 2
    assert unavailable.computed_class == 2
    assert unavailable.predicted_class == 0


def test_single_class_null_submodel_stays_valid_outside_a_classifier_family():
    family = _ambigious_family()
    null_model = _submodel(1, (0,), (1.0,))

    assert null_model.predict((0.0,)) == 0
    with pytest.raises(
        NeutralClassifierFormatError,
        match="fp_div class_labels must exactly match",
    ):
        NeutralAmbigiousClassifierFamily(
            source_model_sha256=_SOURCE_SHA,
            feature_layout=family.feature_layout,
            ambigious=family.ambigious,
            fp_div=null_model,
            dirtyfp_fn=family.dirtyfp_fn,
            divfp=family.divfp,
        )


@pytest.mark.parametrize(
    ("submodel_name", "labels"),
    (
        ("ambigious", (0, 3, 2)),
        ("fp_div", (0, 3, 1)),
        ("dirtyfp_fn", (0, 2, 1, 3)),
        ("divfp", (0, 3, 1)),
    ),
)
def test_ambigious_family_rejects_reordered_positional_class_labels(
    submodel_name,
    labels,
):
    family = _ambigious_family()
    models = family.submodels
    original = models[submodel_name]
    models[submodel_name] = _submodel(
        original.predictor_count,
        labels,
        (1.0 / len(labels),) * len(labels),
    )

    with pytest.raises(
        NeutralClassifierFormatError,
        match=rf"{submodel_name} class_labels must exactly match",
    ):
        NeutralAmbigiousClassifierFamily(
            source_model_sha256=_SOURCE_SHA,
            feature_layout=family.feature_layout,
            ambigious=models["ambigious"],
            fp_div=models["fp_div"],
            dirtyfp_fn=models["dirtyfp_fn"],
            divfp=models["divfp"],
        )


def test_ambigious_family_rejects_labels_from_another_topology_branch():
    family = _ambigious_family()

    with pytest.raises(
        NeutralClassifierFormatError,
        match=r"ambigious class_labels must exactly match.*\(0, 2, 3\)",
    ):
        NeutralAmbigiousClassifierFamily(
            source_model_sha256=_SOURCE_SHA,
            feature_layout=family.feature_layout,
            ambigious=_submodel(3, (0, 1, 3), (0.1, 0.8, 0.1)),
            fp_div=family.fp_div,
            dirtyfp_fn=family.dirtyfp_fn,
            divfp=family.divfp,
        )


def test_ambigious_family_json_round_trip_and_source_binding(tmp_path: Path):
    path = tmp_path / "ambigious-classifier.json"
    family = _ambigious_family()

    save_neutral_ambigious_classifier(path, family)
    restored = load_neutral_ambigious_classifier(
        path,
        expected_source_model_sha256=_SOURCE_SHA,
    )

    assert restored == family
    assert restored.to_dict()["submodels"]["ambigious"]
    with pytest.raises(NeutralClassifierFormatError, match="not the requested"):
        load_neutral_ambigious_classifier(
            path,
            expected_source_model_sha256="b" * 64,
        )


def test_ambigious_family_json_rejects_reordered_class_labels(tmp_path: Path):
    payload = _ambigious_family().to_dict()
    payload["submodels"]["divfp"]["class_labels"] = [0, 3, 1]
    path = tmp_path / "reordered-ambigious-classifier.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        NeutralClassifierFormatError,
        match="divfp class_labels must exactly match",
    ):
        load_neutral_ambigious_classifier(path)


def test_ambigious_family_accepts_only_explicit_inert_numeric_export():
    family = _ambigious_family()
    exported = {
        "family_name": "ambigious",
        "daughter_keep": np.asarray(family.feature_layout.daughter_keep),
        "back_keep": np.asarray(family.feature_layout.backward_keep),
        "forward_keep": np.asarray(family.feature_layout.forward_keep),
        "submodels": {
            name: submodel.to_dict()
            for name, submodel in family.submodels.items()
        },
    }

    converted = neutral_ambigious_classifier_from_matlab_export(
        {"classifier_family_model": exported},
        source_model_sha256=_SOURCE_SHA,
    )

    assert converted == family
    exported["family_name"] = "ambiguous"
    with pytest.raises(NeutralClassifierFormatError, match="legacy spelling"):
        neutral_ambigious_classifier_from_matlab_export(
            exported,
            source_model_sha256=_SOURCE_SHA,
        )


def test_ambigious_matlab_export_rejects_illegal_branch_class_labels():
    family = _ambigious_family()
    exported = {
        "family_name": "ambigious",
        "daughter_keep": np.asarray(family.feature_layout.daughter_keep),
        "back_keep": np.asarray(family.feature_layout.backward_keep),
        "forward_keep": np.asarray(family.feature_layout.forward_keep),
        "submodels": {
            name: submodel.to_dict()
            for name, submodel in family.submodels.items()
        },
    }
    exported["submodels"]["fp_div"]["class_labels"] = [0, 1, 2]

    with pytest.raises(
        NeutralClassifierFormatError,
        match="fp_div class_labels must exactly match",
    ):
        neutral_ambigious_classifier_from_matlab_export(
            exported,
            source_model_sha256=_SOURCE_SHA,
        )


def test_ambigious_family_rejects_mask_submodel_shape_drift():
    family = _ambigious_family()

    with pytest.raises(NeutralClassifierFormatError, match="masks select 3"):
        NeutralAmbigiousClassifierFamily(
            source_model_sha256=_SOURCE_SHA,
            feature_layout=family.feature_layout,
            ambigious=_submodel(2, (0, 2, 3), (0.1, 0.8, 0.1)),
            fp_div=family.fp_div,
            dirtyfp_fn=family.dirtyfp_fn,
            divfp=family.divfp,
        )


def test_nan_predictors_are_omitted_and_scores_use_log_likelihoods():
    model = _fake_model()
    features = (1.0,) + (math.nan,) * 20

    scores = model.log_scores(features)
    posterior = model.posterior(features)

    expected_scores = tuple(
        math.log(0.25) + math.log(row[0]) for row in _CATEGORY_PROBABILITIES
    )
    assert scores == pytest.approx(expected_scores)
    assert posterior == pytest.approx((0.5, 0.25, 1.0 / 6.0, 1.0 / 12.0))
    assert sum(posterior) == pytest.approx(1.0)
    assert model.predict(features) == 0


def test_all_missing_predictors_fall_back_to_prior_and_ties_are_deterministic():
    model = _fake_model()

    assert model.log_scores((math.nan,) * 21) == pytest.approx(
        (math.log(0.25),) * 4
    )
    assert model.posterior((math.nan,) * 21) == pytest.approx((0.25,) * 4)
    assert model.predict((math.nan,) * 21) == 0


def test_all_zero_numeric_likelihoods_fall_back_to_prior():
    layout = SingleModelFeatureLayout(
        daughter_keep=(True,) + (False,) * 21,
        backward_keep=(False,) * 11,
        forward_keep=(False,) * 13,
    )
    model = _fake_model(layout=layout)

    assert model.posterior((2.0, 1e308)) == pytest.approx((0.25,) * 4)
    assert model.predict((2.0, 1e308)) == 0


def test_high_level_classification_scores_each_distribution_only_once():
    class CountingDistribution:
        class_count = 4

        def __init__(self):
            self.calls = 0

        def log_probability(self, value, class_index):
            self.calls += 1
            return 0.0

    layout = SingleModelFeatureLayout(
        daughter_keep=(True,) + (False,) * 21,
        backward_keep=(False,) * 11,
        forward_keep=(False,) * 13,
    )
    counter = CountingDistribution()
    model = _fake_model(
        layout=layout,
        distributions=(_categorical(), counter),
    )

    result = classify_single_model(model, _feature_input())

    assert result.predicted_class in {0, 1, 2, 3}
    assert counter.calls == 4


def test_legacy_force_mode_excludes_other_and_false_positive_classes():
    model = _fake_model()
    features = (1.0,) + (math.nan,) * 20

    assert model.predict(features) == 0
    assert model.predict(features, force_mode=True) == 1


def test_false_negative_class_is_demoted_without_a_backward_repair_option():
    model = _fake_model()
    # Category 3 has its largest likelihood in class 2.
    features = (3.0,) + (math.nan,) * 20

    assert model.predict(features, backward_repair_available=True) == 2
    assert model.predict(features, backward_repair_available=False) == 0


def test_high_level_single_model_prediction_retains_reproducible_diagnostics():
    model = _fake_model()
    inputs = _feature_input(
        lengths=(3.0, 6.0),
        backward=(True, False),
        forward=(2.0, -1.0),
    )

    result = classify_single_model(model, inputs)

    assert result.topology_class == 1
    assert result.topology_case == "truly_ambiguous"
    assert len(result.features) == 21
    assert len(result.log_scores) == 4
    assert sum(result.posterior) == pytest.approx(1.0)
    assert result.predicted_class in {0, 1, 2, 3}


def test_gaussian_kernel_likelihood_uses_class_specific_samples_weights_and_width():
    kernel = GaussianKernelFeatureDistribution(
        samples=((0.0,), (1.0,), (2.0,), (3.0,)),
        frequencies=((1.0,), (1.0,), (1.0,), (1.0,)),
        bandwidths=(0.25, 0.25, 0.25, 0.25),
    )
    expected_at_center = -math.log(0.25) - 0.5 * math.log(2.0 * math.pi)

    assert kernel.log_probability(2.0, 2) == pytest.approx(expected_at_center)
    assert kernel.log_probability(2.0, 0) < kernel.log_probability(2.0, 2)

    layout = _legacy_21_layout()
    uniform_topology = _categorical(((0.2,) * 5,) * 4)
    distributions = (uniform_topology, kernel) + tuple(_gaussian() for _ in range(19))
    model = _fake_model(layout=layout, distributions=distributions)
    assert model.predict((1.0, 2.0) + (math.nan,) * 19) == 2


def test_kernel_frequency_weights_are_normalized_like_matlab_pdf():
    kernel = GaussianKernelFeatureDistribution(
        samples=((0.0, 2.0),) * 4,
        frequencies=((1.0, 3.0),) * 4,
        bandwidths=(1.0,) * 4,
    )
    first = math.exp(-0.5 * (2.0**2)) / math.sqrt(2.0 * math.pi)
    second = 1.0 / math.sqrt(2.0 * math.pi)

    assert math.exp(kernel.log_probability(2.0, 0)) == pytest.approx(
        (first + 3.0 * second) / 4.0
    )


def test_kernel_far_tail_underflow_uses_matlab_prior_fallback():
    kernel = GaussianKernelFeatureDistribution(
        samples=((0.0,), (1.0,), (2.0,), (3.0,)),
        frequencies=((1.0,),) * 4,
        bandwidths=(0.25,) * 4,
    )
    layout = SingleModelFeatureLayout(
        daughter_keep=(True,) + (False,) * 21,
        backward_keep=(False,) * 11,
        forward_keep=(False,) * 13,
    )
    model = _fake_model(
        layout=layout,
        distributions=(_categorical(((0.2,) * 5,) * 4), kernel),
    )

    assert kernel.log_probability(1e6, 0) == -math.inf
    assert model.posterior((1.0, 1e6)) == pytest.approx((0.25,) * 4)


def test_neutral_json_round_trip_preserves_model_and_binds_source_hash(tmp_path: Path):
    model = _fake_model()
    path = tmp_path / "classifier.json"

    save_neutral_classifier(path, model)
    restored = load_neutral_classifier(
        path,
        expected_source_model_sha256=_SOURCE_SHA,
    )

    assert restored == model
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == NEUTRAL_CLASSIFIER_SCHEMA
    assert payload["version"] == NEUTRAL_CLASSIFIER_VERSION
    assert payload["source_model_sha256"] == _SOURCE_SHA
    assert "NaN" not in path.read_text(encoding="utf-8")

    with pytest.raises(NeutralClassifierFormatError, match="not the requested"):
        load_neutral_classifier(
            path,
            expected_source_model_sha256="b" * 64,
        )


def test_matlab_export_bridge_consumes_oracle_c_by_p_struct_without_matlab_objects():
    exported = _fake_matlab_export()

    model = neutral_classifier_from_matlab_export(
        {"success": True, "classifier_model": exported},
        source_model_sha256=_SOURCE_SHA,
    )

    assert model.classifier_family == "new_classifier"
    assert model.feature_layout == _small_layout()
    assert model.feature_names == (
        "topology_class",
        "daughter_0",
        "back_0",
        "forward_0",
    )
    assert isinstance(model.distributions[0], CategoricalFeatureDistribution)
    assert isinstance(model.distributions[1], GaussianFeatureDistribution)
    assert isinstance(model.distributions[2], GaussianKernelFeatureDistribution)
    kernel = model.distributions[2]
    assert kernel.samples[3] == pytest.approx((3.0, 3.5))
    assert kernel.frequencies[0] == pytest.approx((1.0, 2.0))
    assert all(math.isfinite(value) for value in model.log_scores((1.0, 0.0, 1.0, 0.0)))


def test_matlab_export_bridge_accepts_numeric_legacy_naive_bayes_export():
    exported = _fake_matlab_export()
    exported["matlab_class"] = "NaiveBayes"

    model = neutral_classifier_from_matlab_export(
        exported,
        source_model_sha256=_SOURCE_SHA,
    )

    assert model.classifier_family == "legacy_classifier"
    assert model.feature_layout == _small_layout()


def test_matlab_export_bridge_rejects_schema_and_matrix_shape_drift():
    exported = _fake_matlab_export()
    exported["future_field"] = True

    with pytest.raises(NeutralClassifierFormatError, match="unknown field"):
        neutral_classifier_from_matlab_export(
            exported,
            source_model_sha256=_SOURCE_SHA,
        )

    exported = _fake_matlab_export()
    exported["distributions"] = exported["distributions"][:, :3]
    with pytest.raises(NeutralClassifierFormatError, match=r"shape \(4, 4\)"):
        neutral_classifier_from_matlab_export(
            exported,
            source_model_sha256=_SOURCE_SHA,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("score_transform", "logit", "score_transform"),
        ("mu", np.asarray([0.0, 1.0, 0.0, 0.0]), "nonidentity"),
        ("sigma", np.asarray([1.0, 2.0, 1.0, 1.0]), "nonidentity"),
        ("standardization_state", "identity", "disagrees"),
    ),
)
def test_matlab_export_bridge_rejects_unmodeled_prediction_transforms(
    field,
    value,
    message,
):
    exported = _fake_matlab_export()
    exported[field] = value

    with pytest.raises(NeutralClassifierFormatError, match=message):
        neutral_classifier_from_matlab_export(
            exported,
            source_model_sha256=_SOURCE_SHA,
        )


def test_matlab_export_bridge_accepts_explicit_identity_standardization():
    exported = _fake_matlab_export()
    exported["standardization_state"] = "identity"
    exported["mu"] = np.zeros(4)
    exported["sigma"] = np.ones(4)

    model = neutral_classifier_from_matlab_export(
        exported,
        source_model_sha256=_SOURCE_SHA,
    )

    assert model.predictor_count == 4


def test_matlab_export_bridge_rejects_corrupt_indices_and_censored_kernels():
    exported = _fake_matlab_export()
    entries = exported["distributions"].copy()
    corrupt = dict(entries[1, 2])
    corrupt["class_index_1based"] = 4
    entries[1, 2] = corrupt
    exported["distributions"] = entries

    with pytest.raises(NeutralClassifierFormatError, match="index metadata"):
        neutral_classifier_from_matlab_export(
            exported,
            source_model_sha256=_SOURCE_SHA,
        )

    exported = _fake_matlab_export()
    entries = exported["distributions"].copy()
    censored = dict(entries[0, 2])
    censored["input_censored"] = np.asarray([1.0, 0.0])
    entries[0, 2] = censored
    exported["distributions"] = entries
    with pytest.raises(NeutralClassifierFormatError, match="censored kernel"):
        neutral_classifier_from_matlab_export(
            exported,
            source_model_sha256=_SOURCE_SHA,
        )


def test_neutral_model_and_nested_parameters_are_immutable():
    model = _fake_model()

    with pytest.raises(FrozenInstanceError):
        model.class_priors = (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(TypeError):
        model.distributions[0].probabilities[0][0] = 1.0


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda: SingleModelFeatureLayout(
                daughter_keep=(True,) * 21,
                backward_keep=(False,) * 11,
                forward_keep=(False,) * 13,
            ),
            "exactly 22",
        ),
        (
            lambda: GaussianFeatureDistribution(
                means=(0.0,) * 4,
                standard_deviations=(1.0, 1.0, 0.0, 1.0),
            ),
            "must be positive",
        ),
        (
            lambda: GaussianKernelFeatureDistribution(
                samples=((0.0,),) * 4,
                frequencies=((1.0,),) * 4,
                bandwidths=(1.0, 1.0, -1.0, 1.0),
            ),
            "bandwidths must be positive",
        ),
        (
            lambda: GaussianKernelFeatureDistribution(
                samples=((0.0,),) * 4,
                frequencies=((1.0,),) * 4,
                bandwidths=(1.0,) * 4,
                kernel="epanechnikov",
            ),
            "Unsupported kernel",
        ),
    ),
)
def test_invalid_numeric_model_state_fails_with_actionable_messages(factory, message):
    with pytest.raises(NeutralClassifierFormatError, match=message):
        factory()


def test_model_validation_rejects_bad_hash_and_layout_distribution_mismatch():
    with pytest.raises(NeutralClassifierFormatError, match="source_model_sha256"):
        NeutralNaiveBayesClassifier(
            source_model_sha256="not-a-hash",
            classifier_family="new_classifier",
            feature_layout=_legacy_21_layout(),
            feature_names=("x",) * 21,
            class_labels=(0, 1, 2, 3),
            class_priors=(0.25,) * 4,
            misclassification_costs=_STANDARD_COST,
            distributions=(_categorical(),) + tuple(_gaussian() for _ in range(20)),
        )

    with pytest.raises(NeutralClassifierFormatError, match="disagree"):
        _fake_model(distributions=(_categorical(), _gaussian()))


def test_unknown_json_distribution_and_fields_fail_closed(tmp_path: Path):
    payload = _fake_model().to_dict()
    payload["distributions"][1] = {"kind": "mystery"}
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NeutralClassifierFormatError, match="unsupported"):
        load_neutral_classifier(path)

    payload = _fake_model().to_dict()
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NeutralClassifierFormatError, match="unknown field"):
        load_neutral_classifier(path)


def test_prediction_rejects_wrong_length_but_infinity_and_unknown_category_use_prior():
    model = _fake_model()

    with pytest.raises(ClassifierPredictionError, match="expects 21"):
        model.predict((1.0,))
    infinite = (1.0, math.inf) + (math.nan,) * 19
    assert model.posterior(infinite) == pytest.approx((0.25,) * 4)
    assert model.predict(infinite) == 0
    unknown = (6.0,) + (math.nan,) * 20
    assert model.posterior(unknown) == pytest.approx((0.25,) * 4)
    assert model.predict(unknown) == 0


def test_nonstandard_cost_matrix_changes_ordinary_decision_but_not_force_rule():
    # Always choosing class 1 is cheapest under these costs, despite class 0
    # having the largest posterior for topology category 1.
    costs = (
        (2.0, 0.0, 3.0, 3.0),
        (2.0, 0.0, 3.0, 3.0),
        (2.0, 0.0, 3.0, 3.0),
        (2.0, 0.0, 3.0, 3.0),
    )
    model = _fake_model(costs=costs)

    assert model.predict((1.0,) + (math.nan,) * 20) == 1
