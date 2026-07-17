"""Validated, implementation-neutral StarryNite bifurcation classifiers.

MATLAB ``NaiveBayes`` and ``ClassificationNaiveBayes`` objects are executable
objects, not portable data structures.  This module deliberately does not try
to deserialize those objects.  Instead it consumes a small JSON representation
containing only the numeric state exported by a trusted converter.

The model evaluates one bifurcation at a time.  Predictor zero is the
StarryNite topology class and the remaining predictors are selected, in order,
from the 22 daughter, 11 backward, and 13 forward measurements used by the
single-model classifier.  The bundled 2019 model selects 20 of those
measurements (21 predictors total); other legacy models may select more.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np


NEUTRAL_CLASSIFIER_SCHEMA = "acetree.starrynite-classifier"
NEUTRAL_CLASSIFIER_VERSION = 1
NEUTRAL_AMBIGIOUS_CLASSIFIER_SCHEMA = (
    "acetree.starrynite-ambigious-classifier-family"
)
NEUTRAL_AMBIGIOUS_CLASSIFIER_VERSION = 1
LEGACY_SINGLE_MODEL_FEATURE_COUNT = 21

_CLASS_LABELS = (0, 1, 2, 3)
_DAUGHTER_FEATURE_COUNT = 22
_BACKWARD_FEATURE_COUNT = 11
_FORWARD_FEATURE_COUNT = 13
_LOG_TWO_PI = math.log(2.0 * math.pi)
_PROBABILITY_SUM_TOLERANCE = 1e-8
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AMBIGIOUS_SUBMODEL_NAMES = ("ambigious", "fp_div", "dirtyfp_fn", "divfp")
_AMBIGIOUS_FORCE_CLASS_VALUES = {
    "ambigious": (0, 2, 3),
    "fp_div": (0, 1, 3),
    "dirtyfp_fn": (0, 1, 2, 3),
    "divfp": (0, 1, 3),
}


class StarryNiteClassifierError(ValueError):
    """Base class for neutral-model and prediction failures."""


class NeutralClassifierFormatError(StarryNiteClassifierError):
    """Raised when a neutral classifier is incomplete or inconsistent."""


class ClassifierPredictionError(StarryNiteClassifierError):
    """Raised when a valid model cannot score a supplied feature vector."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (str, bytes, bytearray, bool)) or type(value).__name__ == "bool_":
        raise NeutralClassifierFormatError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NeutralClassifierFormatError(
            f"{label} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise NeutralClassifierFormatError(f"{label} must be finite")
    return result


def _prediction_number(value: Any, label: str, *, allow_nan: bool = True) -> float:
    if isinstance(value, bool):
        raise ClassifierPredictionError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ClassifierPredictionError(f"{label} must be numeric") from exc
    if math.isinf(result) or (math.isnan(result) and not allow_nan):
        raise ClassifierPredictionError(f"{label} must be finite")
    return result


def _bool_mask(
    values: Sequence[bool],
    expected_length: int,
    label: str,
) -> tuple[bool, ...]:
    result = tuple(values)
    if len(result) != expected_length:
        raise NeutralClassifierFormatError(
            f"{label} must contain exactly {expected_length} booleans; "
            f"received {len(result)}"
        )
    if any(type(item) is not bool for item in result):
        raise NeutralClassifierFormatError(f"{label} must contain only booleans")
    return result


def _number_tuple(
    values: Sequence[Any],
    label: str,
    *,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    result = tuple(
        _finite_number(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if nonnegative and any(value < 0 for value in result):
        raise NeutralClassifierFormatError(f"{label} cannot contain negative values")
    return result


def _probability_rows(
    values: Sequence[Sequence[Any]],
    label: str,
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(
        _number_tuple(row, f"{label}[{index}]", nonnegative=True)
        for index, row in enumerate(values)
    )
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise NeutralClassifierFormatError(f"{label} rows must have equal lengths")
    for index, row in enumerate(rows):
        if not row:
            raise NeutralClassifierFormatError(f"{label}[{index}] cannot be empty")
        if abs(sum(row) - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise NeutralClassifierFormatError(
                f"{label}[{index}] must sum to 1; received {sum(row):.17g}"
            )
    return rows


@dataclass(frozen=True, slots=True)
class SingleModelFeatureLayout:
    """Boolean feature masks stored beside a single-model classifier."""

    daughter_keep: tuple[bool, ...]
    backward_keep: tuple[bool, ...]
    forward_keep: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "daughter_keep",
            _bool_mask(
                self.daughter_keep,
                _DAUGHTER_FEATURE_COUNT,
                "feature_layout.daughter_keep",
            ),
        )
        object.__setattr__(
            self,
            "backward_keep",
            _bool_mask(
                self.backward_keep,
                _BACKWARD_FEATURE_COUNT,
                "feature_layout.backward_keep",
            ),
        )
        object.__setattr__(
            self,
            "forward_keep",
            _bool_mask(
                self.forward_keep,
                _FORWARD_FEATURE_COUNT,
                "feature_layout.forward_keep",
            ),
        )

    @property
    def selected_feature_count(self) -> int:
        return sum(self.daughter_keep) + sum(self.backward_keep) + sum(
            self.forward_keep
        )

    @property
    def predictor_count(self) -> int:
        """Topology class plus every selected block feature."""

        return 1 + self.selected_feature_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "daughter_keep": list(self.daughter_keep),
            "backward_keep": list(self.backward_keep),
            "forward_keep": list(self.forward_keep),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SingleModelFeatureLayout:
        data = _mapping(value, "feature_layout")
        _require_keys(
            data,
            {"daughter_keep", "backward_keep", "forward_keep"},
            "feature_layout",
        )
        return cls(
            daughter_keep=tuple(_sequence(data["daughter_keep"], "daughter_keep")),
            backward_keep=tuple(_sequence(data["backward_keep"], "backward_keep")),
            forward_keep=tuple(_sequence(data["forward_keep"], "forward_keep")),
        )


@dataclass(frozen=True, slots=True)
class GaussianFeatureDistribution:
    """One normal predictor with class-specific mean and deviation."""

    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    kind: str = "gaussian"

    def __post_init__(self) -> None:
        if self.kind != "gaussian":
            raise NeutralClassifierFormatError(
                "GaussianFeatureDistribution.kind must be 'gaussian'"
            )
        means = _number_tuple(self.means, "gaussian.means")
        deviations = _number_tuple(
            self.standard_deviations,
            "gaussian.standard_deviations",
        )
        if len(means) != len(deviations) or not means:
            raise NeutralClassifierFormatError(
                "Gaussian means and standard deviations must have equal nonzero length"
            )
        if any(value <= 0 for value in deviations):
            raise NeutralClassifierFormatError(
                "Gaussian standard deviations must be positive"
            )
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "standard_deviations", deviations)

    @property
    def class_count(self) -> int:
        return len(self.means)

    def log_probability(self, value: float, class_index: int) -> float:
        mean = self.means[class_index]
        deviation = self.standard_deviations[class_index]
        normalized = (value - mean) / deviation
        try:
            square = normalized * normalized
        except OverflowError:  # pragma: no cover - platform float detail
            square = math.inf
        return -0.5 * _LOG_TWO_PI - math.log(deviation) - 0.5 * square

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "means": list(self.means),
            "standard_deviations": list(self.standard_deviations),
        }


@dataclass(frozen=True, slots=True)
class CategoricalFeatureDistribution:
    """One finite categorical predictor with class-conditional probabilities."""

    categories: tuple[float, ...]
    probabilities: tuple[tuple[float, ...], ...]
    kind: str = "categorical"

    def __post_init__(self) -> None:
        if self.kind != "categorical":
            raise NeutralClassifierFormatError(
                "CategoricalFeatureDistribution.kind must be 'categorical'"
            )
        categories = _number_tuple(self.categories, "categorical.categories")
        if not categories:
            raise NeutralClassifierFormatError("Categorical categories cannot be empty")
        if len(set(categories)) != len(categories):
            raise NeutralClassifierFormatError("Categorical categories must be unique")
        probabilities = _probability_rows(
            self.probabilities,
            "categorical.probabilities",
        )
        if any(len(row) != len(categories) for row in probabilities):
            raise NeutralClassifierFormatError(
                "Every categorical probability row must match the category count"
            )
        object.__setattr__(self, "categories", categories)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def class_count(self) -> int:
        return len(self.probabilities)

    def log_probability(self, value: float, class_index: int) -> float:
        try:
            category_index = self.categories.index(value)
        except ValueError:
            # MATLAB assigns zero likelihood to an unknown categorical level.
            # If every class is therefore impossible, the model-level prior
            # fallback applies.
            return -math.inf
        probability = self.probabilities[class_index][category_index]
        return -math.inf if probability == 0 else math.log(probability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "categories": list(self.categories),
            "probabilities": [list(row) for row in self.probabilities],
        }


@dataclass(frozen=True, slots=True)
class GaussianKernelFeatureDistribution:
    """Class-specific Gaussian kernel-density likelihoods.

    ``samples`` and ``frequencies`` are kept per class.  This mirrors the
    numeric ``InputData.data``/``InputData.freq`` state exported from MATLAB,
    while avoiding any dependency on a MATLAB classifier object.
    """

    samples: tuple[tuple[float, ...], ...]
    frequencies: tuple[tuple[float, ...], ...]
    bandwidths: tuple[float, ...]
    kernel: str = "normal"
    support: str = "unbounded"
    kind: str = "kernel"
    _sample_arrays: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _frequency_arrays: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.kind != "kernel":
            raise NeutralClassifierFormatError(
                "GaussianKernelFeatureDistribution.kind must be 'kernel'"
            )
        if self.kernel not in {"normal", "gaussian"}:
            raise NeutralClassifierFormatError(
                f"Unsupported kernel {self.kernel!r}; only MATLAB's normal kernel "
                "is supported"
            )
        if self.support not in {"unbounded", "unbounded-support"}:
            raise NeutralClassifierFormatError(
                f"Unsupported kernel support {self.support!r}; only unbounded "
                "Gaussian density is supported"
            )
        samples = tuple(
            _number_tuple(row, f"kernel.samples[{index}]")
            for index, row in enumerate(self.samples)
        )
        frequencies = tuple(
            _number_tuple(
                row,
                f"kernel.frequencies[{index}]",
                nonnegative=True,
            )
            for index, row in enumerate(self.frequencies)
        )
        bandwidths = _number_tuple(self.bandwidths, "kernel.bandwidths")
        if not samples:
            raise NeutralClassifierFormatError("Kernel samples cannot be empty")
        if not (len(samples) == len(frequencies) == len(bandwidths)):
            raise NeutralClassifierFormatError(
                "Kernel samples, frequencies, and bandwidths need one row per class"
            )
        for index, (class_samples, class_frequencies) in enumerate(
            zip(samples, frequencies, strict=True)
        ):
            if not class_samples:
                raise NeutralClassifierFormatError(
                    f"kernel.samples[{index}] cannot be empty"
                )
            if len(class_samples) != len(class_frequencies):
                raise NeutralClassifierFormatError(
                    f"kernel.frequencies[{index}] must match its sample count"
                )
            if sum(class_frequencies) <= 0:
                raise NeutralClassifierFormatError(
                    f"kernel.frequencies[{index}] must contain positive total weight"
                )
        if any(value <= 0 for value in bandwidths):
            raise NeutralClassifierFormatError("Kernel bandwidths must be positive")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "kernel", "normal")
        object.__setattr__(self, "support", "unbounded")
        sample_arrays = tuple(np.asarray(row, dtype=np.float64) for row in samples)
        frequency_arrays = tuple(
            np.asarray(row, dtype=np.float64) for row in frequencies
        )
        for array in (*sample_arrays, *frequency_arrays):
            array.setflags(write=False)
        object.__setattr__(self, "_sample_arrays", sample_arrays)
        object.__setattr__(self, "_frequency_arrays", frequency_arrays)

    @property
    def class_count(self) -> int:
        return len(self.samples)

    def log_probability(self, value: float, class_index: int) -> float:
        samples = self._sample_arrays[class_index]
        frequencies = self._frequency_arrays[class_index]
        bandwidth = self.bandwidths[class_index]
        total_frequency = float(np.sum(frequencies))
        # MATLAB's Naive Bayes path evaluates KernelDistribution.pdf first and
        # then takes its logarithm. Deliberately accumulate in probability
        # space so far-tail underflow and the ensuing prior fallback match the
        # legacy implementation instead of silently choosing a class from
        # numerically stable but non-legacy log-domain tails.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            normalized = (value - samples) / bandwidth
            kernels = np.exp(-0.5 * normalized * normalized)
            weighted_density = float(np.dot(frequencies, kernels))
        if weighted_density == 0.0:
            return -math.inf
        return (
            math.log(weighted_density)
            - math.log(total_frequency)
            - math.log(bandwidth)
            - 0.5 * _LOG_TWO_PI
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "kernel": self.kernel,
            "support": self.support,
            "samples": [list(row) for row in self.samples],
            "frequencies": [list(row) for row in self.frequencies],
            "bandwidths": list(self.bandwidths),
        }


FeatureDistribution: TypeAlias = (
    GaussianFeatureDistribution
    | CategoricalFeatureDistribution
    | GaussianKernelFeatureDistribution
)


@dataclass(frozen=True, slots=True)
class NeutralNaiveBayesClassifier:
    """Immutable numeric representation of a StarryNite Naive Bayes model."""

    source_model_sha256: str
    classifier_family: str
    feature_layout: SingleModelFeatureLayout
    feature_names: tuple[str, ...]
    class_labels: tuple[int, ...]
    class_priors: tuple[float, ...]
    misclassification_costs: tuple[tuple[float, ...], ...]
    distributions: tuple[FeatureDistribution, ...]
    missing_value_policy: str = "omit"
    tie_policy: str = "first_class"
    schema: str = NEUTRAL_CLASSIFIER_SCHEMA
    version: int = NEUTRAL_CLASSIFIER_VERSION

    def __post_init__(self) -> None:
        if self.schema != NEUTRAL_CLASSIFIER_SCHEMA:
            raise NeutralClassifierFormatError(
                f"Unsupported neutral classifier schema: {self.schema!r}"
            )
        if type(self.version) is not int or self.version != NEUTRAL_CLASSIFIER_VERSION:
            raise NeutralClassifierFormatError(
                f"Unsupported neutral classifier version: {self.version!r}"
            )
        if not _SHA256.fullmatch(str(self.source_model_sha256)):
            raise NeutralClassifierFormatError(
                "source_model_sha256 must be 64 lowercase hexadecimal characters"
            )
        if self.classifier_family not in {"legacy_classifier", "new_classifier"}:
            raise NeutralClassifierFormatError(
                "classifier_family must be 'legacy_classifier' or 'new_classifier'"
            )
        if not isinstance(self.feature_layout, SingleModelFeatureLayout):
            raise NeutralClassifierFormatError(
                "feature_layout must be a SingleModelFeatureLayout"
            )
        labels = tuple(self.class_labels)
        if any(type(value) is not int for value in labels) or labels != _CLASS_LABELS:
            raise NeutralClassifierFormatError(
                "StarryNite single-model class_labels must be ordered [0, 1, 2, 3]"
            )
        priors = _number_tuple(self.class_priors, "class_priors", nonnegative=True)
        if len(priors) != len(labels) or any(value <= 0 for value in priors):
            raise NeutralClassifierFormatError(
                "class_priors must contain one positive probability per class"
            )
        if abs(sum(priors) - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise NeutralClassifierFormatError(
                f"class_priors must sum to 1; received {sum(priors):.17g}"
            )
        costs = tuple(
            _number_tuple(row, f"misclassification_costs[{index}]", nonnegative=True)
            for index, row in enumerate(self.misclassification_costs)
        )
        if len(costs) != len(labels) or any(len(row) != len(labels) for row in costs):
            raise NeutralClassifierFormatError(
                "misclassification_costs must be a square class-by-class matrix"
            )
        names = tuple(str(name) for name in self.feature_names)
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise NeutralClassifierFormatError(
                "feature_names must be nonempty and unique"
            )
        distributions = tuple(self.distributions)
        expected = self.feature_layout.predictor_count
        if len(names) != expected or len(distributions) != expected:
            raise NeutralClassifierFormatError(
                "feature layout, names, and distributions disagree: "
                f"layout selects {expected} predictors, names contain {len(names)}, "
                f"and distributions contain {len(distributions)}"
            )
        if not isinstance(distributions[0], CategoricalFeatureDistribution):
            raise NeutralClassifierFormatError(
                "Predictor zero (the topology class) must be categorical"
            )
        required_topology = {1.0, 2.0, 3.0, 4.0, 5.0}
        if not required_topology.issubset(set(distributions[0].categories)):
            raise NeutralClassifierFormatError(
                "Topology distribution must define categories 1 through 5"
            )
        if any(distribution.class_count != len(labels) for distribution in distributions):
            raise NeutralClassifierFormatError(
                "Every predictor distribution needs one parameter set per class"
            )
        if self.missing_value_policy != "omit":
            raise NeutralClassifierFormatError(
                "Only the legacy HandleMissing='on'/'omit' policy is supported"
            )
        if self.tie_policy != "first_class":
            raise NeutralClassifierFormatError(
                "Only deterministic first-class tie handling is supported"
            )
        object.__setattr__(self, "source_model_sha256", str(self.source_model_sha256))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "class_labels", labels)
        object.__setattr__(self, "class_priors", priors)
        object.__setattr__(self, "misclassification_costs", costs)
        object.__setattr__(self, "distributions", distributions)

    @property
    def predictor_count(self) -> int:
        return len(self.distributions)

    def log_scores(self, features: Sequence[Any]) -> tuple[float, ...]:
        """Return unnormalized class log scores, omitting NaN predictors."""

        values = _prediction_features(features, self.predictor_count)
        scores = [math.log(value) for value in self.class_priors]
        observed = 0
        for feature_index, (value, distribution) in enumerate(
            zip(values, self.distributions, strict=True)
        ):
            if math.isnan(value):
                continue
            observed += 1
            for class_index in range(len(self.class_labels)):
                scores[class_index] += distribution.log_probability(
                    value,
                    class_index,
                )
        if observed == 0:
            # MATLAB's missing-feature handling falls back to the class prior.
            return tuple(scores)
        if all(score == -math.inf for score in scores):
            # ClassificationNaiveBayes falls back to the prior when every
            # class-conditional likelihood is numerically zero. Keeping this
            # behavior explicit also prevents an undefined 0/0 normalization.
            return tuple(math.log(value) for value in self.class_priors)
        if any(math.isnan(score) for score in scores):
            raise ClassifierPredictionError(
                "Classifier produced NaN log scores from otherwise valid inputs"
            )
        return tuple(scores)

    def posterior(self, features: Sequence[Any]) -> tuple[float, ...]:
        """Return normalized posterior class probabilities."""

        return self._posterior_from_log_scores(self.log_scores(features))

    def _predict_from_posterior(
        self,
        posterior: Sequence[float],
        *,
        force_mode: bool,
        backward_repair_available: bool,
    ) -> int:
        """Apply costs and StarryNite policy to one already-scored vector."""

        if type(force_mode) is not bool or type(backward_repair_available) is not bool:
            raise ClassifierPredictionError(
                "force_mode and backward_repair_available must be booleans"
            )
        values = tuple(float(value) for value in posterior)
        if len(values) != len(self.class_labels):
            raise ClassifierPredictionError(
                "posterior must contain one probability per classifier class"
            )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ClassifierPredictionError(
                "posterior probabilities must be finite and non-negative"
            )
        total = sum(values)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ClassifierPredictionError(
                f"posterior probabilities must sum to 1; received {total:.17g}"
            )
        posterior = tuple(value / total for value in values)
        expected_costs = tuple(
            sum(
                posterior[actual] * self.misclassification_costs[actual][predicted]
                for actual in range(len(self.class_labels))
            )
            for predicted in range(len(self.class_labels))
        )
        chosen_index = min(
            range(len(self.class_labels)),
            key=lambda index: (expected_costs[index], index),
        )
        chosen = self.class_labels[chosen_index]
        if force_mode and chosen == 0:
            allowed = tuple(
                index
                for index, label in enumerate(self.class_labels)
                if label not in {0, 3}
            )
            chosen_index = max(allowed, key=lambda index: (posterior[index], -index))
            chosen = self.class_labels[chosen_index]
        if chosen == 2 and not backward_repair_available:
            return 0
        return chosen

    @staticmethod
    def _posterior_from_log_scores(
        scores: Sequence[float],
    ) -> tuple[float, ...]:
        """Normalize log scores without re-evaluating model distributions."""

        scores = tuple(float(value) for value in scores)
        denominator = _logsumexp(scores)
        if not math.isfinite(denominator):
            raise ClassifierPredictionError(
                "Classifier posterior is undefined because every class has zero likelihood"
            )
        posterior = tuple(
            0.0 if score == -math.inf else math.exp(score - denominator)
            for score in scores
        )
        total = sum(posterior)
        if total <= 0 or not math.isfinite(total):
            raise ClassifierPredictionError("Classifier posterior could not be normalized")
        return tuple(value / total for value in posterior)

    def predict(
        self,
        features: Sequence[Any],
        *,
        force_mode: bool = False,
        backward_repair_available: bool = True,
    ) -> int:
        """Predict a legacy class, including StarryNite's optional force rule.

        Ordinary prediction minimizes expected misclassification cost.  If
        ``force_mode`` is active and that result is class 0 ("other"), the
        legacy code ignores classes 0 and 3 and chooses the larger posterior
        among division (1) and false-negative repair (2).  Class 2 is finally
        demoted to class 0 when no backward repair candidate exists.
        """

        posterior = self.posterior(features)
        return self._predict_from_posterior(
            posterior,
            force_mode=force_mode,
            backward_repair_available=backward_repair_available,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "source_model_sha256": self.source_model_sha256,
            "classifier_family": self.classifier_family,
            "feature_layout": self.feature_layout.to_dict(),
            "feature_names": list(self.feature_names),
            "class_labels": list(self.class_labels),
            "class_priors": list(self.class_priors),
            "misclassification_costs": [
                list(row) for row in self.misclassification_costs
            ],
            "distributions": [item.to_dict() for item in self.distributions],
            "missing_value_policy": self.missing_value_policy,
            "tie_policy": self.tie_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NeutralNaiveBayesClassifier:
        data = _mapping(value, "neutral classifier")
        required = {
            "schema",
            "version",
            "source_model_sha256",
            "classifier_family",
            "feature_layout",
            "feature_names",
            "class_labels",
            "class_priors",
            "misclassification_costs",
            "distributions",
            "missing_value_policy",
            "tie_policy",
        }
        _require_keys(data, required, "neutral classifier")
        distributions = tuple(
            _distribution_from_dict(item, index)
            for index, item in enumerate(
                _sequence(data["distributions"], "distributions")
            )
        )
        labels_raw = _sequence(data["class_labels"], "class_labels")
        labels: list[int] = []
        for index, value in enumerate(labels_raw):
            if type(value) is not int:
                raise NeutralClassifierFormatError(
                    f"class_labels[{index}] must be an integer"
                )
            labels.append(value)
        version = data["version"]
        if type(version) is not int:
            raise NeutralClassifierFormatError("version must be an integer")
        return cls(
            schema=str(data["schema"]),
            version=version,
            source_model_sha256=str(data["source_model_sha256"]),
            classifier_family=str(data["classifier_family"]),
            feature_layout=SingleModelFeatureLayout.from_dict(
                _mapping(data["feature_layout"], "feature_layout")
            ),
            feature_names=tuple(
                str(item) for item in _sequence(data["feature_names"], "feature_names")
            ),
            class_labels=tuple(labels),
            class_priors=tuple(_sequence(data["class_priors"], "class_priors")),
            misclassification_costs=tuple(
                tuple(_sequence(row, f"misclassification_costs[{index}]"))
                for index, row in enumerate(
                    _sequence(data["misclassification_costs"], "misclassification_costs")
                )
            ),
            distributions=distributions,
            missing_value_policy=str(data["missing_value_policy"]),
            tie_policy=str(data["tie_policy"]),
        )


@dataclass(frozen=True, slots=True)
class NeutralNaiveBayesSubmodel:
    """One numeric classifier in StarryNite's four-model family.

    Unlike :class:`NeutralNaiveBayesClassifier`, a legacy submodel has no
    topology predictor and may contain only the classes observed for its
    topology branch.  The enclosing family owns the feature masks and exact
    branch routing.
    """

    classifier_family: str
    feature_names: tuple[str, ...]
    class_labels: tuple[int, ...]
    class_priors: tuple[float, ...]
    misclassification_costs: tuple[tuple[float, ...], ...]
    distributions: tuple[FeatureDistribution, ...]
    missing_value_policy: str = "omit"
    tie_policy: str = "first_class"

    def __post_init__(self) -> None:
        if self.classifier_family not in {"legacy_classifier", "new_classifier"}:
            raise NeutralClassifierFormatError(
                "classifier_family must be 'legacy_classifier' or 'new_classifier'"
            )
        labels = tuple(self.class_labels)
        if (
            not labels
            or any(type(value) is not int or value not in _CLASS_LABELS for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise NeutralClassifierFormatError(
                "submodel class_labels must be unique StarryNite classes from 0 to 3"
            )
        priors = _number_tuple(self.class_priors, "class_priors", nonnegative=True)
        if len(priors) != len(labels) or any(value <= 0 for value in priors):
            raise NeutralClassifierFormatError(
                "class_priors must contain one positive probability per class"
            )
        if abs(sum(priors) - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise NeutralClassifierFormatError(
                f"class_priors must sum to 1; received {sum(priors):.17g}"
            )
        costs = tuple(
            _number_tuple(row, f"misclassification_costs[{index}]", nonnegative=True)
            for index, row in enumerate(self.misclassification_costs)
        )
        if len(costs) != len(labels) or any(len(row) != len(labels) for row in costs):
            raise NeutralClassifierFormatError(
                "misclassification_costs must be a square class-by-class matrix"
            )
        names = tuple(str(name) for name in self.feature_names)
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise NeutralClassifierFormatError(
                "feature_names must be nonempty and unique"
            )
        distributions = tuple(self.distributions)
        if not names or len(names) != len(distributions):
            raise NeutralClassifierFormatError(
                "submodel feature_names and distributions must have equal nonzero length"
            )
        if any(distribution.class_count != len(labels) for distribution in distributions):
            raise NeutralClassifierFormatError(
                "Every predictor distribution needs one parameter set per class"
            )
        if self.missing_value_policy != "omit":
            raise NeutralClassifierFormatError(
                "Only the legacy HandleMissing='on'/'omit' policy is supported"
            )
        if self.tie_policy != "first_class":
            raise NeutralClassifierFormatError(
                "Only deterministic first-class tie handling is supported"
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "class_labels", labels)
        object.__setattr__(self, "class_priors", priors)
        object.__setattr__(self, "misclassification_costs", costs)
        object.__setattr__(self, "distributions", distributions)

    @property
    def predictor_count(self) -> int:
        return len(self.distributions)

    def log_scores(self, features: Sequence[Any]) -> tuple[float, ...]:
        """Return unnormalized branch-class log scores, omitting NaNs."""

        values = _prediction_features(features, self.predictor_count)
        scores = [math.log(value) for value in self.class_priors]
        observed = 0
        for value, distribution in zip(values, self.distributions, strict=True):
            if math.isnan(value):
                continue
            observed += 1
            for class_index in range(len(self.class_labels)):
                scores[class_index] += distribution.log_probability(value, class_index)
        if observed == 0 or all(score == -math.inf for score in scores):
            return tuple(math.log(value) for value in self.class_priors)
        if any(math.isnan(score) for score in scores):
            raise ClassifierPredictionError(
                "Classifier produced NaN log scores from otherwise valid inputs"
            )
        return tuple(scores)

    def posterior(self, features: Sequence[Any]) -> tuple[float, ...]:
        return NeutralNaiveBayesClassifier._posterior_from_log_scores(
            self.log_scores(features)
        )

    def predict_from_posterior(self, posterior: Sequence[float]) -> int:
        """Apply the submodel's MATLAB cost matrix with first-class ties."""

        values = tuple(float(value) for value in posterior)
        if len(values) != len(self.class_labels):
            raise ClassifierPredictionError(
                "posterior must contain one probability per submodel class"
            )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ClassifierPredictionError(
                "posterior probabilities must be finite and non-negative"
            )
        total = sum(values)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ClassifierPredictionError(
                f"posterior probabilities must sum to 1; received {total:.17g}"
            )
        normalized = tuple(value / total for value in values)
        expected_costs = tuple(
            sum(
                normalized[actual]
                * self.misclassification_costs[actual][predicted]
                for actual in range(len(self.class_labels))
            )
            for predicted in range(len(self.class_labels))
        )
        chosen = min(
            range(len(self.class_labels)),
            key=lambda index: (expected_costs[index], index),
        )
        return self.class_labels[chosen]

    def predict(self, features: Sequence[Any]) -> int:
        return self.predict_from_posterior(self.posterior(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier_family": self.classifier_family,
            "feature_names": list(self.feature_names),
            "class_labels": list(self.class_labels),
            "class_priors": list(self.class_priors),
            "misclassification_costs": [
                list(row) for row in self.misclassification_costs
            ],
            "distributions": [item.to_dict() for item in self.distributions],
            "missing_value_policy": self.missing_value_policy,
            "tie_policy": self.tie_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NeutralNaiveBayesSubmodel:
        data = _mapping(value, "neutral classifier submodel")
        required = {
            "classifier_family",
            "feature_names",
            "class_labels",
            "class_priors",
            "misclassification_costs",
            "distributions",
            "missing_value_policy",
            "tie_policy",
        }
        _require_keys(data, required, "neutral classifier submodel")
        labels_raw = _sequence(data["class_labels"], "class_labels")
        labels: list[int] = []
        for index, value_item in enumerate(labels_raw):
            if type(value_item) is not int:
                raise NeutralClassifierFormatError(
                    f"class_labels[{index}] must be an integer"
                )
            labels.append(value_item)
        return cls(
            classifier_family=str(data["classifier_family"]),
            feature_names=tuple(
                str(item) for item in _sequence(data["feature_names"], "feature_names")
            ),
            class_labels=tuple(labels),
            class_priors=tuple(_sequence(data["class_priors"], "class_priors")),
            misclassification_costs=tuple(
                tuple(_sequence(row, f"misclassification_costs[{index}]"))
                for index, row in enumerate(
                    _sequence(data["misclassification_costs"], "misclassification_costs")
                )
            ),
            distributions=tuple(
                _distribution_from_dict(item, index)
                for index, item in enumerate(
                    _sequence(data["distributions"], "distributions")
                )
            ),
            missing_value_policy=str(data["missing_value_policy"]),
            tie_policy=str(data["tie_policy"]),
        )


@dataclass(frozen=True, slots=True)
class NeutralAmbigiousClassifierFamily:
    """Numeric parity representation of legacy ``predictBifurcationType``.

    ``ambigious`` intentionally retains StarryNite's historical misspelling;
    it is a serialized field name in legacy parameter/model files.
    """

    source_model_sha256: str
    feature_layout: SingleModelFeatureLayout
    ambigious: NeutralNaiveBayesSubmodel
    fp_div: NeutralNaiveBayesSubmodel
    dirtyfp_fn: NeutralNaiveBayesSubmodel
    divfp: NeutralNaiveBayesSubmodel
    schema: str = NEUTRAL_AMBIGIOUS_CLASSIFIER_SCHEMA
    version: int = NEUTRAL_AMBIGIOUS_CLASSIFIER_VERSION

    def __post_init__(self) -> None:
        if self.schema != NEUTRAL_AMBIGIOUS_CLASSIFIER_SCHEMA:
            raise NeutralClassifierFormatError(
                f"Unsupported neutral ambigious-family schema: {self.schema!r}"
            )
        if (
            type(self.version) is not int
            or self.version != NEUTRAL_AMBIGIOUS_CLASSIFIER_VERSION
        ):
            raise NeutralClassifierFormatError(
                f"Unsupported neutral ambigious-family version: {self.version!r}"
            )
        if not _SHA256.fullmatch(str(self.source_model_sha256)):
            raise NeutralClassifierFormatError(
                "source_model_sha256 must be 64 lowercase hexadecimal characters"
            )
        if not isinstance(self.feature_layout, SingleModelFeatureLayout):
            raise NeutralClassifierFormatError(
                "feature_layout must be a SingleModelFeatureLayout"
            )
        models = self.submodels
        if any(
            not isinstance(model, NeutralNaiveBayesSubmodel)
            for model in models.values()
        ):
            raise NeutralClassifierFormatError(
                "ambigious family fields must be NeutralNaiveBayesSubmodel values"
            )
        for name, expected_labels in _AMBIGIOUS_FORCE_CLASS_VALUES.items():
            actual_labels = models[name].class_labels
            if actual_labels != expected_labels:
                raise NeutralClassifierFormatError(
                    f"{name} class_labels must exactly match MATLAB's positional "
                    f"class order {expected_labels}; received {actual_labels}"
                )
        daughter_count = sum(self.feature_layout.daughter_keep)
        backward_count = sum(self.feature_layout.backward_keep)
        forward_count = sum(self.feature_layout.forward_keep)
        expected = {
            "ambigious": daughter_count + backward_count + forward_count,
            "fp_div": daughter_count,
            "dirtyfp_fn": daughter_count + backward_count,
            "divfp": daughter_count + forward_count,
        }
        for name, model in models.items():
            if model.predictor_count != expected[name]:
                raise NeutralClassifierFormatError(
                    f"{name} expects {model.predictor_count} predictors, but the "
                    f"legacy masks select {expected[name]} for that branch"
                )
        object.__setattr__(self, "source_model_sha256", str(self.source_model_sha256))

    @property
    def classifier_family(self) -> str:
        families = {model.classifier_family for model in self.submodels.values()}
        return next(iter(families)) if len(families) == 1 else "mixed_classifier"

    @property
    def submodels(self) -> dict[str, NeutralNaiveBayesSubmodel]:
        return {
            "ambigious": self.ambigious,
            "fp_div": self.fp_div,
            "dirtyfp_fn": self.dirtyfp_fn,
            "divfp": self.divfp,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "source_model_sha256": self.source_model_sha256,
            "feature_layout": self.feature_layout.to_dict(),
            "submodels": {
                name: model.to_dict() for name, model in self.submodels.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NeutralAmbigiousClassifierFamily:
        data = _mapping(value, "neutral ambigious classifier family")
        required = {
            "schema",
            "version",
            "source_model_sha256",
            "feature_layout",
            "submodels",
        }
        _require_keys(data, required, "neutral ambigious classifier family")
        version = data["version"]
        if type(version) is not int:
            raise NeutralClassifierFormatError("version must be an integer")
        submodels = _mapping(data["submodels"], "submodels")
        _require_keys(submodels, set(_AMBIGIOUS_SUBMODEL_NAMES), "submodels")
        parsed = {
            name: NeutralNaiveBayesSubmodel.from_dict(
                _mapping(submodels[name], f"submodels.{name}")
            )
            for name in _AMBIGIOUS_SUBMODEL_NAMES
        }
        return cls(
            schema=str(data["schema"]),
            version=version,
            source_model_sha256=str(data["source_model_sha256"]),
            feature_layout=SingleModelFeatureLayout.from_dict(
                _mapping(data["feature_layout"], "feature_layout")
            ),
            ambigious=parsed["ambigious"],
            fp_div=parsed["fp_div"],
            dirtyfp_fn=parsed["dirtyfp_fn"],
            divfp=parsed["divfp"],
        )


@dataclass(frozen=True, slots=True)
class SingleModelFeatureInput:
    """Raw blocks and topology cues for one legacy bifurcation decision."""

    daughter_features: tuple[float, ...]
    backward_features: tuple[float, ...]
    forward_features: tuple[float, ...]
    daughter_lengths: tuple[float, float]
    backward_candidate_present: tuple[bool, bool]
    best_forward_lengths: tuple[float, float]
    small_cutoff: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "daughter_features",
            _feature_block(
                self.daughter_features,
                _DAUGHTER_FEATURE_COUNT,
                "daughter_features",
            ),
        )
        object.__setattr__(
            self,
            "backward_features",
            _feature_block(
                self.backward_features,
                _BACKWARD_FEATURE_COUNT,
                "backward_features",
            ),
        )
        object.__setattr__(
            self,
            "forward_features",
            _feature_block(
                self.forward_features,
                _FORWARD_FEATURE_COUNT,
                "forward_features",
            ),
        )
        daughter_lengths = _fixed_prediction_tuple(
            self.daughter_lengths,
            2,
            "daughter_lengths",
        )
        if any(value < 0 for value in daughter_lengths):
            raise ClassifierPredictionError("daughter_lengths cannot be negative")
        object.__setattr__(self, "daughter_lengths", daughter_lengths)
        backward = tuple(self.backward_candidate_present)
        if len(backward) != 2 or any(type(value) is not bool for value in backward):
            raise ClassifierPredictionError(
                "backward_candidate_present must contain exactly two booleans"
            )
        object.__setattr__(self, "backward_candidate_present", backward)
        object.__setattr__(
            self,
            "best_forward_lengths",
            _fixed_prediction_tuple(
                self.best_forward_lengths,
                2,
                "best_forward_lengths",
            ),
        )
        cutoff = _prediction_number(self.small_cutoff, "small_cutoff", allow_nan=False)
        if cutoff <= 0:
            raise ClassifierPredictionError("small_cutoff must be positive")
        object.__setattr__(self, "small_cutoff", cutoff)


@dataclass(frozen=True, slots=True)
class AssembledSingleModelFeatures:
    """Classifier vector plus the topology branch that produced it."""

    values: tuple[float, ...]
    topology_class: int
    topology_case: str


@dataclass(frozen=True, slots=True)
class SingleModelPrediction:
    """Complete diagnostic result for one single-model classification."""

    predicted_class: int
    topology_class: int
    topology_case: str
    features: tuple[float, ...]
    log_scores: tuple[float, ...]
    posterior: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AssembledAmbigiousClassifierFeatures:
    """Feature row and submodel selected by legacy topology routing."""

    values: tuple[float, ...]
    topology_class: int
    topology_case: str
    submodel_name: str


@dataclass(frozen=True, slots=True)
class AmbigiousClassifierPrediction:
    """Auditable result from StarryNite's historical four-model classifier."""

    predicted_class: int
    computed_class: int
    topology_class: int
    topology_case: str
    submodel_name: str
    features: tuple[float, ...]
    log_scores: tuple[float, ...]
    posterior: tuple[float, ...]


def assemble_ambigious_classifier_features(
    inputs: SingleModelFeatureInput,
    layout: SingleModelFeatureLayout,
) -> AssembledAmbigiousClassifierFeatures:
    """Reproduce ``predictBifurcationType`` feature routing exactly.

    The multi-model path does not prepend the topology class and does not pad
    unused blocks with NaNs.  Instead it chooses one of four classifiers and
    concatenates only the blocks used by that topology branch.
    """

    if not isinstance(inputs, SingleModelFeatureInput):
        raise TypeError("inputs must be a SingleModelFeatureInput")
    if not isinstance(layout, SingleModelFeatureLayout):
        raise TypeError("layout must be a SingleModelFeatureLayout")

    first_length, second_length = inputs.daughter_lengths
    first_forward, second_forward = inputs.best_forward_lengths
    has_backward = any(inputs.backward_candidate_present)
    both_long = (
        first_length >= inputs.small_cutoff
        and second_length >= inputs.small_cutoff
    )
    one_small = min(first_length, second_length) < inputs.small_cutoff
    small_lacks_forward = (
        first_length < inputs.small_cutoff and not first_forward > 0
    ) or (
        second_length < inputs.small_cutoff and not second_forward > 0
    )

    if both_long and not has_backward:
        topology_case = "fully_division_looking"
        topology_class = 2
        submodel_name = "fp_div"
        include_backward = False
        include_forward = False
    elif both_long and has_backward:
        topology_case = "false_negative_division_looking"
        topology_class = 3
        submodel_name = "dirtyfp_fn"
        include_backward = True
        include_forward = False
    elif one_small and small_lacks_forward and not has_backward:
        topology_case = "fully_false_positive_looking"
        topology_class = 4
        submodel_name = "fp_div"
        include_backward = False
        include_forward = False
    elif one_small and small_lacks_forward and has_backward:
        topology_case = "dirty_false_positive_looking"
        topology_class = 5
        submodel_name = "dirtyfp_fn"
        include_backward = True
        include_forward = False
    elif one_small and not small_lacks_forward and not has_backward:
        topology_case = "division_false_positive_looking"
        topology_class = 5
        submodel_name = "divfp"
        include_backward = False
        include_forward = True
    elif one_small and not small_lacks_forward and has_backward:
        topology_case = "truly_ambiguous"
        topology_class = 1
        submodel_name = "ambigious"
        include_backward = True
        include_forward = True
    else:  # pragma: no cover - exhaustive finite scalar partition
        raise ClassifierPredictionError(
            "Bifurcation topology did not match a legacy multi-model branch"
        )

    selected = [
        value
        for value, keep in zip(
            inputs.daughter_features,
            layout.daughter_keep,
            strict=True,
        )
        if keep
    ]
    if include_backward:
        selected.extend(
            value
            for value, keep in zip(
                inputs.backward_features,
                layout.backward_keep,
                strict=True,
            )
            if keep
        )
    if include_forward:
        selected.extend(
            value
            for value, keep in zip(
                inputs.forward_features,
                layout.forward_keep,
                strict=True,
            )
            if keep
        )
    return AssembledAmbigiousClassifierFeatures(
        values=tuple(selected),
        topology_class=topology_class,
        topology_case=topology_case,
        submodel_name=submodel_name,
    )


def assemble_single_model_features(
    inputs: SingleModelFeatureInput,
    layout: SingleModelFeatureLayout,
) -> AssembledSingleModelFeatures:
    """Reproduce the single-model topology/masking feature assembly.

    The topology branch decides whether backward or forward measurements are
    meaningful.  Inapplicable blocks are replaced with NaNs before the three
    masks are applied, matching MATLAB's missing-feature classifier path.
    """

    if not isinstance(inputs, SingleModelFeatureInput):
        raise TypeError("inputs must be a SingleModelFeatureInput")
    if not isinstance(layout, SingleModelFeatureLayout):
        raise TypeError("layout must be a SingleModelFeatureLayout")

    first_length, second_length = inputs.daughter_lengths
    first_forward, second_forward = inputs.best_forward_lengths
    has_backward = any(inputs.backward_candidate_present)
    both_long = (
        first_length >= inputs.small_cutoff
        and second_length >= inputs.small_cutoff
    )
    one_small = min(first_length, second_length) < inputs.small_cutoff
    small_lacks_forward = (
        first_length < inputs.small_cutoff and not first_forward > 0
    ) or (
        second_length < inputs.small_cutoff and not second_forward > 0
    )

    if both_long and not has_backward:
        topology_case = "fully_division_looking"
        topology_class = 2
        mask_backward = True
        mask_forward = True
    elif both_long and has_backward:
        topology_case = "false_negative_division_looking"
        topology_class = 3
        mask_backward = False
        mask_forward = True
    elif one_small and small_lacks_forward and not has_backward:
        topology_case = "fully_false_positive_looking"
        topology_class = 4
        mask_backward = True
        mask_forward = True
    elif one_small and small_lacks_forward and has_backward:
        topology_case = "dirty_false_positive_looking"
        topology_class = 5
        mask_backward = False
        mask_forward = True
    elif one_small and not small_lacks_forward and not has_backward:
        topology_case = "division_false_positive_looking"
        topology_class = 5
        mask_backward = True
        mask_forward = True
    elif one_small and not small_lacks_forward and has_backward:
        topology_case = "truly_ambiguous"
        topology_class = 1
        mask_backward = False
        mask_forward = False
    else:  # pragma: no cover - exhaustive finite scalar partition
        raise ClassifierPredictionError(
            "Bifurcation topology did not match a legacy single-model branch"
        )

    backward = (
        (math.nan,) * _BACKWARD_FEATURE_COUNT
        if mask_backward
        else inputs.backward_features
    )
    forward = (
        (math.nan,) * _FORWARD_FEATURE_COUNT
        if mask_forward
        else inputs.forward_features
    )
    selected = [float(topology_class)]
    selected.extend(
        value
        for value, keep in zip(
            inputs.daughter_features,
            layout.daughter_keep,
            strict=True,
        )
        if keep
    )
    selected.extend(
        value
        for value, keep in zip(backward, layout.backward_keep, strict=True)
        if keep
    )
    selected.extend(
        value
        for value, keep in zip(forward, layout.forward_keep, strict=True)
        if keep
    )
    if len(selected) != layout.predictor_count:
        raise RuntimeError("Internal feature-layout cardinality drift")
    return AssembledSingleModelFeatures(
        values=tuple(selected),
        topology_class=topology_class,
        topology_case=topology_case,
    )


def classify_single_model(
    model: NeutralNaiveBayesClassifier,
    inputs: SingleModelFeatureInput,
    *,
    force_mode: bool = False,
    backward_repair_available: bool = True,
) -> SingleModelPrediction:
    """Assemble, score, and classify one StarryNite bifurcation."""

    if not isinstance(model, NeutralNaiveBayesClassifier):
        raise TypeError("model must be a NeutralNaiveBayesClassifier")
    assembled = assemble_single_model_features(inputs, model.feature_layout)
    log_scores = model.log_scores(assembled.values)
    posterior = model._posterior_from_log_scores(log_scores)
    predicted = model._predict_from_posterior(
        posterior,
        force_mode=force_mode,
        backward_repair_available=backward_repair_available,
    )
    return SingleModelPrediction(
        predicted_class=predicted,
        topology_class=assembled.topology_class,
        topology_case=assembled.topology_case,
        features=assembled.values,
        log_scores=log_scores,
        posterior=posterior,
    )


def classify_ambigious_family(
    model: NeutralAmbigiousClassifierFamily,
    inputs: SingleModelFeatureInput,
    *,
    force_mode: bool = False,
    backward_repair_available: bool = True,
) -> AmbigiousClassifierPrediction:
    """Classify through the four-model ``predictBifurcationType`` path.

    The force-mode remapping deliberately uses the original hard-coded class
    arrays rather than submodel labels.  That odd positional behavior is part
    of the MATLAB implementation and is retained for compatibility.
    """

    if not isinstance(model, NeutralAmbigiousClassifierFamily):
        raise TypeError("model must be a NeutralAmbigiousClassifierFamily")
    if type(force_mode) is not bool or type(backward_repair_available) is not bool:
        raise ClassifierPredictionError(
            "force_mode and backward_repair_available must be booleans"
        )
    assembled = assemble_ambigious_classifier_features(inputs, model.feature_layout)
    submodel = model.submodels[assembled.submodel_name]
    log_scores = submodel.log_scores(assembled.values)
    posterior = NeutralNaiveBayesClassifier._posterior_from_log_scores(log_scores)
    predicted = submodel.predict_from_posterior(posterior)

    if force_mode and predicted == 0:
        class_values = _AMBIGIOUS_FORCE_CLASS_VALUES[assembled.submodel_name]
        if len(posterior) > len(class_values):
            raise ClassifierPredictionError(
                f"{assembled.submodel_name} has {len(posterior)} posterior values, "
                f"but legacy force mode defines only {len(class_values)} positions"
            )
        forced_posterior = (0.0,) + posterior[1:]
        forced_index = max(
            range(len(forced_posterior)),
            key=lambda index: (forced_posterior[index], -index),
        )
        predicted = class_values[forced_index]

    computed_class = predicted
    if force_mode and predicted == 0:
        predicted = 1
    if predicted == 2 and not backward_repair_available:
        predicted = 0
    return AmbigiousClassifierPrediction(
        predicted_class=predicted,
        computed_class=computed_class,
        topology_class=assembled.topology_class,
        topology_case=assembled.topology_case,
        submodel_name=assembled.submodel_name,
        features=assembled.values,
        log_scores=log_scores,
        posterior=posterior,
    )


def save_neutral_classifier(
    path: str | Path,
    model: NeutralNaiveBayesClassifier,
) -> None:
    """Atomically write a validated, finite JSON neutral model."""

    if not isinstance(model, NeutralNaiveBayesClassifier):
        raise TypeError("model must be a NeutralNaiveBayesClassifier")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        model.to_dict(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_neutral_classifier(
    path: str | Path,
    *,
    expected_source_model_sha256: str | None = None,
) -> NeutralNaiveBayesClassifier:
    """Load a neutral model and optionally bind it to an original MAT hash."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise NeutralClassifierFormatError(
            f"Invalid neutral classifier JSON in {source}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise NeutralClassifierFormatError(
            f"Could not read neutral classifier {source}: {exc}"
        ) from exc
    try:
        model = NeutralNaiveBayesClassifier.from_dict(
            _mapping(raw, "neutral classifier")
        )
    except NeutralClassifierFormatError as exc:
        raise NeutralClassifierFormatError(
            f"Invalid neutral classifier {source}: {exc}"
        ) from exc
    if expected_source_model_sha256 is not None:
        expected = str(expected_source_model_sha256)
        if not _SHA256.fullmatch(expected):
            raise NeutralClassifierFormatError(
                "expected_source_model_sha256 must be 64 lowercase hexadecimal "
                "characters"
            )
        if model.source_model_sha256 != expected:
            raise NeutralClassifierFormatError(
                f"Neutral classifier {source} was exported from MAT SHA-256 "
                f"{model.source_model_sha256}, not the requested {expected}"
            )
    return model


def save_neutral_ambigious_classifier(
    path: str | Path,
    model: NeutralAmbigiousClassifierFamily,
) -> None:
    """Atomically save a validated legacy four-model classifier family."""

    if not isinstance(model, NeutralAmbigiousClassifierFamily):
        raise TypeError("model must be a NeutralAmbigiousClassifierFamily")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        model.to_dict(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_neutral_ambigious_classifier(
    path: str | Path,
    *,
    expected_source_model_sha256: str | None = None,
) -> NeutralAmbigiousClassifierFamily:
    """Load and optionally source-bind a neutral four-model family."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise NeutralClassifierFormatError(
            f"Invalid neutral ambigious classifier JSON in {source}: "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise NeutralClassifierFormatError(
            f"Could not read neutral ambigious classifier {source}: {exc}"
        ) from exc
    try:
        model = NeutralAmbigiousClassifierFamily.from_dict(
            _mapping(raw, "neutral ambigious classifier family")
        )
    except NeutralClassifierFormatError as exc:
        raise NeutralClassifierFormatError(
            f"Invalid neutral ambigious classifier {source}: {exc}"
        ) from exc
    if expected_source_model_sha256 is not None:
        expected = str(expected_source_model_sha256)
        if not _SHA256.fullmatch(expected):
            raise NeutralClassifierFormatError(
                "expected_source_model_sha256 must be 64 lowercase hexadecimal "
                "characters"
            )
        if model.source_model_sha256 != expected:
            raise NeutralClassifierFormatError(
                f"Neutral ambigious classifier {source} was exported from MAT "
                f"SHA-256 {model.source_model_sha256}, not the requested {expected}"
            )
    return model


def neutral_ambigious_classifier_from_matlab_export(
    exported: Mapping[str, Any],
    *,
    source_model_sha256: str,
) -> NeutralAmbigiousClassifierFamily:
    """Convert a compatible MATLAB numeric four-model export.

    A converter running under a MATLAB release that can reconstruct the old
    ``NaiveBayes`` objects must normalize each object to the inert submodel
    fields accepted by :meth:`NeutralNaiveBayesSubmodel.from_dict`.  No live
    or serialized MATLAB object is accepted here.
    """

    outer = _mapping(exported, "MATLAB ambigious classifier export")
    if "classifier_family_model" in outer:
        raw = _mapping(
            outer["classifier_family_model"],
            "MATLAB ambigious classifier export.classifier_family_model",
        )
    else:
        raw = outer
    required = {
        "family_name",
        "daughter_keep",
        "back_keep",
        "forward_keep",
        "submodels",
    }
    _require_keys(raw, required, "MATLAB ambigious classifier family")
    family_name = _matlab_text(raw["family_name"], "family_name")
    if family_name != "ambigious":
        raise NeutralClassifierFormatError(
            "family_name must preserve StarryNite's legacy spelling 'ambigious'"
        )
    layout = SingleModelFeatureLayout(
        daughter_keep=_matlab_mask(raw["daughter_keep"], 22, "daughter_keep"),
        backward_keep=_matlab_mask(raw["back_keep"], 11, "back_keep"),
        forward_keep=_matlab_mask(raw["forward_keep"], 13, "forward_keep"),
    )
    submodels = _mapping(raw["submodels"], "submodels")
    _require_keys(submodels, set(_AMBIGIOUS_SUBMODEL_NAMES), "submodels")
    parsed = {}
    for name in _AMBIGIOUS_SUBMODEL_NAMES:
        normalized = _matlab_inert_json_mapping(
            _mapping(submodels[name], f"submodels.{name}")
        )
        # scipy.io simplify_cells collapses a one-cell MATLAB cell array to
        # its scalar. Reintroduce the two fields whose JSON schema always
        # requires an array, while leaving every other shape strict.
        if isinstance(normalized.get("feature_names"), str):
            normalized["feature_names"] = [normalized["feature_names"]]
        if isinstance(normalized.get("distributions"), Mapping):
            normalized["distributions"] = [normalized["distributions"]]
        parsed[name] = NeutralNaiveBayesSubmodel.from_dict(normalized)
    return NeutralAmbigiousClassifierFamily(
        source_model_sha256=source_model_sha256,
        feature_layout=layout,
        ambigious=parsed["ambigious"],
        fp_div=parsed["fp_div"],
        dirtyfp_fn=parsed["dirtyfp_fn"],
        divfp=parsed["divfp"],
    )


def _matlab_inert_json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize scipy-loaded inert MATLAB state without interpreting objects."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(member) for key, member in item.items()}
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, tuple):
            return [normalize(member) for member in item]
        if isinstance(item, list):
            return [normalize(member) for member in item]
        return item

    return {str(key): normalize(item) for key, item in value.items()}


def neutral_classifier_from_matlab_export(
    exported: Mapping[str, Any],
    *,
    source_model_sha256: str,
) -> NeutralNaiveBayesClassifier:
    """Convert the MATLAB oracle's inert classifier struct into a model.

    ``exported`` may be either the value of the oracle result's
    ``classifier_model`` field or the complete result mapping containing that
    field.  The bridge intentionally accepts the exact schema emitted by
    ``export_classifier_model`` and cross-checks its duplicated metadata.  It
    never attempts to interpret a live or serialized MATLAB object.
    """

    outer = _mapping(exported, "MATLAB classifier export")
    if "classifier_model" in outer:
        raw = _mapping(
            outer["classifier_model"],
            "MATLAB classifier export.classifier_model",
        )
    else:
        raw = outer
    required = {
        "matlab_class",
        "score_transform",
        "standardization_state",
        "mu",
        "sigma",
        "class_names",
        "prior",
        "cost",
        "predictor_names",
        "categorical_predictors_1based",
        "num_observations",
        "distribution_names",
        "categorical_levels",
        "kernel_names",
        "support_names",
        "width",
        "distributions",
        "daughter_keep",
        "back_keep",
        "forward_keep",
        "selected_feature_count",
    }
    _require_keys(raw, required, "MATLAB classifier_model")

    matlab_class = _matlab_text(raw["matlab_class"], "matlab_class")
    supported_matlab_classes = {
        "ClassificationNaiveBayes": "new_classifier",
        "NaiveBayes": "legacy_classifier",
    }
    if matlab_class not in supported_matlab_classes:
        raise NeutralClassifierFormatError(
            "matlab_class must be 'ClassificationNaiveBayes' or 'NaiveBayes'; "
            f"received {matlab_class!r}"
        )

    score_transform = _matlab_text(
        raw["score_transform"],
        "score_transform",
    ).strip().lower()
    if score_transform != "none":
        raise NeutralClassifierFormatError(
            "score_transform must be MATLAB's identity value 'none'; "
            f"received {score_transform!r}"
        )

    layout = SingleModelFeatureLayout(
        daughter_keep=_matlab_mask(raw["daughter_keep"], 22, "daughter_keep"),
        backward_keep=_matlab_mask(raw["back_keep"], 11, "back_keep"),
        forward_keep=_matlab_mask(raw["forward_keep"], 13, "forward_keep"),
    )
    selected_count = _matlab_integer_scalar(
        raw["selected_feature_count"],
        "selected_feature_count",
    )
    if selected_count != layout.predictor_count:
        raise NeutralClassifierFormatError(
            "selected_feature_count disagrees with daughter/back/forward masks: "
            f"export reports {selected_count}, masks select {layout.predictor_count}"
        )

    class_values = _matlab_vector(raw["class_names"], "class_names")
    class_labels = tuple(
        _matlab_integer(value, f"class_names[{index}]")
        for index, value in enumerate(class_values)
    )
    class_count = len(class_labels)
    if class_count != len(_CLASS_LABELS):
        raise NeutralClassifierFormatError(
            f"class_names must contain four StarryNite classes; received {class_count}"
        )

    predictor_name_values = _matlab_vector(
        raw["predictor_names"],
        "predictor_names",
    )
    feature_names = tuple(
        _matlab_text(value, f"predictor_names[{index}]")
        for index, value in enumerate(predictor_name_values)
    )
    distribution_name_values = _matlab_vector(
        raw["distribution_names"],
        "distribution_names",
    )
    distribution_names = tuple(
        _matlab_text(value, f"distribution_names[{index}]").strip().lower()
        for index, value in enumerate(distribution_name_values)
    )
    predictor_count = len(distribution_names)
    if predictor_count != selected_count or len(feature_names) != predictor_count:
        raise NeutralClassifierFormatError(
            "MATLAB predictor metadata disagrees: masks select "
            f"{selected_count}, distribution_names contains {predictor_count}, "
            f"and predictor_names contains {len(feature_names)}"
        )

    mu = _matlab_optional_numeric_vector(raw["mu"], "mu")
    sigma = _matlab_optional_numeric_vector(raw["sigma"], "sigma")
    standardization_state = _matlab_text(
        raw["standardization_state"],
        "standardization_state",
    ).strip().lower()
    if not mu and not sigma:
        expected_standardization = "none"
    elif (
        len(mu) == predictor_count
        and len(sigma) == predictor_count
        and all(value == 0.0 for value in mu)
        and all(value == 1.0 for value in sigma)
    ):
        expected_standardization = "identity"
    else:
        raise NeutralClassifierFormatError(
            "mu/sigma contain unsupported nonidentity predictor standardization"
        )
    if standardization_state != expected_standardization:
        raise NeutralClassifierFormatError(
            "standardization_state disagrees with mu/sigma: received "
            f"{standardization_state!r}, expected {expected_standardization!r}"
        )
    unsupported = sorted(set(distribution_names) - {"mvmn", "normal", "kernel"})
    if unsupported:
        raise NeutralClassifierFormatError(
            "distribution_names contains unsupported value(s): "
            + ", ".join(repr(value) for value in unsupported)
        )

    categorical_indices = tuple(
        _matlab_integer(value, f"categorical_predictors_1based[{index}]")
        for index, value in enumerate(
            _matlab_vector(
                raw["categorical_predictors_1based"],
                "categorical_predictors_1based",
                allow_scalar=True,
            )
        )
    )
    expected_categorical = tuple(
        index + 1
        for index, distribution_name in enumerate(distribution_names)
        if distribution_name == "mvmn"
    )
    if categorical_indices != expected_categorical:
        raise NeutralClassifierFormatError(
            "categorical_predictors_1based disagrees with mvmn predictors: "
            f"received {categorical_indices}, expected {expected_categorical}"
        )

    observations = _matlab_integer_scalar(raw["num_observations"], "num_observations")
    if observations < 0:
        raise NeutralClassifierFormatError("num_observations cannot be negative")

    categorical_level_cells = _matlab_vector(
        raw["categorical_levels"],
        "categorical_levels",
    )
    kernel_name_cells = _matlab_vector(raw["kernel_names"], "kernel_names")
    support_name_cells = _matlab_vector(raw["support_names"], "support_names")
    for label, values in (
        ("categorical_levels", categorical_level_cells),
        ("kernel_names", kernel_name_cells),
        ("support_names", support_name_cells),
    ):
        if len(values) != predictor_count:
            raise NeutralClassifierFormatError(
                f"{label} must contain one entry per predictor; received "
                f"{len(values)}, expected {predictor_count}"
            )
    categorical_levels = tuple(
        tuple(
            _finite_number(value, f"categorical_levels[{predictor}][{level}]")
            for level, value in enumerate(
                _matlab_vector(
                    cell,
                    f"categorical_levels[{predictor}]",
                    allow_scalar=True,
                )
            )
        )
        for predictor, cell in enumerate(categorical_level_cells)
    )
    kernel_names = tuple(
        _matlab_text(value, f"kernel_names[{index}]").strip().lower()
        for index, value in enumerate(kernel_name_cells)
    )
    support_names = tuple(
        _matlab_text(value, f"support_names[{index}]").strip().lower()
        for index, value in enumerate(support_name_cells)
    )

    widths = _matlab_matrix(
        raw["width"],
        class_count,
        predictor_count,
        "width",
    )
    entries = _matlab_matrix(
        raw["distributions"],
        class_count,
        predictor_count,
        "distributions",
    )
    parsed_entries = tuple(
        tuple(
            _matlab_distribution_entry(
                entries[class_index][predictor_index],
                class_index,
                predictor_index,
                distribution_names[predictor_index],
                categorical_levels[predictor_index],
                kernel_names[predictor_index],
                support_names[predictor_index],
            )
            for predictor_index in range(predictor_count)
        )
        for class_index in range(class_count)
    )

    distributions: list[FeatureDistribution] = []
    for predictor_index, distribution_name in enumerate(distribution_names):
        column = tuple(row[predictor_index] for row in parsed_entries)
        if distribution_name == "mvmn":
            distributions.append(
                CategoricalFeatureDistribution(
                    categories=categorical_levels[predictor_index],
                    probabilities=tuple(item["numeric_parameters"] for item in column),
                )
            )
        elif distribution_name == "normal":
            distributions.append(
                GaussianFeatureDistribution(
                    means=tuple(item["numeric_parameters"][0] for item in column),
                    standard_deviations=tuple(
                        item["numeric_parameters"][1] for item in column
                    ),
                )
            )
        else:
            bandwidths = tuple(item["bandwidth"] for item in column)
            for class_index, bandwidth in enumerate(bandwidths):
                width = _matlab_float(
                    widths[class_index][predictor_index],
                    f"width[{class_index}][{predictor_index}]",
                    require_finite=True,
                )
                if not math.isclose(width, bandwidth, rel_tol=1e-12, abs_tol=0.0):
                    raise NeutralClassifierFormatError(
                        f"width[{class_index}][{predictor_index}] ({width:g}) "
                        f"disagrees with kernel bandwidth ({bandwidth:g})"
                    )
            distributions.append(
                GaussianKernelFeatureDistribution(
                    samples=tuple(item["input_data"] for item in column),
                    frequencies=tuple(item["input_frequency"] for item in column),
                    bandwidths=bandwidths,
                    kernel=kernel_names[predictor_index],
                    support=support_names[predictor_index],
                )
            )

    prior = tuple(
        _finite_number(value, f"prior[{index}]")
        for index, value in enumerate(_matlab_vector(raw["prior"], "prior"))
    )
    cost_values = _matlab_matrix(raw["cost"], class_count, class_count, "cost")
    costs = tuple(
        tuple(
            _finite_number(value, f"cost[{row_index}][{column_index}]")
            for column_index, value in enumerate(row)
        )
        for row_index, row in enumerate(cost_values)
    )
    return NeutralNaiveBayesClassifier(
        source_model_sha256=source_model_sha256,
        classifier_family=supported_matlab_classes[matlab_class],
        feature_layout=layout,
        feature_names=feature_names,
        class_labels=class_labels,
        class_priors=prior,
        misclassification_costs=costs,
        distributions=tuple(distributions),
    )


_MATLAB_DISTRIBUTION_FIELDS = {
    "class_index_1based",
    "predictor_index_1based",
    "distribution_name",
    "numeric_parameters",
    "categorical_levels",
    "kernel_name",
    "support",
    "bandwidth",
    "input_data",
    "input_frequency",
    "input_censored",
    "truncation",
    "is_truncated",
}


def _matlab_distribution_entry(
    value: Any,
    class_index: int,
    predictor_index: int,
    distribution_name: str,
    categorical_levels: tuple[float, ...],
    kernel_name: str,
    support_name: str,
) -> dict[str, Any]:
    label = f"distributions[{class_index}][{predictor_index}]"
    data = _mapping(value, label)
    _require_keys(data, _MATLAB_DISTRIBUTION_FIELDS, label)
    exported_class = _matlab_integer_scalar(
        data["class_index_1based"],
        f"{label}.class_index_1based",
    )
    exported_predictor = _matlab_integer_scalar(
        data["predictor_index_1based"],
        f"{label}.predictor_index_1based",
    )
    if exported_class != class_index + 1 or exported_predictor != predictor_index + 1:
        raise NeutralClassifierFormatError(
            f"{label} index metadata is ({exported_class}, {exported_predictor}); "
            f"expected ({class_index + 1}, {predictor_index + 1})"
        )
    item_distribution = _matlab_text(
        data["distribution_name"],
        f"{label}.distribution_name",
    ).strip().lower()
    if item_distribution != distribution_name:
        raise NeutralClassifierFormatError(
            f"{label}.distribution_name is {item_distribution!r}; top-level "
            f"distribution_names specifies {distribution_name!r}"
        )
    item_levels = tuple(
        _finite_number(item, f"{label}.categorical_levels[{index}]")
        for index, item in enumerate(
            _matlab_vector(
                data["categorical_levels"],
                f"{label}.categorical_levels",
                allow_scalar=True,
            )
        )
    )
    if item_levels != categorical_levels:
        raise NeutralClassifierFormatError(
            f"{label}.categorical_levels disagrees with top-level metadata"
        )
    item_kernel = _matlab_text(
        data["kernel_name"],
        f"{label}.kernel_name",
    ).strip().lower()
    item_support = _matlab_text(
        data["support"],
        f"{label}.support",
    ).strip().lower()
    if item_kernel != kernel_name or item_support != support_name:
        raise NeutralClassifierFormatError(
            f"{label} kernel/support metadata disagrees with top-level metadata"
        )

    numeric_parameters = tuple(
        _finite_number(item, f"{label}.numeric_parameters[{index}]")
        for index, item in enumerate(
            _matlab_vector(
                data["numeric_parameters"],
                f"{label}.numeric_parameters",
                allow_scalar=True,
            )
        )
    )
    bandwidth_values = _matlab_vector(
        data["bandwidth"],
        f"{label}.bandwidth",
        allow_scalar=True,
    )
    input_data = tuple(
        _finite_number(item, f"{label}.input_data[{index}]")
        for index, item in enumerate(
            _matlab_vector(
                data["input_data"],
                f"{label}.input_data",
                allow_scalar=True,
            )
        )
    )
    input_frequency = tuple(
        _finite_number(item, f"{label}.input_frequency[{index}]")
        for index, item in enumerate(
            _matlab_vector(
                data["input_frequency"],
                f"{label}.input_frequency",
                allow_scalar=True,
            )
        )
    )
    input_censored = tuple(
        _matlab_float(item, f"{label}.input_censored[{index}]", require_finite=False)
        for index, item in enumerate(
            _matlab_vector(
                data["input_censored"],
                f"{label}.input_censored",
                allow_scalar=True,
            )
        )
    )
    truncation = _matlab_vector(
        data["truncation"],
        f"{label}.truncation",
        allow_scalar=True,
    )
    is_truncated = _matlab_boolean_scalar(
        data["is_truncated"],
        f"{label}.is_truncated",
    )

    if distribution_name == "mvmn":
        if len(numeric_parameters) != len(categorical_levels):
            raise NeutralClassifierFormatError(
                f"{label}.numeric_parameters must match its categorical levels"
            )
        _require_empty_matlab_fields(
            label,
            bandwidth_values=bandwidth_values,
            input_data=input_data,
            input_frequency=input_frequency,
            input_censored=input_censored,
            truncation=truncation,
            is_truncated=is_truncated,
        )
    elif distribution_name == "normal":
        if len(numeric_parameters) != 2:
            raise NeutralClassifierFormatError(
                f"{label}.numeric_parameters must contain [mean, standard deviation]"
            )
        _require_empty_matlab_fields(
            label,
            bandwidth_values=bandwidth_values,
            input_data=input_data,
            input_frequency=input_frequency,
            input_censored=input_censored,
            truncation=truncation,
            is_truncated=is_truncated,
        )
    else:
        if numeric_parameters:
            raise NeutralClassifierFormatError(
                f"{label}.numeric_parameters must be empty for a kernel predictor"
            )
        if len(bandwidth_values) != 1:
            raise NeutralClassifierFormatError(
                f"{label}.bandwidth must be a positive scalar"
            )
        if not input_data or len(input_frequency) != len(input_data):
            raise NeutralClassifierFormatError(
                f"{label} kernel data and frequency vectors must have equal "
                "nonzero length"
            )
        if input_censored and (
            len(input_censored) != len(input_data)
            or any(value != 0.0 for value in input_censored)
        ):
            raise NeutralClassifierFormatError(
                f"{label} contains censored kernel observations, which the neutral "
                "Gaussian KDE does not support"
            )
        if is_truncated:
            raise NeutralClassifierFormatError(
                f"{label} is truncated; only unbounded Gaussian KDEs are supported"
            )

    bandwidth = (
        _finite_number(bandwidth_values[0], f"{label}.bandwidth")
        if bandwidth_values
        else None
    )
    return {
        "numeric_parameters": numeric_parameters,
        "bandwidth": bandwidth,
        "input_data": input_data,
        "input_frequency": input_frequency,
    }


def _require_empty_matlab_fields(
    label: str,
    *,
    bandwidth_values: Sequence[Any],
    input_data: Sequence[Any],
    input_frequency: Sequence[Any],
    input_censored: Sequence[Any],
    truncation: Sequence[Any],
    is_truncated: bool,
) -> None:
    if (
        bandwidth_values
        or input_data
        or input_frequency
        or input_censored
        or truncation
        or is_truncated
    ):
        raise NeutralClassifierFormatError(
            f"{label} contains kernel-only state for a non-kernel predictor"
        )


def _matlab_shape(value: Any) -> tuple[int, ...] | None:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        return None
    try:
        return tuple(int(item) for item in raw_shape)
    except (TypeError, ValueError) as exc:
        raise NeutralClassifierFormatError(
            "MATLAB export contains an object with an invalid array shape"
        ) from exc


def _matlab_vector(
    value: Any,
    label: str,
    *,
    allow_scalar: bool = False,
) -> tuple[Any, ...]:
    shape = _matlab_shape(value)
    if shape is not None:
        if len(shape) == 0:
            if not allow_scalar:
                raise NeutralClassifierFormatError(f"{label} must be a vector")
            item = value.item() if hasattr(value, "item") else value
            return (item,)
        if len(shape) == 1:
            return tuple(value[index] for index in range(shape[0]))
        if len(shape) == 2 and 1 in shape:
            if shape[0] == 1:
                return tuple(value[0, index] for index in range(shape[1]))
            return tuple(value[index, 0] for index in range(shape[0]))
        raise NeutralClassifierFormatError(
            f"{label} must be a vector; received array shape {shape}"
        )
    if isinstance(value, (str, bytes, bytearray)):
        if allow_scalar:
            return (value,)
        raise NeutralClassifierFormatError(f"{label} must be a vector")
    if isinstance(value, Sequence):
        return tuple(value)
    if allow_scalar:
        return (value,)
    raise NeutralClassifierFormatError(f"{label} must be a vector")


def _matlab_optional_numeric_vector(value: Any, label: str) -> tuple[float, ...]:
    shape = _matlab_shape(value)
    if shape is not None and any(size == 0 for size in shape):
        return ()
    values = _matlab_vector(value, label, allow_scalar=True)
    return tuple(
        _matlab_float(item, f"{label}[{index}]", require_finite=True)
        for index, item in enumerate(values)
    )


def _matlab_matrix(
    value: Any,
    rows: int,
    columns: int,
    label: str,
) -> tuple[tuple[Any, ...], ...]:
    shape = _matlab_shape(value)
    if shape is not None:
        if shape != (rows, columns):
            raise NeutralClassifierFormatError(
                f"{label} must have shape ({rows}, {columns}); received {shape}"
            )
        return tuple(
            tuple(value[row, column] for column in range(columns))
            for row in range(rows)
        )
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise NeutralClassifierFormatError(
            f"{label} must be a {rows}-by-{columns} matrix"
        )
    matrix = tuple(value)
    if len(matrix) != rows:
        raise NeutralClassifierFormatError(
            f"{label} must have {rows} rows; received {len(matrix)}"
        )
    result: list[tuple[Any, ...]] = []
    for row_index, row in enumerate(matrix):
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise NeutralClassifierFormatError(f"{label}[{row_index}] must be a row")
        row_values = tuple(row)
        if len(row_values) != columns:
            raise NeutralClassifierFormatError(
                f"{label}[{row_index}] must have {columns} columns; received "
                f"{len(row_values)}"
            )
        result.append(row_values)
    return tuple(result)


def _matlab_text(value: Any, label: str) -> str:
    shape = _matlab_shape(value)
    if shape is not None and any(size == 0 for size in shape):
        return ""
    if shape == () and hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, str):
        raise NeutralClassifierFormatError(f"{label} must be text")
    return value


def _matlab_float(value: Any, label: str, *, require_finite: bool) -> float:
    if isinstance(value, (str, bytes, bytearray, bool)) or type(value).__name__ == "bool_":
        raise NeutralClassifierFormatError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NeutralClassifierFormatError(f"{label} must be numeric") from exc
    if require_finite and not math.isfinite(result):
        raise NeutralClassifierFormatError(f"{label} must be finite")
    return result


def _matlab_integer(value: Any, label: str) -> int:
    number = _matlab_float(value, label, require_finite=True)
    if number != math.trunc(number):
        raise NeutralClassifierFormatError(f"{label} must be an integer")
    return int(number)


def _matlab_integer_scalar(value: Any, label: str) -> int:
    values = _matlab_vector(value, label, allow_scalar=True)
    if len(values) != 1:
        raise NeutralClassifierFormatError(f"{label} must be a scalar integer")
    return _matlab_integer(values[0], label)


def _matlab_boolean_scalar(value: Any, label: str) -> bool:
    values = _matlab_vector(value, label, allow_scalar=True)
    if len(values) != 1:
        raise NeutralClassifierFormatError(f"{label} must be a scalar boolean")
    item = values[0]
    if isinstance(item, bool) or type(item).__name__ == "bool_":
        return bool(item)
    number = _matlab_float(item, label, require_finite=True)
    if number not in {0.0, 1.0}:
        raise NeutralClassifierFormatError(f"{label} must be boolean (0 or 1)")
    return bool(number)


def _matlab_mask(value: Any, expected: int, label: str) -> tuple[bool, ...]:
    values = _matlab_vector(value, label)
    if len(values) != expected:
        raise NeutralClassifierFormatError(
            f"{label} must contain {expected} values; received {len(values)}"
        )
    return tuple(
        _matlab_boolean_scalar(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    )


def _distribution_from_dict(value: Any, index: int) -> FeatureDistribution:
    label = f"distributions[{index}]"
    data = _mapping(value, label)
    kind = data.get("kind")
    if kind == "gaussian":
        _require_keys(data, {"kind", "means", "standard_deviations"}, label)
        return GaussianFeatureDistribution(
            means=tuple(_sequence(data["means"], f"{label}.means")),
            standard_deviations=tuple(
                _sequence(
                    data["standard_deviations"],
                    f"{label}.standard_deviations",
                )
            ),
        )
    if kind == "categorical":
        _require_keys(data, {"kind", "categories", "probabilities"}, label)
        return CategoricalFeatureDistribution(
            categories=tuple(_sequence(data["categories"], f"{label}.categories")),
            probabilities=tuple(
                tuple(_sequence(row, f"{label}.probabilities[{row_index}]"))
                for row_index, row in enumerate(
                    _sequence(data["probabilities"], f"{label}.probabilities")
                )
            ),
        )
    if kind == "kernel":
        _require_keys(
            data,
            {
                "kind",
                "kernel",
                "support",
                "samples",
                "frequencies",
                "bandwidths",
            },
            label,
        )
        return GaussianKernelFeatureDistribution(
            kernel=str(data["kernel"]),
            support=str(data["support"]),
            samples=tuple(
                tuple(_sequence(row, f"{label}.samples[{row_index}]"))
                for row_index, row in enumerate(
                    _sequence(data["samples"], f"{label}.samples")
                )
            ),
            frequencies=tuple(
                tuple(_sequence(row, f"{label}.frequencies[{row_index}]"))
                for row_index, row in enumerate(
                    _sequence(data["frequencies"], f"{label}.frequencies")
                )
            ),
            bandwidths=tuple(
                _sequence(data["bandwidths"], f"{label}.bandwidths")
            ),
        )
    raise NeutralClassifierFormatError(
        f"{label}.kind {kind!r} is unsupported; expected 'categorical', "
        "'gaussian', or Gaussian 'kernel'"
    )


def _prediction_features(values: Sequence[Any], expected: int) -> tuple[float, ...]:
    result_values: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ClassifierPredictionError(f"features[{index}] must be numeric")
        try:
            result_values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ClassifierPredictionError(
                f"features[{index}] must be numeric"
            ) from exc
    result = tuple(result_values)
    if len(result) != expected:
        raise ClassifierPredictionError(
            f"Classifier expects {expected} features; received {len(result)}"
        )
    return result


def _feature_block(values: Sequence[Any], expected: int, label: str) -> tuple[float, ...]:
    result_values: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ClassifierPredictionError(f"{label}[{index}] must be numeric")
        try:
            result_values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ClassifierPredictionError(
                f"{label}[{index}] must be numeric"
            ) from exc
    result = tuple(result_values)
    if len(result) != expected:
        raise ClassifierPredictionError(
            f"{label} must contain exactly {expected} values; received {len(result)}"
        )
    return result


def _fixed_prediction_tuple(
    values: Sequence[Any],
    expected: int,
    label: str,
) -> tuple[float, ...]:
    result = tuple(
        _prediction_number(value, f"{label}[{index}]", allow_nan=False)
        for index, value in enumerate(values)
    )
    if len(result) != expected:
        raise ClassifierPredictionError(
            f"{label} must contain exactly {expected} values"
        )
    return result


def _logsumexp(values: Sequence[float]) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    if maximum == math.inf:
        return math.inf
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NeutralClassifierFormatError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise NeutralClassifierFormatError(f"{label} keys must be strings")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise NeutralClassifierFormatError(f"{label} must be a JSON array")
    return value


def _require_keys(data: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(data)
    unknown = set(data) - required
    if missing:
        raise NeutralClassifierFormatError(
            f"{label} is missing required field(s): {', '.join(sorted(missing))}"
        )
    if unknown:
        raise NeutralClassifierFormatError(
            f"{label} contains unknown field(s): {', '.join(sorted(unknown))}"
        )


__all__ = [
    "AmbigiousClassifierPrediction",
    "AssembledAmbigiousClassifierFeatures",
    "AssembledSingleModelFeatures",
    "CategoricalFeatureDistribution",
    "ClassifierPredictionError",
    "GaussianFeatureDistribution",
    "GaussianKernelFeatureDistribution",
    "LEGACY_SINGLE_MODEL_FEATURE_COUNT",
    "NEUTRAL_AMBIGIOUS_CLASSIFIER_SCHEMA",
    "NEUTRAL_AMBIGIOUS_CLASSIFIER_VERSION",
    "NEUTRAL_CLASSIFIER_SCHEMA",
    "NEUTRAL_CLASSIFIER_VERSION",
    "NeutralAmbigiousClassifierFamily",
    "NeutralClassifierFormatError",
    "NeutralNaiveBayesClassifier",
    "NeutralNaiveBayesSubmodel",
    "SingleModelFeatureInput",
    "SingleModelFeatureLayout",
    "SingleModelPrediction",
    "StarryNiteClassifierError",
    "assemble_ambigious_classifier_features",
    "assemble_single_model_features",
    "classify_ambigious_family",
    "classify_single_model",
    "load_neutral_ambigious_classifier",
    "load_neutral_classifier",
    "neutral_ambigious_classifier_from_matlab_export",
    "neutral_classifier_from_matlab_export",
    "save_neutral_ambigious_classifier",
    "save_neutral_classifier",
]
