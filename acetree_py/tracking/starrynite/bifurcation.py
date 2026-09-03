"""Classifier-to-lineage orchestration for one StarryNite bifurcation.

This module is intentionally a narrow boundary.  It does not calculate or
approximate any of the 22 daughter, 11 backward, or 13 forward legacy feature
measurements.  ``SingleModelLineageRequest.features`` must come from an exact,
independently validated legacy feature extractor before this API is called.
The same evidence can be routed through either the later single-model
classifier or the historical four-model ``ambigious`` family.

Keeping feature extraction out of the orchestration layer is important for
clean-room parity: a plausible geometric substitute must never be presented to
the learned MATLAB model as though it were the original feature vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from .classifier import (
    AmbigiousClassifierPrediction,
    NeutralAmbigiousClassifierFamily,
    NeutralNaiveBayesClassifier,
    SingleModelFeatureInput,
    SingleModelPrediction,
    classify_ambigious_family,
    classify_single_model,
)
from .lineage import (
    BifurcationDecision,
    FalseNegativeRewirePlan,
    LineageGraphState,
    LineageReattachmentCandidate,
    LineageResolutionDiagnostics,
    LineageResolutionResult,
    resolve_bifurcation,
)


class BifurcationOrchestrationError(ValueError):
    """Raised when classifier and lineage inputs cannot form one safe action."""


def _optional_score(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a number or None")
    # Legacy ``log(1 ./ mvnpdf(...))`` legitimately produces Inf, -Inf, or
    # NaN in the Gaussian tails.  MATLAB then uses an ordinary strict ``>``
    # comparison (NaN therefore falls through to daughter 2), so retain the
    # scalar rather than rejecting an otherwise reproducible event.
    return float(value)


@dataclass(frozen=True, slots=True)
class SingleModelLineageRequest:
    """Exact classifier evidence and graph context for one tentative split.

    ``features`` is an explicit trust boundary: callers must supply the exact
    legacy single-model blocks and topology cues.  This object validates their
    type but deliberately does not synthesize missing measurements.

    A class-2 false-negative prediction is enabled only when
    ``false_negative_plan`` is present.  Candidate selection therefore remains
    the responsibility of the exact legacy candidate extractor.  Class 0 uses
    the two nondivision scores and optional ordered reattachment candidates.
    """

    state: LineageGraphState
    features: SingleModelFeatureInput
    parent_id: str
    daughter1_id: str
    daughter2_id: str
    daughter1_nondivision_score: float | None = None
    daughter2_nondivision_score: float | None = None
    reattachment_candidates: tuple[LineageReattachmentCandidate, ...] = ()
    false_negative_plan: FalseNegativeRewirePlan | None = None
    force_mode: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.state, LineageGraphState):
            raise TypeError("state must be a LineageGraphState")
        if not isinstance(self.features, SingleModelFeatureInput):
            raise TypeError(
                "features must be an exact SingleModelFeatureInput; this API "
                "does not approximate legacy feature blocks"
            )

        identifiers = {
            "parent_id": self.parent_id,
            "daughter1_id": self.daughter1_id,
            "daughter2_id": self.daughter2_id,
        }
        for label, value in identifiers.items():
            if type(value) is not str or not value:
                raise TypeError(f"{label} must be a non-empty string")
        if len(set(identifiers.values())) != 3:
            raise BifurcationOrchestrationError(
                "Bifurcation parent and daughter IDs must be distinct"
            )

        inactive = set(identifiers.values()) - self.state.active_ids
        if inactive:
            raise BifurcationOrchestrationError(
                "Bifurcation nodes must be active in the lineage graph: "
                + ", ".join(sorted(inactive))
            )
        actual_daughters = set(self.state.successors(self.parent_id))
        requested_daughters = {self.daughter1_id, self.daughter2_id}
        if actual_daughters != requested_daughters:
            raise BifurcationOrchestrationError(
                f"Parent {self.parent_id!r} must have exactly the two requested "
                "daughters"
            )

        first_score = _optional_score(
            self.daughter1_nondivision_score,
            "daughter1_nondivision_score",
        )
        second_score = _optional_score(
            self.daughter2_nondivision_score,
            "daughter2_nondivision_score",
        )
        if (first_score is None) != (second_score is None):
            raise BifurcationOrchestrationError(
                "Nondivision scores must either both be supplied or both be omitted"
            )
        object.__setattr__(self, "daughter1_nondivision_score", first_score)
        object.__setattr__(self, "daughter2_nondivision_score", second_score)

        try:
            candidates = tuple(self.reattachment_candidates)
        except TypeError as exc:
            raise TypeError(
                "reattachment_candidates must be an iterable of "
                "LineageReattachmentCandidate values"
            ) from exc
        if any(
            not isinstance(candidate, LineageReattachmentCandidate)
            for candidate in candidates
        ):
            raise TypeError(
                "reattachment_candidates must contain only "
                "LineageReattachmentCandidate values"
            )
        object.__setattr__(self, "reattachment_candidates", candidates)

        if self.false_negative_plan is not None and not isinstance(
            self.false_negative_plan,
            FalseNegativeRewirePlan,
        ):
            raise TypeError(
                "false_negative_plan must be a FalseNegativeRewirePlan or None"
            )
        if type(self.force_mode) is not bool:
            raise TypeError("force_mode must be a boolean")

    @classmethod
    def from_legacy_extraction(
        cls,
        state: LineageGraphState,
        extraction: object,
        *,
        reattachment_candidates: tuple[LineageReattachmentCandidate, ...] = (),
        force_mode: bool = False,
    ) -> SingleModelLineageRequest:
        """Bridge one exact extractor result into classifier/lineage runtime.

        For a fully legacy class-0 retry loop, call
        :func:`repair_legacy_class_zero_bifurcation` after classification; the
        optional candidates here support the simpler generic lineage boundary.
        """

        # Local import avoids a module cycle: legacy_features owns the raw
        # 22/11/13 contract and imports classifier primitives from this package.
        from .legacy_features import LegacyBifurcationExtraction

        if not isinstance(extraction, LegacyBifurcationExtraction):
            raise TypeError("extraction must be a LegacyBifurcationExtraction")
        return cls(
            state=state,
            features=extraction.feature_input,
            parent_id=extraction.parent_id,
            daughter1_id=extraction.daughter_ids[0],
            daughter2_id=extraction.daughter_ids[1],
            daughter1_nondivision_score=extraction.nondivision_scores[0],
            daughter2_nondivision_score=extraction.nondivision_scores[1],
            reattachment_candidates=reattachment_candidates,
            false_negative_plan=extraction.false_negative_plan,
            force_mode=force_mode,
        )


@dataclass(frozen=True, slots=True)
class SingleModelLineageResult:
    """Auditable classifier prediction, decision, and immutable graph result."""

    prediction: SingleModelPrediction | AmbigiousClassifierPrediction
    decision: BifurcationDecision
    resolution: LineageResolutionResult

    def __post_init__(self) -> None:
        if not isinstance(
            self.prediction,
            (SingleModelPrediction, AmbigiousClassifierPrediction),
        ):
            raise TypeError(
                "prediction must be a single-model or ambigious-family prediction"
            )
        if not isinstance(self.decision, BifurcationDecision):
            raise TypeError("decision must be a BifurcationDecision")
        if not isinstance(self.resolution, LineageResolutionResult):
            raise TypeError("resolution must be a LineageResolutionResult")
        classes = {
            self.prediction.predicted_class,
            self.decision.classification,
            self.resolution.diagnostics.classification,
        }
        if len(classes) != 1:
            raise BifurcationOrchestrationError(
                "Prediction, decision, and resolution classes disagree"
            )

    @property
    def prediction_diagnostics(
        self,
    ) -> SingleModelPrediction | AmbigiousClassifierPrediction:
        """Classifier scores, posterior, topology, and selected class."""

        return self.prediction

    @property
    def resolution_diagnostics(self) -> LineageResolutionDiagnostics:
        """Graph action diagnostics produced by the lineage resolver."""

        return self.resolution.diagnostics


def classify_and_resolve_bifurcation(
    model: NeutralNaiveBayesClassifier | NeutralAmbigiousClassifierFamily,
    request: SingleModelLineageRequest,
) -> SingleModelLineageResult:
    """Classify and atomically resolve one tentative StarryNite split.

    No feature extraction or repair-candidate discovery occurs here.  A
    false-negative repair is advertised to the classifier only when the exact
    upstream extractor supplied an explicit ``FalseNegativeRewirePlan``.  The
    classifier may be the 2019 single model or the legacy four-model family.
    """

    if not isinstance(
        model,
        (NeutralNaiveBayesClassifier, NeutralAmbigiousClassifierFamily),
    ):
        raise TypeError(
            "model must be a neutral single-model or ambigious classifier family"
        )
    if not isinstance(request, SingleModelLineageRequest):
        raise TypeError("request must be a SingleModelLineageRequest")

    classifier = (
        classify_ambigious_family
        if isinstance(model, NeutralAmbigiousClassifierFamily)
        else classify_single_model
    )
    prediction = classifier(
        model,
        request.features,
        force_mode=request.force_mode,
        backward_repair_available=request.false_negative_plan is not None,
    )
    if prediction.predicted_class == 0 and (
        request.daughter1_nondivision_score is None
        or request.daughter2_nondivision_score is None
    ):
        raise BifurcationOrchestrationError(
            "Classifier predicted class 0, but both exact legacy nondivision "
            "scores were not supplied"
        )
    if prediction.predicted_class == 2 and request.false_negative_plan is None:
        # ``classify_single_model`` normally demotes this condition to class 0.
        # Keep this guard so the graph resolver can never receive an implicit
        # class-2 request if classifier behavior changes in the future.
        raise BifurcationOrchestrationError(
            "Classifier predicted class 2 without an explicit false-negative plan"
        )

    decision = BifurcationDecision(
        classification=prediction.predicted_class,
        parent_id=request.parent_id,
        daughter1_id=request.daughter1_id,
        daughter2_id=request.daughter2_id,
        daughter1_nondivision_score=request.daughter1_nondivision_score,
        daughter2_nondivision_score=request.daughter2_nondivision_score,
        reattachment_candidates=request.reattachment_candidates,
        false_negative_plan=request.false_negative_plan,
    )
    resolution = resolve_bifurcation(request.state, decision)
    return SingleModelLineageResult(prediction, decision, resolution)


__all__ = [
    "BifurcationOrchestrationError",
    "SingleModelLineageRequest",
    "SingleModelLineageResult",
    "classify_and_resolve_bifurcation",
]
