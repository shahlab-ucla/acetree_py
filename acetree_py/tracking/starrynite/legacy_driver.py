"""Whole-movie orchestration for StarryNite's legacy bifurcation pass.

This module starts at the exact boundary immediately before MATLAB
``greedydeleteFPbranches`` scans tentative divisions.  It deliberately does
not synthesize the preceding easy-link, greedy end-score, or polar-body
passes.  Given a complete raw :class:`LegacyTrackingContext`,
tracking statistics, and a neutralized legacy classifier, it reproduces the
function's isolated-fragment prepass, dynamic frame/row scan, inline class-0
retry classifications, and class 0/1/2/3 graph mutations without modifying
its input.

The output is transaction-like.  If any event cannot be represented exactly,
``status == "unsupported"`` and the returned context/state are the original
objects.  Completed prefix events and ordered classification records remain
available only as diagnostics; no partial lineage is exposed as executable
output.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Literal

from .classifier import (
    AmbigiousClassifierPrediction,
    ClassifierPredictionError,
    NeutralAmbigiousClassifierFamily,
    NeutralNaiveBayesClassifier,
    SingleModelPrediction,
    classify_ambigious_family,
    classify_single_model,
)
from .legacy_class_zero import (
    LegacyClassZeroClassificationObservation,
    LegacyClassZeroRepairDiagnostics,
    LegacyClassZeroRepairError,
    LegacyClassZeroRepairStatus,
    repair_legacy_class_zero_bifurcation,
    synchronize_legacy_context_after_resolution,
)
from .legacy_features import (
    LegacyBifurcationExtraction,
    LegacyFeatureExtractionError,
    LegacyTrackingStatistics,
    extract_legacy_bifurcation_features,
)
from .legacy_mutations import (
    LegacyMutationError,
    resolve_legacy_false_positive_bifurcation,
)
from .legacy_isolated import (
    LegacyIsolatedFragmentParameters,
    apply_legacy_isolated_fragment_prepass,
)
from .legacy_state import LegacyStateError, LegacyTrackingContext
from .lineage import (
    BifurcationDecision,
    LineageGraphState,
    LineageResolutionAction,
    LineageResolutionDiagnostics,
    LineageResolutionError,
    LineageResolutionResult,
    resolve_bifurcation,
)
from .repair_candidates import ClassZeroRepairCandidate, LegacyRepairCandidateError


LegacyClassifierModel = (
    NeutralNaiveBayesClassifier | NeutralAmbigiousClassifierFamily
)
LegacyClassifierPrediction = SingleModelPrediction | AmbigiousClassifierPrediction
LegacyMovieDecisionStatus = Literal["completed", "unsupported"]
LegacyMovieClassifierMode = Literal["single_model", "ambigious_four_model"]
LegacyMovieFailureStage = Literal[
    "feature_extraction",
    "classification",
    "observer",
    "class_zero_repair",
    "lineage_resolution",
    "context_synchronization",
    "postcondition",
]


class LegacyMovieDecisionError(ValueError):
    """Raised when the movie-driver request itself is invalid."""


@dataclass(frozen=True, slots=True)
class LegacyMovieDecisionConfig:
    """Frame range and force-mode controls for the MATLAB-style scan.

    ``end_frame`` has MATLAB ``trackingparameters.endtime`` semantics: it is
    the final movie frame, while tentative bifurcations are scanned only
    through ``end_frame - 1``.  With ``force_mode`` enabled and
    ``force_end_frame`` supplied, force mode applies exactly when
    ``frame < force_end_frame``.
    """

    start_frame: int = 1
    end_frame: int | None = None
    force_mode: bool = False
    force_end_frame: int | None = None
    record_answers: bool = False
    delete_isolated: bool = False
    fp_size_threshold: int = 2
    early_cell_threshold: int = 250
    fp_size_threshold_small: int = 1

    def __post_init__(self) -> None:
        _positive_integer(self.start_frame, "start_frame")
        if self.end_frame is not None:
            _positive_integer(self.end_frame, "end_frame")
            if self.end_frame < self.start_frame:
                raise LegacyMovieDecisionError(
                    "end_frame cannot precede start_frame"
                )
        if type(self.force_mode) is not bool:
            raise TypeError("force_mode must be a boolean")
        if self.force_end_frame is not None:
            _positive_integer(self.force_end_frame, "force_end_frame")
            if not self.force_mode:
                raise LegacyMovieDecisionError(
                    "force_end_frame requires force_mode=True"
                )
        if type(self.record_answers) is not bool:
            raise TypeError("record_answers must be a boolean")
        if type(self.delete_isolated) is not bool:
            raise TypeError("delete_isolated must be a boolean")
        for name in (
            "fp_size_threshold",
            "early_cell_threshold",
            "fp_size_threshold_small",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise LegacyMovieDecisionError(f"{name} cannot be negative")

    def resolved_end_frame(self, context: LegacyTrackingContext) -> int:
        end_frame = (
            context.parameters.end_frame
            if self.end_frame is None
            else self.end_frame
        )
        if end_frame > context.parameters.end_frame:
            raise LegacyMovieDecisionError(
                "end_frame cannot exceed the legacy context end frame"
            )
        if end_frame < self.start_frame:
            raise LegacyMovieDecisionError(
                "start_frame cannot exceed the resolved end frame"
            )
        return end_frame

    def force_at(self, frame: int) -> bool:
        return self.force_mode and (
            self.force_end_frame is None or frame < self.force_end_frame
        )

    @classmethod
    def from_early_tracking_parameters(
        cls,
        parameters: "LegacyEarlyTrackingParameters",
        *,
        force_mode: bool = False,
        force_end_frame: int | None = None,
        record_answers: bool = False,
    ) -> "LegacyMovieDecisionConfig":
        """Carry the shared movie/prepass controls across the exact seam."""

        from .legacy_early import LegacyEarlyTrackingParameters

        if not isinstance(parameters, LegacyEarlyTrackingParameters):
            raise TypeError(
                "parameters must be LegacyEarlyTrackingParameters"
            )
        return cls(
            start_frame=parameters.start_frame,
            end_frame=parameters.end_frame,
            force_mode=force_mode,
            force_end_frame=force_end_frame,
            record_answers=record_answers,
            delete_isolated=parameters.delete_isolated,
            fp_size_threshold=parameters.fp_size_threshold,
            early_cell_threshold=parameters.early_cell_threshold,
            fp_size_threshold_small=parameters.fp_size_threshold_small,
        )


@dataclass(frozen=True, slots=True)
class LegacyMovieClassificationRecord:
    """One top-level or inline classifier call in exact execution order."""

    sequence_index: int
    top_level_event_index: int
    parent_id: str
    daughter_ids: tuple[str, str]
    frame: int
    matlab_row: int
    classifier_round: int
    recursion_depth: int
    force_mode: bool
    prediction: LegacyClassifierPrediction
    raw_attempt_rank: int | None = None
    attachment_source_id: str | None = None
    attachment_target_id: str | None = None

    def __post_init__(self) -> None:
        if self.sequence_index < 0 or self.top_level_event_index < 0:
            raise LegacyMovieDecisionError("Classification indices cannot be negative")
        if type(self.parent_id) is not str or not self.parent_id:
            raise TypeError("parent_id must be a non-empty string")
        if len(self.daughter_ids) != 2 or len(set(self.daughter_ids)) != 2:
            raise LegacyMovieDecisionError(
                "daughter_ids must contain two distinct IDs"
            )
        _positive_integer(self.frame, "frame")
        if (
            isinstance(self.matlab_row, bool)
            or not isinstance(self.matlab_row, Integral)
            or self.matlab_row < 0
        ):
            raise TypeError("matlab_row must be a non-negative integer")
        if self.classifier_round not in {1, 2}:
            raise LegacyMovieDecisionError("classifier_round must be 1 or 2")
        if (
            isinstance(self.recursion_depth, bool)
            or not isinstance(self.recursion_depth, Integral)
            or self.recursion_depth < 0
        ):
            raise TypeError("recursion_depth must be a non-negative integer")
        if type(self.force_mode) is not bool:
            raise TypeError("force_mode must be a boolean")
        if not isinstance(
            self.prediction,
            (SingleModelPrediction, AmbigiousClassifierPrediction),
        ):
            raise TypeError("prediction must be a legacy classifier prediction")
        if self.raw_attempt_rank is not None:
            _positive_integer(self.raw_attempt_rank, "raw_attempt_rank")
        if self.classifier_round == 1 and any(
            value is not None
            for value in (
                self.raw_attempt_rank,
                self.attachment_source_id,
                self.attachment_target_id,
            )
        ):
            raise LegacyMovieDecisionError(
                "Top-level classifications cannot carry class-0 attachment metadata"
            )
        if self.classifier_round == 2 and (
            self.raw_attempt_rank is None
            or self.attachment_source_id is None
            or self.attachment_target_id is None
        ):
            raise LegacyMovieDecisionError(
                "Round-2 classifications require attachment metadata"
            )

    @property
    def effective_class(self) -> int:
        return self.prediction.predicted_class

    @property
    def computed_class(self) -> int:
        if isinstance(self.prediction, AmbigiousClassifierPrediction):
            return self.prediction.computed_class
        return self.prediction.predicted_class


@dataclass(frozen=True, slots=True)
class LegacyMovieClassificationObservation:
    """Transient raw contexts for an optional parity/event-trace observer.

    For round 1, both contexts are the current movie state.  For round 2,
    ``before_attachment_context`` is the detached state and
    ``classification_context`` is the provisional state after writing the raw
    attachment slots but before applying the predicted class.
    """

    record: LegacyMovieClassificationRecord
    extraction: LegacyBifurcationExtraction
    before_attachment_context: LegacyTrackingContext
    classification_context: LegacyTrackingContext
    candidate: ClassZeroRepairCandidate | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record, LegacyMovieClassificationRecord):
            raise TypeError("record must be a LegacyMovieClassificationRecord")
        if not isinstance(self.extraction, LegacyBifurcationExtraction):
            raise TypeError("extraction must be a LegacyBifurcationExtraction")
        if not isinstance(self.before_attachment_context, LegacyTrackingContext):
            raise TypeError(
                "before_attachment_context must be a LegacyTrackingContext"
            )
        if not isinstance(self.classification_context, LegacyTrackingContext):
            raise TypeError("classification_context must be a LegacyTrackingContext")
        if self.record.classifier_round == 1 and self.candidate is not None:
            raise LegacyMovieDecisionError(
                "Top-level observations cannot carry a class-0 candidate"
            )
        if self.record.classifier_round == 2 and not isinstance(
            self.candidate, ClassZeroRepairCandidate
        ):
            raise LegacyMovieDecisionError(
                "Round-2 observations require a class-0 candidate"
            )


LegacyMovieClassificationObserver = Callable[
    [LegacyMovieClassificationObservation], None
]


@dataclass(frozen=True, slots=True)
class LegacyMovieDecisionEvent:
    """One completed top-level bifurcation decision in frame/row order."""

    event_index: int
    classification_sequence_index: int
    extraction: LegacyBifurcationExtraction
    prediction: LegacyClassifierPrediction
    force_mode: bool
    actions: tuple[LineageResolutionAction, ...]
    nested_classification_count: int
    deleted_ids_before: frozenset[str]
    deleted_ids_after: frozenset[str]
    class_zero_status: LegacyClassZeroRepairStatus | None = None
    class_zero_diagnostics: LegacyClassZeroRepairDiagnostics | None = None
    lineage_diagnostics: LineageResolutionDiagnostics | None = None

    def __post_init__(self) -> None:
        if self.event_index < 0 or self.classification_sequence_index < 0:
            raise LegacyMovieDecisionError("Event indices cannot be negative")
        if not isinstance(self.extraction, LegacyBifurcationExtraction):
            raise TypeError("extraction must be a LegacyBifurcationExtraction")
        if not isinstance(
            self.prediction,
            (SingleModelPrediction, AmbigiousClassifierPrediction),
        ):
            raise TypeError("prediction must be a legacy classifier prediction")
        if self.prediction.predicted_class == 0:
            if (
                self.class_zero_status is None
                or self.class_zero_diagnostics is None
                or self.lineage_diagnostics is not None
            ):
                raise LegacyMovieDecisionError(
                    "Class 0 requires only class-zero status/diagnostics"
                )
        elif (
            self.lineage_diagnostics is None
            or self.class_zero_status is not None
            or self.class_zero_diagnostics is not None
        ):
            raise LegacyMovieDecisionError(
                "Classes 1/2/3 require only lineage diagnostics"
            )
        if any(not isinstance(item, LineageResolutionAction) for item in self.actions):
            raise TypeError("actions must contain LineageResolutionAction values")
        object.__setattr__(self, "actions", tuple(self.actions))
        if self.nested_classification_count < 0:
            raise LegacyMovieDecisionError(
                "nested_classification_count cannot be negative"
            )

    @property
    def parent_id(self) -> str:
        return self.extraction.parent_id

    @property
    def daughter_ids(self) -> tuple[str, str]:
        return self.extraction.daughter_ids

    @property
    def effective_class(self) -> int:
        return self.prediction.predicted_class


@dataclass(frozen=True, slots=True)
class LegacyMovieDecisionFailure:
    """Why an exact event was rejected and the transaction rolled back."""

    stage: LegacyMovieFailureStage
    top_level_event_index: int
    parent_id: str
    frame: int
    matlab_row: int
    reason: str
    exception_type: str


@dataclass(frozen=True, slots=True)
class LegacyMovieDecisionResult:
    """Atomic movie result plus an ordered compatibility audit trail."""

    context: LegacyTrackingContext
    state: LineageGraphState
    status: LegacyMovieDecisionStatus
    events: tuple[LegacyMovieDecisionEvent, ...]
    classifications: tuple[LegacyMovieClassificationRecord, ...]
    source_model_sha256: str
    classifier_family: str
    classifier_mode: LegacyMovieClassifierMode
    failure: LegacyMovieDecisionFailure | None = None
    classifier_entry_context: LegacyTrackingContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, LegacyTrackingContext):
            raise TypeError("context must be a LegacyTrackingContext")
        if not isinstance(self.state, LineageGraphState):
            raise TypeError("state must be a LineageGraphState")
        if self.status not in {"completed", "unsupported"}:
            raise LegacyMovieDecisionError("Unknown movie decision status")
        if self.classifier_mode not in {"single_model", "ambigious_four_model"}:
            raise LegacyMovieDecisionError("Unknown movie classifier mode")
        if self.status == "completed" and self.failure is not None:
            raise LegacyMovieDecisionError("A completed result cannot have a failure")
        if self.classifier_entry_context is not None and not isinstance(
            self.classifier_entry_context,
            LegacyTrackingContext,
        ):
            raise TypeError(
                "classifier_entry_context must be LegacyTrackingContext or None"
            )
        if self.status == "unsupported" and self.failure is None:
            raise LegacyMovieDecisionError(
                "An unsupported result must explain its failure"
            )
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "classifications", tuple(self.classifications))
        if self.context.to_lineage_graph_state() != self.state:
            raise LegacyMovieDecisionError(
                "Returned legacy context and active lineage state disagree"
            )

    @property
    def supported(self) -> bool:
        return self.status == "completed"


class _ObserverAbort(ValueError):
    """Internal marker used to convert observer errors into atomic failure."""


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise TypeError(f"{label} must be a positive integer")
    return int(value)


def _classifier_family(model: LegacyClassifierModel) -> str:
    if isinstance(model, NeutralAmbigiousClassifierFamily):
        return model.classifier_family
    return model.classifier_family


def _classifier_mode(model: LegacyClassifierModel) -> LegacyMovieClassifierMode:
    return (
        "ambigious_four_model"
        if isinstance(model, NeutralAmbigiousClassifierFamily)
        else "single_model"
    )


def _same_active_topology(
    first: LineageGraphState,
    second: LineageGraphState,
) -> bool:
    """Compare executable pointers without conflating AT edge annotations.

    MATLAB has no counterpart to AT's ``TrackEdge.kind``.  Synchronizing raw
    successor slots may consistently relabel both members of an untouched
    split as ``split`` even when an imported edge retained the older first-slot
    ``link`` annotation.  Frames, delete flags, and directed pairs are the
    compatibility-bearing postcondition.
    """

    return (
        first.frames == second.frames
        and first.deleted_ids == second.deleted_ids
        and {
            (edge.source_id, edge.target_id) for edge in first.edges
        }
        == {(edge.source_id, edge.target_id) for edge in second.edges}
    )


def _predict(
    model: LegacyClassifierModel,
    extraction: LegacyBifurcationExtraction,
    *,
    force_mode: bool,
) -> LegacyClassifierPrediction:
    classify = (
        classify_ambigious_family
        if isinstance(model, NeutralAmbigiousClassifierFamily)
        else classify_single_model
    )
    return classify(
        model,
        extraction.feature_input,
        force_mode=force_mode,
        backward_repair_available=extraction.false_negative_plan is not None,
    )


def _record(
    *,
    sequence_index: int,
    top_level_event_index: int,
    extraction: LegacyBifurcationExtraction,
    prediction: LegacyClassifierPrediction,
    context: LegacyTrackingContext,
    classifier_round: int,
    recursion_depth: int,
    force_mode: bool,
    candidate: ClassZeroRepairCandidate | None = None,
) -> LegacyMovieClassificationRecord:
    parent = context.nucleus(extraction.parent_id)
    return LegacyMovieClassificationRecord(
        sequence_index=sequence_index,
        top_level_event_index=top_level_event_index,
        parent_id=extraction.parent_id,
        daughter_ids=extraction.daughter_ids,
        frame=parent.frame,
        matlab_row=parent.matlab_row,
        classifier_round=classifier_round,
        recursion_depth=recursion_depth,
        force_mode=force_mode,
        prediction=prediction,
        raw_attempt_rank=None if candidate is None else candidate.raw_rank,
        attachment_source_id=None if candidate is None else candidate.source_id,
        attachment_target_id=(
            None
            if candidate is None
            else extraction.daughter_ids[1]
        ),
    )


def _failure_result(
    *,
    original_context: LegacyTrackingContext,
    original_state: LineageGraphState,
    events: list[LegacyMovieDecisionEvent],
    classifications: list[LegacyMovieClassificationRecord],
    model: LegacyClassifierModel,
    stage: LegacyMovieFailureStage,
    top_level_event_index: int,
    parent_id: str,
    error: BaseException | str,
) -> LegacyMovieDecisionResult:
    parent = original_context.nucleus(parent_id)
    reason = str(error)
    exception_type = type(error).__name__ if isinstance(error, BaseException) else ""
    return LegacyMovieDecisionResult(
        context=original_context,
        state=original_state,
        status="unsupported",
        events=tuple(events),
        classifications=tuple(classifications),
        source_model_sha256=model.source_model_sha256,
        classifier_family=_classifier_family(model),
        classifier_mode=_classifier_mode(model),
        failure=LegacyMovieDecisionFailure(
            stage=stage,
            top_level_event_index=top_level_event_index,
            parent_id=parent_id,
            frame=parent.frame,
            matlab_row=parent.matlab_row,
            reason=reason,
            exception_type=exception_type,
        ),
    )


def run_legacy_movie_decisions(
    context: LegacyTrackingContext,
    model: LegacyClassifierModel,
    statistics: LegacyTrackingStatistics,
    *,
    config: LegacyMovieDecisionConfig | None = None,
    classification_observer: LegacyMovieClassificationObserver | None = None,
) -> LegacyMovieDecisionResult:
    """Run the exact post-greedy classifier/repair pass across one movie.

    The scan order is the literal MATLAB order: increasing frame, then original
    nucleus row.  The current raw state is checked immediately before each row,
    so a branch deleted by an earlier event is skipped later in the same pass.
    Class-0 provisional classifications are recorded inline as round 2 before
    scanning resumes.

    Operational incompatibilities return an atomic unsupported result.  Invalid
    API argument types/configuration still raise immediately.
    """

    if not isinstance(context, LegacyTrackingContext):
        raise TypeError("context must be a LegacyTrackingContext")
    if not isinstance(
        model,
        (NeutralNaiveBayesClassifier, NeutralAmbigiousClassifierFamily),
    ):
        raise TypeError("model must be a supported neutral legacy classifier")
    if not isinstance(statistics, LegacyTrackingStatistics):
        raise TypeError("statistics must be LegacyTrackingStatistics")
    if config is None:
        config = LegacyMovieDecisionConfig()
    if not isinstance(config, LegacyMovieDecisionConfig):
        raise TypeError("config must be a LegacyMovieDecisionConfig or None")
    if classification_observer is not None and not callable(
        classification_observer
    ):
        raise TypeError("classification_observer must be callable or None")

    end_frame = config.resolved_end_frame(context)
    original_context = context
    original_state = context.to_lineage_graph_state()
    isolated_result = apply_legacy_isolated_fragment_prepass(
        context,
        LegacyIsolatedFragmentParameters(
            enabled=config.delete_isolated,
            start_frame=config.start_frame,
            end_frame=end_frame,
            fp_size_threshold=config.fp_size_threshold,
            early_cell_threshold=config.early_cell_threshold,
            fp_size_threshold_small=config.fp_size_threshold_small,
        ),
    )
    working_context = isolated_result.context
    working_state = working_context.to_lineage_graph_state()
    events: list[LegacyMovieDecisionEvent] = []
    classifications: list[LegacyMovieClassificationRecord] = []

    for frame in range(config.start_frame, end_frame):
        # finalpoints row count/order is immutable even when delete flags change.
        for parent_id in context.frame_ids(frame, include_deleted=True):
            if parent_id in working_context.deleted_ids:
                continue
            first_id, second_id = working_context.successor_slots(parent_id)
            if first_id is None or second_id is None:
                continue

            top_level_event_index = len(events)
            force_mode = config.force_at(frame)
            try:
                extraction = extract_legacy_bifurcation_features(
                    working_context,
                    parent_id,
                    statistics,
                    record_answers=config.record_answers,
                )
            except (
                LegacyFeatureExtractionError,
                LegacyRepairCandidateError,
                LegacyStateError,
                LineageResolutionError,
                ValueError,
            ) as exc:
                return _failure_result(
                    original_context=original_context,
                    original_state=original_state,
                    events=events,
                    classifications=classifications,
                    model=model,
                    stage="feature_extraction",
                    top_level_event_index=top_level_event_index,
                    parent_id=parent_id,
                    error=exc,
                )

            try:
                prediction = _predict(
                    model,
                    extraction,
                    force_mode=force_mode,
                )
            except (ClassifierPredictionError, ValueError) as exc:
                return _failure_result(
                    original_context=original_context,
                    original_state=original_state,
                    events=events,
                    classifications=classifications,
                    model=model,
                    stage="classification",
                    top_level_event_index=top_level_event_index,
                    parent_id=parent_id,
                    error=exc,
                )

            top_record = _record(
                sequence_index=len(classifications),
                top_level_event_index=top_level_event_index,
                extraction=extraction,
                prediction=prediction,
                context=working_context,
                classifier_round=1,
                recursion_depth=0,
                force_mode=force_mode,
            )
            classifications.append(top_record)
            if classification_observer is not None:
                try:
                    classification_observer(
                        LegacyMovieClassificationObservation(
                            record=top_record,
                            extraction=extraction,
                            before_attachment_context=working_context,
                            classification_context=working_context,
                        )
                    )
                except Exception as exc:  # observer is an explicit safety boundary
                    return _failure_result(
                        original_context=original_context,
                        original_state=original_state,
                        events=events,
                        classifications=classifications,
                        model=model,
                        stage="observer",
                        top_level_event_index=top_level_event_index,
                        parent_id=parent_id,
                        error=exc,
                    )

            deleted_before = working_context.deleted_ids
            nested_count_before = len(classifications)
            observer_error: Exception | None = None

            def observe_nested(
                observation: LegacyClassZeroClassificationObservation,
            ) -> None:
                nonlocal observer_error
                nested_record = _record(
                    sequence_index=len(classifications),
                    top_level_event_index=top_level_event_index,
                    extraction=observation.extraction,
                    prediction=observation.prediction,
                    context=observation.classification_context,
                    classifier_round=observation.classifier_round,
                    recursion_depth=observation.recursion_depth,
                    force_mode=observation.force_mode,
                    candidate=observation.candidate,
                )
                classifications.append(nested_record)
                if classification_observer is None:
                    return
                try:
                    classification_observer(
                        LegacyMovieClassificationObservation(
                            record=nested_record,
                            extraction=observation.extraction,
                            before_attachment_context=(
                                observation.before_attachment_context
                            ),
                            classification_context=observation.classification_context,
                            candidate=observation.candidate,
                        )
                    )
                except Exception as exc:  # see top-level observer boundary
                    observer_error = exc
                    raise _ObserverAbort(str(exc)) from exc

            if prediction.predicted_class == 0:
                try:
                    class_zero_result = repair_legacy_class_zero_bifurcation(
                        working_context,
                        working_state,
                        extraction,
                        model,
                        statistics,
                        force_mode=force_mode,
                        record_answers=config.record_answers,
                        classification_observer=observe_nested,
                    )
                except (
                    LegacyClassZeroRepairError,
                    LegacyFeatureExtractionError,
                    LegacyRepairCandidateError,
                    LegacyStateError,
                    LineageResolutionError,
                    ClassifierPredictionError,
                    ValueError,
                ) as exc:
                    return _failure_result(
                        original_context=original_context,
                        original_state=original_state,
                        events=events,
                        classifications=classifications,
                        model=model,
                        stage="observer" if observer_error is not None else "class_zero_repair",
                        top_level_event_index=top_level_event_index,
                        parent_id=parent_id,
                        error=observer_error if observer_error is not None else exc,
                    )
                if not class_zero_result.supported:
                    return _failure_result(
                        original_context=original_context,
                        original_state=original_state,
                        events=events,
                        classifications=classifications,
                        model=model,
                        stage="observer" if observer_error is not None else "class_zero_repair",
                        top_level_event_index=top_level_event_index,
                        parent_id=parent_id,
                        error=(
                            observer_error
                            if observer_error is not None
                            else (
                                class_zero_result.diagnostics.failure_reason
                                or "Legacy class-0 repair was unsupported"
                            )
                        ),
                    )
                next_context = class_zero_result.context
                next_state = class_zero_result.state
                actions = class_zero_result.actions
                lineage_result = None
            else:
                if prediction.predicted_class == 3:
                    try:
                        raw_resolution = (
                            resolve_legacy_false_positive_bifurcation(
                                working_context,
                                extraction.parent_id,
                                extraction.daughter_ids,
                            )
                        )
                        next_context = raw_resolution.context
                        lineage_result = LineageResolutionResult(
                            raw_resolution.state,
                            raw_resolution.actions,
                            raw_resolution.diagnostics,
                        )
                        next_state = lineage_result.state
                    except (
                        LegacyMutationError,
                        LegacyStateError,
                        LineageResolutionError,
                        ValueError,
                    ) as exc:
                        return _failure_result(
                            original_context=original_context,
                            original_state=original_state,
                            events=events,
                            classifications=classifications,
                            model=model,
                            stage="lineage_resolution",
                            top_level_event_index=top_level_event_index,
                            parent_id=parent_id,
                            error=exc,
                        )
                    actions = lineage_result.actions
                    class_zero_result = None
                    nested_count = len(classifications) - nested_count_before
                    events.append(
                        LegacyMovieDecisionEvent(
                            event_index=top_level_event_index,
                            classification_sequence_index=top_record.sequence_index,
                            extraction=extraction,
                            prediction=prediction,
                            force_mode=force_mode,
                            actions=actions,
                            nested_classification_count=nested_count,
                            deleted_ids_before=deleted_before,
                            deleted_ids_after=next_context.deleted_ids,
                            lineage_diagnostics=lineage_result.diagnostics,
                        )
                    )
                    working_context = next_context
                    working_state = next_state
                    continue
                decision = BifurcationDecision(
                    classification=prediction.predicted_class,
                    parent_id=extraction.parent_id,
                    daughter1_id=extraction.daughter_ids[0],
                    daughter2_id=extraction.daughter_ids[1],
                    daughter1_nondivision_score=extraction.nondivision_scores[0],
                    daughter2_nondivision_score=extraction.nondivision_scores[1],
                    false_negative_plan=extraction.false_negative_plan,
                )
                try:
                    lineage_result = resolve_bifurcation(working_state, decision)
                except (LineageResolutionError, ValueError) as exc:
                    return _failure_result(
                        original_context=original_context,
                        original_state=original_state,
                        events=events,
                        classifications=classifications,
                        model=model,
                        stage="lineage_resolution",
                        top_level_event_index=top_level_event_index,
                        parent_id=parent_id,
                        error=exc,
                    )
                if prediction.predicted_class == 1:
                    next_context = working_context
                    next_state = working_state
                else:
                    try:
                        next_context = synchronize_legacy_context_after_resolution(
                            working_context,
                            lineage_result.state,
                        )
                        next_state = next_context.to_lineage_graph_state()
                    except (
                        LegacyClassZeroRepairError,
                        LegacyStateError,
                        LineageResolutionError,
                        ValueError,
                    ) as exc:
                        return _failure_result(
                            original_context=original_context,
                            original_state=original_state,
                            events=events,
                            classifications=classifications,
                            model=model,
                            stage="context_synchronization",
                            top_level_event_index=top_level_event_index,
                            parent_id=parent_id,
                            error=exc,
                        )
                    if not _same_active_topology(next_state, lineage_result.state):
                        return _failure_result(
                            original_context=original_context,
                            original_state=original_state,
                            events=events,
                            classifications=classifications,
                            model=model,
                            stage="postcondition",
                            top_level_event_index=top_level_event_index,
                            parent_id=parent_id,
                            error=(
                                "Raw legacy synchronization changed the active "
                                "lineage resolution"
                            ),
                        )
                actions = lineage_result.actions
                class_zero_result = None

            nested_count = len(classifications) - nested_count_before
            events.append(
                LegacyMovieDecisionEvent(
                    event_index=top_level_event_index,
                    classification_sequence_index=top_record.sequence_index,
                    extraction=extraction,
                    prediction=prediction,
                    force_mode=force_mode,
                    actions=actions,
                    nested_classification_count=nested_count,
                    deleted_ids_before=deleted_before,
                    deleted_ids_after=next_context.deleted_ids,
                    class_zero_status=(
                        class_zero_result.status
                        if class_zero_result is not None
                        else None
                    ),
                    class_zero_diagnostics=(
                        class_zero_result.diagnostics
                        if class_zero_result is not None
                        else None
                    ),
                    lineage_diagnostics=(
                        lineage_result.diagnostics
                        if lineage_result is not None
                        else None
                    ),
                )
            )
            working_context = next_context
            working_state = next_state

    return LegacyMovieDecisionResult(
        context=working_context,
        state=working_state,
        status="completed",
        events=tuple(events),
        classifications=tuple(classifications),
        source_model_sha256=model.source_model_sha256,
        classifier_family=_classifier_family(model),
        classifier_mode=_classifier_mode(model),
        classifier_entry_context=isolated_result.context,
    )


__all__ = [
    "LegacyMovieClassificationObservation",
    "LegacyMovieClassificationObserver",
    "LegacyMovieClassificationRecord",
    "LegacyMovieClassifierMode",
    "LegacyMovieDecisionConfig",
    "LegacyMovieDecisionError",
    "LegacyMovieDecisionEvent",
    "LegacyMovieDecisionFailure",
    "LegacyMovieDecisionResult",
    "LegacyMovieDecisionStatus",
    "LegacyMovieFailureStage",
    "run_legacy_movie_decisions",
]
