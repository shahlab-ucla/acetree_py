"""Executable, source-bound StarryNite whole-movie compatibility tracker."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..api import (
    Calibration,
    ComponentSpec,
    Detection,
    TrackerGraphResult,
    WholeMoviePreflightContext,
)
from .classifier import (
    NEUTRAL_AMBIGIOUS_CLASSIFIER_SCHEMA,
    NEUTRAL_CLASSIFIER_SCHEMA,
    NeutralAmbigiousClassifierFamily,
    NeutralNaiveBayesClassifier,
    load_neutral_ambigious_classifier,
    load_neutral_classifier,
)
from .compatibility import (
    LEGACY_EXACT_REFINEMENT_BACKEND,
    build_compatibility_report,
    select_compatibility_backend,
)
from .legacy_driver import (
    LegacyMovieDecisionConfig,
    LegacyMovieDecisionResult,
    run_legacy_movie_decisions,
)
from .legacy_early import LegacyEarlyStageSummary, run_legacy_early_tracking
from .legacy_runtime import PreparedLegacyRuntime, prepare_legacy_runtime
from .legacy_state import LegacyNucleus, LegacyTrackingContext
from .models import sha256_file
from .presets import StarryNiteTuningProfile, legacy_stage_index, load_tuning_profile


STARRYNITE_LEGACY_EXACT_TRACKER_ID = "acetree.starrynite_legacy_exact"
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_PARAMETER_FILE = "STARRYNITE_PARAMETER_FILE"
_PARAMETER_SHA256 = "STARRYNITE_PARAMETER_SHA256"
_MODEL_FILE = "STARRYNITE_MODEL_FILE"
_MODEL_SHA256 = "STARRYNITE_MODEL_SHA256"
_CLASSIFIER_FILE = "STARRYNITE_NEUTRAL_CLASSIFIER_FILE"
_CLASSIFIER_SHA256 = "STARRYNITE_NEUTRAL_CLASSIFIER_SHA256"
_MODE = "STARRYNITE_COMPATIBILITY_MODE"
_FORCE_MODE = "STARRYNITE_FORCE_MODE"
_FORCE_END_FRAME = "STARRYNITE_FORCE_END_FRAME"
_RECORD_ANSWERS = "STARRYNITE_RECORD_ANSWERS"
_REQUIRE_EXACT_TAIL = "STARRYNITE_REQUIRE_EXACT_DETECTOR_TAIL"
_STATIC_DIAMETER = "STARRYNITE_USE_STATIC_DIAMETER"
_DISTRIBUTION_FILE = "STARRYNITE_DISTRIBUTION_FILE"
_DISTRIBUTION_SHA256 = "STARRYNITE_DISTRIBUTION_SHA256"
_DISTRIBUTION_SOURCE_SHA256 = "STARRYNITE_DISTRIBUTION_SOURCE_SHA256"
_CANDIDATE_DIAMETERS = "LEGACY_CANDIDATE_DIAMETERS_XY_PX"
_CANDIDATE_COUNT = "LEGACY_CANDIDATE_DIAMETER_COUNT"
_CANDIDATE_MEDIAN = "LEGACY_CANDIDATE_DIAMETER_MEDIAN_XY_PX"
_PREVIOUS_CANDIDATE_COUNT = "LEGACY_PREVIOUS_CANDIDATE_DIAMETER_COUNT"
_PREVIOUS_CANDIDATE_MEDIAN = (
    "LEGACY_PREVIOUS_CANDIDATE_DIAMETER_MEDIAN_XY_PX"
)

_DEFAULT_SETTINGS = MappingProxyType(
    {
        _MODE: LEGACY_EXACT_REFINEMENT_BACKEND,
        _PARAMETER_FILE: "",
        _PARAMETER_SHA256: "",
        _MODEL_FILE: "",
        _MODEL_SHA256: "",
        _CLASSIFIER_FILE: "",
        _CLASSIFIER_SHA256: "",
        _FORCE_MODE: False,
        _FORCE_END_FRAME: 0,
        _RECORD_ANSWERS: False,
        _REQUIRE_EXACT_TAIL: True,
        _STATIC_DIAMETER: False,
        "ALLOW_TRACK_SPLITTING": True,
    }
)


class StarryNiteLegacyExactError(ValueError):
    """Raised before any proposal when exact execution is not proven safe."""


LegacyNeutralClassifier = (
    NeutralNaiveBayesClassifier | NeutralAmbigiousClassifierFamily
)


class StarryNiteLegacyExactTracker:
    """Compose exact detector rows, early geometry, and classifier mutations."""

    plugin_id = STARRYNITE_LEGACY_EXACT_TRACKER_ID
    display_name = "StarryNite legacy exact (whole movie)"
    default_settings = _DEFAULT_SETTINGS

    def track(
        self,
        _detections: Sequence[Detection],
        _settings: Mapping[str, Any],
    ):
        raise StarryNiteLegacyExactError(
            "The legacy exact tracker is global-only and must run through the "
            "whole-movie refine_movie boundary"
        )

    def preflight_movie(
        self,
        settings: Mapping[str, Any],
        *,
        context: WholeMoviePreflightContext,
    ) -> None:
        """Fail closed on exact-source incompatibility before image detection."""

        if not isinstance(context, WholeMoviePreflightContext):
            raise TypeError("context must be WholeMoviePreflightContext")
        if not context.covers_complete_global_movie:
            raise StarryNiteLegacyExactError(
                "Legacy exact tracking requires an unseeded global request covering "
                "the complete image source from frame 1 through frame "
                f"{context.source_num_timepoints}"
            )
        _load_and_validate_exact_inputs(
            settings,
            detector_spec=context.detector_spec,
            calibration=context.calibration,
            start_frame=context.scope.start_frame,
            end_frame=context.scope.end_frame,
        )

    def refine_movie(
        self,
        detections: Sequence[Detection],
        settings: Mapping[str, Any],
        *,
        detector_spec: ComponentSpec,
        calibration: Calibration,
        start_frame: int,
        end_frame: int,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> TrackerGraphResult:
        """Run the complete supported legacy movie path atomically."""

        _check_cancelled(cancelled)
        values, profile, prepared, classifier = _load_and_validate_exact_inputs(
            settings,
            detector_spec=detector_spec,
            calibration=calibration,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        _check_cancelled(cancelled)

        ordered_detections = tuple(detections)
        _validate_detections(
            ordered_detections,
            profile=profile,
            values=values,
            start_frame=start_frame,
            end_frame=end_frame,
            require_exact_tail=values[_REQUIRE_EXACT_TAIL],
        )
        initial_context = LegacyTrackingContext.from_nuclei_and_edges(
            tuple(LegacyNucleus.from_detection(item) for item in ordered_detections),
            (),
            prepared.feature_parameters,
        )
        if progress is not None:
            progress(end_frame, end_frame, "Building legacy tentative lineage")
        early = run_legacy_early_tracking(
            initial_context,
            prepared.early_parameters,
            prepared.statistics,
            capture_snapshots=False,
            capture_summaries=True,
        )
        _check_cancelled(cancelled)

        cancellation_observed = False

        def observe_classification(_observation) -> None:
            nonlocal cancellation_observed
            if cancelled is not None and cancelled():
                cancellation_observed = True
                raise RuntimeError("Tracking was cancelled")

        force_end = values[_FORCE_END_FRAME] or None
        decision = run_legacy_movie_decisions(
            early.context,
            classifier,
            prepared.statistics,
            config=LegacyMovieDecisionConfig.from_early_tracking_parameters(
                prepared.early_parameters,
                force_mode=values[_FORCE_MODE],
                force_end_frame=force_end,
                record_answers=values[_RECORD_ANSWERS],
            ),
            classification_observer=observe_classification,
        )
        if cancellation_observed:
            _raise_cancelled()
        _check_cancelled(cancelled)
        if not decision.supported:
            assert decision.failure is not None
            failure = decision.failure
            raise StarryNiteLegacyExactError(
                "Legacy classifier refinement failed closed at "
                f"{failure.stage}, frame {failure.frame}, row "
                f"{failure.matlab_row}: {failure.reason}"
            )
        _validate_event_order(decision)

        state = decision.state
        retained = tuple(
            item
            for item in ordered_detections
            if item.detection_id not in state.deleted_ids
        )
        rejected = tuple(
            item.detection_id
            for item in ordered_detections
            if item.detection_id in state.deleted_ids
        )
        provenance = _provenance(
            profile,
            values,
            prepared,
            early.stage_summaries,
            decision,
            retained_count=len(retained),
            rejected_count=len(rejected),
        )
        return TrackerGraphResult(
            detections=retained,
            edges=state.edges,
            rejected_detection_ids=rejected,
            warnings=(),
            provenance=provenance,
        )


def _load_and_validate_exact_inputs(
    settings: Mapping[str, Any],
    *,
    detector_spec: ComponentSpec,
    calibration: Calibration,
    start_frame: int,
    end_frame: int,
) -> tuple[
    dict[str, Any],
    StarryNiteTuningProfile,
    PreparedLegacyRuntime,
    LegacyNeutralClassifier,
]:
    """Read and validate every source-bound exact input without retaining state."""

    values = _validated_settings(settings)
    if not isinstance(calibration, Calibration):
        raise TypeError("calibration must be Calibration")
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise TypeError("start_frame must be an integer")
    if start_frame != 1:
        raise StarryNiteLegacyExactError(
            "Legacy exact tracking must start at frame 1 so initialization and "
            "dynamic row order have complete temporal history"
        )
    if isinstance(end_frame, bool) or not isinstance(end_frame, int):
        raise TypeError("end_frame must be an integer")
    if end_frame < start_frame:
        raise StarryNiteLegacyExactError("end_frame cannot precede start_frame")

    profile = load_tuning_profile(values[_PARAMETER_FILE])
    report = build_compatibility_report(
        profile,
        neutral_classifier_path=values[_CLASSIFIER_FILE],
        runtime_capabilities=(LEGACY_EXACT_REFINEMENT_BACKEND,),
    )
    select_compatibility_backend(report, LEGACY_EXACT_REFINEMENT_BACKEND)
    _validate_bound_sources(profile, values)
    _validate_detector_spec(profile, values, detector_spec)
    _validate_calibration(profile, calibration)

    prepared = prepare_legacy_runtime(
        profile,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if prepared.model.sha256 != values[_MODEL_SHA256]:
        raise StarryNiteLegacyExactError(
            "The tracking MAT file changed while exact inputs were being loaded"
        )
    classifier = _load_classifier(
        values[_CLASSIFIER_FILE],
        expected_model_sha256=prepared.model.sha256,
    )
    if sha256_file(values[_CLASSIFIER_FILE]) != values[_CLASSIFIER_SHA256]:
        raise StarryNiteLegacyExactError(
            "The neutral classifier export changed while exact inputs were being "
            "loaded"
        )
    return values, profile, prepared, classifier


def _validated_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")
    unknown = set(settings) - set(_DEFAULT_SETTINGS)
    if unknown:
        raise StarryNiteLegacyExactError(
            "Unsupported legacy exact tracker setting(s): "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    values = dict(_DEFAULT_SETTINGS)
    values.update(settings)
    if values[_MODE] != LEGACY_EXACT_REFINEMENT_BACKEND:
        raise StarryNiteLegacyExactError(
            f"{_MODE} must explicitly select {LEGACY_EXACT_REFINEMENT_BACKEND!r}"
        )
    for key in (_PARAMETER_FILE, _MODEL_FILE, _CLASSIFIER_FILE):
        raw = values[key]
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            raise StarryNiteLegacyExactError(f"{key} must select an existing file")
        path = Path(raw).expanduser().resolve(strict=False)
        if not path.is_file():
            raise StarryNiteLegacyExactError(f"{key} was not found: {path}")
        values[key] = str(path)
    for key in (_PARAMETER_SHA256, _MODEL_SHA256, _CLASSIFIER_SHA256):
        digest = values[key]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise StarryNiteLegacyExactError(
                f"{key} must be a lowercase SHA-256 digest"
            )
    for key in (
        _FORCE_MODE,
        _RECORD_ANSWERS,
        _REQUIRE_EXACT_TAIL,
        _STATIC_DIAMETER,
        "ALLOW_TRACK_SPLITTING",
    ):
        if type(values[key]) is not bool:
            raise TypeError(f"{key} must be boolean")
    if not values["ALLOW_TRACK_SPLITTING"]:
        raise StarryNiteLegacyExactError(
            "Legacy exact tracking intrinsically includes division hypotheses"
        )
    if not values[_REQUIRE_EXACT_TAIL]:
        raise StarryNiteLegacyExactError(
            "Legacy exact tracking requires source-bound exact detector-tail rows"
        )
    force_end = values[_FORCE_END_FRAME]
    if isinstance(force_end, bool) or not isinstance(force_end, int) or force_end < 0:
        raise TypeError(f"{_FORCE_END_FRAME} must be a non-negative integer")
    if force_end and not values[_FORCE_MODE]:
        raise StarryNiteLegacyExactError(
            f"{_FORCE_END_FRAME} requires {_FORCE_MODE}=True"
        )
    return values


def _validate_bound_sources(
    profile: StarryNiteTuningProfile,
    values: Mapping[str, Any],
) -> None:
    parameter_path = profile.parameters.source_path
    if parameter_path is None or parameter_path.resolve(strict=False) != Path(
        values[_PARAMETER_FILE]
    ):
        raise StarryNiteLegacyExactError(
            "The loaded parameter source does not match the requested source"
        )
    if profile.parameter_sha256 != values[_PARAMETER_SHA256]:
        raise StarryNiteLegacyExactError(
            "The parameter-file SHA-256 does not match the loaded source"
        )
    if profile.model_path is None or profile.model_path.resolve(strict=False) != Path(
        values[_MODEL_FILE]
    ):
        raise StarryNiteLegacyExactError(
            "The active MAT model path does not match the exact tracker request"
        )
    if profile.model_sha256 != values[_MODEL_SHA256]:
        raise StarryNiteLegacyExactError(
            "The MAT model SHA-256 does not match the loaded source"
        )
    if sha256_file(values[_PARAMETER_FILE]) != values[_PARAMETER_SHA256]:
        raise StarryNiteLegacyExactError(
            "The parameter file changed after the exact request was created"
        )
    if sha256_file(values[_MODEL_FILE]) != values[_MODEL_SHA256]:
        raise StarryNiteLegacyExactError(
            "The MAT model changed after the exact request was created"
        )
    if sha256_file(values[_CLASSIFIER_FILE]) != values[_CLASSIFIER_SHA256]:
        raise StarryNiteLegacyExactError(
            "The neutral classifier changed after the exact request was created"
        )


def _validate_calibration(
    profile: StarryNiteTuningProfile,
    calibration: Calibration,
) -> None:
    assert profile.xy_um is not None and profile.z_um is not None
    for label, expected, actual in (
        ("xyres", float(profile.xy_um), calibration.xy_um),
        ("zres", float(profile.z_um), calibration.z_um),
    ):
        tolerance = max(1e-9, 1e-6 * max(abs(expected), abs(actual)))
        if abs(expected - actual) > tolerance:
            raise StarryNiteLegacyExactError(
                f"Legacy {label} is {expected:g}, but the dataset calibration is "
                f"{actual:g}; exact tracking will not rescale one source silently"
            )


def _validate_detector_spec(
    profile: StarryNiteTuningProfile,
    values: Mapping[str, Any],
    detector_spec: ComponentSpec,
) -> None:
    if not isinstance(detector_spec, ComponentSpec):
        raise TypeError("detector_spec must be ComponentSpec")
    if detector_spec.plugin_id != "acetree.starrynite_detector":
        raise StarryNiteLegacyExactError(
            "Legacy exact tracking requires the registered StarryNite detector"
        )
    settings = detector_spec.settings
    parameter_path = _mapping_path(settings, _PARAMETER_FILE)
    if parameter_path != Path(values[_PARAMETER_FILE]):
        raise StarryNiteLegacyExactError(
            "The detector parameter source does not match the exact tracker request"
        )
    if settings.get(_PARAMETER_SHA256) != values[_PARAMETER_SHA256]:
        raise StarryNiteLegacyExactError(
            "The detector parameter hash does not match the exact tracker request"
        )
    distribution_value = profile.detector_settings.get(_DISTRIBUTION_FILE)
    if not distribution_value:
        raise StarryNiteLegacyExactError(
            "The loaded parameter profile does not select an exact detector "
            "distribution file"
        )
    expected_distribution = Path(str(distribution_value)).resolve(strict=False)
    if not expected_distribution.is_file():
        raise StarryNiteLegacyExactError(
            "The exact detector distribution file was not found: "
            f"{expected_distribution}. No alternate file was substituted."
        )
    if _mapping_path(settings, "STARRYNITE_DISTRIBUTION_FILE") != expected_distribution:
        raise StarryNiteLegacyExactError(
            "The detector distribution source does not match the parameter profile"
        )
    expected_distribution_sha256 = _distribution_source_sha256(profile)
    if settings.get(_DISTRIBUTION_SOURCE_SHA256) != expected_distribution_sha256:
        raise StarryNiteLegacyExactError(
            "The detector distribution hash binding does not match the parameter "
            "profile"
        )
    current_distribution_sha256 = sha256_file(expected_distribution)
    if current_distribution_sha256 != expected_distribution_sha256:
        raise StarryNiteLegacyExactError(
            "The exact detector distribution changed after the parameter profile "
            "was loaded; reload the source before tracking"
        )

    for key in (
        "DO_SUBPIXEL_LOCALIZATION",
        "DO_MEDIAN_FILTERING",
        "DARK_NUCLEI",
        "ROI_CROPPED",
    ):
        if bool(settings.get(key, False)):
            raise StarryNiteLegacyExactError(
                f"{key}=True is a native detector mutation outside exact "
                "processVolume replay"
            )
    for key in (
        "MIN_LOCAL_CONTRAST",
        "MIN_SEPARATION",
        "THRESHOLD",
        "ROI_X_OFFSET",
        "ROI_Y_OFFSET",
    ):
        raw = settings.get(key, 0.0)
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise StarryNiteLegacyExactError(f"{key} must be numeric") from exc
        if number != 0.0:
            raise StarryNiteLegacyExactError(
                f"{key}={number:g} is an explicit native-only detector override"
            )
    roi_points = settings.get("ROI_POINTS_XY", ())
    if roi_points not in (None, "", (), []):
        raise StarryNiteLegacyExactError(
            "ROI_POINTS_XY must come from the source-bound parameter file in exact "
            "mode, not from an implicit detector override"
        )
    static_diameter = settings.get(_STATIC_DIAMETER, False)
    if type(static_diameter) is not bool:
        raise TypeError(f"Detector {_STATIC_DIAMETER} must be boolean")
    if static_diameter != values[_STATIC_DIAMETER]:
        raise StarryNiteLegacyExactError(
            "The detector dynamic/static diameter mode does not match the exact "
            "tracker request"
        )
    normalized_settings = profile.parameters.normalized_settings
    if "downsampling" not in normalized_settings:
        raise StarryNiteLegacyExactError(
            "Exact mode requires an explicit scalar downsampling=1 assignment in "
            "the source parameter file"
        )
    downsampling = normalized_settings["downsampling"]
    if isinstance(downsampling, (bool, str, tuple)):
        raise StarryNiteLegacyExactError(
            "Legacy downsampling must resolve to the scalar value 1 in exact mode"
        )
    if float(downsampling) != 1.0:
        raise StarryNiteLegacyExactError(
            "Legacy downsampling other than 1 is not reproduced by the AT movie "
            "boundary"
        )

    for key in (
        "RADIUS",
        "SIGMA",
        "INTENSITY_THRESHOLD",
        "BOUNDARY_PERCENT",
        "LARGE_RAY_THRESHOLD",
        "SMALL_RAY_THRESHOLD",
        "RANGE_THRESHOLD",
        "MERGE_LOWER",
        "MERGE_SPLIT",
        "SPLIT_THRESHOLD",
        "NNDIST_MERGE",
        "AR_MERGE",
        "STARRYNITE_CELL_COUNT",
        "STARRYNITE_STAGE_INDEX",
    ):
        if key not in profile.detector_settings:
            continue
        if not _same_setting(settings.get(key), profile.detector_settings[key]):
            raise StarryNiteLegacyExactError(
                f"Detector setting {key} differs from the source-bound parameter "
                "profile. Save the tuned parameter copy and reload it before exact "
                "tracking."
            )


def _mapping_path(settings: Mapping[str, Any], key: str) -> Path | None:
    value = settings.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    return Path(value).expanduser().resolve(strict=False)


def _distribution_source_sha256(profile: StarryNiteTuningProfile) -> str:
    value = profile.detector_settings.get(_DISTRIBUTION_SOURCE_SHA256)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StarryNiteLegacyExactError(
            "The parameter profile has no valid request-time detector distribution "
            "SHA-256 binding"
        )
    return value


def _same_setting(first: Any, second: Any) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second
    try:
        left = float(first)
        right = float(second)
    except (TypeError, ValueError):
        return first == second
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _load_classifier(
    path: str | Path,
    *,
    expected_model_sha256: str,
) -> LegacyNeutralClassifier:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarryNiteLegacyExactError(
            f"Could not inspect neutral classifier {source}: {exc}"
        ) from exc
    schema = payload.get("schema") if isinstance(payload, Mapping) else None
    if schema == NEUTRAL_CLASSIFIER_SCHEMA:
        return load_neutral_classifier(
            source,
            expected_source_model_sha256=expected_model_sha256,
        )
    if schema == NEUTRAL_AMBIGIOUS_CLASSIFIER_SCHEMA:
        return load_neutral_ambigious_classifier(
            source,
            expected_source_model_sha256=expected_model_sha256,
        )
    raise StarryNiteLegacyExactError(
        f"Unsupported neutral classifier schema {schema!r}"
    )


def _validate_detections(
    detections: Sequence[Detection],
    *,
    profile: StarryNiteTuningProfile,
    values: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
    require_exact_tail: bool,
) -> None:
    if any(not isinstance(item, Detection) for item in detections):
        raise TypeError("detections must contain Detection values")
    identifiers = [item.detection_id for item in detections]
    if len(identifiers) != len(set(identifiers)):
        raise StarryNiteLegacyExactError("Detector IDs must be unique")
    outside = [
        item.detection_id
        for item in detections
        if item.frame < start_frame or item.frame > end_frame
    ]
    if outside:
        raise StarryNiteLegacyExactError(
            "Detector rows lie outside the exact movie scope: "
            + ", ".join(outside[:5])
        )
    if require_exact_tail:
        distribution_value = profile.detector_settings.get(_DISTRIBUTION_FILE)
        if not distribution_value:
            raise StarryNiteLegacyExactError(
                "The parameter profile does not select a detector distribution "
                "file for the exact tail"
            )
        distribution_path = Path(str(distribution_value)).resolve(strict=False)
        if not distribution_path.is_file():
            raise StarryNiteLegacyExactError(
                "The exact detector distribution file was not found: "
                f"{distribution_path}. No alternate model was substituted."
            )
        expected_distribution_sha256 = _distribution_source_sha256(profile)
        distribution_sha256 = sha256_file(distribution_path)
        if distribution_sha256 != expected_distribution_sha256:
            raise StarryNiteLegacyExactError(
                "The exact detector distribution changed after the request was "
                "created; reload the parameter source before tracking"
            )
        incompatible = [
            item.detection_id
            for item in detections
            if (
                item.features.get("LEGACY_EXACT_TAIL") is not True
                or _feature_path(
                    item,
                    "STARRYNITE_PARAMETER_FILE",
                )
                != Path(values[_PARAMETER_FILE])
                or item.features.get("STARRYNITE_PARAMETER_SHA256")
                != values[_PARAMETER_SHA256]
                or _feature_path(
                    item,
                    _DISTRIBUTION_FILE,
                )
                != distribution_path
                or item.features.get(_DISTRIBUTION_SHA256)
                != expected_distribution_sha256
                or item.features.get(_DISTRIBUTION_SOURCE_SHA256)
                != expected_distribution_sha256
            )
        ]
        if incompatible:
            raise StarryNiteLegacyExactError(
                "Legacy exact tracking requires every detector row to be bound to "
                "the same parameter source and exact distribution MAT file; "
                "incompatible detection IDs include "
                + ", ".join(incompatible[:5])
            )
        _validate_dynamic_detector_provenance(
            detections,
            profile=profile,
            values=values,
            start_frame=start_frame,
            end_frame=end_frame,
        )


def _validate_dynamic_detector_provenance(
    detections: Sequence[Detection],
    *,
    profile: StarryNiteTuningProfile,
    values: Mapping[str, Any],
    start_frame: int,
    end_frame: int,
) -> None:
    """Prove that rows came from one sequential ``processVolume`` detector."""

    by_frame: dict[int, list[Detection]] = {
        frame: [] for frame in range(start_frame, end_frame + 1)
    }
    for detection in detections:
        by_frame[detection.frame].append(detection)

    normalized = profile.parameters.normalized_settings
    initial_count = int(normalized["firsttimestepnumcells"])
    initial_diameter = float(normalized["firsttimestepdiam"])
    use_static_diameter = values[_STATIC_DIAMETER]
    previous_final_count = 0
    # A non-empty detector frame exposes and proves its full candidate vector.
    # A zero-output frame cannot carry row features, so ``None`` means the next
    # non-empty frame must supply that missing frame's behaviorally sufficient
    # count/median handoff. processVolume's diameter update uses no other part
    # of the vector.
    previous_candidate_summary: tuple[int, float | None] | None = (0, None)

    for frame in range(start_frame, end_frame + 1):
        frame_rows = by_frame[frame]
        row_indices = [_row_index(item) for item in frame_rows]
        if sorted(row_indices) != list(range(len(frame_rows))):
            raise StarryNiteLegacyExactError(
                f"Legacy detector row indices at frame {frame} must be unique and "
                f"contiguous from 0; received {sorted(row_indices)!r}"
            )

        expected_count = initial_count if frame == start_frame else previous_final_count
        expected_stage = legacy_stage_index(profile.parameters, expected_count)
        if frame_rows:
            first = frame_rows[0]
            signature = _dynamic_detector_scalar_signature(first)
            (
                effective_diameter,
                cell_count,
                stage_index,
                static_mode,
                candidate_count,
                candidate_median,
                reported_previous_count,
                reported_previous_median,
            ) = signature
            reported_previous_summary = (
                reported_previous_count,
                reported_previous_median,
            )
            if (
                previous_candidate_summary is not None
                and reported_previous_summary != previous_candidate_summary
            ):
                raise StarryNiteLegacyExactError(
                    f"Legacy detector frame {frame} reports previous candidate "
                    f"summary {reported_previous_summary!r}, but sequential replay "
                    f"requires {previous_candidate_summary!r}"
                )
            expected_diameter = _diameter_from_candidate_summary(
                initial_diameter,
                reported_previous_summary,
                use_static_diameter=use_static_diameter,
            )
            if cell_count != expected_count:
                raise StarryNiteLegacyExactError(
                    f"Legacy detector frame {frame} reports cell count {cell_count}, "
                    f"but sequential replay requires {expected_count}"
                )
            if stage_index != expected_stage:
                raise StarryNiteLegacyExactError(
                    f"Legacy detector frame {frame} reports stage {stage_index}, "
                    f"but cell count {expected_count} selects stage {expected_stage}"
                )
            if static_mode is not use_static_diameter:
                raise StarryNiteLegacyExactError(
                    f"Legacy detector frame {frame} does not match the requested "
                    "dynamic/static diameter mode"
                )
            if not math.isclose(
                effective_diameter,
                expected_diameter,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise StarryNiteLegacyExactError(
                    f"Legacy detector frame {frame} reports effective diameter "
                    f"{effective_diameter:g}, but sequential replay requires "
                    f"{expected_diameter:g}"
                )
            first_raw_candidates, candidate_diameters = (
                _validated_candidate_diameters(first)
            )
            _validate_candidate_summary(
                first,
                candidate_diameters,
                expected_count=candidate_count,
                expected_median=candidate_median,
            )
            for item in frame_rows[1:]:
                if _dynamic_detector_scalar_signature(item) != signature:
                    raise StarryNiteLegacyExactError(
                        f"Legacy detector rows at frame {frame} disagree on dynamic "
                        "diameter, count, stage, static mode, or candidate summaries"
                    )
                raw_candidates = item.features.get(_CANDIDATE_DIAMETERS)
                if raw_candidates is first_raw_candidates:
                    # Detector rows intentionally share this immutable tuple;
                    # avoid rescanning O(candidate_count) values per nucleus.
                    continue
                _raw, row_candidates = _validated_candidate_diameters(item)
                if row_candidates != candidate_diameters:
                    raise StarryNiteLegacyExactError(
                        f"Legacy detector rows at frame {frame} disagree on candidate "
                        "diameters"
                    )
            previous_candidate_summary = (candidate_count, candidate_median)
        else:
            # No row exists on which to preserve this frame's raw candidate
            # vector. The live detector writes its count/median onto the next
            # emitted frame; defer validation until that handoff is observable.
            previous_candidate_summary = None
        previous_final_count = len(frame_rows)


def _row_index(detection: Detection) -> int:
    raw = detection.features.get("LEGACY_ROW_INDEX")
    if isinstance(raw, bool) or not isinstance(raw, Integral) or int(raw) < 0:
        raise StarryNiteLegacyExactError(
            f"Detection {detection.detection_id} has an invalid LEGACY_ROW_INDEX"
        )
    return int(raw)


def _dynamic_detector_scalar_signature(
    detection: Detection,
) -> tuple[float, int, int, bool, int, float | None, int, float | None]:
    features = detection.features
    diameter = features.get("LEGACY_EFFECTIVE_DIAMETER_XY_PX")
    if (
        isinstance(diameter, bool)
        or not isinstance(diameter, Real)
        or not math.isfinite(float(diameter))
        or float(diameter) <= 0
    ):
        raise StarryNiteLegacyExactError(
            f"Detection {detection.detection_id} lacks a valid effective diameter"
        )
    cell_count = _nonnegative_integer_feature(detection, "STARRYNITE_CELL_COUNT")
    stage_index = _nonnegative_integer_feature(detection, "STARRYNITE_STAGE_INDEX")
    static_mode = features.get(_STATIC_DIAMETER)
    if type(static_mode) is not bool:
        raise StarryNiteLegacyExactError(
            f"Detection {detection.detection_id} lacks a boolean {_STATIC_DIAMETER}"
        )
    candidate_count, candidate_median = _candidate_summary(
        detection,
        count_key=_CANDIDATE_COUNT,
        median_key=_CANDIDATE_MEDIAN,
    )
    previous_count, previous_median = _candidate_summary(
        detection,
        count_key=_PREVIOUS_CANDIDATE_COUNT,
        median_key=_PREVIOUS_CANDIDATE_MEDIAN,
    )
    return (
        float(diameter),
        cell_count,
        stage_index,
        static_mode,
        candidate_count,
        candidate_median,
        previous_count,
        previous_median,
    )


def _candidate_summary(
    detection: Detection,
    *,
    count_key: str,
    median_key: str,
) -> tuple[int, float | None]:
    count = _nonnegative_integer_feature(detection, count_key)
    raw_median = detection.features.get(median_key)
    if count == 0:
        if raw_median is not None:
            raise StarryNiteLegacyExactError(
                f"Detection {detection.detection_id} must report {median_key}=None "
                f"when {count_key}=0"
            )
        return 0, None
    if (
        isinstance(raw_median, bool)
        or not isinstance(raw_median, Real)
        or not math.isfinite(float(raw_median))
        or float(raw_median) <= 0
    ):
        raise StarryNiteLegacyExactError(
            f"Detection {detection.detection_id} lacks a valid {median_key}"
        )
    return count, float(raw_median)


def _validated_candidate_diameters(
    detection: Detection,
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    raw_candidates = detection.features.get(_CANDIDATE_DIAMETERS)
    if not isinstance(raw_candidates, tuple):
        raise StarryNiteLegacyExactError(
            f"Detection {detection.detection_id} lacks the candidate-diameter tuple"
        )
    candidates: list[float] = []
    for raw in raw_candidates:
        if (
            isinstance(raw, bool)
            or not isinstance(raw, Real)
            or not math.isfinite(float(raw))
            or float(raw) <= 0
        ):
            raise StarryNiteLegacyExactError(
                f"Detection {detection.detection_id} has an invalid candidate "
                "diameter"
            )
        candidates.append(float(raw))
    return raw_candidates, tuple(candidates)


def _validate_candidate_summary(
    detection: Detection,
    candidates: tuple[float, ...],
    *,
    expected_count: int,
    expected_median: float | None,
) -> None:
    if len(candidates) != expected_count:
        raise StarryNiteLegacyExactError(
            f"Detection {detection.detection_id} reports candidate count "
            f"{expected_count}, but its candidate tuple contains {len(candidates)}"
        )
    actual_median = None if not candidates else float(np.median(candidates))
    if actual_median is None:
        matches = expected_median is None
    else:
        matches = expected_median is not None and math.isclose(
            actual_median,
            expected_median,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    if not matches:
        raise StarryNiteLegacyExactError(
            f"Detection {detection.detection_id} reports candidate median "
            f"{expected_median!r}, but its candidate tuple requires "
            f"{actual_median!r}"
        )


def _diameter_from_candidate_summary(
    initial_diameter: float,
    summary: tuple[int, float | None],
    *,
    use_static_diameter: bool,
) -> float:
    count, median = summary
    if not use_static_diameter and count > 10:
        if median is None:
            raise StarryNiteLegacyExactError(
                "A dynamic detector summary with more than ten candidates must "
                "include a positive median"
            )
        return median
    return initial_diameter


def _nonnegative_integer_feature(detection: Detection, key: str) -> int:
    raw = detection.features.get(key)
    if isinstance(raw, bool) or not isinstance(raw, Integral) or int(raw) < 0:
        raise StarryNiteLegacyExactError(
            f"Detection {detection.detection_id} lacks a valid {key}"
        )
    return int(raw)


def _feature_path(detection: Detection, key: str) -> Path | None:
    value = detection.features.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    return Path(value).expanduser().resolve(strict=False)


def _validate_event_order(result: LegacyMovieDecisionResult) -> None:
    classifications = result.classifications
    events = result.events
    if tuple(item.sequence_index for item in classifications) != tuple(
        range(len(classifications))
    ):
        raise StarryNiteLegacyExactError(
            "Classifier sequence indices are not contiguous"
        )
    if tuple(item.event_index for item in events) != tuple(range(len(events))):
        raise StarryNiteLegacyExactError("Movie event indices are not contiguous")
    scan_order: list[tuple[int, int]] = []
    for event in events:
        start = event.classification_sequence_index
        stop = start + event.nested_classification_count + 1
        if start < 0 or stop > len(classifications):
            raise StarryNiteLegacyExactError(
                f"Movie event {event.event_index} has an invalid trace span"
            )
        group = classifications[start:stop]
        top = group[0]
        if (
            top.classifier_round != 1
            or top.top_level_event_index != event.event_index
            or top.parent_id != event.parent_id
            or top.daughter_ids != event.daughter_ids
        ):
            raise StarryNiteLegacyExactError(
                f"Movie event {event.event_index} disagrees with its top-level "
                "classification record"
            )
        if any(
            item.top_level_event_index != event.event_index for item in group
        ):
            raise StarryNiteLegacyExactError(
                f"Movie event {event.event_index} contains a foreign classifier record"
            )
        if any(item.classifier_round != 2 for item in group[1:]):
            raise StarryNiteLegacyExactError(
                f"Movie event {event.event_index} has a non-inline nested record"
            )
        scan_order.append((top.frame, top.matlab_row))
    if scan_order != sorted(scan_order) or len(scan_order) != len(set(scan_order)):
        raise StarryNiteLegacyExactError(
            "Top-level classifier events do not follow unique MATLAB frame/row order"
        )
    covered = sum(1 + item.nested_classification_count for item in events)
    if covered != len(classifications):
        raise StarryNiteLegacyExactError(
            "Classifier records are not completely covered by movie events"
        )


def _provenance(
    profile: StarryNiteTuningProfile,
    values: Mapping[str, Any],
    prepared: PreparedLegacyRuntime,
    stages: Sequence[LegacyEarlyStageSummary],
    decision: LegacyMovieDecisionResult,
    *,
    retained_count: int,
    rejected_count: int,
) -> dict[str, Any]:
    effective_counts = Counter(
        str(item.effective_class) for item in decision.classifications
    )
    computed_counts = Counter(
        str(item.computed_class) for item in decision.classifications
    )
    round_counts = Counter(
        str(item.classifier_round) for item in decision.classifications
    )
    trace = [
        {
            "sequence": item.sequence_index,
            "event": item.top_level_event_index,
            "frame": item.frame,
            "row": item.matlab_row,
            "round": item.classifier_round,
            "depth": item.recursion_depth,
            "parent": item.parent_id,
            "daughters": list(item.daughter_ids),
            "effective_class": item.effective_class,
            "computed_class": item.computed_class,
            "force_mode": item.force_mode,
        }
        for item in decision.classifications
    ]
    trace_bytes = json.dumps(
        trace,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    distribution_path = profile.detector_settings.get(
        "STARRYNITE_DISTRIBUTION_FILE"
    )
    distribution_hash = None
    if distribution_path and Path(str(distribution_path)).is_file():
        distribution_hash = sha256_file(str(distribution_path))
    return {
        "backend": LEGACY_EXACT_REFINEMENT_BACKEND,
        "boundary": "detections_through_legacy_classifier_movie",
        "implementation": "StarryNiteLegacyExactTracker",
        "parameter_file": values[_PARAMETER_FILE],
        "parameter_sha256": profile.parameter_sha256,
        "model_file": values[_MODEL_FILE],
        "model_sha256": prepared.model.sha256,
        "neutral_classifier_file": values[_CLASSIFIER_FILE],
        "neutral_classifier_sha256": values[_CLASSIFIER_SHA256],
        "classifier_source_model_sha256": decision.source_model_sha256,
        "classifier_family": decision.classifier_family,
        "classifier_mode": decision.classifier_mode,
        "distribution_file": (
            None if distribution_path is None else str(distribution_path)
        ),
        "distribution_source_sha256": profile.detector_settings.get(
            _DISTRIBUTION_SOURCE_SHA256
        ),
        "distribution_sha256": distribution_hash,
        "exact_detector_tail_required": values[_REQUIRE_EXACT_TAIL],
        "nondivision_cost_function": prepared.nondivision_cost_function,
        "division_cost_function": prepared.division_cost_function,
        "retained_detection_count": retained_count,
        "rejected_detection_count": rejected_count,
        "early_stages": _stage_summaries(stages),
        "event_order_validated": True,
        "event_order_validation_scope": (
            "structural_sequence_trace_coverage_and_unique_frame_row_order"
        ),
        "event_omission_completeness_proven": False,
        "top_level_event_count": len(decision.events),
        "classification_count": len(decision.classifications),
        "classification_round_counts": dict(sorted(round_counts.items())),
        "effective_class_counts": dict(sorted(effective_counts.items())),
        "computed_class_counts": dict(sorted(computed_counts.items())),
        "classification_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "first_classification": None if not trace else trace[0],
        "last_classification": None if not trace else trace[-1],
    }


def _stage_summaries(
    stages: Sequence[LegacyEarlyStageSummary],
) -> tuple[dict[str, Any], ...]:
    return tuple(stage.as_provenance() for stage in stages)


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        _raise_cancelled()


def _raise_cancelled() -> None:
    # Imported lazily to avoid a registry -> tracker -> pipeline cycle.
    from ..pipeline import TrackingCancelled

    raise TrackingCancelled("Tracking was cancelled")


__all__ = [
    "STARRYNITE_LEGACY_EXACT_TRACKER_ID",
    "StarryNiteLegacyExactError",
    "StarryNiteLegacyExactTracker",
]
