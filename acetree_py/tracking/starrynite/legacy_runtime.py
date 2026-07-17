"""Shared preparation for the executable legacy whole-movie boundary.

The compatibility report and the runtime must interpret the parameter script
the same way.  In particular, assignments after the active ``load`` statement
override fields from the MAT file, while earlier assignments are replaced by
that load.  Only values accepted by the safe parameter parser are applied;
function handles are recognized from a small, explicit static dispatch table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, MutableMapping

from .legacy_early import (
    LegacyEarlyTrackingParameters,
)
from .legacy_features import LegacyTrackingStatistics
from .legacy_state import LegacyFeatureParameters
from .models import StarryNiteModel, load_matlab_model
from .parameters import ParameterRecord, normalize_parameter_name
from .presets import StarryNiteTuningProfile


_COST_ASSIGNMENT = re.compile(
    r"^trackingparameters\.(?P<field>nonDivCostFunction|DivCostFunction)"
    r"\s*=\s*@(?P<function>"
    r"distanceCostFunction|nondivScoreModelCostFunction|"
    r"divDistanceCostFunction|divScoreModelCostFunction)\s*;?$",
    re.ASCII,
)


class LegacyRuntimePreparationError(ValueError):
    """Raised when source-bound legacy runtime inputs cannot be prepared."""


@dataclass(frozen=True, slots=True)
class PreparedLegacyRuntime:
    """Numeric state shared by early geometry and classifier refinement."""

    model: StarryNiteModel
    tracking_parameters: Mapping[str, Any]
    statistics: LegacyTrackingStatistics
    feature_parameters: LegacyFeatureParameters
    early_parameters: LegacyEarlyTrackingParameters
    nondivision_cost_function: str | None
    division_cost_function: str | None


def supported_static_cost_assignment(source: str) -> bool:
    """Return whether one opaque record is an implemented static dispatch."""

    return _COST_ASSIGNMENT.fullmatch(source.strip()) is not None


def prepare_legacy_runtime(
    profile: StarryNiteTuningProfile,
    *,
    start_frame: int = 1,
    end_frame: int | None = None,
) -> PreparedLegacyRuntime:
    """Load and validate all inert numeric state for an exact movie run."""

    if not isinstance(profile, StarryNiteTuningProfile):
        raise TypeError("profile must be a StarryNiteTuningProfile")
    if profile.model_path is None:
        raise LegacyRuntimePreparationError(
            "The parameter file does not identify one tracking MAT file"
        )
    if profile.xy_um is None or profile.z_um is None:
        raise LegacyRuntimePreparationError(
            "Exact tracking requires both xyres and zres"
        )
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 1:
        raise TypeError("start_frame must be a positive integer")
    if end_frame is not None and (
        isinstance(end_frame, bool) or not isinstance(end_frame, int) or end_frame < 1
    ):
        raise TypeError("end_frame must be a positive integer or None")

    model = load_matlab_model(profile.model_path)
    parameters = overlay_tracking_parameters(model, profile)
    nondivision, division = static_cost_functions(profile)
    anisotropy = (1.0, 1.0, float(profile.z_um) / float(profile.xy_um))
    parameters = dict(parameters)
    parameters[_matching_key(parameters, "starttime")] = start_frame
    parameters[_matching_key(parameters, "anisotropyvector")] = anisotropy
    if end_frame is not None:
        parameters[_matching_key(parameters, "endtime")] = end_frame
    runtime_source = {"trackingparameters": parameters}
    statistics = LegacyTrackingStatistics.from_model(runtime_source)
    feature_parameters = LegacyFeatureParameters.from_model(
        runtime_source,
        anisotropy_xyz=anisotropy,
        end_frame=end_frame,
    )
    early_parameters = LegacyEarlyTrackingParameters.from_model(
        runtime_source,
        end_frame=end_frame,
        nondivision_cost=nondivision,
        division_cost=division,
    )
    return PreparedLegacyRuntime(
        model=model,
        tracking_parameters=parameters,
        statistics=statistics,
        feature_parameters=feature_parameters,
        early_parameters=early_parameters,
        nondivision_cost_function=nondivision,
        division_cost_function=division,
    )


def overlay_tracking_parameters(
    model: StarryNiteModel,
    profile: StarryNiteTuningProfile,
) -> Mapping[str, Any]:
    """Apply safe post-load script assignments to decoded MAT parameters."""

    if not isinstance(model, StarryNiteModel):
        raise TypeError("model must be a StarryNiteModel")
    source = model.fields.get("trackingparameters")
    if not isinstance(source, Mapping):
        raise LegacyRuntimePreparationError(
            "The MAT file has no decoded trackingparameters struct"
        )
    result = _mutable_copy(source)
    load_index = _active_load_index(profile)
    for index, record in enumerate(profile.parameters.records):
        if index <= load_index or record.kind != "assignment" or record.name is None:
            continue
        normalized = normalize_parameter_name(record.name)
        if not normalized.startswith("trackingparameters."):
            continue
        segments = record.name.split(".")[1:]
        if not segments:
            continue
        _assign_nested(result, segments, record.value)
    return result


def static_cost_functions(
    profile: StarryNiteTuningProfile,
) -> tuple[str | None, str | None]:
    """Resolve implemented function handles using source-order semantics."""

    load_index = _active_load_index(profile)
    resolved: dict[str, str] = {}
    for index, record in enumerate(profile.parameters.records):
        if index <= load_index or record.kind != "opaque":
            continue
        match = _COST_ASSIGNMENT.fullmatch(record.source.strip())
        if match is not None:
            resolved[match.group("field")] = match.group("function")
    return (
        resolved.get("nonDivCostFunction"),
        resolved.get("DivCostFunction"),
    )


def _active_load_index(profile: StarryNiteTuningProfile) -> int:
    load_references = tuple(
        reference
        for reference in profile.parameters.model_references
        if reference.source_kind == "load"
    )
    active = load_references or profile.parameters.model_references
    if len(active) != 1:
        return -1
    return active[0].record_index if active[0].source_kind == "load" else -1


def _mutable_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _mutable_copy(member) if isinstance(member, Mapping) else member
        for key, member in value.items()
    }


def _matching_key(values: Mapping[str, Any], requested: str) -> str:
    lowered = requested.lower()
    matches = [str(key) for key in values if str(key).lower() == lowered]
    if len(matches) > 1:
        raise LegacyRuntimePreparationError(
            f"Ambiguous case-insensitive tracking parameter field {requested!r}"
        )
    return matches[0] if matches else requested


def _assign_nested(
    values: MutableMapping[str, Any],
    segments: list[str],
    value: Any,
) -> None:
    current = values
    for segment in segments[:-1]:
        key = _matching_key(current, segment)
        child = current.get(key)
        if child is None:
            replacement: dict[str, Any] = {}
        elif isinstance(child, Mapping):
            replacement = _mutable_copy(child)
        else:
            raise LegacyRuntimePreparationError(
                "Cannot apply nested assignment below non-struct field "
                f"{segment!r}"
            )
        current[key] = replacement
        current = replacement
    final_key = _matching_key(current, segments[-1])
    current[final_key] = _python_parameter_value(value)


def _python_parameter_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_python_parameter_value(item) for item in value)
    if isinstance(value, (str, bool, Real)) or value is None:
        return value
    raise LegacyRuntimePreparationError(
        f"Unsupported safe tracking parameter value {type(value).__name__}"
    )


__all__ = [
    "LegacyRuntimePreparationError",
    "PreparedLegacyRuntime",
    "overlay_tracking_parameters",
    "prepare_legacy_runtime",
    "static_cost_functions",
    "supported_static_cost_assignment",
]
