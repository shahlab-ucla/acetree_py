"""Translate legacy StarryNite parameter files into native tracking settings.

This adapter is deliberately conservative.  It maps only parameters whose
meaning has a direct native equivalent and keeps the source/model identities
as provenance.  Unknown MATLAB statements remain in :class:`StarryNiteParameters`
and are never executed.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .models import sha256_file
from .parameters import ParameterValue, StarryNiteParameters, read_parameter_file


class StarryNitePresetError(ValueError):
    """Raised when a recognized legacy value cannot be translated safely."""


@dataclass(frozen=True, slots=True)
class StarryNiteTuningProfile:
    """Native starting settings and provenance derived from one parameter file."""

    parameters: StarryNiteParameters
    detector_settings: Mapping[str, Any]
    tracker_settings: Mapping[str, Any]
    stage_index: int
    cell_count: int
    xy_um: float | None = None
    z_um: float | None = None
    model_path: Path | None = None
    parameter_sha256: str | None = None
    model_sha256: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "detector_settings",
            MappingProxyType(dict(self.detector_settings)),
        )
        object.__setattr__(
            self,
            "tracker_settings",
            MappingProxyType(dict(self.tracker_settings)),
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path))


@dataclass(frozen=True, slots=True)
class StarryNiteTuningSavePlan:
    """Lossless legacy overrides produced by the basic tuning controls."""

    overrides: Mapping[str, ParameterValue]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))


def legacy_stage_index(parameters: StarryNiteParameters, cell_count: int) -> int:
    """Return StarryNite's zero-based stage for a detected-cell count.

    The MATLAB implementation advances only after ``cell_count`` is strictly
    greater than a staging boundary.  ``bisect_left`` reproduces that detail.
    """

    if (
        isinstance(cell_count, bool)
        or not isinstance(cell_count, Integral)
        or cell_count < 0
    ):
        raise StarryNitePresetError("cell_count must be a non-negative integer")
    staging = _lookup(parameters, "parameters.staging")
    if staging is None:
        return 0
    boundaries = _numeric_vector("parameters.staging", staging)
    if any(next_value < value for value, next_value in zip(boundaries, boundaries[1:])):
        raise StarryNitePresetError("parameters.staging must be sorted")
    return bisect.bisect_left(boundaries, int(cell_count))


def select_stage_value(
    name: str,
    value: ParameterValue,
    stage_index: int,
) -> ParameterValue:
    """Apply StarryNite's scalar-or-stage-vector parameter semantics."""

    if not isinstance(value, tuple):
        return value
    if not value:
        raise StarryNitePresetError(f"{name} cannot be an empty stage vector")
    if len(value) == 1:
        # MATLAB getParameter.m treats every length-one array as a scalar,
        # regardless of the current stage.
        return value[0]
    if stage_index >= len(value):
        raise StarryNitePresetError(
            f"{name} has {len(value)} stage value(s), but stage {stage_index + 1} "
            "was requested"
        )
    return value[stage_index]


def tuning_profile_from_parameters(
    parameters: StarryNiteParameters,
    *,
    cell_count: int | None = None,
    fallback_radius_um: float = 4.0,
) -> StarryNiteTuningProfile:
    """Build a native detector/tracker starting point from parsed parameters."""

    if not math.isfinite(float(fallback_radius_um)) or fallback_radius_um <= 0:
        raise StarryNitePresetError("fallback_radius_um must be positive and finite")
    assumed_initial_cell_count = False
    if cell_count is None:
        initial_count = _optional_number(parameters, "firsttimestepnumcells")
        if initial_count is not None and not initial_count.is_integer():
            raise StarryNitePresetError("firsttimestepnumcells must be an integer")
        assumed_initial_cell_count = initial_count is None
        cell_count = 0 if initial_count is None else int(initial_count)
    stage_index = legacy_stage_index(parameters, cell_count)
    source_path = parameters.source_path
    base_directory = source_path.parent if source_path is not None else Path.cwd()
    source_text = str(source_path.resolve(strict=False)) if source_path else ""
    parameter_digest = (
        sha256_file(source_path) if source_path is not None and source_path.is_file() else None
    )

    warnings: list[str] = []
    if assumed_initial_cell_count:
        warnings.append(
            "firsttimestepnumcells is missing; stage 1 was selected from an "
            "assumed starting cell count of 0"
        )
    detector: dict[str, Any] = {
        "STARRYNITE_CELL_COUNT": int(cell_count),
        "STARRYNITE_STAGE_INDEX": stage_index,
        "STARRYNITE_PARAMETER_FILE": source_text,
        "STARRYNITE_PARAMETER_SHA256": parameter_digest or "",
        # A zero direct threshold asks the native detector to use the staged,
        # robust intensity threshold below.
        "THRESHOLD": 0.0,
        # Legacy processVolume exports the integer ray-recentered point by
        # default.  Broad support-centroid refinement remains an explicit
        # native opt-in because it collapses close daughter coordinates.
        "DO_SUBPIXEL_LOCALIZATION": False,
    }
    tracker: dict[str, Any] = {
        "STARRYNITE_PARAMETER_FILE": source_text,
        "STARRYNITE_PARAMETER_SHA256": parameter_digest or "",
    }

    xy_um = _optional_number(parameters, "xyres")
    z_um = _optional_number(parameters, "zres")
    diameter_px = _optional_number(parameters, "firsttimestepdiam")
    radius_um = fallback_radius_um
    if diameter_px is not None:
        if xy_um is None:
            warnings.append(
                "firsttimestepdiam was present without xyres; the current dataset radius was kept"
            )
        else:
            radius_um = diameter_px * xy_um / 2.0
    detector["RADIUS"] = radius_um

    detector_mappings = {
        "parameters.sigma": "SIGMA",
        "parameters.intensitythreshold": "INTENSITY_THRESHOLD",
        "parameters.boundary_percent": "BOUNDARY_PERCENT",
        "parameters.large_ray_threshold": "LARGE_RAY_THRESHOLD",
        "parameters.small_ray_threshold": "SMALL_RAY_THRESHOLD",
        "parameters.nndist_merge": "NNDIST_MERGE",
        "parameters.armerge": "AR_MERGE",
        "parameters.rangethreshold": "RANGE_THRESHOLD",
        "parameters.mergelower": "MERGE_LOWER",
        "parameters.mergesplit": "MERGE_SPLIT",
        "parameters.split": "SPLIT_THRESHOLD",
    }
    for legacy_name, native_name in detector_mappings.items():
        value = _lookup(parameters, legacy_name)
        if value is None:
            continue
        selected = select_stage_value(legacy_name, value, stage_index)
        detector[native_name] = _number(legacy_name, selected)
    # The exact detector tail consumes the second distribution file when the
    # script names one; otherwise it consumes the primary file.  Preserve that
    # selection even when the referenced file is absent.  Falling back from a
    # missing distribution_file2 to distribution_file would silently change
    # the detector model and is therefore forbidden.
    normalized = parameters.normalized_settings
    distribution_name = (
        "distribution_file2"
        if "distribution_file2" in normalized
        else "distribution_file"
    )
    distribution_value = normalized.get(distribution_name)
    if distribution_value is not None:
        if not isinstance(distribution_value, str) or not distribution_value.strip():
            warnings.append(
                f"{distribution_name} was not a non-empty path; the exact "
                "disk-distribution detector tail remains disabled"
            )
        else:
            distribution_path = Path(distribution_value)
            if not distribution_path.is_absolute():
                distribution_path = base_directory / distribution_path
            distribution_path = distribution_path.resolve(strict=False)
            detector["STARRYNITE_DISTRIBUTION_FILE"] = str(distribution_path)
            if distribution_path.is_file():
                detector["STARRYNITE_DISTRIBUTION_SOURCE_SHA256"] = sha256_file(
                    distribution_path
                )
            else:
                warnings.append(
                    f"Referenced detector distribution file was not found: "
                    f"{distribution_path}. No alternate file was substituted."
                )

    tracker_mappings = {
        "trackingparameters.candidate_cutoff": ("CANDIDATE_CUTOFF", float),
        "trackingparameters.safefactor": ("SAFE_FACTOR", float),
        "trackingparameters.nnnumber": ("NN_NUMBER", int),
        "trackingparameters.forwardnnnumber": ("FORWARD_NN_NUMBER", int),
        "trackingparameters.temporalcutoff": ("MAX_FRAME_GAP", int),
    }
    for legacy_name, (native_name, converter) in tracker_mappings.items():
        value = _lookup(parameters, legacy_name)
        if value is None:
            continue
        number = _number(legacy_name, select_stage_value(legacy_name, value, stage_index))
        if converter is int and not number.is_integer():
            raise StarryNitePresetError(f"{legacy_name} must be an integer")
        tracker[native_name] = converter(number)
    if int(tracker.get("MAX_FRAME_GAP", 1)) > 1:
        tracker["ALLOW_GAP_CLOSING"] = True

    # Explicit ``load`` statements identify the active tracking classifier in
    # standard files.  Other model-like paths often point to detector
    # distribution tables and must not displace that classifier.
    model_references = tuple(
        reference
        for reference in parameters.model_references
        if reference.source_kind == "load"
    ) or parameters.model_references
    resolved_models = tuple(
        reference.resolve(base_directory) for reference in model_references
    )
    model_path = resolved_models[0] if resolved_models else None
    model_digest = None
    if model_path is not None:
        tracker["STARRYNITE_MODEL_FILE"] = str(model_path)
        if model_path.is_file():
            model_digest = sha256_file(model_path)
            tracker["STARRYNITE_MODEL_SHA256"] = model_digest
        else:
            warnings.append(f"Referenced tracking model was not found: {model_path}")
        warnings.append(
            "The MATLAB classifier is retained for provenance and can be exported to "
            "the validated neutral runtime with exact legacy feature extraction; "
            "the registered native-fast tracker intentionally continues to use its "
            "explicit geometry scorer"
        )
    if len(resolved_models) > 1:
        warnings.append(
            "Multiple active model references were found; the first is retained as the preset model"
        )
    if parameters.opaque_records:
        warnings.append(
            f"{len(parameters.opaque_records)} unsupported MATLAB statement(s) were preserved "
            "but not executed"
        )

    return StarryNiteTuningProfile(
        parameters=parameters,
        detector_settings=detector,
        tracker_settings=tracker,
        stage_index=stage_index,
        cell_count=int(cell_count),
        xy_um=xy_um,
        z_um=z_um,
        model_path=model_path,
        parameter_sha256=parameter_digest,
        model_sha256=model_digest,
        warnings=tuple(warnings),
    )


def load_tuning_profile(
    path: str | Path,
    *,
    cell_count: int | None = None,
    fallback_radius_um: float = 4.0,
    encoding: str = "utf-8-sig",
) -> StarryNiteTuningProfile:
    """Read a standard StarryNite file and build a tunable native profile."""

    parameters = read_parameter_file(path, encoding=encoding)
    return tuning_profile_from_parameters(
        parameters,
        cell_count=cell_count,
        fallback_radius_um=fallback_radius_um,
    )


def build_tuning_save_plan(
    profile: StarryNiteTuningProfile,
    *,
    radius_um: float,
    intensity_threshold: float,
    max_frame_gap: int,
) -> StarryNiteTuningSavePlan:
    """Map editable native controls back to compatible legacy assignments.

    Stage vectors are retained and only the active stage is replaced. Settings
    without an exact legacy-unit equivalent, such as the native ROI radius and
    ambiguity ratio, deliberately remain native UI preferences.
    """

    if not isinstance(profile, StarryNiteTuningProfile):
        raise TypeError("profile must be a StarryNiteTuningProfile")
    radius = float(radius_um)
    threshold = float(intensity_threshold)
    if not math.isfinite(radius) or radius <= 0:
        raise StarryNitePresetError("radius_um must be positive and finite")
    if not math.isfinite(threshold) or threshold < 0:
        raise StarryNitePresetError(
            "intensity_threshold must be non-negative and finite"
        )
    if isinstance(max_frame_gap, bool) or int(max_frame_gap) != max_frame_gap:
        raise StarryNitePresetError("max_frame_gap must be an integer")
    frame_gap = int(max_frame_gap)
    if frame_gap < 1:
        raise StarryNitePresetError("max_frame_gap must be at least 1")

    overrides: dict[str, ParameterValue] = {
        "parameters.intensitythreshold": _replace_active_stage(
            profile.parameters,
            "parameters.intensitythreshold",
            threshold,
            profile.stage_index,
        ),
        "trackingparameters.temporalcutoff": _replace_active_stage(
            profile.parameters,
            "trackingparameters.temporalcutoff",
            frame_gap,
            profile.stage_index,
        ),
    }
    warnings: list[str] = []
    if profile.xy_um is None:
        warnings.append(
            "Expected radius was not written because the source file has no xyres; "
            "the original diameter was preserved"
        )
    else:
        overrides["firsttimestepdiam"] = radius * 2.0 / profile.xy_um
    return StarryNiteTuningSavePlan(overrides, tuple(warnings))


def _replace_active_stage(
    parameters: StarryNiteParameters,
    name: str,
    value: ParameterValue,
    stage_index: int,
) -> ParameterValue:
    original = _lookup(parameters, name)
    if not isinstance(original, tuple):
        return value
    if not original:
        raise StarryNitePresetError(f"{name} cannot be an empty stage vector")
    if stage_index >= len(original):
        raise StarryNitePresetError(
            f"{name} has {len(original)} stage value(s), but stage "
            f"{stage_index + 1} was requested"
        )
    return tuple(
        value if index == stage_index else item
        for index, item in enumerate(original)
    )


def _lookup(parameters: StarryNiteParameters, name: str) -> ParameterValue | None:
    return parameters.normalized_settings.get(name)


def _number(name: str, value: ParameterValue) -> float:
    if isinstance(value, (bool, str, tuple)):
        raise StarryNitePresetError(f"{name} must resolve to one numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise StarryNitePresetError(f"{name} must be finite")
    return number


def _optional_number(parameters: StarryNiteParameters, name: str) -> float | None:
    value = _lookup(parameters, name)
    return None if value is None else _number(name, value)


def _numeric_vector(name: str, value: ParameterValue) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        return (_number(name, value),)
    return tuple(_number(name, item) for item in value)


__all__ = [
    "StarryNitePresetError",
    "StarryNiteTuningProfile",
    "StarryNiteTuningSavePlan",
    "build_tuning_save_plan",
    "legacy_stage_index",
    "load_tuning_profile",
    "select_stage_value",
    "tuning_profile_from_parameters",
]
