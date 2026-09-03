"""Actionable StarryNite parameter/model compatibility reporting.

The native tracker deliberately does not execute a MATLAB classifier.  This
module makes that boundary machine-readable: it inventories a tuning profile,
validates an optional neutral classifier against the referenced MAT-file hash,
and decides whether a requested backend may run.  In particular, exact mode is
never selected by fallback; every prerequisite must be positively validated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .classifier import (
    NEUTRAL_AMBIGIOUS_CLASSIFIER_SCHEMA,
    NEUTRAL_CLASSIFIER_SCHEMA,
    NeutralClassifierFormatError,
    load_neutral_ambigious_classifier,
    load_neutral_classifier,
)
from .legacy_early import LegacyEarlyTrackingError
from .legacy_detector_tail import (
    LegacyDetectorTailError,
    load_legacy_disk_distributions,
)
from .legacy_features import LegacyFeatureExtractionError
from .legacy_runtime import (
    LegacyRuntimePreparationError,
    prepare_legacy_runtime,
    supported_static_cost_assignment,
)
from .legacy_state import LegacyStateError
from .models import MatlabModelLoadError, sha256_file
from .parameters import ParameterValue, normalize_parameter_name
from .parameter_view import build_legacy_region_table
from .presets import StarryNiteTuningProfile


NATIVE_FAST_BACKEND = "native_fast"
LEGACY_EXACT_REFINEMENT_BACKEND = "legacy_exact_refinement"
_SEVERITIES = frozenset({"info", "warning", "blocker"})

# These are the staged fields read unconditionally by the standard detector
# route in the pinned StarryNite sources (``processVolume``,
# ``pickCenterIndicies``, ``vcalculateMaximalRange``,
# ``calculateSphereDiameters_geometric``, and ``resolveConflicts``).  The six
# ``a_*`` fields belong only to the optional DLSM adaptive filter and are not
# required for the published example-file pipeline.
_REQUIRED_LEGACY_DETECTOR_FIELDS = (
    "staging",
    "sigma",
    "intensitythreshold",
    "rangethreshold",
    "boundary_percent",
    "large_ray_threshold",
    "small_ray_threshold",
    "mergelower",
    "mergesplit",
    "split",
    "nndist_merge",
    "armerge",
)
class StarryNiteCompatibilityError(ValueError):
    """Raised when a requested compatibility backend is not validated."""


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One stable, user-actionable compatibility finding."""

    code: str
    severity: str
    message: str
    record_index: int | None = None

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("Compatibility issue code cannot be empty")
        if self.severity not in _SEVERITIES:
            raise ValueError(f"Unsupported compatibility severity: {self.severity!r}")
        if not self.message.strip():
            raise ValueError("Compatibility issue message cannot be empty")
        if self.record_index is not None and self.record_index < 0:
            raise ValueError("record_index cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.record_index is not None:
            value["record_index"] = self.record_index
        return value


@dataclass(frozen=True, slots=True)
class EffectiveParameter:
    """One effective last-assignment-wins value and its source record."""

    normalized_name: str
    original_name: str
    value: ParameterValue
    record_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_name": self.normalized_name,
            "original_name": self.original_name,
            "value": self.value,
            "record_index": self.record_index,
        }


@dataclass(frozen=True, slots=True)
class ModelReferenceStatus:
    """Resolved state of one model reference in the parameter source."""

    raw_path: str
    resolved_path: Path
    source_kind: str
    record_index: int
    exists: bool
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_path", Path(self.resolved_path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_path": self.raw_path,
            "resolved_path": str(self.resolved_path),
            "source_kind": self.source_kind,
            "record_index": self.record_index,
            "exists": self.exists,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class BackendReadiness:
    """Positive validation result for one tracking behavior."""

    backend_id: str
    display_name: str
    issues: tuple[CompatibilityIssue, ...] = ()

    def __post_init__(self) -> None:
        if not self.backend_id.strip() or not self.display_name.strip():
            raise ValueError("Backend identifiers and labels cannot be empty")
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def runnable(self) -> bool:
        return not any(issue.severity == "blocker" for issue in self.issues)

    @property
    def blockers(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "blocker")

    def require(self) -> None:
        """Fail closed with all actionable blockers when this backend cannot run."""

        if self.runnable:
            return
        details = "; ".join(issue.message for issue in self.blockers)
        raise StarryNiteCompatibilityError(
            f"{self.display_name} is not available: {details}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "display_name": self.display_name,
            "runnable": self.runnable,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class StarryNiteCompatibilityReport:
    """Complete compatibility inventory for one loaded tuning profile."""

    syntax: str
    encoding: str
    parameter_path: Path | None
    parameter_sha256: str | None
    stage_index: int
    cell_count: int
    effective_parameters: tuple[EffectiveParameter, ...]
    opaque_issues: tuple[CompatibilityIssue, ...]
    model_references: tuple[ModelReferenceStatus, ...]
    native_overrides: Mapping[str, Mapping[str, Any]]
    backends: Mapping[str, BackendReadiness]
    regional_definition_count: int = 0
    regional_consumed_record_indices: tuple[int, ...] = ()
    neutral_classifier_path: Path | None = None
    neutral_classifier_schema: str | None = None
    neutral_classifier_family: str | None = None
    neutral_source_model_sha256: str | None = None
    neutral_classifier_source_bound: bool = False
    current_parameter_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.stage_index < 0 or self.cell_count < 0:
            raise ValueError("Stage index and cell count cannot be negative")
        object.__setattr__(
            self,
            "parameter_path",
            None if self.parameter_path is None else Path(self.parameter_path),
        )
        object.__setattr__(
            self,
            "neutral_classifier_path",
            (
                None
                if self.neutral_classifier_path is None
                else Path(self.neutral_classifier_path)
            ),
        )
        if self.neutral_classifier_source_bound and (
            self.neutral_classifier_path is None
            or self.neutral_source_model_sha256 is None
        ):
            raise ValueError(
                "A source-bound neutral classifier needs a path and source hash"
            )
        frozen_overrides = {
            str(component): MappingProxyType(dict(settings))
            for component, settings in self.native_overrides.items()
        }
        object.__setattr__(
            self,
            "native_overrides",
            MappingProxyType(frozen_overrides),
        )
        object.__setattr__(self, "backends", MappingProxyType(dict(self.backends)))
        if self.regional_definition_count < 0:
            raise ValueError("regional_definition_count cannot be negative")
        object.__setattr__(
            self,
            "regional_consumed_record_indices",
            tuple(int(item) for item in self.regional_consumed_record_indices),
        )

    def backend(self, backend_id: str) -> BackendReadiness:
        try:
            return self.backends[str(backend_id)]
        except KeyError as exc:
            raise StarryNiteCompatibilityError(
                f"Unknown StarryNite backend: {backend_id!r}"
            ) from exc

    def select_backend(self, backend_id: str) -> str:
        """Return a validated backend ID; never silently fall back to native."""

        readiness = self.backend(backend_id)
        readiness.require()
        return readiness.backend_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acetree.starrynite-compatibility-report",
            "version": 1,
            "syntax": self.syntax,
            "encoding": self.encoding,
            "parameter_path": (
                None if self.parameter_path is None else str(self.parameter_path)
            ),
            "parameter_sha256": self.parameter_sha256,
            "current_parameter_sha256": self.current_parameter_sha256,
            "stage_index": self.stage_index,
            "cell_count": self.cell_count,
            "effective_parameters": [
                item.to_dict() for item in self.effective_parameters
            ],
            "regional_parameters": {
                "definition_count": self.regional_definition_count,
                "consumed_record_indices": list(
                    self.regional_consumed_record_indices
                ),
            },
            "opaque_issues": [item.to_dict() for item in self.opaque_issues],
            "model_references": [item.to_dict() for item in self.model_references],
            "native_overrides": {
                component: dict(settings)
                for component, settings in self.native_overrides.items()
            },
            "neutral_classifier": {
                "path": (
                    None
                    if self.neutral_classifier_path is None
                    else str(self.neutral_classifier_path)
                ),
                "schema": self.neutral_classifier_schema,
                "classifier_family": self.neutral_classifier_family,
                "source_model_sha256": self.neutral_source_model_sha256,
                "source_bound": self.neutral_classifier_source_bound,
            },
            "backends": {
                name: backend.to_dict() for name, backend in self.backends.items()
            },
        }

    def format_text(self) -> str:
        """Render a compact report suitable for a non-technical details dialog."""

        source = "in-memory parameters" if self.parameter_path is None else str(
            self.parameter_path
        )
        lines = [
            f"Parameter file: {source}",
            f"Syntax: {self.syntax}; decoded as {self.encoding}",
            f"Selected stage: {self.stage_index + 1} ({self.cell_count} cells)",
            f"Recognized effective settings: {len(self.effective_parameters)}",
            f"Recognized regional stages: {self.regional_definition_count}",
            f"Unsupported statements: {len(self.opaque_issues)}",
        ]
        if not self.model_references:
            lines.append("Tracking model: none referenced")
        else:
            for reference in self.model_references:
                state = "found" if reference.exists else "missing"
                lines.append(f"Tracking model ({state}): {reference.resolved_path}")
        if self.neutral_classifier_path is not None:
            binding = (
                "source-bound"
                if self.neutral_classifier_source_bound
                else "not proven source-bound"
            )
            lines.append(
                f"Neutral classifier ({binding}): {self.neutral_classifier_path}"
            )
        lines.append("")
        lines.append("Available tracking behaviors:")
        for backend in self.backends.values():
            state = "ready" if backend.runnable else "not available"
            lines.append(f"- {backend.display_name}: {state}")
            for issue in backend.issues:
                lines.append(f"  - {issue.message}")
        return "\n".join(lines)


def build_compatibility_report(
    profile: StarryNiteTuningProfile,
    *,
    neutral_classifier_path: str | Path | None = None,
    runtime_capabilities: Iterable[str] = (),
    encoding: str = "utf-8-sig",
) -> StarryNiteCompatibilityReport:
    """Inventory one profile and positively validate available backends.

    ``legacy_exact_refinement`` is ready only when the caller advertises that
    runtime capability *and* the parameter source, referenced MAT model, and
    source-bound neutral export all validate.  Merely finding a ``.mat`` file is
    never treated as permission or ability to execute it.
    """

    if not isinstance(profile, StarryNiteTuningProfile):
        raise TypeError("profile must be a StarryNiteTuningProfile")
    parameters = profile.parameters
    effective = _effective_parameters(profile)
    regional_table = build_legacy_region_table(parameters, strict=False)
    regional_consumed = set(regional_table.consumed_record_indices)
    regional_issue_indices = {issue.record_index for issue in regional_table.issues}
    opaque_items = [
        CompatibilityIssue(
            code="opaque_parameter_statement",
            severity="warning",
            message=(
                f"Record {index + 1} was preserved but not interpreted"
                + (f": {record.reason}" if record.reason else "")
            ),
            record_index=index,
        )
        for index, record in enumerate(parameters.records)
        if record.kind == "opaque"
        and not _is_supported_exact_static_dispatch(record.source)
        and index not in regional_consumed
        and index not in regional_issue_indices
    ]
    opaque_items.extend(
        CompatibilityIssue(
            code="regional_parameter_unresolved",
            severity="warning",
            message=(
                f"Record {issue.record_index + 1} could not be represented as an "
                f"exact regional setting: {issue.message}"
            ),
            record_index=issue.record_index,
        )
        for issue in regional_table.issues
    )
    opaque = tuple(sorted(opaque_items, key=lambda item: item.record_index or -1))
    references = _model_reference_statuses(profile)
    capabilities = {str(item).strip().lower() for item in runtime_capabilities}

    native_issues = [
        CompatibilityIssue(
            code="native_geometry_backend",
            severity="info",
            message=(
                "This behavior uses the native geometry scorer; referenced MATLAB "
                "classifier files are provenance only."
            ),
        )
    ]
    native_issues.extend(
        CompatibilityIssue(
            code=f"profile_warning_{index + 1}",
            severity="warning",
            message=warning,
        )
        for index, warning in enumerate(profile.warnings)
    )
    native_issues.extend(_missing_legacy_detector_issues(profile))

    exact_issues: list[CompatibilityIssue] = []
    if LEGACY_EXACT_REFINEMENT_BACKEND not in capabilities:
        exact_issues.append(
            CompatibilityIssue(
                code="exact_runtime_unavailable",
                severity="blocker",
                message=(
                    "The installed tracker does not expose the validated whole-movie "
                    "legacy refinement runtime."
                ),
            )
        )
    parameter_path = parameters.source_path
    current_parameter_hash: str | None = None
    if parameter_path is None or not parameter_path.is_file():
        exact_issues.append(
            CompatibilityIssue(
                code="parameter_source_unbound",
                severity="blocker",
                message=(
                    "Exact refinement requires an existing parameter source so its "
                    "identity can be recorded."
                ),
            )
        )
    else:
        try:
            current_parameter_hash = sha256_file(parameter_path)
        except OSError as exc:
            exact_issues.append(
                CompatibilityIssue(
                    code="parameter_source_unreadable",
                    severity="blocker",
                    message=(
                        "The parameter source could not be re-read for identity "
                        f"validation: {exc}"
                    ),
                )
            )
        else:
            if profile.parameter_sha256 is None:
                exact_issues.append(
                    CompatibilityIssue(
                        code="parameter_identity_unavailable",
                        severity="blocker",
                        message=(
                            "The loaded profile has no parameter-file identity; "
                            "reload it before exact refinement."
                        ),
                    )
                )
            elif current_parameter_hash != profile.parameter_sha256:
                exact_issues.append(
                    CompatibilityIssue(
                        code="parameter_source_changed",
                        severity="blocker",
                        message=(
                            "The parameter file changed after this profile was "
                            "loaded; reload it before exact refinement."
                        ),
                    )
                )
    for issue in opaque:
        exact_issues.append(
            CompatibilityIssue(
                code=issue.code,
                severity="blocker",
                message=(
                    issue.message
                    + "; exact mode cannot assume that it is behaviorally inert"
                ),
                record_index=issue.record_index,
            )
        )
    exact_issues.extend(_exact_detector_source_issues(profile))

    active_references = _active_model_references(profile, references)
    if not active_references:
        exact_issues.append(
            CompatibilityIssue(
                code="tracking_model_not_referenced",
                severity="blocker",
                message="No tracking classifier model is referenced by this file.",
            )
        )
    elif len(active_references) > 1:
        exact_issues.append(
            CompatibilityIssue(
                code="tracking_model_ambiguous",
                severity="blocker",
                message=(
                    "Multiple active tracking model references were found; select a "
                    "single source explicitly before exact refinement."
                ),
            )
        )
    for reference in active_references:
        if not reference.exists:
            exact_issues.append(
                CompatibilityIssue(
                    code="tracking_model_missing",
                    severity="blocker",
                    message=(
                        "The referenced tracking model was not found: "
                        f"{reference.resolved_path}. Place the exact MAT file at "
                        "that path or update the load statement to one existing "
                        "model; no directory search or basename substitution was "
                        "attempted."
                    ),
                    record_index=reference.record_index,
                )
            )

    model_identity_valid = False
    if len(active_references) == 1 and active_references[0].exists:
        current_model_hash = active_references[0].sha256
        if current_model_hash is None or profile.model_sha256 is None:
            exact_issues.append(
                CompatibilityIssue(
                    code="tracking_model_identity_unavailable",
                    severity="blocker",
                    message=(
                        "The tracking model identity could not be validated; reload "
                        "the parameter file before exact refinement."
                    ),
                )
            )
        elif current_model_hash != profile.model_sha256:
            exact_issues.append(
                CompatibilityIssue(
                    code="tracking_model_changed",
                    severity="blocker",
                    message=(
                        "The referenced tracking model changed after this profile "
                        "was loaded; reload it and regenerate its classifier export."
                    ),
                )
            )
        else:
            model_identity_valid = True

    if model_identity_valid:
        exact_issues.extend(
            _legacy_model_runtime_issues(
                profile,
                active_references[0].resolved_path,
            )
        )

    neutral_path = (
        None
        if neutral_classifier_path is None
        else Path(neutral_classifier_path).expanduser().resolve(strict=False)
    )
    neutral_schema: str | None = None
    neutral_family: str | None = None
    neutral_source_hash: str | None = None
    source_hash = (
        active_references[0].sha256
        if model_identity_valid
        else None
    )
    if neutral_path is None:
        exact_issues.append(
            CompatibilityIssue(
                code="neutral_classifier_not_selected",
                severity="blocker",
                message=(
                    "Select a neutral numeric classifier export bound to the "
                    "referenced MAT model."
                ),
            )
        )
    else:
        try:
            neutral_schema, neutral_family, neutral_source_hash = (
                _validate_neutral_classifier(neutral_path, source_hash)
            )
        except (FileNotFoundError, OSError, NeutralClassifierFormatError, ValueError) as exc:
            exact_issues.append(
                CompatibilityIssue(
                    code="neutral_classifier_invalid",
                    severity="blocker",
                    message=f"The neutral classifier export is not usable: {exc}",
                )
            )

    neutral_source_bound = bool(
        neutral_path is not None
        and neutral_schema is not None
        and source_hash is not None
        and neutral_source_hash == source_hash
    )
    if neutral_path is not None and neutral_schema is not None and not neutral_source_bound:
        exact_issues.append(
            CompatibilityIssue(
                code="neutral_classifier_unbound",
                severity="blocker",
                message=(
                    "The classifier export is structurally valid, but it cannot be "
                    "proven to match one unchanged active tracking model."
                ),
            )
        )

    if not exact_issues:
        exact_issues.append(
            CompatibilityIssue(
                code="exact_inputs_validated",
                severity="info",
                message=(
                    "The exact refinement runtime, parameter source, MAT model, and "
                    "source-bound neutral export are validated."
                ),
            )
        )

    backends = {
        NATIVE_FAST_BACKEND: BackendReadiness(
            NATIVE_FAST_BACKEND,
            "Native StarryNite geometry tracking",
            tuple(native_issues),
        ),
        LEGACY_EXACT_REFINEMENT_BACKEND: BackendReadiness(
            LEGACY_EXACT_REFINEMENT_BACKEND,
            "Legacy classifier refinement",
            tuple(exact_issues),
        ),
    }
    return StarryNiteCompatibilityReport(
        syntax=parameters.syntax,
        encoding=str(encoding),
        parameter_path=parameter_path,
        parameter_sha256=profile.parameter_sha256,
        current_parameter_sha256=current_parameter_hash,
        stage_index=profile.stage_index,
        cell_count=profile.cell_count,
        effective_parameters=effective,
        opaque_issues=opaque,
        model_references=references,
        native_overrides={
            "detector": dict(profile.detector_settings),
            "tracker": dict(profile.tracker_settings),
        },
        backends=backends,
        regional_definition_count=len(regional_table.regions),
        regional_consumed_record_indices=regional_table.consumed_record_indices,
        neutral_classifier_path=neutral_path,
        neutral_classifier_schema=neutral_schema,
        neutral_classifier_family=neutral_family,
        neutral_source_model_sha256=neutral_source_hash,
        neutral_classifier_source_bound=neutral_source_bound,
    )


def select_compatibility_backend(
    report: StarryNiteCompatibilityReport,
    requested_backend: str,
) -> str:
    """Validate an explicit backend request without compatibility fallback."""

    if not isinstance(report, StarryNiteCompatibilityReport):
        raise TypeError("report must be a StarryNiteCompatibilityReport")
    return report.select_backend(requested_backend)


def _effective_parameters(
    profile: StarryNiteTuningProfile,
) -> tuple[EffectiveParameter, ...]:
    latest: dict[str, EffectiveParameter] = {}
    for index, record in enumerate(profile.parameters.records):
        if record.kind != "assignment" or record.name is None:
            continue
        normalized = normalize_parameter_name(record.name)
        latest[normalized] = EffectiveParameter(
            normalized_name=normalized,
            original_name=record.name,
            value=record.value,  # type: ignore[arg-type]
            record_index=index,
        )
    return tuple(sorted(latest.values(), key=lambda item: item.record_index))


def _missing_legacy_detector_issues(
    profile: StarryNiteTuningProfile,
) -> tuple[CompatibilityIssue, ...]:
    """Report required upstream fields without inventing compatibility values."""

    settings = profile.parameters.settings
    return tuple(
        CompatibilityIssue(
            code="legacy_detector_parameter_missing",
            severity="warning",
            message=(
                f"parameters.{field_name} is required by the standard legacy "
                "detector path but is absent. No value was guessed; native-fast "
                "can still use its explicit detector behavior, but exact legacy "
                "detector replay requires an explicit value in the parameter file."
            ),
        )
        for field_name in _REQUIRED_LEGACY_DETECTOR_FIELDS
        if f"parameters.{field_name}" not in settings
    )


def _exact_detector_source_issues(
    profile: StarryNiteTuningProfile,
) -> tuple[CompatibilityIssue, ...]:
    """Validate source values that native presets may otherwise guess."""

    issues: list[CompatibilityIssue] = []
    settings = profile.parameters.settings
    normalized = profile.parameters.normalized_settings
    for field_name in _REQUIRED_LEGACY_DETECTOR_FIELDS:
        if f"parameters.{field_name}" not in settings:
            issues.append(
                CompatibilityIssue(
                    code="legacy_exact_detector_parameter_missing",
                    severity="blocker",
                    message=(
                        f"Exact tracking requires an explicit "
                        f"parameters.{field_name} assignment for the production "
                        "legacy detector path; no native default is accepted at "
                        "this boundary."
                    ),
                )
            )

    diameter_name = "firsttimestepdiam"
    if diameter_name not in normalized:
        issues.append(
            CompatibilityIssue(
                code="legacy_exact_parameter_missing",
                severity="blocker",
                message=(
                    f"Exact tracking requires an explicit scalar {diameter_name} "
                    "in the parameter file; the native preset fallback is not "
                    "accepted at this boundary."
                ),
            )
        )
    else:
        diameter = normalized[diameter_name]
        if _not_finite_real(diameter) or float(diameter) <= 0:
            issues.append(
                CompatibilityIssue(
                    code="legacy_exact_parameter_invalid",
                    severity="blocker",
                    message=(
                        "Exact tracking requires positive scalar "
                        f"{diameter_name}."
                    ),
                )
            )

    count_name = "firsttimestepnumcells"
    if count_name not in normalized:
        issues.append(
            CompatibilityIssue(
                code="legacy_exact_parameter_missing",
                severity="blocker",
                message=(
                    f"Exact tracking requires an explicit scalar {count_name} in "
                    "the parameter file; the native preset fallback is not "
                    "accepted at this boundary."
                ),
            )
        )
    else:
        count = normalized[count_name]
        if (
            _not_finite_real(count)
            or float(count) < 0
            or not float(count).is_integer()
        ):
            issues.append(
                CompatibilityIssue(
                    code="legacy_exact_parameter_invalid",
                    severity="blocker",
                    message=(
                        "Exact tracking requires non-negative integer scalar "
                        f"{count_name}."
                    ),
                )
            )

    if "downsampling" not in normalized:
        issues.append(
            CompatibilityIssue(
                code="legacy_downsampling_missing",
                severity="blocker",
                message=(
                    "Exact AT movie replay requires an explicit scalar "
                    "downsampling=1 assignment in the parameter file; an omitted "
                    "value is not assumed."
                ),
            )
        )
    else:
        downsampling = normalized["downsampling"]
        if _not_finite_real(downsampling) or float(downsampling) != 1.0:
            issues.append(
                CompatibilityIssue(
                    code="legacy_downsampling_unsupported",
                    severity="blocker",
                    message=(
                        "Exact AT movie replay requires scalar downsampling=1; "
                        "other legacy image-resampling states are not reproduced."
                    ),
                )
            )

    distribution = profile.detector_settings.get("STARRYNITE_DISTRIBUTION_FILE")
    distribution_digest = profile.detector_settings.get(
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256"
    )
    if not distribution:
        issues.append(
            CompatibilityIssue(
                code="legacy_distribution_not_selected",
                severity="blocker",
                message=(
                    "Exact tracking requires distribution_file or "
                    "distribution_file2 for the source-bound detector tail."
                ),
            )
        )
    elif not Path(str(distribution)).is_file():
        issues.append(
            CompatibilityIssue(
                code="legacy_distribution_missing",
                severity="blocker",
                message=(
                    "The selected detector distribution file was not found: "
                    f"{distribution}. No alternate file was substituted."
                ),
            )
        )
    else:
        valid_distribution_digest = (
            isinstance(distribution_digest, str)
            and len(distribution_digest) == 64
            and all(
                character in "0123456789abcdef"
                for character in distribution_digest
            )
        )
        if not valid_distribution_digest:
            issues.append(
                CompatibilityIssue(
                    code="legacy_distribution_identity_unavailable",
                    severity="blocker",
                    message=(
                        "The detector distribution has no valid request-time "
                        "SHA-256 binding; reload the parameter source before exact "
                        "tracking."
                    ),
                )
            )
        try:
            loaded_distribution = load_legacy_disk_distributions(distribution)
        except (LegacyDetectorTailError, OSError, TypeError, ValueError) as exc:
            issues.append(
                CompatibilityIssue(
                    code="legacy_distribution_invalid",
                    severity="blocker",
                    message=(
                        "The selected detector distribution MAT payload is not "
                        f"usable by the exact tail: {exc}"
                    ),
                )
            )
        else:
            if (
                valid_distribution_digest
                and loaded_distribution.source_sha256 != distribution_digest
            ):
                issues.append(
                    CompatibilityIssue(
                        code="legacy_distribution_changed",
                        severity="blocker",
                        message=(
                            "The detector distribution changed after the parameter "
                            "profile was loaded; reload the source before exact "
                            "tracking."
                        ),
                    )
                )
    return tuple(issues)


def _not_finite_real(value: object) -> bool:
    return (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    )


def _is_supported_exact_static_dispatch(source: str) -> bool:
    """Recognize only cost functions implemented by exact early tracking."""

    return supported_static_cost_assignment(source)


def _legacy_model_runtime_issues(
    profile: StarryNiteTuningProfile,
    model_path: Path,
) -> tuple[CompatibilityIssue, ...]:
    """Validate numeric state needed before exact whole-movie execution."""

    issues: list[CompatibilityIssue] = []
    if profile.xy_um is None or profile.z_um is None:
        issues.append(
            CompatibilityIssue(
                code="legacy_calibration_unavailable",
                severity="blocker",
                message=(
                    "Exact refinement requires both xyres and zres in the legacy "
                    "parameter file so detector coordinates can be validated."
                ),
            )
        )
        return tuple(issues)
    try:
        prepared = prepare_legacy_runtime(profile)
        if prepared.model.path.resolve(strict=False) != model_path.resolve(strict=False):
            raise LegacyRuntimePreparationError(
                "The prepared MAT model does not match the active reference"
            )
    except (
        MatlabModelLoadError,
        LegacyEarlyTrackingError,
        LegacyFeatureExtractionError,
        LegacyRuntimePreparationError,
        LegacyStateError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        issues.append(
            CompatibilityIssue(
                code="legacy_tracking_state_unavailable",
                severity="blocker",
                message=(
                    "The source MAT file does not expose the numeric tracking "
                    "statistics and feature controls required by exact refinement: "
                    f"{exc}"
                ),
            )
        )
    else:
        unsupported_measurements: list[str] = []
        if prepared.early_parameters.polar_body_filter:
            unsupported_measurements.append("polar-body disk maxima")
        if prepared.early_parameters.hysteresis:
            unsupported_measurements.append("hysteresis local maxima")
        if unsupported_measurements:
            issues.append(
                CompatibilityIssue(
                    code="legacy_detector_measurement_unavailable",
                    severity="blocker",
                    message=(
                        "Exact early tracking needs raw detector measurements that "
                        "are not yet carried through the AT detection boundary: "
                        + ", ".join(unsupported_measurements)
                        + ". Disable the corresponding legacy option or add an "
                        "exact measurement adapter; no proxy values will be used."
                    ),
                )
            )
    return tuple(issues)


def _model_reference_statuses(
    profile: StarryNiteTuningProfile,
) -> tuple[ModelReferenceStatus, ...]:
    parameters = profile.parameters
    base = parameters.source_path.parent if parameters.source_path else Path.cwd()
    load_references = tuple(
        reference
        for reference in parameters.model_references
        if reference.source_kind == "load"
    )
    active_references = load_references or parameters.model_references
    # Hash only the sole active classifier. Multiple active references are
    # already ambiguous and cannot establish a source binding; streaming every
    # potentially large detector asset would add latency without more certainty.
    hashed_record_indices = {
        active_references[0].record_index
    } if len(active_references) == 1 else set()
    statuses: list[ModelReferenceStatus] = []
    for reference in parameters.model_references:
        resolved = reference.resolve(base)
        exists = resolved.is_file()
        digest = None
        if exists and reference.record_index in hashed_record_indices:
            try:
                digest = sha256_file(resolved)
            except OSError:
                digest = None
        statuses.append(
            ModelReferenceStatus(
                raw_path=reference.raw_path,
                resolved_path=resolved,
                source_kind=reference.source_kind,
                record_index=reference.record_index,
                exists=exists,
                sha256=digest,
            )
        )
    return tuple(statuses)


def _active_model_references(
    profile: StarryNiteTuningProfile,
    statuses: tuple[ModelReferenceStatus, ...],
) -> tuple[ModelReferenceStatus, ...]:
    load_references = tuple(item for item in statuses if item.source_kind == "load")
    selected = load_references or statuses
    if profile.model_path is None:
        return selected
    return selected


def _validate_neutral_classifier(
    path: Path,
    expected_source_hash: str | None,
) -> tuple[str, str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NeutralClassifierFormatError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise NeutralClassifierFormatError("neutral classifier root must be an object")
    schema = str(raw.get("schema", ""))
    if schema == NEUTRAL_CLASSIFIER_SCHEMA:
        model = load_neutral_classifier(
            path,
            expected_source_model_sha256=expected_source_hash,
        )
        family = model.classifier_family
    elif schema == NEUTRAL_AMBIGIOUS_CLASSIFIER_SCHEMA:
        model = load_neutral_ambigious_classifier(
            path,
            expected_source_model_sha256=expected_source_hash,
        )
        family = model.classifier_family
    else:
        raise NeutralClassifierFormatError(
            f"unsupported neutral classifier schema {schema!r}"
        )
    return schema, family, model.source_model_sha256


__all__ = [
    "BackendReadiness",
    "CompatibilityIssue",
    "EffectiveParameter",
    "LEGACY_EXACT_REFINEMENT_BACKEND",
    "ModelReferenceStatus",
    "NATIVE_FAST_BACKEND",
    "StarryNiteCompatibilityError",
    "StarryNiteCompatibilityReport",
    "build_compatibility_report",
    "select_compatibility_backend",
]
