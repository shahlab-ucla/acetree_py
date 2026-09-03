"""Registry and entry-point discovery for detector and tracker plugins."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .api import TRACKING_API_MAJOR, TRACKING_API_VERSION


DETECTOR_ENTRY_POINT_GROUP = "acetree_py.tracking.detectors"
TRACKER_ENTRY_POINT_GROUP = "acetree_py.tracking.trackers"
_VALID_PLUGIN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _api_major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tracking API version: {version!r}") from exc


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """Stable metadata exposed before a plugin implementation is created."""

    plugin_id: str
    kind: str
    display_name: str
    api_version: str = TRACKING_API_VERSION
    implementation_version: str = "0.1.0"
    description: str = ""
    settings_schema: Mapping[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _VALID_PLUGIN_ID.fullmatch(self.plugin_id):
            raise ValueError(
                "plugin_id must contain lowercase letters, digits, '.', '_' or '-'"
            )
        if self.kind not in {"detector", "tracker"}:
            raise ValueError("Component kind must be 'detector' or 'tracker'")
        if not self.display_name.strip():
            raise ValueError("display_name cannot be empty")
        if _api_major(self.api_version) != TRACKING_API_MAJOR:
            raise ValueError(
                f"Plugin {self.plugin_id!r} targets tracking API {self.api_version}; "
                f"this package supports API major {TRACKING_API_MAJOR}"
            )
        object.__setattr__(self, "settings_schema", _freeze_mapping(self.settings_schema))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))


@dataclass(frozen=True, slots=True)
class PluginContribution:
    """Object returned by an installed tracking entry point."""

    descriptor: ComponentDescriptor
    factory: Callable[[], Any]

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("Plugin factory must be callable")


class TrackingRegistry:
    """Detector/tracker catalog with isolated third-party discovery failures."""

    def __init__(self) -> None:
        self._detectors: dict[str, PluginContribution] = {}
        self._trackers: dict[str, PluginContribution] = {}
        self._discovery_errors: list[str] = []

    @property
    def discovery_errors(self) -> tuple[str, ...]:
        return tuple(self._discovery_errors)

    def register(self, contribution: PluginContribution, *, replace: bool = False) -> None:
        descriptor = contribution.descriptor
        catalog = self._detectors if descriptor.kind == "detector" else self._trackers
        if descriptor.plugin_id in catalog and not replace:
            raise ValueError(f"Duplicate {descriptor.kind} plugin ID: {descriptor.plugin_id}")
        catalog[descriptor.plugin_id] = contribution

    def register_detector(
        self,
        descriptor: ComponentDescriptor,
        factory: Callable[[], Any],
        *,
        replace: bool = False,
    ) -> None:
        if descriptor.kind != "detector":
            raise ValueError("Detector registration requires a detector descriptor")
        self.register(PluginContribution(descriptor, factory), replace=replace)

    def register_tracker(
        self,
        descriptor: ComponentDescriptor,
        factory: Callable[[], Any],
        *,
        replace: bool = False,
    ) -> None:
        if descriptor.kind != "tracker":
            raise ValueError("Tracker registration requires a tracker descriptor")
        self.register(PluginContribution(descriptor, factory), replace=replace)

    def detector_descriptors(self) -> tuple[ComponentDescriptor, ...]:
        return tuple(
            self._detectors[key].descriptor for key in sorted(self._detectors)
        )

    def tracker_descriptors(self) -> tuple[ComponentDescriptor, ...]:
        return tuple(self._trackers[key].descriptor for key in sorted(self._trackers))

    def get_descriptor(self, plugin_id: str) -> ComponentDescriptor:
        if plugin_id in self._detectors:
            return self._detectors[plugin_id].descriptor
        if plugin_id in self._trackers:
            return self._trackers[plugin_id].descriptor
        raise KeyError(f"Unknown tracking plugin: {plugin_id}")

    def default_settings(self, plugin_id: str) -> dict[str, Any]:
        """Return JSON-schema defaults advertised by a component."""
        descriptor = self.get_descriptor(plugin_id)
        defaults: dict[str, Any] = {}
        for key, schema in descriptor.settings_schema.items():
            if isinstance(schema, Mapping) and "default" in schema:
                value = schema["default"]
                defaults[key] = dict(value) if isinstance(value, Mapping) else value
        return defaults

    def create_detector(self, plugin_id: str):
        try:
            contribution = self._detectors[plugin_id]
        except KeyError as exc:
            raise KeyError(f"Unknown detector plugin: {plugin_id}") from exc
        component = contribution.factory()
        if not callable(getattr(component, "detect", None)):
            raise TypeError(f"Detector plugin {plugin_id!r} does not expose detect()")
        return component

    def create_tracker(self, plugin_id: str):
        try:
            contribution = self._trackers[plugin_id]
        except KeyError as exc:
            raise KeyError(f"Unknown tracker plugin: {plugin_id}") from exc
        component = contribution.factory()
        if not callable(getattr(component, "track", None)):
            raise TypeError(f"Tracker plugin {plugin_id!r} does not expose track()")
        if (
            "whole_movie_preflight" in contribution.descriptor.capabilities
            and not callable(getattr(component, "preflight_movie", None))
        ):
            raise TypeError(
                f"Tracker plugin {plugin_id!r} advertises whole_movie_preflight "
                "but does not expose preflight_movie()"
            )
        return component

    def discover_entry_points(self) -> None:
        """Load installed contributions without letting one bad plugin abort startup."""
        groups = (
            (DETECTOR_ENTRY_POINT_GROUP, "detector"),
            (TRACKER_ENTRY_POINT_GROUP, "tracker"),
        )
        available = metadata.entry_points()
        for group, expected_kind in groups:
            if hasattr(available, "select"):
                entries = available.select(group=group)
            else:  # pragma: no cover - compatibility with older importlib.metadata
                entries = available.get(group, ())
            for entry_point in entries:
                try:
                    loaded = entry_point.load()
                    contribution = self._coerce_contribution(loaded)
                    if contribution.descriptor.kind != expected_kind:
                        raise ValueError(
                            f"entry-point group expects {expected_kind}, got "
                            f"{contribution.descriptor.kind}"
                        )
                    self.register(contribution)
                except Exception as exc:  # plugin failures are reported, not fatal
                    self._discovery_errors.append(
                        f"{group}:{entry_point.name}: {type(exc).__name__}: {exc}"
                    )

    @staticmethod
    def _coerce_contribution(loaded: Any) -> PluginContribution:
        if isinstance(loaded, PluginContribution):
            return loaded
        descriptor = getattr(loaded, "descriptor", None)
        factory = getattr(loaded, "factory", None)
        if isinstance(descriptor, ComponentDescriptor) and callable(factory):
            return PluginContribution(descriptor, factory)
        if callable(loaded):
            result = loaded()
            if isinstance(result, PluginContribution):
                return result
        raise TypeError("Tracking entry point must expose or return PluginContribution")


def _detector_schema() -> dict[str, Any]:
    return {
        "TARGET_CHANNEL": {"type": "integer", "minimum": 1, "default": 1},
        "RADIUS": {"type": "number", "exclusiveMinimum": 0, "default": 4.0},
        "THRESHOLD": {"type": "number", "minimum": 0, "default": 0.0},
        "DO_SUBPIXEL_LOCALIZATION": {"type": "boolean", "default": True},
        "DO_MEDIAN_FILTERING": {"type": "boolean", "default": False},
    }


def _tracker_schema() -> dict[str, Any]:
    return {
        "LINKING_MAX_DISTANCE": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 15.0,
        },
        "LINKING_FEATURE_PENALTIES": {"type": "object", "default": {}},
        "ALLOW_GAP_CLOSING": {"type": "boolean", "default": True},
        "GAP_CLOSING_MAX_DISTANCE": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 15.0,
        },
        "GAP_CLOSING_FEATURE_PENALTIES": {"type": "object", "default": {}},
        "MAX_FRAME_GAP": {"type": "integer", "minimum": 1, "default": 2},
        "ALLOW_TRACK_SPLITTING": {"type": "boolean", "const": False},
        "ALLOW_TRACK_MERGING": {"type": "boolean", "const": False},
        "ALTERNATIVE_LINKING_COST_FACTOR": {
            "type": "number",
            "exclusiveMinimum": 1,
            "default": 1.05,
        },
    }


def _starrynite_detector_schema() -> dict[str, Any]:
    schema = _detector_schema()
    schema["DO_SUBPIXEL_LOCALIZATION"] = {
        "type": "boolean",
        "default": False,
    }
    schema.update(
        {
            "SIGMA": {"type": "number", "exclusiveMinimum": 0, "default": 1.0},
            "INTENSITY_THRESHOLD": {
                "type": "number",
                "minimum": 0,
                "default": 4.0,
            },
            "MIN_LOCAL_CONTRAST": {
                "type": "number",
                "minimum": 0,
                "default": 0.0,
            },
            "BOUNDARY_PERCENT": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 1,
                "default": 0.5,
            },
            "LARGE_RAY_THRESHOLD": {
                "type": "number",
                "exclusiveMinimum": 0,
                "default": 1.5,
            },
            "SMALL_RAY_THRESHOLD": {
                "type": "number",
                "exclusiveMinimum": 0,
                "default": 1.0 / 3.0,
            },
            "NNDIST_MERGE": {
                "type": "number",
                "minimum": 0,
                "default": 0.8,
            },
            "AR_MERGE": {
                "type": "number",
                "minimum": 0,
                "default": 1.6,
            },
            "RANGE_THRESHOLD": {
                "type": "number",
                "minimum": 0,
                "default": 1.0,
            },
            "SPLIT_THRESHOLD": {
                "type": "number",
                "minimum": 0,
                "default": 100.0,
            },
            "MERGE_LOWER": {
                "type": "number",
                "default": -300.0,
            },
            "MERGE_SPLIT": {
                "type": "number",
                "minimum": 0,
                "default": 1.0,
            },
            "MIN_SEPARATION": {
                "type": "number",
                "minimum": 0,
                "default": 0.0,
            },
            "DARK_NUCLEI": {"type": "boolean", "default": False},
            "ROI_POINTS_XY": {
                "type": "array",
                "default": [],
                "minItems": 3,
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
            },
            "ROI_CROPPED": {"type": "boolean", "default": False},
            "ROI_X_OFFSET": {
                "type": "number",
                "minimum": 0,
                "default": 0.0,
            },
            "ROI_Y_OFFSET": {
                "type": "number",
                "minimum": 0,
                "default": 0.0,
            },
            "ROI_X_MAX": {
                "type": "number",
                "minimum": 0,
                "default": 0.0,
            },
            "ROI_Y_MAX": {
                "type": "number",
                "minimum": 0,
                "default": 0.0,
            },
            "STARRYNITE_CELL_COUNT": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
            },
            "STARRYNITE_STAGE_INDEX": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
            },
            "STARRYNITE_PARAMETER_FILE": {"type": "string", "default": ""},
            "STARRYNITE_PARAMETER_SHA256": {"type": "string", "default": ""},
            "STARRYNITE_DISTRIBUTION_FILE": {"type": "string", "default": ""},
            "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": {
                "type": "string",
                "default": "",
            },
            "STARRYNITE_USE_STATIC_DIAMETER": {
                "type": "boolean",
                "default": False,
            },
        }
    )
    return schema


def _starrynite_tracker_schema() -> dict[str, Any]:
    schema = _tracker_schema()
    schema.update(
        {
            "ALLOW_TRACK_SPLITTING": {"type": "boolean", "default": True},
            "CANDIDATE_CUTOFF": {
                "type": "number",
                "exclusiveMinimum": 0,
                "default": 1.2,
            },
            "NN_NUMBER": {"type": "integer", "minimum": 1, "default": 2},
            "FORWARD_NN_NUMBER": {
                "type": "integer",
                "minimum": 1,
                "default": 4,
            },
            "SAFE_FACTOR": {
                "type": "number",
                "minimum": 0,
                "default": 2.0,
            },
            "DIVISION_COST_THRESHOLD": {
                "type": "number",
                "minimum": 0,
                "default": 0.85,
            },
            "DIVISION_MAX_DAUGHTER_DISTANCE": {
                "type": "number",
                "exclusiveMinimum": 0,
                "default": 15.0,
            },
            "DIVISION_MAX_DAUGHTER_SEPARATION": {
                "type": "number",
                "exclusiveMinimum": 0,
                "default": 12.0,
            },
            "DIVISION_MAX_MIDPOINT_ERROR": {
                "type": "number",
                "exclusiveMinimum": 0,
                "default": 6.0,
            },
            "DIVISION_MIN_QUALITY_RATIO": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "default": 0.25,
            },
            "DIVISION_MIN_VOLUME_RATIO": {
                "type": "number",
                "minimum": 0,
                "default": 0.15,
            },
            "DIVISION_MAX_VOLUME_RATIO": {
                "type": "number",
                "minimum": 0,
                "default": 2.5,
            },
            "DIVISION_REQUIRE_EXCESS_TARGET": {
                "type": "boolean",
                "default": True,
            },
            "MAX_ACTIVE_BRANCHES": {
                "type": "integer",
                "minimum": 2,
                "default": 8,
            },
            "RADIUS_PENALTY_WEIGHT": {
                "type": "number",
                "minimum": 0,
                "default": 0.25,
            },
            "QUALITY_PENALTY_WEIGHT": {
                "type": "number",
                "minimum": 0,
                "default": 0.0,
            },
            "STARRYNITE_PARAMETER_FILE": {"type": "string", "default": ""},
            "STARRYNITE_PARAMETER_SHA256": {"type": "string", "default": ""},
            "STARRYNITE_MODEL_FILE": {"type": "string", "default": ""},
            "STARRYNITE_MODEL_SHA256": {"type": "string", "default": ""},
            "STARRYNITE_COMPATIBILITY_MODE": {
                "type": "string",
                "enum": ["native_fast"],
                "default": "native_fast",
            },
        }
    )
    return schema


def _starrynite_legacy_exact_tracker_schema() -> dict[str, Any]:
    return {
        "STARRYNITE_COMPATIBILITY_MODE": {
            "type": "string",
            "const": "legacy_exact_refinement",
            "default": "legacy_exact_refinement",
        },
        "STARRYNITE_PARAMETER_FILE": {"type": "string", "default": ""},
        "STARRYNITE_PARAMETER_SHA256": {"type": "string", "default": ""},
        "STARRYNITE_MODEL_FILE": {"type": "string", "default": ""},
        "STARRYNITE_MODEL_SHA256": {"type": "string", "default": ""},
        "STARRYNITE_NEUTRAL_CLASSIFIER_FILE": {
            "type": "string",
            "default": "",
        },
        "STARRYNITE_NEUTRAL_CLASSIFIER_SHA256": {
            "type": "string",
            "default": "",
        },
        "STARRYNITE_FORCE_MODE": {"type": "boolean", "default": False},
        "STARRYNITE_FORCE_END_FRAME": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
        },
        "STARRYNITE_RECORD_ANSWERS": {
            "type": "boolean",
            "default": False,
        },
        "STARRYNITE_REQUIRE_EXACT_DETECTOR_TAIL": {
            "type": "boolean",
            "const": True,
            "default": True,
        },
        "STARRYNITE_USE_STATIC_DIAMETER": {
            "type": "boolean",
            "default": False,
        },
        "ALLOW_TRACK_SPLITTING": {
            "type": "boolean",
            "const": True,
            "default": True,
        },
    }


def build_default_registry(*, discover_plugins: bool = True) -> TrackingRegistry:
    from .detectors import DoGDetector, LoGDetector
    from .lap import SimpleLAPTracker
    from .starrynite.detector import StarryNiteDetector
    from .starrynite.legacy_exact_tracker import StarryNiteLegacyExactTracker
    from .starrynite.tracker import StarryNiteDivisionTracker

    registry = TrackingRegistry()
    for detector_type in (LoGDetector, DoGDetector):
        registry.register_detector(
            ComponentDescriptor(
                plugin_id=detector_type.plugin_id,
                kind="detector",
                display_name=detector_type.display_name,
                description="Anisotropic bright-blob detection on a 3D ZYX stack.",
                settings_schema=_detector_schema(),
                capabilities=("3d", "anisotropic", "subpixel"),
            ),
            detector_type,
        )
    registry.register_detector(
        ComponentDescriptor(
            plugin_id=StarryNiteDetector.plugin_id,
            kind="detector",
            display_name=StarryNiteDetector.display_name,
            description=(
                "Stage-aware anisotropic detection with safe legacy StarryNite "
                "parameter adapters."
            ),
            settings_schema=_starrynite_detector_schema(),
            capabilities=(
                "3d",
                "anisotropic",
                "subpixel",
                "legacy_parameter_presets",
                "native_starrynite",
            ),
        ),
        StarryNiteDetector,
    )
    registry.register_tracker(
        ComponentDescriptor(
            plugin_id=SimpleLAPTracker.plugin_id,
            kind="tracker",
            display_name=SimpleLAPTracker.display_name,
            description="One-to-one physical-distance LAP linking with gap closing.",
            settings_schema=_tracker_schema(),
            capabilities=("gap_closing",),
        ),
        SimpleLAPTracker,
    )
    registry.register_tracker(
        ComponentDescriptor(
            plugin_id=StarryNiteDivisionTracker.plugin_id,
            kind="tracker",
            display_name=StarryNiteDivisionTracker.display_name,
            description=(
                "Candidate-limited LAP linking with deterministic two-daughter "
                "division hypotheses and gap closure."
            ),
            settings_schema=_starrynite_tracker_schema(),
            capabilities=(
                "gap_closing",
                "splitting",
                "frontier_tracking",
                "selected_forward",
                "legacy_parameter_presets",
                "native_starrynite",
            ),
        ),
        StarryNiteDivisionTracker,
    )
    registry.register_tracker(
        ComponentDescriptor(
            plugin_id=StarryNiteLegacyExactTracker.plugin_id,
            kind="tracker",
            display_name=StarryNiteLegacyExactTracker.display_name,
            description=(
                "Source-bound whole-movie replay of StarryNite's staged geometry "
                "and legacy classifier/repair decisions."
            ),
            settings_schema=_starrynite_legacy_exact_tracker_schema(),
            capabilities=(
                "splitting",
                "global_only",
                "whole_movie_preflight",
                "whole_movie_refinement",
                "legacy_parameter_presets",
                "legacy_exact_refinement",
            ),
        ),
        StarryNiteLegacyExactTracker,
    )
    if discover_plugins:
        registry.discover_entry_points()
    return registry


@lru_cache(maxsize=1)
def get_default_registry() -> TrackingRegistry:
    """Return the process-wide built-in and installed-plugin registry."""
    return build_default_registry(discover_plugins=True)
