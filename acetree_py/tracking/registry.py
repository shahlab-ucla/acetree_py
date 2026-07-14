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


def build_default_registry(*, discover_plugins: bool = True) -> TrackingRegistry:
    from .detectors import DoGDetector, LoGDetector
    from .lap import SimpleLAPTracker

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
    if discover_plugins:
        registry.discover_entry_points()
    return registry


@lru_cache(maxsize=1)
def get_default_registry() -> TrackingRegistry:
    """Return the process-wide built-in and installed-plugin registry."""
    return build_default_registry(discover_plugins=True)
