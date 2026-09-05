"""Build tracking component settings consistently across configuration forms.

Registry defaults are the base, source/preset settings override those defaults,
and explicit form controls win last. Only controls advertised by the selected
component are added; source settings retain their provenance and advanced keys.
Workflow-specific validation (scope, source bindings and capabilities) stays
with the caller.
"""

from __future__ import annotations

from typing import Any, Mapping

from .api import ComponentSpec
from .registry import TrackingRegistry


def _component_spec(
    registry: TrackingRegistry,
    plugin_id: str,
    kind: str,
    source_settings: Mapping[str, Any] | None,
    controls: Mapping[str, Any],
) -> ComponentSpec:
    descriptor = registry.get_descriptor(plugin_id)
    if descriptor.kind != kind:
        raise ValueError(f"{plugin_id!r} is not a {kind} plugin")
    settings = registry.default_settings(plugin_id)
    settings.update(source_settings or {})
    settings.update(
        (key, value)
        for key, value in controls.items()
        if key in descriptor.settings_schema
    )
    return ComponentSpec(plugin_id, settings)


def build_detector_spec(
    registry: TrackingRegistry,
    plugin_id: str,
    *,
    channel: int,
    radius_um: float,
    threshold: float,
    median_filter: bool = False,
    subpixel: bool | None = None,
    source_settings: Mapping[str, Any] | None = None,
    exact_starrynite: bool = False,
) -> ComponentSpec:
    """Apply detector controls, preserving default localization when omitted.

    StarryNite uses an absolute intensity threshold. Its distribution binding is
    retained only when the caller explicitly selects the exact movie workflow.
    """
    controls = {
        "TARGET_CHANNEL": channel,
        "RADIUS": radius_um,
        "THRESHOLD": threshold,
        "DO_MEDIAN_FILTERING": median_filter,
    }
    if subpixel is not None:
        controls["DO_SUBPIXEL_LOCALIZATION"] = subpixel
    if plugin_id == "acetree.starrynite_detector":
        controls["THRESHOLD"] = 0.0
        controls["INTENSITY_THRESHOLD"] = threshold
        if not exact_starrynite:
            from .starrynite.presets import native_detector_settings

            source_settings = native_detector_settings(source_settings or {})
    return _component_spec(registry, plugin_id, "detector", source_settings, controls)


def build_tracker_spec(
    registry: TrackingRegistry,
    plugin_id: str,
    *,
    max_distance_um: float,
    missing_frames: int,
    allow_splitting: bool,
    source_settings: Mapping[str, Any] | None = None,
) -> ComponentSpec:
    """Map missed frames to frame delta and apply supported tracking controls.

    Exact trackers advertise no native linking controls, so their validated
    source binding is preserved without adding geometry settings they ignore.
    """
    controls = {
        "LINKING_MAX_DISTANCE": max_distance_um,
        "ALLOW_GAP_CLOSING": missing_frames > 0,
        "GAP_CLOSING_MAX_DISTANCE": max_distance_um,
        "MAX_FRAME_GAP": missing_frames + 1 if missing_frames > 0 else 1,
        "ALLOW_TRACK_SPLITTING": allow_splitting,
        "ALLOW_TRACK_MERGING": False,
    }
    return _component_spec(registry, plugin_id, "tracker", source_settings, controls)
