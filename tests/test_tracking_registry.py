"""Tests for tracking API serialization and plugin discovery."""

from __future__ import annotations

import pytest

from acetree_py.tracking import (
    ComponentDescriptor,
    ComponentSpec,
    Detection,
    PluginContribution,
    TrackEdge,
    TrackingRegistry,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
    build_default_registry,
)


def test_builtin_registry_has_stable_ids_and_factories():
    registry = build_default_registry(discover_plugins=False)
    assert [item.plugin_id for item in registry.detector_descriptors()] == [
        "acetree.dog3d",
        "acetree.log3d",
        "acetree.starrynite_detector",
    ]
    assert [item.plugin_id for item in registry.tracker_descriptors()] == [
        "acetree.simple_lap",
        "acetree.starrynite_division",
        "acetree.starrynite_legacy_exact",
    ]
    assert callable(registry.create_detector("acetree.log3d").detect)
    assert callable(registry.create_tracker("acetree.simple_lap").track)
    assert registry.default_settings("acetree.dog3d")["RADIUS"] == 4.0
    assert registry.default_settings("acetree.simple_lap")["MAX_FRAME_GAP"] == 2
    assert (
        registry.default_settings("acetree.starrynite_division")[
            "ALLOW_TRACK_SPLITTING"
        ]
        is True
    )
    exact = registry.get_descriptor("acetree.starrynite_legacy_exact")
    assert {
        "global_only",
        "whole_movie_preflight",
        "whole_movie_refinement",
        "legacy_exact_refinement",
    } <= set(exact.capabilities)
    assert (
        registry.default_settings(exact.plugin_id)["STARRYNITE_COMPATIBILITY_MODE"]
        == "legacy_exact_refinement"
    )
    exact_tracker = registry.create_tracker(exact.plugin_id)
    assert callable(exact_tracker.preflight_movie)
    assert callable(exact_tracker.refine_movie)


def test_descriptor_enforces_id_kind_and_api_major():
    with pytest.raises(ValueError, match="plugin_id"):
        ComponentDescriptor("Bad ID", "detector", "Bad")
    with pytest.raises(ValueError, match="kind"):
        ComponentDescriptor("example.bad", "widget", "Bad")
    with pytest.raises(ValueError, match="API major"):
        ComponentDescriptor("example.future", "detector", "Future", api_version="2.0")


def test_registry_rejects_duplicates_and_invalid_component():
    descriptor = ComponentDescriptor("example.detector", "detector", "Example")
    registry = TrackingRegistry()
    registry.register_detector(descriptor, lambda: object())
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register_detector(descriptor, lambda: object())
    with pytest.raises(TypeError, match="detect"):
        registry.create_detector("example.detector")


def test_registry_rejects_tracker_missing_advertised_preflight_hook():
    class Tracker:
        def track(self, *_args):
            return ()

    registry = TrackingRegistry()
    registry.register_tracker(
        ComponentDescriptor(
            "example.preflight",
            "tracker",
            "Incomplete preflight tracker",
            capabilities=("whole_movie_preflight",),
        ),
        Tracker,
    )

    with pytest.raises(TypeError, match="does not expose preflight_movie"):
        registry.create_tracker("example.preflight")


def test_entry_point_discovery_isolated_and_validated(monkeypatch):
    class Detector:
        def detect(self, *args, **kwargs):
            return ()

    contribution = PluginContribution(
        ComponentDescriptor("example.detect", "detector", "Example detector"),
        Detector,
    )

    class EntryPoint:
        def __init__(self, name, loaded=None, error=None):
            self.name = name
            self.loaded = loaded
            self.error = error

        def load(self):
            if self.error:
                raise self.error
            return self.loaded

    class EntryPoints(list):
        def select(self, *, group):
            if group.endswith("detectors"):
                return self
            return []

    from acetree_py.tracking import registry as registry_module

    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda: EntryPoints(
            [
                EntryPoint("valid", contribution),
                EntryPoint("broken", error=RuntimeError("boom")),
            ]
        ),
    )
    registry = TrackingRegistry()
    registry.discover_entry_points()
    assert registry.get_descriptor("example.detect").display_name == "Example detector"
    assert len(registry.discovery_errors) == 1
    assert "broken" in registry.discovery_errors[0]


def test_tracking_result_round_trip_preserves_existing_seed_boundary():
    request = TrackingRequest(
        ComponentSpec("acetree.log3d", {"RADIUS": 3.0}),
        ComponentSpec("acetree.simple_lap", {"LINKING_MAX_DISTANCE": 5.0}),
        TrackingScope(
            "selected_forward", 2, 4, seed_anchors=((2, 7),), roi_radius_um=8.0
        ),
    )
    seed = Detection("seed", 2, 1.0, 2.0, 3.0, 2.0, 1.0)
    found = Detection("found", 3, 2.0, 2.0, 3.0, 2.0, 5.0)
    result = TrackingResult(
        request,
        (seed, found),
        (TrackEdge("seed", "found", 1.0),),
        existing_anchors={"seed": (2, 7)},
        warnings=("review",),
        provenance={"engine": "test"},
    )

    restored = TrackingResult.from_dict(result.to_dict())
    assert restored.to_dict() == result.to_dict()
    assert restored.new_detections == (found,)
    with pytest.raises(TypeError):
        result.existing_anchors["other"] = (1, 1)


def test_nested_schema_defaults_are_detached_for_each_caller():
    registry = TrackingRegistry()
    defaults = {"weights": [0.1, 0.2]}
    descriptor = ComponentDescriptor(
        "test.nested", "detector", "Nested defaults",
        settings_schema={"options": {"default": defaults}},
    )
    registry.register_detector(descriptor, lambda: None)
    defaults["weights"].append(99)
    first = registry.default_settings("test.nested")
    first["options"]["weights"].clear()

    assert registry.default_settings("test.nested") == {
        "options": {"weights": [0.1, 0.2]},
    }
