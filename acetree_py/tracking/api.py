"""Stable, dependency-light data contracts for image tracking plugins.

Coordinates crossing this API are always expressed in physical microns.  AceTree's
legacy pixel/plane coordinates are available through :class:`Calibration` helpers.
Frames and image planes are 1-based; x/y pixel coordinates remain zero-based.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


TRACKING_API_VERSION = "1.0"
TRACKING_API_MAJOR = 1
TRACKING_OUTCOME_CODES = frozenset(
    {"completed", "lost", "ambiguity", "division", "conflict"}
)
TRACKING_BRANCH_POLICIES = frozenset({"stop", "follow_best", "follow_both"})


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Calibration:
    """Voxel calibration and legacy coordinate conversion.

    ``pixel_to_physical`` and ``physical_to_pixel`` accept and return coordinates
    ordered as ``(x, y, z)``.  The z pixel coordinate is an AceTree image plane,
    so ``plane_start`` maps to physical z=0.
    """

    xy_um: float
    z_um: float
    plane_start: int = 1

    def __post_init__(self) -> None:
        _require_finite("xy_um", self.xy_um)
        _require_finite("z_um", self.z_um)
        if self.xy_um <= 0 or self.z_um <= 0:
            raise ValueError("Voxel calibration must be positive")
        if not isinstance(self.plane_start, int):
            raise TypeError("plane_start must be an integer")

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        """Physical voxel spacing ordered like a ZYX NumPy stack."""
        return (self.z_um, self.xy_um, self.xy_um)

    def pixel_to_physical(
        self, x_px: float, y_px: float, z_plane: float
    ) -> tuple[float, float, float]:
        """Convert image pixel/plane coordinates to physical microns."""
        return (
            float(x_px) * self.xy_um,
            float(y_px) * self.xy_um,
            (float(z_plane) - self.plane_start) * self.z_um,
        )

    def physical_to_pixel(
        self, x_um: float, y_um: float, z_um: float
    ) -> tuple[float, float, float]:
        """Convert physical microns to image pixel/plane coordinates."""
        return (
            float(x_um) / self.xy_um,
            float(y_um) / self.xy_um,
            float(z_um) / self.z_um + self.plane_start,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "xy_um": self.xy_um,
            "z_um": self.z_um,
            "plane_start": self.plane_start,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Calibration:
        return cls(
            xy_um=float(data["xy_um"]),
            z_um=float(data["z_um"]),
            plane_start=int(data.get("plane_start", 1)),
        )


@dataclass(frozen=True, slots=True)
class Detection:
    """One detector result at a 1-based frame in physical coordinates."""

    detection_id: str
    frame: int
    x_um: float
    y_um: float
    z_um: float
    radius_um: float
    quality: float
    features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.detection_id:
            raise ValueError("detection_id cannot be empty")
        if self.frame < 1:
            raise ValueError("frame must be 1-based and positive")
        for name in ("x_um", "y_um", "z_um", "radius_um", "quality"):
            _require_finite(name, float(getattr(self, name)))
        if self.radius_um <= 0:
            raise ValueError("radius_um must be positive")
        object.__setattr__(self, "features", _immutable_mapping(self.features))

    @property
    def position_um(self) -> tuple[float, float, float]:
        return (self.x_um, self.y_um, self.z_um)

    def to_pixel(self, calibration: Calibration) -> tuple[float, float, float]:
        """Return ``(x_px, y_px, z_plane)`` for AceTree integration."""
        return calibration.physical_to_pixel(self.x_um, self.y_um, self.z_um)

    @classmethod
    def from_pixel(
        cls,
        detection_id: str,
        frame: int,
        x_px: float,
        y_px: float,
        z_plane: float,
        radius_um: float,
        quality: float,
        calibration: Calibration,
        features: Mapping[str, Any] | None = None,
    ) -> Detection:
        x_um, y_um, z_um = calibration.pixel_to_physical(x_px, y_px, z_plane)
        return cls(
            detection_id=detection_id,
            frame=frame,
            x_um=x_um,
            y_um=y_um,
            z_um=z_um,
            radius_um=radius_um,
            quality=quality,
            features=features or {},
        )

    def feature(self, key: str) -> Any:
        """Return a TrackMate-style core feature or a detector feature."""
        core = {
            "POSITION_X": self.x_um,
            "POSITION_Y": self.y_um,
            "POSITION_Z": self.z_um,
            "FRAME": self.frame,
            "RADIUS": self.radius_um,
            "QUALITY": self.quality,
        }
        return core[key] if key in core else self.features.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "frame": self.frame,
            "x_um": self.x_um,
            "y_um": self.y_um,
            "z_um": self.z_um,
            "radius_um": self.radius_um,
            "quality": self.quality,
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Detection:
        return cls(
            detection_id=str(data["detection_id"]),
            frame=int(data["frame"]),
            x_um=float(data["x_um"]),
            y_um=float(data["y_um"]),
            z_um=float(data["z_um"]),
            radius_um=float(data["radius_um"]),
            quality=float(data["quality"]),
            features=data.get("features", {}),
        )


@dataclass(frozen=True, slots=True)
class TrackingOutcome:
    """Structured selected-forward completion or stopping diagnostic.

    ``review_candidates`` are detector observations that explain why tracking
    stopped.  They are intentionally separate from ``TrackingResult.detections``
    and are never materialized when a proposal is accepted.
    """

    code: str
    stop_frame: int | None
    last_accepted_frame: int
    predicted_position_um: tuple[float, float, float] | None
    search_radius_um: float
    review_candidates: tuple[Detection, ...] = ()

    def __post_init__(self) -> None:
        if self.code not in TRACKING_OUTCOME_CODES:
            raise ValueError(f"Unsupported tracking outcome code: {self.code!r}")
        if (
            isinstance(self.last_accepted_frame, bool)
            or not isinstance(self.last_accepted_frame, int)
            or self.last_accepted_frame < 1
        ):
            raise ValueError("last_accepted_frame must be a positive integer")
        if self.stop_frame is not None and (
            isinstance(self.stop_frame, bool)
            or not isinstance(self.stop_frame, int)
            or self.stop_frame < 1
        ):
            raise ValueError("stop_frame must be a positive integer or None")

        if self.code == "completed":
            if self.stop_frame is not None:
                raise ValueError("A completed outcome cannot have a stop_frame")
            if self.predicted_position_um is not None:
                raise ValueError(
                    "A completed outcome cannot have a stopped-frame prediction"
                )
        else:
            if self.stop_frame is None:
                raise ValueError(f"{self.code} outcome requires a stop_frame")
            if self.stop_frame <= self.last_accepted_frame:
                raise ValueError("stop_frame must follow last_accepted_frame")
            if self.predicted_position_um is None:
                raise ValueError(
                    f"{self.code} outcome requires predicted_position_um"
                )

        if self.predicted_position_um is not None:
            position = tuple(float(value) for value in self.predicted_position_um)
            if len(position) != 3:
                raise ValueError("predicted_position_um must contain x, y, and z")
            for name, value in zip(("x", "y", "z"), position):
                _require_finite(f"predicted_position_um.{name}", value)
            object.__setattr__(self, "predicted_position_um", position)

        search_radius_um = float(self.search_radius_um)
        _require_finite("search_radius_um", search_radius_um)
        if search_radius_um <= 0:
            raise ValueError("search_radius_um must be positive")
        object.__setattr__(self, "search_radius_um", search_radius_um)

        candidates = tuple(self.review_candidates)
        if any(not isinstance(candidate, Detection) for candidate in candidates):
            raise TypeError("review_candidates must contain Detection values")
        ids = [candidate.detection_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Review candidate IDs must be unique")
        if self.stop_frame is None and candidates:
            raise ValueError("A completed outcome cannot have review candidates")
        if self.stop_frame is not None and any(
            candidate.frame != self.stop_frame for candidate in candidates
        ):
            raise ValueError("Review candidates must belong to the stop frame")
        object.__setattr__(self, "review_candidates", candidates)

    @property
    def stopped_early(self) -> bool:
        return self.code != "completed"

    @property
    def frame(self) -> int | None:
        """Compatibility alias for stopped-frame presentation code."""
        return self.stop_frame

    @property
    def candidates(self) -> tuple[Detection, ...]:
        """Compatibility alias for review-only detector observations."""
        return self.review_candidates

    def to_dict(self) -> dict[str, Any]:
        position = self.predicted_position_um
        return {
            "code": self.code,
            "stop_frame": self.stop_frame,
            "last_accepted_frame": self.last_accepted_frame,
            "predicted_position_um": (
                None
                if position is None
                else {"x_um": position[0], "y_um": position[1], "z_um": position[2]}
            ),
            "search_radius_um": self.search_radius_um,
            "review_candidates": [
                candidate.to_dict() for candidate in self.review_candidates
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackingOutcome:
        position_data = data.get("predicted_position_um")
        position = None
        if position_data is not None:
            if not isinstance(position_data, Mapping):
                raise TypeError("predicted_position_um must be a mapping")
            position = (
                float(position_data["x_um"]),
                float(position_data["y_um"]),
                float(position_data["z_um"]),
            )
        return cls(
            code=str(data["code"]),
            stop_frame=(
                None if data.get("stop_frame") is None else int(data["stop_frame"])
            ),
            last_accepted_frame=int(data["last_accepted_frame"]),
            predicted_position_um=position,
            search_radius_um=float(data["search_radius_um"]),
            review_candidates=tuple(
                Detection.from_dict(item)
                for item in data.get("review_candidates", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class TrackEdge:
    """A directed temporal link between two detection IDs."""

    source_id: str
    target_id: str
    cost: float
    kind: str = "link"
    features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise ValueError("Track edge IDs cannot be empty")
        if self.source_id == self.target_id:
            raise ValueError("A detection cannot link to itself")
        _require_finite("cost", self.cost)
        if self.cost < 0:
            raise ValueError("cost cannot be negative")
        if self.kind not in {"link", "gap", "split"}:
            raise ValueError(f"Unsupported edge kind: {self.kind}")
        object.__setattr__(self, "features", _immutable_mapping(self.features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "cost": self.cost,
            "kind": self.kind,
            "features": dict(self.features),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackEdge:
        return cls(
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            cost=float(data["cost"]),
            kind=str(data.get("kind", "link")),
            features=data.get("features", {}),
        )


@dataclass(frozen=True, slots=True)
class TrackingScope:
    """Temporal and optional seeded scope for a tracking run."""

    kind: str
    start_frame: int
    end_frame: int
    seed_anchors: tuple[tuple[int, int], ...] = ()
    roi_radius_um: float | None = None
    ambiguity_ratio: float = 1.2
    branch_policy: str = "stop"

    def __post_init__(self) -> None:
        if self.kind not in {"global", "selected_forward"}:
            raise ValueError(f"Unsupported tracking scope: {self.kind}")
        if self.branch_policy not in TRACKING_BRANCH_POLICIES:
            raise ValueError(
                f"Unsupported tracking branch policy: {self.branch_policy!r}"
            )
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError("Tracking scope has an invalid frame range")
        anchors = tuple((int(t), int(i)) for t, i in self.seed_anchors)
        if any(t < 1 or i < 1 for t, i in anchors):
            raise ValueError("Seed anchors must use positive 1-based coordinates")
        if self.kind == "selected_forward" and not anchors:
            raise ValueError("selected_forward scope requires at least one seed anchor")
        object.__setattr__(self, "seed_anchors", anchors)
        if self.roi_radius_um is not None:
            _require_finite("roi_radius_um", self.roi_radius_um)
            if self.roi_radius_um <= 0:
                raise ValueError("roi_radius_um must be positive")
        _require_finite("ambiguity_ratio", self.ambiguity_ratio)
        if self.ambiguity_ratio <= 1:
            raise ValueError("ambiguity_ratio must be greater than 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "seed_anchors": [list(anchor) for anchor in self.seed_anchors],
            "roi_radius_um": self.roi_radius_um,
            "ambiguity_ratio": self.ambiguity_ratio,
            "branch_policy": self.branch_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackingScope:
        return cls(
            kind=str(data["kind"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            seed_anchors=tuple(tuple(map(int, anchor)) for anchor in data.get("seed_anchors", ())),
            roi_radius_um=(
                None if data.get("roi_radius_um") is None else float(data["roi_radius_um"])
            ),
            ambiguity_ratio=float(data.get("ambiguity_ratio", 1.2)),
            branch_policy=str(data.get("branch_policy", "stop")),
        )


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A selected detector/tracker plugin and its JSON-safe settings."""

    plugin_id: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plugin_id:
            raise ValueError("plugin_id cannot be empty")
        object.__setattr__(self, "settings", _immutable_mapping(self.settings))

    def to_dict(self) -> dict[str, Any]:
        return {"plugin_id": self.plugin_id, "settings": dict(self.settings)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComponentSpec:
        return cls(plugin_id=str(data["plugin_id"]), settings=data.get("settings", {}))


@dataclass(frozen=True, slots=True)
class TrackingRequest:
    """Serializable request selecting the detector, tracker, and run scope."""

    detector: ComponentSpec
    tracker: ComponentSpec
    scope: TrackingScope

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector.to_dict(),
            "tracker": self.tracker.to_dict(),
            "scope": self.scope.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackingRequest:
        return cls(
            detector=ComponentSpec.from_dict(data["detector"]),
            tracker=ComponentSpec.from_dict(data["tracker"]),
            scope=TrackingScope.from_dict(data["scope"]),
        )


@dataclass(frozen=True, slots=True)
class WholeMoviePreflightContext:
    """Immutable source facts supplied to an optional tracker preflight.

    A global tracker may expose ``preflight_movie(settings, *, context)`` to
    validate its detector binding and whole-movie inputs before the pipeline
    reads the first image stack.  Preflight is validation-only: plugins must
    not retain or mutate this context, the selected component settings, or
    external source files. ``target_channel`` is the zero-based channel index
    used by :class:`~acetree_py.io.image_provider.ImageProvider`.
    """

    detector_spec: ComponentSpec
    calibration: Calibration
    scope: TrackingScope
    source_num_timepoints: int
    source_num_channels: int
    target_channel: int

    def __post_init__(self) -> None:
        if not isinstance(self.detector_spec, ComponentSpec):
            raise TypeError("detector_spec must be ComponentSpec")
        if not isinstance(self.calibration, Calibration):
            raise TypeError("calibration must be Calibration")
        if not isinstance(self.scope, TrackingScope):
            raise TypeError("scope must be TrackingScope")
        for name, value in (
            ("source_num_timepoints", self.source_num_timepoints),
            ("source_num_channels", self.source_num_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.scope.end_frame > self.source_num_timepoints:
            raise ValueError("Tracking scope extends beyond the image source")
        if isinstance(self.target_channel, bool) or not isinstance(
            self.target_channel, int
        ):
            raise TypeError("target_channel must be an integer")
        if self.target_channel < 0 or self.target_channel >= self.source_num_channels:
            raise ValueError(
                "Target channel extends beyond the image source channel count"
            )

    @property
    def covers_complete_global_movie(self) -> bool:
        """Whether the request spans every frame without selected-cell seeds."""

        return (
            self.scope.kind == "global"
            and self.scope.start_frame == 1
            and self.scope.end_frame == self.source_num_timepoints
            and not self.scope.seed_anchors
        )


@dataclass(frozen=True, slots=True)
class TrackerGraphResult:
    """Optional whole-graph refinement returned by lineage-aware trackers.

    Basic trackers continue to return only edges. A classifier-backed tracker
    may additionally reject detector artifacts or rewrite tentative links, so
    the global pipeline accepts this richer result through an optional
    ``refine_graph`` method. Whole-movie compatibility backends may instead
    expose ``refine_movie`` to receive the immutable frame range and voxel
    calibration. Trackers advertising whole-movie preflight may additionally
    expose ``preflight_movie`` as documented by
    :class:`WholeMoviePreflightContext`.
    """

    detections: tuple[Detection, ...]
    edges: tuple[TrackEdge, ...]
    rejected_detection_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        detections = tuple(self.detections)
        edges = tuple(self.edges)
        identifiers = [item.detection_id for item in detections]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Refined graph detection IDs must be unique")
        known = set(identifiers)
        if any(edge.source_id not in known or edge.target_id not in known for edge in edges):
            raise ValueError("Refined graph edges must reference retained detections")
        rejected = tuple(str(item) for item in self.rejected_detection_ids)
        if any(not item for item in rejected) or len(rejected) != len(set(rejected)):
            raise ValueError("Rejected detection IDs must be unique and non-empty")
        if known & set(rejected):
            raise ValueError("A refined graph detection cannot also be rejected")
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "rejected_detection_ids", rejected)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "provenance", _immutable_mapping(self.provenance))


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Immutable proposal returned by a tracking run.

    ``existing_anchors`` distinguishes seed detections already present in the
    AceTree record from new detections that an integration layer should add.
    """

    request: TrackingRequest
    detections: tuple[Detection, ...]
    edges: tuple[TrackEdge, ...]
    existing_anchors: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    outcome: TrackingOutcome | None = None

    def __post_init__(self) -> None:
        detections = tuple(self.detections)
        edges = tuple(self.edges)
        ids = [d.detection_id for d in detections]
        if len(ids) != len(set(ids)):
            raise ValueError("Detection IDs must be unique")
        known = set(ids)
        if any(edge.source_id not in known or edge.target_id not in known for edge in edges):
            raise ValueError("Every edge endpoint must reference a result detection")
        anchors = {
            str(key): (int(value[0]), int(value[1]))
            for key, value in self.existing_anchors.items()
        }
        if any(key not in known for key in anchors):
            raise ValueError("Existing anchors must reference result detections")
        if any(t < 1 or i < 1 for t, i in anchors.values()):
            raise ValueError("Existing anchors must be positive and 1-based")
        if self.outcome is not None:
            if not isinstance(self.outcome, TrackingOutcome):
                raise TypeError("outcome must be a TrackingOutcome or None")
            scope = self.request.scope
            if scope.kind != "selected_forward":
                raise ValueError(
                    "Structured outcomes are only valid for selected-forward results"
                )
            if not detections:
                raise ValueError(
                    "A selected-forward outcome requires its accepted seed detection"
                )
            final_detection_frame = max(detection.frame for detection in detections)
            if self.outcome.last_accepted_frame != final_detection_frame:
                raise ValueError(
                    "Outcome last_accepted_frame must match the final proposal detection"
                )
            if not (
                scope.start_frame
                <= self.outcome.last_accepted_frame
                <= scope.end_frame
            ):
                raise ValueError(
                    "Outcome last_accepted_frame must be inside the tracking scope"
                )
            if self.outcome.code == "completed":
                if self.outcome.last_accepted_frame != scope.end_frame:
                    raise ValueError(
                        "A completed outcome must reach the tracking scope end frame"
                    )
            elif not (
                scope.start_frame < self.outcome.stop_frame <= scope.end_frame
            ):
                raise ValueError("Outcome stop_frame must be inside the tracking scope")
            review_ids = {
                candidate.detection_id
                for candidate in self.outcome.review_candidates
            }
            if known & review_ids:
                raise ValueError(
                    "Review-only candidates cannot also be proposal detections"
                )
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "existing_anchors", MappingProxyType(anchors))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(self, "provenance", _immutable_mapping(self.provenance))

    @property
    def new_detections(self) -> tuple[Detection, ...]:
        return tuple(d for d in self.detections if d.detection_id not in self.existing_anchors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "detections": [detection.to_dict() for detection in self.detections],
            "edges": [edge.to_dict() for edge in self.edges],
            "existing_anchors": {
                key: list(anchor) for key, anchor in self.existing_anchors.items()
            },
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackingResult:
        return cls(
            request=TrackingRequest.from_dict(data["request"]),
            detections=tuple(Detection.from_dict(item) for item in data.get("detections", ())),
            edges=tuple(TrackEdge.from_dict(item) for item in data.get("edges", ())),
            existing_anchors={
                str(key): tuple(map(int, anchor))
                for key, anchor in data.get("existing_anchors", {}).items()
            },
            warnings=tuple(data.get("warnings", ())),
            provenance=data.get("provenance", {}),
            outcome=(
                None
                if data.get("outcome") is None
                else TrackingOutcome.from_dict(data["outcome"])
            ),
        )
