"""Versioned persistence for uncommitted tracking proposals.

Tracking runs are deliberately side-effect free until a user accepts them.
This module stores the complete proposal beside a dataset so it can be
reviewed, reproduced, or accepted in a later session.  The file is replaced
atomically; a failed write therefore never destroys the last good proposal.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .api import (
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)

TRACKING_PROPOSAL_SCHEMA = "acetree.tracking-proposal"
TRACKING_PROPOSAL_VERSION = 1
TRACKING_PROPOSAL_SUFFIX = ".tracking.json"


class TrackingProposalFormatError(ValueError):
    """Raised when a tracking-proposal sidecar is malformed or unsupported."""


def write_tracking_proposal(
    path: str | Path,
    result: TrackingResult,
) -> Path:
    """Atomically write *result* to a versioned JSON sidecar.

    The temporary file is created in the destination directory, flushed to
    disk, assigned the destination's existing mode (or the process' normal
    creation mode), then committed with :func:`os.replace`.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = tracking_result_to_dict(result)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        # Sidecars are exchange artifacts, so emit strict RFC-compatible JSON.
        # The public API rejects non-finite result values and plugin settings
        # are required to use JSON-safe finite values as well.
        allow_nan=False,
    ) + "\n"

    fd, temp_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, _replacement_mode(destination))
        os.replace(temp_path, destination)
    except BaseException:
        # os.fdopen owns fd after it succeeds.  If it did not, close the raw
        # descriptor before removing the staging file.
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    finally:
        temp_path.unlink(missing_ok=True)
    return destination


def tracking_sidecar_path(dataset_config_or_zip_path: str | Path) -> Path:
    """Return conventional ``<dataset stem>.tracking.json`` sidecar path."""
    dataset_path = Path(dataset_config_or_zip_path)
    return dataset_path.with_name(dataset_path.stem + TRACKING_PROPOSAL_SUFFIX)


def read_tracking_proposal(path: str | Path) -> TrackingResult:
    """Read and validate a tracking proposal sidecar."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise TrackingProposalFormatError(
            f"Invalid tracking proposal JSON in {source}: {exc.msg}"
        ) from exc
    return tracking_result_from_dict(payload)


def tracking_result_to_dict(result: TrackingResult) -> dict[str, Any]:
    """Return the stable v1 JSON representation of *result*."""
    request = result.request
    return {
        "schema": TRACKING_PROPOSAL_SCHEMA,
        "schema_version": TRACKING_PROPOSAL_VERSION,
        "request": {
            "detector": _component_to_dict(request.detector),
            "tracker": _component_to_dict(request.tracker),
            "scope": {
                **request.scope.to_dict(),
                # Retain the v1 sidecar anchor shape while sharing scope fields.
                "seed_anchors": [
                    {"time": time, "index": index}
                    for time, index in request.scope.seed_anchors
                ],
            },
        },
        "result": {
            "detections": [
                _detection_to_dict(detection)
                for detection in result.detections
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "cost": edge.cost,
                    "kind": edge.kind,
                    "features": _json_value(edge.features),
                }
                for edge in result.edges
            ],
            "existing_anchors": {
                detection_id: {"time": time, "index": index}
                for detection_id, (time, index) in result.existing_anchors.items()
            },
            "warnings": list(result.warnings),
            "provenance": _json_value(result.provenance),
            "outcome": _outcome_to_dict(result.outcome),
        },
    }


def tracking_result_from_dict(payload: Any) -> TrackingResult:
    """Construct a :class:`TrackingResult` from its validated v1 mapping."""
    try:
        root = _mapping(payload, "tracking proposal")
        schema = _string(root.get("schema"), "schema")
        if schema != TRACKING_PROPOSAL_SCHEMA:
            raise TrackingProposalFormatError(
                f"Unsupported tracking proposal schema: {schema!r}"
            )
        version = _integer(root.get("schema_version"), "schema_version")
        if version != TRACKING_PROPOSAL_VERSION:
            raise TrackingProposalFormatError(
                "Unsupported tracking proposal schema version "
                f"{version}; expected {TRACKING_PROPOSAL_VERSION}"
            )

        request_data = _mapping(root.get("request"), "request")
        scope_data = _mapping(request_data.get("scope"), "request.scope")
        scope = TrackingScope(
            kind=_string(scope_data.get("kind"), "request.scope.kind"),
            start_frame=_integer(
                scope_data.get("start_frame"), "request.scope.start_frame"
            ),
            end_frame=_integer(
                scope_data.get("end_frame"), "request.scope.end_frame"
            ),
            seed_anchors=tuple(
                (
                    _integer(
                        _mapping(anchor, "seed anchor").get("time"),
                        "seed anchor time",
                    ),
                    _integer(
                        _mapping(anchor, "seed anchor").get("index"),
                        "seed anchor index",
                    ),
                )
                for anchor in _sequence(
                    scope_data.get("seed_anchors", []),
                    "request.scope.seed_anchors",
                )
            ),
            roi_radius_um=(
                None
                if scope_data.get("roi_radius_um") is None
                else _number(
                    scope_data.get("roi_radius_um"),
                    "request.scope.roi_radius_um",
                )
            ),
            ambiguity_ratio=_number(
                scope_data.get("ambiguity_ratio"),
                "request.scope.ambiguity_ratio",
            ),
            branch_policy=_string(
                scope_data.get("branch_policy", "stop"),
                "request.scope.branch_policy",
            ),
        )
        request = TrackingRequest(
            detector=_component_from_dict(
                request_data.get("detector"), "request.detector"
            ),
            tracker=_component_from_dict(
                request_data.get("tracker"), "request.tracker"
            ),
            scope=scope,
        )

        result_data = _mapping(root.get("result"), "result")
        detections = tuple(
            _detection_from_dict(item)
            for item in _sequence(result_data.get("detections"), "result.detections")
        )
        edges = tuple(
            _edge_from_dict(item)
            for item in _sequence(result_data.get("edges"), "result.edges")
        )
        anchors_data = _mapping(
            result_data.get("existing_anchors", {}),
            "result.existing_anchors",
        )
        anchors = {
            _string(detection_id, "existing anchor detection_id"): (
                _integer(
                    _mapping(anchor, "existing anchor").get("time"),
                    "existing anchor time",
                ),
                _integer(
                    _mapping(anchor, "existing anchor").get("index"),
                    "existing anchor index",
                ),
            )
            for detection_id, anchor in anchors_data.items()
        }
        warnings = tuple(
            _string(item, "warning")
            for item in _sequence(result_data.get("warnings", []), "result.warnings")
        )
        provenance = dict(
            _mapping(result_data.get("provenance", {}), "result.provenance")
        )
        outcome_data = result_data.get("outcome")
        outcome = (
            None
            if outcome_data is None
            else _outcome_from_dict(outcome_data)
        )
        return TrackingResult(
            request=request,
            detections=detections,
            edges=edges,
            existing_anchors=anchors,
            warnings=warnings,
            provenance=provenance,
            outcome=outcome,
        )
    except TrackingProposalFormatError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackingProposalFormatError(
            f"Invalid tracking proposal payload: {exc}"
        ) from exc


def _component_to_dict(component: ComponentSpec) -> dict[str, Any]:
    return {
        "plugin_id": component.plugin_id,
        "settings": _json_value(component.settings),
    }


def _component_from_dict(value: Any, label: str) -> ComponentSpec:
    data = _mapping(value, label)
    return ComponentSpec(
        plugin_id=_string(data.get("plugin_id"), f"{label}.plugin_id"),
        settings=dict(_mapping(data.get("settings", {}), f"{label}.settings")),
    )


def _detection_to_dict(detection: Detection) -> dict[str, Any]:
    return {
        "detection_id": detection.detection_id,
        "frame": detection.frame,
        "x_um": detection.x_um,
        "y_um": detection.y_um,
        "z_um": detection.z_um,
        "radius_um": detection.radius_um,
        "quality": detection.quality,
        "features": _json_value(detection.features),
    }


def _detection_from_dict(value: Any) -> Detection:
    data = _mapping(value, "detection")
    return Detection(
        detection_id=_string(data.get("detection_id"), "detection.detection_id"),
        frame=_integer(data.get("frame"), "detection.frame"),
        x_um=_number(data.get("x_um"), "detection.x_um"),
        y_um=_number(data.get("y_um"), "detection.y_um"),
        z_um=_number(data.get("z_um"), "detection.z_um"),
        radius_um=_number(data.get("radius_um"), "detection.radius_um"),
        quality=_number(data.get("quality"), "detection.quality"),
        features=dict(_mapping(data.get("features", {}), "detection.features")),
    )


def _edge_from_dict(value: Any) -> TrackEdge:
    data = _mapping(value, "edge")
    return TrackEdge(
        source_id=_string(data.get("source_id"), "edge.source_id"),
        target_id=_string(data.get("target_id"), "edge.target_id"),
        cost=_number(data.get("cost"), "edge.cost"),
        kind=_string(data.get("kind"), "edge.kind"),
        features=dict(_mapping(data.get("features", {}), "edge.features")),
    )


def _outcome_to_dict(outcome: TrackingOutcome | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    position = outcome.predicted_position_um
    return {
        "code": outcome.code,
        "stop_frame": outcome.stop_frame,
        "last_accepted_frame": outcome.last_accepted_frame,
        "predicted_position_um": (
            None
            if position is None
            else {"x_um": position[0], "y_um": position[1], "z_um": position[2]}
        ),
        "search_radius_um": outcome.search_radius_um,
        "review_candidates": [
            _detection_to_dict(candidate)
            for candidate in outcome.review_candidates
        ],
    }


def _outcome_from_dict(value: Any) -> TrackingOutcome:
    data = _mapping(value, "result.outcome")
    stop_value = data.get("stop_frame")
    position_value = data.get("predicted_position_um")
    position = None
    if position_value is not None:
        position_data = _mapping(
            position_value,
            "result.outcome.predicted_position_um",
        )
        position = (
            _number(
                position_data.get("x_um"),
                "result.outcome.predicted_position_um.x_um",
            ),
            _number(
                position_data.get("y_um"),
                "result.outcome.predicted_position_um.y_um",
            ),
            _number(
                position_data.get("z_um"),
                "result.outcome.predicted_position_um.z_um",
            ),
        )
    return TrackingOutcome(
        code=_string(data.get("code"), "result.outcome.code"),
        stop_frame=(
            None
            if stop_value is None
            else _integer(stop_value, "result.outcome.stop_frame")
        ),
        last_accepted_frame=_integer(
            data.get("last_accepted_frame"),
            "result.outcome.last_accepted_frame",
        ),
        predicted_position_um=position,
        search_radius_um=_number(
            data.get("search_radius_um"),
            "result.outcome.search_radius_um",
        ),
        review_candidates=tuple(
            _detection_from_dict(candidate)
            for candidate in _sequence(
                data.get("review_candidates", []),
                "result.outcome.review_candidates",
            )
        ),
    )


def _json_value(value: Any) -> Any:
    """Copy a JSON-compatible value out of immutable/proxy containers."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("Tracking proposal mapping keys must be strings")
            result[key] = _json_value(child)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(child) for child in value]
    raise TypeError(
        "Tracking proposal values must be JSON-compatible; "
        f"got {type(value).__name__}"
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrackingProposalFormatError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TrackingProposalFormatError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TrackingProposalFormatError(f"{label} must be a string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackingProposalFormatError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackingProposalFormatError(f"{label} must be a number")
    return float(value)


def _replacement_mode(destination: Path) -> int:
    """Mode for an atomic replacement, preserving target or normal defaults."""
    try:
        return stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        previous = os.umask(0)
        os.umask(previous)
        return 0o666 & ~previous


def _reject_json_constant(value: str) -> None:
    """Reject Python's non-standard ``NaN``/``Infinity`` JSON extensions."""
    raise TrackingProposalFormatError(
        f"Invalid non-finite number in tracking proposal JSON: {value}"
    )


# Concise aliases for callers that speak in terms of persisted run results.
write_tracking_result = write_tracking_proposal
read_tracking_result = read_tracking_proposal
