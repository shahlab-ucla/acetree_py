"""Tests for durable, versioned tracking proposal sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from acetree_py.tracking.api import (
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)
from acetree_py.tracking.persistence import (
    TRACKING_PROPOSAL_SCHEMA,
    TRACKING_PROPOSAL_VERSION,
    TrackingProposalFormatError,
    read_tracking_proposal,
    tracking_result_from_dict,
    tracking_result_to_dict,
    tracking_sidecar_path,
    write_tracking_proposal,
)


def _result(*, settings: dict | None = None, branch_policy: str = "stop") -> TrackingResult:
    request = TrackingRequest(
        detector=ComponentSpec(
            "org.acetree.detector.dog",
            settings
            or {
                "TARGET_CHANNEL": 1,
                "RADIUS": 2.5,
                "nested": {"thresholds": [0.1, 0.2], "enabled": True},
            },
        ),
        tracker=ComponentSpec(
            "org.acetree.tracker.lap",
            {"LINKING_MAX_DISTANCE": 8.0, "MAX_FRAME_GAP": 2},
        ),
        scope=TrackingScope(
            kind="selected_forward",
            start_frame=3,
            end_frame=5,
            seed_anchors=((3, 7),),
            roi_radius_um=12.5,
            ambiguity_ratio=1.4,
            branch_policy=branch_policy,
        ),
    )
    detections = (
        Detection("seed", 3, 1.0, 2.0, 3.0, 2.5, 99.0, {"manual": 1.0}),
        Detection("next", 4, 1.5, 2.5, 3.5, 2.4, 8.0, {"response": 8.0}),
    )
    review_candidate = Detection(
        "ambiguous-next",
        5,
        2.0,
        3.0,
        4.0,
        2.3,
        7.5,
        {"review_only": True},
    )
    return TrackingResult(
        request=request,
        detections=detections,
        edges=(TrackEdge("seed", "next", 0.75, "link", {"distance_um": 0.9}),),
        existing_anchors={"seed": (3, 7)},
        warnings=("Stopped before an ambiguous division",),
        provenance={
            "run_id": "run-123",
            "plugins": {
                "detector": {"version": "1.2.3", "distribution": "demo"},
                "tracker": {"version": "4.5.6"},
            },
            "command": ["acetree", "track"],
        },
        outcome=TrackingOutcome(
            code="ambiguity",
            stop_frame=5,
            last_accepted_frame=4,
            predicted_position_um=(2.1, 3.1, 4.1),
            search_radius_um=12.5,
            review_candidates=(review_candidate,),
        ),
    )


@pytest.mark.parametrize("branch_policy", ["stop", "follow_best", "follow_both"])
def test_tracking_proposal_round_trip_is_lossless(tmp_path: Path, branch_policy):
    path = tmp_path / "embryo.tracking.json"
    original = _result(branch_policy=branch_policy)

    assert write_tracking_proposal(path, original) == path
    loaded = read_tracking_proposal(path)

    assert loaded == original
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == TRACKING_PROPOSAL_SCHEMA
    assert payload["schema_version"] == TRACKING_PROPOSAL_VERSION
    assert payload["request"]["detector"]["plugin_id"] == (
        "org.acetree.detector.dog"
    )
    assert payload["result"]["existing_anchors"]["seed"] == {
        "time": 3,
        "index": 7,
    }
    assert payload["result"]["provenance"]["plugins"]["tracker"][
        "version"
    ] == "4.5.6"
    assert payload["result"]["outcome"]["code"] == "ambiguity"
    assert payload["result"]["outcome"]["predicted_position_um"] == {
        "x_um": 2.1,
        "y_um": 3.1,
        "z_um": 4.1,
    }
    assert payload["result"]["outcome"]["review_candidates"][0][
        "detection_id"
    ] == "ambiguous-next"


def test_v1_sidecar_without_outcome_remains_readable():
    payload = tracking_result_to_dict(_result())
    payload["result"].pop("outcome")
    payload["request"]["scope"].pop("branch_policy")

    loaded = tracking_result_from_dict(payload)

    assert loaded.outcome is None
    assert loaded.request.scope.branch_policy == "stop"


def test_tracking_sidecar_path_uses_dataset_stem(tmp_path: Path):
    assert tracking_sidecar_path(tmp_path / "embryo.xml") == (
        tmp_path / "embryo.tracking.json"
    )
    assert tracking_sidecar_path(tmp_path / "embryo.zip") == (
        tmp_path / "embryo.tracking.json"
    )


def test_unknown_schema_version_is_rejected():
    payload = {
        "schema": TRACKING_PROPOSAL_SCHEMA,
        "schema_version": TRACKING_PROPOSAL_VERSION + 1,
    }
    with pytest.raises(TrackingProposalFormatError, match="schema version"):
        tracking_result_from_dict(payload)


def test_malformed_json_reports_a_format_error(tmp_path: Path):
    path = tmp_path / "broken.tracking.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(TrackingProposalFormatError, match="Invalid tracking proposal JSON"):
        read_tracking_proposal(path)


def test_non_standard_non_finite_json_number_is_rejected(tmp_path: Path):
    path = tmp_path / "non-finite.tracking.json"
    path.write_text(
        '{"schema": "acetree.tracking-proposal", "schema_version": NaN}',
        encoding="utf-8",
    )

    with pytest.raises(TrackingProposalFormatError, match="non-finite number"):
        read_tracking_proposal(path)


def test_atomic_replace_failure_preserves_previous_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import acetree_py.tracking.persistence as persistence

    path = tmp_path / "embryo.tracking.json"
    path.write_text("last-good\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        write_tracking_proposal(path, _result())

    assert path.read_text(encoding="utf-8") == "last-good\n"
    assert list(tmp_path.glob(".embryo.tracking.json.*.tmp")) == []


def test_non_finite_plugin_setting_is_rejected_before_replacement(tmp_path: Path):
    path = tmp_path / "embryo.tracking.json"
    path.write_text("last-good\n", encoding="utf-8")
    result = _result(settings={"invalid": float("inf")})

    with pytest.raises(ValueError, match="Out of range float values"):
        write_tracking_proposal(path, result)

    assert path.read_text(encoding="utf-8") == "last-good\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not stable on Windows")
def test_atomic_replace_preserves_existing_mode(tmp_path: Path):
    path = tmp_path / "embryo.tracking.json"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o640)

    write_tracking_proposal(path, _result())

    assert path.stat().st_mode & 0o777 == 0o640
