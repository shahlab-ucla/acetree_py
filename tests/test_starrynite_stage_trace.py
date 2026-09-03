"""Strict decoding tests for MATLAB geometry-stage checkpoints."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.tracking import TrackEdge
from acetree_py.tracking.starrynite.legacy_early import (
    LegacyCandidateState,
    LegacyEarlyStageSnapshot,
    LegacyEarlyTrackingResult,
)
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
)
from acetree_py.tracking.starrynite.oracle import (
    EventTraceFormatError,
    MatlabOracleRun,
    compare_geometry_stage_traces,
    geometry_stage_trace_from_legacy_early,
    matlab_geometry_stage_trace,
)


def _snapshot(*, linked: bool = False) -> np.ndarray:
    if linked:
        return np.asarray(
            [
                [0, 0, 0, -1, -1, 1, 0, -1, -1],
                [1, 0, 0, 0, 0, -1, -1, -1, -1],
            ],
            dtype=float,
        )
    return np.asarray(
        [
            [0, 0, 0, -1, -1, -1, -1, -1, -1],
            [1, 0, 0, -1, -1, -1, -1, -1, -1],
        ],
        dtype=float,
    )


def _run() -> MatlabOracleRun:
    labels = np.asarray(
        [
            "detected",
            "initialized",
            "easy_links",
            "post_polar_filter",
            "candidates",
            "geometry_final",
        ],
        dtype=object,
    )
    snapshots = np.empty(6, dtype=object)
    snapshots[:5] = [_snapshot() for _ in range(5)]
    snapshots[5] = _snapshot(linked=True)
    candidates = np.empty(6, dtype=object)
    candidates[:4] = [np.empty((0, 5)) for _ in range(4)]
    candidates[4:] = [
        np.asarray([[0, 0, 0, 1, 0], [1, 0, 0, 1, 0]], dtype=float)
        for _ in range(2)
    ]
    trace = {
        "schema_version": 1,
        "finished": True,
        "stage_count": 6,
        "snapshot_columns": np.asarray(
            [
                "frame_0based",
                "node_0based",
                "deleted",
                "predecessor_frame_0based",
                "predecessor_node_0based",
                "successor1_frame_0based",
                "successor1_node_0based",
                "successor2_frame_0based",
                "successor2_node_0based",
            ],
            dtype=object,
        ),
        "candidate_columns": np.asarray(
            [
                "direction_code",
                "source_frame_0based",
                "source_node_0based",
                "target_frame_0based",
                "target_node_0based",
            ],
            dtype=object,
        ),
        "stage_labels": labels,
        "stage_thresholds": np.full(6, np.nan),
        "stage_snapshots": snapshots,
        "stage_candidate_tables": candidates,
        "stage_node_counts": np.full(6, 2),
        "stage_active_counts": np.full(6, 2),
        "stage_deleted_counts": np.zeros(6),
        "stage_edge_counts": np.asarray([0, 0, 0, 0, 0, 1]),
        "stage_candidate_counts": np.asarray([0, 0, 0, 0, 2, 2]),
    }
    return MatlabOracleRun(
        "full_tracking",
        {
            "matlab_version": "test",
            "starrynite_root": "external",
            "upstream_function": "tracking_driver.m",
            "tracking_stage_trace": trace,
        },
        "",
        "",
        0.0,
    )


def test_matlab_geometry_stage_trace_decodes_candidates_and_raw_links() -> None:
    trace = matlab_geometry_stage_trace(_run())

    assert [item.label for item in trace.stages] == [
        "detected",
        "initialized",
        "easy_links",
        "post_polar_filter",
        "candidates",
        "geometry_final",
    ]
    assert trace.stages[4].candidate_pairs == (("matlab:0:0", "matlab:1:0"),)
    assert trace.stages[4].backward_candidate_pairs == (
        ("matlab:0:0", "matlab:1:0"),
    )
    assert trace.final.edge_count == 1
    assert trace.final.by_id["matlab:0:0"].successor_slots == (
        "matlab:1:0",
        None,
    )


def test_matlab_geometry_stage_trace_rejects_inconsistent_summaries() -> None:
    run = _run()
    run.result["tracking_stage_trace"]["stage_edge_counts"] = np.zeros(6)

    with pytest.raises(EventTraceFormatError, match="edge count"):
        matlab_geometry_stage_trace(run)


def test_matlab_geometry_stage_trace_rejects_changed_row_identity() -> None:
    run = _run()
    snapshots = run.result["tracking_stage_trace"]["stage_snapshots"]
    changed = snapshots[5].copy()
    changed[1, 1] = 1
    snapshots[5] = changed

    with pytest.raises(EventTraceFormatError, match="known detector rows|identity"):
        matlab_geometry_stage_trace(run)


def test_python_early_stages_normalize_to_matlab_rows_and_candidate_directions() -> None:
    feature_parameters = LegacyFeatureParameters(
        interval=1.0,
        candidate_cutoff=2.0,
        temporal_cutoff=3,
        temporal_cutoff_start=2,
        small_cutoff=4.0,
        anisotropy_xyz=(1.0, 1.0, 1.0),
        end_frame=2,
    )
    nuclei = (
        LegacyNucleus("a", 1, 0, (0.0, 0.0, 0.0), 4.0, 1, 1, 1, 0, 1, 1),
        LegacyNucleus("b", 2, 0, (1.0, 0.0, 0.0), 4.0, 1, 1, 1, 0, 1, 1),
    )
    initial = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei, (), feature_parameters
    )
    linked = LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        (
            TrackEdge(
                "a",
                "b",
                0.0,
                "link",
                {"LEGACY_SUCCESSOR_SLOT": 0},
            ),
        ),
        feature_parameters,
    )
    empty = LegacyEarlyStageSnapshot("initialized", None, (), (), (), (), 0, 0)
    easy = LegacyEarlyStageSnapshot(
        "easy_links", None, (("a", 0, "b"),), (), (), (), 1, 0
    )
    post_polar = LegacyEarlyStageSnapshot(
        "post_polar_filter", None, easy.pointers, (), (), (), 1, 0
    )
    candidates = LegacyEarlyStageSnapshot(
        "candidates",
        None,
        (("a", 0, "b"),),
        (),
        (("a", "b"),),
        (("a", "b"),),
        1,
        0,
    )
    nondivision = LegacyEarlyStageSnapshot(
        "nondivision",
        0.5,
        candidates.pointers,
        (),
        candidates.forward_candidates,
        candidates.backward_candidates,
        1,
        0,
    )
    division = LegacyEarlyStageSnapshot(
        "division",
        np.inf,
        candidates.pointers,
        (),
        candidates.forward_candidates,
        candidates.backward_candidates,
        1,
        0,
    )
    final = LegacyEarlyStageSnapshot(
        "geometry_final",
        None,
        candidates.pointers,
        (),
        candidates.forward_candidates,
        candidates.backward_candidates,
        1,
        0,
    )
    result = LegacyEarlyTrackingResult(
        linked,
        LegacyCandidateState({"a": ("b",), "b": ()}, {"a": (), "b": ("a",)}),
        (empty, easy, post_polar, candidates, nondivision, division, final),
    )

    trace = geometry_stage_trace_from_legacy_early(initial, result)

    assert [item.label for item in trace.stages] == [
        "detected",
        "initialized",
        "easy_links",
        "post_polar_filter",
        "candidates",
        "nondivision",
        "division",
        "geometry_final",
    ]
    assert trace.stages[4].forward_candidate_pairs == (
        ("matlab:0:0", "matlab:1:0"),
    )
    assert trace.stages[4].backward_candidate_pairs == (
        ("matlab:0:0", "matlab:1:0"),
    )
    assert compare_geometry_stage_traces(trace, trace).matched
