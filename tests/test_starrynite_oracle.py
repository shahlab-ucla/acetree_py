"""Always-on tests for the StarryNite differential-oracle foundations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import acetree_py.tracking.starrynite.oracle.matlab_backend as matlab_backend
from acetree_py.tracking.api import Calibration
from acetree_py.tracking.api import Detection, TrackEdge
from acetree_py.tracking.starrynite.detector import legacy_dog_response
from acetree_py.tracking.starrynite.oracle.metrics import (
    compare_detections,
    compare_lineage_graphs,
    compare_lineages,
    compare_sensitivity_curves,
    compare_volumes,
    match_point_sets,
    LineageNodeState,
)
from acetree_py.tracking.starrynite.oracle.matlab_backend import (
    MatlabOracleConfig,
    MatlabOracleRun,
    MatlabStarryNiteOracle,
    full_detection_request,
    full_tracking_request,
    separable_dog_request,
)
from acetree_py.tracking.starrynite.oracle.lineage import (
    compare_lineage_snapshots,
    matlab_lineage_snapshot,
    python_lineage_snapshot,
    read_lineage_snapshot,
    write_lineage_snapshot,
)
from acetree_py.tracking.starrynite.oracle.lineage_experiment import (
    run_lineage_parity_suite,
)
from acetree_py.tracking.starrynite.oracle.synthetic import (
    default_synthetic_suite,
    lineage_synthetic_suite,
    resolution_noise_suite,
)


def test_synthetic_suite_is_deterministic_and_contains_division_truth():
    first = default_synthetic_suite(seed=7)
    second = default_synthetic_suite(seed=7)

    assert [item.name for item in first] == [
        "isolated",
        "close_pair",
        "unequal_pair",
        "boundary",
        "division",
    ]
    assert all(
        np.array_equal(left.frames_tzyx, right.frames_tzyx)
        for left, right in zip(first, second, strict=True)
    )
    division = first[-1]
    assert len(division.truth_for_frame(3)) == 1
    assert len(division.truth_for_frame(4)) == 2
    assert {item.parent_id for item in division.truth_for_frame(4)} == {"parent"}


def test_neutral_classifier_export_hash_binds_and_saves_requested_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    executable = tmp_path / "matlab.exe"
    executable.write_bytes(b"fake executable")
    starrynite_root = tmp_path / "StarryNite"
    (starrynite_root / "distribution_code").mkdir(parents=True)
    model_path = (
        starrynite_root
        / "distribution_lineaging"
        / "2019TrackingModelv2.mat"
    )
    model_path.parent.mkdir(parents=True)
    model_bytes = b"immutable classifier source"
    model_path.write_bytes(model_bytes)
    oracle = MatlabStarryNiteOracle(
        MatlabOracleConfig(executable, starrynite_root)
    )

    captured: dict[str, object] = {}
    sentinel_model = object()

    def fake_export(self, *, model_file=None):
        captured["requested_model"] = model_file
        return MatlabOracleRun(
            operation="export_classifier_model",
            result={
                "source_model": {"path": str(model_path.resolve())},
                "classifier_model": {"neutral": True},
            },
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )

    def fake_convert(exported, *, source_model_sha256):
        captured["exported"] = exported
        captured["source_hash"] = source_model_sha256
        return sentinel_model

    def fake_save(destination, model):
        captured["destination"] = destination
        captured["saved_model"] = model

    monkeypatch.setattr(
        MatlabStarryNiteOracle,
        "export_classifier_model",
        fake_export,
    )
    monkeypatch.setattr(
        matlab_backend,
        "neutral_classifier_from_matlab_export",
        fake_convert,
    )
    monkeypatch.setattr(matlab_backend, "save_neutral_classifier", fake_save)

    destination = tmp_path / "neutral" / "classifier.json"
    returned = oracle.export_neutral_classifier(destination)

    expected_hash = hashlib.sha256(model_bytes).hexdigest()
    assert captured["requested_model"] == model_path.resolve()
    assert captured["source_hash"] == expected_hash
    assert captured["destination"] == destination
    assert captured["saved_model"] is sentinel_model
    assert returned is sentinel_model
    assert model_path.read_bytes() == model_bytes


def test_neutral_classifier_export_cannot_overwrite_its_source_model(tmp_path: Path):
    executable = tmp_path / "matlab.exe"
    executable.write_bytes(b"fake executable")
    starrynite_root = tmp_path / "StarryNite"
    (starrynite_root / "distribution_code").mkdir(parents=True)
    model_path = (
        starrynite_root
        / "distribution_lineaging"
        / "2019TrackingModelv2.mat"
    )
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"source-model")
    oracle = MatlabStarryNiteOracle(
        MatlabOracleConfig(executable, starrynite_root)
    )

    with pytest.raises(ValueError, match="cannot overwrite"):
        oracle.export_neutral_classifier(model_path, model_file=model_path)

    assert model_path.read_bytes() == b"source-model"


def test_resolution_noise_suite_is_multiseed_and_includes_noise_controls():
    first = resolution_noise_suite(seed=9, seed_count=2)
    second = resolution_noise_suite(seed=9, seed_count=2)

    assert len(first) == 2 * 3 * 9
    assert [item.name for item in first] == [item.name for item in second]
    assert all(
        np.array_equal(left.frames_tzyx, right.frames_tzyx)
        for left, right in zip(first, second, strict=True)
    )
    assert sum(not item.objects for item in first) == 2 * 3
    assert all(
        len(item.truth_for_frame(1)) == 2 for item in first if item.objects
    )


def test_lineage_suite_is_deterministic_and_isolates_five_topology_behaviors():
    first = lineage_synthetic_suite(seed=12)
    second = lineage_synthetic_suite(seed=12)

    assert [item.name for item in first] == [
        "lineage-translation",
        "lineage-one-frame-gap",
        "lineage-long-division",
        "lineage-model-division",
        "lineage-transient-artifact",
    ]
    assert all(
        np.array_equal(left.frames_tzyx, right.frames_tzyx)
        for left, right in zip(first, second, strict=True)
    )
    assert [item.frame_count for item in first] == [5, 8, 11, 7, 8]
    assert len(first[1].truth_for_frame(4)) == 2
    assert len(first[1].truth_for_frame(5)) == 3
    assert len(first[2].truth_for_frame(5)) == 4
    assert len(first[2].truth_for_frame(6)) == 5
    assert {item.parent_id for item in first[2].truth_for_frame(6)} == {
        None,
        "parent",
    }
    assert len(first[3].truth_for_frame(2)) == 4
    assert len(first[3].truth_for_frame(3)) == 5
    assert {item.parent_id for item in first[3].truth_for_frame(3)} == {
        None,
        "parent",
    }
    assert len(first[4].truth_for_frame(4)) == 4
    assert len(first[4].truth_for_frame(5)) == 3


def test_volume_similarity_reports_exact_agreement_and_peak_shift():
    reference = np.zeros((5, 6, 7), dtype=np.float32)
    reference[2, 3, 4] = 5.0

    exact = compare_volumes(reference, reference.copy())
    shifted = np.roll(reference, shift=1, axis=2)
    different = compare_volumes(reference, shifted)

    assert exact.relative_l2 == 0
    assert exact.pearson_correlation == pytest.approx(1.0)
    assert exact.peak_displacement_px == 0
    assert different.peak_displacement_px == pytest.approx(1.0)
    assert different.relative_l2 > 0


def test_point_matching_maximizes_cardinality_before_distance():
    reference = np.asarray([[0.0], [1.1]])
    candidate = np.asarray([[1.0], [2.0]])

    result = match_point_sets(reference, candidate, tolerance=1.05)

    assert [(item.reference_index, item.candidate_index) for item in result.matches] == [
        (0, 0),
        (1, 1),
    ]
    assert result.unmatched_reference == ()
    assert result.unmatched_candidate == ()


def test_detection_similarity_handles_empty_sets_and_anisotropic_spacing():
    both_empty = compare_detections(
        np.empty((0, 3)),
        np.empty((0, 3)),
        tolerance=1.0,
    )
    one_shifted_z = compare_detections(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.6]]),
        tolerance=1.0,
        spacing=(1.0, 1.0, 2.0),
    )

    assert both_empty.f1 == 1.0
    assert one_shifted_z.f1 == 0.0
    assert one_shifted_z.count_delta == 0


def test_sensitivity_comparison_detects_matching_and_reversed_response():
    values = (0.5, 1.0, 1.5, 2.0)
    reference = (1.0, 2.0, 4.0, 4.0)

    exact = compare_sensitivity_curves(values, reference, reference)
    reversed_curve = compare_sensitivity_curves(values, reference, reference[::-1])

    assert exact.normalized_curve_rmse == 0
    assert exact.slope_sign_agreement == 1.0
    assert exact.transition_distance == 0
    assert reversed_curve.slope_sign_agreement < 1.0
    assert reversed_curve.normalized_area_between_curves > 0


def test_lineage_similarity_treats_daughter_order_as_irrelevant_and_half_split_as_wrong():
    reference = {("p", "a", "split"), ("p", "b", "split")}
    reordered = {("parent", "right", "split"), ("parent", "left", "split")}
    mapped = {"parent": "p", "right": "b", "left": "a"}
    half = {("p", "a", "split")}

    exact = compare_lineages(reference, reordered, candidate_to_reference=mapped)
    incomplete = compare_lineages(reference, half)

    assert exact.f1 == 1.0
    assert exact.division_f1 == 1.0
    assert incomplete.matched_edge_count == 1
    assert incomplete.matched_division_count == 0
    assert incomplete.division_f1 == 0.0


def test_lineage_graph_similarity_aligns_ids_and_exposes_cleanup_and_ancestry_errors():
    reference_nodes = (
        LineageNodeState("p0", 0, (0.0, 0.0, 0.0)),
        LineageNodeState("p1", 1, (1.0, 0.0, 0.0)),
        LineageNodeState("a", 2, (2.0, -1.0, 0.0)),
        LineageNodeState("b", 2, (2.0, 1.0, 0.0)),
        LineageNodeState("artifact", 2, (8.0, 8.0, 0.0), retained=False),
    )
    candidate_nodes = (
        LineageNodeState(10, 0, (0.01, 0.0, 0.0)),
        LineageNodeState(20, 1, (1.01, 0.0, 0.0)),
        LineageNodeState(30, 2, (2.01, 1.0, 0.0)),
        LineageNodeState(40, 2, (2.01, -1.0, 0.0)),
        # Same raw detection, but a cleanup regression retains it.
        LineageNodeState(50, 2, (8.01, 8.0, 0.0), retained=True),
    )
    reference_edges = {
        ("p0", "p1", "link"),
        ("p1", "a", "split"),
        ("p1", "b", "split"),
    }
    exact_candidate_edges = {
        (10, 20, "link"),
        (20, 30, "split"),
        (20, 40, "split"),
    }

    result = compare_lineage_graphs(
        reference_nodes,
        reference_edges,
        candidate_nodes,
        exact_candidate_edges,
        tolerance=0.1,
    )

    assert result.node_f1 == 1.0
    assert result.edge_similarity.f1 == 1.0
    assert result.edge_similarity.division_f1 == 1.0
    assert result.edge_by_kind["split"]["f1"] == 1.0
    assert result.state_accuracy == pytest.approx(4 / 5)
    assert result.retained_precision == pytest.approx(4 / 5)
    assert result.ancestry_agreement == 1.0
    assert result.candidate_to_reference[30] == "b"


def test_lineage_graph_similarity_scores_gap_kind_and_ancestry_not_only_endpoints():
    nodes = (
        LineageNodeState("a", 0, (0.0, 0.0, 0.0)),
        LineageNodeState("b", 2, (1.0, 0.0, 0.0)),
        LineageNodeState("c", 3, (2.0, 0.0, 0.0)),
    )
    candidate = tuple(
        LineageNodeState(f"candidate-{item.node_id}", item.frame, item.position)
        for item in nodes
    )
    result = compare_lineage_graphs(
        nodes,
        {("a", "b", "gap"), ("b", "c", "link")},
        candidate,
        {
            ("candidate-a", "candidate-b", "link"),
            ("candidate-a", "candidate-c", "gap"),
        },
        tolerance=0.1,
    )

    assert result.node_f1 == 1.0
    assert result.edge_similarity.f1 == 0.0
    assert result.edge_by_kind["gap"]["f1"] == 0.0
    assert result.ancestry_agreement < 1.0


def test_public_legacy_dog_stage_returns_finite_same_shape_volume():
    image = default_synthetic_suite(seed=3)[0].frames_tzyx[0]

    response = legacy_dog_response(
        image,
        radius_um=2.0,
        sigma_factor=1.0,
        calibration=Calibration(0.5, 1.0),
    )

    assert response.shape == image.shape
    assert response.dtype == np.float32
    assert np.all(np.isfinite(response))
    assert np.unravel_index(np.argmax(response), response.shape) == (8, 24, 24)


def test_matlab_requests_make_orientation_and_sweep_precedence_explicit():
    movie = default_synthetic_suite(seed=4)[0]
    image = movie.frames_tzyx[0]

    dog_request = separable_dog_request(
        image,
        radius_um=2.0,
        sigma_factor=1.0,
        calibration=movie.calibration,
    )
    detection_request = full_detection_request(
        image,
        radius_um=2.0,
        sigma_factor=0.8,
        intensity_threshold=12.0,
        boundary_percent=0.4,
        calibration=movie.calibration,
        num_cells=4,
        legacy_parameters={
            "sigma": 99.0,
            "intensitythreshold": 99.0,
            "boundary_percent": 0.99,
        },
    )

    assert dog_request["volume_yxz"].shape == (49, 49, 17)
    assert dog_request["inner_sigma_yxz"][2] < dog_request["inner_sigma_yxz"][0]
    parameters = detection_request["legacy_parameters"]
    assert parameters["sigma"] == pytest.approx([0.8])
    assert parameters["intensitythreshold"] == pytest.approx([12.0])
    assert parameters["boundary_percent"] == pytest.approx([0.4])


def test_tracking_request_and_result_normalize_movie_axes_nodes_and_edge_kinds():
    movie = default_synthetic_suite(seed=5)[-1]
    request = full_tracking_request(
        movie.frames_tzyx,
        radius_um=2.0,
        sigma_factor=1.0,
        intensity_threshold=18.0,
        boundary_percent=0.35,
        calibration=movie.calibration,
        tracking_overrides={"candidateCutoff": 1.5},
    )
    run = MatlabOracleRun(
        "full_tracking",
        {
            "edge_table": np.asarray(
                [
                    [0, 0, 1, 0, 0, 1, 0],
                    [1, 0, 3, 2, 1, 2, 0],
                    [3, 2, 4, 0, 2, 1, 1],
                ]
            )
        },
        "",
        "",
        0.1,
    )

    assert request["movie_yxzt"].shape == (49, 49, 17, 6)
    assert request["tracking_overrides"]["candidateCutoff"] == 1.5
    assert run.normalized_edges() == (
        ((0, 0), (1, 0), "link"),
        ((1, 0), (3, 2), "gap"),
    )
    assert run.normalized_edges(include_deleted_sources=True)[-1] == (
        (3, 2),
        (4, 0),
        "split",
    )


def test_matlab_python_lineage_snapshots_round_trip_and_compare_cleanup(tmp_path):
    run = MatlabOracleRun(
        "full_tracking",
        {
            "matlab_version": "test",
            "node_table": np.asarray(
                [
                    [0, 0, 10, 10, 2, 8, 100, 0],
                    [1, 0, 11, 10, 2, 8, 100, 0],
                    [1, 1, 20, 20, 2, 4, 20, 1],
                ],
                dtype=float,
            ),
            "edge_table": np.asarray(
                [[0, 0, 1, 0, 0, 1, 0]], dtype=float
            ),
        },
        "",
        "",
        0.1,
    )
    calibration = Calibration(0.5, 1.5)
    reference = matlab_lineage_snapshot(run, calibration)
    raw = (
        Detection("p", 1, 5.0, 5.0, 3.0, 2.0, 100.0),
        Detection("c", 2, 5.5, 5.0, 3.0, 2.0, 100.0),
        Detection("artifact", 2, 10.0, 10.0, 3.0, 1.0, 20.0),
    )
    candidate = python_lineage_snapshot(
        raw,
        {"p", "c"},
        (TrackEdge("p", "c", 0.0),),
    )

    comparison = compare_lineage_snapshots(
        reference, candidate, tolerance_um=0.01
    )
    destination = tmp_path / "lineage.json"
    write_lineage_snapshot(destination, reference)

    assert comparison.node_f1 == 1.0
    assert comparison.state_accuracy == 1.0
    assert comparison.retained_f1 == 1.0
    assert comparison.edge_similarity.f1 == 1.0
    assert read_lineage_snapshot(destination) == reference


def test_python_lineage_snapshot_requires_explicit_raw_retention_accounting():
    detection = Detection("only", 1, 0.0, 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="Retained IDs"):
        python_lineage_snapshot((detection,), {"missing"}, ())


def test_lineage_parity_suite_batches_matlab_startup_and_keeps_case_names():
    class EmptyGraphOracle:
        def __init__(self):
            self.batch_count = 0

        def run_many(self, requests):
            self.batch_count += 1
            return tuple(
                MatlabOracleRun(
                    "full_tracking",
                    {
                        "matlab_version": "test",
                        "node_table": np.empty((0, 8), dtype=float),
                        "edge_table": np.empty((0, 7), dtype=float),
                        "classifier_computed_classes": np.asarray(
                            [1.0, 0.0]
                        ),
                        "classifier_rounds": np.asarray([1.0, 2.0]),
                    },
                    "",
                    "",
                    0.01,
                )
                for _request in requests
            )

        def installation_provenance(self):
            return {"oracle": "fake"}

    movies = lineage_synthetic_suite(seed=21)[:2]
    oracle = EmptyGraphOracle()

    results = run_lineage_parity_suite(
        movies,
        oracle,  # type: ignore[arg-type]
        radius_um=4.0,
        sigma_factor=0.5,
        intensity_threshold=0.25,
        boundary_percent=0.35,
    )

    assert oracle.batch_count == 1
    assert [item.scenario for item in results] == [item.name for item in movies]
    assert all(item.matlab_snapshot.nodes == () for item in results)
    serialized = results[0].to_dict()
    assert serialized["matlab_snapshot"]["schema"] == (
        "acetree.starrynite.lineage-snapshot/v1"
    )
    assert serialized["python_snapshot"]["nodes"]
    assert serialized["matlab_classifier_diagnostics"] == {
        "classifier_computed_classes": [1.0, 0.0],
        "classifier_rounds": [1.0, 2.0],
    }
