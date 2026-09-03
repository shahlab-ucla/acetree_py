"""Optional live checks against a user-supplied MATLAB StarryNite checkout."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

from acetree_py.io.image_provider import NumpyProvider
from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    TrackingRequest,
    TrackingScope,
)
from acetree_py.tracking.pipeline import TrackingPipeline
from acetree_py.tracking.registry import build_default_registry
from acetree_py.tracking.starrynite.classifier import (
    GaussianKernelFeatureDistribution,
    SingleModelFeatureInput,
    classify_single_model,
    neutral_classifier_from_matlab_export,
    save_neutral_classifier,
)
from acetree_py.tracking.starrynite.detector import (
    StarryNiteDetector,
    _canonical_maxima,
    legacy_dog_response,
    legacy_radial_geometry,
)
from acetree_py.tracking.starrynite import (
    LEGACY_EXACT_REFINEMENT_BACKEND,
    LegacyMovieDecisionConfig,
    LegacyTrackingStatistics,
    build_legacy_region_table,
    load_tuning_profile,
    load_matlab_model,
    parse_parameter_text,
    resolve_legacy_parameter,
    run_legacy_movie_decisions,
    sha256_file,
)
from acetree_py.tracking.starrynite.legacy_state import legacy_single_round
from acetree_py.tracking.starrynite.legacy_early import (
    LegacyEarlyTrackingParameters,
    run_legacy_early_tracking,
)
from acetree_py.tracking.starrynite.legacy_state import LegacyTrackingContext
from acetree_py.tracking.starrynite.oracle.matlab_backend import (
    MatlabOracleConfig,
    MatlabOracleRun,
    MatlabStarryNiteOracle,
    _float32_ulp_distances,
    classifier_model_export_request,
    classifier_prediction_request,
    full_detection_request,
    full_tracking_request,
    parameter_resolution_request,
    separable_dog_request,
    slice_candidates_request,
)
from acetree_py.tracking.starrynite.oracle.metrics import compare_detections, compare_volumes
from acetree_py.tracking.starrynite.oracle.lineage_experiment import (
    run_lineage_parity_case,
)
from acetree_py.tracking.starrynite.oracle.lineage import (
    compare_lineage_snapshots,
    matlab_lineage_snapshot,
    python_lineage_snapshot,
)
from acetree_py.tracking.starrynite.oracle.event_trace import (
    assert_event_trace_parity,
    legacy_context_from_checkpoint,
    trace_from_legacy_movie_result,
)
from acetree_py.tracking.starrynite.oracle.stage_trace import (
    assert_geometry_stage_parity,
    geometry_stage_trace_from_legacy_early,
)
from acetree_py.tracking.starrynite.oracle.synthetic import (
    SyntheticObject,
    default_synthetic_suite,
    lineage_synthetic_suite,
    render_movie,
)


pytestmark = pytest.mark.matlab_oracle


def test_float32_ulp_metric_handles_adjacent_signs_and_nan_pairs() -> None:
    values = np.asarray([0.0, 1.0, -1.0, np.nan], dtype=np.float32)
    adjacent = np.asarray(
        [
            np.nextafter(values[0], np.float32(1.0)),
            np.nextafter(values[1], np.float32(2.0)),
            np.nextafter(values[2], np.float32(-2.0)),
            np.nan,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        _float32_ulp_distances(values, adjacent),
        [1, 1, 1, 0],
    )


def test_oracle_run_rehydrates_legacy_state_and_successor_slots() -> None:
    confidence = (
        (0.5, 0.0, legacy_single_round(math.log(2.0)), 1.0, 1.0, 0.5),
        (1.5, 0.0, legacy_single_round(math.log(2.0)), 1.0, 1.0, 1.5),
    )
    rows = []
    for frame, x_values in enumerate(((0.0, 10.0), (1.0, 11.0))):
        for row, x_value in enumerate(x_values):
            rows.append(
                (
                    frame,
                    row,
                    x_value,
                    0.0,
                    0.0,
                    4.0,
                    (10.0, 30.0)[row],
                    (2.0, 6.0)[row],
                    1.0,
                    4.0,
                    2.0,
                    8.0,
                    2.0,
                    10.0,
                    *confidence[row],
                    float(frame == 1 and row == 1),
                )
            )
    run = MatlabOracleRun(
        operation="full_tracking",
        result={
            "legacy_node_measurements": np.asarray(rows, dtype=float),
            # Edge order is MATLAB suc slot order: higher-row daughter first.
            "edge_table": np.asarray(
                (
                    (0, 0, 1, 1, 2, 1, 0),
                    (0, 0, 1, 0, 2, 1, 0),
                ),
                dtype=float,
            ),
            "tracking_parameter_summary": {
                "interval": 1.0,
                "candidateCutoff": 1.2,
                "temporalcutoff": 2.0,
                "temporalcutoffstart": 2.0,
                "smallcutoff": 4.0,
                "endtime": 2.0,
                "anisotropyvector": np.asarray((1.0, 1.0, 1.0)),
            },
        },
        stdout="",
        stderr="",
        duration_seconds=0.0,
    )

    context = run.legacy_tracking_context()

    assert context.successor_slots("matlab:0:0") == (
        "matlab:1:1",
        "matlab:1:0",
    )
    assert context.deleted_ids == frozenset({"matlab:1:1"})
    assert context.self_distance("matlab:0:0") == pytest.approx(10.0)


def _classifier_request_for_topology(
    topology_class: int,
    *,
    force_mode: bool = False,
) -> dict[str, object]:
    topology = {
        # d1, d2, back1, back2, forward1, forward2
        1: (2.0, 5.0, 2.0, -1.0, 3.0, -1.0),
        2: (5.0, 5.0, -1.0, -1.0, -1.0, -1.0),
        3: (5.0, 5.0, 2.0, -1.0, -1.0, -1.0),
        4: (2.0, 5.0, -1.0, -1.0, -1.0, -1.0),
        5: (2.0, 5.0, 2.0, -1.0, -1.0, -1.0),
    }[topology_class]
    d1, d2, back1, back2, forward1, forward2 = topology
    return classifier_prediction_request(
        np.ones(22, dtype=float),
        np.ones(11, dtype=float),
        np.ones(13, dtype=float),
        d1_length=d1,
        d2_length=d2,
        fn_back_candidate_1_length=back1,
        fn_back_candidate_2_length=back2,
        best_fn_forward_length_d1=forward1,
        best_fn_forward_length_d2=forward2,
        best_fn_back_correct=False,
        best_index=1,
        force_mode=force_mode,
    )


def _classifier_request_for_division_fp() -> dict[str, object]:
    return classifier_prediction_request(
        np.ones(22, dtype=float),
        np.ones(11, dtype=float),
        np.ones(13, dtype=float),
        d1_length=2.0,
        d2_length=5.0,
        fn_back_candidate_1_length=-1.0,
        fn_back_candidate_2_length=-1.0,
        best_fn_forward_length_d1=3.0,
        best_fn_forward_length_d2=-1.0,
        best_fn_back_correct=False,
        best_index=1,
        force_mode=False,
    )


def _python_classifier_prediction(model, request):
    inputs = SingleModelFeatureInput(
        daughter_features=tuple(np.asarray(request["daughter_data"]).reshape(-1)),
        backward_features=tuple(np.asarray(request["back_data"]).reshape(-1)),
        forward_features=tuple(np.asarray(request["forward_data"]).reshape(-1)),
        daughter_lengths=(
            float(request["d1_length"]),
            float(request["d2_length"]),
        ),
        backward_candidate_present=(
            float(request["fn_back_candidate_1_length"]) > 0,
            float(request["fn_back_candidate_2_length"]) > 0,
        ),
        best_forward_lengths=(
            float(request["best_fn_forward_length_d1"]),
            float(request["best_fn_forward_length_d2"]),
        ),
        small_cutoff=4.0,
    )
    return classify_single_model(
        model,
        inputs,
        force_mode=bool(request["force_mode"]),
        backward_repair_available=int(request["best_index"]) != -1,
    )


def _perturbed_classifier_request(
    request: dict[str, object],
    block_name: str,
    feature_index: int,
    delta: float,
) -> dict[str, object]:
    blocks = {
        "daughter_data": np.asarray(request["daughter_data"], dtype=float).copy(),
        "back_data": np.asarray(request["back_data"], dtype=float).copy(),
        "forward_data": np.asarray(request["forward_data"], dtype=float).copy(),
    }
    blocks[block_name].reshape(-1)[feature_index] += delta
    return classifier_prediction_request(
        blocks["daughter_data"],
        blocks["back_data"],
        blocks["forward_data"],
        d1_length=float(request["d1_length"]),
        d2_length=float(request["d2_length"]),
        fn_back_candidate_1_length=float(
            request["fn_back_candidate_1_length"]
        ),
        fn_back_candidate_2_length=float(
            request["fn_back_candidate_2_length"]
        ),
        best_fn_forward_length_d1=float(
            request["best_fn_forward_length_d1"]
        ),
        best_fn_forward_length_d2=float(
            request["best_fn_forward_length_d2"]
        ),
        best_fn_back_correct=bool(request["best_fn_back_correct"]),
        best_index=int(request["best_index"]),
        force_mode=bool(request["force_mode"]),
    )


def _classifier_request_from_selected_features(
    model,
    selected_features,
    *,
    model_file: Path,
) -> dict[str, object]:
    selected = iter(selected_features)
    blocks: dict[str, np.ndarray] = {
        "daughter_data": np.zeros(22, dtype=float),
        "back_data": np.zeros(11, dtype=float),
        "forward_data": np.zeros(13, dtype=float),
    }
    masks = (
        ("daughter_data", model.feature_layout.daughter_keep),
        ("back_data", model.feature_layout.backward_keep),
        ("forward_data", model.feature_layout.forward_keep),
    )
    for block_name, mask in masks:
        for feature_index, keep in enumerate(mask):
            if keep:
                blocks[block_name][feature_index] = next(selected)
    try:
        next(selected)
    except StopIteration:
        pass
    else:
        raise AssertionError("Too many selected classifier features for model masks")
    return classifier_prediction_request(
        blocks["daughter_data"],
        blocks["back_data"],
        blocks["forward_data"],
        d1_length=2.0,
        d2_length=5.0,
        fn_back_candidate_1_length=2.0,
        fn_back_candidate_2_length=-1.0,
        best_fn_forward_length_d1=3.0,
        best_fn_forward_length_d2=-1.0,
        best_fn_back_correct=False,
        best_index=1,
        force_mode=False,
        model_file=model_file,
    )


def _live_config() -> MatlabOracleConfig:
    root = os.environ.get("STARRYNITE_ROOT")
    executable = os.environ.get("MATLAB_EXECUTABLE")
    if not root or not executable:
        pytest.skip("Set STARRYNITE_ROOT and MATLAB_EXECUTABLE for live oracle checks")
    return MatlabOracleConfig(Path(executable), Path(root), timeout_seconds=600)


def _write_live_exact_parameter_file(
    destination: Path,
    *,
    model_file: Path,
    distribution_file: Path,
    initial_cell_count: int,
    initial_diameter_xy_px: float = 8.0,
    staging: tuple[int, ...] = (1_000_000_000,),
) -> Path:
    """Write the inert parameter source shared by both live movie engines."""

    parameter_path = destination / "synthetic_starrynite_parameters.m"
    staging_text = ",".join(str(value) for value in staging)
    parameter_path.write_text(
        "xyres=1;\n"
        "zres=3;\n"
        f"firsttimestepdiam={initial_diameter_xy_px:g};\n"
        f"firsttimestepnumcells={initial_cell_count};\n"
        "downsampling=1;\n"
        f"distribution_file='{distribution_file.as_posix()}';\n"
        f"parameters.staging=[{staging_text}];\n"
        "parameters.sigma=.5;\n"
        "parameters.intensitythreshold=.25;\n"
        "parameters.rangethreshold=100;\n"
        "parameters.selection_dist=.5;\n"
        "parameters.boundary_percent=.35;\n"
        "parameters.large_ray_threshold=1.5;\n"
        "parameters.small_ray_threshold=.3333333333333333;\n"
        "parameters.mergelower=-300;\n"
        "parameters.mergesplit=1;\n"
        "parameters.split=100;\n"
        "parameters.nndist_merge=.8;\n"
        "parameters.armerge=1.6;\n"
        f"load '{model_file.as_posix()}';\n"
        "trackingparameters.nonDivCostFunction=@distanceCostFunction;\n"
        "trackingparameters.DivCostFunction=@divScoreModelCostFunction;\n",
        encoding="utf-8",
    )
    return parameter_path


def _assert_live_early_geometry_parity(
    tracking_run: MatlabOracleRun,
    config: MatlabOracleConfig,
):
    """Replay detector rows through Python and require every MATLAB stage."""

    model_path = (
        config.starrynite_root
        / "distribution_lineaging"
        / "2019TrackingModelv2.mat"
    )
    model = load_matlab_model(model_path)
    measurement_template = tracking_run.legacy_tracking_context()
    initial_context = LegacyTrackingContext.from_nuclei_and_edges(
        measurement_template.nuclei,
        (),
        measurement_template.parameters,
    )
    early_parameters = LegacyEarlyTrackingParameters.from_model(
        model,
        end_frame=measurement_template.parameters.end_frame,
        nondivision_cost="distanceCostFunction",
        division_cost="divScoreModelCostFunction",
    )
    early_result = run_legacy_early_tracking(
        initial_context,
        early_parameters,
        LegacyTrackingStatistics.from_model(model),
    )
    assert_geometry_stage_parity(
        tracking_run.tracking_stage_trace(),
        geometry_stage_trace_from_legacy_early(initial_context, early_result),
    )
    return initial_context, early_result


def _live_decision_config(model, *, end_frame: int) -> LegacyMovieDecisionConfig:
    early = LegacyEarlyTrackingParameters.from_model(
        model,
        end_frame=end_frame,
        nondivision_cost="distanceCostFunction",
        division_cost="divScoreModelCostFunction",
    )
    return LegacyMovieDecisionConfig.from_early_tracking_parameters(early)


def _post_greedy_class_two_and_three_movies():
    """Return deterministic image fixtures selected by a bounded live sweep.

    The class-3 scene uses a one-frame daughter (below the model's four-frame
    ``smallcutoff``) that is attached to a real parent, so it survives the
    isolated-fragment prepass.  The class-2 scene leaves track B absent for
    one frame and keeps its endpoint within the model's 2--6-frame backward
    candidate window.  The 8.5-pixel cross-track offset is resolved by the
    detector while remaining close enough for the greedy false split.
    """

    calibration = Calibration(1.0, 3.0)
    shape_zyx = (13, 49, 65)
    sigma_zyx = (2.2 / 3.0, 2.2, 2.2)
    controls_xyz = (
        (10.0, 9.0, 5.0),
        (32.0, 9.0, 6.0),
        (54.0, 9.0, 7.0),
    )

    def add_controls(objects, frame):
        for index, (x, y, z) in enumerate(controls_xyz):
            objects.append(
                SyntheticObject(
                    f"control-{index}",
                    frame,
                    (z - 1.0, y - 1.0, x - 1.0 + 0.2 * (frame - 1)),
                    sigma_zyx,
                    110.0,
                )
            )

    false_positive_objects = []
    daughter_sigma = tuple(value * 0.95 for value in sigma_zyx)
    for frame in range(1, 10):
        add_controls(false_positive_objects, frame)
        if frame <= 3:
            false_positive_objects.append(
                SyntheticObject(
                    "parent",
                    frame,
                    (6.0, 32.0, 31.0 + 0.2 * (frame - 1)),
                    sigma_zyx,
                    110.0,
                )
            )
            continue
        center_x = 31.0 + 0.2 * (frame - 1)
        false_positive_objects.append(
            SyntheticObject(
                "long-branch",
                frame,
                (6.0, 32.0, center_x - 5.0),
                daughter_sigma,
                99.0,
                parent_id="parent" if frame == 4 else None,
            )
        )
        if frame == 4:
            false_positive_objects.append(
                SyntheticObject(
                    "one-frame-branch",
                    frame,
                    (6.0, 32.0, center_x + 5.0),
                    daughter_sigma,
                    99.0,
                    parent_id="parent",
                )
            )
    false_positive = render_movie(
        "lineage-model-class-3-short-branch",
        shape_zyx,
        false_positive_objects,
        calibration=calibration,
        expected_radius_um=4.0,
        background=0.0,
        description="Attached one-frame branch classified as legacy class 3.",
    )

    false_negative_objects = []
    post_gap_sigma = tuple(value * 0.92 for value in sigma_zyx)
    for frame in range(1, 10):
        add_controls(false_negative_objects, frame)
        false_negative_objects.append(
            SyntheticObject(
                "track-a",
                frame,
                (6.0, 28.0, 27.0 + 0.7 * (frame - 1)),
                sigma_zyx,
                110.0,
            )
        )
        if frame == 4:
            continue
        false_negative_objects.append(
            SyntheticObject(
                "track-b",
                frame,
                (6.0, 36.5, 35.0 - 0.7 * (frame - 1)),
                sigma_zyx if frame < 4 else post_gap_sigma,
                110.0 if frame < 4 else 75.0,
            )
        )
    false_negative = render_movie(
        "lineage-model-class-2-missed-crossing",
        shape_zyx,
        false_negative_objects,
        calibration=calibration,
        expected_radius_um=4.0,
        background=0.0,
        description="Near crossing with one missed observation and backward repair.",
    )
    return false_positive, false_negative


def test_classifier_prediction_request_rejects_invalid_feature_contracts():
    common = {
        "d1_length": 5.0,
        "d2_length": 5.0,
        "fn_back_candidate_1_length": -1.0,
        "fn_back_candidate_2_length": -1.0,
        "best_fn_forward_length_d1": -1.0,
        "best_fn_forward_length_d2": -1.0,
    }
    with pytest.raises(ValueError, match="daughter_data.*22"):
        classifier_prediction_request(
            np.ones(21), np.ones(11), np.ones(13), **common
        )
    invalid_back = np.ones(11)
    invalid_back[3] = np.inf
    with pytest.raises(ValueError, match="back_data.*infinite"):
        classifier_prediction_request(
            np.ones(22), invalid_back, np.ones(13), **common
        )
    with pytest.raises(ValueError, match="force_mode.*boolean"):
        classifier_prediction_request(
            np.ones(22),
            np.ones(11),
            np.ones(13),
            force_mode=1,  # type: ignore[arg-type]
            **common,
        )


def test_parameter_resolution_request_is_strict_and_non_executable() -> None:
    request = parameter_resolution_request(
        "rangethreshold",
        (25, 80, 181),
        (100, 20, 10, 5),
        cell_count=80,
        location_xyz=(40, 30, 6),
        regional_stage_index=2,
        regional_area=(20, 40, 0, 30, 2, 6),
        regional_value=16,
    )
    assert request["operation"] == "resolve_parameter"
    assert np.asarray(request["regional_value"]).tolist() == [16.0]

    with pytest.raises(ValueError, match="direct MATLAB struct field"):
        parameter_resolution_request(
            "parameters.rangethreshold",
            (25,),
            (1, 2),
            cell_count=0,
            location_xyz=(0, 0, 0),
        )
    with pytest.raises(ValueError, match="supplied together"):
        parameter_resolution_request(
            "split",
            (25,),
            (1, 2),
            cell_count=0,
            location_xyz=(0, 0, 0),
            regional_stage_index=1,
        )
    with pytest.raises(ValueError, match="lower bound"):
        parameter_resolution_request(
            "split",
            (25,),
            (1, 2),
            cell_count=0,
            location_xyz=(0, 0, 0),
            regional_stage_index=1,
            regional_area=(1, 1, 0, 2, 0, 2),
            regional_value=3,
        )


def test_live_getparameter_matches_python_staging_and_region_boundaries() -> None:
    config = _live_config()
    oracle = MatlabStarryNiteOracle(config)
    parameters = parse_parameter_text(
        "parameters.staging=[25,80,181];\n"
        "parameters.rangethreshold=[100,20,10,5];\n"
        "parameters.regions=cell(4,1);\n"
        "parameters.regions{2}.rangethreshold=16;\n"
        "parameters.regions{2}.area=[20,40,0,30,2,6];\n"
    )
    region_table = build_legacy_region_table(parameters)
    cases = (
        (25, (21, 1, 3)),
        (26, (21, 1, 3)),
        (80, (20, 1, 3)),
        (80, (21, 1, 3)),
        (80, (40, 30, 6)),
        (80, (40.00001, 30, 6)),
        (81, (21, 1, 3)),
    )
    requests = tuple(
        parameter_resolution_request(
            "rangethreshold",
            (25, 80, 181),
            (100, 20, 10, 5),
            cell_count=cell_count,
            location_xyz=location,
            regional_stage_index=2,
            regional_area=(20, 40, 0, 30, 2, 6),
            regional_value=16,
        )
        for cell_count, location in cases
    )

    runs = oracle.run_many(requests)

    for run, (cell_count, location) in zip(runs, cases, strict=True):
        expected = resolve_legacy_parameter(
            parameters,
            "rangethreshold",
            cell_count=cell_count,
            location=location,
            region_table=region_table,
        )
        actual = float(np.asarray(run.result["resolved_parameter"]).reshape(-1)[0])
        assert actual == pytest.approx(float(expected.value))


def test_live_classifier_export_covers_normal_and_kernel_models():
    config = _live_config()
    alternate_model = (
        config.starrynite_root
        / "example_parameter_files"
        / "newmatlab"
        / "gaussianlatedispimmodel_withoptimizedfeatures_ignoringFPstillpoorFN.mat"
    )
    if not alternate_model.is_file():
        pytest.skip("Bundled alternate diSPIM classifier model was not found")
    oracle = MatlabStarryNiteOracle(config)

    normal_run, kernel_run = oracle.run_many(
        (
            classifier_model_export_request(),
            classifier_model_export_request(model_file=alternate_model),
        )
    )

    normal = normal_run.result["classifier_model"]
    assert normal["matlab_class"] == "ClassificationNaiveBayes"
    assert normal["score_transform"] == "none"
    assert normal["standardization_state"] == "none"
    assert np.asarray(normal["mu"]).size == 0
    assert np.asarray(normal["sigma"]).size == 0
    assert np.asarray(normal["class_names"]).reshape(-1) == pytest.approx(
        [0, 1, 2, 3]
    )
    assert int(normal["selected_feature_count"]) == 21
    assert np.asarray(normal["prior"]).reshape(-1).sum() == pytest.approx(1.0)
    assert list(np.asarray(normal["distribution_names"]).reshape(-1)) == [
        "mvmn",
        *("normal" for _ in range(20)),
    ]
    assert np.asarray(normal["width"]).shape == (4, 21)

    kernel = kernel_run.result["classifier_model"]
    assert kernel["matlab_class"] == "ClassificationNaiveBayes"
    assert kernel["score_transform"] == "none"
    assert kernel["standardization_state"] == "none"
    assert int(kernel["selected_feature_count"]) == 37
    kernel_names = list(np.asarray(kernel["distribution_names"]).reshape(-1))
    assert kernel_names == ["mvmn", *("kernel" for _ in range(36))]
    assert np.asarray(kernel["width"]).shape == (4, 37)
    entries = np.asarray(kernel["distributions"], dtype=object)
    assert entries.shape == (4, 37)
    first_kernel = entries[0, 1]
    assert float(first_kernel["bandwidth"]) > 0
    input_data = np.asarray(first_kernel["input_data"]).reshape(-1)
    input_frequency = np.asarray(first_kernel["input_frequency"]).reshape(-1)
    assert len(input_data) > 0
    assert len(input_frequency) == len(input_data)


def test_live_kernel_classifier_matches_matlab_at_centers_missing_and_far_tail():
    config = _live_config()
    alternate_model = (
        config.starrynite_root
        / "example_parameter_files"
        / "newmatlab"
        / "gaussianlatedispimmodel_withoptimizedfeatures_ignoringFPstillpoorFN.mat"
    )
    if not alternate_model.is_file():
        pytest.skip("Bundled alternate diSPIM classifier model was not found")
    oracle = MatlabStarryNiteOracle(config)
    export_run = oracle.export_classifier_model(model_file=alternate_model)
    source_hash = hashlib.sha256(alternate_model.read_bytes()).hexdigest()
    python_model = neutral_classifier_from_matlab_export(
        export_run.result,
        source_model_sha256=source_hash,
    )
    kernels = python_model.distributions[1:]
    assert len(kernels) == 36
    assert all(
        isinstance(distribution, GaussianKernelFeatureDistribution)
        for distribution in kernels
    )

    center = tuple(distribution.samples[1][0] for distribution in kernels)
    mixture = tuple(
        (distribution.samples[1][0] + distribution.samples[1][-1]) / 2.0
        for distribution in kernels
    )
    missing = list(center)
    for index in range(0, len(missing), 5):
        missing[index] = np.nan
    far_tail = (1e12,) * len(kernels)
    case_values = (center, mixture, tuple(missing), far_tail)
    requests = tuple(
        _classifier_request_from_selected_features(
            python_model,
            values,
            model_file=alternate_model,
        )
        for values in case_values
    )
    runs = oracle.run_many(requests)

    maximum_posterior_error = 0.0
    for case_name, request, run in zip(
        ("sample-center", "mixture", "missing", "far-tail"),
        requests,
        runs,
        strict=True,
    ):
        python = _python_classifier_prediction(python_model, request)
        matlab_features = np.asarray(run.result["classifier_input"]).reshape(-1)
        matlab_posterior = np.asarray(run.result["posterior_scores"]).reshape(-1)
        np.testing.assert_allclose(
            python.features,
            matlab_features,
            rtol=0,
            atol=0,
            equal_nan=True,
            err_msg=f"kernel feature drift for {case_name}",
        )
        maximum_posterior_error = max(
            maximum_posterior_error,
            float(np.max(np.abs(np.asarray(python.posterior) - matlab_posterior))),
        )
        np.testing.assert_allclose(
            python.posterior,
            matlab_posterior,
            rtol=3e-6,
            atol=3e-8,
            err_msg=f"kernel posterior drift for {case_name}",
        )
        assert python.predicted_class == int(run.result["predicted_class"])
        assert int(run.result["predicted_class"]) == int(
            run.result["direct_predicted_class"]
        )

    np.testing.assert_allclose(
        runs[-1].result["posterior_scores"],
        python_model.class_priors,
        rtol=0,
        atol=1e-15,
    )
    print(
        "kernel classifier parity: "
        f"cases={len(runs)}, "
        f"max_posterior_abs_error={maximum_posterior_error:.3g}"
    )


def test_live_bifurcation_classifier_entrypoint_covers_topology_and_force_mode():
    config = _live_config()
    oracle = MatlabStarryNiteOracle(config)
    requests = tuple(
        _classifier_request_for_topology(value) for value in range(1, 6)
    )
    requests += (
        _classifier_request_for_division_fp(),
        _classifier_request_for_topology(1, force_mode=True),
    )

    export_run, *runs = oracle.run_many(
        (classifier_model_export_request(), *requests)
    )
    source_hash = oracle.installation_provenance()[
        "upstream_tracking_model_sha256"
    ]
    python_model = neutral_classifier_from_matlab_export(
        export_run.result,
        source_model_sha256=source_hash,
    )

    expected_topologies = (1, 2, 3, 4, 5, 5)
    for expected_topology, request, run in zip(
        expected_topologies, requests[:6], runs[:6], strict=True
    ):
        result = run.result
        python = _python_classifier_prediction(python_model, request)
        assert int(result["topology_class"]) == expected_topology
        assert python.topology_class == expected_topology
        classifier_input = np.asarray(result["classifier_input"]).reshape(-1)
        assert classifier_input.shape == (21,)
        np.testing.assert_allclose(
            python.features,
            classifier_input,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        classes = np.asarray(result["posterior_class_names"]).reshape(-1)
        scores = np.asarray(result["posterior_scores"]).reshape(-1)
        assert classes == pytest.approx([0, 1, 2, 3])
        assert np.all(np.isfinite(scores))
        assert scores.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(
            python.posterior,
            scores,
            rtol=2e-6,
            atol=2e-8,
        )
        assert python.predicted_class == int(result["predicted_class"])
        assert int(result["predicted_class"]) == int(result["direct_predicted_class"])
        assert int(result["computed_class"]) == int(result["predicted_class"])
        assert int(result["reference_class"]) == int(result["predicted_class"])

    assert bool(runs[4].result["topology_flags"]["dirty_false_positive_looking"])
    assert bool(
        runs[5].result["topology_flags"]["division_false_positive_looking"]
    )

    forced = runs[-1].result
    python_forced = _python_classifier_prediction(python_model, requests[-1])
    classes = np.asarray(forced["posterior_class_names"]).reshape(-1)
    scores = np.asarray(forced["posterior_scores"]).reshape(-1).copy()
    np.testing.assert_allclose(
        python_forced.features,
        np.asarray(forced["classifier_input"]).reshape(-1),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        python_forced.posterior,
        scores,
        rtol=2e-6,
        atol=2e-8,
    )
    scores[np.isin(classes, [0, 3])] = 0.0
    expected_forced = int(classes[int(np.argmax(scores))])
    assert bool(forced["force_mode"])
    assert python_forced.predicted_class == expected_forced
    assert int(forced["computed_class"]) == expected_forced
    assert int(forced["predicted_class"]) == expected_forced
    assert int(forced["reference_class"]) == expected_forced


def test_live_classifier_parameter_sensitivity_matches_matlab_feature_by_feature():
    oracle = MatlabStarryNiteOracle(_live_config())
    export_run = oracle.export_classifier_model()
    source_hash = oracle.installation_provenance()[
        "upstream_tracking_model_sha256"
    ]
    python_model = neutral_classifier_from_matlab_export(
        export_run.result,
        source_model_sha256=source_hash,
    )

    baseline = _classifier_request_for_topology(1)
    requests = [baseline]
    perturbations: list[tuple[str, int, float]] = [("baseline", -1, 0.0)]
    block_masks = (
        ("daughter_data", python_model.feature_layout.daughter_keep),
        ("back_data", python_model.feature_layout.backward_keep),
        ("forward_data", python_model.feature_layout.forward_keep),
    )
    for block_name, mask in block_masks:
        for feature_index, selected in enumerate(mask):
            if not selected:
                continue
            for delta in (-0.05, 0.05):
                requests.append(
                    _perturbed_classifier_request(
                        baseline,
                        block_name,
                        feature_index,
                        delta,
                    )
                )
                perturbations.append((block_name, feature_index, delta))

    runs = oracle.run_many(requests)
    maximum_posterior_error = 0.0
    matching_classes = 0
    for perturbation, request, run in zip(
        perturbations,
        requests,
        runs,
        strict=True,
    ):
        python = _python_classifier_prediction(python_model, request)
        matlab_features = np.asarray(run.result["classifier_input"]).reshape(-1)
        matlab_posterior = np.asarray(run.result["posterior_scores"]).reshape(-1)
        np.testing.assert_allclose(
            python.features,
            matlab_features,
            rtol=0,
            atol=0,
            equal_nan=True,
            err_msg=f"assembled feature drift for perturbation {perturbation}",
        )
        error = float(
            np.max(np.abs(np.asarray(python.posterior) - matlab_posterior))
        )
        maximum_posterior_error = max(maximum_posterior_error, error)
        matlab_class = int(run.result["predicted_class"])
        matching_classes += int(python.predicted_class == matlab_class)
        np.testing.assert_allclose(
            python.posterior,
            matlab_posterior,
            rtol=2e-6,
            atol=2e-8,
            err_msg=f"posterior drift for perturbation {perturbation}",
        )
        assert python.predicted_class == matlab_class, perturbation

    class_agreement = matching_classes / len(runs)
    print(
        "classifier sensitivity parity: "
        f"cases={len(runs)}, "
        f"max_posterior_abs_error={maximum_posterior_error:.3g}, "
        f"class_agreement={class_agreement:.6f}"
    )
    assert len(runs) == 1 + 2 * python_model.feature_layout.selected_feature_count
    assert class_agreement == 1.0


def test_live_matlab_filter_and_full_detector_match_isolated_simulation():
    movie = default_synthetic_suite(seed=29)[0]
    image = movie.frames_tzyx[0]
    oracle = MatlabStarryNiteOracle(_live_config())
    requests = (
        separable_dog_request(
            image,
            radius_um=2.0,
            sigma_factor=1.0,
            calibration=movie.calibration,
        ),
        full_detection_request(
            image,
            radius_um=2.0,
            sigma_factor=1.0,
            intensity_threshold=18.0,
            boundary_percent=0.35,
            calibration=movie.calibration,
            num_cells=4,
        ),
    )

    filter_run, detector_run = oracle.run_many(requests)

    python_response = legacy_dog_response(
        image,
        radius_um=2.0,
        sigma_factor=1.0,
        calibration=movie.calibration,
    )
    filter_similarity = compare_volumes(filter_run.volume_zyx(), python_response)
    matlab_xyz = detector_run.points_zyx0("final_points_zyx_0based")[:, ::-1]
    detection_similarity = compare_detections(
        movie.truth_positions_xyz_px(1),
        matlab_xyz,
        tolerance=2.0,
        spacing=(0.5, 0.5, 1.0),
    )

    assert filter_similarity.relative_l2 < 5e-6
    assert filter_similarity.pearson_correlation == pytest.approx(1.0, abs=1e-7)
    assert len(matlab_xyz) == 1
    assert detection_similarity.f1 == 1.0
    assert np.asarray(detector_run.result["final_diameters_xy"]).size == 1


def test_live_matlab_exact_detector_tail_matches_full_detection_outputs(
    monkeypatch: pytest.MonkeyPatch,
):
    import acetree_py.tracking.starrynite.legacy_detector_tail as detector_tail

    captured_features: list[np.ndarray] = []
    original_log_odds = detector_tail.legacy_calculate_disk_log_odds

    def capture_log_odds(*args, **kwargs):
        captured_features.append(
            detector_tail.legacy_disk_feature_vectors(*args[:6]).T.copy()
        )
        return original_log_odds(*args, **kwargs)

    monkeypatch.setattr(
        detector_tail, "legacy_calculate_disk_log_odds", capture_log_odds
    )
    suite = default_synthetic_suite(seed=1731)
    cases = ((0, 1),)
    config = _live_config()
    distribution = (
        config.starrynite_root
        / "distribution_code"
        / "clean_distributions_newimage.mat"
    )
    requests = tuple(
        full_detection_request(
            suite[scenario].frames_tzyx[frame - 1],
            radius_um=2.0,
            sigma_factor=1.0,
            intensity_threshold=18.0,
            boundary_percent=0.35,
            calibration=suite[scenario].calibration,
            num_cells=4,
            distribution_file=distribution,
        )
        for scenario, frame in cases
    )
    runs = MatlabStarryNiteOracle(config).run_many(requests)

    for (scenario, frame), run in zip(cases, runs, strict=True):
        movie = suite[scenario]
        detections = StarryNiteDetector().detect(
            movie.frames_tzyx[frame - 1],
            frame,
            movie.calibration,
            {
                "RADIUS": 2.0,
                "SIGMA": 1.0,
                "INTENSITY_THRESHOLD": 18.0,
                "BOUNDARY_PERCENT": 0.35,
                "STARRYNITE_CELL_COUNT": 4,
                "STARRYNITE_DISTRIBUTION_FILE": str(distribution),
                "STARRYNITE_DISTRIBUTION_SOURCE_SHA256": sha256_file(
                    distribution
                ),
                "RANGE_THRESHOLD": 100.0,
                "SPLIT_THRESHOLD": 100.0,
                "MERGE_LOWER": -300.0,
                "MERGE_SPLIT": 1.0,
            },
        )
        python_points = np.asarray(
            [
                (
                    item.features["VOXEL_Z"],
                    item.features["VOXEL_Y"],
                    item.features["VOXEL_X"],
                )
                for item in detections
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        matlab_points = run.points_zyx0("final_points_zyx_0based")
        np.testing.assert_allclose(python_points, matlab_points, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            [item.features["LEGACY_DIAMETER_XY_PX"] for item in detections],
            np.asarray(run.result["final_diameters_xy"]).reshape(-1),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            [item.features["LEGACY_ASPECT_RATIO"] for item in detections],
            np.asarray(run.result["aspectratio"]).reshape(-1),
            rtol=1e-6,
            atol=1e-6,
        )
        serial_log_odds = np.asarray(
            run.result["nucleus_serial_log_odds"], dtype=np.float64
        ).reshape(-1)
        # SciPy's single-precision separable convolution and MATLAB imfilter
        # differ by a few float32 ULPs.  Verify that this bounded DoG drift is
        # the only input difference before comparing the learned log odds.
        assert captured_features
        np.testing.assert_allclose(
            captured_features[0].T,
            np.asarray(run.result["nucleus_disk_features"], dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
        )
        retained = (
            np.asarray(run.result["nucleus_ranges_1based"], dtype=int).reshape(-1)
            - 1
        )
        retained_serial = serial_log_odds[retained]
        np.testing.assert_allclose(
            detections[0].features["LEGACY_CLAIMED_LOG_ODDS"],
            retained_serial,
            rtol=5e-7,
            atol=5e-6,
        )
        assert detections[0].features["LEGACY_LOG_ODDS_SUM"] == pytest.approx(
            float(np.sum(retained_serial)), rel=5e-7, abs=5e-6
        )
        stored_parallel = np.asarray(
            run.result["mergedlogoddssum"], dtype=np.float64
        ).reshape(-1)
        # R2025a executes the upstream parfor with empty worker-global
        # distribution state and stores zeros.  Re-evaluating the unmodified
        # calculateLogodds function serially produces the intended learned
        # values.  Older MATLAB releases may store those values directly.
        assert np.allclose(stored_parallel, 0.0) or np.allclose(
            stored_parallel, [float(np.sum(retained_serial))], rtol=1e-7, atol=1e-7
        )
        assert all(item.features["LEGACY_EXACT_TAIL"] is True for item in detections)


def test_live_matlab_slice_candidate_geometry_matches_radial_port():
    cases = (
        (0, 1, 0.2),
        (0, 1, 0.35),
        (4, 4, 0.2),
        (4, 4, 0.35),
        (4, 4, 0.65),
    )
    expected = []
    requests = []
    suite = default_synthetic_suite(seed=1731)
    for scenario_index, frame, boundary_percent in cases:
        movie = suite[scenario_index]
        response = legacy_dog_response(
            movie.frames_tzyx[frame - 1],
            radius_um=2.0,
            sigma_factor=1.0,
            calibration=movie.calibration,
        )
        peaks = _canonical_maxima(
            response,
            threshold=18.0,
            footprint=np.ones((3, 3, 3), dtype=bool),
        )
        expected.append(
            tuple(
                legacy_radial_geometry(
                    response,
                    peak,
                    expected_diameter_xy_px=8.0,
                    boundary_percent=boundary_percent,
                )
                for peak in peaks
            )
        )
        requests.append(
            slice_candidates_request(
                response,
                maxima_threshold=18.0,
                cell_diameter_xy_px=8.0,
                anisotropy=2.0,
                num_cells=4,
                legacy_parameters={"boundary_percent": boundary_percent},
            )
        )
    oracle = MatlabStarryNiteOracle(_live_config())
    runs = oracle.run_many(requests)

    for run, python_geometry in zip(runs, expected, strict=True):
        matlab_centers = run.candidate_centers_zyx0()
        python_centers = np.asarray(
            [item.center_zyx_px for item in python_geometry], dtype=float
        ).reshape((-1, 3))
        matlab_diameters = np.asarray(
            run.result["candidate_diameters_xy"]
        ).reshape(-1)
        python_diameters = np.asarray(
            [item.diameter_xy_px for item in python_geometry], dtype=float
        )
        matlab_coverage = np.asarray(
            run.result["candidate_xy_coverage"]
        ).reshape(-1)
        python_coverage = np.asarray(
            [item.valid_ray_count for item in python_geometry], dtype=float
        )

        assert python_centers == pytest.approx(matlab_centers)
        assert python_diameters == pytest.approx(matlab_diameters)
        assert python_coverage == pytest.approx(matlab_coverage)


def test_live_legacy_2019_model_tracks_four_translating_nuclei():
    objects = []
    starts_xyz = ((14, 13, 5), (37, 13, 6), (14, 30, 7), (37, 30, 8))
    for frame in range(1, 6):
        for index, (x, y, z) in enumerate(starts_xyz):
            objects.append(
                SyntheticObject(
                    f"cell-{index}",
                    frame,
                    (z - 1.0, y - 1.0, x - 1.0 + frame - 1),
                    (2.2 / 3.0, 2.2, 2.2),
                    100.0,
                )
            )
    movie = render_movie(
        "translation",
        (13, 41, 51),
        objects,
        calibration=Calibration(1.0, 3.0),
        expected_radius_um=4.0,
        background=0.0,
    )
    config = _live_config()
    oracle = MatlabStarryNiteOracle(config)

    run = oracle.full_tracking(
        movie.frames_tzyx,
        radius_um=4.0,
        sigma_factor=0.5,
        intensity_threshold=0.25,
        boundary_percent=0.35,
        calibration=movie.calibration,
        num_cells=4,
    )

    edges = run.normalized_edges()
    stage_trace = run.tracking_stage_trace()
    assert run.matlab_version
    assert run.result["frame_detection_counts"] == pytest.approx([4, 4, 4, 4, 4])
    assert len(edges) == 16
    assert all(kind == "link" for _source, _target, kind in edges)
    assert [stage.label for stage in stage_trace.stages[:5]] == [
        "detected",
        "initialized",
        "easy_links",
        "post_polar_filter",
        "candidates",
    ]
    assert stage_trace.final.nodes == run.tracking_event_trace().checkpoints[0].nodes
    assert stage_trace.final.edge_count == 16

    _assert_live_early_geometry_parity(run, config)


def test_live_registered_exact_pipeline_matches_matlab_full_movie(tmp_path: Path):
    """Exercise the public detector -> whole-movie tracker integration boundary."""

    movie = lineage_synthetic_suite(seed=2718)[3]
    dynamic_objects = []
    for frame in (1, 2):
        for index, (x, y) in enumerate(
            (x, y)
            for y in (12.0, 34.0, 56.0)
            for x in (12.0, 34.0, 56.0, 78.0)
        ):
            dynamic_objects.append(
                SyntheticObject(
                    f"dynamic-{index}",
                    frame,
                    (
                        3.0 + float(index % 5),
                        y - 1.0,
                        x - 1.0 + 0.25 * (frame - 1),
                    ),
                    (2.2 / 3.0, 2.21, 2.21),
                    120.0,
                )
            )
    dynamic_movie = render_movie(
        "dynamic-detector-state",
        (13, 69, 91),
        dynamic_objects,
        calibration=movie.calibration,
        expected_radius_um=4.0,
        background=0.0,
    )
    config = _live_config()
    oracle = MatlabStarryNiteOracle(config)
    distribution_file = (
        config.starrynite_root
        / "distribution_code"
        / "clean_distributions_newimage.mat"
    ).resolve()
    model_file = (
        config.starrynite_root
        / "distribution_lineaging"
        / "2019TrackingModelv2.mat"
    ).resolve()
    tracking_run, dynamic_run, classifier_run = oracle.run_many(
        (
            full_tracking_request(
                movie.frames_tzyx,
                radius_um=4.0,
                sigma_factor=0.5,
                intensity_threshold=0.25,
                boundary_percent=0.35,
                calibration=movie.calibration,
                num_cells=4,
                use_static_diameter=False,
                legacy_parameters={"selection_dist": 0.5},
                distribution_file=distribution_file,
                model_file=model_file,
            ),
            full_tracking_request(
                dynamic_movie.frames_tzyx,
                radius_um=5.0,
                sigma_factor=0.5,
                intensity_threshold=0.25,
                boundary_percent=0.35,
                calibration=dynamic_movie.calibration,
                num_cells=1,
                use_static_diameter=False,
                legacy_parameters={
                    "selection_dist": 0.5,
                    "staging": np.asarray([10.0, 1_000_000_000.0]),
                },
                distribution_file=distribution_file,
                model_file=model_file,
            ),
            classifier_model_export_request(model_file=model_file),
        )
    )

    model_sha256 = sha256_file(model_file)
    classifier = neutral_classifier_from_matlab_export(
        classifier_run.result,
        source_model_sha256=model_sha256,
    )
    classifier_file = tmp_path / "2019TrackingModelv2.neutral.json"
    save_neutral_classifier(classifier_file, classifier)
    parameter_file = _write_live_exact_parameter_file(
        tmp_path,
        model_file=model_file,
        distribution_file=distribution_file,
        initial_cell_count=4,
    )
    profile = load_tuning_profile(parameter_file)
    assert profile.parameter_sha256 is not None
    assert profile.model_sha256 == model_sha256
    expected_exact_tail_settings = {
        "RANGE_THRESHOLD": 100.0,
        "SPLIT_THRESHOLD": 100.0,
        "MERGE_LOWER": -300.0,
        "MERGE_SPLIT": 1.0,
    }
    assert {
        name: profile.detector_settings[name]
        for name in expected_exact_tail_settings
    } == expected_exact_tail_settings
    detector_settings = dict(profile.detector_settings)
    detector_settings["STARRYNITE_USE_STATIC_DIAMETER"] = False
    tracker_settings = {
        "STARRYNITE_COMPATIBILITY_MODE": LEGACY_EXACT_REFINEMENT_BACKEND,
        "STARRYNITE_PARAMETER_FILE": str(parameter_file.resolve()),
        "STARRYNITE_PARAMETER_SHA256": profile.parameter_sha256,
        "STARRYNITE_MODEL_FILE": str(model_file),
        "STARRYNITE_MODEL_SHA256": model_sha256,
        "STARRYNITE_NEUTRAL_CLASSIFIER_FILE": str(classifier_file.resolve()),
        "STARRYNITE_NEUTRAL_CLASSIFIER_SHA256": sha256_file(classifier_file),
        "STARRYNITE_USE_STATIC_DIAMETER": False,
        "ALLOW_TRACK_SPLITTING": True,
    }
    request = TrackingRequest(
        detector=ComponentSpec("acetree.starrynite_detector", detector_settings),
        tracker=ComponentSpec("acetree.starrynite_legacy_exact", tracker_settings),
        scope=TrackingScope("global", 1, movie.frames_tzyx.shape[0]),
    )
    result = TrackingPipeline(
        build_default_registry(discover_plugins=False)
    ).run(
        NumpyProvider(movie.frames_tzyx),
        movie.calibration,
        request,
    )

    refinement = result.provenance["graph_refinement"]
    assert refinement["backend"] == LEGACY_EXACT_REFINEMENT_BACKEND
    assert refinement["boundary"] == "detections_through_legacy_classifier_movie"
    assert refinement["parameter_sha256"] == profile.parameter_sha256
    assert refinement["model_sha256"] == model_sha256
    assert refinement["neutral_classifier_sha256"] == sha256_file(classifier_file)
    assert refinement["distribution_sha256"] == sha256_file(distribution_file)
    assert refinement["event_order_validated"] is True
    assert refinement["rejected_detection_count"] == 0

    detections_by_id = {item.detection_id: item for item in result.detections}

    def row_key(detection_id: str) -> tuple[int, int]:
        detection = detections_by_id[detection_id]
        return (
            detection.frame - 1,
            int(detection.features["LEGACY_ROW_INDEX"]),
        )

    matlab_rows = {
        (int(row[0]), int(row[1])): row for row in tracking_run.node_table()
    }
    retained_matlab_rows = {
        key: row for key, row in matlab_rows.items() if not bool(row[7])
    }
    pipeline_rows = {row_key(item.detection_id): item for item in result.detections}
    assert pipeline_rows.keys() == retained_matlab_rows.keys()
    for key, detection in pipeline_rows.items():
        matlab_row = retained_matlab_rows[key]
        assert detection.features["VOXEL_X"] == pytest.approx(matlab_row[2])
        assert detection.features["VOXEL_Y"] == pytest.approx(matlab_row[3])
        assert detection.features["VOXEL_Z"] == pytest.approx(matlab_row[4])
        assert 2.0 * detection.radius_um / movie.calibration.xy_um == pytest.approx(
            matlab_row[5]
        )

    matlab_edges = sorted(tracking_run.normalized_edges())
    pipeline_edges = sorted(
        (row_key(item.source_id), row_key(item.target_id), item.kind)
        for item in result.edges
    )
    assert pipeline_edges == matlab_edges

    matlab_stages = tracking_run.tracking_stage_trace().stages[1:]
    pipeline_stages = tuple(refinement["early_stages"])
    assert [item["name"] for item in pipeline_stages] == [
        item.label for item in matlab_stages
    ]
    for pipeline_stage, matlab_stage in zip(
        pipeline_stages, matlab_stages, strict=True
    ):
        threshold = matlab_stage.threshold
        if threshold is not None and math.isinf(threshold):
            threshold = "inf" if threshold > 0 else "-inf"
        assert pipeline_stage["threshold"] == threshold
        assert pipeline_stage["link_count"] == matlab_stage.edge_count
        assert pipeline_stage["division_count"] == sum(
            item.successor_slots[1] is not None for item in matlab_stage.nodes
        )
        assert pipeline_stage["deleted_count"] == len(matlab_stage.deleted_ids)
        assert pipeline_stage["forward_candidate_count"] == len(
            matlab_stage.forward_candidate_pairs
        )
        assert pipeline_stage["backward_candidate_count"] == len(
            matlab_stage.backward_candidate_pairs
        )

    matlab_events = tracking_run.tracking_event_trace().classifications
    assert len(matlab_events) == 1
    event = matlab_events[0]
    assert refinement["top_level_event_count"] == 1
    assert refinement["classification_count"] == 1
    assert refinement["classification_round_counts"] == {"1": 1}
    assert refinement["computed_class_counts"] == {
        str(event.computed_class): 1
    }
    assert refinement["effective_class_counts"] == {
        str(event.effective_class): 1
    }
    first = refinement["first_classification"]
    assert first == refinement["last_classification"]
    assert first["round"] == event.classifier_round
    assert first["computed_class"] == event.computed_class
    assert first["effective_class"] == event.effective_class
    assert f"matlab:{first['frame'] - 1}:{first['row']}" == event.parent_id
    assert tuple(
        f"matlab:{frame}:{row}" for frame, row in map(row_key, first["daughters"])
    ) == event.daughter_ids

    reference_lineage = matlab_lineage_snapshot(
        tracking_run,
        movie.calibration,
    )
    candidate_lineage = python_lineage_snapshot(
        result.detections,
        set(detections_by_id),
        result.edges,
        provenance=refinement,
    )
    similarity = compare_lineage_snapshots(
        reference_lineage,
        candidate_lineage,
        tolerance_um=1e-4,
    )
    assert similarity.node_f1 == 1.0
    assert similarity.state_accuracy == 1.0
    assert similarity.edge_similarity.f1 == 1.0
    assert similarity.edge_similarity.division_f1 == 1.0
    assert similarity.ancestry_agreement == 1.0

    # The second batched movie crosses both of processVolume's stateful
    # boundaries: more than ten prior candidate diameters enables median
    # diameter adaptation, while the prior final row count selects stage 2.
    dynamic_directory = tmp_path / "dynamic"
    dynamic_directory.mkdir()
    dynamic_parameter_file = _write_live_exact_parameter_file(
        dynamic_directory,
        model_file=model_file,
        distribution_file=distribution_file,
        initial_cell_count=1,
        initial_diameter_xy_px=10.0,
        staging=(10, 1_000_000_000),
    )
    dynamic_profile = load_tuning_profile(dynamic_parameter_file)
    assert {
        name: dynamic_profile.detector_settings[name]
        for name in expected_exact_tail_settings
    } == expected_exact_tail_settings
    dynamic_settings = dict(dynamic_profile.detector_settings)
    dynamic_settings["STARRYNITE_USE_STATIC_DIAMETER"] = False
    dynamic_detector = build_default_registry(
        discover_plugins=False
    ).create_detector("acetree.starrynite_detector")
    dynamic_detections = tuple(
        tuple(
            dynamic_detector.detect(
                image,
                frame,
                dynamic_movie.calibration,
                dynamic_settings,
            )
        )
        for frame, image in enumerate(dynamic_movie.frames_tzyx, start=1)
    )
    matlab_frame_counts = tuple(
        int(value)
        for value in np.asarray(
            dynamic_run.result["frame_detection_counts"]
        ).reshape(-1)
    )
    assert tuple(map(len, dynamic_detections)) == matlab_frame_counts
    assert matlab_frame_counts[0] == 12

    def uniform_dynamic_feature(frame_index: int, name: str):
        rows = dynamic_detections[frame_index]
        assert rows
        value = rows[0].features[name]
        assert all(item.features[name] == value for item in rows)
        return value

    matlab_effective_counts = tuple(
        int(value)
        for value in np.asarray(
            dynamic_run.result["detector_effective_cell_counts"]
        ).reshape(-1)
    )
    python_effective_counts = tuple(
        int(uniform_dynamic_feature(index, "STARRYNITE_CELL_COUNT"))
        for index in range(2)
    )
    assert python_effective_counts == matlab_effective_counts
    assert python_effective_counts == (1, 12)
    assert tuple(
        int(uniform_dynamic_feature(index, "STARRYNITE_STAGE_INDEX"))
        for index in range(2)
    ) == (0, 1)

    python_candidate_diameters = tuple(
        tuple(
            float(value)
            for value in uniform_dynamic_feature(
                index, "LEGACY_CANDIDATE_DIAMETERS_XY_PX"
            )
        )
        for index in range(2)
    )
    matlab_candidate_counts = tuple(
        int(value)
        for value in np.asarray(
            dynamic_run.result["detector_candidate_diameter_counts"]
        ).reshape(-1)
    )
    assert tuple(map(len, python_candidate_diameters)) == matlab_candidate_counts
    assert matlab_candidate_counts[0] > 10

    matlab_effective_diameters = tuple(
        float(value)
        for value in np.asarray(
            dynamic_run.result["detector_cell_diameters_xy"]
        ).reshape(-1)
    )
    python_effective_diameters = tuple(
        float(
            uniform_dynamic_feature(
                index, "LEGACY_EFFECTIVE_DIAMETER_XY_PX"
            )
        )
        for index in range(2)
    )
    assert python_effective_diameters == pytest.approx(matlab_effective_diameters)
    assert python_effective_diameters[0] == pytest.approx(10.0)
    assert python_effective_diameters[1] == pytest.approx(
        float(np.median(python_candidate_diameters[0]))
    )
    assert python_effective_diameters[1] != pytest.approx(
        python_effective_diameters[0]
    )

    dynamic_rows_by_key = {
        (int(row[0]), int(row[1])): row for row in dynamic_run.node_table()
    }
    python_dynamic_rows = {
        (
            detection.frame - 1,
            int(detection.features["LEGACY_ROW_INDEX"]),
        ): detection
        for frame_rows in dynamic_detections
        for detection in frame_rows
    }
    assert python_dynamic_rows.keys() == dynamic_rows_by_key.keys()
    for key, detection in python_dynamic_rows.items():
        matlab_row = dynamic_rows_by_key[key]
        assert detection.features["VOXEL_X"] == pytest.approx(matlab_row[2])
        assert detection.features["VOXEL_Y"] == pytest.approx(matlab_row[3])
        assert detection.features["VOXEL_Z"] == pytest.approx(matlab_row[4])
        assert 2.0 * detection.radius_um / dynamic_movie.calibration.xy_um == (
            pytest.approx(matlab_row[5])
        )


def test_live_positive_class_one_division_matches_complete_python_graph():
    movie = lineage_synthetic_suite(seed=2718)[3]
    result = run_lineage_parity_case(
        movie,
        MatlabStarryNiteOracle(_live_config()),
        radius_um=4.0,
        sigma_factor=0.5,
        intensity_threshold=0.25,
        boundary_percent=0.35,
        tracker_settings={"DIVISION_MAX_DAUGHTER_SEPARATION": 20.0},
    )
    similarity = result.similarity

    assert similarity.node_f1 == 1.0
    assert similarity.state_accuracy == 1.0
    assert similarity.edge_similarity.f1 == 1.0
    assert similarity.edge_similarity.division_f1 == 1.0
    assert similarity.edge_similarity.reference_division_count == 1
    assert similarity.edge_similarity.candidate_division_count == 1
    assert similarity.ancestry_agreement == 1.0
    diagnostics = result.matlab_classifier_diagnostics
    np.testing.assert_array_equal(
        np.asarray(diagnostics["classifier_computed_classes"]).reshape(-1),
        [1.0],
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics["classifier_reference_classes"]).reshape(-1),
        [1.0],
    )
    np.testing.assert_array_equal(
        np.asarray(diagnostics["classifier_rounds"]).reshape(-1),
        [1.0],
    )
    event_trace = result.matlab_event_trace
    assert event_trace is not None
    assert len(event_trace.checkpoints) == 3
    assert [event.classifier_round for event in event_trace.classifications] == [1]
    assert [event.computed_class for event in event_trace.classifications] == [1]
    assert [event.effective_class for event in event_trace.classifications] == [1]


def test_live_positive_class_one_matches_exact_post_greedy_driver_trace():
    """Replay MATLAB's initial raw checkpoint through the exact Python pass."""

    movie = lineage_synthetic_suite(seed=2718)[3]
    config = _live_config()
    oracle = MatlabStarryNiteOracle(config)
    tracking_run, export_run = oracle.run_many(
        (
            full_tracking_request(
                movie.frames_tzyx,
                radius_um=4.0,
                sigma_factor=0.5,
                intensity_threshold=0.25,
                boundary_percent=0.35,
                calibration=movie.calibration,
                num_cells=0,
            ),
            classifier_model_export_request(),
        )
    )
    _assert_live_early_geometry_parity(tracking_run, config)
    source_model = (
        config.starrynite_root
        / "distribution_lineaging"
        / "2019TrackingModelv2.mat"
    )
    source_hash = hashlib.sha256(source_model.read_bytes()).hexdigest()
    classifier = neutral_classifier_from_matlab_export(
        export_run.result,
        source_model_sha256=source_hash,
    )
    decoded_model = load_matlab_model(source_model)
    statistics = LegacyTrackingStatistics.from_model(decoded_model)
    reference = tracking_run.tracking_event_trace()
    measurement_template = tracking_run.legacy_tracking_context()
    initial_context = legacy_context_from_checkpoint(
        measurement_template,
        reference.checkpoints[0],
    )
    observations = []
    result = run_legacy_movie_decisions(
        initial_context,
        classifier,
        statistics,
        config=_live_decision_config(
            decoded_model,
            end_frame=initial_context.parameters.end_frame,
        ),
        classification_observer=observations.append,
    )

    assert result.supported, result.failure
    candidate = trace_from_legacy_movie_result(
        initial_context,
        result,
        observations,
    )
    assert_event_trace_parity(reference, candidate)
    assert [event.effective_class for event in candidate.classifications] == [1]


def test_live_class_zero_recursive_retries_match_exact_post_greedy_driver_trace():
    """Cover dynamic round-two retries in two real 2019-model scenarios."""

    movies = lineage_synthetic_suite(seed=2718)
    scenarios = (movies[2], movies[4])
    config = _live_config()
    oracle = MatlabStarryNiteOracle(config)
    requests = [
        full_tracking_request(
            movie.frames_tzyx,
            radius_um=4.0,
            sigma_factor=0.5,
            intensity_threshold=0.25,
            boundary_percent=0.35,
            calibration=movie.calibration,
            num_cells=0,
        )
        for movie in scenarios
    ]
    requests.append(classifier_model_export_request())
    *tracking_runs, export_run = oracle.run_many(requests)
    source_model = (
        config.starrynite_root
        / "distribution_lineaging"
        / "2019TrackingModelv2.mat"
    )
    source_hash = hashlib.sha256(source_model.read_bytes()).hexdigest()
    classifier = neutral_classifier_from_matlab_export(
        export_run.result,
        source_model_sha256=source_hash,
    )
    decoded_model = load_matlab_model(source_model)
    statistics = LegacyTrackingStatistics.from_model(decoded_model)

    for movie, tracking_run in zip(scenarios, tracking_runs, strict=True):
        _assert_live_early_geometry_parity(tracking_run, config)
        reference = tracking_run.tracking_event_trace()
        measurement_template = tracking_run.legacy_tracking_context()
        initial_context = legacy_context_from_checkpoint(
            measurement_template,
            reference.checkpoints[0],
        )
        observations = []
        result = run_legacy_movie_decisions(
            initial_context,
            classifier,
            statistics,
            config=_live_decision_config(
                decoded_model,
                end_frame=initial_context.parameters.end_frame,
            ),
            classification_observer=observations.append,
        )

        assert result.supported, f"{movie.name}: {result.failure}"
        candidate = trace_from_legacy_movie_result(
            initial_context,
            result,
            observations,
        )
        assert_event_trace_parity(reference, candidate)
        assert len(candidate.ordered_events) == 9
        assert [
            event.effective_class for event in candidate.classifications
        ] == [0, 0, 0, 0]
        assert [
            event.classifier_round for event in candidate.classifications
        ] == [1, 2, 2, 2]


def test_live_classes_two_and_three_match_exact_post_greedy_driver_trace():
    """Exercise real false-negative rewiring and false-positive deletion."""

    scenarios = _post_greedy_class_two_and_three_movies()
    config = _live_config()
    oracle = MatlabStarryNiteOracle(config)
    requests = [
        full_tracking_request(
            movie.frames_tzyx,
            radius_um=4.0,
            sigma_factor=0.5,
            intensity_threshold=0.25,
            boundary_percent=0.35,
            calibration=movie.calibration,
            num_cells=0,
        )
        for movie in scenarios
    ]
    requests.append(classifier_model_export_request())
    *tracking_runs, export_run = oracle.run_many(requests)
    source_model = (
        config.starrynite_root
        / "distribution_lineaging"
        / "2019TrackingModelv2.mat"
    )
    source_hash = hashlib.sha256(source_model.read_bytes()).hexdigest()
    classifier = neutral_classifier_from_matlab_export(
        export_run.result,
        source_model_sha256=source_hash,
    )
    decoded_model = load_matlab_model(source_model)
    statistics = LegacyTrackingStatistics.from_model(decoded_model)
    expected_classes = (3, 2)
    expected_detection_counts = (
        [4, 4, 4, 5, 4, 4, 4, 4, 4],
        [5, 5, 5, 4, 5, 5, 5, 5, 5],
    )

    for movie, tracking_run, expected_class, expected_counts in zip(
        scenarios,
        tracking_runs,
        expected_classes,
        expected_detection_counts,
        strict=True,
    ):
        _assert_live_early_geometry_parity(tracking_run, config)
        np.testing.assert_array_equal(
            np.asarray(tracking_run.result["frame_detection_counts"]).reshape(-1),
            expected_counts,
        )
        np.testing.assert_array_equal(
            np.asarray(
                tracking_run.result["classifier_computed_classes"]
            ).reshape(-1),
            [expected_class],
        )
        np.testing.assert_array_equal(
            np.asarray(
                tracking_run.result["classifier_reference_classes"]
            ).reshape(-1),
            [expected_class],
        )
        np.testing.assert_array_equal(
            np.asarray(tracking_run.result["classifier_rounds"]).reshape(-1),
            [1],
        )

        reference = tracking_run.tracking_event_trace()
        measurement_template = tracking_run.legacy_tracking_context()
        initial_context = legacy_context_from_checkpoint(
            measurement_template,
            reference.checkpoints[0],
        )
        observations = []
        result = run_legacy_movie_decisions(
            initial_context,
            classifier,
            statistics,
            config=_live_decision_config(
                decoded_model,
                end_frame=initial_context.parameters.end_frame,
            ),
            classification_observer=observations.append,
        )

        assert result.supported, f"{movie.name}: {result.failure}"
        candidate = trace_from_legacy_movie_result(
            initial_context,
            result,
            observations,
        )
        assert_event_trace_parity(reference, candidate)
        assert len(candidate.ordered_events) == 3
        assert [
            event.effective_class for event in candidate.classifications
        ] == [expected_class]
        assert result.events[0].lineage_diagnostics is not None
        assert result.events[0].lineage_diagnostics.classification == expected_class
        if expected_class == 3:
            assert len(result.events[0].lineage_diagnostics.deleted_ids) == 1
        else:
            assert len(result.events[0].lineage_diagnostics.applied_gap_edges) == 1
