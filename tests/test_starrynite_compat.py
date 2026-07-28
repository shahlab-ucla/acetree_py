"""Focused clean-room tests for StarryNite parameter/model compatibility I/O."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from acetree_py.tracking.starrynite import (
    LegacyParameterResolutionError,
    MissingScipyError,
    OpaqueMatlabValue,
    ParameterParseError,
    StarryNitePresetError,
    UnsupportedMatlabVersionError,
    build_tuning_save_plan,
    build_legacy_region_table,
    identify_classifier_flavor,
    legacy_stage_index,
    load_matlab_model,
    load_tuning_profile,
    normalize_parameter_name,
    parse_parameter_text,
    parse_parameter_expression,
    parse_parameter_value,
    read_parameter_file,
    resolve_legacy_parameter,
    select_stage_value,
    tuning_profile_from_parameters,
    write_parameter_file,
)


def test_matlab_parameters_are_safe_typed_lossless_and_last_assignment_wins(
    tmp_path: Path,
) -> None:
    source = (
        "% original StarryNite-style commands\n"
        "tracking.nucleusRadius = 4; threshold = [40, 60]./10;\n"
        "enabled = true; label = 'O''Brien';\n"
        "threshold = 9; % repeated assignment\n"
        "load 'models/new_classifier.mat';\n"
        "danger = system('this must never run');\n"
        "if enabled\n"
        "    display('also never run');\n"
        "end\n"
    )

    parameters = parse_parameter_text(
        source,
        source_path=tmp_path / "embryo-parameters.m",
    )

    assert parameters.render() == source
    assert parameters.settings["tracking.nucleusRadius"] == 4
    assert parameters.settings["threshold"] == 9
    assert parameters.settings["enabled"] is True
    assert parameters.settings["label"] == "O'Brien"
    assert parameters.normalized_settings["tracking.nucleus_radius"] == 4
    assert [record.kind for record in parameters.records].count("load") == 1
    assert any("system" in record.source for record in parameters.opaque_records)
    assert any("display" in record.source for record in parameters.opaque_records)
    assert parameters.model_references[0].raw_path == "models/new_classifier.mat"
    assert parameters.resolve_model_references() == (
        (tmp_path / "models" / "new_classifier.mat").resolve(),
    )


def test_classic_whitespace_parameters_support_comments_vectors_and_model_settings(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models" / "old.mat"
    model_path.parent.mkdir()
    model_path.write_bytes(b"placeholder")
    source = (
        "# classic StarryNite key-value form\n"
        "radius 4.5\n"
        "maxFrames 100 # an inline comment\n"
        "feature.flags [1, 2, 3]\n"
        "modelFile models/old.mat\n"
        "enabled false\n"
        "unsupported call(value)\n"
        "radius 5\n"
    )

    parameters = parse_parameter_text(source, source_path=tmp_path / "parameters.txt")

    assert parameters.syntax == "classic"
    assert parameters.render() == source
    assert dict(parameters.settings) == {
        "radius": 5,
        "maxFrames": 100,
        "feature.flags": (1, 2, 3),
        "modelFile": "models/old.mat",
        "enabled": False,
    }
    assert parameters.normalized_settings["max_frames"] == 100
    assert len(parameters.opaque_records) == 1
    assert parameters.resolve_model_references(must_exist=True) == (model_path.resolve(),)


def test_render_appends_overrides_without_rewriting_original_source() -> None:
    source = "% hand-tuned\nthreshold = 5;\n"
    parameters = parse_parameter_text(source)

    rendered = parameters.render(
        {"threshold": 2.5},
        append={"division.followBoth": True, "search.radii": (4, 6)},
    )

    assert rendered.startswith(source)
    assert "threshold = 2.5;" in rendered
    assert "division.followBoth = true;" in rendered
    reparsed = parse_parameter_text(rendered)
    assert reparsed.settings["threshold"] == 2.5
    assert reparsed.settings["division.followBoth"] is True
    assert reparsed.settings["search.radii"] == (4, 6)
    with pytest.raises(ValueError, match="both overridden and appended"):
        parameters.render({"threshold": 1}, append={"threshold": 2})

    classic_crlf = parse_parameter_text("radius 4\r\n")
    assert classic_crlf.render(append={"enabled": True}).endswith(
        "enabled true\r\n"
    )


def test_value_grammar_has_no_name_lookup_or_function_execution(tmp_path: Path) -> None:
    assert parse_parameter_value("[40, 60]./10") == (4.0, 6.0)
    assert parse_parameter_value("([2, 4] + 2).*3") == (12, 18)
    assert parse_parameter_value("'a''b'") == "a'b"
    assert normalize_parameter_name("Tracking.MaxFrameGap") == "tracking.max_frame_gap"

    marker = tmp_path / "should-not-exist"
    malicious = parse_parameter_text(
        f"result = __import__('pathlib').Path('{marker}').touch();\n"
    )
    assert "result" not in malicious.settings
    assert len(malicious.opaque_records) == 1
    assert not marker.exists()
    with pytest.raises(ParameterParseError, match="unsupported identifier"):
        parse_parameter_value("workspaceVariable + 1")


def test_literal_numeric_roi_matrix_is_typed_rectangular_and_round_trips() -> None:
    source = "ROIpoints=[161 14; 252 10; 345 25; ];\n"
    parameters = parse_parameter_text(source)

    assert parameters.settings["ROIpoints"] == (
        (161, 14),
        (252, 10),
        (345, 25),
    )
    assert parameters.opaque_records == ()
    rendered = parameters.render(
        {"ROIpoints": ((1.5, 2.5), (3.5, 4.5), (5.5, 6.5))}
    )
    assert "ROIpoints = [1.5, 2.5; 3.5, 4.5; 5.5, 6.5];" in rendered
    assert parse_parameter_text(rendered).settings["ROIpoints"] == (
        (1.5, 2.5),
        (3.5, 4.5),
        (5.5, 6.5),
    )
    # Column matrices keep legacy stage-vector behavior.
    assert parse_parameter_value("[1; 2; 3]") == (1, 2, 3)
    with pytest.raises(ParameterParseError, match="equal lengths"):
        parse_parameter_value("[1 2; 3]")
    with pytest.raises(ParameterParseError, match="numeric scalar"):
        parse_parameter_value("[1 'not numeric'; 2 3]")


def test_explicit_constant_expression_environment_remains_non_executable() -> None:
    assert parse_parameter_expression(
        "[450 * downsample, 720 * downsample]",
        constants={"downsample": 2},
    ) == (900, 1440)

    with pytest.raises(ParameterParseError, match="unsupported identifier"):
        parse_parameter_expression("unknown + 1", constants={"downsample": 2})
    with pytest.raises(ParameterParseError):
        parse_parameter_expression("system('touch marker')", constants={})
    with pytest.raises(ValueError, match="must be numeric"):
        parse_parameter_expression("label", constants={"label": "embryo"})


def test_parameter_script_resolves_prior_inert_aliases_in_source_order() -> None:
    parameters = parse_parameter_text(
        "distribution_file='clean.mat';\n"
        "distribution_file2=distribution_file;\n"
        "scale=2;\n"
        "parameters.intensitythreshold=[4, 6]./scale;\n"
    )

    assert parameters.settings["distribution_file2"] == "clean.mat"
    assert parameters.settings["parameters.intensitythreshold"] == (2.0, 3.0)
    assert parameters.opaque_records == ()


def test_parameter_script_invalidates_alias_after_dynamic_reassignment() -> None:
    parameters = parse_parameter_text(
        "scale=2;\n"
        "scale=system('never execute');\n"
        "parameters.intensitythreshold=scale;\n"
    )

    assert parameters.settings["scale"] == 2
    assert "parameters.intensitythreshold" not in parameters.settings
    assert len(parameters.opaque_records) == 2
    assert "unsupported identifier 'system'" in (
        parameters.opaque_records[0].reason or ""
    )
    assert "unsupported identifier 'scale'" in (
        parameters.opaque_records[1].reason or ""
    )


def test_model_loader_hashes_and_recursively_exposes_classic_mat_fields(
    tmp_path: Path,
) -> None:
    scipy_io = pytest.importorskip("scipy.io")
    model_path = tmp_path / "newmatlab" / "classifier.mat"
    model_path.parent.mkdir()
    scipy_io.savemat(
        model_path,
        {
            "weights": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "label": "embryo",
            "configuration": {
                "threshold": 0.75,
                "names": np.array(["left", "right"], dtype=object),
            },
            "classifier_flavor": "new",
        },
    )

    model = load_matlab_model(model_path)

    assert model.path == model_path.resolve()
    assert model.sha256 == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model.fields["weights"] == ((1.0, 2.0), (3.0, 4.0))
    assert model.fields["label"] == "embryo"
    assert model.fields["configuration"]["threshold"] == 0.75
    assert model.fields["configuration"]["names"] == ("left", "right")
    assert model.classifier_flavor == "new_classifier"
    assert model.classifier_evidence
    assert model.metadata["source_size_bytes"] == model_path.stat().st_size


def test_large_model_arrays_become_metadata_only_values(tmp_path: Path) -> None:
    scipy_io = pytest.importorskip("scipy.io")
    model_path = tmp_path / "large.mat"
    scipy_io.savemat(model_path, {"weights": np.arange(12).reshape(3, 4)})

    model = load_matlab_model(model_path, max_array_elements=4)

    opaque = model.fields["weights"]
    assert isinstance(opaque, OpaqueMatlabValue)
    assert opaque.shape == (3, 4)
    assert opaque.dtype is not None
    assert "extraction limit" in opaque.reason


def test_hdf5_and_missing_scipy_fail_with_actionable_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hdf_path = tmp_path / "v73.mat"
    hdf_path.write_bytes(b"MATLAB 7.3 MAT-file" + b"\0" * 600)
    with pytest.raises(UnsupportedMatlabVersionError, match="-v7"):
        load_matlab_model(hdf_path)

    classic_path = tmp_path / "classic.mat"
    classic_path.write_bytes(b"classic-placeholder")
    from acetree_py.tracking.starrynite import models as model_module

    def missing_scipy():
        raise ImportError("simulated optional dependency")

    monkeypatch.setattr(model_module, "_import_scipy_io", missing_scipy)
    with pytest.raises(MissingScipyError, match="scipy.io"):
        load_matlab_model(classic_path)


def test_classifier_identification_is_conservative_and_reports_conflicts() -> None:
    legacy = identify_classifier_flavor({"classifier_type": "NaiveBayes"})
    modern = identify_classifier_flavor(
        {"classifier_type": "CompactClassificationNaiveBayes"}
    )
    conflict = identify_classifier_flavor(
        {
            "legacy": {"classifier_type": "NaiveBayes"},
            "modern": {"classifier_type": "ClassificationNaiveBayes"},
        }
    )
    unknown = identify_classifier_flavor({"weights": (1.0, 2.0)})

    assert legacy.flavor == "legacy_classifier"
    assert modern.flavor == "new_classifier"
    assert conflict.flavor == "unknown"
    assert "conflicting" in conflict.evidence[0]
    assert unknown.flavor == "unknown" and unknown.evidence == ()


def test_tuning_profile_warns_when_stage_one_uses_an_assumed_zero_count() -> None:
    profile = tuning_profile_from_parameters(
        parse_parameter_text(
            "parameters.staging=[25,80];\n"
            "parameters.intensitythreshold=[5,8,12];\n"
        )
    )

    assert profile.cell_count == 0
    assert profile.stage_index == 0
    assert any(
        "firsttimestepnumcells is missing" in warning
        and "assumed starting cell count of 0" in warning
        for warning in profile.warnings
    )


def test_tuning_profile_maps_standard_staged_parameters_and_model(tmp_path: Path) -> None:
    model_path = tmp_path / "legacy-model.mat"
    model_path.write_bytes(b"legacy model fixture")
    parameter_path = tmp_path / "parameters.txt"
    parameter_path.write_text(
        "\n".join(
            (
                "xyres=.25;",
                "zres=1;",
                "firsttimestepdiam=40;",
                "parameters.staging=[25,80,181];",
                "parameters.sigma=1;",
                "parameters.intensitythreshold=[4,6,8,10];",
                "parameters.rangethreshold=[100,20,10,5];",
                "parameters.mergelower=[-300,-200,-100,-35];",
                "parameters.mergesplit=[1,.8,.5,.3];",
                "parameters.split=[100,50,20,10];",
                "parameters.boundary_percent=.4;",
                "parameters.large_ray_threshold=1.7;",
                "parameters.small_ray_threshold=.25;",
                "parameters.nndist_merge=[.8,.6,.4,.3];",
                "parameters.armerge=[1.6,1.4,1.2,1.0];",
                "trackingparameters.candidateCutoff=1.3;",
                "trackingparameters.safefactor=2;",
                "trackingparameters.nnnumber=2;",
                "trackingparameters.forwardnnnumber=4;",
                "trackingparameters.temporalcutoff=6;",
                "load 'legacy-model.mat';",
                "trackingparameters.DivCostFunction=@divScoreModelCostFunction;",
            )
        ),
        encoding="utf-8",
    )

    profile = load_tuning_profile(parameter_path, cell_count=80)

    assert profile.stage_index == 1  # boundaries advance only when strictly exceeded
    assert profile.detector_settings["RADIUS"] == pytest.approx(5.0)
    assert profile.detector_settings["INTENSITY_THRESHOLD"] == pytest.approx(6.0)
    assert profile.detector_settings["DO_SUBPIXEL_LOCALIZATION"] is False
    assert profile.detector_settings["LARGE_RAY_THRESHOLD"] == pytest.approx(1.7)
    assert profile.detector_settings["SMALL_RAY_THRESHOLD"] == pytest.approx(0.25)
    assert profile.detector_settings["NNDIST_MERGE"] == pytest.approx(0.6)
    assert profile.detector_settings["AR_MERGE"] == pytest.approx(1.4)
    assert "MIN_LOCAL_CONTRAST" not in profile.detector_settings
    assert profile.detector_settings["RANGE_THRESHOLD"] == pytest.approx(20.0)
    assert profile.detector_settings["MERGE_LOWER"] == pytest.approx(-200.0)
    assert profile.detector_settings["MERGE_SPLIT"] == pytest.approx(0.8)
    assert profile.detector_settings["SPLIT_THRESHOLD"] == pytest.approx(50.0)
    assert profile.tracker_settings["CANDIDATE_CUTOFF"] == pytest.approx(1.3)
    assert profile.tracker_settings["MAX_FRAME_GAP"] == 6
    assert profile.tracker_settings["STARRYNITE_MODEL_FILE"] == str(model_path)
    assert len(profile.parameter_sha256 or "") == 64
    assert len(profile.model_sha256 or "") == 64
    assert profile.warnings


def test_tuning_profile_explicit_stage_overrides_sparse_cell_count_inference() -> None:
    parameters = parse_parameter_text(
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[5,8,12];\n"
        "parameters.rangethreshold=[100,50,25];\n"
        "trackingparameters.temporalcutoff=[1,2,3];\n"
    )

    assert legacy_stage_index(parameters, 1) == 0

    profile = tuning_profile_from_parameters(
        parameters,
        cell_count=1,
        stage_index=2,
    )

    assert profile.cell_count == 1
    assert profile.stage_index == 2
    assert profile.detector_settings["STARRYNITE_CELL_COUNT"] == 1
    assert profile.detector_settings["STARRYNITE_STAGE_INDEX"] == 2
    assert profile.detector_settings["INTENSITY_THRESHOLD"] == pytest.approx(12.0)
    assert profile.detector_settings["RANGE_THRESHOLD"] == pytest.approx(25.0)
    assert profile.tracker_settings["MAX_FRAME_GAP"] == 3
    assert profile.tracker_settings["ALLOW_GAP_CLOSING"] is True


def test_load_tuning_profile_forwards_explicit_stage_override(tmp_path: Path) -> None:
    parameter_path = tmp_path / "parameters.txt"
    parameter_path.write_text(
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[5,8,12];\n",
        encoding="utf-8",
    )

    profile = load_tuning_profile(parameter_path, cell_count=1, stage_index=1)

    assert profile.stage_index == 1
    assert profile.detector_settings["STARRYNITE_STAGE_INDEX"] == 1
    assert profile.detector_settings["INTENSITY_THRESHOLD"] == pytest.approx(8.0)


@pytest.mark.parametrize("stage_index", [-1, 1.5, True, "1"])
def test_tuning_profile_rejects_invalid_explicit_stage_index(stage_index: object) -> None:
    parameters = parse_parameter_text(
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[5,8,12];\n"
    )

    with pytest.raises(StarryNitePresetError, match="non-negative integer"):
        tuning_profile_from_parameters(
            parameters,
            cell_count=1,
            stage_index=stage_index,  # type: ignore[arg-type]
        )


def test_tuning_profile_binds_selected_distribution_to_request_time_sha256(
    tmp_path: Path,
) -> None:
    distribution_path = tmp_path / "distribution.mat"
    distribution_path.write_bytes(b"immutable detector source")
    parameter_path = tmp_path / "parameters.m"
    parameter_path.write_text(
        "xyres=1;\n"
        "firsttimestepdiam=8;\n"
        "distribution_file='distribution.mat';\n",
        encoding="utf-8",
    )

    profile = load_tuning_profile(parameter_path)

    assert profile.detector_settings["STARRYNITE_DISTRIBUTION_FILE"] == str(
        distribution_path.resolve()
    )
    assert profile.detector_settings[
        "STARRYNITE_DISTRIBUTION_SOURCE_SHA256"
    ] == hashlib.sha256(distribution_path.read_bytes()).hexdigest()


def test_legacy_stage_selection_matches_strict_matlab_boundaries() -> None:
    parameters = parse_parameter_text("parameters.staging=[25,80,181];")

    assert legacy_stage_index(parameters, 25) == 0
    assert legacy_stage_index(parameters, 26) == 1
    assert legacy_stage_index(parameters, 80) == 1
    assert legacy_stage_index(parameters, 81) == 2
    assert select_stage_value("threshold", (4, 6, 8), 2) == 8


def test_stage_selection_matches_matlab_scalar_and_duplicate_boundary_semantics() -> None:
    parameters = parse_parameter_text("parameters.staging=[25,25,80];")

    # getParameter.m chooses max(find(staging < numcells)) + 1. Equal duplicate
    # boundaries therefore jump two stages as soon as the count exceeds them.
    assert legacy_stage_index(parameters, 25) == 0
    assert legacy_stage_index(parameters, 26) == 2
    assert legacy_stage_index(parameters, 80) == 2
    assert legacy_stage_index(parameters, 81) == 3
    assert select_stage_value("sigma", (1,), 999) == 1

    with pytest.raises(StarryNitePresetError, match="non-negative integer"):
        legacy_stage_index(parameters, 25.5)  # type: ignore[arg-type]


def test_exact_parameter_resolution_accepts_duplicate_staging_and_scalar_vector() -> None:
    parameters = parse_parameter_text(
        "parameters.staging=[25,25,80];\n"
        "parameters.split=[1,2,3,4];\n"
        "parameters.sigma=[7];\n"
    )

    jumped = resolve_legacy_parameter(parameters, "split", cell_count=26)
    scalar = resolve_legacy_parameter(parameters, "sigma", cell_count=81)

    assert jumped.stage_index == 2
    assert jumped.matlab_stage_index == 3
    assert jumped.value == 3
    assert scalar.stage_index == 3
    assert scalar.value == 7


def test_exact_regional_parameter_resolution_matches_getparameter_boundaries() -> None:
    parameters = parse_parameter_text(
        "downsample=2;\n"
        "parameters.staging=[25,80,181];\n"
        "parameters.rangethreshold=[100,20,10,5];\n"
        "parameters.regions=cell(4,1);\n"
        "parameters.regions{2}.rangethreshold=16;\n"
        "parameters.regions{2}.area=[10*downsample,20*downsample,0,30,2,6];\n"
    )

    table = build_legacy_region_table(parameters)
    assert table.capacity == 4
    assert table.consumed_record_indices
    assert table.issues == ()

    inside = resolve_legacy_parameter(
        parameters,
        "rangethreshold",
        cell_count=80,
        location=(21, 1, 3),
        region_table=table,
    )
    assert inside.stage_index == 1
    assert inside.base_value == 20
    assert inside.value == 16
    assert inside.regional_override_applied

    # MATLAB boxes use lower-exclusive and upper-inclusive comparisons.
    lower_edge = resolve_legacy_parameter(
        parameters,
        "parameters.rangethreshold",
        cell_count=80,
        location=(20, 1, 3),
        region_table=table,
    )
    upper_edge = resolve_legacy_parameter(
        parameters,
        "rangethreshold",
        cell_count=80,
        location=(40, 30, 6),
        region_table=table,
    )
    assert lower_edge.value == 20 and not lower_edge.regional_override_applied
    assert upper_edge.value == 16 and upper_edge.regional_override_applied

    # Equality with staging boundaries stays in the earlier stage.
    stage_one = resolve_legacy_parameter(
        parameters,
        "rangethreshold",
        cell_count=25,
        location=(21, 1, 3),
        region_table=table,
    )
    assert stage_one.stage_index == 0 and stage_one.value == 100


def test_regional_parameter_resolution_fails_closed_when_context_is_ambiguous() -> None:
    missing_location = parse_parameter_text(
        "parameters.staging=[25];\n"
        "parameters.split=[4,5];\n"
        "parameters.regions=cell(2,1);\n"
        "parameters.regions{1}.area=[0,10,0,10,0,10];\n"
        "parameters.regions{1}.split=9;\n"
    )
    with pytest.raises(LegacyParameterResolutionError, match="location is required"):
        resolve_legacy_parameter(missing_location, "split", cell_count=10)

    conditional = parse_parameter_text(
        "parameters.staging=[25];\n"
        "parameters.split=[4,5];\n"
        "if useSpecialRegion\n"
        "parameters.regions=cell(2,1);\n"
        "parameters.regions{1}.area=[0,10,0,10,0,10];\n"
        "parameters.regions{1}.split=9;\n"
        "end\n"
    )
    with pytest.raises(LegacyParameterResolutionError, match="control flow"):
        build_legacy_region_table(conditional)

    unresolved_expression = parse_parameter_text(
        "parameters.staging=[25];\n"
        "parameters.split=[4,5];\n"
        "parameters.regions=cell(2,1);\n"
        "parameters.regions{1}.area=[0,width,0,10,0,10];\n"
        "parameters.regions{1}.split=9;\n"
    )
    diagnostic = build_legacy_region_table(unresolved_expression, strict=False)
    assert diagnostic.issues
    with pytest.raises(LegacyParameterResolutionError, match="could not safely resolve"):
        build_legacy_region_table(unresolved_expression)

    dynamic_initializer = parse_parameter_text(
        "parameters.staging=[25];\n"
        "parameters.regions=cell(size(parameters.staging));\n"
    )
    with pytest.raises(LegacyParameterResolutionError, match="unsupported.*regions"):
        build_legacy_region_table(dynamic_initializer)


def test_parameter_file_reader_preserves_crlf_newlines(tmp_path: Path) -> None:
    path = tmp_path / "windows-parameters.txt"
    path.write_bytes(b"radius=4;\r\nthreshold=5;\r\n")

    parameters = read_parameter_file(path)

    assert parameters.source == "radius=4;\r\nthreshold=5;\r\n"
    assert parameters.render() == parameters.source


def test_tuning_save_plan_updates_only_active_stage_and_writes_atomically(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "standard.txt"
    source_path.write_text(
        "% retain this comment\n"
        "xyres=.25;\n"
        "firsttimestepdiam=40;\n"
        "parameters.staging=[25,80];\n"
        "parameters.intensitythreshold=[10,20,30];\n"
        "trackingparameters.temporalcutoff=[2,3,4];\n",
        encoding="utf-8",
    )
    profile = load_tuning_profile(source_path, cell_count=40)

    plan = build_tuning_save_plan(
        profile,
        radius_um=6.0,
        intensity_threshold=27.0,
        max_frame_gap=5,
    )
    destination = tmp_path / "standard-tuned.txt"
    written = write_parameter_file(
        profile.parameters,
        destination,
        overrides=plan.overrides,
    )
    reparsed = read_parameter_file(written)

    assert not plan.warnings
    assert reparsed.source.startswith(profile.parameters.source)
    assert reparsed.normalized_settings["parameters.intensitythreshold"] == (
        10,
        27.0,
        30,
    )
    assert reparsed.normalized_settings["trackingparameters.temporalcutoff"] == (
        2,
        5,
        4,
    )
    assert reparsed.normalized_settings["firsttimestepdiam"] == pytest.approx(48.0)


def test_tuning_save_plan_preserves_diameter_when_xy_resolution_is_unknown() -> None:
    parameters = parse_parameter_text("parameters.intensitythreshold=4;\n")

    plan = build_tuning_save_plan(
        tuning_profile_from_parameters(parameters),
        radius_um=5.0,
        intensity_threshold=6.0,
        max_frame_gap=2,
    )

    assert "firsttimestepdiam" not in plan.overrides
    assert plan.warnings


@pytest.mark.parametrize(
    ("matlab_class", "expected"),
    [
        ("ClassificationNaiveBayes", "new_classifier"),
        ("NaiveBayes", "legacy_classifier"),
    ],
)
def test_scipy_matlab_opaque_retains_inert_classifier_class(
    matlab_class: str,
    expected: str,
) -> None:
    scipy_matlab = pytest.importorskip("scipy.io.matlab")
    from acetree_py.tracking.starrynite import models as model_module

    payload = np.array(
        [("MCOS", matlab_class, np.array([1, 2, 3], dtype=np.uint32))],
        dtype=[("_TypeSystem", "O"), ("_Class", "O"), ("_ObjectMetadata", "O")],
    )
    opaque = scipy_matlab.MatlabOpaque(payload)

    normalized = model_module._normalize_matlab_value(
        opaque,
        max_array_elements=100,
        path="classifier",
    )

    assert isinstance(normalized, OpaqueMatlabValue)
    assert normalized.matlab_class == matlab_class
    assert identify_classifier_flavor({"classifier": normalized}).flavor == expected
