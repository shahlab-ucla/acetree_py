from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from acetree_py.tracking.starrynite.classifier import (
    GaussianFeatureDistribution,
    NeutralNaiveBayesSubmodel,
    SingleModelFeatureLayout,
)
from acetree_py.tracking.starrynite.oracle.matlab_backend import (
    MatlabOracleError,
    MatlabOracleRun,
)
from acetree_py.tracking.starrynite.oracle.model_export import (
    CLASSIFIER_EXPORT_MANIFEST_SCHEMA,
    ClassifierExportArtifact,
    compatible_classifier_export_request,
    export_classifier_numeric,
    validate_classifier_export_artifact,
)


def _submodel(
    name: str,
    predictor_count: int,
    labels: tuple[int, ...],
) -> NeutralNaiveBayesSubmodel:
    class_count = len(labels)
    costs = tuple(
        tuple(float(row != column) for column in range(class_count))
        for row in range(class_count)
    )
    return NeutralNaiveBayesSubmodel(
        classifier_family="legacy_classifier",
        feature_names=tuple(
            f"{name}_feature_{index:03d}" for index in range(predictor_count)
        ),
        class_labels=labels,
        class_priors=tuple(1.0 / class_count for _ in labels),
        misclassification_costs=costs,
        distributions=tuple(
            GaussianFeatureDistribution(
                means=tuple(float(index) for index in range(class_count)),
                standard_deviations=(1.0,) * class_count,
            )
            for _ in range(predictor_count)
        ),
    )


def _family_export(source: Path) -> dict[str, object]:
    layout = SingleModelFeatureLayout(
        daughter_keep=(True,) + (False,) * 21,
        backward_keep=(True,) + (False,) * 10,
        forward_keep=(True,) + (False,) * 12,
    )
    models = {
        "ambigious": _submodel("ambigious", 3, (0, 2, 3)),
        "fp_div": _submodel("fp_div", 1, (0, 1, 3)),
        "dirtyfp_fn": _submodel("dirtyfp_fn", 2, (0, 1, 2, 3)),
        "divfp": _submodel("divfp", 2, (0, 1, 3)),
    }
    submodels = {name: model.to_dict() for name, model in models.items()}
    # Match scipy.io's scalar collapse for a one-cell MATLAB feature/distribution
    # cell array and its ndarray representation for numeric vectors/matrices.
    fp_div = submodels["fp_div"]
    fp_div["feature_names"] = fp_div["feature_names"][0]
    fp_div["distributions"] = fp_div["distributions"][0]
    fp_div["class_labels"] = np.asarray(fp_div["class_labels"])
    fp_div["class_priors"] = np.asarray(fp_div["class_priors"])
    fp_div["misclassification_costs"] = np.asarray(
        fp_div["misclassification_costs"]
    )
    probes = {}
    for name, model in models.items():
        features = [
            [float(class_index)] * model.predictor_count
            for class_index in range(len(model.class_labels))
        ]
        features.extend(
            [float(class_index) + 0.5] * model.predictor_count
            for class_index in range(len(model.class_labels))
        )
        probes[name] = {
            "features": features,
            "predicted_classes": [model.predict(row) for row in features],
            "posteriors": [model.posterior(row) for row in features],
            "class_labels": model.class_labels,
        }
    return {
        "success": True,
        "export_schema": "acetree.starrynite-matlab-classifier-export",
        "export_version": 1,
        "matlab_version": "8.6.0.267246 (R2015b)",
        "matlab_release": "R2015b",
        "model_kind": "ambigious",
        "export_complete": True,
        "classifier_matlab_classes": ["NaiveBayes"] * 4,
        "source_model": {
            "path": str(source),
            "bytes": source.stat().st_size,
        },
        "validation_probes": probes,
        "classifier_family_model": {
            "family_name": "ambigious",
            "daughter_keep": layout.daughter_keep,
            "back_keep": layout.backward_keep,
            "forward_keep": layout.forward_keep,
            "submodels": submodels,
        },
    }


def test_compatible_export_request_is_absolute_and_fail_closed(tmp_path: Path) -> None:
    request = compatible_classifier_export_request(
        tmp_path / "model.mat",
        expected_kind="ambigious",
    )

    assert request["operation"] == "export_classifier_numeric_compatible"
    assert request["expected_kind"] == "ambigious"
    assert Path(str(request["model_file"])).is_absolute()
    with pytest.raises(ValueError, match="auto, single, or ambigious"):
        compatible_classifier_export_request(tmp_path / "model.mat", expected_kind="guess")


def test_family_export_writes_source_bound_model_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "StarryNite"
    (root / "distribution_code").mkdir(parents=True)
    source = root / "legacy-family.mat"
    source.write_bytes(b"historical MATLAB model")
    executable = tmp_path / "matlab.exe"
    executable.write_bytes(b"placeholder")
    destination = tmp_path / "family.json"

    def fake_run(*args, **kwargs):
        return MatlabOracleRun(
            operation="export_classifier_numeric_compatible",
            result=_family_export(source),
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )

    monkeypatch.setattr(
        "acetree_py.tracking.starrynite.oracle.model_export."
        "run_compatible_classifier_export",
        fake_run,
    )

    artifact = export_classifier_numeric(
        matlab_executable=executable,
        starrynite_root=root,
        model_file=source,
        destination=destination,
        expected_kind="ambigious",
    )

    assert isinstance(artifact, ClassifierExportArtifact)
    assert artifact.model_kind == "ambigious"
    payload = validate_classifier_export_artifact(destination)
    assert payload["schema"] == CLASSIFIER_EXPORT_MANIFEST_SCHEMA
    assert payload["source_model"]["sha256"] == artifact.source_model_sha256
    assert payload["neutral_model"]["sha256"] == artifact.neutral_model_sha256
    assert json.loads(destination.read_text(encoding="utf-8"))["submodels"]


def test_artifact_validation_rejects_changed_neutral_or_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "StarryNite"
    (root / "distribution_code").mkdir(parents=True)
    source = root / "legacy-family.mat"
    source.write_bytes(b"historical MATLAB model")
    executable = tmp_path / "matlab.exe"
    executable.write_bytes(b"placeholder")
    destination = tmp_path / "family.json"
    monkeypatch.setattr(
        "acetree_py.tracking.starrynite.oracle.model_export."
        "run_compatible_classifier_export",
        lambda *args, **kwargs: MatlabOracleRun(
            operation="export_classifier_numeric_compatible",
            result=_family_export(source),
            stdout="",
            stderr="",
            duration_seconds=0.1,
        ),
    )
    export_classifier_numeric(
        matlab_executable=executable,
        starrynite_root=root,
        model_file=source,
        destination=destination,
    )

    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MatlabOracleError, match="no longer matches"):
        validate_classifier_export_artifact(destination)

    # Restore through a new export, then prove the source binding is active too.
    export_classifier_numeric(
        matlab_executable=executable,
        starrynite_root=root,
        model_file=source,
        destination=destination,
    )
    source.write_bytes(b"changed historical MATLAB model")
    with pytest.raises(MatlabOracleError, match="Source MAT file"):
        validate_classifier_export_artifact(destination)


def test_artifact_validation_rejects_semantic_manifest_or_model_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "StarryNite"
    (root / "distribution_code").mkdir(parents=True)
    source = root / "legacy-family.mat"
    source.write_bytes(b"historical MATLAB model")
    executable = tmp_path / "matlab.exe"
    executable.write_bytes(b"placeholder")
    destination = tmp_path / "family.json"
    monkeypatch.setattr(
        "acetree_py.tracking.starrynite.oracle.model_export."
        "run_compatible_classifier_export",
        lambda *args, **kwargs: MatlabOracleRun(
            operation="export_classifier_numeric_compatible",
            result=_family_export(source),
            stdout="",
            stderr="",
            duration_seconds=0.1,
        ),
    )
    artifact = export_classifier_numeric(
        matlab_executable=executable,
        starrynite_root=root,
        model_file=source,
        destination=destination,
    )
    manifest_path = artifact.manifest_path
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    wrong_kind = json.loads(json.dumps(original_manifest))
    wrong_kind["model_kind"] = "single"
    write_json(manifest_path, wrong_kind)
    with pytest.raises(MatlabOracleError, match="model kind or source MAT identity"):
        validate_classifier_export_artifact(destination)

    wrong_runtime_class = json.loads(json.dumps(original_manifest))
    wrong_runtime_class["export_runtime"]["classifier_matlab_classes"] = [
        "ClassificationNaiveBayes"
    ]
    write_json(manifest_path, wrong_runtime_class)
    with pytest.raises(MatlabOracleError, match="model kind disagrees"):
        validate_classifier_export_artifact(destination)

    wrong_size = json.loads(json.dumps(original_manifest))
    wrong_size["neutral_model"]["bytes"] += 1
    write_json(manifest_path, wrong_size)
    with pytest.raises(MatlabOracleError, match="size"):
        validate_classifier_export_artifact(destination)

    # Rehashing a semantically altered JSON must not bypass its binding to the
    # unchanged MAT source recorded by the manifest.
    neutral = json.loads(destination.read_text(encoding="utf-8"))
    neutral["source_model_sha256"] = "f" * 64
    write_json(destination, neutral)
    rebound_manifest = json.loads(json.dumps(original_manifest))
    rebound_manifest["neutral_model"]["bytes"] = destination.stat().st_size
    rebound_manifest["neutral_model"]["sha256"] = hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()
    write_json(manifest_path, rebound_manifest)
    with pytest.raises(MatlabOracleError, match="source MAT identity"):
        validate_classifier_export_artifact(destination)


def test_export_rejects_matlab_source_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "StarryNite"
    (root / "distribution_code").mkdir(parents=True)
    source = root / "legacy-family.mat"
    source.write_bytes(b"historical MATLAB model")
    executable = tmp_path / "matlab.exe"
    executable.write_bytes(b"placeholder")
    result = _family_export(source)
    result["source_model"]["bytes"] = source.stat().st_size + 1
    monkeypatch.setattr(
        "acetree_py.tracking.starrynite.oracle.model_export."
        "run_compatible_classifier_export",
        lambda *args, **kwargs: MatlabOracleRun(
            operation="export_classifier_numeric_compatible",
            result=result,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        ),
    )

    with pytest.raises(MatlabOracleError, match="byte size"):
        export_classifier_numeric(
            matlab_executable=executable,
            starrynite_root=root,
            model_file=source,
            destination=tmp_path / "family.json",
        )


def test_export_rejects_numeric_payload_that_changes_matlab_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "StarryNite"
    (root / "distribution_code").mkdir(parents=True)
    source = root / "legacy-family.mat"
    source.write_bytes(b"historical MATLAB model")
    executable = tmp_path / "matlab.exe"
    executable.write_bytes(b"placeholder")
    result = _family_export(source)
    result["validation_probes"]["fp_div"]["posteriors"][0] = [0.8, 0.1, 0.1]
    monkeypatch.setattr(
        "acetree_py.tracking.starrynite.oracle.model_export."
        "run_compatible_classifier_export",
        lambda *args, **kwargs: MatlabOracleRun(
            operation="export_classifier_numeric_compatible",
            result=result,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        ),
    )

    with pytest.raises(MatlabOracleError, match="changed.*posterior"):
        export_classifier_numeric(
            matlab_executable=executable,
            starrynite_root=root,
            model_file=source,
            destination=tmp_path / "family.json",
        )


def test_modern_handoff_failure_requests_a_current_matlab_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "StarryNite"
    (root / "distribution_code").mkdir(parents=True)
    source = root / "modern.mat"
    source.write_bytes(b"modern MATLAB model")
    executable = tmp_path / "matlab.exe"
    executable.write_bytes(b"placeholder")
    result = {
        "success": True,
        "export_schema": "acetree.starrynite-matlab-classifier-export",
        "export_version": 1,
        "matlab_version": "8.6.0.267246 (R2015b)",
        "matlab_release": "R2015b",
        "model_kind": "single",
        "export_complete": False,
        "classifier_matlab_classes": ["ClassificationNaiveBayes"],
        "source_model": {
            "path": str(source),
            "bytes": source.stat().st_size,
        },
    }
    monkeypatch.setattr(
        "acetree_py.tracking.starrynite.oracle.model_export."
        "run_compatible_classifier_export",
        lambda *args, **kwargs: MatlabOracleRun(
            operation="export_classifier_numeric_compatible",
            result=result,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        ),
    )
    monkeypatch.setattr(
        "acetree_py.tracking.starrynite.oracle.model_export."
        "MatlabStarryNiteOracle.export_classifier_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            MatlabOracleError("-batch is unavailable")
        ),
    )

    with pytest.raises(MatlabOracleError, match="R2019a or newer"):
        export_classifier_numeric(
            matlab_executable=executable,
            starrynite_root=root,
            model_file=source,
            destination=tmp_path / "modern.json",
        )


@pytest.mark.matlab_oracle
def test_live_version_aware_command_exports_and_revalidates_modern_model(
    tmp_path: Path,
) -> None:
    root_value = os.environ.get("STARRYNITE_ROOT")
    executable_value = os.environ.get("MATLAB_EXECUTABLE")
    if not root_value or not executable_value:
        pytest.skip("STARRYNITE_ROOT and MATLAB_EXECUTABLE are required")
    root = Path(root_value)
    model_file = root / "distribution_lineaging" / "2019TrackingModelv2.mat"
    if not model_file.is_file():
        pytest.skip("Bundled modern StarryNite classifier was not found")
    destination = tmp_path / "modern-classifier.json"

    artifact = export_classifier_numeric(
        matlab_executable=executable_value,
        starrynite_root=root,
        model_file=model_file,
        destination=destination,
        expected_kind="single",
        timeout_seconds=180.0,
    )
    manifest = validate_classifier_export_artifact(destination)

    assert artifact.model_kind == "single"
    assert manifest["export_runtime"]["numeric_export_path"] == "current_oracle"
    assert manifest["neutral_model"]["sha256"] == artifact.neutral_model_sha256
