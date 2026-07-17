"""Version-aware, fail-closed MATLAB classifier numeric export workflow.

The normal differential oracle intentionally targets current MATLAB releases
and launches them with ``-batch``.  StarryNite's historical ``NaiveBayes``
objects require older MATLAB versions, some of which predate that switch and
APIs used by the main oracle wrapper.  This module therefore has a deliberately
small compatibility bridge that exchanges only v7 MAT files and launches
MATLAB through the older ``-nodesktop -nosplash -r`` interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import savemat

from ..classifier import (
    NeutralClassifierFormatError,
    NeutralAmbigiousClassifierFamily,
    NeutralNaiveBayesClassifier,
    load_neutral_ambigious_classifier,
    load_neutral_classifier,
    neutral_ambigious_classifier_from_matlab_export,
    neutral_classifier_from_matlab_export,
    save_neutral_ambigious_classifier,
    save_neutral_classifier,
)
from .matlab_backend import (
    MatlabOracleConfig,
    MatlabOracleError,
    MatlabOracleRun,
    MatlabOracleUnavailable,
    MatlabStarryNiteOracle,
    _git_revision,
    _load_result,
    _matlab_quote,
)


CLASSIFIER_EXPORT_MANIFEST_SCHEMA = (
    "acetree.starrynite-classifier-numeric-export-manifest"
)
CLASSIFIER_EXPORT_MANIFEST_VERSION = 1
MATLAB_CLASSIFIER_EXPORT_SCHEMA = "acetree.starrynite-matlab-classifier-export"
MATLAB_CLASSIFIER_EXPORT_VERSION = 1
_EXPORT_KINDS = {"auto", "single", "ambigious"}


@dataclass(frozen=True, slots=True)
class ClassifierExportArtifact:
    """Paths and validated identity of one completed numeric export."""

    model_path: Path
    manifest_path: Path
    model_kind: str
    source_model_sha256: str
    neutral_model_sha256: str
    matlab_version: str
    matlab_release: str


def compatible_classifier_export_request(
    model_file: str | Path,
    *,
    expected_kind: str = "auto",
) -> dict[str, Any]:
    """Build the old-release-compatible classifier export request."""

    kind = str(expected_kind).strip().lower()
    if kind not in _EXPORT_KINDS:
        raise ValueError("expected_kind must be auto, single, or ambigious")
    source = Path(model_file).resolve(strict=False)
    return {
        "schema_version": np.uint32(1),
        "operation": "export_classifier_numeric_compatible",
        "model_file": str(source),
        "expected_kind": kind,
    }


def run_compatible_classifier_export(
    matlab_executable: str | Path,
    request: Mapping[str, Any],
    *,
    timeout_seconds: float = 300.0,
) -> MatlabOracleRun:
    """Run the small compatibility helper under an old or current MATLAB."""

    executable = Path(matlab_executable).resolve(strict=False)
    if not executable.is_file():
        raise MatlabOracleUnavailable(f"MATLAB executable was not found: {executable}")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    helper = Path(__file__).with_name("matlab") / "export_starrynite_classifier_numeric.m"
    if not helper.is_file():
        raise MatlabOracleUnavailable(
            f"MATLAB compatible classifier exporter is missing: {helper}"
        )

    normalized = dict(request)
    normalized.setdefault("schema_version", np.uint32(1))
    if normalized.get("operation") != "export_classifier_numeric_compatible":
        raise ValueError(
            "Compatible classifier export request has an unsupported operation"
        )

    with tempfile.TemporaryDirectory(
        prefix="at_starrynite_model_export_",
        ignore_cleanup_errors=True,
    ) as temporary:
        temporary_path = Path(temporary)
        request_path = temporary_path / "request.mat"
        result_path = temporary_path / "result.mat"
        savemat(
            request_path,
            {"request": normalized},
            do_compression=False,
            long_field_names=True,
        )
        expression = (
            f"addpath('{_matlab_quote(helper.parent)}'); "
            "try, export_starrynite_classifier_numeric("
            f"'{_matlab_quote(request_path)}','{_matlab_quote(result_path)}'); "
            "catch exception, disp(getReport(exception,'extended')); exit(1); end; "
            "exit(0);"
        )
        command = [str(executable), "-nodesktop", "-nosplash"]
        if os.name == "nt":
            command.append("-wait")
        command.extend(("-r", expression))
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=temporary_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MatlabOracleError(
                f"MATLAB classifier export exceeded {timeout:g} seconds"
            ) from exc
        except OSError as exc:
            raise MatlabOracleUnavailable(f"Could not start MATLAB: {exc}") from exc
        elapsed = time.perf_counter() - started
        result = _load_result(result_path)
        success = bool(result.get("success", False))
        if completed.returncode != 0 or not success:
            identifier = str(result.get("error_identifier", "MATLAB process error"))
            message = str(
                result.get("error_message", "No structured export error was saved")
            )
            output_tail = (completed.stderr or completed.stdout)[-4000:]
            raise MatlabOracleError(
                f"MATLAB compatible classifier export failed ({identifier}): "
                f"{message}\n{output_tail}".rstrip()
            )
        return MatlabOracleRun(
            operation="export_classifier_numeric_compatible",
            result=result,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=float(result.get("elapsed_seconds", elapsed)),
        )


def export_classifier_numeric(
    *,
    matlab_executable: str | Path,
    starrynite_root: str | Path,
    model_file: str | Path,
    destination: str | Path,
    expected_kind: str = "auto",
    timeout_seconds: float = 300.0,
    manifest_path: str | Path | None = None,
) -> ClassifierExportArtifact:
    """Export, validate, source-bind, and save one StarryNite classifier.

    The source MAT file is hashed before MATLAB starts and again after the
    numeric payload has passed Python's neutral-model validation.  The export
    is refused if the source changes, if MATLAB reports another source path or
    byte size, or if the requested single/family shape is not exact.
    """

    source = Path(model_file).resolve(strict=False)
    output = Path(destination).resolve(strict=False)
    root = Path(starrynite_root).resolve(strict=False)
    executable = Path(matlab_executable).resolve(strict=False)
    if not source.is_file():
        raise MatlabOracleUnavailable(f"StarryNite tracking model was not found: {source}")
    _validate_export_destination(source, output)
    provenance_path = (
        Path(manifest_path).resolve(strict=False)
        if manifest_path is not None
        else output.with_name(output.name + ".provenance.json")
    )
    if os.path.normcase(str(provenance_path)) in {
        os.path.normcase(str(source)),
        os.path.normcase(str(output)),
    }:
        raise ValueError("Export manifest must not overwrite the source or model JSON")
    if not (root / "distribution_code").is_dir():
        raise MatlabOracleUnavailable(
            f"StarryNite distribution_code was not found under: {root}"
        )

    source_stat = source.stat()
    source_hash = _sha256(source)
    request = compatible_classifier_export_request(
        source,
        expected_kind=expected_kind,
    )
    compatibility_run = run_compatible_classifier_export(
        executable,
        request,
        timeout_seconds=timeout_seconds,
    )
    _validate_compatible_result(
        compatibility_run.result,
        source=source,
        source_size=source_stat.st_size,
        expected_kind=str(request["expected_kind"]),
    )
    model_kind = str(compatibility_run.result["model_kind"])
    export_complete = bool(compatibility_run.result["export_complete"])
    run = compatibility_run

    model: NeutralNaiveBayesClassifier | NeutralAmbigiousClassifierFamily
    if model_kind == "single" and not export_complete:
        config = MatlabOracleConfig(
            executable,
            root,
            timeout_seconds=timeout_seconds,
        )
        try:
            run = MatlabStarryNiteOracle(config).export_classifier_model(
                model_file=source
            )
        except MatlabOracleError as exc:
            raise MatlabOracleError(
                "The compatibility helper identified a modern "
                "ClassificationNaiveBayes model, but the selected MATLAB could "
                "not run the current oracle exporter. Use MATLAB R2019a or newer "
                "for modern models; reserve the older executable for historical "
                f"NaiveBayes MAT files. Current-export error: {exc}"
            ) from exc
        _validate_source_metadata(
            run.result.get("source_model"),
            source=source,
            source_size=source_stat.st_size,
        )
        model = neutral_classifier_from_matlab_export(
            run.result,
            source_model_sha256=source_hash,
        )
    elif model_kind == "single" and export_complete:
        model = neutral_classifier_from_matlab_export(
            compatibility_run.result,
            source_model_sha256=source_hash,
        )
    elif model_kind == "ambigious" and export_complete:
        model = neutral_ambigious_classifier_from_matlab_export(
            compatibility_run.result,
            source_model_sha256=source_hash,
        )
    else:  # pragma: no cover - guarded by the MATLAB/Python schema checks
        raise MatlabOracleError(
            "MATLAB did not return a complete supported classifier payload"
        )

    if export_complete:
        validation_probes = compatibility_run.result.get("validation_probes")
        if model_kind == "single":
            _validate_prediction_probes(
                model,
                validation_probes,
                label="single classifier",
            )
        else:
            if not isinstance(validation_probes, Mapping):
                raise MatlabOracleError(
                    "MATLAB ambigious-family export omitted validation probes"
                )
            assert isinstance(model, NeutralAmbigiousClassifierFamily)
            for name, submodel in model.submodels.items():
                _validate_prediction_probes(
                    submodel,
                    validation_probes.get(name),
                    label=f"ambigious family member {name}",
                )

    if source.stat().st_size != source_stat.st_size or _sha256(source) != source_hash:
        raise MatlabOracleError(
            "Tracking model changed while its neutral classifier was exported"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, NeutralAmbigiousClassifierFamily):
        save_neutral_ambigious_classifier(output, model)
    else:
        save_neutral_classifier(output, model)
    neutral_hash = _sha256(output)

    matlab_version = str(compatibility_run.result.get("matlab_version", ""))
    matlab_release = str(compatibility_run.result.get("matlab_release", ""))
    manifest = {
        "schema": CLASSIFIER_EXPORT_MANIFEST_SCHEMA,
        "version": CLASSIFIER_EXPORT_MANIFEST_VERSION,
        "model_kind": model_kind,
        "source_model": {
            "path": str(source),
            "sha256": source_hash,
            "bytes": source_stat.st_size,
            "modified_time_ns": source_stat.st_mtime_ns,
        },
        "neutral_model": {
            "path": str(output),
            "sha256": neutral_hash,
            "bytes": output.stat().st_size,
        },
        "export_runtime": {
            "matlab_executable": str(executable),
            "matlab_version": matlab_version,
            "matlab_release": matlab_release,
            "compatibility_helper_sha256": _sha256(
                Path(__file__).with_name("matlab")
                / "export_starrynite_classifier_numeric.m"
            ),
            "numeric_export_path": (
                "compatible_legacy_helper" if export_complete else "current_oracle"
            ),
            "classifier_matlab_classes": _text_values(
                compatibility_run.result.get("classifier_matlab_classes")
            ),
            "requested_kind": str(request["expected_kind"]),
        },
        "starrynite": {
            "root": str(root),
            "revision": _git_revision(root),
        },
    }
    _write_json_atomic(provenance_path, manifest)
    return ClassifierExportArtifact(
        model_path=output,
        manifest_path=provenance_path,
        model_kind=model_kind,
        source_model_sha256=source_hash,
        neutral_model_sha256=neutral_hash,
        matlab_version=matlab_version,
        matlab_release=matlab_release,
    )


def validate_classifier_export_artifact(
    model_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Fail closed unless a saved neutral model still matches its manifest."""

    model = Path(model_path).resolve(strict=False)
    manifest = (
        Path(manifest_path).resolve(strict=False)
        if manifest_path is not None
        else model.with_name(model.name + ".provenance.json")
    )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatlabOracleError(f"Could not read classifier export manifest: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise MatlabOracleError("Classifier export manifest must be a JSON object")
    if payload.get("schema") != CLASSIFIER_EXPORT_MANIFEST_SCHEMA:
        raise MatlabOracleError("Classifier export manifest has an unsupported schema")
    if payload.get("version") != CLASSIFIER_EXPORT_MANIFEST_VERSION:
        raise MatlabOracleError("Classifier export manifest has an unsupported version")
    neutral = payload.get("neutral_model")
    source = payload.get("source_model")
    if not isinstance(neutral, Mapping) or not isinstance(source, Mapping):
        raise MatlabOracleError("Classifier export manifest is missing model identity")
    model_kind = payload.get("model_kind")
    if model_kind not in {"single", "ambigious"}:
        raise MatlabOracleError(
            "Classifier export manifest has an unsupported model_kind"
        )
    if Path(str(neutral.get("path", ""))).resolve(strict=False) != model:
        raise MatlabOracleError("Classifier export manifest refers to another model path")
    if not model.is_file():
        raise MatlabOracleError("Neutral classifier JSON no longer matches its manifest")
    if (
        _manifest_file_size(neutral.get("bytes"), "neutral_model.bytes")
        != model.stat().st_size
    ):
        raise MatlabOracleError("Neutral classifier JSON size no longer matches its manifest")
    if neutral.get("sha256") != _sha256(model):
        raise MatlabOracleError("Neutral classifier JSON no longer matches its manifest")
    source_path = Path(str(source.get("path", ""))).resolve(strict=False)
    if not source_path.is_file():
        raise MatlabOracleError("Source MAT file no longer matches the numeric export")
    if (
        _manifest_file_size(source.get("bytes"), "source_model.bytes")
        != source_path.stat().st_size
    ):
        raise MatlabOracleError("Source MAT file size no longer matches the numeric export")
    source_hash = source.get("sha256")
    if source_hash != _sha256(source_path):
        raise MatlabOracleError("Source MAT file no longer matches the numeric export")

    try:
        if model_kind == "single":
            loaded_model = load_neutral_classifier(
                model,
                expected_source_model_sha256=str(source_hash),
            )
        else:
            loaded_model = load_neutral_ambigious_classifier(
                model,
                expected_source_model_sha256=str(source_hash),
            )
    except (FileNotFoundError, OSError, NeutralClassifierFormatError, ValueError) as exc:
        raise MatlabOracleError(
            "Neutral classifier JSON does not match the manifest model kind or "
            f"source MAT identity: {exc}"
        ) from exc

    runtime = payload.get("export_runtime")
    if not isinstance(runtime, Mapping):
        raise MatlabOracleError("Classifier export manifest is missing export_runtime")
    raw_classes = runtime.get("classifier_matlab_classes")
    if not isinstance(raw_classes, list) or any(
        type(item) is not str for item in raw_classes
    ):
        raise MatlabOracleError(
            "Classifier export manifest has invalid classifier MATLAB classes"
        )
    matlab_classes = tuple(raw_classes)
    if model_kind == "ambigious":
        expected_classes = ("NaiveBayes",) * 4
        expected_family = "legacy_classifier"
    elif matlab_classes == ("NaiveBayes",):
        expected_classes = ("NaiveBayes",)
        expected_family = "legacy_classifier"
    else:
        expected_classes = ("ClassificationNaiveBayes",)
        expected_family = "new_classifier"
    if matlab_classes != expected_classes:
        raise MatlabOracleError(
            "Classifier export manifest model kind disagrees with its MATLAB classes"
        )
    if loaded_model.classifier_family != expected_family:
        raise MatlabOracleError(
            "Neutral classifier family disagrees with the manifest MATLAB class"
        )
    return payload


def _validate_compatible_result(
    result: Mapping[str, Any],
    *,
    source: Path,
    source_size: int,
    expected_kind: str,
) -> None:
    if result.get("export_schema") != MATLAB_CLASSIFIER_EXPORT_SCHEMA:
        raise MatlabOracleError("MATLAB classifier export returned an unknown schema")
    if int(result.get("export_version", -1)) != MATLAB_CLASSIFIER_EXPORT_VERSION:
        raise MatlabOracleError("MATLAB classifier export returned an unknown version")
    kind = str(result.get("model_kind", ""))
    if kind not in {"single", "ambigious"}:
        raise MatlabOracleError(f"MATLAB returned unsupported model_kind {kind!r}")
    if expected_kind != "auto" and kind != expected_kind:
        raise MatlabOracleError(
            f"MATLAB model kind is {kind!r}, not requested {expected_kind!r}"
        )
    export_complete = _matlab_logical_scalar(
        result.get("export_complete"),
        "export_complete",
    )
    matlab_classes = _text_values(result.get("classifier_matlab_classes"))
    expected_classes = (
        ["NaiveBayes"] * 4
        if kind == "ambigious"
        else ["NaiveBayes" if export_complete else "ClassificationNaiveBayes"]
    )
    if matlab_classes != expected_classes:
        raise MatlabOracleError(
            "MATLAB classifier class provenance disagrees with the exported "
            f"model kind: received {matlab_classes}, expected {expected_classes}"
        )
    if kind == "ambigious" and not export_complete:
        raise MatlabOracleError(
            "Historical ambigious-family export did not contain numeric state"
        )
    if export_complete and not isinstance(
        result.get("validation_probes"), Mapping
    ):
        raise MatlabOracleError(
            "MATLAB numeric classifier export omitted prediction validation probes"
        )
    _validate_source_metadata(
        result.get("source_model"),
        source=source,
        source_size=source_size,
    )
    if not str(result.get("matlab_version", "")).strip():
        raise MatlabOracleError("MATLAB classifier export omitted matlab_version")


def _validate_source_metadata(
    metadata: Any,
    *,
    source: Path,
    source_size: int,
) -> None:
    if not isinstance(metadata, Mapping):
        raise MatlabOracleError("Classifier export omitted source_model metadata")
    returned_source = Path(str(metadata.get("path", ""))).resolve(strict=False)
    if returned_source != source:
        raise MatlabOracleError(
            "Classifier export source path does not match the requested model: "
            f"{returned_source} != {source}"
        )
    try:
        returned_size = int(metadata.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise MatlabOracleError("Classifier export source byte size is invalid") from exc
    if returned_size != source_size:
        raise MatlabOracleError(
            "Classifier export source byte size does not match the requested model"
        )


def _matlab_logical_scalar(value: Any, label: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise MatlabOracleError(f"MATLAB {label} must be scalar logical")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, np.integer, float, np.floating)) and float(
        scalar
    ) in {0.0, 1.0}:
        # scipy.io represents logical scalars from older v7 MAT files as
        # uint8; values other than exact zero/one remain invalid.
        return bool(scalar)
    raise MatlabOracleError(f"MATLAB {label} must be scalar logical")


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    array = np.asarray(value, dtype=object)
    return [str(item) for item in array.reshape(-1)]


def _validate_prediction_probes(
    model: Any,
    probes: Any,
    *,
    label: str,
) -> None:
    """Replay MATLAB predictions through inert numeric state before saving."""

    if not isinstance(probes, Mapping):
        raise MatlabOracleError(f"MATLAB {label} export omitted validation probes")
    required = {"features", "predicted_classes", "posteriors", "class_labels"}
    if set(probes) != required:
        raise MatlabOracleError(
            f"MATLAB {label} validation probes have unexpected fields"
        )
    try:
        features = np.asarray(probes["features"], dtype=np.float64)
        expected_classes = np.asarray(
            probes["predicted_classes"], dtype=np.float64
        ).reshape(-1)
        expected_posteriors = np.asarray(
            probes["posteriors"], dtype=np.float64
        )
        expected_labels = np.asarray(
            probes["class_labels"], dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise MatlabOracleError(
            f"MATLAB {label} validation probes are not numeric"
        ) from exc
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if expected_posteriors.ndim == 1:
        expected_posteriors = expected_posteriors.reshape(1, -1)
    class_labels = np.asarray(model.class_labels, dtype=np.float64)
    expected_shape = (features.shape[0], len(model.class_labels))
    if (
        features.ndim != 2
        or features.shape[0] != 2 * len(model.class_labels)
        or features.shape[1] != model.predictor_count
        or expected_classes.shape != (features.shape[0],)
        or expected_posteriors.shape != expected_shape
        or expected_labels.shape != class_labels.shape
        or not np.array_equal(expected_labels, class_labels)
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(expected_classes))
        or not np.all(np.isfinite(expected_posteriors))
        or np.any(expected_posteriors < 0)
    ):
        raise MatlabOracleError(
            f"MATLAB {label} validation probe shapes or values are invalid"
        )
    if not np.allclose(
        expected_posteriors.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1e-7,
    ):
        raise MatlabOracleError(
            f"MATLAB {label} validation posterior rows do not sum to one"
        )
    for index, row in enumerate(features):
        actual_posterior = np.asarray(model.posterior(row), dtype=np.float64)
        if not np.allclose(
            actual_posterior,
            expected_posteriors[index],
            rtol=2e-7,
            atol=2e-9,
        ):
            difference = float(
                np.max(np.abs(actual_posterior - expected_posteriors[index]))
            )
            raise MatlabOracleError(
                f"Numeric export changed {label} posterior at validation probe "
                f"{index}: maximum absolute difference {difference:.6g}"
            )
        actual_class = int(model.predict(row))
        expected_class = float(expected_classes[index])
        if not expected_class.is_integer() or actual_class != int(expected_class):
            raise MatlabOracleError(
                f"Numeric export changed {label} class at validation probe "
                f"{index}: Python {actual_class}, MATLAB {expected_class:g}"
            )


def _validate_export_destination(source: Path, output: Path) -> None:
    same_file = os.path.normcase(str(source)) == os.path.normcase(str(output))
    if not same_file and output.exists():
        try:
            same_file = os.path.samefile(source, output)
        except OSError:
            same_file = False
    if same_file:
        raise ValueError("Neutral classifier destination cannot overwrite the source MAT")
    if output.suffix.lower() == ".mat":
        raise ValueError("Neutral classifier destination must not use the .mat extension")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_file_size(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MatlabOracleError(f"Classifier export manifest {label} must be non-negative")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI for one-command old/new StarryNite classifier conversion."""

    parser = argparse.ArgumentParser(
        description=(
            "Export a StarryNite MATLAB classifier to validated inert numeric JSON. "
            "The exporter supports historical NaiveBayes runtimes without -batch."
        )
    )
    parser.add_argument(
        "--starrynite-root",
        default=os.environ.get("STARRYNITE_ROOT"),
        help="StarryNite checkout (or STARRYNITE_ROOT).",
    )
    parser.add_argument(
        "--matlab",
        default=os.environ.get("MATLAB_EXECUTABLE"),
        help="Compatible matlab executable (or MATLAB_EXECUTABLE).",
    )
    parser.add_argument("--model", required=True, type=Path, help="Source model MAT file.")
    parser.add_argument("--output", required=True, type=Path, help="Neutral model JSON.")
    parser.add_argument(
        "--kind",
        choices=tuple(sorted(_EXPORT_KINDS)),
        default="auto",
        help="Require a single model or legacy ambigious family (default: auto).",
    )
    parser.add_argument("--manifest", type=Path, help="Optional provenance JSON path.")
    parser.add_argument("--timeout", type=float, default=300.0)
    arguments = parser.parse_args(argv)
    if not arguments.starrynite_root:
        parser.error("--starrynite-root or STARRYNITE_ROOT is required")
    if not arguments.matlab:
        parser.error("--matlab or MATLAB_EXECUTABLE is required")
    try:
        artifact = export_classifier_numeric(
            matlab_executable=arguments.matlab,
            starrynite_root=arguments.starrynite_root,
            model_file=arguments.model,
            destination=arguments.output,
            expected_kind=arguments.kind,
            timeout_seconds=arguments.timeout,
            manifest_path=arguments.manifest,
        )
    except (OSError, ValueError, MatlabOracleError) as exc:
        parser.exit(2, f"StarryNite classifier export failed: {exc}\n")
    print(f"Exported {artifact.model_kind} classifier: {artifact.model_path}")
    print(f"Provenance: {artifact.manifest_path}")
    return 0


__all__ = [
    "CLASSIFIER_EXPORT_MANIFEST_SCHEMA",
    "CLASSIFIER_EXPORT_MANIFEST_VERSION",
    "ClassifierExportArtifact",
    "compatible_classifier_export_request",
    "export_classifier_numeric",
    "run_compatible_classifier_export",
    "validate_classifier_export_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
