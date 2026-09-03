"""Safe subprocess bridge to a locally installed MATLAB StarryNite checkout."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat, savemat

from ...api import Calibration, TrackEdge
from ..classifier import (
    NeutralNaiveBayesClassifier,
    neutral_classifier_from_matlab_export,
    save_neutral_classifier,
)
from ..detector import legacy_dog_filter_parameters

if TYPE_CHECKING:
    from ..legacy_state import LegacyFeatureParameters, LegacyTrackingContext
    from .event_trace import TrackingEventTrace
    from .stage_trace import GeometryStageTrace


class MatlabOracleError(RuntimeError):
    """MATLAB ran but could not produce a valid successful oracle result."""


class MatlabOracleUnavailable(MatlabOracleError):
    """MATLAB or the user-supplied StarryNite checkout is unavailable."""


@dataclass(frozen=True, slots=True)
class MatlabOracleConfig:
    matlab_executable: Path
    starrynite_root: Path
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        executable = Path(self.matlab_executable).resolve(strict=False)
        root = Path(self.starrynite_root).resolve(strict=False)
        if not executable.is_file():
            raise MatlabOracleUnavailable(f"MATLAB executable was not found: {executable}")
        if not (root / "distribution_code").is_dir():
            raise MatlabOracleUnavailable(
                f"StarryNite distribution_code was not found under: {root}"
            )
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        object.__setattr__(self, "matlab_executable", executable)
        object.__setattr__(self, "starrynite_root", root)
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

    @classmethod
    def discover(
        cls,
        starrynite_root: str | Path,
        *,
        matlab_executable: str | Path | None = None,
        timeout_seconds: float = 300.0,
    ) -> MatlabOracleConfig:
        """Find MATLAB from an explicit value, environment, PATH, or standard install."""

        candidate = matlab_executable or os.environ.get("MATLAB_EXECUTABLE")
        if candidate is None:
            candidate = shutil.which("matlab")
        if candidate is None and os.name == "nt":
            installs = sorted(
                Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"),
                reverse=True,
            )
            candidate = installs[0] if installs else None
        if candidate is None:
            raise MatlabOracleUnavailable(
                "MATLAB was not found; pass --matlab or set MATLAB_EXECUTABLE"
            )
        return cls(Path(candidate), Path(starrynite_root), timeout_seconds)


@dataclass(frozen=True, slots=True)
class MatlabOracleRun:
    operation: str
    result: Mapping[str, Any]
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def matlab_version(self) -> str:
        return str(self.result.get("matlab_version", ""))

    def volume_zyx(self, field: str = "filtered_volume_yxz") -> np.ndarray:
        if field not in self.result:
            raise KeyError(f"MATLAB result did not contain {field!r}")
        value = np.asarray(self.result[field])
        if value.ndim != 3:
            raise MatlabOracleError(
                f"MATLAB result {field!r} has shape {value.shape}, expected YXZ"
            )
        return np.asarray(np.transpose(value, (2, 0, 1)), dtype=np.float32)

    def candidate_centers_zyx0(self) -> np.ndarray:
        return self.points_zyx0("candidate_centers_zyx_0based")

    def points_zyx0(self, field: str) -> np.ndarray:
        value = self.result.get(field, np.empty((0, 3)))
        points = np.asarray(value, dtype=np.float64)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        points = np.atleast_2d(points)
        if points.shape[1] != 3:
            raise MatlabOracleError(
                f"MATLAB {field} was not an N-by-3 table"
            )
        return points

    def node_table(self) -> np.ndarray:
        return self._table("node_table", 8)

    def edge_table(self) -> np.ndarray:
        return self._table("edge_table", 7)

    def legacy_node_measurement_table(self) -> np.ndarray:
        """Return the neutral per-node inputs used by the legacy extractor."""

        return self._table("legacy_node_measurements", 21)

    def legacy_self_nn_table(self) -> np.ndarray:
        """Return exact nearest rows and confidence-geometry intermediates."""

        return self._table("legacy_self_nn", 12)

    def extracted_bifurcations(self) -> Mapping[str, Any]:
        """Return MATLAB's retained-bifurcation 22/11/13 diagnostics."""

        value = self.result.get("extracted_bifurcations", {})
        if not isinstance(value, Mapping):
            raise MatlabOracleError(
                "MATLAB extracted_bifurcations result was not a structure"
            )
        return value

    def tracking_event_trace(self) -> TrackingEventTrace:
        """Decode ordered classifier checkpoints and intervening mutations."""

        from .event_trace import matlab_event_trace

        return matlab_event_trace(self)

    def tracking_stage_trace(self) -> GeometryStageTrace:
        """Decode staged geometry state preceding classifier/repair work."""

        from .stage_trace import matlab_geometry_stage_trace

        return matlab_geometry_stage_trace(self)

    def normalized_edges(
        self,
        *,
        include_deleted_sources: bool = False,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int], str], ...]:
        """Return ID-independent zero-based lineage edges from a tracking run."""

        kinds = {0: "link", 1: "gap", 2: "split"}
        edges = []
        for row in self.edge_table():
            if not include_deleted_sources and bool(row[6]):
                continue
            kind_code = int(row[4])
            if kind_code not in kinds:
                raise MatlabOracleError(f"Unknown MATLAB edge kind code: {kind_code}")
            edges.append(
                (
                    (int(row[0]), int(row[1])),
                    (int(row[2]), int(row[3])),
                    kinds[kind_code],
                )
            )
        return tuple(edges)

    def legacy_track_edges(
        self,
        *,
        include_deleted_sources: bool = True,
    ) -> tuple[TrackEdge, ...]:
        """Return ID-bearing edges while retaining MATLAB successor slots.

        ``normalize_tracking_result`` emits edges by source row and then suc
        slot 1/2.  Carrying that order into ``LEGACY_SUCCESSOR_SLOT`` avoids
        silently sorting daughters after StarryNite has deliberately swapped
        them during a repair.
        """

        kinds = {0: "link", 1: "gap", 2: "split"}
        next_slot: dict[tuple[int, int], int] = {}
        result: list[TrackEdge] = []
        for row in self.edge_table():
            source = (int(row[0]), int(row[1]))
            slot = next_slot.get(source, 0)
            next_slot[source] = slot + 1
            if slot > 1:
                raise MatlabOracleError(
                    f"MATLAB source {source} returned more than two successors"
                )
            if not include_deleted_sources and bool(row[6]):
                continue
            kind_code = int(row[4])
            if kind_code not in kinds:
                raise MatlabOracleError(
                    f"Unknown MATLAB edge kind code: {kind_code}"
                )
            target = (int(row[2]), int(row[3]))
            result.append(
                TrackEdge(
                    f"matlab:{source[0]}:{source[1]}",
                    f"matlab:{target[0]}:{target[1]}",
                    0.0,
                    kinds[kind_code],
                    {
                        "LEGACY_SUCCESSOR_SLOT": slot,
                        "FRAME_DELTA": int(row[5]),
                        "MATLAB_SOURCE_DELETED": bool(row[6]),
                    },
                )
            )
        return tuple(result)

    def legacy_tracking_context(
        self,
        parameters: LegacyFeatureParameters | None = None,
        *,
        validate_snapshots: bool = True,
    ) -> LegacyTrackingContext:
        """Rehydrate exact Python extractor state from a full-tracking run.

        The local imports keep the generic MATLAB bridge usable without
        coupling every oracle operation to the extractor implementation.
        ``parameters`` may be supplied explicitly for archived oracle files;
        current runs carry the required scalar summary themselves.
        """

        from ..legacy_state import (
            LegacyFeatureParameters,
            LegacyNucleus,
            LegacyTrackingContext,
        )

        if parameters is None:
            summary = self.result.get("tracking_parameter_summary")
            if not isinstance(summary, Mapping):
                raise MatlabOracleError(
                    "MATLAB result lacks tracking_parameter_summary"
                )
            required = (
                "interval",
                "candidateCutoff",
                "temporalcutoff",
                "temporalcutoffstart",
                "smallcutoff",
                "endtime",
                "anisotropyvector",
            )
            missing = [name for name in required if name not in summary]
            if missing:
                raise MatlabOracleError(
                    "MATLAB tracking summary lacks extractor parameter(s): "
                    + ", ".join(missing)
                )
            anisotropy = tuple(
                float(item)
                for item in np.asarray(summary["anisotropyvector"]).reshape(-1)
            )
            parameters = LegacyFeatureParameters(
                interval=float(summary["interval"]),
                candidate_cutoff=float(summary["candidateCutoff"]),
                temporal_cutoff=int(summary["temporalcutoff"]),
                temporal_cutoff_start=int(summary["temporalcutoffstart"]),
                small_cutoff=float(summary["smallcutoff"]),
                anisotropy_xyz=anisotropy,
                end_frame=int(summary["endtime"]),
                absolute_cutoff=bool(summary.get("abscutoff", False)),
            )
        if not isinstance(parameters, LegacyFeatureParameters):
            raise TypeError("parameters must be LegacyFeatureParameters or None")

        table = self.legacy_node_measurement_table()
        nuclei = tuple(
            LegacyNucleus(
                nucleus_id=f"matlab:{int(row[0])}:{int(row[1])}",
                frame=int(row[0]) + 1,
                matlab_row=int(row[1]),
                position_xyz=(float(row[2]), float(row[3]), float(row[4])),
                diameter=float(row[5]),
                total_gfp=float(row[6]),
                avg_gfp=float(row[7]),
                aspect_ratio=float(row[8]),
                log_odds_sum=float(row[9]),
                slice_count=int(row[10]),
                xy_principal_variance=float(row[11]),
                xy_secondary_variance=float(row[12]),
            )
            for row in table
        )
        deleted_ids = tuple(
            nucleus.nucleus_id
            for nucleus, row in zip(nuclei, table, strict=True)
            if bool(row[20])
        )
        context = LegacyTrackingContext.from_nuclei_and_edges(
            nuclei,
            self.legacy_track_edges(include_deleted_sources=True),
            parameters,
            deleted_ids=deleted_ids,
        )
        if validate_snapshots:
            validation_issues: list[str] = []
            nearest_table = self.legacy_self_nn_table()
            nearest_by_id = {
                f"matlab:{int(row[0])}:{int(row[1])}": (
                    f"matlab:{int(row[0])}:{int(row[2])}"
                )
                for row in nearest_table
            }
            nearest_diagnostics = {
                f"matlab:{int(row[0])}:{int(row[1])}": row[3:].tolist()
                for row in nearest_table
            }
            for nucleus, row in zip(nuclei, table, strict=True):
                expected_distance = float(row[13])
                actual_distance = context.self_distance(nucleus.nucleus_id)
                if not np.isclose(
                    actual_distance,
                    expected_distance,
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                ):
                    validation_issues.append(
                        f"selfdistance {nucleus.nucleus_id}: "
                        f"{actual_distance} != {expected_distance}"
                    )
                actual_confidence = np.asarray(
                    context.confidence_vector(nucleus.nucleus_id), dtype=float
                )
                expected = np.asarray(row[14:20], dtype=float)
                actual = np.asarray(actual_confidence, dtype=float)
                close = np.isclose(
                    actual,
                    expected,
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True,
                )
                # MATLAB's platform single-precision libm can select the
                # adjacent float for log(single(...)) even when all captured
                # arithmetic inputs agree bit for bit.  Only the two NN-log
                # columns receive this explicit one-ULP allowance; the four
                # algebraic columns retain the strict double comparison.
                log_ulp = _float32_ulp_distances(actual[1:3], expected[1:3])
                close[1:3] = log_ulp <= 1
                if not bool(np.all(close)):
                    differing = np.flatnonzero(~close)
                    validation_issues.append(
                        "confidence-vector "
                        f"{nucleus.nucleus_id}; columns={differing.tolist()}, "
                        f"nn_log_ulp={log_ulp.tolist()}, "
                        f"actual={actual.tolist()}, expected={expected.tolist()}, "
                        f"python_self_nn={context._self_nn_by_id[nucleus.nucleus_id]!r}, "
                        "matlab_self_nn="
                        f"{nearest_by_id.get(nucleus.nucleus_id)!r}, "
                        "matlab_geometry="
                        f"{nearest_diagnostics.get(nucleus.nucleus_id)!r}"
                    )
            if validation_issues:
                raise MatlabOracleError(
                    "Python legacy measurement reconstruction drifted in "
                    f"{len(validation_issues)} row(s): "
                    + " | ".join(validation_issues[:12])
                )
        return context

    def _table(self, field: str, columns: int) -> np.ndarray:
        value = np.asarray(self.result.get(field, np.empty((0, columns))), dtype=float)
        if value.size == 0:
            return np.empty((0, columns), dtype=float)
        table = np.atleast_2d(value)
        if table.shape[1] != columns:
            raise MatlabOracleError(
                f"MATLAB {field} has shape {table.shape}; expected N-by-{columns}"
            )
        return table

    def to_provenance(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "matlab_version": self.matlab_version,
            "upstream_function": str(self.result.get("upstream_function", "")),
            "duration_seconds": self.duration_seconds,
        }


class MatlabStarryNiteOracle:
    """Execute the checked-in adapter without evaluating legacy parameter files."""

    def __init__(self, config: MatlabOracleConfig) -> None:
        self.config = config
        self.wrapper_directory = Path(__file__).with_name("matlab")
        self.wrapper_path = self.wrapper_directory / "run_starrynite_matlab_oracle.m"
        if not self.wrapper_path.is_file():
            raise MatlabOracleUnavailable(f"MATLAB oracle wrapper is missing: {self.wrapper_path}")

    def run(self, request: Mapping[str, Any]) -> MatlabOracleRun:
        """Run one schema-v1 request in a fresh MATLAB batch process."""

        return self.run_many((request,))[0]

    def run_many(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> tuple[MatlabOracleRun, ...]:
        """Run a whole scenario/sweep matrix with one MATLAB startup."""

        if not requests:
            return ()
        normalized_requests: list[dict[str, Any]] = []
        operations: list[str] = []
        for request in requests:
            normalized = dict(request)
            normalized.setdefault("schema_version", np.uint32(1))
            operation = str(normalized.get("operation", ""))
            if not operation:
                raise ValueError("Every oracle request requires an operation")
            normalized_requests.append(normalized)
            operations.append(operation)

        with tempfile.TemporaryDirectory(
            prefix="at_starrynite_oracle_",
            ignore_cleanup_errors=True,
        ) as temporary:
            temporary_path = Path(temporary)
            request_paths = tuple(
                temporary_path / f"request_{index:05d}.mat"
                for index in range(len(normalized_requests))
            )
            result_paths = tuple(
                temporary_path / f"result_{index:05d}.mat"
                for index in range(len(normalized_requests))
            )
            for request_path, normalized in zip(
                request_paths, normalized_requests, strict=True
            ):
                savemat(
                    request_path,
                    {"request": normalized},
                    do_compression=False,
                    long_field_names=True,
                )
            manifest_path = temporary_path / "batch_manifest.mat"
            savemat(
                manifest_path,
                {
                    "request_paths": np.asarray(
                        [str(path) for path in request_paths], dtype=object
                    ),
                    "result_paths": np.asarray(
                        [str(path) for path in result_paths], dtype=object
                    ),
                },
                do_compression=False,
            )
            expression = (
                f"addpath('{_matlab_quote(self.wrapper_directory)}'); "
                "run_starrynite_matlab_oracle_batch("
                f"'{_matlab_quote(manifest_path)}',"
                f"'{_matlab_quote(self.config.starrynite_root)}')"
            )
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    [str(self.config.matlab_executable), "-batch", expression],
                    cwd=temporary_path,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise MatlabOracleError(
                    f"MATLAB oracle batch exceeded "
                    f"{self.config.timeout_seconds:g} seconds"
                ) from exc
            except OSError as exc:
                raise MatlabOracleUnavailable(f"Could not start MATLAB: {exc}") from exc
            duration = time.perf_counter() - started
            results = tuple(_load_result(path) for path in result_paths)
            failures = []
            for index, (operation, result) in enumerate(
                zip(operations, results, strict=True)
            ):
                if not bool(result.get("success", False)):
                    failures.append(
                        (
                            index,
                            operation,
                            str(result.get("error_identifier", "MATLAB process error")),
                            str(
                                result.get(
                                    "error_message", "No structured error was saved"
                                )
                            ),
                        )
                    )
            if completed.returncode != 0 or failures:
                output_tail = (completed.stderr or completed.stdout)[-4000:]
                if failures:
                    summary = "; ".join(
                        f"trial {index} {operation!r} ({identifier}): {message}"
                        for index, operation, identifier, message in failures
                    )
                else:
                    summary = f"MATLAB exited with status {completed.returncode}"
                raise MatlabOracleError(
                    f"MATLAB oracle batch failed: {summary}\n{output_tail}".rstrip()
                )
            return tuple(
                MatlabOracleRun(
                    operation=operation,
                    result=result,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    duration_seconds=float(result.get("elapsed_seconds", duration)),
                )
                for operation, result in zip(operations, results, strict=True)
            )

    def resolve_parameter(
        self,
        name: str,
        staging: Sequence[float],
        parameter_values: Sequence[float] | float,
        *,
        cell_count: float,
        location_xyz: Sequence[float],
        regional_stage_index: int | None = None,
        regional_area: Sequence[float] | None = None,
        regional_value: Sequence[float] | float | None = None,
    ) -> MatlabOracleRun:
        """Invoke upstream ``getParameter`` on an inert numeric request."""

        return self.run(
            parameter_resolution_request(
                name,
                staging,
                parameter_values,
                cell_count=cell_count,
                location_xyz=location_xyz,
                regional_stage_index=regional_stage_index,
                regional_area=regional_area,
                regional_value=regional_value,
            )
        )

    def separable_dog(
        self,
        volume_zyx: np.ndarray,
        *,
        radius_um: float,
        sigma_factor: float,
        calibration: Calibration,
    ) -> MatlabOracleRun:
        """Execute StarryNite's separable DoG with native-resolved kernel values."""

        return self.run(
            separable_dog_request(
                volume_zyx,
                radius_um=radius_um,
                sigma_factor=sigma_factor,
                calibration=calibration,
            )
        )

    def slice_candidates(
        self,
        filtered_volume_zyx: np.ndarray,
        *,
        maxima_threshold: float,
        cell_diameter_xy_px: float,
        anisotropy: float,
        num_cells: int = 0,
        legacy_parameters: Mapping[str, Any] | None = None,
    ) -> MatlabOracleRun:
        """Execute upstream 2-D disk extraction and 3-D candidate selection."""

        return self.run(
            slice_candidates_request(
                filtered_volume_zyx,
                maxima_threshold=maxima_threshold,
                cell_diameter_xy_px=cell_diameter_xy_px,
                anisotropy=anisotropy,
                num_cells=num_cells,
                legacy_parameters=legacy_parameters,
            )
        )

    def full_detection(
        self,
        volume_zyx: np.ndarray,
        *,
        radius_um: float,
        sigma_factor: float,
        intensity_threshold: float,
        boundary_percent: float,
        calibration: Calibration,
        num_cells: int = 0,
        legacy_parameters: Mapping[str, Any] | None = None,
        distribution_file: str | Path | None = None,
    ) -> MatlabOracleRun:
        """Execute the original ``processVolume.m`` detector script."""

        return self.run(
            full_detection_request(
                volume_zyx,
                radius_um=radius_um,
                sigma_factor=sigma_factor,
                intensity_threshold=intensity_threshold,
                boundary_percent=boundary_percent,
                calibration=calibration,
                num_cells=num_cells,
                legacy_parameters=legacy_parameters,
                distribution_file=distribution_file,
            )
        )

    def full_tracking(
        self,
        movie_tzyx: np.ndarray,
        *,
        radius_um: float,
        sigma_factor: float,
        intensity_threshold: float,
        boundary_percent: float,
        calibration: Calibration,
        num_cells: int = 0,
        use_static_diameter: bool = False,
        legacy_parameters: Mapping[str, Any] | None = None,
        tracking_overrides: Mapping[str, float] | None = None,
        distribution_file: str | Path | None = None,
        model_file: str | Path | None = None,
    ) -> MatlabOracleRun:
        """Execute original frame detection plus the legacy tracking model."""

        return self.run(
            full_tracking_request(
                movie_tzyx,
                radius_um=radius_um,
                sigma_factor=sigma_factor,
                intensity_threshold=intensity_threshold,
                boundary_percent=boundary_percent,
                calibration=calibration,
                num_cells=num_cells,
                use_static_diameter=use_static_diameter,
                legacy_parameters=legacy_parameters,
                tracking_overrides=tracking_overrides,
                distribution_file=distribution_file,
                model_file=model_file,
            )
        )

    def export_classifier_model(
        self,
        *,
        model_file: str | Path | None = None,
    ) -> MatlabOracleRun:
        """Export a neutral view of a user-supplied MATLAB classifier model."""

        return self.run(classifier_model_export_request(model_file=model_file))

    def export_neutral_classifier(
        self,
        destination: str | Path,
        model_file: str | Path | None = None,
    ) -> NeutralNaiveBayesClassifier:
        """Export, provenance-bind, and atomically save a neutral classifier."""

        output = Path(destination).resolve(strict=False)
        source = (
            Path(model_file)
            if model_file is not None
            else self.config.starrynite_root
            / "distribution_lineaging"
            / "2019TrackingModelv2.mat"
        ).resolve(strict=False)
        same_file = os.path.normcase(str(output)) == os.path.normcase(str(source))
        if not same_file and output.exists():
            try:
                same_file = os.path.samefile(output, source)
            except OSError:
                same_file = False
        if same_file:
            raise ValueError(
                "Neutral classifier destination cannot overwrite the source MAT model"
            )
        if output.suffix.lower() == ".mat":
            raise ValueError(
                "Neutral classifier destination must not use the .mat extension"
            )
        if not source.is_file():
            raise MatlabOracleUnavailable(
                f"StarryNite tracking model was not found: {source}"
            )
        source_hash = _sha256(source)
        run = self.export_classifier_model(model_file=source)
        metadata = run.result.get("source_model")
        if not isinstance(metadata, Mapping) or "path" not in metadata:
            raise MatlabOracleError(
                "Classifier export did not include source_model.path metadata"
            )
        returned_source = Path(str(metadata["path"])).resolve(strict=False)
        if returned_source != source:
            raise MatlabOracleError(
                "Classifier export source path does not match the requested model: "
                f"{returned_source} != {source}"
            )
        if _sha256(source) != source_hash:
            raise MatlabOracleError(
                "Tracking model changed while its neutral classifier was exported"
            )
        model = neutral_classifier_from_matlab_export(
            run.result,
            source_model_sha256=source_hash,
        )
        save_neutral_classifier(output, model)
        return model

    def predict_bifurcation(
        self,
        daughter_data: Sequence[float],
        back_data: Sequence[float],
        forward_data: Sequence[float],
        *,
        d1_length: float,
        d2_length: float,
        fn_back_candidate_1_length: float,
        fn_back_candidate_2_length: float,
        best_fn_forward_length_d1: float,
        best_fn_forward_length_d2: float,
        best_fn_back_correct: bool = False,
        best_index: int = 1,
        force_mode: bool = False,
        model_file: str | Path | None = None,
    ) -> MatlabOracleRun:
        """Invoke StarryNite's saved-model bifurcation classifier entry point."""

        return self.run(
            classifier_prediction_request(
                daughter_data,
                back_data,
                forward_data,
                d1_length=d1_length,
                d2_length=d2_length,
                fn_back_candidate_1_length=fn_back_candidate_1_length,
                fn_back_candidate_2_length=fn_back_candidate_2_length,
                best_fn_forward_length_d1=best_fn_forward_length_d1,
                best_fn_forward_length_d2=best_fn_forward_length_d2,
                best_fn_back_correct=best_fn_back_correct,
                best_index=best_index,
                force_mode=force_mode,
                model_file=model_file,
            )
        )

    def installation_provenance(self) -> dict[str, str]:
        """Return revision and hashes that identify the interoperability boundary."""

        provenance = {
            "matlab_executable_name": self.config.matlab_executable.name,
            "starrynite_revision": _git_revision(self.config.starrynite_root),
            "wrapper_sha256": _sha256(self.wrapper_path),
        }
        for label, relative_path in (
            ("upstream_process_volume_sha256", "distribution_code/processVolume.m"),
            (
                "upstream_tracking_driver_sha256",
                "distribution_lineaging/tracking_driver_new_classifier_based_version.m",
            ),
            (
                "upstream_distribution_sha256",
                "distribution_code/clean_distributions_newimage.mat",
            ),
            (
                "upstream_tracking_model_sha256",
                "distribution_lineaging/2019TrackingModelv2.mat",
            ),
        ):
            candidate = self.config.starrynite_root / relative_path
            if candidate.is_file():
                provenance[label] = _sha256(candidate)
        return provenance


def minimal_candidate_parameters() -> dict[str, np.ndarray | float]:
    """Conservative defaults required by upstream ``createDiskSet`` helpers."""

    return {
        "staging": np.asarray([1_000_000_000.0]),
        "boundary_percent": 0.35,
        "large_ray_threshold": 1.5,
        "small_ray_threshold": 1.0 / 3.0,
    }


def minimal_detector_parameters(
    *,
    sigma_factor: float,
    intensity_threshold: float,
    boundary_percent: float,
) -> dict[str, np.ndarray | float]:
    """Stable early-stage defaults sufficient for ``processVolume.m``."""

    result = minimal_candidate_parameters()
    result.update(
        {
            "sigma": _positive("sigma_factor", sigma_factor),
            "intensitythreshold": _finite(
                "intensity_threshold", intensity_threshold
            ),
            "boundary_percent": _positive("boundary_percent", boundary_percent),
            "rangethreshold": 100.0,
            "nndist_merge": 0.8,
            "mergelower": -300.0,
            "armerge": 1.6,
            "mergesplit": 1.0,
            "split": 100.0,
        }
    )
    return result


def separable_dog_request(
    volume_zyx: np.ndarray,
    *,
    radius_um: float,
    sigma_factor: float,
    calibration: Calibration,
) -> dict[str, Any]:
    volume = _volume_zyx(volume_zyx)
    parameters = legacy_dog_filter_parameters(
        radius_um,
        sigma_factor,
        calibration,
    )
    return {
        "operation": "separable_dog",
        "volume_yxz": np.transpose(volume, (1, 2, 0)),
        "inner_sigma_yxz": _zyx_to_yxz(parameters.inner_sigma_zyx),
        "inner_kernel_size_yxz": _zyx_to_yxz(parameters.inner_support_zyx),
        "outer_sigma_yxz": _zyx_to_yxz(parameters.outer_sigma_zyx),
        "outer_kernel_size_yxz": _zyx_to_yxz(parameters.outer_support_zyx),
    }


def parameter_resolution_request(
    name: str,
    staging: Sequence[float],
    parameter_values: Sequence[float] | float,
    *,
    cell_count: float,
    location_xyz: Sequence[float],
    regional_stage_index: int | None = None,
    regional_area: Sequence[float] | None = None,
    regional_value: Sequence[float] | float | None = None,
) -> dict[str, Any]:
    """Build a strict, non-executable golden query for MATLAB ``getParameter``."""

    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise ValueError("name must be one direct MATLAB struct field")
    request: dict[str, Any] = {
        "operation": "resolve_parameter",
        "parameter_name": name,
        "staging": _finite_vector("staging", staging),
        "parameter_values": _finite_vector("parameter_values", parameter_values),
        "num_cells": _finite("cell_count", cell_count),
        "location_xyz": _finite_vector(
            "location_xyz", location_xyz, expected_length=3
        ),
    }
    regional_values = (
        regional_stage_index,
        regional_area,
        regional_value,
    )
    if any(value is not None for value in regional_values):
        if not all(value is not None for value in regional_values):
            raise ValueError(
                "regional_stage_index, regional_area, and regional_value "
                "must be supplied together"
            )
        if (
            isinstance(regional_stage_index, bool)
            or int(regional_stage_index) != regional_stage_index
            or int(regional_stage_index) < 1
        ):
            raise ValueError("regional_stage_index must be a positive integer")
        area = _finite_vector(
            "regional_area", regional_area, expected_length=6  # type: ignore[arg-type]
        )
        if any(area[index] >= area[index + 1] for index in (0, 2, 4)):
            raise ValueError(
                "each regional_area lower bound must be smaller than its upper bound"
            )
        request.update(
            {
                "regional_stage_index": int(regional_stage_index),
                "regional_area": area,
                "regional_value": _finite_vector(
                    "regional_value", regional_value  # type: ignore[arg-type]
                ),
            }
        )
    return request


def slice_candidates_request(
    filtered_volume_zyx: np.ndarray,
    *,
    maxima_threshold: float,
    cell_diameter_xy_px: float,
    anisotropy: float,
    num_cells: int = 0,
    legacy_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parameters = minimal_candidate_parameters()
    parameters.update(dict(legacy_parameters or {}))
    return {
        "operation": "slice_candidates",
        "filtered_volume_yxz": np.transpose(
            _volume_zyx(filtered_volume_zyx), (1, 2, 0)
        ),
        "maxima_threshold": _finite("maxima_threshold", maxima_threshold),
        "cell_diameter_xy": _positive("cell_diameter_xy_px", cell_diameter_xy_px),
        "anisotropy": _positive("anisotropy", anisotropy),
        "num_cells": _nonnegative_integer("num_cells", num_cells),
        "legacy_parameters": _numeric_parameter_struct(parameters),
    }


def full_detection_request(
    volume_zyx: np.ndarray,
    *,
    radius_um: float,
    sigma_factor: float,
    intensity_threshold: float,
    boundary_percent: float,
    calibration: Calibration,
    num_cells: int = 0,
    legacy_parameters: Mapping[str, Any] | None = None,
    distribution_file: str | Path | None = None,
) -> dict[str, Any]:
    parameters = minimal_detector_parameters(
        sigma_factor=sigma_factor,
        intensity_threshold=intensity_threshold,
        boundary_percent=boundary_percent,
    )
    parameters.update(dict(legacy_parameters or {}))
    # Explicit trial values are the independent variables of a sweep and must
    # override the starting legacy parameter file rather than being masked by it.
    parameters.update(
        {
            "sigma": _positive("sigma_factor", sigma_factor),
            "intensitythreshold": _finite(
                "intensity_threshold", intensity_threshold
            ),
            "boundary_percent": _positive("boundary_percent", boundary_percent),
        }
    )
    radius = _positive("radius_um", radius_um)
    request: dict[str, Any] = {
        "operation": "full_detection",
        "volume_yxz": np.transpose(_volume_zyx(volume_zyx), (1, 2, 0)),
        "cell_diameter_xy": 2.0 * radius / calibration.xy_um,
        "anisotropy": calibration.z_um / calibration.xy_um,
        "num_cells": _nonnegative_integer("num_cells", num_cells),
        "legacy_parameters": _numeric_parameter_struct(parameters),
    }
    if distribution_file is not None:
        request["distribution_file"] = str(Path(distribution_file).resolve(strict=False))
    return request


def full_tracking_request(
    movie_tzyx: np.ndarray,
    *,
    radius_um: float,
    sigma_factor: float,
    intensity_threshold: float,
    boundary_percent: float,
    calibration: Calibration,
    num_cells: int = 0,
    use_static_diameter: bool = False,
    legacy_parameters: Mapping[str, Any] | None = None,
    tracking_overrides: Mapping[str, float] | None = None,
    distribution_file: str | Path | None = None,
    model_file: str | Path | None = None,
) -> dict[str, Any]:
    movie = np.asarray(movie_tzyx)
    if movie.ndim != 4 or any(size == 0 for size in movie.shape):
        raise ValueError("Oracle movie must be a non-empty TZYX array")
    if movie.shape[0] < 2 or movie.shape[1] < 4:
        raise ValueError("Oracle tracking movies require T >= 2 and Z >= 4")
    if not np.issubdtype(movie.dtype, np.number) or not np.all(np.isfinite(movie)):
        raise ValueError("Oracle movie must contain finite numeric values")
    parameters = minimal_detector_parameters(
        sigma_factor=sigma_factor,
        intensity_threshold=intensity_threshold,
        boundary_percent=boundary_percent,
    )
    parameters.update(dict(legacy_parameters or {}))
    parameters.update(
        {
            "sigma": _positive("sigma_factor", sigma_factor),
            "intensitythreshold": _finite(
                "intensity_threshold", intensity_threshold
            ),
            "boundary_percent": _positive("boundary_percent", boundary_percent),
        }
    )
    radius = _positive("radius_um", radius_um)
    if type(use_static_diameter) is not bool:
        raise TypeError("use_static_diameter must be a boolean")
    request: dict[str, Any] = {
        "operation": "full_tracking",
        "movie_yxzt": np.transpose(np.asarray(movie, dtype=np.float32), (2, 3, 1, 0)),
        "cell_diameter_xy": 2.0 * radius / calibration.xy_um,
        "anisotropy": calibration.z_um / calibration.xy_um,
        "num_cells": _nonnegative_integer("num_cells", num_cells),
        "use_static_diameter": use_static_diameter,
        "legacy_parameters": _numeric_parameter_struct(parameters),
    }
    if tracking_overrides:
        request["tracking_overrides"] = {
            str(name): _finite(f"tracking_overrides.{name}", value)
            for name, value in tracking_overrides.items()
        }
    if distribution_file is not None:
        request["distribution_file"] = str(Path(distribution_file).resolve(strict=False))
    if model_file is not None:
        request["model_file"] = str(Path(model_file).resolve(strict=False))
    return request


def classifier_model_export_request(
    *,
    model_file: str | Path | None = None,
) -> dict[str, Any]:
    """Build a request for a neutral, non-executable classifier model view."""

    request: dict[str, Any] = {"operation": "export_classifier_model"}
    if model_file is not None:
        request["model_file"] = str(Path(model_file).resolve(strict=False))
    return request


def classifier_prediction_request(
    daughter_data: Sequence[float],
    back_data: Sequence[float],
    forward_data: Sequence[float],
    *,
    d1_length: float,
    d2_length: float,
    fn_back_candidate_1_length: float,
    fn_back_candidate_2_length: float,
    best_fn_forward_length_d1: float,
    best_fn_forward_length_d2: float,
    best_fn_back_correct: bool = False,
    best_index: int = 1,
    force_mode: bool = False,
    model_file: str | Path | None = None,
) -> dict[str, Any]:
    """Build one strict request for ``predictBifurcationTypeSinglemodel``."""

    request: dict[str, Any] = {
        "operation": "predict_bifurcation",
        "daughter_data": _feature_row("daughter_data", daughter_data, 22),
        "back_data": _feature_row("back_data", back_data, 11),
        "forward_data": _feature_row("forward_data", forward_data, 13),
        "d1_length": _positive("d1_length", d1_length),
        "d2_length": _positive("d2_length", d2_length),
        "fn_back_candidate_1_length": _finite(
            "fn_back_candidate_1_length", fn_back_candidate_1_length
        ),
        "fn_back_candidate_2_length": _finite(
            "fn_back_candidate_2_length", fn_back_candidate_2_length
        ),
        "best_fn_forward_length_d1": _finite(
            "best_fn_forward_length_d1", best_fn_forward_length_d1
        ),
        "best_fn_forward_length_d2": _finite(
            "best_fn_forward_length_d2", best_fn_forward_length_d2
        ),
        "best_fn_back_correct": _boolean(
            "best_fn_back_correct", best_fn_back_correct
        ),
        "best_index": _classifier_index(best_index),
        "force_mode": _boolean("force_mode", force_mode),
    }
    if model_file is not None:
        request["model_file"] = str(Path(model_file).resolve(strict=False))
    return request


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write a stable JSON report while rejecting accidental non-finite values."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = loadmat(path, simplify_cells=True)
    result = loaded.get("result")
    if result is None:
        raise MatlabOracleError("MATLAB result MAT-file did not contain 'result'")
    if isinstance(result, Mapping):
        return {
            str(name): _normalize_loaded_value(value)
            for name, value in result.items()
        }
    if hasattr(result, "dtype") and getattr(result.dtype, "names", None):
        return {
            str(name): _normalize_loaded_value(result[name])
            for name in result.dtype.names
        }
    raise MatlabOracleError("MATLAB result was not a scalar structure")


def _normalize_loaded_value(value: Any) -> Any:
    """Recursively turn SciPy ``mat_struct`` values into inert Python data."""

    if isinstance(value, Mapping):
        return {
            str(name): _normalize_loaded_value(item)
            for name, item in value.items()
        }
    field_names = getattr(value, "_fieldnames", None)
    if field_names:
        return {
            str(name): _normalize_loaded_value(getattr(value, name))
            for name in field_names
        }
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim == 0:
            return _normalize_loaded_value(value.item())
        normalized = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            normalized[index] = _normalize_loaded_value(value[index])
        return normalized
    if isinstance(value, np.void) and value.dtype.names:
        return {
            str(name): _normalize_loaded_value(value[name])
            for name in value.dtype.names
        }
    return value


def _volume_zyx(value: np.ndarray) -> np.ndarray:
    volume = np.asarray(value)
    if volume.ndim != 3 or any(size == 0 for size in volume.shape):
        raise ValueError("Oracle volume must be a non-empty ZYX array")
    if not np.issubdtype(volume.dtype, np.number) or not np.all(np.isfinite(volume)):
        raise ValueError("Oracle volume must contain finite numeric values")
    if volume.shape[0] < 4:
        raise ValueError(
            "StarryNite treats fewer than four Z planes as a color image; use Z >= 4"
        )
    return np.asarray(volume, dtype=np.float32)


def _zyx_to_yxz(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("Kernel vectors must contain Z, Y, and X values")
    return vector[[1, 2, 0]]


def _numeric_parameter_struct(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
            raise ValueError(f"Legacy MATLAB parameter {name!r} must be a finite vector")
        result[str(name)] = array
    return result


def _feature_row(name: str, value: Sequence[float], length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.shape != (length,):
        raise ValueError(f"{name} must contain exactly {length} values")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} cannot contain infinite values")
    return array.reshape(1, length)


def _finite_vector(
    name: str,
    value: Sequence[float] | float,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite numeric vector")
    if expected_length is not None and len(array) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    return array


def _float32_ulp_distances(
    actual: Sequence[float] | np.ndarray,
    expected: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return elementwise binary32 ULP distances with NaN-pair equality."""

    left = np.asarray(actual, dtype=np.float32)
    right = np.asarray(expected, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError("ULP comparison arrays must have equal shapes")
    left_bits = left.view(np.uint32)
    right_bits = right.view(np.uint32)

    def ordered(bits: np.ndarray) -> np.ndarray:
        magnitude = (bits & np.uint32(0x7FFFFFFF)).astype(np.uint64)
        negative = (bits & np.uint32(0x80000000)) != 0
        return np.where(
            negative,
            np.uint64(0x80000000) - magnitude,
            np.uint64(0x80000000) + magnitude,
        )

    left_ordered = ordered(left_bits)
    right_ordered = ordered(right_bits)
    distance = np.maximum(left_ordered, right_ordered) - np.minimum(
        left_ordered,
        right_ordered,
    )
    equal = (left == right) | (np.isnan(left) & np.isnan(right))
    finite_pair = np.isfinite(left) & np.isfinite(right)
    return np.where(
        equal,
        np.uint64(0),
        np.where(finite_pair, distance, np.iinfo(np.uint64).max),
    )


def _matlab_quote(path: str | Path) -> str:
    return str(Path(path).resolve(strict=False)).replace("'", "''")


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _boolean(name: str, value: bool) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _classifier_index(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise ValueError("best_index must be -1 or a positive integer")
    index = int(value)
    if index == 0 or index < -1:
        raise ValueError("best_index must be -1 or a positive integer")
    return index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    git_directory = root / ".git"
    if git_directory.is_file():
        marker = git_directory.read_text(encoding="utf-8", errors="replace").strip()
        if marker.lower().startswith("gitdir:"):
            git_directory = (root / marker.split(":", 1)[1].strip()).resolve()
    head_path = git_directory / "HEAD"
    if not head_path.is_file():
        return "unknown"
    head = head_path.read_text(encoding="ascii", errors="replace").strip()
    if not head.startswith("ref:"):
        return head or "unknown"
    reference = head.split(":", 1)[1].strip()
    loose_reference = git_directory / reference
    if loose_reference.is_file():
        return loose_reference.read_text(encoding="ascii", errors="replace").strip()
    packed = git_directory / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="ascii", errors="replace").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            revision, name = line.split(" ", 1)
            if name == reference:
                return revision
    return "unknown"


__all__ = [
    "MatlabOracleConfig",
    "MatlabOracleError",
    "MatlabOracleRun",
    "MatlabOracleUnavailable",
    "MatlabStarryNiteOracle",
    "classifier_model_export_request",
    "classifier_prediction_request",
    "full_detection_request",
    "full_tracking_request",
    "minimal_candidate_parameters",
    "minimal_detector_parameters",
    "parameter_resolution_request",
    "separable_dog_request",
    "slice_candidates_request",
    "write_run_manifest",
]
