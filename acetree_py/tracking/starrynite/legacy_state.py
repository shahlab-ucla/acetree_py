"""Immutable StarryNite state used by exact legacy feature extraction.

The MATLAB tracker stores more than its final lineage edges.  Feature and
repair-candidate extraction also depends on the original nucleus row order,
per-frame nearest-neighbour distances, directed nearest-neighbour claimants,
and two ordered successor slots.  This module reconstructs that shared state
without coupling it to classifier or graph-repair policy.

``matlab_row`` is deliberately zero based in Python.  Ordering by it is
equivalent to MATLAB's first-index ``min`` tie behaviour while avoiding a
mixture of indexing conventions inside the Python implementation.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..api import Detection, TrackEdge


class LegacyStateError(ValueError):
    """Raised when legacy feature state is incomplete or inconsistent."""


_MEASUREMENT_FIELDS = frozenset(
    {
        "total_gfp",
        "avg_gfp",
        "aspect_ratio",
        "log_odds_sum",
        "slice_count",
        "xy_principal_variance",
        "xy_secondary_variance",
    }
)


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LegacyStateError(f"{label} must be finite")
    if positive and result <= 0:
        raise LegacyStateError(f"{label} must be positive")
    return result


def _measurement_number(
    value: Any,
    label: str,
    *,
    nonnegative: bool = False,
) -> float:
    """Validate one scalar measurement while retaining an unavailable NaN."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if math.isinf(result):
        raise LegacyStateError(f"{label} cannot be infinite")
    if not math.isnan(result) and nonnegative and result < 0:
        raise LegacyStateError(f"{label} cannot be negative")
    return result


def _matlab_single_geometry(value: Any, label: str) -> float:
    """Round one legacy geometry scalar exactly as MATLAB ``single``.

    ``processVolume.m`` commits ``finalpoints`` and ``finaldiams`` to binary32
    before any tracking features or candidate distances are calculated.  The
    public :class:`Detection` remains double precision; only conversion into
    the legacy state applies this rounding boundary.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LegacyStateError(f"{label} must be finite")
    try:
        return struct.unpack("=f", struct.pack("=f", result))[0]
    except OverflowError as exc:
        raise LegacyStateError(f"{label} is outside MATLAB single range") from exc


def _matlab_single_zero_based_coordinate(value: Any, label: str) -> float:
    """Cast a native zero-based coordinate through MATLAB's 1-based domain."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite number")
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise LegacyStateError(f"{label} must be finite")
    # Native detector VOXEL_* values index NumPy arrays from zero.  MATLAB's
    # merged positions are one based when processVolume casts finalpoints to
    # single, and the adapter translates them back afterward.  Since binary32
    # rounding is not translation invariant, the ordering here is observable.
    return _matlab_single_geometry(coordinate + 1.0, label) - 1.0


def _matlab_single_arithmetic(value: float) -> float:
    """Round an arithmetic intermediate to binary32, retaining Inf/NaN."""

    try:
        return struct.unpack("=f", struct.pack("=f", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, float(value))


def legacy_single_round(value: float) -> float:
    """Expose one MATLAB ``single`` arithmetic boundary to sibling modules."""

    return _matlab_single_arithmetic(value)


def legacy_single_log(value: float) -> float:
    """Evaluate MATLAB ``log(single(value))`` at the binary32 boundary.

    The input and output are both committed to binary32; the intermediate
    transcendental evaluation retains enough precision to reproduce the live
    MATLAB scalar path on the supported platform.
    """

    single = _matlab_single_arithmetic(value)
    return _matlab_single_arithmetic(math.log(single))


def _matlab_single_sum(values: Iterable[float]) -> float:
    result = _matlab_single_arithmetic(0.0)
    for value in values:
        result = _matlab_single_arithmetic(result + value)
    return result


def legacy_single_sum(values: Iterable[float]) -> float:
    """Expose MATLAB single-precision accumulation to sibling modules."""

    return _matlab_single_sum(values)


def legacy_single_dot(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Evaluate MATLAB single ``a' * b``/``dot(a, b)`` semantics.

    MATLAB routes a single-precision matrix product through its native dot
    kernel. Its fused accumulation can differ from rounding each multiply and
    add separately, including by enough to change a Gram distance by one ULP.
    """

    left = np.asarray(tuple(first), dtype=np.float32)
    right = np.asarray(tuple(second), dtype=np.float32)
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        raise LegacyStateError(
            "Legacy dot inputs must have equal one-dimensional shapes"
        )
    if left.size == 0:
        raise LegacyStateError("Legacy dot inputs cannot be empty")
    with np.errstate(over="ignore", invalid="ignore"):
        return float(left @ right)


def legacy_single_mean(values: Sequence[float]) -> float:
    """Return MATLAB's binary32 mean for a legacy ``single`` vector."""

    numeric = tuple(float(value) for value in values)
    if not numeric:
        return math.nan
    return _matlab_single_arithmetic(
        _matlab_single_sum(numeric) / len(numeric)
    )


def legacy_gram_distance(
    first: Sequence[float],
    second: Sequence[float],
    anisotropy: Sequence[float] | None = None,
    *,
    zero_based: bool = True,
) -> float:
    """Evaluate StarryNite ``distance[_anisotropic]`` in MATLAB single.

    ``finalpoints`` are one based when MATLAB stores them as ``single``.  AT's
    compatibility state exposes zero-based coordinates, so the default path
    reconstructs that one-based binary32 representation before applying the
    Gram expression.  Callers with already reconstructed intermediates (the
    mixed-coordinate triple feature) can set ``zero_based=False``.
    """

    left_values = tuple(float(value) for value in first)
    right_values = tuple(float(value) for value in second)
    if len(left_values) != len(right_values) or not left_values:
        raise LegacyStateError(
            "Legacy distance inputs must have equal, non-zero dimensionality"
        )
    if anisotropy is None:
        scales = (1.0,) * len(left_values)
    else:
        scales = tuple(float(value) for value in anisotropy)
        if len(scales) != len(left_values):
            raise LegacyStateError(
                "Legacy distance anisotropy must match input dimensionality"
            )
        if any(not math.isfinite(value) for value in scales):
            raise LegacyStateError("Legacy distance anisotropy must be finite")

    translation = 1.0 if zero_based else 0.0
    left = tuple(
        _matlab_single_arithmetic(
            _matlab_single_arithmetic(value + translation) * scale
        )
        for value, scale in zip(left_values, scales, strict=True)
    )
    right = tuple(
        _matlab_single_arithmetic(
            _matlab_single_arithmetic(value + translation) * scale
        )
        for value, scale in zip(right_values, scales, strict=True)
    )
    aa = _matlab_single_sum(
        _matlab_single_arithmetic(value * value) for value in left
    )
    bb = _matlab_single_sum(
        _matlab_single_arithmetic(value * value) for value in right
    )
    ab = legacy_single_dot(left, right)
    doubled = _matlab_single_arithmetic(2.0 * ab)
    squared = _matlab_single_arithmetic(
        _matlab_single_arithmetic(aa + bb) - doubled
    )
    return _matlab_single_arithmetic(math.sqrt(abs(squared)))


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise LegacyStateError(f"{label} must be at least {minimum}")
    return result


def _feature_value(
    detection: Detection,
    key: str,
    *,
    unavailable: float = math.nan,
) -> Any:
    value = detection.features.get(key, unavailable)
    return unavailable if value is None else value


@dataclass(frozen=True, slots=True)
class LegacyNucleus:
    """Scalar per-nucleus measurements needed by the legacy tracker.

    Measurements that are unavailable from an older detector may be NaN (or
    zero for ``slice_count``).  The feature extractor can then call
    :meth:`require_measurements` only for predictors selected by the loaded
    model.  Position, diameter, frame, and row are always mandatory because
    candidate topology cannot be reconstructed without them.
    """

    nucleus_id: str
    frame: int
    matlab_row: int
    position_xyz: tuple[float, float, float]
    diameter: float
    total_gfp: float
    avg_gfp: float
    aspect_ratio: float
    log_odds_sum: float
    slice_count: int
    xy_principal_variance: float
    xy_secondary_variance: float = math.nan

    def __post_init__(self) -> None:
        if type(self.nucleus_id) is not str or not self.nucleus_id:
            raise TypeError("nucleus_id must be a non-empty string")
        object.__setattr__(self, "frame", _integer(self.frame, "frame", minimum=1))
        object.__setattr__(
            self,
            "matlab_row",
            _integer(self.matlab_row, "matlab_row", minimum=0),
        )
        try:
            position = tuple(self.position_xyz)
        except TypeError as exc:
            raise TypeError("position_xyz must contain three numbers") from exc
        if len(position) != 3:
            raise LegacyStateError("position_xyz must contain exactly three values")
        object.__setattr__(
            self,
            "position_xyz",
            tuple(
                _finite_number(value, f"position_xyz[{index}]")
                for index, value in enumerate(position)
            ),
        )
        object.__setattr__(
            self,
            "diameter",
            _finite_number(self.diameter, "diameter", positive=True),
        )
        for name, nonnegative in (
            ("total_gfp", True),
            ("avg_gfp", True),
            ("aspect_ratio", True),
            ("log_odds_sum", False),
            ("xy_principal_variance", True),
            ("xy_secondary_variance", True),
        ):
            object.__setattr__(
                self,
                name,
                _measurement_number(
                    getattr(self, name),
                    name,
                    nonnegative=nonnegative,
                ),
            )
        object.__setattr__(
            self,
            "slice_count",
            _integer(self.slice_count, "slice_count", minimum=0),
        )

    @classmethod
    def from_detection(
        cls,
        detection: Detection,
        *,
        required_measurements: Iterable[str] = (),
    ) -> LegacyNucleus:
        """Build exact scalar state from a StarryNite detector result.

        No approximate TrackMate-style fallback is used for legacy-only
        values.  Missing measurements remain explicit and are rejected only
        if named in ``required_measurements``.
        """

        if not isinstance(detection, Detection):
            raise TypeError("detection must be a Detection")
        required_keys = (
            "LEGACY_ROW_INDEX",
            "VOXEL_X",
            "VOXEL_Y",
            "VOXEL_Z",
            "LEGACY_DIAMETER_XY_PX",
        )
        missing = [key for key in required_keys if detection.features.get(key) is None]
        if missing:
            raise LegacyStateError(
                "Detection is missing legacy topology measurement(s): "
                + ", ".join(missing)
            )
        slice_value = _feature_value(
            detection,
            "LEGACY_SLICE_COUNT",
            unavailable=0,
        )
        # StarryNite processVolume.m:149-151 stores finalaveragepoints,
        # finalpoints, and finaldiams as MATLAB single arrays.  Apply that
        # boundary here rather than lowering the precision of generic
        # Detection coordinates or detector diagnostic features.
        legacy_x = _matlab_single_zero_based_coordinate(
            _feature_value(detection, "VOXEL_X"),
            "VOXEL_X",
        )
        legacy_y = _matlab_single_zero_based_coordinate(
            _feature_value(detection, "VOXEL_Y"),
            "VOXEL_Y",
        )
        legacy_z = _matlab_single_zero_based_coordinate(
            _feature_value(detection, "VOXEL_Z"),
            "VOXEL_Z",
        )
        legacy_diameter = _matlab_single_geometry(
            _feature_value(detection, "LEGACY_DIAMETER_XY_PX"),
            "LEGACY_DIAMETER_XY_PX",
        )
        nucleus = cls(
            nucleus_id=detection.detection_id,
            frame=detection.frame,
            matlab_row=_feature_value(detection, "LEGACY_ROW_INDEX"),
            position_xyz=(legacy_x, legacy_y, legacy_z),
            diameter=legacy_diameter,
            total_gfp=_feature_value(detection, "LEGACY_TOTAL_GFP"),
            avg_gfp=_feature_value(detection, "LEGACY_AVG_GFP"),
            aspect_ratio=_feature_value(detection, "LEGACY_ASPECT_RATIO"),
            log_odds_sum=_feature_value(detection, "LEGACY_LOG_ODDS_SUM"),
            slice_count=slice_value,
            xy_principal_variance=_feature_value(
                detection,
                "LEGACY_XY_PRINCIPAL_VARIANCE",
            ),
            xy_secondary_variance=_feature_value(
                detection,
                "LEGACY_XY_SECONDARY_VARIANCE",
            ),
        )
        nucleus.require_measurements(*tuple(required_measurements))
        return nucleus

    @property
    def detection_id(self) -> str:
        """Compatibility alias for code working with :class:`Detection`."""

        return self.nucleus_id

    @property
    def x(self) -> float:
        return self.position_xyz[0]

    @property
    def y(self) -> float:
        return self.position_xyz[1]

    @property
    def z(self) -> float:
        return self.position_xyz[2]

    def measurement_available(self, name: str) -> bool:
        if name not in _MEASUREMENT_FIELDS:
            raise KeyError(f"Unknown legacy measurement {name!r}")
        if name == "slice_count":
            return self.slice_count > 0
        return math.isfinite(float(getattr(self, name)))

    def require_measurements(self, *names: str) -> LegacyNucleus:
        missing = [name for name in names if not self.measurement_available(name)]
        if missing:
            raise LegacyStateError(
                f"Nucleus {self.nucleus_id!r} lacks required legacy "
                "measurement(s): " + ", ".join(missing)
            )
        return self


@dataclass(frozen=True, slots=True)
class LegacyFeatureParameters:
    """Validated parameters that affect legacy features and candidates."""

    interval: float
    candidate_cutoff: float
    temporal_cutoff: int
    temporal_cutoff_start: int
    small_cutoff: float
    anisotropy_xyz: tuple[float, float, float]
    end_frame: int
    absolute_cutoff: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interval",
            _finite_number(self.interval, "interval", positive=True),
        )
        object.__setattr__(
            self,
            "candidate_cutoff",
            _finite_number(
                self.candidate_cutoff,
                "candidate_cutoff",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "temporal_cutoff",
            _integer(self.temporal_cutoff, "temporal_cutoff", minimum=1),
        )
        object.__setattr__(
            self,
            "temporal_cutoff_start",
            _integer(
                self.temporal_cutoff_start,
                "temporal_cutoff_start",
                minimum=1,
            ),
        )
        if self.temporal_cutoff_start > self.temporal_cutoff:
            raise LegacyStateError(
                "temporal_cutoff_start cannot exceed temporal_cutoff"
            )
        object.__setattr__(
            self,
            "small_cutoff",
            _finite_number(self.small_cutoff, "small_cutoff", positive=True),
        )
        try:
            anisotropy = tuple(self.anisotropy_xyz)
        except TypeError as exc:
            raise TypeError("anisotropy_xyz must contain three numbers") from exc
        if len(anisotropy) != 3:
            raise LegacyStateError(
                "anisotropy_xyz must contain exactly three values"
            )
        object.__setattr__(
            self,
            "anisotropy_xyz",
            tuple(
                _finite_number(
                    value,
                    f"anisotropy_xyz[{index}]",
                    positive=True,
                )
                for index, value in enumerate(anisotropy)
            ),
        )
        object.__setattr__(
            self,
            "end_frame",
            _integer(self.end_frame, "end_frame", minimum=1),
        )
        if type(self.absolute_cutoff) is not bool:
            raise TypeError("absolute_cutoff must be a boolean")

    @classmethod
    def from_model(
        cls,
        model: Mapping[str, Any] | Any,
        *,
        anisotropy_xyz: Sequence[Real] | None = None,
        end_frame: int | None = None,
    ) -> LegacyFeatureParameters:
        """Load exact extractor controls from a decoded MATLAB model.

        ``model`` may be the ``trackingparameters`` struct itself, a decoded
        :class:`StarryNiteModel`, or its top-level ``fields`` mapping.  Dataset
        anisotropy/end-frame overrides are explicit because the tracking
        driver replaces those two model values for each run.
        """

        source: Any = model
        fields = getattr(source, "fields", None)
        if isinstance(fields, Mapping):
            source = fields
        if isinstance(source, Mapping) and "trackingparameters" in source:
            source = source["trackingparameters"]

        def read(name: str, *, default: Any = None, optional: bool = False) -> Any:
            if isinstance(source, Mapping):
                if name in source:
                    return source[name]
            else:
                try:
                    return getattr(source, name)
                except AttributeError:
                    pass
            if optional:
                return default
            raise LegacyStateError(
                f"trackingparameters is missing required field {name!r}"
            )

        resolved_anisotropy = (
            tuple(anisotropy_xyz)
            if anisotropy_xyz is not None
            else tuple(read("anisotropyvector"))
        )
        resolved_end = end_frame if end_frame is not None else read("endtime")

        def integer_field(name: str, value: Any) -> int:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise LegacyStateError(
                    f"trackingparameters.{name} must be an integer"
                )
            number = float(value)
            if not math.isfinite(number) or not number.is_integer():
                raise LegacyStateError(
                    f"trackingparameters.{name} must be an integer"
                )
            return int(number)

        absolute = read("abscutoff", default=False, optional=True)
        if isinstance(absolute, Real) and not isinstance(absolute, bool):
            absolute = bool(float(absolute))
        return cls(
            interval=read("interval"),
            candidate_cutoff=read("candidateCutoff"),
            temporal_cutoff=integer_field(
                "temporalcutoff", read("temporalcutoff")
            ),
            temporal_cutoff_start=integer_field(
                "temporalcutoffstart", read("temporalcutoffstart")
            ),
            small_cutoff=read("smallcutoff"),
            anisotropy_xyz=resolved_anisotropy,
            end_frame=integer_field("endtime", resolved_end),
            absolute_cutoff=absolute,
        )

    @property
    def anisotropy(self) -> tuple[float, float, float]:
        """Compatibility alias matching the MATLAB parameter name."""

        return self.anisotropy_xyz


def _distance(
    first: LegacyNucleus,
    second: LegacyNucleus,
    anisotropy: tuple[float, float, float],
) -> float:
    return legacy_gram_distance(
        first.position_xyz,
        second.position_xyz,
        anisotropy,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    result = 0.0
    for value in values:
        result += float(value)
    return result / len(values)


def _matlab_divide(numerator: float, denominator: float) -> float:
    """Scalar division with MATLAB/NumPy Inf and NaN semantics."""

    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if denominator == 0:
        if numerator == 0:
            return math.nan
        return math.copysign(math.inf, numerator * math.copysign(1.0, denominator))
    return numerator / denominator


def _immutable_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class LegacyTrackingContext:
    """Deeply immutable graph, row-order, and nearest-neighbour state."""

    nuclei: tuple[LegacyNucleus, ...]
    edges: tuple[TrackEdge, ...]
    parameters: LegacyFeatureParameters
    deleted_ids: frozenset[str] = frozenset()
    stale_predecessor_by_id: Mapping[str, str] = field(default_factory=dict)
    _nuclei_by_id: Mapping[str, LegacyNucleus] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _ordered_ids_by_frame: Mapping[int, tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _successor_slots_by_id: Mapping[str, tuple[str | None, str | None]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _predecessor_by_id: Mapping[str, str | None] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _self_distance_by_id: Mapping[str, float] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _self_nn_by_id: Mapping[str, str] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _mean_self_distance_by_frame: Mapping[int, float] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _forward_cutoff_by_frame: Mapping[int, float] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _f_nn_by_id: Mapping[str, str | None] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _b_nn_by_id: Mapping[str, str | None] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _successor_suitors_by_id: Mapping[str, tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _predecessor_suitors_by_id: Mapping[str, tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _confidence_vector_by_id: Mapping[str, tuple[float, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, LegacyFeatureParameters):
            raise TypeError("parameters must be LegacyFeatureParameters")
        nuclei = tuple(self.nuclei)
        if any(not isinstance(item, LegacyNucleus) for item in nuclei):
            raise TypeError("nuclei must contain only LegacyNucleus values")
        identifiers = [item.nucleus_id for item in nuclei]
        if len(identifiers) != len(set(identifiers)):
            raise LegacyStateError("Legacy nucleus IDs must be unique")
        nuclei_by_id = {item.nucleus_id: item for item in nuclei}
        if any(item.frame > self.parameters.end_frame for item in nuclei):
            raise LegacyStateError("A nucleus lies after parameters.end_frame")

        ordered_ids_by_frame: dict[int, tuple[str, ...]] = {}
        for frame in range(1, self.parameters.end_frame + 1):
            frame_nuclei = sorted(
                (item for item in nuclei if item.frame == frame),
                key=lambda item: (item.matlab_row, item.nucleus_id),
            )
            rows = [item.matlab_row for item in frame_nuclei]
            if len(rows) != len(set(rows)):
                raise LegacyStateError(
                    f"MATLAB rows must be unique within frame {frame}"
                )
            if rows and rows != list(range(len(rows))):
                raise LegacyStateError(
                    f"MATLAB rows in frame {frame} must be contiguous and zero based"
                )
            ordered_ids_by_frame[frame] = tuple(
                item.nucleus_id for item in frame_nuclei
            )

        deleted = frozenset(str(item) for item in self.deleted_ids)
        if "" in deleted or len(deleted) != len(self.deleted_ids):
            raise LegacyStateError("deleted_ids must be unique non-empty strings")
        unknown_deleted = deleted - nuclei_by_id.keys()
        if unknown_deleted:
            raise LegacyStateError(
                "Deleted nucleus IDs are unknown: "
                + ", ".join(sorted(unknown_deleted))
            )

        edges = tuple(self.edges)
        if any(not isinstance(edge, TrackEdge) for edge in edges):
            raise TypeError("edges must contain only TrackEdge values")
        pairs: set[tuple[str, str]] = set()
        outgoing: dict[str, list[tuple[int | None, str]]] = {
            item: [] for item in identifiers
        }
        predecessor: dict[str, str | None] = {item: None for item in identifiers}
        for edge in edges:
            pair = (edge.source_id, edge.target_id)
            if pair in pairs:
                raise LegacyStateError(
                    f"Duplicate legacy edge {pair[0]!r} -> {pair[1]!r}"
                )
            pairs.add(pair)
            if edge.source_id not in nuclei_by_id or edge.target_id not in nuclei_by_id:
                raise LegacyStateError("Legacy edges must reference known nuclei")
            # MATLAB marks rows deleted without consistently clearing their
            # stored suc/pred slots.  Exact feature/candidate extraction must
            # retain that raw topology: a stale successor on an active row,
            # for example, still makes ``suc(:, 1) ~= -1`` during candidate
            # discovery.  ``active_ids`` remains the separate filtered view
            # used when constructing an executable lineage graph.
            source = nuclei_by_id[edge.source_id]
            target = nuclei_by_id[edge.target_id]
            if target.frame <= source.frame:
                raise LegacyStateError("Legacy edges must point strictly forward in time")
            if predecessor[edge.target_id] is not None:
                raise LegacyStateError(
                    f"Legacy lineage merge at {edge.target_id!r} is not supported"
                )
            predecessor[edge.target_id] = edge.source_id
            raw_slot = edge.features.get("LEGACY_SUCCESSOR_SLOT")
            if raw_slot is None:
                slot: int | None = None
            elif isinstance(raw_slot, bool) or not isinstance(raw_slot, Integral):
                raise LegacyStateError(
                    "LEGACY_SUCCESSOR_SLOT must be the zero-based integer 0 or 1"
                )
            else:
                slot = int(raw_slot)
                if slot not in {0, 1}:
                    raise LegacyStateError(
                        "LEGACY_SUCCESSOR_SLOT must be the zero-based integer 0 or 1"
                    )
            outgoing[edge.source_id].append((slot, edge.target_id))
            if len(outgoing[edge.source_id]) > 2:
                raise LegacyStateError(
                    f"Nucleus {edge.source_id!r} has more than two successors"
                )

        successor_slots: dict[str, tuple[str | None, str | None]] = {}
        for nucleus_id, slot_targets in outgoing.items():
            explicit = [slot for slot, _target in slot_targets if slot is not None]
            if explicit and len(explicit) != len(slot_targets):
                raise LegacyStateError(
                    "A source must specify LEGACY_SUCCESSOR_SLOT on all or none "
                    "of its outgoing edges"
                )
            if explicit:
                if len(explicit) != len(set(explicit)):
                    raise LegacyStateError(
                        f"Nucleus {nucleus_id!r} has duplicate legacy successor slots"
                    )
                by_slot = {slot: target for slot, target in slot_targets}
                if 1 in by_slot and 0 not in by_slot:
                    raise LegacyStateError(
                        f"Nucleus {nucleus_id!r} has successor slot 1 without slot 0"
                    )
                targets = [by_slot[item] for item in sorted(by_slot)]
            else:
                # Generic TrackEdge collections have no ordered-slot field.
                # MATLAB initially enumerates ordinary daughter pairs by row,
                # so row order is the deterministic compatibility fallback.
                # Oracle/native adapters should retain LEGACY_SUCCESSOR_SLOT
                # because later rewires can intentionally reverse that order.
                targets = [target for _slot, target in slot_targets]
                targets.sort(
                    key=lambda target_id: (
                        nuclei_by_id[target_id].frame,
                        nuclei_by_id[target_id].matlab_row,
                        target_id,
                    )
                )
            successor_slots[nucleus_id] = (
                targets[0] if targets else None,
                targets[1] if len(targets) > 1 else None,
            )

        if not isinstance(self.stale_predecessor_by_id, Mapping):
            raise TypeError("stale_predecessor_by_id must be a mapping")
        stale_predecessors: dict[str, str] = {}
        for raw_target, raw_source in self.stale_predecessor_by_id.items():
            if type(raw_target) is not str or not raw_target:
                raise TypeError("Stale predecessor targets must be non-empty strings")
            if type(raw_source) is not str or not raw_source:
                raise TypeError("Stale predecessors must be non-empty strings")
            if raw_target in stale_predecessors:
                raise LegacyStateError(
                    f"Duplicate stale predecessor target {raw_target!r}"
                )
            if raw_target not in nuclei_by_id or raw_source not in nuclei_by_id:
                raise LegacyStateError(
                    "Stale predecessor pointers must reference known nuclei"
                )
            if raw_target not in deleted:
                raise LegacyStateError(
                    "Only a deleted row may retain a stale predecessor pointer"
                )
            if predecessor[raw_target] is not None:
                raise LegacyStateError(
                    f"Stale predecessor target {raw_target!r} still has an edge"
                )
            if raw_target in successor_slots[raw_source]:
                raise LegacyStateError(
                    "A stale predecessor cannot have a matching successor slot"
                )
            if nuclei_by_id[raw_source].frame >= nuclei_by_id[raw_target].frame:
                raise LegacyStateError(
                    "Stale predecessor pointers must point backward in time"
                )
            stale_predecessors[raw_target] = raw_source
        raw_predecessor = dict(predecessor)
        raw_predecessor.update(stale_predecessors)

        (
            self_distance,
            self_nn,
            mean_self_distance,
            forward_cutoff,
        ) = self._build_frame_distances(
            nuclei_by_id,
            ordered_ids_by_frame,
        )
        (
            f_nn,
            b_nn,
            successor_suitors,
            predecessor_suitors,
        ) = self._build_directed_nearest_neighbors(
            nuclei_by_id,
            ordered_ids_by_frame,
        )
        confidence_vectors = self._build_confidence_vectors(
            nuclei_by_id,
            ordered_ids_by_frame,
            self_nn,
            mean_self_distance,
        )

        object.__setattr__(self, "nuclei", nuclei)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "deleted_ids", deleted)
        object.__setattr__(
            self,
            "stale_predecessor_by_id",
            _immutable_mapping(stale_predecessors),
        )
        object.__setattr__(self, "_nuclei_by_id", _immutable_mapping(nuclei_by_id))
        object.__setattr__(
            self,
            "_ordered_ids_by_frame",
            _immutable_mapping(ordered_ids_by_frame),
        )
        object.__setattr__(
            self,
            "_successor_slots_by_id",
            _immutable_mapping(successor_slots),
        )
        object.__setattr__(
            self,
            "_predecessor_by_id",
            _immutable_mapping(raw_predecessor),
        )
        object.__setattr__(
            self,
            "_self_distance_by_id",
            _immutable_mapping(self_distance),
        )
        object.__setattr__(self, "_self_nn_by_id", _immutable_mapping(self_nn))
        object.__setattr__(
            self,
            "_mean_self_distance_by_frame",
            _immutable_mapping(mean_self_distance),
        )
        object.__setattr__(
            self,
            "_forward_cutoff_by_frame",
            _immutable_mapping(forward_cutoff),
        )
        object.__setattr__(self, "_f_nn_by_id", _immutable_mapping(f_nn))
        object.__setattr__(self, "_b_nn_by_id", _immutable_mapping(b_nn))
        object.__setattr__(
            self,
            "_successor_suitors_by_id",
            _immutable_mapping(successor_suitors),
        )
        object.__setattr__(
            self,
            "_predecessor_suitors_by_id",
            _immutable_mapping(predecessor_suitors),
        )
        object.__setattr__(
            self,
            "_confidence_vector_by_id",
            _immutable_mapping(confidence_vectors),
        )

    @classmethod
    def from_nuclei_and_edges(
        cls,
        nuclei: Sequence[LegacyNucleus],
        edges: Sequence[TrackEdge],
        parameters: LegacyFeatureParameters,
        *,
        deleted_ids: Iterable[str] = (),
        stale_predecessor_by_id: Mapping[str, str] | None = None,
    ) -> LegacyTrackingContext:
        return cls(
            tuple(nuclei),
            tuple(edges),
            parameters,
            frozenset(deleted_ids),
            {} if stale_predecessor_by_id is None else stale_predecessor_by_id,
        )

    def _build_frame_distances(
        self,
        nuclei: Mapping[str, LegacyNucleus],
        frames: Mapping[int, tuple[str, ...]],
    ) -> tuple[
        dict[str, float],
        dict[str, str],
        dict[int, float],
        dict[int, float],
    ]:
        self_distance: dict[str, float] = {}
        self_nn: dict[str, str] = {}
        mean_distance: dict[int, float] = {}
        forward_cutoff: dict[int, float] = {}
        for frame, ordered in frames.items():
            for nucleus_id in ordered:
                alternatives = tuple(item for item in ordered if item != nucleus_id)
                if not alternatives:
                    self_distance[nucleus_id] = math.inf
                    self_nn[nucleus_id] = nucleus_id
                    continue
                nearest = min(
                    alternatives,
                    key=lambda candidate: (
                        _distance(
                            nuclei[nucleus_id],
                            nuclei[candidate],
                            self.parameters.anisotropy_xyz,
                        ),
                        nuclei[candidate].matlab_row,
                        candidate,
                    ),
                )
                self_nn[nucleus_id] = nearest
                self_distance[nucleus_id] = _distance(
                    nuclei[nucleus_id],
                    nuclei[nearest],
                    self.parameters.anisotropy_xyz,
                )
            frame_distances = [self_distance[item] for item in ordered]
            if not ordered:
                mean_distance[frame] = math.nan
                forward_cutoff[frame] = -1.0
            else:
                # MATLAB calculates selfdistance, normv and forwardcutoff before
                # classifier-driven deletion.  These arrays are detection-time
                # state and must not be recomputed when a row is marked deleted.
                mean_distance[frame] = legacy_single_mean(frame_distances)
                forward_cutoff[frame] = (
                    self.parameters.candidate_cutoff
                    if self.parameters.absolute_cutoff
                    else _matlab_single_arithmetic(
                        mean_distance[frame] * self.parameters.candidate_cutoff
                    )
                )
        return self_distance, self_nn, mean_distance, forward_cutoff

    def _build_directed_nearest_neighbors(
        self,
        nuclei: Mapping[str, LegacyNucleus],
        frames: Mapping[int, tuple[str, ...]],
    ) -> tuple[
        dict[str, str | None],
        dict[str, str | None],
        dict[str, tuple[str, ...]],
        dict[str, tuple[str, ...]],
    ]:
        f_nn: dict[str, str | None] = {item: None for item in nuclei}
        b_nn: dict[str, str | None] = {item: None for item in nuclei}
        successor_suitors_work: dict[str, list[str]] = {
            item: [] for item in nuclei
        }
        predecessor_suitors_work: dict[str, list[str]] = {
            item: [] for item in nuclei
        }
        for frame in range(1, self.parameters.end_frame):
            # fNN/bNN and their inverse suitor lists are also snapshots made
            # before later mutations of the per-row delete flag.  Consumers
            # decide whether a frozen candidate is still eligible.
            sources = frames[frame]
            targets = frames[frame + 1]
            if not sources or not targets:
                continue
            for source_id in sources:
                target_id = min(
                    targets,
                    key=lambda candidate: (
                        _distance(
                            nuclei[source_id],
                            nuclei[candidate],
                            self.parameters.anisotropy_xyz,
                        ),
                        nuclei[candidate].matlab_row,
                        candidate,
                    ),
                )
                f_nn[source_id] = target_id
                predecessor_suitors_work[target_id].append(source_id)
            for target_id in targets:
                source_id = min(
                    sources,
                    key=lambda candidate: (
                        _distance(
                            nuclei[target_id],
                            nuclei[candidate],
                            self.parameters.anisotropy_xyz,
                        ),
                        nuclei[candidate].matlab_row,
                        candidate,
                    ),
                )
                b_nn[target_id] = source_id
                successor_suitors_work[source_id].append(target_id)

        def ordered(values: list[str]) -> tuple[str, ...]:
            return tuple(
                sorted(
                    values,
                    key=lambda item: (
                        nuclei[item].frame,
                        nuclei[item].matlab_row,
                        item,
                    ),
                )
            )

        successor_suitors = {
            item: ordered(values) for item, values in successor_suitors_work.items()
        }
        predecessor_suitors = {
            item: ordered(values)
            for item, values in predecessor_suitors_work.items()
        }
        return f_nn, b_nn, successor_suitors, predecessor_suitors

    def _build_confidence_vectors(
        self,
        nuclei: Mapping[str, LegacyNucleus],
        frames: Mapping[int, tuple[str, ...]],
        self_nn: Mapping[str, str],
        mean_self_distance: Mapping[int, float],
    ) -> dict[str, tuple[float, ...]]:
        confidence: dict[str, tuple[float, ...]] = {}
        for frame, identifiers in frames.items():
            if not identifiers:
                continue
            frame_nuclei = tuple(nuclei[item] for item in identifiers)
            mean_total = _mean([item.total_gfp for item in frame_nuclei])
            mean_aspect = _mean([item.aspect_ratio for item in frame_nuclei])
            mean_avg = _mean([item.avg_gfp for item in frame_nuclei])
            log_density = [
                _matlab_divide(item.log_odds_sum, float(item.slice_count))
                for item in frame_nuclei
            ]
            mean_log_density = _mean(log_density)
            for index, nucleus in enumerate(frame_nuclei):
                nearest = nuclei[self_nn[nucleus.nucleus_id]]
                normalizer = mean_self_distance[frame]
                z_distance = legacy_gram_distance(
                    (nucleus.z,),
                    (nearest.z,),
                )
                z_distance = legacy_single_round(
                    z_distance * self.parameters.anisotropy_xyz[2]
                )
                xy_distance = legacy_gram_distance(
                    (nucleus.x, nucleus.y),
                    (nearest.x, nearest.y),
                )
                z_ratio = legacy_single_round(
                    _matlab_divide(z_distance, normalizer)
                )
                xy_ratio = legacy_single_round(
                    _matlab_divide(xy_distance, normalizer)
                )
                average_value = nucleus.avg_gfp
                # calculateConfidenceVector leaves this column unnormalised
                # when the frame mean is exactly zero.
                if mean_avg != 0:
                    average_value = _matlab_divide(average_value, mean_avg)
                density_value = log_density[index]
                # calculateConfidenceVector retains the raw density column
                # when the frame mean is exactly zero.
                if mean_log_density != 0:
                    density_value = _matlab_divide(
                        density_value,
                        mean_log_density,
                    )
                confidence[nucleus.nucleus_id] = (
                    _matlab_divide(nucleus.total_gfp, mean_total),
                    legacy_single_log(legacy_single_round(z_ratio + 1.0)),
                    legacy_single_log(legacy_single_round(xy_ratio + 1.0)),
                    _matlab_divide(nucleus.aspect_ratio, mean_aspect),
                    density_value,
                    average_value,
                )
        return confidence

    @property
    def nuclei_by_id(self) -> Mapping[str, LegacyNucleus]:
        return self._nuclei_by_id

    @property
    def ordered_ids_by_frame(self) -> Mapping[int, tuple[str, ...]]:
        return self._ordered_ids_by_frame

    @property
    def successor_slots_by_id(
        self,
    ) -> Mapping[str, tuple[str | None, str | None]]:
        return self._successor_slots_by_id

    @property
    def predecessor_by_id(self) -> Mapping[str, str | None]:
        return self._predecessor_by_id

    @property
    def self_distance_by_id(self) -> Mapping[str, float]:
        return self._self_distance_by_id

    @property
    def mean_self_distance_by_frame(self) -> Mapping[int, float]:
        return self._mean_self_distance_by_frame

    @property
    def forward_cutoff_by_frame(self) -> Mapping[int, float]:
        return self._forward_cutoff_by_frame

    @property
    def f_nn_by_id(self) -> Mapping[str, str | None]:
        return self._f_nn_by_id

    @property
    def b_nn_by_id(self) -> Mapping[str, str | None]:
        return self._b_nn_by_id

    @property
    def successor_suitors_by_id(self) -> Mapping[str, tuple[str, ...]]:
        return self._successor_suitors_by_id

    @property
    def predecessor_suitors_by_id(self) -> Mapping[str, tuple[str, ...]]:
        return self._predecessor_suitors_by_id

    @property
    def confidence_vector_by_id(self) -> Mapping[str, tuple[float, ...]]:
        return self._confidence_vector_by_id

    @property
    def active_ids(self) -> frozenset[str]:
        return frozenset(self._nuclei_by_id.keys() - self.deleted_ids)

    def to_lineage_graph_state(self) -> "LineageGraphState":
        """Project raw MATLAB state into the executable active lineage graph.

        MATLAB can retain stale predecessor/successor slots on rows marked for
        deletion.  Those slots stay available on this context for exact
        candidate extraction, while the lineage projection removes every edge
        incident to a deleted row as required by the mutation runtime.
        """

        from .lineage import LineageGraphState

        active = self.active_ids
        return LineageGraphState(
            {item.nucleus_id: item.frame for item in self.nuclei},
            tuple(
                edge
                for edge in self.edges
                if edge.source_id in active and edge.target_id in active
            ),
            self.deleted_ids,
        )

    def nucleus(self, nucleus_id: str) -> LegacyNucleus:
        try:
            return self._nuclei_by_id[nucleus_id]
        except KeyError as exc:
            raise LegacyStateError(f"Unknown legacy nucleus {nucleus_id!r}") from exc

    def frame_ids(
        self,
        frame: int,
        *,
        include_deleted: bool = False,
    ) -> tuple[str, ...]:
        frame_number = _integer(frame, "frame", minimum=1)
        if frame_number > self.parameters.end_frame:
            raise LegacyStateError("frame lies after parameters.end_frame")
        identifiers = self._ordered_ids_by_frame[frame_number]
        if include_deleted:
            return identifiers
        return tuple(item for item in identifiers if item not in self.deleted_ids)

    def successor_slots(self, nucleus_id: str) -> tuple[str | None, str | None]:
        self.nucleus(nucleus_id)
        return self._successor_slots_by_id[nucleus_id]

    def successors(self, nucleus_id: str) -> tuple[str, ...]:
        return tuple(
            item for item in self.successor_slots(nucleus_id) if item is not None
        )

    def predecessor(self, nucleus_id: str) -> str | None:
        self.nucleus(nucleus_id)
        return self._predecessor_by_id[nucleus_id]

    def self_distance(self, nucleus_id: str) -> float:
        self.nucleus(nucleus_id)
        return self._self_distance_by_id[nucleus_id]

    def mean_self_distance(self, frame: int) -> float:
        self.frame_ids(frame, include_deleted=True)
        return self._mean_self_distance_by_frame[int(frame)]

    def forward_cutoff(self, frame: int) -> float:
        self.frame_ids(frame, include_deleted=True)
        return self._forward_cutoff_by_frame[int(frame)]

    def f_nn(self, nucleus_id: str) -> str | None:
        self.nucleus(nucleus_id)
        return self._f_nn_by_id[nucleus_id]

    def b_nn(self, nucleus_id: str) -> str | None:
        self.nucleus(nucleus_id)
        return self._b_nn_by_id[nucleus_id]

    def successor_suitors(self, nucleus_id: str) -> tuple[str, ...]:
        self.nucleus(nucleus_id)
        return self._successor_suitors_by_id[nucleus_id]

    def predecessor_suitors(self, nucleus_id: str) -> tuple[str, ...]:
        self.nucleus(nucleus_id)
        return self._predecessor_suitors_by_id[nucleus_id]

    def confidence_vector(self, nucleus_id: str) -> tuple[float, ...]:
        self.nucleus(nucleus_id)
        return self._confidence_vector_by_id[nucleus_id]

    def require_measurements(
        self,
        nucleus_ids: Iterable[str],
        *names: str,
    ) -> None:
        for nucleus_id in nucleus_ids:
            self.nucleus(nucleus_id).require_measurements(*names)

    def traverse_forward(self, nucleus_id: str) -> tuple[str, ...]:
        """Follow successor slot zero, including the starting nucleus."""

        self.nucleus(nucleus_id)
        branch: list[str] = []
        current: str | None = nucleus_id
        visited: set[str] = set()
        while current is not None:
            if current in visited:
                raise LegacyStateError("Cycle encountered during forward traversal")
            visited.add(current)
            branch.append(current)
            current = self._successor_slots_by_id[current][0]
        return tuple(branch)

    def traverse_backward(
        self,
        nucleus_id: str,
        *,
        stop_at_division: bool = False,
    ) -> tuple[str, ...]:
        """Follow predecessors, optionally stopping before a dividing parent."""

        self.nucleus(nucleus_id)
        branch = [nucleus_id]
        current = nucleus_id
        visited = {nucleus_id}
        while True:
            predecessor = self._predecessor_by_id[current]
            if predecessor is None:
                return tuple(branch)
            if stop_at_division and self._successor_slots_by_id[predecessor][1] is not None:
                return tuple(branch)
            if predecessor in visited:
                raise LegacyStateError("Cycle encountered during backward traversal")
            visited.add(predecessor)
            branch.append(predecessor)
            current = predecessor

    def forward_depth(self, nucleus_id: str) -> int:
        return len(self.traverse_forward(nucleus_id))

    def backward_depth(self, nucleus_id: str, *, stop_at_division: bool = False) -> int:
        return len(
            self.traverse_backward(nucleus_id, stop_at_division=stop_at_division)
        )


__all__ = [
    "LegacyFeatureParameters",
    "LegacyNucleus",
    "LegacyStateError",
    "LegacyTrackingContext",
    "legacy_single_dot",
    "legacy_single_log",
]
