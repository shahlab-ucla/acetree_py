"""Session-level storage for arbitrary-channel AceTree measurements.

Legacy nuclei files have room for one selected AT expression value only.  The
Measure workflow, however, computes every image channel.  This module keeps
those additional values available to analysis windows without changing the
legacy XML/nuclei ZIP contract.

The store is deliberately revision-bound.  A measurement made against one
document revision must never be presented as current after a nucleus is moved,
resized, added, deleted, or relinked.  Per-sample geometry signatures provide a
second fail-closed boundary for callers that mutate a record outside the normal
edit-history path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TYPE_CHECKING

from ..core.nucleus import RED_CORRECTIONS
from .expression_plot import ExpressionChannel


EXPRESSION_MEASUREMENT_CACHE_VERSION = 2

if TYPE_CHECKING:
    from ..core.cell import Cell
    from ..core.nuclei_manager import NucleiManager
    from ..core.nucleus import Nucleus
    from .expression_comparison import ExpressionDataset


@dataclass(frozen=True, slots=True)
class NucleusGeometrySignature:
    """Geometry and liveness state that determine a pixel measurement."""

    x: int
    y: int
    z: float
    size: int
    status: int

    @classmethod
    def from_nucleus(cls, nucleus: Nucleus) -> NucleusGeometrySignature:
        return cls(
            x=int(nucleus.x),
            y=int(nucleus.y),
            z=float(nucleus.z),
            size=int(nucleus.size),
            status=int(nucleus.status),
        )


@dataclass(frozen=True, slots=True)
class MeasuredExpressionSample:
    """Pixel aggregates and selected expression value for one nucleus."""

    value: float
    raw: float
    annulus_background: float | None
    blot_background: float | None
    inner_pixel_count: int
    annulus_pixel_count: int
    blot_pixel_count: int


@dataclass(frozen=True, slots=True)
class MeasuredExpressionAggregate:
    """Correction-neutral pixel aggregates for one nucleus and image channel."""

    raw: float
    annulus_background: float | None
    blot_background: float | None
    inner_pixel_count: int
    annulus_pixel_count: int
    blot_pixel_count: int

    def corrected_value(self, correction_method: str) -> float:
        """Derive a supported correction without copying this aggregate."""

        if correction_method not in RED_CORRECTIONS:
            choices = ", ".join(RED_CORRECTIONS)
            raise ValueError(
                f"Unknown correction_method={correction_method!r}; "
                f"choose one of: {choices}"
            )
        if correction_method == "none":
            return self.raw
        global_background = self.annulus_background or 0.0
        if correction_method == "blot":
            blot_background = (
                self.blot_background
                if self.blot_background is not None
                else global_background
            )
            return self.raw - blot_background
        # Python Measure intentionally aliases legacy local/cross requests to
        # the freshly measured global annulus fallback.
        return self.raw - global_background


@dataclass(frozen=True, slots=True)
class MeasuredExpressionChannel:
    """One image channel measured for all available nuclei."""

    image_channel: int
    label: str
    samples: Mapping[tuple[int, int], MeasuredExpressionSample]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))

    @property
    def key(self) -> str:
        return f"measured_channel_{self.image_channel + 1}"


@dataclass(frozen=True, slots=True)
class MeasuredExpressionAggregateChannel:
    """One channel in a shared correction-neutral measurement family."""

    image_channel: int
    label: str
    samples: Mapping[tuple[int, int], MeasuredExpressionAggregate]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))

    @property
    def key(self) -> str:
        return f"measured_channel_{self.image_channel + 1}"


@dataclass(frozen=True, slots=True)
class ExpressionMeasurementFamily:
    """One immutable aggregate store shared by every correction method."""

    source_revision: int
    source_dependency_fingerprint: str
    source_calibration: tuple[float, float, int, float]
    channels: tuple[MeasuredExpressionAggregateChannel, ...]
    geometries: Mapping[tuple[int, int], NucleusGeometrySignature]
    source_plane_start: int = 1
    measurement_algorithm_version: int = EXPRESSION_MEASUREMENT_CACHE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(
            self,
            "geometries",
            MappingProxyType(dict(self.geometries)),
        )

    @property
    def available_corrections(self) -> tuple[str, ...]:
        return tuple(RED_CORRECTIONS)

    def is_current(self, manager: NucleiManager) -> bool:
        return (
            self.source_revision == int(getattr(manager, "data_revision", 0))
            and self.source_plane_start == expression_measurement_plane_start(manager)
            and self.measurement_algorithm_version == EXPRESSION_MEASUREMENT_CACHE_VERSION
        )

    def dependencies_current(
        self,
        manager: NucleiManager,
        correction_method: str = "blot",
    ) -> bool:
        """Validate only the dependencies required by one correction mode.

        Calibration affects every aggregate. Raw/global/local/cross values only
        depend on the sampled nucleus, whose geometry is checked at lookup.
        Blot additionally masks every neighbouring nucleus and therefore needs
        the movie-wide dependency fingerprint.
        """

        if correction_method not in RED_CORRECTIONS:
            choices = ", ".join(RED_CORRECTIONS)
            raise ValueError(
                f"Unknown correction_method={correction_method!r}; "
                f"choose one of: {choices}"
            )
        if (
            expression_measurement_calibration(manager) != self.source_calibration
            or expression_measurement_plane_start(manager) != self.source_plane_start
        ):
            return False
        if correction_method != "blot":
            return True
        return (
            expression_measurement_dependency_fingerprint(manager)
            == self.source_dependency_fingerprint
        )

    def channel(self, image_channel: int) -> MeasuredExpressionAggregateChannel:
        for channel in self.channels:
            if channel.image_channel == image_channel:
                return channel
        raise KeyError(f"No measured expression channel {image_channel + 1}")

    def sample(
        self,
        manager: NucleiManager,
        image_channel: int,
        time: int,
        nucleus: Nucleus,
        correction_method: str = "blot",
    ) -> MeasuredExpressionAggregate | None:
        """Resolve one aggregate, failing closed on its required dependencies.

        Pass the correction that will be derived. The safe default is
        ``"blot"``, whose neighbour mask requires movie-wide validation;
        non-blot callers avoid that unnecessary whole-movie hash.
        """

        if not self.is_current(manager) or not self.dependencies_current(
            manager,
            correction_method,
        ):
            return None
        return self._sample_from_current_source(time, nucleus, image_channel)

    def _sample_from_current_source(
        self,
        time: int,
        nucleus: Nucleus,
        image_channel: int,
    ) -> MeasuredExpressionAggregate | None:
        """Resolve one sample after revision/dependency validation."""

        key = (int(time), int(nucleus.index))
        sample = self.channel(image_channel).samples.get(key)
        if sample is None:
            return None
        if self.geometries.get(key) != NucleusGeometrySignature.from_nucleus(nucleus):
            return None
        return sample

    def corrected_value(
        self,
        manager: NucleiManager,
        image_channel: int,
        time: int,
        nucleus: Nucleus,
        correction_method: str,
    ) -> float | None:
        if not self.is_current(manager) or not self.dependencies_current(
            manager,
            correction_method,
        ):
            return None
        sample = self._sample_from_current_source(time, nucleus, image_channel)
        if sample is None:
            return None
        return sample.corrected_value(correction_method)


@dataclass(frozen=True, slots=True)
class FrozenCellMeasurements:
    """Stable cell-to-sample index embedded in a full-dataset cache.

    ``cell_id`` is independent of the displayed name so duplicate canonical
    names remain separate, explicit records instead of being overwritten.
    """

    cell_id: str
    cell_name: str
    start_time: int
    end_time: int
    sample_keys: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("frozen cell_id cannot be blank")
        if not self.cell_name.strip():
            raise ValueError("frozen cell_name cannot be blank")
        keys = tuple((int(time), int(index)) for time, index in self.sample_keys)
        if not keys:
            raise ValueError("a frozen cell requires at least one sample key")
        if len(set(keys)) != len(keys):
            raise ValueError("frozen cell sample keys must be unique")
        if any(right[0] <= left[0] for left, right in zip(keys, keys[1:])):
            raise ValueError("frozen cell sample times must be strictly increasing")
        start = int(self.start_time)
        end = int(self.end_time)
        if end < start or keys[0][0] < start or keys[-1][0] > end:
            raise ValueError("frozen cell samples must lie within its lifetime")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "sample_keys", keys)


@dataclass(frozen=True, slots=True)
class FrozenDatasetMeasurementCache:
    """Portable correction-neutral measurements for one complete dataset.

    The cache contains every named observed cell and every measured image
    channel.  Correction choices are derived from raw/global-annulus/blot
    aggregates, so changing a plot request never needs image I/O.
    """

    dataset_id: str
    source_uri: str
    source_fingerprint: str
    snapshot_token: str
    dataset_generation: int
    image_manifest_token: str | None
    measured_at: str
    measurement_algorithm_version: int
    source_revision: int
    source_dependency_fingerprint: str
    source_calibration: tuple[float, float, int, float]
    channels: tuple[MeasuredExpressionAggregateChannel, ...]
    cells: tuple[FrozenCellMeasurements, ...]
    geometries: Mapping[tuple[int, int], NucleusGeometrySignature]
    missing_reasons: Mapping[tuple[int, int, int], str]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("frozen dataset_id cannot be blank")
        if not self.source_fingerprint.strip():
            raise ValueError("frozen source_fingerprint cannot be blank")
        if not self.snapshot_token.strip():
            raise ValueError("frozen snapshot_token cannot be blank")
        if not self.measured_at.strip():
            raise ValueError("frozen measured_at cannot be blank")
        if int(self.measurement_algorithm_version) < 1:
            raise ValueError("measurement_algorithm_version must be positive")
        channels = tuple(self.channels)
        channel_indices = [channel.image_channel for channel in channels]
        if len(set(channel_indices)) != len(channel_indices):
            raise ValueError("frozen measurement channel indices must be unique")
        cells = tuple(self.cells)
        cell_ids = [cell.cell_id for cell in cells]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("frozen cell ids must be unique")
        missing = {
            (int(channel), int(time), int(index)): str(reason)
            for (channel, time, index), reason in self.missing_reasons.items()
        }
        if any(not reason.strip() for reason in missing.values()):
            raise ValueError("frozen missing reasons cannot be blank")
        object.__setattr__(self, "dataset_generation", int(self.dataset_generation))
        object.__setattr__(self, "source_revision", int(self.source_revision))
        object.__setattr__(
            self,
            "measurement_algorithm_version",
            int(self.measurement_algorithm_version),
        )
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "geometries", MappingProxyType(dict(self.geometries)))
        object.__setattr__(self, "missing_reasons", MappingProxyType(missing))

    @property
    def available_corrections(self) -> tuple[str, ...]:
        return tuple(RED_CORRECTIONS)

    @property
    def cell_names(self) -> tuple[str, ...]:
        """Return unique display names while retaining duplicates in ``cells``."""

        return tuple(sorted({cell.cell_name for cell in self.cells}, key=str.casefold))

    def channel(self, image_channel: int) -> MeasuredExpressionAggregateChannel:
        for channel in self.channels:
            if channel.image_channel == image_channel:
                return channel
        raise KeyError(f"No frozen expression channel {image_channel + 1}")

    def is_current(self, manager: NucleiManager) -> bool:
        """Return whether every dependency of this full cache still matches."""

        return (
            self.measurement_algorithm_version == EXPRESSION_MEASUREMENT_CACHE_VERSION
            and self.source_revision == int(getattr(manager, "data_revision", 0))
            and self.source_calibration == expression_measurement_calibration(manager)
            and self.source_dependency_fingerprint
            == expression_measurement_dependency_fingerprint(manager)
        )

    def materialize_dataset(
        self,
        cell_name: str,
        image_channel: int,
        correction_method: str,
        overrides: Mapping[str, object] | None = None,
    ) -> ExpressionDataset:
        """Materialize one comparison dataset without opening source files.

        Missing samples remain explicit gaps.  Missing/duplicate cells and an
        unavailable channel become acquisition-status records.
        """

        from .expression_comparison import (
            DatasetAcquisitionStatus,
            DatasetExpressionTrace,
            DatasetProvenance,
            ExpressionDataset,
            TraceAvailability,
        )

        if not isinstance(cell_name, str) or not cell_name.strip():
            raise ValueError("cell_name cannot be blank")
        if isinstance(image_channel, bool) or not isinstance(image_channel, int):
            raise TypeError("image_channel must be a zero-based integer")
        if image_channel < 0:
            raise ValueError("image_channel cannot be negative")
        if correction_method not in RED_CORRECTIONS:
            choices = ", ".join(RED_CORRECTIONS)
            raise ValueError(
                f"Unknown correction_method={correction_method!r}; "
                f"choose one of: {choices}"
            )
        options: dict[str, Any] = dict(overrides or {})
        dataset_id = str(options.pop("dataset_id", self.dataset_id))
        label = str(options.pop("label", Path(self.source_uri).stem or dataset_id))
        group_id = str(options.pop("group_id", "all"))
        source_uri = str(options.pop("source_uri", self.source_uri))
        series_label = options.pop("series_label", label)
        color = options.pop("color", None)
        if options:
            raise ValueError(
                "Unknown materialization override(s): " + ", ".join(sorted(options))
            )

        requested_channel = f"measured_channel_{image_channel + 1}"
        matches = tuple(cell for cell in self.cells if cell.cell_name == cell_name)
        status: DatasetAcquisitionStatus | None = None
        measured_channel = None
        if not matches:
            status = DatasetAcquisitionStatus(
                cell_name=cell_name,
                channel_key=requested_channel,
                availability=TraceAvailability.MISSING_CELL,
                message=f"Cell {cell_name!r} is absent from this frozen dataset cache.",
            )
        elif len(matches) > 1:
            status = DatasetAcquisitionStatus(
                cell_name=cell_name,
                channel_key=requested_channel,
                availability=TraceAvailability.AMBIGUOUS,
                message=(
                    f"Cell {cell_name!r} has {len(matches)} cached lineages; "
                    "repair duplicate names before exact comparison."
                ),
            )
        else:
            try:
                measured_channel = self.channel(image_channel)
            except KeyError:
                status = DatasetAcquisitionStatus(
                    cell_name=cell_name,
                    channel_key=requested_channel,
                    availability=TraceAvailability.MISSING_CHANNEL,
                    message=(
                        f"Image channel {image_channel + 1} is absent from this "
                        "frozen dataset cache."
                    ),
                )

        metadata = (
            ("trace_source", "recomputed"),
            ("trace_freshness", "frozen_cache"),
            ("session_snapshot_token", self.snapshot_token),
            ("image_manifest_token", self.image_manifest_token or ""),
            ("image_channel", str(image_channel + 1)),
            ("correction_method", correction_method),
            (
                "correction_exact",
                "false" if correction_method in ("local", "cross") else "true",
            ),
            ("measurement_algorithm_version", str(self.measurement_algorithm_version)),
            ("measured_at", self.measured_at),
        )
        provenance = DatasetProvenance(
            dataset_id=dataset_id,
            label=label,
            group_id=group_id,
            source_uri=source_uri,
            source_fingerprint=self.source_fingerprint,
            source_revision=self.dataset_generation,
            metadata=metadata,
        )
        if status is not None:
            return ExpressionDataset(
                provenance=provenance,
                traces=(),
                acquisition_statuses=(status,),
            )

        assert measured_channel is not None and len(matches) == 1
        cell = matches[0]
        values: list[float | None] = []
        reasons: list[str | None] = []
        for time, index in cell.sample_keys:
            aggregate = measured_channel.samples.get((time, index))
            reason = self.missing_reasons.get((image_channel, time, index))
            if aggregate is None:
                values.append(None)
                reasons.append(reason or "measurement unavailable")
                continue
            value, derived_reason = _frozen_corrected_value(
                aggregate,
                correction_method,
            )
            values.append(value)
            reasons.append(derived_reason)
        correction_label = _correction_label(correction_method)
        trace = DatasetExpressionTrace(
            cell_name=cell.cell_name,
            channel_key=measured_channel.key,
            channel_label=f"{measured_channel.label} ({correction_label})",
            channel_unit="scaled mean intensity",
            absolute_times=tuple(float(time) for time, _index in cell.sample_keys),
            values=tuple(values),
            birth_time=float(cell.start_time),
            end_time=float(cell.end_time),
            missing_reasons=tuple(reasons),
            series_label=None if series_label is None else str(series_label),
            color=None if color is None else str(color),
        )
        return ExpressionDataset(provenance=provenance, traces=(trace,))


@dataclass(frozen=True, slots=True)
class ExpressionMeasurementSet:
    """Immutable, revision-bound result of a successful Measure run."""

    source_revision: int
    source_dependency_fingerprint: str
    source_calibration: tuple[float, float, int, float]
    correction_method: str
    at_channel: int
    channels: tuple[MeasuredExpressionChannel, ...]
    geometries: Mapping[tuple[int, int], NucleusGeometrySignature]
    csv_paths: tuple[Path, ...] = ()
    source_plane_start: int = 1
    measurement_algorithm_version: int = EXPRESSION_MEASUREMENT_CACHE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(
            self,
            "geometries",
            MappingProxyType(dict(self.geometries)),
        )
        object.__setattr__(
            self,
            "csv_paths",
            tuple(Path(path) for path in self.csv_paths),
        )

    def is_current(self, manager: NucleiManager) -> bool:
        """Return whether this result belongs to the manager's current edit."""

        return (
            self.source_revision == int(getattr(manager, "data_revision", 0))
            and self.source_plane_start == expression_measurement_plane_start(manager)
            and self.measurement_algorithm_version == EXPRESSION_MEASUREMENT_CACHE_VERSION
        )

    def dependencies_current(self, manager: NucleiManager) -> bool:
        """Return whether non-revision measurement inputs still match.

        Calibration affects every measurement.  Blot correction additionally
        masks every nucleus in a frame, so an untracked geometry mutation to
        an *unselected* neighbour must invalidate the whole measured set.
        Other correction modes remain protected by each sampled nucleus's
        geometry signature without paying for a movie-wide hash on every
        sample lookup.
        """

        if (
            expression_measurement_calibration(manager) != self.source_calibration
            or expression_measurement_plane_start(manager) != self.source_plane_start
        ):
            return False
        if self.correction_method != "blot":
            return True
        return (
            expression_measurement_dependency_fingerprint(manager)
            == self.source_dependency_fingerprint
        )

    def channel(self, image_channel: int) -> MeasuredExpressionChannel:
        for channel in self.channels:
            if channel.image_channel == image_channel:
                return channel
        raise KeyError(f"No measured expression channel {image_channel + 1}")

    def sample(
        self,
        manager: NucleiManager,
        image_channel: int,
        time: int,
        nucleus: Nucleus,
    ) -> MeasuredExpressionSample | None:
        """Resolve a sample, failing closed on stale revision or geometry."""

        if not self.is_current(manager) or not self.dependencies_current(manager):
            return None
        return self._sample_from_current_source(time, nucleus, image_channel)

    def _sample_from_current_source(
        self,
        time: int,
        nucleus: Nucleus,
        image_channel: int,
    ) -> MeasuredExpressionSample | None:
        """Resolve one sample after revision/dependency validation."""

        key = (int(time), int(nucleus.index))
        sample = self.channel(image_channel).samples.get(key)
        if sample is None:
            return None
        if self.geometries.get(key) != NucleusGeometrySignature.from_nucleus(nucleus):
            return None
        return sample

    def expression_channels(self, manager: NucleiManager) -> tuple[ExpressionChannel, ...]:
        """Return plot-service channels backed by this measurement snapshot."""

        output: list[ExpressionChannel] = []
        correction = _correction_label(self.correction_method)
        # Validate movie-wide dependencies once when the channel readers are
        # constructed.  The plot window repeats this check before every build
        # and export; individual reads still check the cheap revision and the
        # sampled nucleus's geometry.
        dependencies_current = self.dependencies_current(manager)
        for measured in self.channels:
            at_suffix = ", AT" if measured.image_channel == self.at_channel else ""
            label = f"{measured.label} (measured, {correction}{at_suffix})"

            def read(_cell, time, nucleus, *, index=measured.image_channel):
                if not dependencies_current or not self.is_current(manager):
                    return None
                sample = self._sample_from_current_source(time, nucleus, index)
                return None if sample is None else sample.value

            output.append(
                ExpressionChannel(
                    key=measured.key,
                    label=label,
                    reader=read,
                    unit="scaled mean intensity",
                )
            )
        return tuple(output)

    def coverage(
        self,
        manager: NucleiManager,
        cells: Iterable[Cell],
        image_channel: int,
        *,
        metric: str = "value",
    ) -> tuple[int, int]:
        """Return ``(valid, expected)`` samples for the selected cell lifetimes."""

        cells = tuple(cells)
        expected = sum(
            1
            for cell in cells
            for time in range(int(cell.start_time), int(cell.end_time) + 1)
            if cell.get_nucleus_at(time) is not None
        )
        if not self.is_current(manager) or not self.dependencies_current(manager):
            return 0, expected

        valid = 0
        for cell in cells:
            for time in range(int(cell.start_time), int(cell.end_time) + 1):
                nucleus = cell.get_nucleus_at(time)
                if nucleus is None:
                    continue
                sample = self._sample_from_current_source(
                    time,
                    nucleus,
                    image_channel,
                )
                if sample is not None and _sample_has_metric(sample, metric):
                    valid += 1
        return valid, expected


def freeze_expression_measurement_family(
    manager: NucleiManager,
    family: ExpressionMeasurementFamily,
    *,
    dataset_id: str,
    source_uri: str,
    source_fingerprint: str,
    snapshot_token: str,
    dataset_generation: int,
    image_manifest_token: str | None,
    measured_at: str,
) -> FrozenDatasetMeasurementCache:
    """Detach a complete family from its manager for offline reuse."""

    tree = manager.lineage_tree
    if tree is None:
        raise ValueError("cannot freeze expression measurements without a lineage tree")
    if not family.is_current(manager):
        raise ValueError("cannot freeze stale expression measurements")

    cells: list[FrozenCellMeasurements] = []
    used_ids: set[str] = set()
    ordered_cells = sorted(
        (cell for cell in tree.all_cells() if cell.name.strip() and cell.nuclei),
        key=lambda cell: (
            int(cell.start_time),
            int(cell.nuclei[0][1].index) if cell.nuclei else -1,
            cell.name.casefold(),
            cell.hash_key or "",
        ),
    )
    for cell in ordered_cells:
        by_time: dict[int, list[int]] = {}
        for time, nucleus in cell.nuclei:
            by_time.setdefault(int(time), []).append(int(nucleus.index))
        for indices in by_time.values():
            indices.sort()
        # Malformed legacy lineages can collapse two same-named nuclei at the
        # same time into one Cell object. Preserve them as explicit duplicate
        # cached cells rather than dropping one or creating duplicate X rows.
        lane_count = max((len(indices) for indices in by_time.values()), default=0)
        for lane in range(lane_count):
            sample_keys = tuple(
                (time, indices[lane])
                for time, indices in sorted(by_time.items())
                if lane < len(indices)
            )
            if not sample_keys:
                continue
            base_id = cell.hash_key or f"{sample_keys[0][0]}:{sample_keys[0][1]}"
            if lane_count > 1:
                base_id = f"{base_id}:duplicate-{lane + 1}"
            cell_id = str(base_id)
            if cell_id in used_ids:
                suffix = 2
                while f"{cell_id}#{suffix}" in used_ids:
                    suffix += 1
                cell_id = f"{cell_id}#{suffix}"
            used_ids.add(cell_id)
            cells.append(
                FrozenCellMeasurements(
                    cell_id=cell_id,
                    cell_name=cell.name,
                    start_time=min(int(cell.start_time), sample_keys[0][0]),
                    end_time=max(int(cell.end_time), sample_keys[-1][0]),
                    sample_keys=sample_keys,
                )
            )

    missing: dict[tuple[int, int, int], str] = {}
    for channel in family.channels:
        for cell in cells:
            for time, index in cell.sample_keys:
                if (time, index) not in channel.samples:
                    missing[(channel.image_channel, time, index)] = (
                        "no measurable inner pixels"
                    )

    return FrozenDatasetMeasurementCache(
        dataset_id=dataset_id,
        source_uri=source_uri,
        source_fingerprint=source_fingerprint,
        snapshot_token=snapshot_token,
        dataset_generation=dataset_generation,
        image_manifest_token=image_manifest_token,
        measured_at=measured_at,
        measurement_algorithm_version=family.measurement_algorithm_version,
        source_revision=family.source_revision,
        source_dependency_fingerprint=family.source_dependency_fingerprint,
        source_calibration=family.source_calibration,
        channels=family.channels,
        cells=tuple(cells),
        geometries=family.geometries,
        missing_reasons=missing,
    )


def legacy_expression_coverage(
    cells: Iterable[Cell],
    channel_key: str = "rweight",
) -> tuple[int, int]:
    """Estimate populated legacy AT samples for an actionable UI warning.

    A numeric zero is a valid expression measurement, so it cannot alone mean
    "missing".  The legacy format offers no explicit validity flag; we regard a
    sample as populated when any stored expression aggregate/correction is
    non-zero.  This intentionally errs toward asking the user to remeasure.
    """

    populated = 0
    expected = 0
    for cell in cells:
        for _time, nucleus in cell.nuclei:
            expected += 1
            raw_valid = nucleus.rwraw != 0 or nucleus.rcount > 0
            if channel_key == "weight":
                has_value = nucleus.weight != 0
            elif channel_key == "rweight":
                has_value = nucleus.rweight != 0 or nucleus.rcount > 0
            elif channel_key == "rwraw":
                has_value = raw_valid
            elif channel_key == "red_global":
                # A zero annulus is valid but indistinguishable from missing in
                # legacy files; rcount is the best available extraction flag.
                has_value = raw_valid and nucleus.rcount > 0
            elif channel_key == "red_blot":
                has_value = raw_valid and nucleus.rwcorr3 != 0
            elif channel_key == "red_local":
                has_value = raw_valid and nucleus.rwcorr2 != 0
            elif channel_key == "red_cross":
                has_value = raw_valid and nucleus.rwcorr4 != 0
            else:
                has_value = False
            if has_value:
                populated += 1
    return populated, expected


def expression_document_fingerprint(manager: NucleiManager) -> str:
    """Hash source, lineage, and expression fields used by plots/Measure.

    Normal GUI edits are guarded by ``data_revision``. This fingerprint is the
    second boundary for direct record mutation and for detecting a dataset
    change while Measure yields to Qt progress events.
    """

    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                float(getattr(manager.movie, "xy_res", 0.0)),
                float(getattr(manager.movie, "z_res", 0.0)),
                int(getattr(manager.movie, "num_planes", 0)),
                float(getattr(manager, "z_pix_res", 0.0)),
                expression_measurement_plane_start(manager),
                EXPRESSION_MEASUREMENT_CACHE_VERSION,
            )
        ).encode("ascii")
    )
    digest.update(b"\n")
    digest.update(f"frames:{len(manager.nuclei_record)}\n".encode("ascii"))
    for t0, nuclei in enumerate(manager.nuclei_record):
        digest.update(f"t:{t0 + 1}:count:{len(nuclei)}\n".encode("ascii"))
        for offset, nucleus in enumerate(nuclei):
            fields = (
                offset,
                nucleus.index,
                nucleus.status,
                nucleus.x,
                nucleus.y,
                float(nucleus.z),
                nucleus.size,
                nucleus.predecessor,
                nucleus.successor1,
                nucleus.successor2,
                nucleus.identity,
                nucleus.assigned_id,
                nucleus.weight,
                nucleus.rweight,
                nucleus.rwraw,
                nucleus.rwcorr1,
                nucleus.rwcorr2,
                nucleus.rwcorr3,
                nucleus.rwcorr4,
            )
            digest.update(repr(fields).encode("utf-8", errors="surrogatepass"))
            digest.update(b"\n")
    return digest.hexdigest()


def expression_measurement_plane_start(manager: NucleiManager) -> int:
    """Return the absolute Z coordinate of the first image plane."""

    return int(getattr(getattr(manager, "config", None), "plane_start", 1))


def expression_measurement_calibration(
    manager: NucleiManager,
) -> tuple[float, float, int, float]:
    """Return calibration values that affect pixel sampling geometry."""

    return (
        float(getattr(manager.movie, "xy_res", 0.0)),
        float(getattr(manager.movie, "z_res", 0.0)),
        int(getattr(manager.movie, "num_planes", 0)),
        float(getattr(manager, "z_pix_res", 0.0)),
    )


def expression_measurement_dependency_fingerprint(manager: NucleiManager) -> str:
    """Hash all geometry that can influence a movie-wide Measure result.

    Blot correction masks every live nucleus in a frame, not just the cells
    selected for a plot.  This compact fingerprint therefore includes the
    ordered geometry/liveness of every record plus image calibration, while
    deliberately excluding expression fields that Measure itself updates.
    """

    digest = hashlib.sha256()
    digest.update(
        repr((
            expression_measurement_calibration(manager),
            expression_measurement_plane_start(manager),
            EXPRESSION_MEASUREMENT_CACHE_VERSION,
        )).encode("ascii")
    )
    digest.update(b"\n")
    digest.update(f"frames:{len(manager.nuclei_record)}\n".encode("ascii"))
    for t0, nuclei in enumerate(manager.nuclei_record):
        digest.update(f"t:{t0 + 1}:count:{len(nuclei)}\n".encode("ascii"))
        for offset, nucleus in enumerate(nuclei):
            fields = (
                offset,
                nucleus.index,
                nucleus.status,
                nucleus.x,
                nucleus.y,
                float(nucleus.z),
                nucleus.size,
            )
            digest.update(repr(fields).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def expression_cells_fingerprint(cells: Iterable[Cell]) -> str:
    """Hash the selected cell samples that feed one plot snapshot."""

    digest = hashlib.sha256()
    ordered = sorted(
        cells,
        key=lambda cell: (cell.hash_key or "", cell.name, cell.start_time, cell.end_time),
    )
    for cell in ordered:
        digest.update(
            repr((cell.hash_key, cell.name, cell.start_time, cell.end_time)).encode(
                "utf-8", errors="surrogatepass"
            )
        )
        digest.update(b"\n")
        for time, nucleus in cell.nuclei:
            fields = (
                time,
                nucleus.index,
                nucleus.status,
                nucleus.x,
                nucleus.y,
                float(nucleus.z),
                nucleus.size,
                nucleus.predecessor,
                nucleus.successor1,
                nucleus.successor2,
                nucleus.identity,
                nucleus.assigned_id,
                nucleus.weight,
                nucleus.rweight,
                nucleus.rwraw,
                nucleus.rwcorr1,
                nucleus.rwcorr2,
                nucleus.rwcorr3,
                nucleus.rwcorr4,
            )
            digest.update(repr(fields).encode("utf-8", errors="surrogatepass"))
            digest.update(b"\n")
    return digest.hexdigest()


def _sample_has_metric(sample: MeasuredExpressionSample, metric: str) -> bool:
    if metric in ("value", "raw"):
        return True
    if metric == "global":
        return sample.annulus_background is not None
    if metric == "blot":
        return sample.blot_background is not None
    return False


def _frozen_corrected_value(
    sample: MeasuredExpressionAggregate,
    correction_method: str,
) -> tuple[float | None, str | None]:
    # Delegate to the live family derivation so a frozen cache has exact
    # parity for every RED_CORRECTIONS choice, including its documented
    # absent-background and local/cross fallback semantics.
    return float(sample.corrected_value(correction_method)), None


def _correction_label(method: str) -> str:
    return {
        "none": "raw",
        "global": "global corrected",
        "blot": "blot corrected",
        "local": "global fallback; local unavailable",
        "cross": "global fallback; cross-talk unavailable",
    }.get(method, method or "raw")


__all__ = [
    "EXPRESSION_MEASUREMENT_CACHE_VERSION",
    "ExpressionMeasurementFamily",
    "ExpressionMeasurementSet",
    "FrozenCellMeasurements",
    "FrozenDatasetMeasurementCache",
    "MeasuredExpressionAggregate",
    "MeasuredExpressionAggregateChannel",
    "MeasuredExpressionChannel",
    "MeasuredExpressionSample",
    "NucleusGeometrySignature",
    "expression_cells_fingerprint",
    "expression_document_fingerprint",
    "expression_measurement_calibration",
    "expression_measurement_plane_start",
    "expression_measurement_dependency_fingerprint",
    "freeze_expression_measurement_family",
    "legacy_expression_coverage",
]
