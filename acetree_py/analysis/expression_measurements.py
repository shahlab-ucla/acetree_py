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
from typing import Iterable, Mapping, TYPE_CHECKING

from ..core.nucleus import RED_CORRECTIONS
from .expression_plot import ExpressionChannel

if TYPE_CHECKING:
    from ..core.cell import Cell
    from ..core.nuclei_manager import NucleiManager
    from ..core.nucleus import Nucleus


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
        return self.source_revision == int(getattr(manager, "data_revision", 0))

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
        if expression_measurement_calibration(manager) != self.source_calibration:
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

        return self.source_revision == int(getattr(manager, "data_revision", 0))

    def dependencies_current(self, manager: NucleiManager) -> bool:
        """Return whether non-revision measurement inputs still match.

        Calibration affects every measurement.  Blot correction additionally
        masks every nucleus in a frame, so an untracked geometry mutation to
        an *unselected* neighbour must invalidate the whole measured set.
        Other correction modes remain protected by each sampled nucleus's
        geometry signature without paying for a movie-wide hash on every
        sample lookup.
        """

        if expression_measurement_calibration(manager) != self.source_calibration:
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
    digest.update(repr(expression_measurement_calibration(manager)).encode("ascii"))
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


def _correction_label(method: str) -> str:
    return {
        "none": "raw",
        "global": "global corrected",
        "blot": "blot corrected",
        "local": "global fallback; local unavailable",
        "cross": "global fallback; cross-talk unavailable",
    }.get(method, method or "raw")


__all__ = [
    "ExpressionMeasurementFamily",
    "ExpressionMeasurementSet",
    "MeasuredExpressionAggregate",
    "MeasuredExpressionAggregateChannel",
    "MeasuredExpressionChannel",
    "MeasuredExpressionSample",
    "NucleusGeometrySignature",
    "expression_cells_fingerprint",
    "expression_document_fingerprint",
    "expression_measurement_calibration",
    "expression_measurement_dependency_fingerprint",
    "legacy_expression_coverage",
]
