"""Explicit coordinate transforms at the MATLAB/AceTree compatibility boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..api import Calibration


class LegacyCoordinateError(ValueError):
    """Raised for an invalid or ambiguous legacy coordinate conversion."""


@dataclass(frozen=True, slots=True)
class LegacyCoordinateTransform:
    """Reproduce the transform used by the pinned StarryNite launcher/exporter.

    ``finalpoints`` are MATLAB one-based XYZ coordinates in the downsampled,
    ROI-local detector image.  Before tracking the launcher divides X/Y by
    ``downsampling``.  ``saveGreedyNucleiFiles`` then adds ``ROI[x/y]min - 2``
    to obtain AceTree's zero-based X/Y and one-based Z convention.

    The transform deliberately models that observable contract; it does not
    claim that MATLAB ``imresize`` has a pointwise inverse for arbitrary image
    content.
    """

    calibration: Calibration
    downsampling: float = 1.0
    roi_x_min: float = 1.0
    roi_y_min: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, Calibration):
            raise TypeError("calibration must be Calibration")
        for name in ("downsampling", "roi_x_min", "roi_y_min"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise LegacyCoordinateError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.downsampling <= 0:
            raise LegacyCoordinateError("downsampling must be positive")

    @property
    def tracking_anisotropy(self) -> float:
        """Z/XY scale used while XY coordinates remain downsampled."""

        return (
            self.calibration.z_um
            / self.calibration.xy_um
            * self.downsampling
        )

    def matlab_local_to_acetree(
        self,
        xyz: Sequence[float],
    ) -> tuple[float, float, float]:
        """Convert one-based downsampled ROI-local XYZ to AceTree pixels/plane."""

        x, y, z = _point3(xyz, "xyz")
        return (
            x / self.downsampling + self.roi_x_min - 2.0,
            y / self.downsampling + self.roi_y_min - 2.0,
            z,
        )

    def acetree_to_matlab_local(
        self,
        xyz: Sequence[float],
    ) -> tuple[float, float, float]:
        """Invert the unrounded launcher/export coordinate transform."""

        x, y, z = _point3(xyz, "xyz")
        return (
            (x - self.roi_x_min + 2.0) * self.downsampling,
            (y - self.roi_y_min + 2.0) * self.downsampling,
            z,
        )

    def matlab_local_to_physical(
        self,
        xyz: Sequence[float],
    ) -> tuple[float, float, float]:
        """Convert the legacy detector point to AT's physical XYZ contract."""

        return self.calibration.pixel_to_physical(*self.matlab_local_to_acetree(xyz))

    def physical_to_matlab_local(
        self,
        xyz_um: Sequence[float],
    ) -> tuple[float, float, float]:
        """Convert AT physical XYZ back to the unrounded legacy detector domain."""

        point = _point3(xyz_um, "xyz_um")
        return self.acetree_to_matlab_local(
            self.calibration.physical_to_pixel(*point)
        )

    def diameter_to_acetree_pixels(self, diameter: float) -> float:
        """Undo the launcher's XY downsampling compensation for a diameter."""

        value = _finite_scalar(diameter, "diameter")
        if value < 0:
            raise LegacyCoordinateError("diameter must be non-negative")
        return value / self.downsampling

    def diameter_to_radius_um(self, diameter: float) -> float:
        """Convert a downsampled XY diameter to an AT physical radius."""

        return self.diameter_to_acetree_pixels(diameter) * self.calibration.xy_um / 2.0

    def rounded_acetree_export_xyz(
        self,
        xyz: Sequence[float],
    ) -> tuple[int, int, int]:
        """Apply MATLAB's half-away-from-zero ``round`` used by nuclei export."""

        return tuple(
            matlab_round(value) for value in self.matlab_local_to_acetree(xyz)
        )  # type: ignore[return-value]


def matlab_round(value: float) -> int:
    """Round one finite scalar the way MATLAB ``round`` handles half values."""

    number = _finite_scalar(value, "value")
    if number >= 0:
        return int(math.floor(number + 0.5))
    return int(math.ceil(number - 0.5))


def _finite_scalar(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LegacyCoordinateError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise LegacyCoordinateError(f"{name} must be finite")
    return number


def _point3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes, bytearray)) or len(values) != 3:
        raise LegacyCoordinateError(f"{name} must contain three coordinates")
    result = tuple(
        _finite_scalar(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    return result  # type: ignore[return-value]


__all__ = [
    "LegacyCoordinateError",
    "LegacyCoordinateTransform",
    "matlab_round",
]
