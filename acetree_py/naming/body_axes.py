"""Validated anatomical body frames from explicitly labelled landmarks.

The vectors in this module use AceTree's anatomical direction convention:

* ``AP`` points from posterior to anterior.
* ``LR`` points from right to left.
* ``DV`` points from ventral to dorsal.

The ordered basis ``(AP, LR, DV)`` is right-handed, so ``DV = AP x LR``.
Landmark positions are supplied in raw image coordinates and converted to
physical coordinates by multiplying their z component by ``z_pix_res``.
AuxInfo v2 vectors, by contrast, are already direction vectors and therefore
are not rescaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_PARALLEL_TOLERANCE = 1e-3
_ZERO_NORM = 1e-12
_FRAME_TOLERANCE = 1e-7


class BodyAxisValidationError(ValueError):
    """Raised when labelled landmarks cannot define a reliable body frame."""


class BodyAxisLabel(str, Enum):
    """An anatomical endpoint that can be attached to a 3D landmark."""

    POSTERIOR = "posterior"
    ANTERIOR = "anterior"
    VENTRAL = "ventral"
    DORSAL = "dorsal"
    RIGHT = "right"
    LEFT = "left"


def _as_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise BodyAxisValidationError(f"{name} must be a finite 3D position or vector")
    if not np.all(np.isfinite(vector)):
        raise BodyAxisValidationError(f"{name} must be a finite 3D position or vector")
    return vector.copy()


def _unit_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = _as_vector(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= _ZERO_NORM:
        raise BodyAxisValidationError(f"{name} has zero length")
    return vector / norm


def _validate_parallel_tolerance(value: float) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise BodyAxisValidationError("parallel_tolerance must be between 0 and 1")
    return tolerance


def _orthogonal_unit(
    primary: NDArray[np.float64],
    secondary: ArrayLike,
    name: str,
    parallel_tolerance: float,
) -> tuple[NDArray[np.float64], float]:
    """Gram-Schmidt a secondary vector and return its angular quality.

    The quality is the sine of the angle between the original primary and
    secondary vectors. It is 1 for perpendicular observations and approaches
    0 as the observations become parallel.
    """

    raw = _as_vector(secondary, name)
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm <= _ZERO_NORM:
        raise BodyAxisValidationError(f"{name} has zero length")

    perpendicular = raw - float(np.dot(raw, primary)) * primary
    perpendicular_norm = float(np.linalg.norm(perpendicular))
    angular_quality = perpendicular_norm / raw_norm
    if angular_quality < parallel_tolerance:
        raise BodyAxisValidationError(
            f"{name} is parallel or nearly parallel to the AP direction"
        )
    return perpendicular / perpendicular_norm, float(np.clip(angular_quality, 0.0, 1.0))


@dataclass(frozen=True)
class BodyAxisLandmark:
    """One labelled anatomical point in raw ``(x, y, z)`` image coordinates."""

    label: BodyAxisLabel | str
    position: ArrayLike

    def __post_init__(self) -> None:
        label = self.label
        if not isinstance(label, BodyAxisLabel):
            try:
                label = BodyAxisLabel(str(label).strip().lower())
            except ValueError as exc:
                raise BodyAxisValidationError(f"Unknown body-axis label: {self.label!r}") from exc

        position = _as_vector(self.position, f"{label.value} landmark")
        position.setflags(write=False)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class BodyAxisLabels:
    """Endpoint positions used to solve an anatomical body frame.

    Posterior and anterior are required. At least one complete secondary pair
    (ventral/dorsal or right/left) is also required. Supplying both pairs gives
    an independent handedness check.
    """

    posterior: ArrayLike
    anterior: ArrayLike
    ventral: ArrayLike | None = None
    dorsal: ArrayLike | None = None
    right: ArrayLike | None = None
    left: ArrayLike | None = None

    @classmethod
    def from_landmarks(cls, landmarks: Iterable[BodyAxisLandmark]) -> BodyAxisLabels:
        """Build endpoint labels from an unordered landmark collection."""

        positions: dict[BodyAxisLabel, ArrayLike] = {}
        for item in landmarks:
            landmark = item if isinstance(item, BodyAxisLandmark) else BodyAxisLandmark(*item)
            if landmark.label in positions:
                raise BodyAxisValidationError(
                    f"Duplicate {landmark.label.value} body-axis landmark"
                )
            positions[landmark.label] = landmark.position

        missing = [
            label.value
            for label in (BodyAxisLabel.POSTERIOR, BodyAxisLabel.ANTERIOR)
            if label not in positions
        ]
        if missing:
            raise BodyAxisValidationError(
                "Missing required body-axis landmark(s): " + ", ".join(missing)
            )

        return cls(
            posterior=positions[BodyAxisLabel.POSTERIOR],
            anterior=positions[BodyAxisLabel.ANTERIOR],
            ventral=positions.get(BodyAxisLabel.VENTRAL),
            dorsal=positions.get(BodyAxisLabel.DORSAL),
            right=positions.get(BodyAxisLabel.RIGHT),
            left=positions.get(BodyAxisLabel.LEFT),
        )


@dataclass(frozen=True)
class BodyAxisFrame:
    """An orthonormal anatomical frame in physical laboratory coordinates."""

    ap: NDArray[np.float64]
    dv: NDArray[np.float64]
    lr: NDArray[np.float64]
    provenance: str
    reference_time: int | None
    quality: float

    def __post_init__(self) -> None:
        ap = _as_vector(self.ap, "AP axis")
        dv = _as_vector(self.dv, "DV axis")
        lr = _as_vector(self.lr, "LR axis")

        for name, vector in (("AP", ap), ("DV", dv), ("LR", lr)):
            if not np.isclose(np.linalg.norm(vector), 1.0, atol=_FRAME_TOLERANCE):
                raise BodyAxisValidationError(f"{name} axis must be a unit vector")

        if any(
            abs(float(np.dot(first, second))) > _FRAME_TOLERANCE
            for first, second in ((ap, dv), (ap, lr), (dv, lr))
        ):
            raise BodyAxisValidationError("AP, DV, and LR axes must be orthogonal")
        if not np.allclose(np.cross(ap, lr), dv, atol=_FRAME_TOLERANCE):
            raise BodyAxisValidationError(
                "Body frame must be right-handed according to DV = AP x LR"
            )

        provenance = str(self.provenance).strip()
        if not provenance:
            raise BodyAxisValidationError("provenance must not be empty")

        reference_time = self.reference_time
        if reference_time is not None:
            if isinstance(reference_time, bool) or not isinstance(reference_time, Integral):
                raise BodyAxisValidationError("reference_time must be an integer or None")
            reference_time = int(reference_time)

        quality = float(self.quality)
        if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise BodyAxisValidationError("quality must be between 0 and 1")

        for vector in (ap, dv, lr):
            vector.setflags(write=False)
        object.__setattr__(self, "ap", ap)
        object.__setattr__(self, "dv", dv)
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "reference_time", reference_time)
        object.__setattr__(self, "quality", quality)

    @property
    def basis_matrix(self) -> NDArray[np.float64]:
        """Return the right-handed ``(AP, LR, DV)`` basis as matrix columns."""

        return np.column_stack((self.ap, self.lr, self.dv))

    @classmethod
    def from_landmarks(
        cls,
        labels: BodyAxisLabels | Iterable[BodyAxisLandmark],
        *,
        z_pix_res: float,
        provenance: str = "manual_landmarks",
        reference_time: int | None = None,
        parallel_tolerance: float = DEFAULT_PARALLEL_TOLERANCE,
    ) -> BodyAxisFrame:
        """Solve a frame from labelled raw-pixel positions."""

        return solve_body_axes(
            labels,
            z_pix_res=z_pix_res,
            provenance=provenance,
            reference_time=reference_time,
            parallel_tolerance=parallel_tolerance,
        )

    @classmethod
    def from_auxinfo_vectors(
        cls,
        ap_orientation: ArrayLike,
        lr_orientation: ArrayLike,
        *,
        provenance: str = "auxinfo_v2",
        reference_time: int | None = None,
        parallel_tolerance: float = DEFAULT_PARALLEL_TOLERANCE,
    ) -> BodyAxisFrame:
        """Build a frame from AuxInfo-style AP and LR direction vectors."""

        tolerance = _validate_parallel_tolerance(parallel_tolerance)
        ap = _unit_vector(ap_orientation, "AuxInfo AP orientation")
        lr, angular_quality = _orthogonal_unit(
            ap,
            lr_orientation,
            "AuxInfo LR orientation",
            tolerance,
        )
        dv = np.cross(ap, lr)
        dv /= np.linalg.norm(dv)
        return cls(
            ap=ap,
            dv=dv,
            lr=lr,
            provenance=provenance,
            reference_time=reference_time,
            quality=angular_quality,
        )

    def to_auxinfo_vectors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return AP/LR arrays compatible with AuxInfo v2 orientation fields."""

        return self.ap.copy(), self.lr.copy()


def _physical_position(position: ArrayLike, name: str, z_pix_res: float) -> NDArray[np.float64]:
    physical = _as_vector(position, name)
    physical[2] *= z_pix_res
    if not np.all(np.isfinite(physical)):
        raise BodyAxisValidationError(f"{name} is not finite after z scaling")
    return physical


def _pair_is_complete(
    first: ArrayLike | None,
    second: ArrayLike | None,
    pair_name: str,
) -> bool:
    if (first is None) != (second is None):
        raise BodyAxisValidationError(f"{pair_name} requires both endpoint landmarks")
    return first is not None


def solve_body_axes(
    labels: BodyAxisLabels | Iterable[BodyAxisLandmark],
    *,
    z_pix_res: float,
    provenance: str = "manual_landmarks",
    reference_time: int | None = None,
    parallel_tolerance: float = DEFAULT_PARALLEL_TOLERANCE,
) -> BodyAxisFrame:
    """Solve an anatomical frame from labelled raw-pixel positions.

    Gram-Schmidt projection removes AP components from the observed secondary
    directions. If both DV and LR pairs are present, their projected directions
    are averaged after confirming that they encode the same handedness.
    """

    if not isinstance(labels, BodyAxisLabels):
        labels = BodyAxisLabels.from_landmarks(labels)

    z_resolution = float(z_pix_res)
    if not np.isfinite(z_resolution) or z_resolution <= 0.0:
        raise BodyAxisValidationError("z_pix_res must be a finite positive number")
    tolerance = _validate_parallel_tolerance(parallel_tolerance)

    has_dv = _pair_is_complete(labels.ventral, labels.dorsal, "DV axis")
    has_lr = _pair_is_complete(labels.right, labels.left, "LR axis")
    if not has_dv and not has_lr:
        raise BodyAxisValidationError(
            "A complete ventral/dorsal or right/left landmark pair is required"
        )

    posterior = _physical_position(labels.posterior, "posterior landmark", z_resolution)
    anterior = _physical_position(labels.anterior, "anterior landmark", z_resolution)
    ap = _unit_vector(anterior - posterior, "posterior-to-anterior direction")

    dv_observed: NDArray[np.float64] | None = None
    lr_observed: NDArray[np.float64] | None = None
    quality_terms: list[float] = []

    if has_dv:
        ventral = _physical_position(labels.ventral, "ventral landmark", z_resolution)
        dorsal = _physical_position(labels.dorsal, "dorsal landmark", z_resolution)
        dv_observed, dv_quality = _orthogonal_unit(
            ap,
            dorsal - ventral,
            "ventral-to-dorsal direction",
            tolerance,
        )
        quality_terms.append(dv_quality)

    if has_lr:
        right = _physical_position(labels.right, "right landmark", z_resolution)
        left = _physical_position(labels.left, "left landmark", z_resolution)
        lr_observed, lr_quality = _orthogonal_unit(
            ap,
            left - right,
            "right-to-left direction",
            tolerance,
        )
        quality_terms.append(lr_quality)

    if dv_observed is not None and lr_observed is not None:
        handedness = float(np.dot(np.cross(ap, lr_observed), dv_observed))
        if handedness < tolerance:
            raise BodyAxisValidationError(
                "DV and LR landmarks have inconsistent or indeterminate handedness"
            )
        quality_terms.append(handedness)

        # Each observed secondary direction independently predicts LR. Average
        # those predictions to avoid arbitrarily privileging one landmark pair.
        lr = _unit_vector(
            lr_observed + np.cross(dv_observed, ap),
            "combined right-to-left direction",
        )
    elif lr_observed is not None:
        lr = lr_observed
    else:
        assert dv_observed is not None
        lr = np.cross(dv_observed, ap)
        lr /= np.linalg.norm(lr)

    # Recompute the third axis from the final pair so the returned frame is
    # exactly orthonormal and right-handed after measurement reconciliation.
    lr = lr - float(np.dot(lr, ap)) * ap
    lr /= np.linalg.norm(lr)
    dv = np.cross(ap, lr)
    dv /= np.linalg.norm(dv)

    return BodyAxisFrame(
        ap=ap,
        dv=dv,
        lr=lr,
        provenance=provenance,
        reference_time=reference_time,
        quality=float(np.clip(min(quality_terms), 0.0, 1.0)),
    )


__all__ = [
    "BodyAxisFrame",
    "BodyAxisLabel",
    "BodyAxisLabels",
    "BodyAxisLandmark",
    "BodyAxisValidationError",
    "DEFAULT_PARALLEL_TOLERANCE",
    "solve_body_axes",
]
