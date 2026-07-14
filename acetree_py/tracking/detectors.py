"""Dependency-light anisotropic 3D LoG and DoG spot detectors."""

from __future__ import annotations

import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from .api import Calibration, Detection


_DEFAULT_SETTINGS = MappingProxyType(
    {
        "TARGET_CHANNEL": 1,
        "RADIUS": 4.0,
        "THRESHOLD": 0.0,
        "DO_SUBPIXEL_LOCALIZATION": True,
        "DO_MEDIAN_FILTERING": False,
    }
)
_SUPPORTED_SETTINGS = frozenset(_DEFAULT_SETTINGS)


def _settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(settings) - _SUPPORTED_SETTINGS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported detector setting(s): {names}")
    values = dict(_DEFAULT_SETTINGS)
    values.update(settings)

    target = values["TARGET_CHANNEL"]
    if isinstance(target, bool) or not isinstance(target, (int, np.integer)) or target < 1:
        raise ValueError("TARGET_CHANNEL must be a positive 1-based integer")
    for key in ("RADIUS", "THRESHOLD"):
        try:
            values[key] = float(values[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(values[key]):
            raise ValueError(f"{key} must be finite")
    if values["RADIUS"] <= 0:
        raise ValueError("RADIUS must be positive")
    if values["THRESHOLD"] < 0:
        raise ValueError("THRESHOLD cannot be negative")
    for key in ("DO_SUBPIXEL_LOCALIZATION", "DO_MEDIAN_FILTERING"):
        if not isinstance(values[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be boolean")
        values[key] = bool(values[key])
    values["TARGET_CHANNEL"] = int(target)
    return values


def _select_image(stack_zyx: np.ndarray, target_channel: int) -> np.ndarray:
    image = np.asarray(stack_zyx)
    if image.ndim == 4:
        channel = target_channel - 1
        if not 0 <= channel < image.shape[0]:
            raise IndexError(
                f"TARGET_CHANNEL {target_channel} out of range for {image.shape[0]} channels"
            )
        image = image[channel]
    elif image.ndim != 3:
        raise ValueError(f"Expected a ZYX or CZYX image, got shape {image.shape}")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("Detector input must be a numeric array")
    # A private floating copy prevents preprocessing from mutating provider data.
    return np.array(image, dtype=np.float32, copy=True)


def _ellipsoid_footprint(radius_um: float, calibration: Calibration) -> np.ndarray:
    spacing = np.asarray(calibration.spacing_zyx, dtype=float)
    half_width = np.maximum(1, np.ceil(radius_um / spacing).astype(int))
    z, y, x = np.ogrid[
        -half_width[0] : half_width[0] + 1,
        -half_width[1] : half_width[1] + 1,
        -half_width[2] : half_width[2] + 1,
    ]
    distance_sq = (
        (z * spacing[0]) ** 2
        + (y * spacing[1]) ** 2
        + (x * spacing[2]) ** 2
    )
    return distance_sq <= radius_um**2 + np.finfo(float).eps


def _quadratic_offset(response: np.ndarray, peak: tuple[int, int, int]) -> np.ndarray:
    """Return a bounded separable quadratic refinement in ZYX order."""
    refined = np.asarray(peak, dtype=float)
    for axis, coordinate in enumerate(peak):
        if coordinate == 0 or coordinate == response.shape[axis] - 1:
            continue
        before = list(peak)
        after = list(peak)
        before[axis] -= 1
        after[axis] += 1
        y_before = float(response[tuple(before)])
        y_zero = float(response[peak])
        y_after = float(response[tuple(after)])
        denominator = y_before - 2.0 * y_zero + y_after
        scale = max(abs(y_before), abs(y_zero), abs(y_after), 1.0)
        if denominator >= -np.finfo(float).eps * scale:
            continue
        delta = 0.5 * (y_before - y_after) / denominator
        refined[axis] += float(np.clip(delta, -0.5, 0.5))
    return refined


def _local_peaks(
    response: np.ndarray,
    threshold: float,
    footprint: np.ndarray,
) -> list[tuple[int, int, int]]:
    maxima = ndimage.maximum_filter(response, footprint=footprint, mode="nearest")
    mask = (response > 0) & (response >= threshold) & (response == maxima)
    if not np.any(mask):
        return []

    # A flat maximum can contain several equal voxels. Collapse each connected
    # plateau and use response, then Z/Y/X, as deterministic tie-breakers.
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    peaks: list[tuple[int, int, int]] = []
    for label_id in range(1, count + 1):
        coords = np.argwhere(labels == label_id)
        qualities = response[tuple(coords.T)]
        best_quality = float(np.max(qualities))
        tied = coords[qualities == best_quality]
        chosen = min(tuple(int(value) for value in coord) for coord in tied)
        peaks.append(chosen)
    peaks.sort()
    return peaks


class _BlobDetector:
    plugin_id = ""
    display_name = ""
    default_settings = _DEFAULT_SETTINGS

    def _response(self, image: np.ndarray, sigma_zyx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def detect(
        self,
        stack_zyx: np.ndarray,
        frame: int,
        calibration: Calibration,
        settings: Mapping[str, Any],
        *,
        offset_zyx: tuple[float, float, float] = (0, 0, 0),
    ) -> tuple[Detection, ...]:
        """Detect bright blobs and return physical-coordinate detections.

        ``offset_zyx`` is the zero-based voxel origin of a cropped input stack in
        the full image. It is applied before conversion to microns.
        """
        if frame < 1:
            raise ValueError("frame must be 1-based and positive")
        if len(offset_zyx) != 3 or not all(math.isfinite(float(v)) for v in offset_zyx):
            raise ValueError("offset_zyx must contain three finite values")
        values = _settings(settings)
        image = _select_image(stack_zyx, values["TARGET_CHANNEL"])
        if any(length == 0 for length in image.shape):
            return ()
        if values["DO_MEDIAN_FILTERING"]:
            # TrackMate's option is a 2D 3x3 median, applied independently in Z.
            image = ndimage.median_filter(image, size=(1, 3, 3), mode="nearest")

        radius_um = values["RADIUS"]
        sigma_um = radius_um / math.sqrt(3.0)
        sigma_zyx = sigma_um / np.asarray(calibration.spacing_zyx, dtype=float)
        response = np.asarray(self._response(image, sigma_zyx), dtype=np.float32)
        peaks = _local_peaks(
            response,
            values["THRESHOLD"],
            _ellipsoid_footprint(radius_um, calibration),
        )

        offset = np.asarray(offset_zyx, dtype=float)
        spacing = np.asarray(calibration.spacing_zyx, dtype=float)
        found: list[Detection] = []
        for peak in peaks:
            voxel_zyx = (
                _quadratic_offset(response, peak)
                if values["DO_SUBPIXEL_LOCALIZATION"]
                else np.asarray(peak, dtype=float)
            )
            full_zyx = voxel_zyx + offset
            physical_zyx = full_zyx * spacing
            quality = float(response[peak])
            coordinate_key = ",".join(f"{float(value):.6f}" for value in full_zyx)
            digest = hashlib.blake2b(
                coordinate_key.encode("ascii"), digest_size=8
            ).hexdigest()
            found.append(
                Detection(
                    # Coordinate-derived IDs remain unique when selected-forward
                    # mode detects several cropped ROIs in the same frame.
                    detection_id=f"{self.plugin_id}:t{frame}:{digest}",
                    frame=frame,
                    x_um=float(physical_zyx[2]),
                    y_um=float(physical_zyx[1]),
                    z_um=float(physical_zyx[0]),
                    radius_um=radius_um,
                    quality=quality,
                    features={
                        "DETECTOR_ID": self.plugin_id,
                        "TARGET_CHANNEL": values["TARGET_CHANNEL"],
                        "VOXEL_Z": float(full_zyx[0]),
                        "VOXEL_Y": float(full_zyx[1]),
                        "VOXEL_X": float(full_zyx[2]),
                    },
                )
            )
        return tuple(found)


class LoGDetector(_BlobDetector):
    """Scale-normalized Laplacian-of-Gaussian detector for bright 3D blobs."""

    plugin_id = "acetree.log3d"
    display_name = "3D Laplacian of Gaussian"

    def _response(self, image: np.ndarray, sigma_zyx: np.ndarray) -> np.ndarray:
        response = np.zeros(image.shape, dtype=np.float32)
        for axis in range(3):
            order = [0, 0, 0]
            order[axis] = 2
            derivative = ndimage.gaussian_filter(
                image,
                sigma=tuple(float(value) for value in sigma_zyx),
                order=tuple(order),
                mode="nearest",
            )
            response -= float(sigma_zyx[axis] ** 2) * derivative
        return response


class DoGDetector(_BlobDetector):
    """Difference-of-Gaussians approximation of a 3D LoG detector."""

    plugin_id = "acetree.dog3d"
    display_name = "3D Difference of Gaussians"
    _K = math.sqrt(2.0)

    def _response(self, image: np.ndarray, sigma_zyx: np.ndarray) -> np.ndarray:
        sigma = tuple(float(value) for value in sigma_zyx)
        inner = ndimage.gaussian_filter(image, sigma=sigma, mode="nearest")
        outer = ndimage.gaussian_filter(
            image,
            sigma=tuple(self._K * value for value in sigma),
            mode="nearest",
        )
        return (inner - outer) / (self._K - 1.0)


LogDetector = LoGDetector
DogDetector = DoGDetector
