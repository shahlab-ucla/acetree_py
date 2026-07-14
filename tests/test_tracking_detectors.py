"""Focused tests for dependency-light 3D blob detectors."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.tracking import Calibration, DoGDetector, LoGDetector


def _gaussian_blob(
    shape: tuple[int, int, int],
    center_zyx: tuple[float, float, float],
    calibration: Calibration,
    sigma_um: float,
    amplitude: float = 100.0,
) -> np.ndarray:
    z, y, x = np.indices(shape, dtype=float)
    spacing = calibration.spacing_zyx
    squared = (
        ((z - center_zyx[0]) * spacing[0] / sigma_um) ** 2
        + ((y - center_zyx[1]) * spacing[1] / sigma_um) ** 2
        + ((x - center_zyx[2]) * spacing[2] / sigma_um) ** 2
    )
    return (amplitude * np.exp(-0.5 * squared)).astype(np.float32)


@pytest.mark.parametrize("detector_type", [LoGDetector, DoGDetector])
def test_detects_anisotropic_subpixel_gaussian(detector_type):
    calibration = Calibration(xy_um=0.5, z_um=1.5)
    radius_um = 3.0
    center = (5.2, 20.35, 18.7)
    image = _gaussian_blob(
        (11, 41, 41), center, calibration, radius_um / np.sqrt(3.0)
    )
    original = image.copy()

    detections = detector_type().detect(
        image,
        frame=7,
        calibration=calibration,
        settings={
            "RADIUS": radius_um,
            "THRESHOLD": 0.01,
            "DO_SUBPIXEL_LOCALIZATION": True,
        },
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.frame == 7
    assert detection.radius_um == radius_um
    assert detection.x_um == pytest.approx(center[2] * calibration.xy_um, abs=0.35)
    assert detection.y_um == pytest.approx(center[1] * calibration.xy_um, abs=0.35)
    assert detection.z_um == pytest.approx(center[0] * calibration.z_um, abs=0.8)
    assert detection.quality > 0
    assert np.array_equal(image, original)


def test_crop_offset_is_applied_before_physical_conversion():
    calibration = Calibration(xy_um=0.25, z_um=2.0)
    center = (3.0, 6.0, 7.0)
    image = _gaussian_blob((7, 15, 15), center, calibration, sigma_um=1.5)

    detection = LoGDetector().detect(
        image,
        frame=2,
        calibration=calibration,
        settings={"RADIUS": 2.6, "THRESHOLD": 0.01},
        offset_zyx=(4, 10, 20),
    )[0]

    assert detection.z_um == pytest.approx((center[0] + 4) * 2.0, abs=0.7)
    assert detection.y_um == pytest.approx((center[1] + 10) * 0.25, abs=0.2)
    assert detection.x_um == pytest.approx((center[2] + 20) * 0.25, abs=0.2)


def test_target_channel_is_one_based_for_czyx_input():
    calibration = Calibration(xy_um=1.0, z_um=1.0)
    first = np.zeros((9, 21, 21), dtype=np.float32)
    second = _gaussian_blob((9, 21, 21), (4, 10, 10), calibration, 1.5)
    image = np.stack([first, second])

    detections = DoGDetector().detect(
        image,
        frame=1,
        calibration=calibration,
        settings={"TARGET_CHANNEL": 2, "RADIUS": 2.6, "THRESHOLD": 0.01},
    )

    assert len(detections) == 1
    assert detections[0].features["TARGET_CHANNEL"] == 2
    with pytest.raises(IndexError, match="TARGET_CHANNEL"):
        DoGDetector().detect(
            image,
            1,
            calibration,
            {"TARGET_CHANNEL": 3, "RADIUS": 2.6},
        )


def test_threshold_and_empty_stack_produce_no_detections():
    calibration = Calibration(xy_um=1.0, z_um=1.0)
    image = _gaussian_blob((7, 15, 15), (3, 7, 7), calibration, 1.5)
    assert not LoGDetector().detect(
        image, 1, calibration, {"RADIUS": 2.6, "THRESHOLD": 1.0e9}
    )
    assert not DoGDetector().detect(
        np.zeros_like(image), 1, calibration, {"RADIUS": 2.6}
    )


def test_detection_ids_and_plateau_tie_break_are_deterministic():
    calibration = Calibration(xy_um=1.0, z_um=1.0)
    image = _gaussian_blob((9, 25, 25), (4, 12, 12), calibration, 1.5)
    detector = LoGDetector()
    first = detector.detect(image, 3, calibration, {"RADIUS": 2.6})
    second = detector.detect(image, 3, calibration, {"RADIUS": 2.6})
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]


def test_calibration_and_detection_pixel_round_trip():
    calibration = Calibration(xy_um=0.2, z_um=1.5, plane_start=1)
    from acetree_py.tracking import Detection

    detection = Detection.from_pixel(
        "seed", 4, 12.5, 8.0, 3.5, 2.0, 1.0, calibration
    )
    assert detection.to_pixel(calibration) == pytest.approx((12.5, 8.0, 3.5))
    with pytest.raises(TypeError):
        detection.features["new"] = 1


def test_detector_settings_fail_early():
    image = np.zeros((3, 3, 3), dtype=np.float32)
    calibration = Calibration(1.0, 1.0)
    with pytest.raises(ValueError, match="Unsupported detector"):
        LoGDetector().detect(image, 1, calibration, {"UNKNOWN": 1})
    with pytest.raises(ValueError, match="RADIUS"):
        LoGDetector().detect(image, 1, calibration, {"RADIUS": 0})
    with pytest.raises(ValueError, match="boolean"):
        LoGDetector().detect(
            image, 1, calibration, {"DO_SUBPIXEL_LOCALIZATION": 1}
        )
