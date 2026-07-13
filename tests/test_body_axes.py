"""Tests for validated anatomical body-axis construction."""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.naming.body_axes import (
    BodyAxisFrame,
    BodyAxisLabel,
    BodyAxisLabels,
    BodyAxisLandmark,
    BodyAxisValidationError,
    solve_body_axes,
)


def _assert_frame(frame: BodyAxisFrame) -> None:
    np.testing.assert_allclose(np.linalg.norm(frame.ap), 1.0)
    np.testing.assert_allclose(np.linalg.norm(frame.dv), 1.0)
    np.testing.assert_allclose(np.linalg.norm(frame.lr), 1.0)
    np.testing.assert_allclose(np.dot(frame.ap, frame.dv), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(frame.ap, frame.lr), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(frame.dv, frame.lr), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(frame.ap, frame.lr), frame.dv, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(frame.basis_matrix), 1.0, atol=1e-12)


def test_solve_from_ap_and_dv_landmarks_applies_z_pixel_resolution() -> None:
    labels = BodyAxisLabels(
        posterior=[0, 0, 0],
        anterior=[3, 0, 1],
        ventral=[0, 0, 0],
        dorsal=[0, 1, 0],
    )

    frame = solve_body_axes(
        labels,
        z_pix_res=4.0,
        provenance="four_cell_manual",
        reference_time=7,
    )

    np.testing.assert_allclose(frame.ap, [0.6, 0.0, 0.8])
    np.testing.assert_allclose(frame.dv, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(frame.lr, [0.8, 0.0, -0.6])
    assert frame.provenance == "four_cell_manual"
    assert frame.reference_time == 7
    assert frame.quality == pytest.approx(1.0)
    _assert_frame(frame)


def test_solve_from_ap_and_lr_landmarks_uses_gram_schmidt() -> None:
    labels = BodyAxisLabels(
        posterior=[0, 0, 0],
        anterior=[2, 0, 0],
        right=[0, 0, 0],
        left=[0.5, 3, 0],  # Contains measurement error along AP.
    )

    frame = BodyAxisFrame.from_landmarks(labels, z_pix_res=2.0)

    np.testing.assert_allclose(frame.ap, [1, 0, 0])
    np.testing.assert_allclose(frame.lr, [0, 1, 0])
    np.testing.assert_allclose(frame.dv, [0, 0, 1])
    assert 0.98 < frame.quality < 1.0
    _assert_frame(frame)


def test_both_secondary_pairs_are_reconciled_when_handedness_agrees() -> None:
    labels = BodyAxisLabels(
        posterior=[0, 0, 0],
        anterior=[1, 0, 0],
        right=[0, 0, 0],
        left=[0, 1, 0.05],
        ventral=[0, 0, 0],
        dorsal=[0, -0.02, 1],
    )

    frame = solve_body_axes(labels, z_pix_res=1.0)

    assert frame.lr[1] > 0.99
    assert frame.dv[2] > 0.99
    assert 0.99 < frame.quality < 1.0
    _assert_frame(frame)


def test_both_secondary_pairs_reject_opposite_handedness() -> None:
    labels = BodyAxisLabels(
        posterior=[0, 0, 0],
        anterior=[1, 0, 0],
        right=[0, 0, 0],
        left=[0, 1, 0],
        ventral=[0, 0, 0],
        dorsal=[0, 0, -1],
    )

    with pytest.raises(BodyAxisValidationError, match="handedness"):
        solve_body_axes(labels, z_pix_res=1.0)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (
            BodyAxisLabels(
                posterior=[0, 0, 0],
                anterior=[0, 0, 0],
                right=[0, 0, 0],
                left=[0, 1, 0],
            ),
            "zero length",
        ),
        (
            BodyAxisLabels(
                posterior=[0, 0, 0],
                anterior=[1, 0, 0],
                right=[0, 0, 0],
                left=[1, 1e-8, 0],
            ),
            "nearly parallel",
        ),
    ],
)
def test_zero_and_nearly_parallel_vectors_are_rejected(
    labels: BodyAxisLabels,
    message: str,
) -> None:
    with pytest.raises(BodyAxisValidationError, match=message):
        solve_body_axes(labels, z_pix_res=1.0)


def test_requires_complete_ap_and_secondary_landmark_pairs() -> None:
    with pytest.raises(BodyAxisValidationError, match="requires both"):
        solve_body_axes(
            BodyAxisLabels(
                posterior=[0, 0, 0],
                anterior=[1, 0, 0],
                right=[0, 0, 0],
            ),
            z_pix_res=1.0,
        )

    with pytest.raises(BodyAxisValidationError, match="complete ventral/dorsal"):
        solve_body_axes(
            BodyAxisLabels(posterior=[0, 0, 0], anterior=[1, 0, 0]),
            z_pix_res=1.0,
        )


def test_unordered_landmarks_are_supported_and_duplicates_rejected() -> None:
    landmarks = [
        BodyAxisLandmark("left", [0, 1, 0]),
        BodyAxisLandmark(BodyAxisLabel.ANTERIOR, [1, 0, 0]),
        BodyAxisLandmark("posterior", [0, 0, 0]),
        BodyAxisLandmark("right", [0, 0, 0]),
    ]

    frame = solve_body_axes(landmarks, z_pix_res=1.0)
    np.testing.assert_allclose(frame.ap, [1, 0, 0])
    np.testing.assert_allclose(frame.lr, [0, 1, 0])
    _assert_frame(frame)

    with pytest.raises(BodyAxisValidationError, match="Duplicate left"):
        BodyAxisLabels.from_landmarks(
            landmarks + [BodyAxisLandmark("left", [0, 2, 0])]
        )


def test_auxinfo_ap_lr_round_trip_orthogonalizes_without_z_rescaling() -> None:
    frame = BodyAxisFrame.from_auxinfo_vectors(
        ap_orientation=[-2, 0, 0],
        lr_orientation=[0.25, 0, 3],
        provenance="auxinfo_v2",
        reference_time=160,
    )

    np.testing.assert_allclose(frame.ap, [-1, 0, 0])
    np.testing.assert_allclose(frame.lr, [0, 0, 1])
    np.testing.assert_allclose(frame.dv, [0, 1, 0])
    assert frame.provenance == "auxinfo_v2"
    assert frame.reference_time == 160
    assert 0.99 < frame.quality < 1.0

    ap, lr = frame.to_auxinfo_vectors()
    restored = BodyAxisFrame.from_auxinfo_vectors(ap, lr)
    np.testing.assert_allclose(restored.ap, frame.ap)
    np.testing.assert_allclose(restored.lr, frame.lr)
    np.testing.assert_allclose(restored.dv, frame.dv)
    _assert_frame(restored)


@pytest.mark.parametrize("z_pix_res", [0.0, -1.0, np.nan, np.inf])
def test_invalid_z_pixel_resolution_is_rejected(z_pix_res: float) -> None:
    labels = BodyAxisLabels(
        posterior=[0, 0, 0],
        anterior=[1, 0, 0],
        right=[0, 0, 0],
        left=[0, 1, 0],
    )
    with pytest.raises(BodyAxisValidationError, match="z_pix_res"):
        solve_body_axes(labels, z_pix_res=z_pix_res)


def test_frame_metadata_and_axes_are_validated() -> None:
    with pytest.raises(BodyAxisValidationError, match="right-handed"):
        BodyAxisFrame(
            ap=np.array([1.0, 0.0, 0.0]),
            dv=np.array([0.0, 0.0, -1.0]),
            lr=np.array([0.0, 1.0, 0.0]),
            provenance="test",
            reference_time=1,
            quality=1.0,
        )

    with pytest.raises(BodyAxisValidationError, match="quality"):
        BodyAxisFrame(
            ap=np.array([1.0, 0.0, 0.0]),
            dv=np.array([0.0, 0.0, 1.0]),
            lr=np.array([0.0, 1.0, 0.0]),
            provenance="test",
            reference_time=1,
            quality=1.1,
        )
