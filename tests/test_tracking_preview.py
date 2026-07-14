"""Tests for the exact, non-mutating representation shown before acceptance."""

from __future__ import annotations

import pytest

from acetree_py.gui.tracking_preview import expand_tracking_preview
from acetree_py.tracking.api import (
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)


def _gap_result() -> TrackingResult:
    request = TrackingRequest(
        detector=ComponentSpec("acetree.dog3d", {}),
        tracker=ComponentSpec("acetree.simple_lap", {}),
        scope=TrackingScope(
            "selected_forward",
            1,
            3,
            seed_anchors=((1, 1),),
            roi_radius_um=10.0,
        ),
    )
    seed = Detection("seed", 1, 2.0, 4.0, 6.0, 2.0, 5.0)
    target = Detection("target", 3, 6.0, 8.0, 10.0, 4.0, 9.0)
    return TrackingResult(
        request=request,
        detections=(seed, target),
        edges=(TrackEdge("seed", "target", 4.0, kind="gap"),),
        existing_anchors={"seed": (1, 1)},
    )


def test_gap_preview_matches_materialized_interpolation():
    preview = expand_tracking_preview(_gap_result())

    assert [(spot.frame, spot.kind) for spot in preview.spots] == [
        (1, "seed"),
        (2, "interpolated"),
        (3, "detection"),
    ]
    gap = preview.spots[1]
    assert gap.x_um == pytest.approx(4.0)
    assert gap.y_um == pytest.approx(6.0)
    assert gap.z_um == pytest.approx(8.0)
    assert gap.radius_um == pytest.approx(3.0)
    assert gap.quality == pytest.approx(7.0)
    assert preview.proposed_count == 2
    assert preview.interpolated_count == 1
    assert [link.kind for link in preview.links] == ["gap", "gap"]


def test_expanding_preview_does_not_change_result():
    result = _gap_result()
    before = result.to_dict()

    first = expand_tracking_preview(result)
    second = expand_tracking_preview(result)

    assert first == second
    assert result.to_dict() == before
