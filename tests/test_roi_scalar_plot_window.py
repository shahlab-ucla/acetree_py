"""Controller and Qt contract tests for scalar ROI time-series plots."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from acetree_py.analysis.roi_measurements import RoiMeasurementEngine
from acetree_py.core.roi_manager import RoiManager
from acetree_py.core.subcellular_roi import (
    CoordinateSpaceSnapshot,
    ObjectClass,
    Polygon2D,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
)
from acetree_py.gui.roi_scalar_plot_window import (
    RoiScalarPlotController,
    RoiScalarPlotWindow,
)
from acetree_py.gui import roi_scalar_plot_window as roi_scalar_plot_module


class _ConstantProvider:
    num_timepoints = 3
    num_planes = 1
    num_channels = 1
    image_shape = (6, 6)
    manifest_token = "constant-images-v1"

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        assert 1 <= time <= self.num_timepoints
        assert plane == 1
        assert channel == 0
        return np.full(self.image_shape, 2.0)


def _measured_roi_fixture():
    object_class = ObjectClass(
        "Golgi",
        (0.2, 0.8, 1.0, 1.0),
        next_instance_index=2,
    )
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=1,
        expected_start_time=1,
        expected_end_time=3,
        frames={
            1: RoiFrameRecord(
                timepoint=1,
                presence=Presence.SEGMENTED,
                review_state=ReviewState.REVIEWED,
                geometry=Polygon2D(
                    z_plane=1,
                    exterior_xy_px=((1, 1), (3, 1), (3, 3), (1, 3)),
                ),
            ),
            2: RoiFrameRecord(
                timepoint=2,
                presence=Presence.ABSENT,
                review_state=ReviewState.REVIEWED,
            ),
        },
    )
    document = SubcellularRoiDocument(
        coordinate_space=CoordinateSpaceSnapshot(
            xy_res=0.5,
            z_res=1.0,
            image_width_px=6,
            image_height_px=6,
            plane_count=1,
            time_end=3,
        ),
        object_classes=(object_class,),
        objects=(track,),
    )
    manager = RoiManager(document)
    provider = _ConstantProvider()
    engine = RoiMeasurementEngine(provider)
    snapshot = engine.measure(manager)
    return manager, provider, engine, track, snapshot


def test_controller_builds_gap_preserving_series_and_exact_export(tmp_path: Path):
    manager, provider, _engine, track, snapshot = _measured_roi_fixture()
    controller = RoiScalarPlotController(
        manager,
        snapshot,
        image_provider=provider,
        object_ids=(track.object_id,),
        image_channel=0,
        metric_key="intensity.mean",
        time_mode="relative",
    )

    data = controller.build()
    series = data.series[0]

    assert controller.selection_metadata["object_ids"] == (str(track.object_id),)
    assert controller.selection_metadata["source_image_channel"] == 1
    assert controller.selection_metadata["metric_key"] == "intensity.mean"
    assert data.series_kind == "subcellular_objects"
    assert data.x_label == "Time since first segmentation (timepoints)"
    assert series.measurement_key == f"roi:{track.object_id}:ch1:intensity.mean"
    assert series.y_values == (2.0, None, None)
    assert series.missing_reasons == (None, "roi_absent", "not_measured")

    exported = controller.export_csv(tmp_path / "roi-series.csv")
    with exported.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["measurement_key"] == series.measurement_key
    assert rows[0]["source_image_channel"] == "1"
    assert rows[0]["metric_key"] == "intensity.mean"
    assert rows[1]["missing_reason"] == "roi_absent"
    assert rows[2]["missing_reason"] == "not_measured"


def test_controller_blocks_stale_export_until_snapshot_is_replaced(tmp_path: Path):
    manager, provider, engine, track, snapshot = _measured_roi_fixture()
    controller = RoiScalarPlotController(
        manager,
        snapshot,
        image_provider=provider,
        object_ids=(track.object_id,),
    )
    manager.update_frame_geometry(
        track.object_id,
        1,
        Polygon2D(
            z_plane=1,
            exterior_xy_px=((2, 1), (4, 1), (4, 3), (2, 3)),
        ),
    )

    stale_data = controller.build()
    assert stale_data.series[0].missing_reasons[0] == "stale"
    assert controller.stale_samples(stale_data) == 1
    destination = tmp_path / "stale.csv"
    with pytest.raises(RuntimeError, match="stale"):
        controller.export_csv(destination)
    assert not destination.exists()

    controller.update_snapshot(engine.measure(manager))
    assert controller.stale_samples(controller.build()) == 0
    assert controller.export_csv(destination) == destination
    assert destination.exists()


@pytest.mark.skipif(
    not roi_scalar_plot_module._GUI_AVAILABLE,
    reason="Qt/Matplotlib GUI unavailable",
)
def test_window_from_app_tracks_selection_and_published_snapshots(qtbot, tmp_path: Path):
    manager, provider, engine, track, snapshot = _measured_roi_fixture()
    app = SimpleNamespace(
        roi_manager=manager,
        roi_measurement_engine=engine,
        image_provider=provider,
        current_roi_object_id=track.object_id,
    )
    window = RoiScalarPlotWindow.from_app(app, metric_key="intensity.mean")
    qtbot.addWidget(window)

    assert window.selected_object_ids == (str(track.object_id),)
    assert window.image_channel == 0
    assert window.metric_key == "intensity.mean"
    assert window.selection_metadata["object_ids"] == (str(track.object_id),)
    assert window.plot_data is not None
    assert window.plot_data.series[0].y_values == (2.0, None, None)
    assert window._export_button.isEnabled()
    assert window._export_svg_button.isEnabled()
    svg_path = window.export_svg(tmp_path / "roi-plot")
    assert svg_path.suffix == ".svg"
    assert "<svg" in svg_path.read_text(encoding="utf-8")

    manager.update_frame_geometry(
        track.object_id,
        1,
        Polygon2D(
            z_plane=1,
            exterior_xy_px=((2, 1), (4, 1), (4, 3), (2, 3)),
        ),
    )
    window.refresh_plot()
    assert not window._export_button.isEnabled()
    assert not window._export_svg_button.isEnabled()
    assert "stale" in window._status.text()
    stale_svg = tmp_path / "stale.svg"
    with pytest.raises(RuntimeError, match="stale"):
        window.export_svg(stale_svg)
    assert not stale_svg.exists()

    replacement = engine.measure(manager)
    assert replacement is not snapshot
    window.on_measurements_updated()
    assert window._export_button.isEnabled()
    assert window.plot_data.series[0].missing_reasons[0] is None
