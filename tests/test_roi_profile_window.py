from dataclasses import replace

import numpy as np
import pytest

from acetree_py.analysis import roi_measurements
from acetree_py.analysis.roi_measurements import RoiMeasurementEngine
from acetree_py.core.roi_manager import RoiManager
from acetree_py.core.subcellular_roi import (
    CellRef,
    CoordinateSpaceSnapshot,
    NucleusAnchor,
    ObjectClass,
    Presence,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
    Thickness,
    ThickPolyline2D,
)
from acetree_py.editing.roi_commands import AssociateRoiFrame, SetRoiFrameGeometry
from acetree_py.gui.roi_profile_window import RoiProfileWindow
from acetree_py.gui.roi_scalar_plot_window import RoiScalarPlotWindow
from tests.test_gui_app import _make_app


class _ProfileProvider:
    num_timepoints = 3
    num_planes = 1
    num_channels = 2
    image_shape = (6, 6)
    manifest_token = "profile-images-v1"

    def __init__(self):
        self.pixel_reads = 0

    def get_plane(self, time, plane, channel=0):
        self.pixel_reads += 1
        return np.full(self.image_shape, 2.0 + time + channel)


def _profile_app():
    object_class = ObjectClass("Membrane", (0.2, 0.8, 1.0, 1.0), next_instance_index=2)
    geometry = ThickPolyline2D(1, ((1, 2), (4, 2)), Thickness(0.5, "um"))
    track = RoiObjectTrack(
        class_id=object_class.class_id,
        instance_index=1,
        frames={
            time: RoiFrameRecord(time, Presence.SEGMENTED, ReviewState.REVIEWED, geometry)
            for time in (1, 2)
        },
    )
    manager = RoiManager(SubcellularRoiDocument(
        coordinate_space=CoordinateSpaceSnapshot(
            xy_res=0.5, z_res=1.0, image_width_px=6, image_height_px=6,
            plane_count=1, time_end=3,
        ),
        object_classes=(object_class,), objects=(track,),
    ))
    app = _make_app()
    app.roi_manager = manager
    app.image_provider = _ProfileProvider()
    app.roi_measurement_engine = RoiMeasurementEngine(app.image_provider)
    app.roi_measurement_engine.measure(manager, include_profiles=True)
    app.current_roi_object_id = track.object_id
    return app, track


def test_live_plots_block_all_stale_exports_and_recover_after_measurement(qtbot, tmp_path, monkeypatch):
    app, track = _profile_app()
    profile = RoiProfileWindow.from_app(app)
    scalar = RoiScalarPlotWindow.from_app(app)
    qtbot.addWidget(profile)
    qtbot.addWidget(scalar)
    app._roi_profile_windows.append(profile)
    app._roi_scalar_plot_windows.append(scalar)
    assert len(profile.profiles) == 4
    assert profile.export_csv(tmp_path / "fresh.csv").exists()
    assert profile.export_svg(tmp_path / "fresh.svg").exists()

    app.edit_history.do(SetRoiFrameGeometry(
        app.roi_manager, track.object_id, 1,
        ThickPolyline2D(1, ((1, 3), (4, 3)), Thickness(0.5, "um")),
    ))
    assert "stale" in profile._status_label.text()
    assert "stale" in scalar._status.text()
    for window in (profile, scalar):
        assert not window._export_button.isEnabled()
        assert not window._export_svg_button.isEnabled()
        assert not window._toolbar._save_action.isEnabled()
        with pytest.raises(RuntimeError, match="stale"):
            window.export_csv(tmp_path / "stale.csv")
        with pytest.raises(RuntimeError, match="stale"):
            window.export_svg(tmp_path / "stale.svg")
    assert not (tmp_path / "stale.csv").exists()
    assert not (tmp_path / "stale.svg").exists()

    warnings = []
    monkeypatch.setattr(
        "acetree_py.gui.roi_profile_window.QFileDialog.getSaveFileName",
        lambda *_args: (str(tmp_path / "toolbar.svg"), ""),
    )
    monkeypatch.setattr(
        "acetree_py.gui.roi_profile_window.QMessageBox.warning",
        lambda _owner, _title, message: warnings.append(message),
    )
    profile._toolbar.save_figure()
    assert warnings and "stale" in warnings[-1]
    assert not (tmp_path / "toolbar.svg").exists()

    snapshot = app.roi_measurement_engine.measure(app.roi_manager, include_profiles=True)
    app._refresh_roi_scalar_plot_windows(snapshot)
    for window in (profile, scalar):
        assert window._export_button.isEnabled()
        assert window._toolbar._save_action.isEnabled()
    assert profile.export_csv(tmp_path / "remeasured.csv").exists()
    before_reads = app.image_provider.pixel_reads
    app.edit_history.do(AssociateRoiFrame(
        app.roi_manager, track.object_id, 1,
        CellRef(nucleus_anchor=NucleusAnchor(1, 1), name_snapshot="ABa"),
    ))
    assert profile._export_button.isEnabled()
    assert app.image_provider.pixel_reads == before_reads
    app.image_provider.manifest_token = "profile-images-v2"
    with pytest.raises(RuntimeError, match="stale"):
        profile.export_csv(tmp_path / "changed-source.csv")
    assert not (tmp_path / "changed-source.csv").exists()

    # An open plot stays bound to its original dataset and image provider,
    # even when a replacement carries identical document/manifest metadata.
    manager, provider = app.roi_manager, app.image_provider
    provider.manifest_token = "profile-images-v1"
    for replacement_manager, replacement_provider in (
        (manager, None), (manager, _ProfileProvider()),
        (RoiManager(manager.document), provider),
    ):
        app.roi_manager = replacement_manager
        app.image_provider = replacement_provider
        for window in (profile, scalar):
            window.on_document_edited()
            assert not window._export_button.isEnabled()
            assert not window._toolbar._save_action.isEnabled()
            with pytest.raises(RuntimeError, match="stale"):
                window.export_csv(tmp_path / "replacement.csv")
            with pytest.raises(RuntimeError, match="stale"):
                window.export_svg(tmp_path / "replacement.svg")
    assert not (tmp_path / "replacement.csv").exists()
    assert not (tmp_path / "replacement.svg").exists()
    app.roi_manager, app.image_provider = manager, provider


def test_plot_and_export_validate_profile_dependencies_once_per_operation(qtbot, tmp_path, monkeypatch):
    app, _track = _profile_app()
    profile = RoiProfileWindow.from_app(app)
    scalar = RoiScalarPlotWindow.from_app(app)
    qtbot.addWidget(profile)
    qtbot.addWidget(scalar)
    original = roi_measurements._image_token
    checks = []

    def image_token(*args, **kwargs):
        checks.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(roi_measurements, "_image_token", image_token)
    for action in (
        profile.refresh_plot,
        lambda: profile.export_csv(tmp_path / "profiles.csv"),
        lambda: profile.export_svg(tmp_path / "profiles.svg"),
        lambda: scalar.export_svg(tmp_path / "scalars.svg"),
    ):
        checks.clear()
        action()
        assert len(checks) == 1

    document = app.roi_manager.document
    app.roi_manager.replace_document(replace(
        document, coordinate_space=replace(document.coordinate_space, xy_res=1.0),
    ))
    profile.on_document_edited()
    assert not profile._export_button.isEnabled()
    with pytest.raises(RuntimeError, match="stale"):
        profile.export_csv(tmp_path / "changed-calibration.csv")
