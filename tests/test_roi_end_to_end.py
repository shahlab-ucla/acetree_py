import numpy as np

from acetree_py.editing.roi_commands import (
    CreateObjectClass,
    CreateRoiObject,
    MarkRoiFrameReviewed,
    SetRoiFrameGeometry,
)
from acetree_py.gui.app import AceTreeApp
from acetree_py.io.config import AceTreeConfig
from acetree_py.io.config_writer import write_config_xml
from acetree_py.io.image_provider import NumpyProvider
from acetree_py.io.nuclei_writer import write_nuclei_zip
from acetree_py.io.roi_sidecar import roi_sidecar_path
from acetree_py.core.subcellular_roi import Polygon2D


def _dataset(tmp_path):
    xml_path = tmp_path / "embryo.xml"
    zip_path = tmp_path / "nuclei.zip"
    config = AceTreeConfig(
        config_file=xml_path,
        zip_file=zip_path,
        starting_index=1,
        ending_index=1,
        plane_start=1,
        plane_end=1,
        xy_res=0.5,
        z_res=1.0,
        split=0,
        flip=0,
    )
    write_nuclei_zip([[]], zip_path)
    write_config_xml(config, xml_path)
    images = np.empty((1, 2, 1, 8, 8), dtype=np.float32)
    images[:, 0] = 2.0
    images[:, 1] = 3.0
    return xml_path, zip_path, images


def test_roi_draw_measure_save_reopen_reproduces_identity_and_values(tmp_path):
    xml_path, zip_path, images = _dataset(tmp_path)
    app = AceTreeApp.from_config(xml_path, image_provider=NumpyProvider(images))

    create_class = CreateObjectClass(
        app.roi_manager,
        "Golgi",
        (0.9, 0.5, 0.1, 1.0),
    )
    app.edit_history.do(create_class)
    create_object = CreateRoiObject(
        app.roi_manager,
        create_class.created_class_id,
        expected_start_time=1,
        expected_end_time=1,
    )
    app.edit_history.do(create_object)
    object_id = create_object.created_object_id
    geometry = Polygon2D(
        z_plane=1,
        exterior_xy_px=((1, 1), (4, 1), (4, 4), (1, 4)),
    )
    app.edit_history.do(
        SetRoiFrameGeometry(app.roi_manager, object_id, 1, geometry)
    )
    app.edit_history.do(MarkRoiFrameReviewed(app.roi_manager, object_id, 1))

    before = app.roi_measurement_engine.measure(app.roi_manager)
    before_values = {
        channel: before.value(object_id, 1, channel, "intensity.sum")
        for channel in (0, 1)
    }
    assert before_values[0] and before_values[1]
    assert app.save() == zip_path
    assert roi_sidecar_path(xml_path).exists()
    assert not app.edit_history.modified
    assert not app.roi_manager.is_dirty

    reopened = AceTreeApp.from_config(
        xml_path,
        image_provider=NumpyProvider(images.copy()),
    )
    reopened_track = reopened.roi_manager.get_object(object_id)
    assert reopened_track is not None
    assert reopened_track.instance_index == 1
    assert reopened.roi_manager.get_class(create_class.created_class_id).name == "Golgi"
    assert reopened_track.frames[1].geometry == geometry
    assert reopened_track.frames[1].review_state.value == "reviewed"

    after = reopened.roi_measurement_engine.measure(reopened.roi_manager)
    assert {
        channel: after.value(object_id, 1, channel, "intensity.sum")
        for channel in (0, 1)
    } == before_values


def test_late_xml_failure_restores_nuclei_roi_and_config_generation(
    tmp_path, monkeypatch
):
    import acetree_py.io.dataset_transaction as transaction_module

    xml_path, zip_path, images = _dataset(tmp_path)
    app = AceTreeApp.from_config(xml_path, image_provider=NumpyProvider(images))
    object_class = app.roi_manager.create_class(
        "Membrane", (0.2, 0.8, 0.9, 1.0)
    )
    track = app.roi_manager.create_object(object_class.class_id)
    original_geometry = Polygon2D(
        1, ((1, 1), (4, 1), (4, 4), (1, 4))
    )
    app.roi_manager.update_frame_geometry(track.object_id, 1, original_geometry)
    assert app.save() == zip_path
    sidecar = roi_sidecar_path(xml_path)
    previous = (zip_path.read_bytes(), sidecar.read_bytes(), xml_path.read_bytes())

    app.edit_history.do(
        SetRoiFrameGeometry(
            app.roi_manager,
            track.object_id,
            1,
            Polygon2D(1, ((2, 2), (5, 2), (5, 5), (2, 5))),
        )
    )
    app.manager.config.expr_corr = "global"
    app.manager._config_dirty = True
    real_replace = transaction_module.os.replace
    failed = False

    def fail_new_xml_install(source, destination):
        nonlocal failed
        if (
            not failed
            and destination == xml_path
            and str(source).endswith(".save-config.tmp")
        ):
            failed = True
            raise OSError("injected final XML failure")
        return real_replace(source, destination)

    monkeypatch.setattr(transaction_module.os, "replace", fail_new_xml_install)

    assert app.save() is None
    assert failed
    assert (zip_path.read_bytes(), sidecar.read_bytes(), xml_path.read_bytes()) == previous
    assert app.edit_history.modified
    assert app.roi_manager.is_dirty
    assert app.manager._config_dirty
