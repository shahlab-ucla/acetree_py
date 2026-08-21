from __future__ import annotations

import csv
from types import SimpleNamespace

from acetree_py.analysis.roi_measure import RoiSpatialProfile
from acetree_py.core.roi_manager import RoiManager
from acetree_py.core.subcellular_roi import (
    ObjectClass,
    RoiObjectTrack,
    SubcellularRoiDocument,
)
from acetree_py.gui.roi_measure_dialog import RoiMeasureDialog
from acetree_py.gui.roi_profile_window import (
    RoiProfileSeries,
    export_roi_profiles_csv,
)


def test_measure_dialog_builds_headless_engine_request(qtbot):
    object_class = ObjectClass("Golgi", (0.2, 0.8, 1.0, 1.0), next_instance_index=2)
    track = RoiObjectTrack(class_id=object_class.class_id, instance_index=1)
    manager = RoiManager(SubcellularRoiDocument(
        object_classes=(object_class,),
        objects=(track,),
    ))
    app = SimpleNamespace(
        roi_manager=manager,
        image_provider=SimpleNamespace(num_channels=2),
        current_roi_object_id=track.object_id,
        current_time=4,
    )
    dialog = RoiMeasureDialog(app)
    qtbot.addWidget(dialog)
    dialog._scope_combo.setCurrentIndex(1)
    dialog._time_combo.setCurrentIndex(1)
    request = dialog.build_request()
    assert request.document is manager
    assert request.object_ids == (str(track.object_id),)
    assert request.timepoints == (4,)
    assert request.channels == (0, 1)
    assert request.metric_keys == (
        "intensity.sum",
        "intensity.mean",
        "intensity.median",
    )


def test_profile_csv_preserves_zero_and_missing_reason(tmp_path):
    profile = RoiSpatialProfile(
        distance_um=(0.0, 1.0),
        mean=(0.0, None),
        median=(0.0, None),
        sample_count=(2, 0),
        missing_reasons=(None, "no_finite_pixels"),
        step_um=1.0,
        width_step_um=1.0,
    )
    path = export_roi_profiles_csv(
        tmp_path / "profile.csv",
        (RoiProfileSeries("Golgi #1, t=4", profile, "object-1", 4, 1),),
    )
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["value"] == "0"
    assert rows[0]["image_channel"] == "2"
    assert rows[1]["value"] == ""
    assert rows[1]["missing_reason"] == "no_finite_pixels"
