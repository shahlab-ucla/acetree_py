"""Focused strictness, recovery, and concurrency tests for ROI sidecars."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from acetree_py.core.roi_manager import RoiManager, RoiWriteProtectedError
from acetree_py.core.subcellular_roi import (
    CellRef,
    ContourSlice,
    ContourStack3D,
    CoordinateSpaceSnapshot,
    NucleusAnchor,
    ObjectClass,
    Polygon2D,
    RoiFrameRecord,
    RoiObjectTrack,
    SubcellularRoiDocument,
    ThickPolyline2D,
    Thickness,
)
from acetree_py.io.roi_sidecar import (
    ROI_MIGRATIONS,
    ROI_SIDECAR_VERSION,
    RoiJsonLimits,
    RoiSidecarChecksumError,
    RoiSidecarConflictError,
    RoiSidecarFormatError,
    canonical_document_bytes,
    commit_staged_roi_sidecar,
    discard_staged_roi_sidecar,
    document_checksum,
    read_roi_sidecar,
    register_roi_migration,
    roi_envelope_to_dict,
    roi_sidecar_path,
    stage_roi_sidecar,
    write_roi_sidecar,
)


def _document() -> SubcellularRoiDocument:
    space = CoordinateSpaceSnapshot(
        plane_start=3,
        xy_res=0.2,
        z_res=1.0,
        image_width_px=100,
        image_height_px=80,
        plane_count=5,
        time_start=1,
        time_end=10,
    )
    object_class = ObjectClass(
        "Golgi", (0.1, 0.2, 0.8, 1.0), next_instance_index=4
    )
    ref = CellRef(
        NucleusAnchor(2, 7),
        NucleusAnchor(1, 3),
        "ABa",
        (10, 11, 3),
    )
    geometries = (
        Polygon2D(3, ((1, 1), (5, 1), (5, 5), (1, 5))),
        ThickPolyline2D(4, ((1, 1), (4, 5)), Thickness(0.8, "um")),
        ContourStack3D(
            (
                ContourSlice(3, ((1, 1), (5, 1), (3, 5))),
                ContourSlice(4, ((2, 1), (6, 1), (4, 5))),
            ),
            "inner_shell",
            0.6,
        ),
    )
    frames = {
        index: RoiFrameRecord(index, "segmented", "reviewed", geometry, ref if index == 2 else None)
        for index, geometry in enumerate(geometries, start=1)
    }
    track = RoiObjectTrack(object_class.class_id, 3, frames=frames)
    return SubcellularRoiDocument(
        coordinate_space=space,
        object_classes=(object_class,),
        objects=(track,),
        file_revision=7,
        roi_revision=11,
        created_at="2026-08-21T00:00:00Z",
        saved_at="2026-08-21T00:01:00Z",
        dataset_fingerprint="dataset-1",
        extensions={"vendor": {"enabled": True}},
    )


def _rewrite_envelope(path: Path, envelope: dict) -> None:
    envelope["checksum"]["sha256"] = document_checksum(envelope["document"])
    path.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")


def test_xml_derived_sidecar_naming():
    assert roi_sidecar_path(Path("embryo.xml")) == Path("embryo.subcellular-rois.json")
    assert roi_sidecar_path(None, Path("nuclei.zip")) == Path(
        "nuclei.subcellular-rois.json"
    )


def test_every_geometry_round_trips_losslessly_with_checksum(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    original = _document()
    token = write_roi_sidecar(path, original)
    loaded = read_roi_sidecar(path)

    assert loaded is not None
    assert loaded.document == original
    assert loaded.token == token
    assert token.checksum == document_checksum(roi_envelope_to_dict(original)["document"])
    assert b"NaN" not in canonical_document_bytes(roi_envelope_to_dict(original)["document"])


def test_tampered_document_checksum_is_rejected(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    write_roi_sidecar(path, _document())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["document"]["objects"][0]["instance_index"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RoiSidecarChecksumError, match="checksum mismatch"):
        read_roi_sidecar(path)


def test_duplicate_keys_and_nonfinite_numbers_are_rejected(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(RoiSidecarFormatError, match="duplicate"):
        read_roi_sidecar(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema":NaN}', encoding="utf-8")
    with pytest.raises(RoiSidecarFormatError, match="non-finite"):
        read_roi_sidecar(nonfinite)


def test_file_depth_and_collection_limits_are_enforced(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    write_roi_sidecar(path, _document())
    with pytest.raises(RoiSidecarFormatError, match="byte limit"):
        read_roi_sidecar(path, limits=RoiJsonLimits(max_file_bytes=10))
    with pytest.raises(RoiSidecarFormatError, match="nesting-depth"):
        read_roi_sidecar(path, limits=RoiJsonLimits(max_depth=2))
    with pytest.raises(RoiSidecarFormatError, match="collection limit"):
        read_roi_sidecar(path, limits=RoiJsonLimits(max_items_per_collection=2))


def test_one_invalid_frame_is_quarantined_without_losing_track(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    envelope = roi_envelope_to_dict(_document())
    envelope["document"]["objects"][0]["frames"][1]["geometry"][
        "points_xy_px"
    ] = [[1, 1]]
    _rewrite_envelope(path, envelope)

    loaded = read_roi_sidecar(path)

    assert loaded is not None
    assert len(loaded.document.objects) == 1
    assert tuple(loaded.document.objects[0].frames) == (1, 3)
    assert len(loaded.document.quarantine) == 1
    assert ".frames[1]" in loaded.document.quarantine[0].path


def test_unknown_newer_version_loads_read_only(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    envelope = roi_envelope_to_dict(_document())
    envelope["schema_version"] = ROI_SIDECAR_VERSION + 1
    envelope["document"]["future_document_field"] = {"preserved_by_source": True}
    envelope["document"]["coordinate_space"]["future_axis_mode"] = "native"
    envelope["document"]["objects"][0]["frames"][0]["geometry"][
        "future_ring_metadata"
    ] = {"origin": "v2"}
    envelope["checksum"]["sha256"] = document_checksum(envelope["document"])
    path.write_text(json.dumps(envelope), encoding="utf-8")

    loaded = read_roi_sidecar(path)
    assert loaded is not None and loaded.read_only
    manager = RoiManager.from_load(loaded)
    with pytest.raises(RoiWriteProtectedError, match="write-protected"):
        manager.stage_save()


def test_pure_migration_interface_never_mutates_source():
    from acetree_py.io.roi_sidecar import migrate_roi_payload

    source = {"schema_version": 0, "document": {"legacy": True}}

    def migrate(value):
        value["schema_version"] = 1
        value["document"]["migrated"] = True
        return value

    previous = ROI_MIGRATIONS.get(0)
    register_roi_migration(0, migrate)
    try:
        migrated = migrate_roi_payload(source)
    finally:
        if previous is None:
            ROI_MIGRATIONS.pop(0, None)
        else:
            ROI_MIGRATIONS[0] = previous
    assert source == {"schema_version": 0, "document": {"legacy": True}}
    assert migrated["document"]["migrated"] is True


def test_external_revision_or_checksum_change_blocks_staging(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    original = _document()
    write_roi_sidecar(path, original)
    loaded = read_roi_sidecar(path)
    assert loaded is not None
    external = replace(original, file_revision=8)
    write_roi_sidecar(path, external, expected_token=loaded.token)

    with pytest.raises(RoiSidecarConflictError, match="changed externally"):
        stage_roi_sidecar(path, replace(original, file_revision=8), expected_token=loaded.token)


def test_commit_rechecks_destination_after_staging(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    original = _document()
    initial = write_roi_sidecar(path, original)
    staged = stage_roi_sidecar(
        path, replace(original, file_revision=8), expected_token=initial
    )
    write_roi_sidecar(path, replace(original, file_revision=9), expected_token=initial)
    try:
        with pytest.raises(RoiSidecarConflictError, match="changed externally"):
            commit_staged_roi_sidecar(staged)
    finally:
        discard_staged_roi_sidecar(staged)


def test_manager_stage_and_external_finalize_do_not_replace_twice(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    manager = RoiManager(_document(), sidecar_path=path)
    # This manager represents a new destination, so its loaded token is absent.
    staged = manager.stage_save()
    assert staged.temp_path.exists()
    commit_staged_roi_sidecar(staged.sidecar)
    token = manager.finalize_external_commit(staged)

    assert token == manager.loaded_token
    assert manager.file_revision == 8
    assert not manager.is_dirty
    assert path.exists() and not staged.temp_path.exists()


def test_manager_save_as_uses_new_destination_precondition(tmp_path: Path):
    source = tmp_path / "source.subcellular-rois.json"
    target = tmp_path / "copy.subcellular-rois.json"
    manager = RoiManager(_document(), sidecar_path=source)
    manager.save()
    source_bytes = source.read_bytes()

    staged = manager.stage_save(target)
    assert staged.sidecar.expected_token is None
    manager.commit_save(staged)

    assert source.read_bytes() == source_bytes
    assert target.exists()
    assert manager.sidecar_path == target


def test_authorizing_malformed_replacement_marks_explicit_empty_document_dirty(
    tmp_path: Path,
):
    path = tmp_path / "broken.subcellular-rois.json"
    path.write_text("{broken", encoding="utf-8")
    manager = RoiManager(
        SubcellularRoiDocument.empty(),
        sidecar_path=path,
        load_error="invalid JSON",
    )

    manager.authorize_replace_invalid_sidecar()
    assert manager.is_dirty
    manager.save()

    loaded = read_roi_sidecar(path)
    assert loaded is not None
    assert loaded.document.objects == ()
    assert not manager.is_write_protected
    assert not manager.is_dirty


def test_malformed_autoload_is_protected_and_recovery_copy_is_allowed(tmp_path: Path):
    xml = tmp_path / "embryo.xml"
    xml.write_text("<config/>", encoding="utf-8")
    sidecar = tmp_path / "embryo.subcellular-rois.json"
    sidecar.write_text("{broken", encoding="utf-8")
    config = SimpleNamespace(
        config_file=xml,
        zip_file=tmp_path / "nuclei.zip",
        plane_start=1,
        plane_end=3,
        xy_res=0.2,
        z_res=1.0,
        starting_index=1,
        ending_index=10,
        split=1,
        flip=1,
    )
    manager = RoiManager.from_config(config)

    assert manager.is_write_protected
    with pytest.raises(RoiWriteProtectedError):
        manager.create_class("Golgi", (0.1, 0.2, 0.8, 1.0))
    recovery = tmp_path / "recovery.json"
    manager.save_recovery_copy(recovery)
    assert recovery.exists()
    assert sidecar.read_text(encoding="utf-8") == "{broken"


def test_explicit_empty_document_overwrites_last_deleted_object(tmp_path: Path):
    path = tmp_path / "embryo.subcellular-rois.json"
    manager = RoiManager(_document(), sidecar_path=path)
    manager.save()
    manager.delete_object(manager.objects[0].object_id)
    manager.save()

    loaded = read_roi_sidecar(path)
    assert loaded is not None
    assert loaded.document.objects == ()
