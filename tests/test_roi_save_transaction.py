from pathlib import Path

import pytest

from acetree_py.io import dataset_transaction as transaction_module
from acetree_py.io.dataset_transaction import DatasetTransaction


def _stage(destination: Path, payload: bytes) -> Path:
    staged = destination.with_name(f".{destination.name}.stage")
    staged.write_bytes(payload)
    return staged


def test_transaction_installs_payloads_in_order(tmp_path):
    roi = tmp_path / "embryo.subcellular-rois.json"
    xml = tmp_path / "embryo.xml"
    roi.write_bytes(b"old-roi")
    xml.write_bytes(b"old-xml")

    transaction = DatasetTransaction()
    transaction.add(_stage(roi, b"new-roi"), roi)
    transaction.add(_stage(xml, b"new-xml"), xml)
    transaction.commit()

    assert roi.read_bytes() == b"new-roi"
    assert xml.read_bytes() == b"new-xml"
    assert list(tmp_path.glob("*.rollback")) == []


def test_transaction_restores_every_prior_artifact_on_late_failure(
    tmp_path, monkeypatch
):
    roi = tmp_path / "embryo.subcellular-rois.json"
    xml = tmp_path / "embryo.xml"
    roi.write_bytes(b"old-roi")
    xml.write_bytes(b"old-xml")
    roi_stage = _stage(roi, b"new-roi")
    xml_stage = _stage(xml, b"new-xml")
    real_replace = transaction_module.os.replace

    def fail_xml_install(source, destination):
        if Path(source) == xml_stage and Path(destination) == xml:
            raise OSError("injected XML commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(transaction_module.os, "replace", fail_xml_install)
    transaction = DatasetTransaction()
    transaction.add(roi_stage, roi)
    transaction.add(xml_stage, xml)

    with pytest.raises(OSError, match="injected"):
        transaction.commit()

    assert roi.read_bytes() == b"old-roi"
    assert xml.read_bytes() == b"old-xml"
    assert list(tmp_path.glob("*.rollback")) == []


def test_transaction_rejects_cross_directory_stage(tmp_path):
    destination_dir = tmp_path / "destination"
    stage_dir = tmp_path / "stage"
    destination_dir.mkdir()
    stage_dir.mkdir()
    destination = destination_dir / "embryo.xml"
    staged = stage_dir / "embryo.xml.stage"
    staged.write_bytes(b"new")

    transaction = DatasetTransaction()
    transaction.add(staged, destination)
    with pytest.raises(ValueError, match="private siblings"):
        transaction.commit()


def test_failed_rollback_preserves_recoverable_backup(tmp_path, monkeypatch):
    roi = tmp_path / "embryo.subcellular-rois.json"
    xml = tmp_path / "embryo.xml"
    roi.write_bytes(b"old-roi")
    xml.write_bytes(b"old-xml")
    roi_stage = _stage(roi, b"new-roi")
    xml_stage = _stage(xml, b"new-xml")
    real_replace = transaction_module.os.replace

    def fail_install_and_roi_restore(source, destination):
        source = Path(source)
        destination = Path(destination)
        if source == xml_stage and destination == xml:
            raise OSError("injected XML install failure")
        if source.suffix == ".rollback" and destination == roi:
            raise OSError("injected ROI restore failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        transaction_module.os,
        "replace",
        fail_install_and_roi_restore,
    )
    transaction = DatasetTransaction()
    transaction.add(roi_stage, roi)
    transaction.add(xml_stage, xml)

    with pytest.raises(RuntimeError, match="could not be restored"):
        transaction.commit()

    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"old-roi"
    assert not roi.exists()
    assert xml.read_bytes() == b"old-xml"
