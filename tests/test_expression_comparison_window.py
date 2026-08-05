"""Focused Qt workflow tests for the multi-dataset comparison window."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from acetree_py.gui.expression_comparison_window import ExpressionComparisonWindow
    from acetree_py.gui.app import AceTreeApp
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMainWindow

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _GUI_AVAILABLE, reason="Qt/Matplotlib GUI unavailable")

from acetree_py.analysis.expression_comparison import (
    BandStatistic,
    CenterStatistic,
)
from acetree_py.analysis.expression_comparison_result import (
    ExpressionComparisonSourceMode,
    load_expression_comparison_result,
)
from acetree_py.analysis.expression_dataset_repository import (
    CanonicalCellAmbiguousError,
    CanonicalCellNotFoundError,
    DatasetSourceChangedError,
    ExpressionDatasetStatus,
    ExpressionChannelUnavailableError,
    ExpressionDataIncompleteError,
    ExpressionTraceFreshness,
    ExpressionTraceProvenance,
    ExpressionTraceSource,
    NativeExpressionTrace,
)


class _FakeRepository:
    def __init__(self) -> None:
        self._statuses: dict[str, ExpressionDatasetStatus] = {}
        self.cells: dict[str, tuple[str, ...]] = {}
        self.values: dict[str, tuple[float, ...]] = {}
        self.measure_calls: list[tuple[str, str]] = []
        self.image_channel_calls: list[str] = []
        self.image_channel_counts: dict[str, int] = {}
        self.remove_calls: list[Path] = []
        self.changed: set[str] = set()
        self.close_calls = 0

    @staticmethod
    def _key(path: str | Path) -> str:
        return str(Path(path).resolve()).casefold()

    def load_dataset(self, path: str | Path) -> ExpressionDatasetStatus:
        canonical = Path(path).resolve()
        key = self._key(canonical)
        status = self._statuses.get(key)
        if status is None:
            status = ExpressionDatasetStatus(
                config_path=canonical,
                source_fingerprint=f"fingerprint:{canonical.name}",
                num_timepoints=3,
                num_cells=1,
                image_provider_loaded=False,
                cached_corrections=(),
                cell_names=("ABa",),
            )
            self._statuses[key] = status
            self.cells.setdefault(key, ("ABa",))
            self.values.setdefault(key, (1.0, 2.0, 3.0))
        return status

    def status(self, path: str | Path) -> ExpressionDatasetStatus:
        key = self._key(path)
        if key in self.changed:
            raise DatasetSourceChangedError("source changed")
        return self._statuses[key]

    def statuses(self) -> tuple[ExpressionDatasetStatus, ...]:
        return tuple(self._statuses.values())

    def session_statuses(self) -> tuple[ExpressionDatasetStatus, ...]:
        return self.statuses()

    def session_status(self, path: str | Path) -> ExpressionDatasetStatus:
        return self._statuses[self._key(path)]

    def reload_dataset(self, path: str | Path) -> ExpressionDatasetStatus:
        key = self._key(path)
        previous = self._statuses[key]
        self.changed.discard(key)
        status = replace(
            previous,
            source_fingerprint=previous.source_fingerprint + ":reloaded",
            snapshot_token=previous.snapshot_token + ":reloaded",
            generation=previous.generation + 1,
            image_provider_loaded=False,
            cached_corrections=(),
        )
        self._statuses[key] = status
        return status

    def cell_names(self, path: str | Path) -> tuple[str, ...]:
        return self.cells[self._key(path)]

    def image_channel_count(self, path: str | Path) -> int:
        key = self._key(path)
        self.image_channel_calls.append(key)
        return self.image_channel_counts.get(key, 2)

    def extract_saved_trace(
        self,
        path: str | Path,
        cell: str,
        channel: str,
    ) -> NativeExpressionTrace:
        key = self._key(path)
        if cell not in self.cells[key]:
            raise CanonicalCellNotFoundError(cell)
        return self._trace(
            path,
            cell,
            channel,
            "Saved AT expression",
            self.values[key],
            ExpressionTraceProvenance(
                source=ExpressionTraceSource.SAVED_LEGACY,
                freshness=ExpressionTraceFreshness.UNVERIFIED,
                image_channel=None,
                correction_method=None,
                at_channel=None,
                channel_verified=False,
                correction_verified=False,
            ),
        )

    def extract_recomputed_trace(
        self,
        path: str | Path,
        cell: str,
        image_channel: int,
        correction_method: str,
        *,
        progress_cb=None,
    ) -> NativeExpressionTrace:
        key = self._key(path)
        if cell not in self.cells[key]:
            raise CanonicalCellNotFoundError(cell)
        status = self._statuses[key]
        if correction_method not in status.cached_corrections:
            self.measure_calls.append((key, correction_method))
            if progress_cb is not None:
                assert progress_cb(0, 2, 1, 3)
                assert progress_cb(1, 2, 3, 3)
            status = replace(
                status,
                image_provider_loaded=True,
                cached_corrections=tuple(
                    sorted((*status.cached_corrections, correction_method))
                ),
            )
            self._statuses[key] = status
        values = tuple(value + 100.0 * (image_channel + 1) for value in self.values[key])
        return self._trace(
            path,
            cell,
            f"measured_channel_{image_channel + 1}",
            f"Channel {image_channel + 1}",
            values,
            ExpressionTraceProvenance(
                source=ExpressionTraceSource.RECOMPUTED,
                freshness=ExpressionTraceFreshness.CURRENT_SESSION,
                image_channel=image_channel,
                correction_method=correction_method,
                at_channel=0,
                channel_verified=True,
                correction_verified=True,
            ),
        )

    def remove_dataset(self, path: str | Path) -> bool:
        self.remove_calls.append(Path(path))
        return True

    def close(self) -> None:
        self.close_calls += 1

    def _trace(
        self,
        path: str | Path,
        cell: str,
        key: str,
        label: str,
        values: tuple[float, ...],
        provenance: ExpressionTraceProvenance,
    ) -> NativeExpressionTrace:
        status = self._statuses[self._key(path)]
        return NativeExpressionTrace(
            dataset_path=status.config_path,
            dataset_fingerprint=status.source_fingerprint,
            cell_name=cell,
            cell_key=f"{status.config_path.stem}:{cell}",
            start_time=1,
            end_time=len(values),
            timepoints=tuple(range(1, len(values) + 1)),
            values=values,
            channel_key=key,
            channel_label=label,
            channel_unit="scaled mean intensity",
            provenance=provenance,
            dataset_snapshot_token=status.snapshot_token,
            dataset_generation=status.generation,
            image_manifest_token=status.image_manifest_token,
        )


class _ExplodingRepository:
    """Repository sentinel proving that frozen windows never call it."""

    def __init__(self) -> None:
        self.calls = 0

    def __getattr__(self, name):
        self.calls += 1
        raise AssertionError(f"frozen mode called repository.{name}")


def _app():
    return SimpleNamespace(
        manager=SimpleNamespace(config=None),
        current_cell_name="ABa",
        _expression_comparison_windows=[],
    )


def _xml(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.xml"
    path.write_text("<embryo/>", encoding="utf-8")
    return path


def _set_source(window: ExpressionComparisonWindow, source: str) -> None:
    index = window._source_combo.findData(source)
    assert index >= 0
    window._source_combo.setCurrentIndex(index)


def test_bulk_add_deduplicates_and_remove_keeps_shared_repository(qtbot, tmp_path):
    repository = _FakeRepository()
    app = _app()
    window = ExpressionComparisonWindow(app, repository)
    qtbot.addWidget(window)
    first = _xml(tmp_path, "first")
    alias = first.parent / "." / first.name

    assert window.add_dataset_paths([first, alias], show_errors=False) == 1
    assert window._dataset_table.rowCount() == 1
    window._dataset_table.selectRow(0)
    window.remove_selected_datasets()

    assert window._dataset_table.rowCount() == 0
    assert repository.remove_calls == []
    assert repository.status(first).config_path == first.resolve()


def test_saved_legacy_requires_explicit_acknowledgement_for_export(qtbot, tmp_path):
    repository = _FakeRepository()
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths([_xml(tmp_path, "saved")], show_errors=False)

    window.prepare_included_datasets()

    assert window._plot_data is not None
    assert not window._btn_export_csv.isEnabled()
    with pytest.raises(RuntimeError, match="unknown channel, correction, and freshness"):
        window._exportable_snapshot()

    window._legacy_ack.setChecked(True)
    assert window._btn_export_csv.isEnabled()
    assert window._exportable_snapshot() is window._plot_data
    destination = window.export_csv(tmp_path / "acknowledged.csv")
    rows = list(csv.DictReader(destination.open(encoding="utf-8")))
    metadata = json.loads(rows[0]["provenance_metadata"])
    assert metadata["legacy_acknowledgement"] == "true"


def test_acquisition_failures_stay_selected_and_export_as_status_rows(qtbot, tmp_path):
    repository = _FakeRepository()
    paths = [
        _xml(tmp_path, name)
        for name in ("available", "ambiguous", "incomplete", "no-channel")
    ]
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths(paths, show_errors=False)
    original = repository.extract_saved_trace

    def extract_saved(path, cell, channel):
        stem = Path(path).stem
        if stem == "ambiguous":
            raise CanonicalCellAmbiguousError("two canonical matches")
        if stem == "incomplete":
            raise ExpressionDataIncompleteError("missing value at time 2")
        if stem == "no-channel":
            raise ExpressionChannelUnavailableError("rweight is absent")
        return original(path, cell, channel)

    repository.extract_saved_trace = extract_saved
    window.prepare_included_datasets()
    window._legacy_ack.setChecked(True)

    data = window._plot_data
    assert data is not None
    assert [status.availability.value for status in data.statuses] == [
        "available",
        "ambiguous",
        "incomplete_data",
        "missing_channel",
    ]
    assert data.summaries[0].n_selected == 4
    assert data.summaries[0].n_available == 1
    assert "Recompute from image channel" in data.statuses[2].message
    assert all(
        state.resolution in ("ready", "acquisition_status")
        for state in window._included_states()
    )

    destination = window.export_csv(tmp_path / "acquisition-statuses.csv")
    rows = list(csv.DictReader(destination.open(encoding="utf-8")))
    statuses = {
        row["dataset_label"]: row["trace_status"]
        for row in rows
        if row["record_type"] == "trace_status"
    }
    assert statuses == {
        "available": "available",
        "ambiguous": "ambiguous",
        "incomplete": "incomplete_data",
        "no-channel": "missing_channel",
    }


def test_status_only_comparison_can_export_csv_but_not_svg(qtbot, tmp_path):
    repository = _FakeRepository()
    path = _xml(tmp_path, "incomplete-only")
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths([path], show_errors=False)

    def incomplete(*_args, **_kwargs):
        raise ExpressionDataIncompleteError("legacy sample missing")

    repository.extract_saved_trace = incomplete
    window.prepare_included_datasets()

    assert window._plot_data is not None
    assert not window._plot_data.has_data
    assert window._btn_export_csv.isEnabled()
    assert not window._btn_export_svg.isEnabled()
    destination = window.export_csv(tmp_path / "status-only.csv")
    rows = list(csv.DictReader(destination.open(encoding="utf-8")))
    assert [row["trace_status"] for row in rows] == ["incomplete_data"]
    assert rows[0]["n_selected"] == "1"
    assert rows[0]["n_available"] == "0"


def test_portable_result_is_offline_retunable_and_resaves_as_child(
    qtbot, tmp_path
):
    repository = _FakeRepository()
    app = _app()
    source_paths = [_xml(tmp_path, "offline-a"), _xml(tmp_path, "offline-b")]
    live = ExpressionComparisonWindow(app, repository)
    qtbot.addWidget(live)
    live.add_dataset_paths(source_paths, show_errors=False)
    _set_source(live, "recomputed")
    live.prepare_included_datasets()
    live._dataset_table.item(0, live.COL_LABEL).setText("Control")
    live._dataset_table.item(0, live.COL_GROUP).setText("control")
    live._dataset_table.item(0, live.COL_COLOR).setText("#123456")
    live._dataset_table.item(1, live.COL_LABEL).setText("Treatment")
    live._dataset_table.item(1, live.COL_GROUP).setText("treated")
    live._dataset_table.item(1, live.COL_USE).setCheckState(Qt.Unchecked)
    live._title_edit.setText("Portable view")
    live._trace_opacity.setValue(0.45)
    captured_path = live.save_portable_result(tmp_path / "offline-capture")
    captured = load_expression_comparison_result(captured_path)
    original_overrides = captured.appearance["dataset_overrides"]
    captured = replace(
        captured,
        appearance={
            **dict(captured.appearance),
            "future_extension": {"enabled": True},
            "dataset_overrides": {
                dataset_id: {
                    **dict(values),
                    "future_dataset_extension": "preserved",
                }
                for dataset_id, values in original_overrides.items()
            },
        },
    )

    assert len(captured.datasets) == 2
    assert captured.appearance["included_dataset_ids"] == (
        captured.datasets[0].provenance.dataset_id,
    )
    for source_path in source_paths:
        source_path.unlink()

    sentinel = _ExplodingRepository()
    frozen_app = _app()
    frozen = ExpressionComparisonWindow(
        frozen_app,
        repository=sentinel,
        result=captured,
        result_path=str(captured_path),
        window_number=7,
    )
    qtbot.addWidget(frozen)

    assert "Frozen Expression Result" in frozen.windowTitle()
    assert "FROZEN RESULT" in frozen._intro_label.text()
    assert not frozen._btn_add.isVisible()
    assert not frozen._btn_prepare.isEnabled()
    assert not frozen._source_combo.isEnabled()
    assert frozen._source_combo.currentData() == "recomputed"
    assert not frozen._saved_channel_combo.isVisible()
    assert not frozen._image_channel.isHidden()
    assert not frozen._correction_combo.isHidden()
    assert frozen._image_channel.value() == 1
    assert frozen._correction_combo.currentData() == "global"
    assert frozen._dataset_table.item(0, frozen.COL_LABEL).text() == "Control"
    assert not frozen._datasets[captured.datasets[1].provenance.dataset_id].included
    assert sentinel.calls == 0

    frozen._time_combo.setCurrentIndex(
        frozen._time_combo.findData("normalized")
    )
    frozen._normalized_points.setValue(7)
    frozen._smoothing_check.setChecked(True)
    frozen._smoothing_sigma.setValue(0.2)
    frozen._center_combo.setCurrentIndex(
        frozen._center_combo.findData(CenterStatistic.MEDIAN.value)
    )
    frozen._band_combo.setCurrentIndex(
        frozen._band_combo.findData(BandStatistic.IQR.value)
    )
    frozen._dataset_table.item(0, frozen.COL_LABEL).setText("Retuned control")
    frozen._dataset_table.item(0, frozen.COL_COLOR).setText("#654321")

    assert frozen._plot_data is not None
    assert frozen._plot_data.spec.grid.normalized_points == 7
    assert frozen._plot_data.spec.smoothing.sigma == pytest.approx(0.2)
    assert frozen._plot_data.spec.summary.center is CenterStatistic.MEDIAN
    csv_path = frozen.export_csv(tmp_path / "offline-values")
    svg_path = frozen.export_svg(tmp_path / "offline-figure")
    assert csv_path.is_file()
    assert "<svg" in svg_path.read_text(encoding="utf-8")
    assert sentinel.calls == 0

    revised_path = frozen.save_portable_result(tmp_path / "offline-revised")
    revised = load_expression_comparison_result(revised_path)
    assert revised.parent_result_id == captured.result_id
    assert revised.captured_at == captured.captured_at
    assert revised.datasets[0].provenance.label == "Retuned control"
    assert revised.datasets[0].traces[0].color == "#654321"
    assert revised.appearance["title"] == "Portable view"
    assert revised.appearance["trace_opacity"] == pytest.approx(0.45)
    assert revised.appearance["future_extension"] == {"enabled": True}
    assert all(
        values["future_dataset_extension"] == "preserved"
        for values in revised.appearance["dataset_overrides"].values()
    )
    assert sentinel.calls == 0


def test_frozen_status_only_result_allows_csv_and_portable_save_not_svg(
    qtbot, tmp_path
):
    repository = _FakeRepository()
    live = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(live)
    live.add_dataset_paths([_xml(tmp_path, "status-capture")], show_errors=False)

    def incomplete(*_args, **_kwargs):
        raise ExpressionDataIncompleteError("legacy sample missing")

    repository.extract_saved_trace = incomplete
    live.prepare_included_datasets()
    path = live.save_portable_result(tmp_path / "status-capture")
    result = load_expression_comparison_result(path)
    sentinel = _ExplodingRepository()
    frozen = ExpressionComparisonWindow(
        _app(), repository=sentinel, result=result, result_path=str(path)
    )
    qtbot.addWidget(frozen)

    assert frozen._btn_export_csv.isEnabled()
    assert frozen._btn_save_result.isEnabled()
    assert not frozen._btn_export_svg.isEnabled()
    assert not frozen._toolbar._save_action.isEnabled()
    assert frozen.export_csv(tmp_path / "frozen-status").is_file()
    with pytest.raises(RuntimeError, match="no frozen expression comparison"):
        frozen.export_svg(tmp_path / "must-not-exist")
    assert sentinel.calls == 0


def test_frozen_unacknowledged_legacy_values_fail_closed(qtbot, tmp_path):
    repository = _FakeRepository()
    live = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(live)
    live.add_dataset_paths([_xml(tmp_path, "legacy-unacknowledged")], show_errors=False)
    live.prepare_included_datasets()
    live._legacy_ack.setChecked(True)
    captured_path = live.save_portable_result(tmp_path / "legacy-acknowledged")
    captured = load_expression_comparison_result(captured_path)
    unacknowledged = replace(captured, legacy_acknowledged=False)

    frozen = ExpressionComparisonWindow(
        _app(),
        repository=_ExplodingRepository(),
        result=unacknowledged,
        result_path=str(captured_path),
    )
    qtbot.addWidget(frozen)

    assert "legacy acknowledgement recorded: no" in frozen._cell_availability.text()
    assert not frozen._btn_export_csv.isEnabled()
    assert not frozen._btn_export_svg.isEnabled()
    assert not frozen._btn_save_result.isEnabled()
    assert captured.acquisition_metadata["saved_channel_key"] == "rweight"
    assert "image_channel_one_based" not in captured.acquisition_metadata
    assert "correction_method" not in captured.acquisition_metadata
    assert not frozen._saved_channel_combo.isHidden()
    assert not frozen._image_channel.isVisible()
    assert not frozen._correction_combo.isVisible()
    with pytest.raises(RuntimeError, match="frozen result contains legacy numeric"):
        frozen.export_csv(tmp_path / "must-not-export")


def test_repeated_live_result_saves_form_revision_lineage(qtbot, tmp_path):
    repository = _FakeRepository()
    live = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(live)
    live.add_dataset_paths([_xml(tmp_path, "live-lineage")], show_errors=False)
    _set_source(live, "recomputed")
    live.prepare_included_datasets()

    first = load_expression_comparison_result(
        live.save_portable_result(tmp_path / "live-first")
    )
    live._title_edit.setText("Presentation revision")
    second = load_expression_comparison_result(
        live.save_portable_result(tmp_path / "live-second")
    )

    assert second.parent_result_id == first.result_id
    assert second.captured_at == first.captured_at
    assert second.appearance["title"] == "Presentation revision"


def test_mixed_unacknowledged_result_allows_recomputed_only_subset(qtbot, tmp_path):
    repository = _FakeRepository()
    live = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(live)
    live.add_dataset_paths(
        [_xml(tmp_path, "mixed-recomputed"), _xml(tmp_path, "mixed-legacy")],
        show_errors=False,
    )
    _set_source(live, "recomputed")
    live.prepare_included_datasets()
    live._dataset_table.item(1, live.COL_USE).setCheckState(Qt.Unchecked)
    path = live.save_portable_result(tmp_path / "mixed-source")
    result = load_expression_comparison_result(path)
    legacy_dataset = result.datasets[1]
    legacy_metadata = dict(legacy_dataset.provenance.metadata)
    legacy_metadata["trace_source"] = ExpressionTraceSource.SAVED_LEGACY.value
    mixed = replace(
        result,
        source_mode=ExpressionComparisonSourceMode.MIXED,
        legacy_acknowledged=False,
        datasets=(
            result.datasets[0],
            replace(
                legacy_dataset,
                provenance=replace(
                    legacy_dataset.provenance,
                    metadata=tuple(legacy_metadata.items()),
                ),
            ),
        ),
    )

    frozen = ExpressionComparisonWindow(_app(), repository=None, result=mixed)
    qtbot.addWidget(frozen)

    assert frozen._source_combo.currentData() == "mixed"
    assert frozen._btn_export_csv.isEnabled()
    assert frozen.export_csv(tmp_path / "mixed-recomputed-only").is_file()


def test_frozen_result_rejects_uneditable_grid_precision(qtbot, tmp_path):
    repository = _FakeRepository()
    live = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(live)
    live.add_dataset_paths([_xml(tmp_path, "precise-grid")], show_errors=False)
    _set_source(live, "recomputed")
    live.prepare_included_datasets()
    result = load_expression_comparison_result(
        live.save_portable_result(tmp_path / "precise-grid")
    )
    precise = replace(
        result,
        spec=replace(
            result.spec,
            grid=replace(result.spec.grid, step=0.0001),
        ),
    )

    with pytest.raises(ValueError, match="outside the range or precision"):
        ExpressionComparisonWindow(_app(), repository=None, result=precise)


def test_comparison_progress_is_monotonic_for_time_major_callbacks(
    qtbot, tmp_path, monkeypatch
):
    from acetree_py.gui import expression_comparison_window as window_module

    class TimeMajorRepository(_FakeRepository):
        def extract_recomputed_trace(self, *args, progress_cb=None, **kwargs):
            if progress_cb is not None:
                for timepoint in (1, 2, 3):
                    for channel in (0, 1):
                        assert progress_cb(channel, 2, timepoint, 3)
            return super().extract_recomputed_trace(
                *args, progress_cb=None, **kwargs
            )

    class Progress:
        instance = None

        def __init__(self, *_args):
            type(self).instance = self
            self.values = []
            self._cancelled = False

        def setWindowTitle(self, _value): pass
        def setWindowModality(self, _value): pass
        def setMinimumDuration(self, _value): pass
        def setLabelText(self, _value): pass
        def close(self): pass
        def cancel(self): self._cancelled = True
        def wasCanceled(self): return self._cancelled
        def setValue(self, value): self.values.append(value)
        def value(self): return self.values[-1] if self.values else 0

    monkeypatch.setattr(window_module, "QProgressDialog", Progress)
    repository = TimeMajorRepository()
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths([_xml(tmp_path, "time-major")], show_errors=False)
    _set_source(window, "recomputed")
    window.prepare_included_datasets()

    assert Progress.instance is not None
    assert Progress.instance.values == sorted(Progress.instance.values)
    assert Progress.instance.values[-1] == 1000


def test_recomputed_measurement_cache_is_reused_across_windows(qtbot, tmp_path):
    repository = _FakeRepository()
    path = _xml(tmp_path, "shared")
    first = ExpressionComparisonWindow(_app(), repository, window_number=1)
    qtbot.addWidget(first)
    first.add_dataset_paths([path], show_errors=False)
    _set_source(first, "recomputed")
    first.prepare_included_datasets()

    second = ExpressionComparisonWindow(_app(), repository, window_number=2)
    qtbot.addWidget(second)
    assert second._dataset_table.rowCount() == 1
    _set_source(second, "recomputed")
    second.prepare_included_datasets()

    for window in (first, second):
        assert window._plot_data is not None
        assert window._btn_export_csv.isEnabled()

    assert len(repository.measure_calls) == 1
    assert "Cached: global" in second._dataset_table.item(0, second.COL_CACHE).text()

    first._dataset_table.selectRow(0)
    first.reload_selected_datasets()
    first.prepare_included_datasets()

    with pytest.raises(RuntimeError, match="no longer matches"):
        second._exportable_snapshot()
    second_state = next(iter(second._datasets.values()))
    assert second_state.resolution == "unprepared"
    assert "Prepare this row again" in second_state.message
    assert not second._btn_export_svg.isEnabled()

    second.prepare_included_datasets()

    assert len(repository.measure_calls) == 2
    assert second._exportable_snapshot() is second._plot_data


def test_image_channels_are_lazy_and_limited_to_common_recomputed_range(
    qtbot, tmp_path
):
    repository = _FakeRepository()
    first_path = _xml(tmp_path, "two-channels")
    second_path = _xml(tmp_path, "four-channels")
    repository.image_channel_counts[repository._key(first_path)] = 2
    repository.image_channel_counts[repository._key(second_path)] = 4
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)

    window.add_dataset_paths([first_path, second_path], show_errors=False)

    assert repository.image_channel_calls == []
    assert window._saved_channel_combo.isEnabled()
    assert not window._image_channel.isEnabled()
    assert not window._correction_combo.isEnabled()

    _set_source(window, "recomputed")

    assert len(repository.image_channel_calls) == 2
    assert window._image_channel.maximum() == 2
    assert not window._saved_channel_combo.isEnabled()
    assert window._image_channel.isEnabled()
    assert window._correction_combo.isEnabled()


def test_prepare_restores_only_controls_for_the_active_source(qtbot, tmp_path):
    repository = _FakeRepository()
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths([_xml(tmp_path, "source-controls")], show_errors=False)

    _set_source(window, "recomputed")
    window.prepare_included_datasets()

    assert not window._saved_channel_combo.isEnabled()
    assert window._image_channel.isEnabled()
    assert window._correction_combo.isEnabled()

    _set_source(window, "saved")
    window.prepare_included_datasets()

    assert window._saved_channel_combo.isEnabled()
    assert not window._image_channel.isEnabled()
    assert not window._correction_combo.isEnabled()


def test_statistics_time_grid_and_plot_controls_feed_snapshot(qtbot, tmp_path):
    repository = _FakeRepository()
    first_path = _xml(tmp_path, "one")
    second_path = _xml(tmp_path, "two")
    first_key = repository._key(first_path)
    second_key = repository._key(second_path)
    repository.load_dataset(first_path)
    repository.load_dataset(second_path)
    repository.values[first_key] = (1.0, 3.0, 5.0)
    repository.values[second_key] = (3.0, 5.0, 7.0)
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths([first_path, second_path], show_errors=False)
    _set_source(window, "recomputed")
    window._center_combo.setCurrentIndex(
        window._center_combo.findData(CenterStatistic.MEAN.value)
    )
    window._band_combo.setCurrentIndex(
        window._band_combo.findData(BandStatistic.SEM.value)
    )
    window._smoothing_sigma.setValue(0.0)
    window.prepare_included_datasets()

    data = window._plot_data
    assert data is not None
    assert data.summaries[0].center == (102.0, 104.0, 106.0)
    assert data.summaries[0].band_statistic is BandStatistic.SEM
    assert data.spec.grid.step == 1.0

    normalized = window._time_combo.findData("normalized")
    window._time_combo.setCurrentIndex(normalized)
    window._normalized_points.setValue(5)
    assert window._plot_data is not None
    assert window._plot_data.summaries[0].grid_x == (0.0, 0.25, 0.5, 0.75, 1.0)

    window._smoothing_check.setChecked(True)
    window._smoothing_sigma.setValue(0.25)
    assert window._plot_data is not None
    assert window._plot_data.spec.smoothing.sigma == 0.25


def test_center_choice_limits_error_bands_to_valid_statistics(qtbot):
    window = ExpressionComparisonWindow(_app(), _FakeRepository())
    qtbot.addWidget(window)

    assert window._band_combo.findData(BandStatistic.SEM.value) >= 0
    assert window._band_combo.findData(BandStatistic.IQR.value) == -1
    window._center_combo.setCurrentIndex(
        window._center_combo.findData(CenterStatistic.MEDIAN.value)
    )

    assert window._band_combo.findData(BandStatistic.SEM.value) == -1
    assert window._band_combo.findData(BandStatistic.STUDENT_T_95.value) == -1
    assert window._band_combo.findData(BandStatistic.IQR.value) >= 0
    assert window._band_combo.findData(BandStatistic.SCALED_MAD.value) >= 0


def test_initial_axis_control_state_and_blank_svg_guard(qtbot, tmp_path):
    window = ExpressionComparisonWindow(_app(), _FakeRepository())
    qtbot.addWidget(window)
    window.add_dataset_paths([_xml(tmp_path, "blank-guard")], show_errors=False)
    _set_source(window, "recomputed")
    window.prepare_included_datasets()

    assert not window._normalized_points.isEnabled()
    assert window._grid_step.isEnabled()
    window._show_traces.setChecked(False)
    window._center_combo.setCurrentIndex(
        window._center_combo.findData(CenterStatistic.NONE.value)
    )

    assert window._btn_export_csv.isEnabled()
    assert not window._btn_export_svg.isEnabled()
    assert "Enable individual traces" in window._status_label.text()


def test_appearance_controls_render_and_toggle_consistently(qtbot, tmp_path):
    from matplotlib.colors import to_rgba

    repository = _FakeRepository()
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths([_xml(tmp_path, "appearance")], show_errors=False)
    _set_source(window, "recomputed")
    window.prepare_included_datasets()

    assert window._auto_x.isChecked()
    assert window._auto_y.isChecked()
    assert not window._x_min.isEnabled()
    assert not window._x_max.isEnabled()
    assert not window._y_min.isEnabled()
    assert not window._y_max.isEnabled()
    # Both endpoints must actually be parented into the rendered form.
    assert window._x_max.parent() is not None
    assert window._y_max.parent() is not None

    window._auto_x.setChecked(False)
    window._auto_y.setChecked(False)
    assert window._x_min.isEnabled() and window._x_max.isEnabled()
    assert window._y_min.isEnabled() and window._y_max.isEnabled()
    window._x_min.setValue(1.0)
    window._x_max.setValue(3.0)
    window._y_min.setValue(100.0)
    window._y_max.setValue(104.0)

    window._title_edit.setText("Styled comparison")
    window._x_label_edit.setText("Custom time")
    window._y_label_edit.setText("Custom signal")
    window._legend_title.setText("Replicates")
    window._legend_columns.setValue(2)
    window._font_size.setValue(12.0)
    window._title_size.setValue(16.0)
    window._trace_opacity.setValue(0.55)
    window._trace_width.setValue(1.7)
    window._center_width.setValue(3.2)
    window._trace_line_style.setCurrentIndex(
        window._trace_line_style.findData("--")
    )
    window._center_line_style.setCurrentIndex(
        window._center_line_style.findData(":")
    )
    window._marker_combo.setCurrentIndex(window._marker_combo.findData("o"))
    window._figure_background = "#abcdef"
    window._axes_background = "#fedcba"
    window._text_color = "#123456"
    window._refresh_plot()

    assert window._axes.get_xlim() == pytest.approx((1.0, 3.0))
    assert window._axes.get_ylim() == pytest.approx((100.0, 104.0))
    assert window._axes.get_title() == "Styled comparison"
    assert window._axes.get_xlabel() == "Custom time"
    assert window._axes.get_ylabel() == "Custom signal"
    assert window._axes.title.get_fontsize() == pytest.approx(16.0)
    assert window._axes.title.get_color() == "#123456"
    assert window._axes.get_facecolor() == pytest.approx(to_rgba("#fedcba"))
    assert window._figure.get_facecolor() == pytest.approx(to_rgba("#abcdef"))

    trace_line = next(line for line in window._axes.lines if line.get_label() == "appearance")
    center_line = next(
        line for line in window._axes.lines if line.get_label().startswith("all: Mean")
    )
    assert trace_line.get_alpha() == pytest.approx(0.55)
    assert trace_line.get_linewidth() == pytest.approx(1.7)
    assert trace_line.get_linestyle() == "--"
    assert trace_line.get_marker() == "o"
    assert center_line.get_linewidth() == pytest.approx(3.2)
    assert center_line.get_linestyle() == ":"
    legend = window._axes.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "Replicates"
    assert legend._ncols == 2

    window._show_traces.setChecked(False)
    for widget in (
        window._trace_opacity,
        window._trace_line_style,
        window._trace_width,
        window._marker_combo,
        window._marker_size,
    ):
        assert not widget.isEnabled()
    window._show_traces.setChecked(True)
    assert window._trace_opacity.isEnabled()


def test_editable_conditions_create_unpooled_colored_group_summaries(qtbot, tmp_path):
    repository = _FakeRepository()
    paths = [_xml(tmp_path, name) for name in ("control-1", "control-2", "treated")]
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths(paths, show_errors=False)
    repository.values[repository._key(paths[0])] = (1.0, 1.0, 1.0)
    repository.values[repository._key(paths[1])] = (3.0, 3.0, 3.0)
    repository.values[repository._key(paths[2])] = (10.0, 10.0, 10.0)

    window._dataset_table.item(0, window.COL_GROUP).setText("control")
    window._dataset_table.item(1, window.COL_GROUP).setText("control")
    window._dataset_table.item(2, window.COL_GROUP).setText("treated")
    window._dataset_table.item(0, window.COL_COLOR).setText("#123456")
    window._dataset_table.item(2, window.COL_COLOR).setText("#654321")
    _set_source(window, "recomputed")
    window.prepare_included_datasets()

    data = window._plot_data
    assert data is not None
    assert [(summary.group_id, summary.center[0]) for summary in data.summaries] == [
        ("control", 102.0),
        ("treated", 110.0),
    ]
    assert [summary.n_selected for summary in data.summaries] == [2, 1]
    summary_lines = {
        line.get_label(): line.get_color()
        for line in window._axes.lines
        if ": Mean " in line.get_label()
    }
    assert summary_lines["control: Mean (n≤2/2)"] == "#123456"
    assert summary_lines["treated: Mean (n≤1/1)"] == "#654321"


def test_exact_csv_svg_export_and_source_revalidation(qtbot, tmp_path):
    repository = _FakeRepository()
    path = _xml(tmp_path, "export")
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths([path], show_errors=False)
    _set_source(window, "recomputed")
    window.prepare_included_datasets()

    csv_path = window.export_csv(tmp_path / "nested" / "values")
    svg_path = window.export_svg(tmp_path / "figure")

    text = csv_path.read_text(encoding="utf-8")
    assert "record_type" in text
    assert "aligned_sample" in text
    assert svg_path.suffix == ".svg"
    assert "<svg" in svg_path.read_text(encoding="utf-8")

    repository.changed.add(repository._key(path))
    with pytest.raises(RuntimeError, match="changed after preparation"):
        window._exportable_snapshot()
    assert not window._btn_export_svg.isEnabled()

    window._dataset_table.selectRow(0)
    window.reload_selected_datasets()
    window.prepare_included_datasets()

    assert window._plot_data is not None
    assert window._btn_export_svg.isEnabled()
    assert window._exportable_snapshot() is window._plot_data


def test_toolbar_save_revalidates_before_opening_svg_dialog(
    qtbot, tmp_path, monkeypatch
):
    from acetree_py.gui import expression_plot_window as plot_window_module

    repository = _FakeRepository()
    path = _xml(tmp_path, "toolbar-export")
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)
    window.add_dataset_paths([path], show_errors=False)
    _set_source(window, "recomputed")
    window.prepare_included_datasets()

    save_action = window._toolbar._save_action
    assert save_action is not None
    assert save_action.isEnabled()
    dialog_calls: list[str] = []
    warning_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(window, "_choose_svg_path", lambda: dialog_calls.append("svg"))
    monkeypatch.setattr(
        plot_window_module,
        "QMessageBox",
        SimpleNamespace(
            warning=lambda _parent, title, message: warning_calls.append(
                (title, message)
            )
        ),
    )

    save_action.trigger()
    assert dialog_calls == ["svg"]
    assert warning_calls == []

    repository.changed.add(repository._key(path))
    save_action.trigger()

    assert dialog_calls == ["svg"]
    assert warning_calls
    assert "changed after preparation" in warning_calls[-1][1]
    assert not save_action.isEnabled()


def test_stale_session_dataset_remains_visible_for_reload(qtbot, tmp_path):
    repository = _FakeRepository()
    path = _xml(tmp_path, "stale-session")
    repository.load_dataset(path)
    repository.changed.add(repository._key(path))

    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)

    assert window._dataset_table.rowCount() == 1
    state = next(iter(window._datasets.values()))
    assert state.resolution == "error"
    assert "Reload selected" in state.message


def test_cell_typing_uses_cached_names_without_movie_revalidation(qtbot, tmp_path):
    repository = _FakeRepository()
    path = _xml(tmp_path, "cached-names")
    repository.load_dataset(path)

    def unexpected_cell_names(_path):
        raise AssertionError("cell_names should not be queried while typing")

    repository.cell_names = unexpected_cell_names  # type: ignore[method-assign]
    window = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(window)

    window._cell_combo.setEditText("A")
    window._cell_combo.setEditText("AB")
    window._cell_combo.setEditText("ABa")
    assert "available in 1/1" in window._cell_availability.text()


def test_dirty_current_dataset_is_marked_before_detached_prepare(qtbot, tmp_path):
    repository = _FakeRepository()
    path = _xml(tmp_path, "dirty-current")
    manager = SimpleNamespace(
        config=SimpleNamespace(config_file=path),
        _config_dirty=False,
    )
    app = SimpleNamespace(
        manager=manager,
        edit_history=SimpleNamespace(modified=True),
        current_cell_name="ABa",
        _expression_comparison_windows=[],
    )

    window = ExpressionComparisonWindow(app, repository)
    qtbot.addWidget(window)

    state = next(iter(window._datasets.values()))
    assert state.resolution == "error"
    assert "unsaved edits" in state.message


def test_close_unregisters_only_this_window(qtbot):
    repository = _FakeRepository()
    app = _app()
    first = ExpressionComparisonWindow(app, repository, window_number=1)
    second = ExpressionComparisonWindow(app, repository, window_number=2)
    app._expression_comparison_windows[:] = [first, second]
    qtbot.addWidget(first)
    qtbot.addWidget(second)

    first.close()

    assert app._expression_comparison_windows == [second]
    assert first not in app._expression_comparison_windows


def test_app_shutdown_closes_and_releases_shared_repository():
    manager = SimpleNamespace(nuclei_record=[], config=None)
    app = AceTreeApp(manager)
    repository = _FakeRepository()
    app._expression_dataset_repository = repository

    app._shutdown_expression_dataset_repository()

    assert repository.close_calls == 1
    assert app._expression_dataset_repository is None
    # The about-to-quit hook can be reached more than once during teardown.
    app._shutdown_expression_dataset_repository()
    assert repository.close_calls == 1


def test_window_menu_exposes_multi_instance_expression_comparison(
    qtbot, monkeypatch
):
    manager = SimpleNamespace(nuclei_record=[], config=None)
    app = AceTreeApp(manager)
    repository = _FakeRepository()
    app._expression_dataset_repository = repository
    qt_window = QMainWindow()
    qtbot.addWidget(qt_window)
    qt_window.menuBar().addMenu("&Window")
    app.viewer = SimpleNamespace(
        window=SimpleNamespace(_qt_window=qt_window, _dock_widgets={})
    )

    app._add_panel_menu_actions()
    action = app._panel_menu_actions["new_expression_comparison"]
    open_result_action = app._panel_menu_actions["open_expression_result"]

    assert "Expression Comparison" in action.text()
    assert open_result_action.text() == "Open Expression Result…"
    assert ".aceexpr" in open_result_action.statusTip()
    assert "multiple AceTree XML datasets" in action.statusTip()
    dialog_calls = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (dialog_calls.append("open") or "", ""),
    )
    open_result_action.trigger()
    assert dialog_calls == ["open"]
    assert app._expression_comparison_windows == []
    action.trigger()
    action.trigger()

    assert len(app._expression_comparison_windows) == 2
    assert app._expression_comparison_windows[0] is not app._expression_comparison_windows[1]
    assert (
        app._expression_comparison_windows[0].windowTitle()
        != app._expression_comparison_windows[1].windowTitle()
    )
    assert all(
        window.repository is repository
        for window in app._expression_comparison_windows
    )
    for window in tuple(app._expression_comparison_windows):
        window.close()


def test_app_opens_result_without_repository_and_malformed_open_is_atomic(
    qtbot, tmp_path, monkeypatch
):
    repository = _FakeRepository()
    live = ExpressionComparisonWindow(_app(), repository)
    qtbot.addWidget(live)
    source = _xml(tmp_path, "app-open")
    live.add_dataset_paths([source], show_errors=False)
    _set_source(live, "recomputed")
    live.prepare_included_datasets()
    result_path = live.save_portable_result(tmp_path / "app-open")
    source.unlink()

    manager = SimpleNamespace(nuclei_record=[], config=None)
    app = AceTreeApp(manager)
    app._expression_dataset_repository = None
    app.viewer = None
    window = app.open_expression_comparison_result_window(result_path)
    assert window is not None
    qtbot.addWidget(window)
    assert app._expression_dataset_repository is None
    assert app._expression_comparison_windows == [window]

    warnings = []
    monkeypatch.setattr(
        "qtpy.QtWidgets.QMessageBox.warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    malformed = tmp_path / "malformed.aceexpr"
    malformed.write_text("{}", encoding="utf-8")
    before_windows = tuple(app._expression_comparison_windows)
    before_counter = app._expression_comparison_window_counter
    assert app.open_expression_comparison_result_window(malformed) is None
    assert tuple(app._expression_comparison_windows) == before_windows
    assert app._expression_comparison_window_counter == before_counter
    assert warnings and warnings[-1][0] == "Cannot open expression result"


def test_panel_actions_create_window_menu_when_napari_has_none(qtbot):
    manager = SimpleNamespace(nuclei_record=[], config=None)
    app = AceTreeApp(manager)
    qt_window = QMainWindow()
    qtbot.addWidget(qt_window)
    app.viewer = SimpleNamespace(
        window=SimpleNamespace(_qt_window=qt_window, _dock_widgets={})
    )

    app._add_panel_menu_actions()

    assert any(
        "window" in action.text().lower().replace("&", "")
        for action in qt_window.menuBar().actions()
    )
    assert "new_expression_comparison" in app._panel_menu_actions
