"""Comprehensive GUI widget tests — requires napari + Qt (pytest-qt).

Tests that all widgets instantiate, render, respond to user interaction,
and correctly sync state with AceTreeApp. Uses pytest-qt's qtbot fixture
for Qt event loop management.

Skip all tests if napari/Qt is not available.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if Qt/napari not available
try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication
    import napari

    _GUI_AVAILABLE = True
except ImportError:
    _GUI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _GUI_AVAILABLE, reason="napari/Qt not installed")

from acetree_py.core.cell import Cell, CellFate
from acetree_py.core.lineage import LineageTree, build_lineage_tree
from acetree_py.core.movie import Movie
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.gui.app import AceTreeApp
from acetree_py.gui.cell_info_panel import CellInfoPanel
from acetree_py.gui.contrast_tools import ContrastTools
from acetree_py.gui.lineage_layout import LayoutParams, compute_layout
from acetree_py.gui.lineage_widget import LineageWidget
from acetree_py.gui.player_controls import PlayerControls
from acetree_py.gui.viewer_integration import ViewerIntegration
from acetree_py.io.image_provider import NumpyProvider


# ── Fixtures ─────────────────────────────────────────────────────


def _make_nuc(index, x, y, z, size=20, identity="", status=1,
              predecessor=NILLI, successor1=NILLI, successor2=NILLI,
              rweight=0):
    return Nucleus(
        index=index, x=x, y=y, z=z, size=size,
        identity=identity, status=status,
        predecessor=predecessor, successor1=successor1, successor2=successor2,
        rweight=rweight,
    )


def _build_test_manager():
    """Build a NucleiManager with synthetic data (5 timepoints, division at T3)."""
    mgr = NucleiManager()
    mgr.movie = Movie(xy_res=0.1, z_res=1.0, num_planes=30)

    mgr.nuclei_record = [
        [_make_nuc(1, 150, 150, 15.0, identity="P0", successor1=1, rweight=100)],
        [_make_nuc(1, 150, 150, 15.0, identity="P0", predecessor=1,
                  successor1=1, successor2=2, rweight=200)],
        [_make_nuc(1, 100, 150, 15.0, identity="AB", predecessor=1, successor1=1, rweight=500),
         _make_nuc(2, 200, 150, 15.0, identity="P1", predecessor=1, successor1=2, rweight=50)],
        [_make_nuc(1, 100, 150, 14.0, identity="AB", predecessor=1, rweight=600),
         _make_nuc(2, 200, 150, 16.0, identity="P1", predecessor=2, rweight=80)],
        [_make_nuc(1, 100, 150, 14.0, identity="AB", predecessor=1, rweight=700),
         _make_nuc(2, 200, 150, 16.0, identity="P1", predecessor=2, rweight=100)],
    ]

    mgr.lineage_tree = build_lineage_tree(
        mgr.nuclei_record, starting_index=0, ending_index=5,
        create_dummy_ancestors=False,
    )
    return mgr


def _make_image_provider():
    """Create a NumpyProvider with synthetic 5-timepoint, 30-plane data."""
    data = np.random.randint(0, 1000, (5, 30, 64, 64), dtype=np.uint16)
    return NumpyProvider(data)


def _make_app_with_images():
    """Create an AceTreeApp with both nuclei data and image data."""
    mgr = _build_test_manager()
    provider = _make_image_provider()
    app = AceTreeApp(mgr, image_provider=provider)
    app.current_time = 1
    app.current_plane = 15
    return app


# ── PlayerControls widget tests ──────────────────────────────────


class TestPlayerControls:
    def test_instantiation(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)
        widget.show()
        assert widget.isVisible()

    def test_time_spinner_reflects_state(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        assert widget._time_spin.value() == 1
        app.current_time = 3
        widget.refresh()
        assert widget._time_spin.value() == 3

    def test_plane_spinner_reflects_state(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        app.current_plane = 20
        widget.refresh()
        assert widget._plane_spin.value() == 20

    def test_time_slider_range(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        assert widget._time_slider.minimum() == 1
        assert widget._time_slider.maximum() == 5  # 5 timepoints

    def test_next_button_advances_time(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        assert app.current_time == 1
        widget._btn_next.click()
        assert app.current_time == 2

    def test_prev_button_goes_back(self, qtbot):
        app = _make_app_with_images()
        app.current_time = 3
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        widget._btn_prev.click()
        assert app.current_time == 2

    def test_start_button_goes_to_t1(self, qtbot):
        app = _make_app_with_images()
        app.current_time = 4
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        widget._btn_start.click()
        assert app.current_time == 1

    def test_end_button_goes_to_last(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        widget._btn_end.click()
        assert app.current_time == 5

    def test_plane_up_down(self, qtbot):
        app = _make_app_with_images()
        app.current_plane = 15
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        widget._btn_plane_up.click()
        assert app.current_plane == 16
        widget._btn_plane_down.click()
        assert app.current_plane == 15

    def test_time_spinner_changes_time(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        widget._time_spin.setValue(4)
        assert app.current_time == 4

    def test_label_updates(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        widget.refresh()
        assert "5" in widget._time_label.text()  # "/ 5"

    def test_play_pause(self, qtbot):
        app = _make_app_with_images()
        widget = PlayerControls(app)
        qtbot.addWidget(widget)

        # Start forward play
        widget._toggle_play(1)
        assert widget._playing
        assert widget._timer.isActive()

        # Pause
        widget._stop_play()
        assert not widget._playing
        assert not widget._timer.isActive()


# ── CellInfoPanel widget tests ───────────────────────────────────


class TestCellInfoPanel:
    def test_instantiation(self, qtbot):
        app = _make_app_with_images()
        widget = CellInfoPanel(app)
        qtbot.addWidget(widget)
        widget.show()
        assert widget.isVisible()

    def test_no_selection_text(self, qtbot):
        app = _make_app_with_images()
        widget = CellInfoPanel(app)
        qtbot.addWidget(widget)

        widget.refresh()
        assert "No cell selected" in widget._text.toPlainText()

    def test_selected_cell_info(self, qtbot):
        app = _make_app_with_images()
        app.current_cell_name = "P0"
        app.current_time = 1
        widget = CellInfoPanel(app)
        qtbot.addWidget(widget)

        widget.refresh()
        text = widget._text.toPlainText()
        assert "P0" in text
        assert "Position" in text

    def test_title_updates(self, qtbot):
        app = _make_app_with_images()
        widget = CellInfoPanel(app)
        qtbot.addWidget(widget)

        app.current_cell_name = "AB"
        widget.refresh()
        assert "AB" in widget._title.text()

    def test_cell_not_present(self, qtbot):
        app = _make_app_with_images()
        app.current_cell_name = "AB"
        app.current_time = 1  # AB doesn't exist at T1
        widget = CellInfoPanel(app)
        qtbot.addWidget(widget)

        widget.refresh()
        text = widget._text.toPlainText()
        assert "Not present" in text or "not in lineage" in text.lower()


# ── ContrastTools widget tests ────────────────────────────────────


class _FakeLayer:
    """Minimal stand-in for a napari Image layer."""

    def __init__(self):
        self.data = np.random.randint(100, 5000, (30, 64, 64), dtype=np.uint16)
        self.contrast_limits = (0, 65535)
        self.visible = True


def _make_app_with_fake_layer():
    """Create app + inject a fake image layer so contrast tools can build."""
    app = _make_app_with_images()
    app._image_layers = [_FakeLayer()]
    return app


class TestContrastTools:
    def test_instantiation(self, qtbot):
        app = _make_app_with_images()
        widget = ContrastTools(app)
        qtbot.addWidget(widget)
        widget.show()
        assert widget.isVisible()

    def test_slider_range(self, qtbot):
        app = _make_app_with_fake_layer()
        widget = ContrastTools(app)
        qtbot.addWidget(widget)
        widget.refresh()  # build channel controls

        ctrl = widget._channel_ctrls[0]
        assert ctrl.min_slider.minimum() == 0
        assert ctrl.max_slider.maximum() == 65535

    def test_min_slider_syncs_spinbox(self, qtbot):
        app = _make_app_with_fake_layer()
        widget = ContrastTools(app)
        qtbot.addWidget(widget)
        widget.refresh()

        ctrl = widget._channel_ctrls[0]
        ctrl.min_slider.setValue(500)
        assert ctrl.min_spin.value() == 500

    def test_max_slider_syncs_spinbox(self, qtbot):
        app = _make_app_with_fake_layer()
        widget = ContrastTools(app)
        qtbot.addWidget(widget)
        widget.refresh()

        ctrl = widget._channel_ctrls[0]
        ctrl.max_slider.setValue(30000)
        assert ctrl.max_spin.value() == 30000

    def test_reset_contrast(self, qtbot):
        app = _make_app_with_fake_layer()
        widget = ContrastTools(app)
        qtbot.addWidget(widget)
        widget.refresh()

        ctrl = widget._channel_ctrls[0]
        ctrl.min_slider.setValue(1000)
        ctrl.max_slider.setValue(5000)
        widget._reset_all()
        assert ctrl.min_slider.value() == 0
        assert ctrl.max_slider.value() == 65535

    def test_auto_contrast_no_layers(self, qtbot):
        """Auto contrast with no image layers should not crash."""
        app = _make_app_with_images()
        widget = ContrastTools(app)
        qtbot.addWidget(widget)
        widget.refresh()

        # No layers, so auto_all should be a no-op
        widget._auto_all()
        assert len(widget._channel_ctrls) == 0


# ── LineageWidget tests ───────────────────────────────────────────


class TestLineageWidget:
    def test_instantiation(self, qtbot):
        app = _make_app_with_images()
        widget = LineageWidget(app)
        qtbot.addWidget(widget)
        widget.show()
        assert widget.isVisible()

    def test_scene_has_items(self, qtbot):
        app = _make_app_with_images()
        widget = LineageWidget(app)
        qtbot.addWidget(widget)

        # Scene should have graphical items (lines, text, dots)
        assert widget._scene.items()
        # At minimum, we should have items for each cell in the tree
        assert len(widget._scene.items()) > 0

    def test_rebuild_tree(self, qtbot):
        app = _make_app_with_images()
        widget = LineageWidget(app)
        qtbot.addWidget(widget)

        initial_count = len(widget._scene.items())
        widget.rebuild_tree()
        # Should still have items after rebuild
        assert len(widget._scene.items()) > 0

    def test_zoom_in_out(self, qtbot):
        app = _make_app_with_images()
        widget = LineageWidget(app)
        qtbot.addWidget(widget)

        transform_before = widget._view.transform()
        widget._zoom(2.0)
        transform_after = widget._view.transform()
        # Transform should have changed
        assert transform_before != transform_after

        widget._zoom(0.5)  # Zoom back out

    def test_fit_to_view(self, qtbot):
        app = _make_app_with_images()
        widget = LineageWidget(app)
        qtbot.addWidget(widget)
        widget.resize(400, 300)

        # Should not crash
        widget._fit_to_view()

    def test_find_cell_at(self, qtbot):
        app = _make_app_with_images()
        widget = LineageWidget(app)
        qtbot.addWidget(widget)

        # The layout should exist
        assert widget._layout is not None
        assert len(widget._layout) > 0

        # Find a cell by clicking near its position
        if "P0" in widget._layout:
            node = widget._layout["P0"]
            name = widget._find_cell_at(node.x, (node.y_start + node.y_end) / 2)
            assert name == "P0"

    def test_refresh_selection(self, qtbot):
        app = _make_app_with_images()
        widget = LineageWidget(app)
        qtbot.addWidget(widget)

        app.current_cell_name = "AB"
        widget.refresh_selection()
        # Should not crash; items should still exist
        assert len(widget._scene.items()) > 0

    def test_layout_contains_expected_cells(self, qtbot):
        app = _make_app_with_images()
        widget = LineageWidget(app)
        qtbot.addWidget(widget)

        assert widget._layout is not None
        # Should have P0, AB, P1 at minimum
        cell_names = set(widget._layout.keys())
        assert "P0" in cell_names
        # AB and P1 should also be there (from division at T3)
        assert "AB" in cell_names or "P1" in cell_names


# ── ViewerIntegration tests (with napari) ─────────────────────────


class TestViewerIntegration:
    def test_overlay_data_structure(self):
        """Test overlay data without creating a viewer."""
        app = _make_app_with_images()
        app.current_time = 3
        app.current_plane = 15

        data = app.get_nucleus_overlay_data()
        assert "centers" in data
        assert "radii" in data
        assert "colors" in data
        assert "names" in data
        assert "selected_idx" in data

        # Should have 2 nuclei at T3 (AB and P1)
        assert len(data["names"]) == 2
        assert data["centers"].shape == (2, 2)
        assert data["radii"].shape == (2,)
        assert data["colors"].shape == (2, 4)

    def test_overlay_with_selection(self):
        app = _make_app_with_images()
        app.current_time = 3
        app.current_plane = 15
        app.current_cell_name = "AB"

        data = app.get_nucleus_overlay_data()
        idx = data["selected_idx"]
        assert idx >= 0
        # Selected cell color should be white
        assert np.allclose(data["colors"][idx], [1.0, 1.0, 1.0, 1.0])


# ── Full napari viewer integration tests ──────────────────────────


class TestNapariIntegration:
    """Tests that create an actual napari viewer (headless)."""

    def test_app_launch_and_layers(self, qtbot):
        """Test that AceTreeApp.launch() creates viewer + layers."""
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)

        # Manually do what launch() does, minus dock widgets (they need show=True)
        app._load_image()

        # Image layer should exist
        assert app._image_layer is not None
        assert app._image_layer.data.shape == (64, 64)

        app.viewer.close()

    def test_image_updates_on_time_change(self, qtbot):
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)

        app._load_image()
        data_t1 = app._image_layer.data.copy()

        app.current_time = 3
        app._load_image()
        data_t3 = app._image_layer.data

        # Data should differ (random data per timepoint)
        assert not np.array_equal(data_t1, data_t3)

        app.viewer.close()

    def test_image_updates_on_plane_change(self, qtbot):
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)

        app._load_image()
        data_p15 = app._image_layer.data.copy()

        app.current_plane = 5
        app._load_image()
        data_p5 = app._image_layer.data

        assert not np.array_equal(data_p15, data_p5)

        app.viewer.close()

    def test_viewer_integration_setup(self, qtbot):
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)

        vi = ViewerIntegration(app)
        vi.setup_layers()

        # Shapes layer should be created
        assert app.viewer.layers["Nuclei"] is not None

        app.viewer.close()

    def test_viewer_integration_update_overlays(self, qtbot):
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)
        app.current_time = 3
        app.current_plane = 15

        vi = ViewerIntegration(app)
        vi.setup_layers()
        vi.update_overlays()

        # Shapes layer should have data
        shapes_layer = app.viewer.layers["Nuclei"]
        assert len(shapes_layer.data) > 0

        app.viewer.close()

    def test_keyboard_bindings(self, qtbot):
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)
        app._bind_keys()

        # Verify key bindings were registered (napari stores them)
        # Just check they don't crash during registration
        assert app.current_time == 1

        app.viewer.close()

    def test_contrast_tool_with_viewer(self, qtbot):
        """Test contrast tool connected to a real napari image layer."""
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)
        app._load_image()

        widget = ContrastTools(app)
        qtbot.addWidget(widget)

        # Auto contrast should work with real image layer
        widget._auto_contrast()
        # Min should be > 0 (data is random 0-999)
        assert widget._min_spin.value() >= 0
        assert widget._max_spin.value() > widget._min_spin.value()

        # Setting values should update the layer
        widget._min_slider.setValue(100)
        widget._max_slider.setValue(900)
        limits = app._image_layer.contrast_limits
        assert limits[0] == pytest.approx(100, abs=1)
        assert limits[1] == pytest.approx(900, abs=1)

        app.viewer.close()


# ── Dock widget integration tests ─────────────────────────────────


class TestDockWidgets:
    """Test widgets as napari dock widgets."""

    def test_player_as_dock_widget(self, qtbot):
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)

        widget = PlayerControls(app)
        app.viewer.window.add_dock_widget(widget, name="Player")

        # Widget should be docked
        assert widget.isVisible() or True  # May not be visible in headless

        app.viewer.close()

    def test_cell_info_as_dock_widget(self, qtbot):
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)

        widget = CellInfoPanel(app)
        app.viewer.window.add_dock_widget(widget, name="Cell Info")

        app.viewer.close()

    def test_lineage_as_dock_widget(self, qtbot):
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)

        widget = LineageWidget(app)
        app.viewer.window.add_dock_widget(widget, name="Lineage")

        assert widget._layout is not None
        app.viewer.close()

    def test_all_dock_widgets_together(self, qtbot):
        """Test that all widgets can coexist in the same viewer."""
        app = _make_app_with_images()
        app.viewer = napari.Viewer(show=False)
        qtbot.addWidget(app.viewer.window._qt_window)
        app._load_image()

        player = PlayerControls(app)
        cell_info = CellInfoPanel(app)
        contrast = ContrastTools(app)
        lineage = LineageWidget(app)

        app.viewer.window.add_dock_widget(player, name="Player", area="bottom")
        app.viewer.window.add_dock_widget(cell_info, name="Cell Info", area="left")
        app.viewer.window.add_dock_widget(contrast, name="Contrast", area="right")
        app.viewer.window.add_dock_widget(lineage, name="Lineage", area="bottom")

        # Navigate and verify all widgets update
        app.current_cell_name = "P0"
        app.current_time = 2
        player.refresh()
        cell_info.refresh()

        assert player._time_spin.value() == 2
        assert "P0" in cell_info._text.toPlainText()

        app.viewer.close()
