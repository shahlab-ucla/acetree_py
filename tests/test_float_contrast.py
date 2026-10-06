"""Floating TIFF display values survive both contrast panels unchanged."""
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

pytest.importorskip('qtpy.QtWidgets')
napari = pytest.importorskip('napari')

from acetree_py.gui.app import AceTreeApp
from acetree_py.gui.contrast_tools import ContrastTools
from acetree_py.gui.viewer_3d_window import Viewer3DWindow
from acetree_py.io.image_provider import StackTiffProvider
from tests.test_gui_widgets import _build_test_manager


@pytest.mark.parametrize('low,high', [(0.0, 1.0), (-0.005, 0.015), (70000.0, 120000.0)])
def test_float_tiff_contrast_in_main_and_detached_viewer(qtbot, tmp_path, low, high):
    original = np.linspace(low, high, 3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)
    tifffile.imwrite(tmp_path / 't001.tif', original, photometric='minisblack')
    provider = StackTiffProvider(tmp_path)
    data = provider.get_stack(1)
    np.testing.assert_array_equal(data, original)
    assert data.dtype == np.float32
    app = AceTreeApp(_build_test_manager(), image_provider=provider)
    layer = napari.layers.Image(data, colormap='magenta')
    app._image_layers = [layer]
    main = ContrastTools(app)
    qtbot.addWidget(main)
    main.refresh()
    detached = Viewer3DWindow(app)
    qtbot.addWidget(detached)
    detached_layer = napari.layers.Image(data, colormap='magenta', contrast_limits=layer.contrast_limits)
    detached._image_layers = [detached_layer]
    detached._rebuild_channel_controls()
    for panel, target in ((main, layer), (detached._contrast_tools, detached_layer)):
        controls = panel._channel_ctrls[0]
        assert (controls.min_spin.value(), controls.max_spin.value()) == pytest.approx(target.contrast_limits)
        panel._auto_all()
        assert target.contrast_limits == pytest.approx(np.percentile(original, [1, 99]))
        lo, hi = low + (high-low)*0.2, low + (high-low)*0.8
        controls.min_spin.setValue(lo)
        controls.max_spin.setValue(hi)
        assert target.contrast_limits == pytest.approx((lo, hi))
        controls.min_slider.setValue(16384)
        assert target.contrast_limits[0] == pytest.approx(low + (high-low)*16384/65535)
        target.contrast_limits = (lo, hi)
        assert (controls.min_spin.value(), controls.max_spin.value()) == pytest.approx((lo, hi))
        panel._reset_all()
        assert target.contrast_limits == pytest.approx((low, high))
        assert target.colormap.name == 'magenta'
        np.testing.assert_array_equal(target.data, original)


def test_float_contrast_nonfinite_constant_and_new_frame(qtbot):
    data = np.array([[np.nan, np.inf, -np.inf, -0.25, 0.125, 0.75]], dtype=np.float32)
    layer = napari.layers.Image(data, contrast_limits=(-0.25, 0.75))
    app = SimpleNamespace(_image_layers=[layer])
    panel = ContrastTools(app)
    qtbot.addWidget(panel)
    panel.refresh()
    panel._auto_all()
    assert layer.contrast_limits == pytest.approx(np.percentile(data[np.isfinite(data)], [1, 99]))
    layer.data = np.full((2, 3), 0.125, dtype=np.float32)
    panel.refresh()
    panel._auto_all()
    lo, hi = layer.contrast_limits
    assert lo < 0.125 < hi
    layer.data = np.array([[-4.5, 8.25]], dtype=np.float32)
    panel.refresh()
    panel._reset_all()
    assert layer.contrast_limits == pytest.approx((-4.5, 8.25))
    layer.data = np.full((2, 3), np.nan, dtype=np.float32)
    panel.refresh()
    panel._auto_all()
    panel._reset_all()
    assert layer.contrast_limits == pytest.approx((0, 1))
