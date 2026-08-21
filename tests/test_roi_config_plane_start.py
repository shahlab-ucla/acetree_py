from types import SimpleNamespace

import numpy as np

from acetree_py.io.config import AceTreeConfig, load_config
from acetree_py.io.config_writer import write_config_xml
from acetree_py.core.nuclei_manager import NucleiManager
from acetree_py.gui.app import AceTreeApp
from acetree_py.io import image_provider as image_provider_module


def test_plane_start_round_trips_through_resolution_element(tmp_path):
    path = tmp_path / "embryo.xml"
    config = AceTreeConfig(
        config_file=path,
        xy_res=0.11,
        z_res=0.75,
        plane_start=7,
        plane_end=41,
    )

    write_config_xml(config, path)
    reopened = load_config(path)

    assert reopened.plane_start == 7
    assert reopened.plane_end == 41
    assert 'planeStart="7"' in path.read_text(encoding="utf-8")


def test_missing_plane_start_retains_legacy_default(tmp_path):
    path = tmp_path / "legacy.xml"
    path.write_text(
        '<embryo><resolution xyRes="0.1" zRes="1" planeEnd="9"/></embryo>',
        encoding="utf-8",
    )

    assert load_config(path).plane_start == 1


def test_movie_plane_count_uses_inclusive_absolute_bounds():
    manager = NucleiManager.new_empty(
        AceTreeConfig(plane_start=7, plane_end=11),
        num_timepoints=2,
    )

    assert manager.movie.num_planes == 5


def test_image_provider_factory_uses_plane_count_not_absolute_end(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "embryo_t001.tif"
    image_path.write_bytes(b"placeholder")
    captured = {}

    class Provider:
        num_planes = 5
        num_timepoints = 1
        num_channels = 1
        image_shape = (8, 8)

    def make_provider(
        image_path, tif_dir, prefix, filename, stem, num_planes, config
    ):
        captured["num_planes"] = num_planes
        return Provider()

    monkeypatch.setattr(image_provider_module, "_create_base_provider", make_provider)
    config = AceTreeConfig(
        image_file=image_path,
        tif_directory=tmp_path,
        plane_start=7,
        plane_end=11,
        split=0,
        flip=0,
    )

    provider = image_provider_module.create_image_provider_from_config(config)

    assert provider is not None
    assert captured["num_planes"] == 5


def test_app_translates_absolute_plane_to_provider_local_index():
    config = AceTreeConfig(plane_start=7, plane_end=11, split=0, flip=0)
    manager = NucleiManager.new_empty(config, num_timepoints=1)

    class Provider:
        num_planes = 5
        num_timepoints = 1
        num_channels = 1
        image_shape = (8, 8)

        def __init__(self):
            self.requested = []

        def get_plane(self, time, plane, channel=0):
            self.requested.append((time, plane, channel))
            return np.zeros((8, 8), dtype=np.uint8)

    provider = Provider()
    app = AceTreeApp(manager, provider)
    app.viewer = SimpleNamespace(
        add_image=lambda data, **kwargs: SimpleNamespace(data=data)
    )
    app.current_time = 1
    app.current_plane = 8

    app._load_image()

    assert provider.requested == [(1, 2, 0)]
