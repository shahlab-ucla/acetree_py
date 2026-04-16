"""Tests for interleaved multichannel TIFF support.

Covers:
- StackTiffProvider de-interleaves CZ and ZC page orderings.
- XML config parse and round-trip for the new attributes.
- Factory wiring: interleaved configs skip SplitChannelProvider.
- Backward compatibility: plain single-file configs still produce
  single-channel providers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acetree_py.io.config import (
    AceTreeConfig,
    NamingMethod,
    _normalize_channel_order,
    load_config,
)
from acetree_py.io.config_writer import write_config_xml
from acetree_py.io.image_provider import (
    SplitChannelProvider,
    StackTiffProvider,
    create_image_provider_from_config,
)

tifffile = pytest.importorskip("tifffile")


# ── Synthetic fixtures ────────────────────────────────────────────


def _make_interleaved_tiff(
    path: Path,
    num_planes: int,
    num_channels: int,
    order: str,
    height: int = 8,
    width: int = 6,
) -> np.ndarray:
    """Write a multi-page TIFF with a deterministic per-(z,c) sentinel.

    Each page is a uniform array with value ``z*100 + c`` so tests can
    uniquely identify which page they just read regardless of ordering.

    Returns the sentinel lookup array ``sentinel[z, c]`` (0-based indices).
    """
    sentinel = np.zeros((num_planes, num_channels), dtype=np.uint16)
    for z in range(num_planes):
        for c in range(num_channels):
            sentinel[z, c] = (z + 1) * 100 + c

    pages: list[np.ndarray] = []
    if order.upper() == "CZ":
        # Channel-fastest: Z1C1, Z1C2, Z2C1, Z2C2, ...
        for z in range(num_planes):
            for c in range(num_channels):
                pages.append(np.full((height, width), sentinel[z, c], dtype=np.uint16))
    elif order.upper() == "ZC":
        # Planar: Z1C1..ZnC1, Z1C2..ZnC2
        for c in range(num_channels):
            for z in range(num_planes):
                pages.append(np.full((height, width), sentinel[z, c], dtype=np.uint16))
    else:
        raise ValueError(f"Unknown order: {order}")

    stack = np.stack(pages)
    tifffile.imwrite(str(path), stack, photometric="minisblack")
    return sentinel


# ── StackTiffProvider: interleaved CZ ─────────────────────────────


class TestStackTiffProviderInterleavedCZ:
    """2-channel CZ-ordered TIFF (pages: Z1C1, Z1C2, Z2C1, ...)."""

    @pytest.fixture
    def provider_and_sentinel(self, tmp_path: Path):
        path = tmp_path / "t001.tif"
        num_planes = 5
        num_channels = 2
        sentinel = _make_interleaved_tiff(path, num_planes, num_channels, "CZ")
        provider = StackTiffProvider(
            directory=tmp_path,
            pattern="t{time:03d}.tif",
            num_channels=num_channels,
            channel_order="CZ",
        )
        return provider, sentinel, num_planes, num_channels

    def test_num_planes_divides_pages_by_channels(self, provider_and_sentinel):
        provider, _, num_planes, _ = provider_and_sentinel
        # Force handle open so metadata is cached
        provider.get_plane(1, 1, 0)
        assert provider.num_planes == num_planes

    def test_num_channels(self, provider_and_sentinel):
        provider, _, _, num_channels = provider_and_sentinel
        assert provider.num_channels == num_channels

    def test_get_plane_channel0(self, provider_and_sentinel):
        provider, sentinel, num_planes, _ = provider_and_sentinel
        for z in range(num_planes):
            page = provider.get_plane(time=1, plane=z + 1, channel=0)
            assert page[0, 0] == sentinel[z, 0], f"plane {z + 1} ch 0 mismatch"

    def test_get_plane_channel1(self, provider_and_sentinel):
        provider, sentinel, num_planes, _ = provider_and_sentinel
        for z in range(num_planes):
            page = provider.get_plane(time=1, plane=z + 1, channel=1)
            assert page[0, 0] == sentinel[z, 1], f"plane {z + 1} ch 1 mismatch"

    def test_get_stack_channel0(self, provider_and_sentinel):
        provider, sentinel, num_planes, _ = provider_and_sentinel
        stack = provider.get_stack(time=1, channel=0)
        assert stack.shape[0] == num_planes
        assert stack.ndim == 3
        for z in range(num_planes):
            assert stack[z, 0, 0] == sentinel[z, 0]

    def test_get_stack_channel1(self, provider_and_sentinel):
        provider, sentinel, num_planes, _ = provider_and_sentinel
        stack = provider.get_stack(time=1, channel=1)
        assert stack.shape[0] == num_planes
        for z in range(num_planes):
            assert stack[z, 0, 0] == sentinel[z, 1]

    def test_get_plane_out_of_range_channel_raises(self, provider_and_sentinel):
        provider, _, _, num_channels = provider_and_sentinel
        with pytest.raises(IndexError):
            provider.get_plane(time=1, plane=1, channel=num_channels)

    def test_get_stack_out_of_range_channel_raises(self, provider_and_sentinel):
        provider, _, _, num_channels = provider_and_sentinel
        with pytest.raises(IndexError):
            provider.get_stack(time=1, channel=num_channels)


# ── StackTiffProvider: interleaved ZC (planar) ────────────────────


class TestStackTiffProviderInterleavedZC:
    """2-channel ZC-ordered TIFF (pages: Z1C1..Z5C1, Z1C2..Z5C2)."""

    @pytest.fixture
    def provider_and_sentinel(self, tmp_path: Path):
        path = tmp_path / "t001.tif"
        num_planes = 5
        num_channels = 2
        sentinel = _make_interleaved_tiff(path, num_planes, num_channels, "ZC")
        provider = StackTiffProvider(
            directory=tmp_path,
            pattern="t{time:03d}.tif",
            num_channels=num_channels,
            channel_order="ZC",
        )
        return provider, sentinel, num_planes, num_channels

    def test_num_planes(self, provider_and_sentinel):
        provider, _, num_planes, _ = provider_and_sentinel
        provider.get_plane(1, 1, 0)
        assert provider.num_planes == num_planes

    def test_get_plane_channel0(self, provider_and_sentinel):
        provider, sentinel, num_planes, _ = provider_and_sentinel
        for z in range(num_planes):
            page = provider.get_plane(time=1, plane=z + 1, channel=0)
            assert page[0, 0] == sentinel[z, 0]

    def test_get_plane_channel1(self, provider_and_sentinel):
        provider, sentinel, num_planes, _ = provider_and_sentinel
        for z in range(num_planes):
            page = provider.get_plane(time=1, plane=z + 1, channel=1)
            assert page[0, 0] == sentinel[z, 1]

    def test_get_stack_round_trip(self, provider_and_sentinel):
        provider, sentinel, num_planes, num_channels = provider_and_sentinel
        for c in range(num_channels):
            stack = provider.get_stack(time=1, channel=c)
            assert stack.shape[0] == num_planes
            for z in range(num_planes):
                assert stack[z, 0, 0] == sentinel[z, c]


# ── StackTiffProvider: 3-channel CZ ───────────────────────────────


class TestStackTiffProvider3Channel:
    """Verify generalization beyond 2 channels."""

    def test_3_channel_cz(self, tmp_path: Path):
        path = tmp_path / "t001.tif"
        num_planes = 4
        num_channels = 3
        sentinel = _make_interleaved_tiff(path, num_planes, num_channels, "CZ")
        provider = StackTiffProvider(
            directory=tmp_path,
            pattern="t{time:03d}.tif",
            num_channels=num_channels,
            channel_order="CZ",
        )
        provider.get_plane(1, 1, 0)
        assert provider.num_planes == num_planes
        assert provider.num_channels == 3
        for c in range(num_channels):
            stack = provider.get_stack(time=1, channel=c)
            for z in range(num_planes):
                assert stack[z, 0, 0] == sentinel[z, c]


# ── StackTiffProvider: backward compatibility (single channel) ────


class TestStackTiffProviderSingleChannel:
    """The default num_channels=1 path must keep working unchanged."""

    def test_single_channel_reads_every_page_as_plane(self, tmp_path: Path):
        path = tmp_path / "t001.tif"
        num_pages = 6
        pages = np.stack([
            np.full((4, 5), (z + 1) * 10, dtype=np.uint16)
            for z in range(num_pages)
        ])
        tifffile.imwrite(str(path), pages, photometric="minisblack")

        provider = StackTiffProvider(
            directory=tmp_path,
            pattern="t{time:03d}.tif",
        )
        stack = provider.get_stack(time=1, channel=0)
        assert stack.shape[0] == num_pages
        for z in range(num_pages):
            assert stack[z, 0, 0] == (z + 1) * 10
        assert provider.num_channels == 1


# ── Config parser ─────────────────────────────────────────────────


class TestConfigParserInterleaved:
    """_parse_xml_config populates the new fields from <image>."""

    def test_normalize_channel_order(self):
        assert _normalize_channel_order("CZ") == "CZ"
        assert _normalize_channel_order("cz") == "CZ"
        assert _normalize_channel_order("interleaved") == "CZ"
        assert _normalize_channel_order("ZC") == "ZC"
        assert _normalize_channel_order("planar") == "ZC"
        assert _normalize_channel_order("garbage") == "CZ"  # warn + fallback
        assert _normalize_channel_order("") == "CZ"

    def test_parse_interleaved_cz(self, tmp_path: Path):
        cfg = tmp_path / "test.xml"
        cfg.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="img.tif" numChannels="2" channelOrder="CZ"/>
</embryo>
""")
        config = load_config(cfg)
        assert config.stack_interleaved is True
        assert config.num_channels == 2
        assert config.stack_channel_order == "CZ"
        # Plain single-file image_file is still populated
        assert config.image_file.name == "img.tif"
        # And the multi-folder dict stays empty
        assert config.image_channels == {}

    def test_parse_interleaved_zc(self, tmp_path: Path):
        cfg = tmp_path / "test.xml"
        cfg.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="img.tif" numChannels="3" channelOrder="ZC"/>
</embryo>
""")
        config = load_config(cfg)
        assert config.stack_interleaved is True
        assert config.num_channels == 3
        assert config.stack_channel_order == "ZC"

    def test_parse_numchannels_1_ignored(self, tmp_path: Path):
        """numChannels=1 on a single-file <image> must not flip stack_interleaved."""
        cfg = tmp_path / "test.xml"
        cfg.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="img.tif" numChannels="1"/>
</embryo>
""")
        config = load_config(cfg)
        assert config.stack_interleaved is False
        # num_channels should remain at its default
        assert config.num_channels == 1

    def test_parse_legacy_single_file(self, tmp_path: Path):
        """Backward compat: plain <image file="..."/> stays non-interleaved."""
        cfg = tmp_path / "test.xml"
        cfg.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="img.tif"/>
</embryo>
""")
        config = load_config(cfg)
        assert config.stack_interleaved is False
        assert config.num_channels == 1

    def test_parse_multichannel_folder_unchanged(self, tmp_path: Path):
        """The existing numChannels+channel1+channel2 branch still works."""
        cfg = tmp_path / "test.xml"
        cfg.write_text("""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image numChannels="2" channel1="ch1/img.tif" channel2="ch2/img.tif"/>
</embryo>
""")
        config = load_config(cfg)
        assert config.stack_interleaved is False
        assert config.num_channels == 2
        assert 1 in config.image_channels
        assert 2 in config.image_channels


# ── Config round-trip ─────────────────────────────────────────────


class TestConfigRoundTripInterleaved:
    """write_config_xml -> load_config preserves interleaved flags."""

    def test_round_trip_cz(self, tmp_path: Path):
        cfg_path = tmp_path / "cfg.xml"
        src = AceTreeConfig(
            config_file=cfg_path,
            zip_file=tmp_path / "nuclei.zip",
            image_file=tmp_path / "img.tif",
            num_channels=2,
            stack_interleaved=True,
            stack_channel_order="CZ",
            naming_method=NamingMethod.NEWCANONICAL,
            xy_res=0.1,
            z_res=1.0,
            plane_end=25,
        )
        write_config_xml(src, cfg_path)

        # Sanity check: the written XML should contain the new attributes
        text = cfg_path.read_text()
        assert "numChannels=\"2\"" in text
        assert "channelOrder=\"CZ\"" in text

        loaded = load_config(cfg_path)
        assert loaded.stack_interleaved is True
        assert loaded.num_channels == 2
        assert loaded.stack_channel_order == "CZ"

    def test_round_trip_zc(self, tmp_path: Path):
        cfg_path = tmp_path / "cfg.xml"
        src = AceTreeConfig(
            config_file=cfg_path,
            zip_file=tmp_path / "nuclei.zip",
            image_file=tmp_path / "img.tif",
            num_channels=3,
            stack_interleaved=True,
            stack_channel_order="ZC",
            naming_method=NamingMethod.NEWCANONICAL,
        )
        write_config_xml(src, cfg_path)
        loaded = load_config(cfg_path)
        assert loaded.stack_interleaved is True
        assert loaded.num_channels == 3
        assert loaded.stack_channel_order == "ZC"

    def test_round_trip_non_interleaved_unchanged(self, tmp_path: Path):
        """Round-trip of a single-channel config should NOT inject new attrs."""
        cfg_path = tmp_path / "cfg.xml"
        src = AceTreeConfig(
            config_file=cfg_path,
            zip_file=tmp_path / "nuclei.zip",
            image_file=tmp_path / "img.tif",
            naming_method=NamingMethod.NEWCANONICAL,
        )
        write_config_xml(src, cfg_path)
        text = cfg_path.read_text()
        assert "channelOrder" not in text
        assert "numChannels" not in text
        loaded = load_config(cfg_path)
        assert loaded.stack_interleaved is False
        assert loaded.num_channels == 1


# ── Factory wiring ────────────────────────────────────────────────


class TestFactoryInterleaved:
    """create_image_provider_from_config routes interleaved configs correctly."""

    def _write_tiff(self, path: Path, num_planes: int, num_channels: int) -> None:
        _make_interleaved_tiff(path, num_planes, num_channels, "CZ")

    def test_factory_returns_unwrapped_stack_provider(self, tmp_path: Path):
        img = tmp_path / "t001.tif"
        self._write_tiff(img, num_planes=4, num_channels=2)

        config = AceTreeConfig(
            config_file=tmp_path / "cfg.xml",
            image_file=img,
            tif_directory=tmp_path,
            tif_prefix="t",
            num_channels=2,
            stack_interleaved=True,
            stack_channel_order="CZ",
            plane_end=4,
            # split/flip off so wrapping wouldn't happen for an unrelated reason
            split=0,
            flip=0,
        )
        provider = create_image_provider_from_config(config)
        assert provider is not None
        # Must NOT be wrapped in SplitChannelProvider
        assert not isinstance(provider, SplitChannelProvider)
        assert isinstance(provider, StackTiffProvider)
        assert provider.num_channels == 2

    def test_factory_skips_split_wrap_when_interleaved(self, tmp_path: Path):
        """Even if split=1, an interleaved config must not get wrapped."""
        img = tmp_path / "t001.tif"
        self._write_tiff(img, num_planes=4, num_channels=2)

        config = AceTreeConfig(
            config_file=tmp_path / "cfg.xml",
            image_file=img,
            tif_directory=tmp_path,
            tif_prefix="t",
            num_channels=2,
            stack_interleaved=True,
            stack_channel_order="CZ",
            plane_end=4,
            split=1,
            flip=1,
        )
        provider = create_image_provider_from_config(config)
        assert provider is not None
        assert not isinstance(provider, SplitChannelProvider)

    def test_factory_legacy_single_channel(self, tmp_path: Path):
        """A plain single-channel config still yields num_channels == 1."""
        img = tmp_path / "t001.tif"
        _make_interleaved_tiff(img, num_planes=3, num_channels=1, order="CZ")

        config = AceTreeConfig(
            config_file=tmp_path / "cfg.xml",
            image_file=img,
            tif_directory=tmp_path,
            tif_prefix="t",
            plane_end=3,
            split=0,
            flip=0,
        )
        provider = create_image_provider_from_config(config)
        assert provider is not None
        assert provider.num_channels == 1

    def test_factory_plane_count_is_pages_over_channels(self, tmp_path: Path):
        """plane_end in the provider should reflect Z, not Z*C."""
        img = tmp_path / "t001.tif"
        self._write_tiff(img, num_planes=4, num_channels=2)

        config = AceTreeConfig(
            config_file=tmp_path / "cfg.xml",
            image_file=img,
            tif_directory=tmp_path,
            tif_prefix="t",
            num_channels=2,
            stack_interleaved=True,
            stack_channel_order="CZ",
            plane_end=50,  # deliberately over-estimated
            split=0,
            flip=0,
        )
        provider = create_image_provider_from_config(config)
        assert provider is not None
        # The factory probes the file and clamps to actual planes.
        # Actual pages = 8, channels = 2, so plane count should be 4.
        provider.get_plane(1, 1, 0)  # force metadata
        assert provider.num_planes == 4
