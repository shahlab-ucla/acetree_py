"""Tests for SplitChannelProvider, MultiChannelFolderProvider, and config auto-detection.

Tests cover:
- SplitChannelProvider: split-only, flip-only, split+flip, passthrough
- MultiChannelFolderProvider: multi-channel from separate files
- create_image_provider_from_config: auto-detection with split/flip flags
- Config parsing: flip and split tags from XML
"""

from __future__ import annotations

import numpy as np
import pytest

from acetree_py.io.image_provider import (
    NumpyProvider,
    SplitChannelProvider,
    MultiChannelFolderProvider,
    StackTiffProvider,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_dual_channel_image(height=64, half_width=32, num_planes=4, num_times=3):
    """Create a synthetic image with two channels side-by-side.

    Left half: green channel (values 100-200)
    Right half: red channel (values 10-20)
    Full width = 2 * half_width
    """
    full_width = half_width * 2
    data = np.zeros((num_times, num_planes, height, full_width), dtype=np.uint16)

    for t in range(num_times):
        for z in range(num_planes):
            # Left half: green channel - distinctive pattern
            data[t, z, :, :half_width] = 100 + t * 10 + z
            # Right half: red channel - different pattern
            data[t, z, :, half_width:] = 10 + t + z

    return data


# ── SplitChannelProvider tests ───────────────────────────────────


class TestSplitChannelProvider:

    def test_split_only_green_channel(self):
        """Split=1, Flip=0: left half is green (ch0)."""
        data = _make_dual_channel_image()
        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=True, flip=False)

        plane = provider.get_plane(1, 1, channel=0)
        assert plane.shape == (64, 32)
        # Left half of t=0, z=0 has value 100
        assert plane[0, 0] == 100

    def test_split_only_red_channel(self):
        """Split=1, Flip=0: right half is red (ch1)."""
        data = _make_dual_channel_image()
        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=True, flip=False)

        plane = provider.get_plane(1, 1, channel=1)
        assert plane.shape == (64, 32)
        # Right half of t=0, z=0 has value 10
        assert plane[0, 0] == 10

    def test_split_image_shape(self):
        """image_shape should return half width when split."""
        data = _make_dual_channel_image(height=64, half_width=32)
        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=True, flip=False)

        assert provider.image_shape == (64, 32)

    def test_split_num_channels(self):
        """num_channels should be 2 when split."""
        data = _make_dual_channel_image()
        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=True, flip=False)
        assert provider.num_channels == 2

    def test_flip_only(self):
        """Flip=1, Split=0: image is horizontally flipped."""
        data = np.zeros((2, 3, 64, 100), dtype=np.uint16)
        # Put a marker at the left edge
        data[:, :, :, 0] = 999
        data[:, :, :, -1] = 111

        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=False, flip=True)

        plane = provider.get_plane(1, 1, 0)
        assert plane.shape == (64, 100)  # No split, full width
        # After flip, left edge should now have the right edge's value
        assert plane[0, 0] == 111
        assert plane[0, -1] == 999

    def test_split_and_flip(self):
        """Split=1, Flip=1: halves are swapped AND each half is flipped."""
        data = np.zeros((2, 3, 10, 20), dtype=np.uint16)
        # Left half (cols 0-9): values 100
        data[:, :, :, :10] = 100
        # Right half (cols 10-19): values 200
        data[:, :, :, 10:] = 200

        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=True, flip=True)

        # With flip+split: right half -> green (ch0), left half -> red (ch1)
        green = provider.get_plane(1, 1, channel=0)
        red = provider.get_plane(1, 1, channel=1)
        assert green.shape == (10, 10)
        assert red.shape == (10, 10)
        # Green should be from right half (value 200), flipped
        assert green[0, 0] == 200
        # Red should be from left half (value 100), flipped
        assert red[0, 0] == 100

    def test_no_split_no_flip_passthrough(self):
        """Split=0, Flip=0: passthrough, no transformation."""
        data = np.ones((2, 3, 64, 100), dtype=np.uint16) * 42
        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=False, flip=False)

        plane = provider.get_plane(1, 1, 0)
        assert plane.shape == (64, 100)
        assert plane[0, 0] == 42

    def test_get_stack_split(self):
        """get_stack should also split correctly."""
        data = _make_dual_channel_image(num_planes=5)
        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=True, flip=False)

        stack = provider.get_stack(1, channel=0)
        assert stack.shape == (5, 64, 32)

        stack_red = provider.get_stack(1, channel=1)
        assert stack_red.shape == (5, 64, 32)

    def test_num_planes_delegates(self):
        data = _make_dual_channel_image(num_planes=7)
        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=True, flip=False)
        assert provider.num_planes == 7

    def test_num_timepoints_delegates(self):
        data = _make_dual_channel_image(num_times=5)
        inner = NumpyProvider(data)
        provider = SplitChannelProvider(inner, split=True, flip=False)
        assert provider.num_timepoints == 5


# ── MultiChannelFolderProvider tests ─────────────────────────────


class TestMultiChannelFolderProvider:

    def test_two_channels(self):
        """Load separate channels from separate providers."""
        ch0_data = np.full((3, 4, 64, 100), 100, dtype=np.uint16)
        ch1_data = np.full((3, 4, 64, 100), 200, dtype=np.uint16)

        provider = MultiChannelFolderProvider([
            NumpyProvider(ch0_data),
            NumpyProvider(ch1_data),
        ])

        assert provider.num_channels == 2
        plane0 = provider.get_plane(1, 1, channel=0)
        plane1 = provider.get_plane(1, 1, channel=1)
        assert plane0[0, 0] == 100
        assert plane1[0, 0] == 200

    def test_three_channels(self):
        ch0 = np.full((2, 3, 32, 32), 10, dtype=np.uint16)
        ch1 = np.full((2, 3, 32, 32), 20, dtype=np.uint16)
        ch2 = np.full((2, 3, 32, 32), 30, dtype=np.uint16)

        provider = MultiChannelFolderProvider([
            NumpyProvider(ch0), NumpyProvider(ch1), NumpyProvider(ch2),
        ])
        assert provider.num_channels == 3
        assert provider.get_plane(1, 1, 2)[0, 0] == 30

    def test_flip(self):
        """Flip should apply to all channels."""
        data = np.zeros((2, 3, 10, 20), dtype=np.uint16)
        data[:, :, :, 0] = 999
        data[:, :, :, -1] = 111

        provider = MultiChannelFolderProvider([NumpyProvider(data)], flip=True)
        plane = provider.get_plane(1, 1, 0)
        assert plane[0, 0] == 111
        assert plane[0, -1] == 999

    def test_channel_out_of_range(self):
        data = np.ones((2, 3, 10, 10), dtype=np.uint16)
        provider = MultiChannelFolderProvider([NumpyProvider(data)])
        with pytest.raises(IndexError):
            provider.get_plane(1, 1, channel=1)

    def test_properties_delegate_to_first_channel(self):
        ch0 = np.ones((5, 8, 64, 128), dtype=np.uint16)
        ch1 = np.ones((5, 8, 64, 128), dtype=np.uint16)
        provider = MultiChannelFolderProvider([
            NumpyProvider(ch0), NumpyProvider(ch1),
        ])
        assert provider.num_timepoints == 5
        assert provider.num_planes == 8
        assert provider.image_shape == (64, 128)


# ── Config parsing tests ────────────────────────────────────────


class TestConfigParsing:

    def test_flip_parsed_from_xml(self, tmp_path):
        """Flip tag should be parsed from XML config."""
        xml_content = """<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="test_t1.tif"/>
    <Flip FlipMode="0"/>
    <Split SplitMode="1"/>
</embryo>"""
        config_file = tmp_path / "test.xml"
        config_file.write_text(xml_content)

        from acetree_py.io.config import load_config
        config = load_config(config_file)
        assert config.flip == 0
        assert config.split == 1

    def test_flip_default(self, tmp_path):
        """Flip defaults to 1 (enabled) when not specified."""
        xml_content = """<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="test_t1.tif"/>
</embryo>"""
        config_file = tmp_path / "test.xml"
        config_file.write_text(xml_content)

        from acetree_py.io.config import load_config
        config = load_config(config_file)
        assert config.flip == 1  # Default

    def test_derive_image_params(self, tmp_path):
        """tif_prefix and tif_directory should be derived from image_file."""
        # Create a dummy image file so resolve doesn't fail
        img = tmp_path / "SPIMA_t1.tif"
        img.write_bytes(b"")

        xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="{img}"/>
</embryo>"""
        config_file = tmp_path / "test.xml"
        config_file.write_text(xml_content)

        from acetree_py.io.config import load_config
        config = load_config(config_file)
        assert config.tif_prefix == "SPIMA_t"
        assert config.tif_directory == tmp_path


# ── Auto-detection integration test ─────────────────────────────


class TestCreateProviderFromConfig:

    def test_stack_with_split(self, tmp_path):
        """Stack TIFFs with split=1 should create SplitChannelProvider."""
        import tifffile

        # Create a dual-channel side-by-side TIFF
        img = np.zeros((4, 64, 100), dtype=np.uint16)
        img[:, :, :50] = 150  # Left half (green)
        img[:, :, 50:] = 25   # Right half (red)
        tifffile.imwrite(str(tmp_path / "test_t1.tif"), img)

        xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="{tmp_path / 'test_t1.tif'}"/>
    <Split SplitMode="1"/>
    <Flip FlipMode="0"/>
    <resolution planeEnd="4"/>
</embryo>"""
        config_file = tmp_path / "test.xml"
        config_file.write_text(xml_content)

        from acetree_py.io.config import load_config
        from acetree_py.io.image_provider import create_image_provider_from_config

        config = load_config(config_file)
        provider = create_image_provider_from_config(config)

        assert isinstance(provider, SplitChannelProvider)
        assert provider.num_channels == 2
        assert provider.image_shape == (64, 50)

        green = provider.get_plane(1, 1, channel=0)
        red = provider.get_plane(1, 1, channel=1)
        assert green[0, 0] == 150
        assert red[0, 0] == 25

    def test_stack_no_split_no_flip(self, tmp_path):
        """Stack TIFFs with split=0, flip=0 should not wrap."""
        import tifffile

        img = np.ones((4, 64, 100), dtype=np.uint16) * 42
        tifffile.imwrite(str(tmp_path / "img_t1.tif"), img)

        xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="{tmp_path / 'img_t1.tif'}"/>
    <Split SplitMode="0"/>
    <Flip FlipMode="0"/>
</embryo>"""
        config_file = tmp_path / "test.xml"
        config_file.write_text(xml_content)

        from acetree_py.io.config import load_config
        from acetree_py.io.image_provider import create_image_provider_from_config

        config = load_config(config_file)
        provider = create_image_provider_from_config(config)

        # Should be a plain StackTiffProvider (no wrapper)
        assert isinstance(provider, StackTiffProvider)
        plane = provider.get_plane(1, 1)
        assert plane[0, 0] == 42

    def test_stack_with_flip_only(self, tmp_path):
        """Stack TIFFs with flip=1 should create SplitChannelProvider(flip=True)."""
        import tifffile

        img = np.zeros((4, 64, 100), dtype=np.uint16)
        img[:, :, 0] = 999  # Marker at left edge
        tifffile.imwrite(str(tmp_path / "test_t1.tif"), img)

        xml_content = f"""<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="test.zip"/>
    <image file="{tmp_path / 'test_t1.tif'}"/>
    <Split SplitMode="0"/>
    <Flip FlipMode="1"/>
</embryo>"""
        config_file = tmp_path / "test.xml"
        config_file.write_text(xml_content)

        from acetree_py.io.config import load_config
        from acetree_py.io.image_provider import create_image_provider_from_config

        config = load_config(config_file)
        provider = create_image_provider_from_config(config)

        assert isinstance(provider, SplitChannelProvider)
        plane = provider.get_plane(1, 1, 0)
        # After flip, the left-edge marker should be at the right edge
        assert plane[0, -1] == 999
        assert plane[0, 0] == 0
