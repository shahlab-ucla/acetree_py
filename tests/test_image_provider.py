"""Tests for acetree_py.io.image_provider — image loading implementations."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from acetree_py.io.image_provider import (
    ImageProvider,
    NumpyProvider,
    OmeTiffProvider,
    StackTiffProvider,
    TiffDirectoryProvider,
    ZipTiffProvider,
)


# ── NumpyProvider tests ─────────────────────────────────────────


class TestNumpyProvider4D:
    """Test NumpyProvider with 4D (T, Z, Y, X) data."""

    @pytest.fixture
    def provider(self) -> NumpyProvider:
        # 3 timepoints, 5 planes, 64x48 images
        data = np.random.randint(0, 255, (3, 5, 64, 48), dtype=np.uint16)
        return NumpyProvider(data)

    def test_protocol_conformance(self, provider: NumpyProvider):
        assert isinstance(provider, ImageProvider)

    def test_num_timepoints(self, provider: NumpyProvider):
        assert provider.num_timepoints == 3

    def test_num_planes(self, provider: NumpyProvider):
        assert provider.num_planes == 5

    def test_num_channels(self, provider: NumpyProvider):
        assert provider.num_channels == 1

    def test_image_shape(self, provider: NumpyProvider):
        assert provider.image_shape == (64, 48)

    def test_get_plane(self, provider: NumpyProvider):
        plane = provider.get_plane(1, 1)
        assert plane.shape == (64, 48)

    def test_get_plane_values(self):
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        prov = NumpyProvider(data)
        # time=1, plane=2 -> index (0, 1)
        plane = prov.get_plane(1, 2)
        np.testing.assert_array_equal(plane, data[0, 1])

    def test_get_stack(self, provider: NumpyProvider):
        stack = provider.get_stack(2)
        assert stack.shape == (5, 64, 48)

    def test_get_stack_values(self):
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        prov = NumpyProvider(data)
        stack = prov.get_stack(2)
        np.testing.assert_array_equal(stack, data[1])


class TestNumpyProvider5D:
    """Test NumpyProvider with 5D (T, C, Z, Y, X) data."""

    @pytest.fixture
    def provider(self) -> NumpyProvider:
        # 2 timepoints, 3 channels, 4 planes, 32x24 images
        data = np.random.randint(0, 255, (2, 3, 4, 32, 24), dtype=np.uint16)
        return NumpyProvider(data)

    def test_num_channels(self, provider: NumpyProvider):
        assert provider.num_channels == 3

    def test_num_planes(self, provider: NumpyProvider):
        assert provider.num_planes == 4

    def test_image_shape(self, provider: NumpyProvider):
        assert provider.image_shape == (32, 24)

    def test_get_plane_channel(self):
        data = np.arange(2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 2, 3, 4, 5)
        prov = NumpyProvider(data)
        plane = prov.get_plane(1, 2, channel=1)
        np.testing.assert_array_equal(plane, data[0, 1, 1])

    def test_get_stack_channel(self):
        data = np.arange(2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 2, 3, 4, 5)
        prov = NumpyProvider(data)
        stack = prov.get_stack(2, channel=1)
        np.testing.assert_array_equal(stack, data[1, 1])


class TestNumpyProviderErrors:
    """Test NumpyProvider error handling."""

    def test_reject_3d(self):
        with pytest.raises(ValueError, match="4D or 5D"):
            NumpyProvider(np.zeros((3, 4, 5)))

    def test_reject_2d(self):
        with pytest.raises(ValueError, match="4D or 5D"):
            NumpyProvider(np.zeros((4, 5)))

    def test_out_of_range_raises(self):
        prov = NumpyProvider(np.zeros((2, 3, 4, 5)))
        with pytest.raises(IndexError):
            prov.get_plane(10, 1)


# ── ZipTiffProvider tests ──────────────────────────────────────


def _write_tiff_bytes(img: np.ndarray) -> bytes:
    """Write a numpy array to TIFF bytes."""
    import tifffile
    buf = io.BytesIO()
    tifffile.imwrite(buf, img)
    return buf.getvalue()


def _create_zip_tiff_dataset(tmp_path: Path, prefix: str = "", num_times: int = 3, num_planes: int = 5) -> Path:
    """Create a synthetic ZIP-TIFF dataset for testing."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for t in range(1, num_times + 1):
        for p in range(1, num_planes + 1):
            img = np.full((32, 24), t * 100 + p, dtype=np.uint16)
            tiff_data = _write_tiff_bytes(img)

            zip_path = img_dir / f"{prefix}t{t:03d}-p{p:02d}.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(f"img_t{t:03d}_p{p:02d}.tif", tiff_data)

    return img_dir


@pytest.fixture
def zip_tiff_dataset(tmp_path: Path) -> Path:
    return _create_zip_tiff_dataset(tmp_path)


class TestZipTiffProvider:
    """Test ZipTiffProvider with synthetic ZIP-TIFF data."""

    def test_protocol_conformance(self, zip_tiff_dataset: Path):
        prov = ZipTiffProvider(zip_tiff_dataset, num_planes=5)
        assert isinstance(prov, ImageProvider)

    def test_num_timepoints(self, zip_tiff_dataset: Path):
        prov = ZipTiffProvider(zip_tiff_dataset, num_planes=5)
        assert prov.num_timepoints == 3

    def test_num_planes(self, zip_tiff_dataset: Path):
        prov = ZipTiffProvider(zip_tiff_dataset, num_planes=5)
        assert prov.num_planes == 5

    def test_get_plane(self, zip_tiff_dataset: Path):
        prov = ZipTiffProvider(zip_tiff_dataset, num_planes=5)
        plane = prov.get_plane(1, 1)
        assert plane.shape == (32, 24)
        # t=1, p=1 -> value 101
        assert plane[0, 0] == 101

    def test_get_plane_values(self, zip_tiff_dataset: Path):
        prov = ZipTiffProvider(zip_tiff_dataset, num_planes=5)
        plane = prov.get_plane(2, 3)
        assert plane[0, 0] == 203  # t=2, p=3

    def test_get_stack(self, zip_tiff_dataset: Path):
        prov = ZipTiffProvider(zip_tiff_dataset, num_planes=5)
        stack = prov.get_stack(1)
        assert stack.shape == (5, 32, 24)
        assert stack[0, 0, 0] == 101
        assert stack[4, 0, 0] == 105

    def test_image_shape_after_read(self, zip_tiff_dataset: Path):
        prov = ZipTiffProvider(zip_tiff_dataset, num_planes=5)
        prov.get_plane(1, 1)
        assert prov.image_shape == (32, 24)

    def test_file_not_found(self, zip_tiff_dataset: Path):
        prov = ZipTiffProvider(zip_tiff_dataset, num_planes=5)
        with pytest.raises(FileNotFoundError):
            prov.get_plane(100, 1)

    def test_with_prefix(self, tmp_path: Path):
        img_dir = _create_zip_tiff_dataset(tmp_path, prefix="myimg_")
        prov = ZipTiffProvider(img_dir, tif_prefix="myimg_", num_planes=5)
        assert prov.num_timepoints == 3
        plane = prov.get_plane(1, 1)
        assert plane[0, 0] == 101


class TestZipTiffProviderLoose:
    """Test ZipTiffProvider in loose-TIFF mode (use_zip=1)."""

    @pytest.fixture
    def loose_tiff_dir(self, tmp_path: Path) -> Path:
        import tifffile

        img_dir = tmp_path / "loose"
        img_dir.mkdir()
        for t in range(1, 3):
            for p in range(1, 4):
                img = np.full((16, 16), t * 100 + p, dtype=np.uint16)
                tifffile.imwrite(str(img_dir / f"t{t:03d}-p{p:02d}.tif"), img)
        return img_dir

    def test_get_plane(self, loose_tiff_dir: Path):
        prov = ZipTiffProvider(loose_tiff_dir, num_planes=3, use_zip=1)
        plane = prov.get_plane(1, 2)
        assert plane.shape == (16, 16)
        assert plane[0, 0] == 102

    def test_num_timepoints(self, loose_tiff_dir: Path):
        prov = ZipTiffProvider(loose_tiff_dir, num_planes=3, use_zip=1)
        assert prov.num_timepoints == 2


# ── TiffDirectoryProvider tests ────────────────────────────────


class TestTiffDirectoryProvider:
    """Test TiffDirectoryProvider with synthetic data."""

    @pytest.fixture
    def tiff_dir(self, tmp_path: Path) -> Path:
        import tifffile

        d = tmp_path / "tiff_dir"
        d.mkdir()
        for t in range(1, 4):
            for p in range(1, 6):
                img = np.full((20, 16), t * 100 + p, dtype=np.uint16)
                tifffile.imwrite(str(d / f"t{t:03d}_p{p:02d}.tif"), img)
        return d

    def test_protocol_conformance(self, tiff_dir: Path):
        prov = TiffDirectoryProvider(tiff_dir, num_planes=5)
        assert isinstance(prov, ImageProvider)

    def test_get_plane(self, tiff_dir: Path):
        prov = TiffDirectoryProvider(tiff_dir, num_planes=5)
        plane = prov.get_plane(2, 3)
        assert plane.shape == (20, 16)
        assert plane[0, 0] == 203

    def test_get_stack(self, tiff_dir: Path):
        prov = TiffDirectoryProvider(tiff_dir, num_planes=5)
        stack = prov.get_stack(1)
        assert stack.shape == (5, 20, 16)

    def test_num_timepoints(self, tiff_dir: Path):
        prov = TiffDirectoryProvider(tiff_dir, num_planes=5)
        assert prov.num_timepoints == 3

    def test_custom_pattern(self, tmp_path: Path):
        import tifffile

        d = tmp_path / "custom"
        d.mkdir()
        for t in range(1, 3):
            img = np.full((8, 8), t, dtype=np.uint16)
            tifffile.imwrite(str(d / f"img_t{t:04d}_z{1:03d}.tif"), img)

        prov = TiffDirectoryProvider(d, pattern="img_t{time:04d}_z{plane:03d}.tif", num_planes=1)
        plane = prov.get_plane(1, 1)
        assert plane[0, 0] == 1
        assert prov.num_timepoints == 2

    def test_file_not_found(self, tiff_dir: Path):
        prov = TiffDirectoryProvider(tiff_dir, num_planes=5)
        with pytest.raises(FileNotFoundError):
            prov.get_plane(100, 1)

    def test_image_shape(self, tiff_dir: Path):
        prov = TiffDirectoryProvider(tiff_dir, num_planes=5)
        prov.get_plane(1, 1)
        assert prov.image_shape == (20, 16)


# ── StackTiffProvider tests ────────────────────────────────────


class TestStackTiffProvider:
    """Test StackTiffProvider with multi-page TIFF stacks."""

    @pytest.fixture
    def stack_dir(self, tmp_path: Path) -> Path:
        import tifffile

        d = tmp_path / "stacks"
        d.mkdir()
        for t in range(1, 4):
            # 5 planes of 16x12 each
            stack = np.full((5, 16, 12), t, dtype=np.uint16)
            for p in range(5):
                stack[p] = t * 100 + p + 1
            tifffile.imwrite(str(d / f"t{t:03d}.tif"), stack)
        return d

    def test_protocol_conformance(self, stack_dir: Path):
        prov = StackTiffProvider(stack_dir)
        assert isinstance(prov, ImageProvider)

    def test_get_plane(self, stack_dir: Path):
        prov = StackTiffProvider(stack_dir)
        plane = prov.get_plane(2, 3)
        assert plane.shape == (16, 12)
        assert plane[0, 0] == 203

    def test_get_stack(self, stack_dir: Path):
        prov = StackTiffProvider(stack_dir)
        stack = prov.get_stack(1)
        assert stack.shape == (5, 16, 12)
        assert stack[0, 0, 0] == 101
        assert stack[4, 0, 0] == 105

    def test_num_timepoints(self, stack_dir: Path):
        prov = StackTiffProvider(stack_dir)
        assert prov.num_timepoints == 3

    def test_num_planes_after_read(self, stack_dir: Path):
        prov = StackTiffProvider(stack_dir)
        prov.get_plane(1, 1)
        assert prov.num_planes == 5

    def test_image_shape(self, stack_dir: Path):
        prov = StackTiffProvider(stack_dir)
        prov.get_plane(1, 1)
        assert prov.image_shape == (16, 12)

    def test_plane_out_of_range(self, stack_dir: Path):
        prov = StackTiffProvider(stack_dir)
        with pytest.raises(IndexError):
            prov.get_plane(1, 100)

    def test_missing_timepoint(self, stack_dir: Path):
        prov = StackTiffProvider(stack_dir)
        with pytest.raises(FileNotFoundError):
            prov.get_plane(99, 1)

    def test_single_plane_tiff(self, tmp_path: Path):
        import tifffile

        d = tmp_path / "single"
        d.mkdir()
        img = np.full((8, 8), 42, dtype=np.uint16)
        tifffile.imwrite(str(d / "t001.tif"), img)

        prov = StackTiffProvider(d)
        plane = prov.get_plane(1, 1)
        assert plane.shape == (8, 8)
        assert plane[0, 0] == 42

    def test_single_plane_rejects_plane_2(self, tmp_path: Path):
        import tifffile

        d = tmp_path / "single2"
        d.mkdir()
        img = np.full((8, 8), 42, dtype=np.uint16)
        tifffile.imwrite(str(d / "t001.tif"), img)

        prov = StackTiffProvider(d)
        with pytest.raises(IndexError):
            prov.get_plane(1, 2)

    def test_custom_pattern(self, tmp_path: Path):
        import tifffile

        d = tmp_path / "custom_stack"
        d.mkdir()
        stack = np.full((3, 10, 10), 7, dtype=np.uint16)
        tifffile.imwrite(str(d / "stack_001.tif"), stack)

        prov = StackTiffProvider(d, pattern="stack_{time:03d}.tif")
        plane = prov.get_plane(1, 2)
        assert plane.shape == (10, 10)


# ── OmeTiffProvider tests ──────────────────────────────────────


class TestOmeTiffProviderSingleFile:
    """Test OmeTiffProvider with a single multi-dimensional TIFF."""

    def test_4d_tczyx(self, tmp_path: Path):
        """Test loading a 4D (T, Z, Y, X) file."""
        import tifffile

        data = np.arange(2 * 3 * 8 * 6, dtype=np.uint16).reshape(2, 3, 8, 6)
        path = tmp_path / "test.tif"
        tifffile.imwrite(str(path), data)

        prov = OmeTiffProvider(path)
        assert prov.num_timepoints == 2
        assert prov.num_planes == 3
        assert prov.image_shape == (8, 6)

        plane = prov.get_plane(1, 2)
        np.testing.assert_array_equal(plane, data[0, 1])

    def test_3d_single_timepoint(self, tmp_path: Path):
        """Test loading a 3D (Z, Y, X) file treated as 1 timepoint."""
        import tifffile

        data = np.arange(4 * 8 * 6, dtype=np.uint16).reshape(4, 8, 6)
        path = tmp_path / "single_t.tif"
        tifffile.imwrite(str(path), data)

        prov = OmeTiffProvider(path)
        assert prov.num_timepoints == 1
        assert prov.num_planes == 4

        stack = prov.get_stack(1)
        np.testing.assert_array_equal(stack, data)

    def test_2d_single_plane(self, tmp_path: Path):
        """Test loading a 2D (Y, X) file treated as 1 timepoint, 1 plane."""
        import tifffile

        data = np.arange(8 * 6, dtype=np.uint16).reshape(8, 6)
        path = tmp_path / "flat.tif"
        tifffile.imwrite(str(path), data)

        prov = OmeTiffProvider(path)
        assert prov.num_timepoints == 1
        assert prov.num_planes == 1

        plane = prov.get_plane(1, 1)
        np.testing.assert_array_equal(plane, data)

    def test_get_stack(self, tmp_path: Path):
        import tifffile

        data = np.arange(2 * 3 * 8 * 6, dtype=np.uint16).reshape(2, 3, 8, 6)
        path = tmp_path / "stack_test.tif"
        tifffile.imwrite(str(path), data)

        prov = OmeTiffProvider(path)
        stack = prov.get_stack(2)
        np.testing.assert_array_equal(stack, data[1])

    def test_protocol_conformance(self, tmp_path: Path):
        import tifffile

        data = np.zeros((2, 3, 8, 6), dtype=np.uint16)
        path = tmp_path / "proto.tif"
        tifffile.imwrite(str(path), data)

        prov = OmeTiffProvider(path)
        assert isinstance(prov, ImageProvider)


class TestOmeTiffProviderDirectory:
    """Test OmeTiffProvider loading from a directory of TIFFs."""

    def test_load_directory(self, tmp_path: Path):
        import tifffile

        d = tmp_path / "ome_dir"
        d.mkdir()
        for t in range(3):
            stack = np.full((4, 10, 8), t + 1, dtype=np.uint16)
            tifffile.imwrite(str(d / f"t{t:03d}.tif"), stack)

        prov = OmeTiffProvider(d)
        assert prov.num_timepoints == 3
        assert prov.num_planes == 4

        plane = prov.get_plane(2, 1)
        assert plane[0, 0] == 2

    def test_empty_directory(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            prov = OmeTiffProvider(d)
            prov.get_plane(1, 1)  # Trigger lazy load


class TestOmeTiffProviderErrors:
    """Test OmeTiffProvider error handling."""

    def test_nonexistent_path(self, tmp_path: Path):
        prov = OmeTiffProvider(tmp_path / "does_not_exist.tif")
        with pytest.raises(FileNotFoundError):
            prov.get_plane(1, 1)


# ── Cross-provider consistency tests ───────────────────────────


class TestProviderConsistency:
    """Test that different providers return consistent results for the same data."""

    def test_numpy_vs_stack(self, tmp_path: Path):
        """NumpyProvider and StackTiffProvider should return the same values."""
        import tifffile

        data = np.random.randint(0, 1000, (2, 4, 16, 12), dtype=np.uint16)

        # Save as stack TIFFs
        d = tmp_path / "consistency"
        d.mkdir()
        for t in range(2):
            tifffile.imwrite(str(d / f"t{t + 1:03d}.tif"), data[t])

        numpy_prov = NumpyProvider(data)
        stack_prov = StackTiffProvider(d)

        for t in range(1, 3):
            for p in range(1, 5):
                np_plane = numpy_prov.get_plane(t, p)
                st_plane = stack_prov.get_plane(t, p)
                np.testing.assert_array_equal(np_plane, st_plane)

    def test_numpy_vs_ome(self, tmp_path: Path):
        """NumpyProvider and OmeTiffProvider should return the same values."""
        import tifffile

        data = np.random.randint(0, 1000, (2, 3, 10, 8), dtype=np.uint16)
        path = tmp_path / "consistency_ome.tif"
        tifffile.imwrite(str(path), data)

        numpy_prov = NumpyProvider(data)
        ome_prov = OmeTiffProvider(path)

        for t in range(1, 3):
            for p in range(1, 4):
                np_plane = numpy_prov.get_plane(t, p)
                ome_plane = ome_prov.get_plane(t, p)
                np.testing.assert_array_equal(np_plane, ome_plane)
