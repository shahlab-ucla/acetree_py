"""Image provider protocol and implementations for loading microscopy images.

Defines a common interface for accessing image data regardless of the
underlying file format (TIFF-in-ZIP, loose TIFFs, OME-TIFF, multi-page stack).

Adding a new image format = implementing one class that conforms to the
ImageProvider protocol. All GUI/analysis code references only ImageProvider.

Implementations:
  - ZipTiffProvider: Backwards compat with existing AceTree ZIP datasets
  - TiffDirectoryProvider: Loose TIFF files with configurable naming pattern
  - StackTiffProvider: Multi-page TIFF stacks (one file per timepoint)
  - OmeTiffProvider: OME-TIFF with metadata-driven loading
  - NumpyProvider: In-memory provider for testing

Ported from: org.rhwlab.image.ZipImage (ZipImage.java)
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class ImageProvider(Protocol):
    """Protocol for accessing image data from any format.

    All implementations must provide these methods. Adding a new image
    format means implementing this protocol.
    """

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        """Get a single 2D plane.

        Args:
            time: 1-based timepoint.
            plane: 1-based z-plane index.
            channel: 0-based channel index.

        Returns:
            2D numpy array (height, width).
        """
        ...

    def get_stack(self, time: int, channel: int = 0) -> np.ndarray:
        """Get a full z-stack for a timepoint.

        Args:
            time: 1-based timepoint.
            channel: 0-based channel index.

        Returns:
            3D numpy array (planes, height, width).
        """
        ...

    @property
    def num_timepoints(self) -> int:
        """Number of available timepoints."""
        ...

    @property
    def num_planes(self) -> int:
        """Number of z-planes per stack."""
        ...

    @property
    def num_channels(self) -> int:
        """Number of channels."""
        ...

    @property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) of each plane."""
        ...


# ── Implementations ──────────────────────────────────────────────


class ZipTiffProvider:
    """Reads TIFF images from ZIP archives (backwards-compatible format).

    This matches the original AceTree format where each image plane is
    stored as a TIFF inside a ZIP file, with naming convention:
        {prefix}t{time:03d}-p{plane:02d}.zip

    Supports two modes via use_zip:
      - use_zip=2: Each plane in a separate ZIP (default)
      - use_zip=1: Loose TIFF files (no ZIP wrapper)
    """

    def __init__(
        self,
        tif_directory: str | Path,
        tif_prefix: str = "",
        num_planes: int = 30,
        use_zip: int = 2,
    ) -> None:
        self.tif_directory = Path(tif_directory)
        self.tif_prefix = tif_prefix
        self._num_planes = num_planes
        self.use_zip = use_zip
        self._shape: tuple[int, int] | None = None
        self._num_timepoints: int | None = None

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        """Load a single plane from a TIFF file (possibly inside a ZIP)."""
        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required for image loading: pip install tifffile")

        file_path = self._build_path(time, plane)

        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        if file_path.suffix.lower() == ".zip":
            return self._read_from_zip(file_path, tifffile)
        else:
            img = tifffile.imread(str(file_path))
            if img.ndim == 3:
                # Multi-page TIFF: select the plane
                plane_idx = plane - 1
                if plane_idx >= img.shape[0]:
                    raise IndexError(f"Plane {plane} out of range (max {img.shape[0]})")
                img = img[plane_idx]
            self._update_shape(img)
            return img

    def get_stack(self, time: int, channel: int = 0) -> np.ndarray:
        """Load all planes for a timepoint."""
        planes = []
        for p in range(1, self._num_planes + 1):
            try:
                plane = self.get_plane(time, p, channel)
                planes.append(plane)
            except (FileNotFoundError, KeyError, IndexError):
                break
        if not planes:
            raise FileNotFoundError(f"No planes found for time={time}")
        return np.stack(planes)

    @property
    def num_timepoints(self) -> int:
        if self._num_timepoints is not None:
            return self._num_timepoints
        self._num_timepoints = self._scan_timepoints()
        return self._num_timepoints

    @property
    def num_planes(self) -> int:
        return self._num_planes

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._shape or (0, 0)

    def _build_path(self, time: int, plane: int) -> Path:
        """Build the file path for a specific time/plane."""
        if self.use_zip == 2:
            filename = f"{self.tif_prefix}t{time:03d}-p{plane:02d}.zip"
        else:
            filename = f"{self.tif_prefix}t{time:03d}-p{plane:02d}.tif"
        return self.tif_directory / filename

    def _read_from_zip(self, zip_path: Path, tifffile) -> np.ndarray:
        """Read a TIFF from inside a ZIP file."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            tif_entries = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tif_entries:
                raise FileNotFoundError(f"No TIFF entries in {zip_path}")
            entry = tif_entries[0]
            data = zf.read(entry)
            img = tifffile.imread(io.BytesIO(data))
            self._update_shape(img)
            return img

    def _scan_timepoints(self) -> int:
        """Scan directory to count available timepoints."""
        if not self.tif_directory.exists():
            return 0
        ext = ".zip" if self.use_zip == 2 else ".tif"
        pattern = re.compile(re.escape(self.tif_prefix) + r"t(\d+)-p\d+" + re.escape(ext))
        times = set()
        for f in self.tif_directory.iterdir():
            m = pattern.match(f.name)
            if m:
                times.add(int(m.group(1)))
        return len(times)

    def _update_shape(self, img: np.ndarray) -> None:
        if self._shape is None and img.ndim >= 2:
            self._shape = (img.shape[-2], img.shape[-1])


class TiffDirectoryProvider:
    """Reads TIFF files from a directory tree.

    Supports flexible naming patterns via a format string. The pattern
    can use {time}, {plane}, and {channel} placeholders.

    Examples:
        "t{time:03d}_p{plane:02d}.tif"
        "img_t{time:03d}-p{plane:02d}.tif"
        "ch{channel}/t{time:04d}_z{plane:03d}.tif"
    """

    def __init__(
        self,
        directory: str | Path,
        pattern: str = "t{time:03d}_p{plane:02d}.tif",
        num_planes: int = 30,
    ) -> None:
        self.directory = Path(directory)
        self.pattern = pattern
        self._num_planes = num_planes
        self._shape: tuple[int, int] | None = None
        self._num_timepoints: int | None = None

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required: pip install tifffile")

        filename = self.pattern.format(time=time, plane=plane, channel=channel)
        path = self.directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = tifffile.imread(str(path))
        if self._shape is None and img.ndim >= 2:
            self._shape = (img.shape[-2], img.shape[-1])
        return img

    def get_stack(self, time: int, channel: int = 0) -> np.ndarray:
        planes = []
        for p in range(1, self._num_planes + 1):
            try:
                plane = self.get_plane(time, p, channel)
                planes.append(plane)
            except FileNotFoundError:
                break
        if not planes:
            raise FileNotFoundError(f"No planes found for time={time}")
        return np.stack(planes)

    @property
    def num_timepoints(self) -> int:
        if self._num_timepoints is not None:
            return self._num_timepoints
        self._num_timepoints = self._scan_timepoints()
        return self._num_timepoints

    @property
    def num_planes(self) -> int:
        return self._num_planes

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._shape or (0, 0)

    def _scan_timepoints(self) -> int:
        """Scan directory to count available timepoints."""
        if not self.directory.exists():
            return 0
        # Try plane 1 for each timepoint starting from 1
        count = 0
        for t in range(1, 10000):
            filename = self.pattern.format(time=t, plane=1, channel=0)
            if (self.directory / filename).exists():
                count += 1
            elif count > 0:
                break  # Stop after first gap
        return count


class StackTiffProvider:
    """Reads multi-page TIFF stacks (one file per timepoint).

    Each timepoint is a single TIFF file containing all z-planes as pages.
    File naming: {prefix}t{time:03d}.tif (configurable via pattern).
    """

    def __init__(
        self,
        directory: str | Path,
        pattern: str = "t{time:03d}.tif",
        num_channels: int = 1,
    ) -> None:
        self.directory = Path(directory)
        self.pattern = pattern
        self._num_channels = num_channels
        self._shape: tuple[int, int] | None = None
        self._num_planes_cached: int | None = None
        self._num_timepoints: int | None = None

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required: pip install tifffile")

        path = self._build_path(time)
        if not path.exists():
            raise FileNotFoundError(f"Stack not found: {path}")

        img = tifffile.imread(str(path))

        # Handle different dimensionalities
        # Could be (Z, Y, X), (C, Z, Y, X), (Z, C, Y, X)
        if img.ndim == 2:
            # Single plane
            if plane != 1:
                raise IndexError(f"Only 1 plane available, requested {plane}")
            self._update_shape(img)
            return img
        elif img.ndim == 3:
            # (Z, Y, X) or (C, Y, X) depending on context
            plane_idx = plane - 1
            if plane_idx >= img.shape[0]:
                raise IndexError(f"Plane {plane} out of range (max {img.shape[0]})")
            result = img[plane_idx]
            self._update_shape(result)
            self._num_planes_cached = img.shape[0]
            return result
        elif img.ndim == 4:
            # (C, Z, Y, X) or (Z, C, Y, X)
            # Assume (C, Z, Y, X) if num_channels > 1
            plane_idx = plane - 1
            if self._num_channels > 1:
                result = img[channel, plane_idx]
            else:
                result = img[plane_idx, channel] if img.shape[1] > 1 else img[plane_idx, 0]
            self._update_shape(result)
            return result
        else:
            raise ValueError(f"Unexpected image dimensions: {img.shape}")

    def get_stack(self, time: int, channel: int = 0) -> np.ndarray:
        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required: pip install tifffile")

        path = self._build_path(time)
        if not path.exists():
            raise FileNotFoundError(f"Stack not found: {path}")

        img = tifffile.imread(str(path))

        if img.ndim == 2:
            self._update_shape(img)
            return img[np.newaxis, ...]  # Add z dimension
        elif img.ndim == 3:
            self._num_planes_cached = img.shape[0]
            if img.shape[0] > 0:
                self._update_shape(img[0])
            return img
        elif img.ndim == 4 and self._num_channels > 1:
            stack = img[channel]
            self._num_planes_cached = stack.shape[0]
            if stack.shape[0] > 0:
                self._update_shape(stack[0])
            return stack
        else:
            return img

    @property
    def num_timepoints(self) -> int:
        if self._num_timepoints is not None:
            return self._num_timepoints
        count = 0
        for t in range(1, 10000):
            if self._build_path(t).exists():
                count += 1
            elif count > 0:
                break
        self._num_timepoints = count
        return count

    @property
    def num_planes(self) -> int:
        return self._num_planes_cached or 0

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._shape or (0, 0)

    def _build_path(self, time: int) -> Path:
        filename = self.pattern.format(time=time)
        return self.directory / filename

    def _update_shape(self, img: np.ndarray) -> None:
        if self._shape is None and img.ndim >= 2:
            self._shape = (img.shape[-2], img.shape[-1])


class OmeTiffProvider:
    """Reads OME-TIFF files with metadata-driven loading.

    OME-TIFF files contain metadata describing the dimensional layout
    (T, C, Z, Y, X). This provider uses tifffile's OME-TIFF support
    to load data correctly based on that metadata.

    Can handle either:
      - Single OME-TIFF containing all timepoints
      - Directory of OME-TIFF files (one per timepoint)
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize from an OME-TIFF file or directory.

        Args:
            path: Path to a single OME-TIFF file or directory of OME-TIFFs.
        """
        self.path = Path(path)
        self._data: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._n_timepoints = 0
        self._n_planes = 0
        self._n_channels = 1

    def _ensure_loaded(self) -> None:
        """Lazy-load the OME-TIFF data."""
        if self._data is not None:
            return

        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required: pip install tifffile")

        if self.path.is_file():
            self._load_single_file(tifffile)
        elif self.path.is_dir():
            self._load_directory(tifffile)
        else:
            raise FileNotFoundError(f"OME-TIFF path not found: {self.path}")

    def _load_single_file(self, tifffile) -> None:
        """Load a single OME-TIFF containing all data."""
        img = tifffile.imread(str(self.path))

        # Reshape based on detected dimensions
        # Common OME-TIFF layouts: (T,Z,Y,X), (T,C,Z,Y,X), (Z,Y,X)
        if img.ndim == 5:
            # (T, C, Z, Y, X)
            self._n_timepoints = img.shape[0]
            self._n_channels = img.shape[1]
            self._n_planes = img.shape[2]
            self._shape = (img.shape[3], img.shape[4])
        elif img.ndim == 4:
            # (T, Z, Y, X)
            self._n_timepoints = img.shape[0]
            self._n_planes = img.shape[1]
            self._shape = (img.shape[2], img.shape[3])
        elif img.ndim == 3:
            # (Z, Y, X) — single timepoint
            self._n_timepoints = 1
            self._n_planes = img.shape[0]
            self._shape = (img.shape[1], img.shape[2])
            img = img[np.newaxis, ...]  # Add T dimension -> (1, Z, Y, X)
        elif img.ndim == 2:
            # (Y, X) — single plane, single timepoint
            self._n_timepoints = 1
            self._n_planes = 1
            self._shape = (img.shape[0], img.shape[1])
            img = img[np.newaxis, np.newaxis, ...]  # -> (1, 1, Y, X)
        else:
            raise ValueError(f"Unexpected OME-TIFF dimensions: {img.shape}")

        self._data = img

    def _load_directory(self, tifffile) -> None:
        """Load a directory of OME-TIFF files (one per timepoint)."""
        files = sorted(self.path.glob("*.tif")) + sorted(self.path.glob("*.tiff"))
        if not files:
            raise FileNotFoundError(f"No TIFF files in {self.path}")

        stacks = []
        for f in files:
            img = tifffile.imread(str(f))
            if img.ndim == 2:
                img = img[np.newaxis, ...]  # Single plane -> (1, Y, X)
            stacks.append(img)

        # Stack into (T, Z, Y, X)
        self._data = np.stack(stacks)
        self._n_timepoints = len(stacks)
        self._n_planes = stacks[0].shape[0]
        self._shape = (stacks[0].shape[-2], stacks[0].shape[-1])

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        self._ensure_loaded()
        assert self._data is not None

        t_idx = time - 1
        p_idx = plane - 1

        if self._data.ndim == 5:
            return self._data[t_idx, channel, p_idx]
        elif self._data.ndim == 4:
            return self._data[t_idx, p_idx]
        else:
            raise ValueError(f"Unexpected data shape: {self._data.shape}")

    def get_stack(self, time: int, channel: int = 0) -> np.ndarray:
        self._ensure_loaded()
        assert self._data is not None

        t_idx = time - 1
        if self._data.ndim == 5:
            return self._data[t_idx, channel]
        elif self._data.ndim == 4:
            return self._data[t_idx]
        else:
            raise ValueError(f"Unexpected data shape: {self._data.shape}")

    @property
    def num_timepoints(self) -> int:
        self._ensure_loaded()
        return self._n_timepoints

    @property
    def num_planes(self) -> int:
        self._ensure_loaded()
        return self._n_planes

    @property
    def num_channels(self) -> int:
        self._ensure_loaded()
        return self._n_channels

    @property
    def image_shape(self) -> tuple[int, int]:
        self._ensure_loaded()
        return self._shape or (0, 0)


class SplitChannelProvider:
    """Wraps another provider to handle side-by-side dual-channel images.

    When split=1 in Java AceTree, a 16-bit image contains two channels
    arranged side-by-side horizontally:
    - Left half:  green channel (GFP / cell imaging)
    - Right half: red channel (expression reporter)

    This provider splits the image and optionally flips it horizontally.

    When flip=1 AND split=1, the halves are swapped:
    - Left half -> red channel, Right half -> green channel

    Channel 0 = green (primary cell imaging channel)
    Channel 1 = red (expression channel)
    """

    def __init__(
        self,
        inner: ImageProvider,
        split: bool = True,
        flip: bool = False,
    ) -> None:
        self._inner = inner
        self._split = split
        self._flip = flip

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        """Get a single plane, splitting and/or flipping as needed."""
        raw = self._inner.get_plane(time, plane, 0)  # Always load channel 0 from inner

        if self._split:
            h, w = raw.shape[-2], raw.shape[-1]
            half_w = w // 2

            if self._flip:
                # Flip=1 + Split=1: right half is green (ch0), left half is red (ch1)
                green_half = raw[..., half_w:]
                red_half = raw[..., :half_w]
                # Also flip each half horizontally
                green_half = np.ascontiguousarray(green_half[..., ::-1])
                red_half = np.ascontiguousarray(red_half[..., ::-1])
            else:
                # Split=1 only: left half is green (ch0), right half is red (ch1)
                green_half = raw[..., :half_w]
                red_half = raw[..., half_w:]

            result = green_half if channel == 0 else red_half
        elif self._flip:
            # Flip only, no split
            result = np.ascontiguousarray(raw[..., ::-1])
        else:
            result = raw

        return result

    def get_stack(self, time: int, channel: int = 0) -> np.ndarray:
        """Get a full z-stack, splitting and/or flipping as needed."""
        raw = self._inner.get_stack(time, 0)

        if self._split:
            h, w = raw.shape[-2], raw.shape[-1]
            half_w = w // 2

            if self._flip:
                green_half = raw[..., half_w:]
                red_half = raw[..., :half_w]
                green_half = np.ascontiguousarray(green_half[..., ::-1])
                red_half = np.ascontiguousarray(red_half[..., ::-1])
            else:
                green_half = raw[..., :half_w]
                red_half = raw[..., half_w:]

            return green_half if channel == 0 else red_half
        elif self._flip:
            return np.ascontiguousarray(raw[..., ::-1])
        else:
            return raw

    @property
    def num_timepoints(self) -> int:
        return self._inner.num_timepoints

    @property
    def num_planes(self) -> int:
        return self._inner.num_planes

    @property
    def num_channels(self) -> int:
        return 2 if self._split else self._inner.num_channels

    @property
    def image_shape(self) -> tuple[int, int]:
        h, w = self._inner.image_shape
        if h == 0 and w == 0:
            # Inner shape not yet known; try to probe by loading one plane
            try:
                self.get_plane(1, 1, 0)
                h, w = self._inner.image_shape
            except (FileNotFoundError, IndexError):
                pass
        if self._split and w > 0:
            return (h, w // 2)
        return (h, w)


class MultiChannelFolderProvider:
    """Loads multi-channel images from separate directories/files.

    Used when the config specifies:
        <image numChannels="2" channel1="path/to/red" channel2="path/to/green"/>

    Each channel is stored in a separate file/directory. This provider
    delegates to per-channel inner providers and composites them.
    """

    def __init__(
        self,
        channel_providers: list[ImageProvider],
        flip: bool = False,
    ) -> None:
        self._channels = channel_providers
        self._flip = flip
        if not channel_providers:
            raise ValueError("At least one channel provider required")

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        if channel >= len(self._channels):
            raise IndexError(f"Channel {channel} out of range (have {len(self._channels)})")
        raw = self._channels[channel].get_plane(time, plane, 0)
        if self._flip:
            raw = np.ascontiguousarray(raw[..., ::-1])
        return raw

    def get_stack(self, time: int, channel: int = 0) -> np.ndarray:
        if channel >= len(self._channels):
            raise IndexError(f"Channel {channel} out of range (have {len(self._channels)})")
        raw = self._channels[channel].get_stack(time, 0)
        if self._flip:
            raw = np.ascontiguousarray(raw[..., ::-1])
        return raw

    @property
    def num_timepoints(self) -> int:
        return self._channels[0].num_timepoints

    @property
    def num_planes(self) -> int:
        return self._channels[0].num_planes

    @property
    def num_channels(self) -> int:
        return len(self._channels)

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._channels[0].image_shape


def create_image_provider_from_config(config) -> ImageProvider | None:
    """Auto-detect and create an appropriate ImageProvider from an AceTreeConfig.

    Handles the following config flags:
    - **split** (SplitMode): When 1, images contain two channels side-by-side
      horizontally (left=green/GFP, right=red/expression). The provider splits
      the image and returns each half as a separate channel.
    - **flip** (FlipMode): When 1, images are horizontally flipped. When
      combined with split, affects which half is which channel.
    - **use_zip**: When 2, images are TIFFs inside ZIP archives.
    - **use_stack**: When 1, images are 16-bit (vs 8-bit for 0).
    - **image_channels**: When present, each channel is in a separate folder.

    Detection logic for base provider:
    1. Multi-channel folders (image_channels) -> MultiChannelFolderProvider
    2. Filename contains '-p' + use_zip=2 -> ZipTiffProvider
    3. Filename contains '-p' -> TiffDirectoryProvider
    4. Otherwise -> StackTiffProvider (multi-page TIFF per timepoint)

    Then wraps with SplitChannelProvider if split=1 or flip=1.

    Args:
        config: An AceTreeConfig instance.

    Returns:
        An ImageProvider, or None if image files can't be found.
    """
    # ── Multi-channel from separate folders ──────────────────────
    if config.image_channels and len(config.image_channels) > 1:
        return _create_multi_channel_provider(config)

    # ── Single image_file based provider ─────────────────────────
    image_file = config.image_file
    if not image_file or not str(image_file) or str(image_file) == ".":
        logger.info("No image file configured, skipping image provider")
        return None

    image_path = Path(image_file)
    if not image_path.exists():
        logger.warning("Image file does not exist: %s", image_path)
        return None

    tif_dir = config.tif_directory if config.tif_directory != Path() else image_path.parent
    prefix = config.tif_prefix
    filename = image_path.name
    stem = image_path.stem

    # Determine number of planes from config or by probing
    num_planes = config.plane_end if config.plane_end > 0 else 50

    # Create the base provider based on file naming pattern
    base_provider = _create_base_provider(
        image_path, tif_dir, prefix, filename, stem, num_planes, config
    )
    if base_provider is None:
        return None

    # Wrap with SplitChannelProvider if split or flip is active
    split_active = getattr(config, "split", 0) == 1
    flip_active = getattr(config, "flip", 0) == 1

    if split_active or flip_active:
        logger.info("Wrapping with SplitChannelProvider (split=%s, flip=%s)",
                    split_active, flip_active)
        return SplitChannelProvider(
            inner=base_provider,
            split=split_active,
            flip=flip_active,
        )

    return base_provider


def _create_base_provider(
    image_path: Path,
    tif_dir: Path,
    prefix: str,
    filename: str,
    stem: str,
    num_planes: int,
    config,
) -> ImageProvider | None:
    """Create the base (unwrapped) image provider from file naming pattern."""

    # Check if this is a per-plane file (contains '-p' in name)
    if "-p" in filename:
        # Per-plane format: {prefix}t{NNN}-p{NN}.tif or .zip
        if config.use_zip == 2:
            logger.info("Creating ZipTiffProvider: dir=%s, prefix='%s', planes=%d",
                        tif_dir, prefix, num_planes)
            return ZipTiffProvider(
                tif_directory=tif_dir,
                tif_prefix=prefix,
                num_planes=num_planes,
                use_zip=2,
            )
        else:
            time_match = re.search(r't(\d+)', filename)
            plane_match = re.search(r'p(\d+)', filename)
            t_width = len(time_match.group(1)) if time_match else 3
            p_width = len(plane_match.group(1)) if plane_match else 2

            pattern = f"{prefix}{{time:0{t_width}d}}-p{{plane:0{p_width}d}}.tif"
            logger.info("Creating TiffDirectoryProvider: dir=%s, pattern='%s'",
                        tif_dir, pattern)
            return TiffDirectoryProvider(
                directory=tif_dir,
                pattern=pattern,
                num_planes=num_planes,
            )
    else:
        # Stack TIFF format: one multi-page TIFF per timepoint
        ext = image_path.suffix
        time_match = re.search(r't(\d+)', stem)
        if time_match:
            t_digits = time_match.group(1)
            t_width = len(t_digits)
            if t_width > 1 and t_digits.startswith("0"):
                pattern = f"{prefix}{{time:0{t_width}d}}{ext}"
            else:
                pattern = f"{prefix}{{time}}{ext}"
        else:
            pattern = f"{prefix}{{time}}{ext}"

        # Probe the first file to determine actual plane count
        actual_planes = _probe_stack_planes(image_path)
        if actual_planes is not None:
            if actual_planes < num_planes:
                logger.info("Stack has %d planes (config planeEnd=%d). "
                           "Using actual plane count.",
                           actual_planes, num_planes)
                num_planes = actual_planes

        logger.info("Creating StackTiffProvider: dir=%s, pattern='%s', planes=%d",
                    tif_dir, pattern, num_planes)
        provider = StackTiffProvider(
            directory=tif_dir,
            pattern=pattern,
            num_channels=1,  # Raw file channels; split handled by wrapper
        )
        provider._num_planes_cached = num_planes
        return provider


def _create_multi_channel_provider(config) -> ImageProvider | None:
    """Create a MultiChannelFolderProvider from per-channel config paths.

    Handles configs like:
        <image numChannels="2" channel1="path/red_t1.tif" channel2="path/green_t1.tif"/>

    Each channel path is treated as a StackTiffProvider.
    """
    channel_providers = []
    flip_active = getattr(config, "flip", 0) == 1

    for ch_num in sorted(config.image_channels.keys()):
        ch_path = Path(config.image_channels[ch_num])
        if not ch_path.exists():
            logger.warning("Channel %d image not found: %s", ch_num, ch_path)
            continue

        ch_dir = ch_path.parent
        ch_stem = ch_path.stem
        ch_ext = ch_path.suffix

        # Parse the channel file pattern
        time_match = re.search(r't(\d+)', ch_stem)
        if time_match:
            # Build prefix up to and including 't'
            t_start = time_match.start()
            t_digits = time_match.group(1)
            prefix_part = ch_stem[:t_start + 1]  # includes 't'
            t_width = len(t_digits)

            if t_width > 1 and t_digits.startswith("0"):
                pattern = f"{prefix_part}{{time:0{t_width}d}}{ch_ext}"
            else:
                pattern = f"{prefix_part}{{time}}{ch_ext}"
        else:
            logger.warning("Could not parse time pattern from channel %d file: %s",
                          ch_num, ch_path.name)
            continue

        # Probe plane count from first file
        actual_planes = _probe_stack_planes(ch_path)
        num_planes = actual_planes or config.plane_end or 50

        provider = StackTiffProvider(directory=ch_dir, pattern=pattern)
        provider._num_planes_cached = num_planes
        channel_providers.append(provider)
        logger.info("Channel %d: dir=%s, pattern='%s', planes=%d",
                    ch_num, ch_dir, pattern, num_planes)

    if not channel_providers:
        logger.warning("No valid channel providers created")
        return None

    logger.info("Creating MultiChannelFolderProvider: %d channels, flip=%s",
                len(channel_providers), flip_active)
    return MultiChannelFolderProvider(channel_providers, flip=flip_active)


def _probe_stack_planes(path: Path) -> int | None:
    """Probe a TIFF file to determine the number of pages/planes.

    Returns None if the file cannot be read.
    """
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tif:
            n_pages = len(tif.pages)
            logger.info("Probed %s: %d pages, shape=%s",
                        path.name, n_pages,
                        tif.pages[0].shape if n_pages > 0 else "N/A")
            return n_pages
    except Exception as e:
        logger.warning("Could not probe TIFF %s: %s", path, e)
        return None


class NumpyProvider:
    """In-memory image provider backed by a numpy array.

    Useful for testing and for synthetic/generated image data.

    The data array should have shape (T, Z, Y, X) or (T, C, Z, Y, X).
    """

    def __init__(self, data: np.ndarray) -> None:
        """Initialize with a numpy array.

        Args:
            data: 4D (T,Z,Y,X) or 5D (T,C,Z,Y,X) array.
        """
        if data.ndim == 4:
            self._data = data
            self._n_channels = 1
        elif data.ndim == 5:
            self._data = data
            self._n_channels = data.shape[1]
        else:
            raise ValueError(f"Expected 4D or 5D array, got {data.ndim}D")

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        t_idx = time - 1
        p_idx = plane - 1
        if self._data.ndim == 5:
            return self._data[t_idx, channel, p_idx]
        return self._data[t_idx, p_idx]

    def get_stack(self, time: int, channel: int = 0) -> np.ndarray:
        t_idx = time - 1
        if self._data.ndim == 5:
            return self._data[t_idx, channel]
        return self._data[t_idx]

    @property
    def num_timepoints(self) -> int:
        return self._data.shape[0]

    @property
    def num_planes(self) -> int:
        if self._data.ndim == 5:
            return self._data.shape[2]
        return self._data.shape[1]

    @property
    def num_channels(self) -> int:
        return self._n_channels

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self._data.shape[-2], self._data.shape[-1])
