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
        t_width: int = 3,
        p_width: int = 2,
    ) -> None:
        """Initialise.

        Args:
            tif_directory: Directory containing the per-plane files.
            tif_prefix: Filename prefix before the ``t`` index.
            num_planes: Number of Z planes per timepoint.
            use_zip: 2 = each plane wrapped in a ``.zip``; 1 = loose ``.tif``.
            t_width: Zero-pad width of the time index.  ``0`` means
                unpadded (e.g. ``t1, t2, ..., t10``); ``3`` means
                3-digit padded (e.g. ``t001, t002, ..., t100``).
                Default ``3`` matches the historical Java AceTree
                convention.
            p_width: Zero-pad width of the plane index.  ``0`` means
                unpadded.  Default ``2``.
        """
        self.tif_directory = Path(tif_directory)
        self.tif_prefix = tif_prefix
        self._num_planes = num_planes
        self.use_zip = use_zip
        self.t_width = t_width
        self.p_width = p_width
        self._shape: tuple[int, int] | None = None
        self._num_timepoints: int | None = None
        # Keep the last-opened ZIP handle open (like Java's ZipImage)
        self._open_zip = None       # zipfile.ZipFile instance
        self._open_zip_path = None  # Path the handle is for

    def _get_zip_handle(self, zip_path: Path):
        """Get (or reuse) an open ZipFile handle."""
        if self._open_zip is not None and self._open_zip_path == zip_path:
            return self._open_zip
        # Close previous handle
        if self._open_zip is not None:
            try:
                self._open_zip.close()
            except Exception:
                pass
        self._open_zip = zipfile.ZipFile(zip_path, "r")
        self._open_zip_path = zip_path
        return self._open_zip

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
        ext = ".zip" if self.use_zip == 2 else ".tif"
        t_fmt = f"{{:0{self.t_width}d}}" if self.t_width > 0 else "{}"
        p_fmt = f"{{:0{self.p_width}d}}" if self.p_width > 0 else "{}"
        filename = (
            f"{self.tif_prefix}t{t_fmt.format(time)}"
            f"-p{p_fmt.format(plane)}{ext}"
        )
        return self.tif_directory / filename

    def _read_from_zip(self, zip_path: Path, tifffile) -> np.ndarray:
        """Read a TIFF from inside a ZIP file, reusing the open handle."""
        zf = self._get_zip_handle(zip_path)
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

    Like Java AceTree, keeps the current TiffFile handle open so that
    switching z-planes within the same timepoint reads a single page
    without re-parsing the entire file.

    Multichannel support
    --------------------
    When ``num_channels > 1`` the file's pages are assumed to carry
    interleaved multichannel data.  ``channel_order`` selects the layout:

    - ``"CZ"`` (channel-fastest, default): pages are ``Z1C1, Z1C2, Z2C1,
      Z2C2, ...``. Page index for ``(plane, channel)`` (1-based plane,
      0-based channel) is ``(plane-1)*num_channels + channel``.
    - ``"ZC"`` (planar): pages are ``Z1C1, Z2C1, ..., ZnC1, Z1C2, ...,
      ZnC2``. Page index is ``channel*num_planes + (plane-1)``.

    For ``num_channels == 1`` the provider behaves as a plain multi-page
    stack — each page is a distinct Z-plane.
    """

    def __init__(
        self,
        directory: str | Path,
        pattern: str = "t{time:03d}.tif",
        num_channels: int = 1,
        channel_order: str = "CZ",
    ) -> None:
        self.directory = Path(directory)
        self.pattern = pattern
        self._num_channels = max(1, int(num_channels))
        co = (channel_order or "CZ").upper()
        if co not in ("CZ", "ZC"):
            logger.warning("Unknown channel_order '%s'; falling back to 'CZ'",
                           channel_order)
            co = "CZ"
        self._channel_order = co
        self._shape: tuple[int, int] | None = None
        self._num_planes_cached: int | None = None
        self._num_timepoints: int | None = None
        # Open file handle cache (like Java's persistent ZipFile)
        self._open_tif = None       # tifffile.TiffFile instance
        self._open_tif_time = None  # timepoint the handle is for

    def _page_index(self, plane: int, channel: int, n_pages: int) -> int:
        """Map (1-based plane, 0-based channel) to a 0-based TIFF page index.

        Falls back to plane-1 when num_channels == 1 (pure Z stack).
        Raises IndexError on out-of-range plane or channel.
        """
        if not (0 <= channel < self._num_channels):
            raise IndexError(
                f"Channel {channel} out of range [0, {self._num_channels})"
            )
        if self._num_channels == 1:
            idx = plane - 1
        elif self._channel_order == "CZ":
            # Z1C1, Z1C2, Z2C1, Z2C2, ...
            idx = (plane - 1) * self._num_channels + channel
        else:  # "ZC"
            # Z1C1..ZnC1, Z1C2..ZnC2, ...
            num_planes = n_pages // self._num_channels if self._num_channels > 0 else n_pages
            idx = channel * num_planes + (plane - 1)
        if idx < 0 or idx >= n_pages:
            raise IndexError(
                f"Plane {plane} channel {channel} maps to page {idx} "
                f"(file has {n_pages} pages)"
            )
        return idx

    def _get_tiff_handle(self, time: int):
        """Get (or reuse) an open TiffFile handle for the given timepoint."""
        if self._open_tif is not None and self._open_tif_time == time:
            return self._open_tif

        # Close previous handle
        if self._open_tif is not None:
            try:
                self._open_tif.close()
            except Exception:
                pass

        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required: pip install tifffile")

        path = self._build_path(time)
        if not path.exists():
            raise FileNotFoundError(f"Stack not found: {path}")

        self._open_tif = tifffile.TiffFile(str(path))
        self._open_tif_time = time

        # Cache metadata from the open handle
        n_pages = len(self._open_tif.pages)
        if n_pages > 0 and self._shape is None:
            page = self._open_tif.pages[0]
            self._shape = (page.shape[-2], page.shape[-1])
        if self._num_planes_cached is None:
            # For interleaved multichannel, the true Z count is pages/channels.
            if self._num_channels > 1:
                self._num_planes_cached = n_pages // self._num_channels
            else:
                self._num_planes_cached = n_pages

        return self._open_tif

    def get_plane(self, time: int, plane: int, channel: int = 0) -> np.ndarray:
        tif = self._get_tiff_handle(time)
        n_pages = len(tif.pages)

        if n_pages == 0:
            raise FileNotFoundError(f"Empty TIFF stack for time={time}")

        if n_pages == 1:
            # Single page — might be 2D or multi-dim
            img = tif.pages[0].asarray()
            if img.ndim == 2:
                if plane != 1:
                    raise IndexError(f"Only 1 plane available, requested {plane}")
                self._update_shape(img)
                return img
            # Multi-dim single page (rare): fall through to full read
            return self._extract_plane_from_array(img, plane, channel)

        # Multi-page: map (plane, channel) to the correct TIFF page
        page_idx = self._page_index(plane, channel, n_pages)
        result = tif.pages[page_idx].asarray()
        self._update_shape(result)
        return result

    def _extract_plane_from_array(self, img, plane, channel):
        """Extract a plane from a pre-loaded multi-dimensional array."""
        if img.ndim == 3:
            plane_idx = plane - 1
            if plane_idx >= img.shape[0]:
                raise IndexError(f"Plane {plane} out of range (max {img.shape[0]})")
            result = img[plane_idx]
            self._update_shape(result)
            self._num_planes_cached = img.shape[0]
            return result
        elif img.ndim == 4:
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
        tif = self._get_tiff_handle(time)
        n_pages = len(tif.pages)

        if n_pages == 0:
            raise FileNotFoundError(f"Empty TIFF stack for time={time}")

        if self._num_channels > 1:
            # De-interleave pages for the requested channel
            if not (0 <= channel < self._num_channels):
                raise IndexError(
                    f"Channel {channel} out of range [0, {self._num_channels})"
                )
            num_planes = n_pages // self._num_channels
            planes = []
            for z in range(1, num_planes + 1):
                page_idx = self._page_index(z, channel, n_pages)
                planes.append(tif.pages[page_idx].asarray())
            img = np.stack(planes) if planes else np.empty((0,))
            self._num_planes_cached = num_planes
            if num_planes > 0:
                self._update_shape(img[0])
            return img

        # Single-channel: read all pages into a stack
        planes = []
        for page in tif.pages:
            planes.append(page.asarray())
        img = np.stack(planes)

        self._num_planes_cached = img.shape[0]
        if img.shape[0] > 0:
            self._update_shape(img[0])
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

    # Wrap with SplitChannelProvider if split or flip is active.
    # Interleaved multichannel stacks already resolve channels at the page
    # level, so do not double-wrap them with the horizontal split logic —
    # that would halve a valid image and produce phantom channels.
    split_active = getattr(config, "split", 0) == 1
    flip_active = getattr(config, "flip", 0) == 1
    interleaved = bool(getattr(config, "stack_interleaved", False))

    if (split_active or flip_active) and not interleaved:
        logger.info("Wrapping with SplitChannelProvider (split=%s, flip=%s)",
                    split_active, flip_active)
        return SplitChannelProvider(
            inner=base_provider,
            split=split_active,
            flip=flip_active,
        )

    if interleaved and (split_active or flip_active):
        logger.info(
            "Ignoring split=%s/flip=%s for interleaved multichannel stack",
            split_active, flip_active,
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
        # Pull example widths to use as fallback if scan finds nothing.
        time_match_example = re.search(r"t(\d+)", filename)
        plane_match_example = re.search(r"-p(\d+)", filename)
        t_width_example = (
            len(time_match_example.group(1)) if time_match_example else 3
        )
        p_width_example = (
            len(plane_match_example.group(1)) if plane_match_example else 2
        )
        # Detect both padding widths by scanning the directory.  This is
        # robust against the example filename happening to land on a
        # high time index (e.g. ``t100`` from a 3-digit padded set,
        # which has no leading zero).
        if time_match_example:
            prefix_before_t = filename[: time_match_example.start()]
        else:
            prefix_before_t = prefix

        zip_ext = ".zip" if config.use_zip == 2 else ".tif"
        t_width, p_width = _detect_per_plane_padding(
            tif_dir, prefix_before_t, zip_ext,
            fallback=(t_width_example, p_width_example),
        )

        if config.use_zip == 2:
            logger.info(
                "Creating ZipTiffProvider: dir=%s, prefix='%s', planes=%d, "
                "t_width=%d, p_width=%d",
                tif_dir, prefix, num_planes, t_width, p_width,
            )
            return ZipTiffProvider(
                tif_directory=tif_dir,
                tif_prefix=prefix,
                num_planes=num_planes,
                use_zip=2,
                t_width=t_width,
                p_width=p_width,
            )
        else:
            t_fmt = f"{{time:0{t_width}d}}" if t_width > 0 else "{time}"
            p_fmt = f"{{plane:0{p_width}d}}" if p_width > 0 else "{plane}"
            # ``prefix`` already includes the literal ``t`` separator
            # (config.tif_prefix is canonically everything up to and
            # including ``t``).  Don't prepend another ``t`` here.
            pattern = f"{prefix}{t_fmt}-p{p_fmt}.tif"
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
            # Re-search on the full filename so the offsets match the
            # directory-scan input exactly.
            tm_in_filename = re.search(r"t(\d+)", filename)
            if tm_in_filename:
                # prefix_before_t excludes the literal 't' so the
                # directory-scan regex (which adds its own 't<digits>')
                # matches sibling files.
                prefix_before_t = filename[: tm_in_filename.start()]
            else:
                prefix_before_t = prefix
            example_width = len(time_match.group(1))
            t_width = _detect_time_padding(
                tif_dir, prefix_before_t, ext, fallback_width=example_width,
            )
            t_fmt = f"{{time:0{t_width}d}}" if t_width > 0 else "{time}"
            # ``prefix`` (config.tif_prefix) canonically includes the
            # literal ``t``, so don't prepend another one here.
            pattern = f"{prefix}{t_fmt}{ext}"
        else:
            pattern = f"{prefix}{{time}}{ext}"

        # Interleaved multichannel stack mode (single file, pages = Z*C)
        interleaved = bool(getattr(config, "stack_interleaved", False))
        stack_channels = int(getattr(config, "num_channels", 1) or 1) if interleaved else 1
        channel_order = (getattr(config, "stack_channel_order", "CZ") or "CZ")

        # Probe the first file to determine actual plane count
        actual_pages = _probe_stack_planes(image_path)
        if actual_pages is not None:
            actual_planes = actual_pages // stack_channels if stack_channels > 1 else actual_pages
            if actual_planes < num_planes:
                logger.info("Stack has %d planes (config planeEnd=%d). "
                           "Using actual plane count.",
                           actual_planes, num_planes)
                num_planes = actual_planes

        logger.info(
            "Creating StackTiffProvider: dir=%s, pattern='%s', planes=%d, "
            "channels=%d, order=%s",
            tif_dir, pattern, num_planes, stack_channels, channel_order,
        )
        provider = StackTiffProvider(
            directory=tif_dir,
            pattern=pattern,
            num_channels=stack_channels,
            channel_order=channel_order,
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
            # Build prefix up to (not including) 't' so we can scan the
            # directory for siblings matching the same shape.
            t_start = time_match.start()
            prefix_part_with_t = ch_stem[:t_start + 1]  # includes 't'
            prefix_before_t = ch_stem[:t_start]         # excludes 't'
            example_width = len(time_match.group(1))
            t_width = _detect_time_padding(
                ch_dir, prefix_before_t, ch_ext,
                fallback_width=example_width,
            )
            t_fmt = f"{{time:0{t_width}d}}" if t_width > 0 else "{time}"
            pattern = f"{prefix_part_with_t}{t_fmt}{ch_ext}"
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


def _detect_time_padding(
    directory: Path,
    prefix_before_t: str,
    suffix_after_t: str,
    fallback_width: int = 0,
) -> int:
    """Detect the zero-padding width of the time index in a directory of
    image files matching ``{prefix_before_t}t{digits}{suffix_after_t}``.

    Scans every file in ``directory``.  The presence of *any* file whose
    digit run starts with ``0`` (and has more than one digit) proves
    padding is in use; the width is taken from that file.  This works
    even when the example filename happens to have a non-leading-zero
    time index (e.g. ``t100.tif`` from a 3-digit padded set).

    If no leading-zero example is found, the convention is unpadded
    (``t1, t2, ..., t10, t100``).

    Args:
        directory: Directory to scan.
        prefix_before_t: Filename portion before the literal ``t`` and
            digit run (e.g. ``"image_"``).  Will be regex-escaped.
        suffix_after_t: Filename portion after the digit run, including
            extension (e.g. ``".tif"``, or ``"-p01.tif"`` for per-plane).
            Will be regex-escaped.
        fallback_width: Width to return if no matching files are found
            in the directory.  Default ``0`` (unpadded).

    Returns:
        The detected pad width (``>= 1``), or ``0`` for unpadded.
    """
    if not directory.exists():
        return fallback_width
    pattern = re.compile(
        re.escape(prefix_before_t) + r"t(\d+)" + re.escape(suffix_after_t) + r"$"
    )
    seen_widths = set()
    leading_zero_widths = set()
    for entry in directory.iterdir():
        m = pattern.match(entry.name)
        if not m:
            continue
        digits = m.group(1)
        seen_widths.add(len(digits))
        if len(digits) > 1 and digits[0] == "0":
            leading_zero_widths.add(len(digits))
    if not seen_widths:
        return fallback_width
    if leading_zero_widths:
        # Padded set proven by the presence of at least one zero-prefix.
        # If multiple widths somehow appear, prefer the longest (most
        # likely the canonical pad width).
        return max(leading_zero_widths)
    if len(seen_widths) == 1:
        # No leading zeros and a single uniform width — could be a
        # single-file dataset or a small unpadded set.  Treat as
        # unpadded; pad-vs-unpadded is indistinguishable for any single
        # value, and the file we're about to load resolves either way.
        return 0
    # Mixed widths and no leading zeros → unpadded (e.g. t1, t2, ..., t100).
    return 0


def _detect_per_plane_padding(
    directory: Path,
    prefix_before_t: str,
    ext: str,
    fallback: tuple[int, int] = (3, 2),
) -> tuple[int, int]:
    """Detect (t_width, p_width) for per-plane filenames.

    Scans ``directory`` for files matching
    ``{prefix_before_t}t<digits>-p<digits>{ext}`` and returns the
    detected pad widths for the time and plane indices.  A leading-zero
    digit run in any matched file proves padding for that index;
    otherwise the index is treated as unpadded (``0``).

    Args:
        directory: Directory to scan.
        prefix_before_t: Filename portion before the ``t``.
        ext: File extension including the leading dot.
        fallback: ``(t_width, p_width)`` to return if no matching files
            are found in the directory.

    Returns:
        ``(t_width, p_width)`` — each ``0`` for unpadded, ``>= 1`` for
        the detected padded width.
    """
    if not directory.exists():
        return fallback
    pattern = re.compile(
        re.escape(prefix_before_t) + r"t(\d+)-p(\d+)" + re.escape(ext) + r"$"
    )
    t_seen: set[int] = set()
    p_seen: set[int] = set()
    t_padded: set[int] = set()
    p_padded: set[int] = set()
    for entry in directory.iterdir():
        m = pattern.match(entry.name)
        if not m:
            continue
        td, pd = m.group(1), m.group(2)
        t_seen.add(len(td))
        p_seen.add(len(pd))
        if len(td) > 1 and td[0] == "0":
            t_padded.add(len(td))
        if len(pd) > 1 and pd[0] == "0":
            p_padded.add(len(pd))
    if not t_seen:
        return fallback
    t_width = max(t_padded) if t_padded else 0
    p_width = max(p_padded) if p_padded else 0
    return t_width, p_width


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
