"""Movie — temporal and spatial bounds of a dataset.

Tracks the overall dimensions and timing of the embryo movie.

Ported from: org.rhwlab.snight.Movie (Movie.java)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Movie:
    """Temporal and spatial bounds of a dataset.

    Attributes:
        start_time: First timepoint (1-based).
        end_time: Last timepoint (1-based).
        num_planes: Number of z-planes per stack.
        width: Image width in pixels.
        height: Image height in pixels.
        xy_res: XY pixel resolution in microns.
        z_res: Z-plane spacing in microns.
    """

    start_time: int = 1
    end_time: int = 1
    num_planes: int = 30
    width: int = 0
    height: int = 0
    xy_res: float = 0.09
    z_res: float = 1.0

    @property
    def num_timepoints(self) -> int:
        """Total number of timepoints."""
        return self.end_time - self.start_time + 1

    @property
    def z_pix_res(self) -> float:
        """Z resolution in pixel units (z_res / xy_res)."""
        if self.xy_res == 0:
            return 1.0
        return self.z_res / self.xy_res
