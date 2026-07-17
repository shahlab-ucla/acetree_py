"""Deterministic synthetic 3-D movies for StarryNite parity testing.

The simulator is intentionally small and transparent.  It is not meant to be
a biologically complete embryo model; each scene isolates one behavior so a
regression report can explain *why* the Python and MATLAB outputs diverged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ...api import Calibration


@dataclass(frozen=True, slots=True)
class SyntheticObject:
    """One rendered nucleus and its lineage identity.

    Positions and Gaussian sigmas are in NumPy ``(z, y, x)`` pixel order.
    ``parent_id`` is populated only on the first frame after a division.
    """

    object_id: str
    frame: int
    position_zyx_px: tuple[float, float, float]
    sigma_zyx_px: tuple[float, float, float]
    amplitude: float
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id cannot be empty")
        if self.frame < 1:
            raise ValueError("frame must be positive and 1-based")
        for name, values in (
            ("position_zyx_px", self.position_zyx_px),
            ("sigma_zyx_px", self.sigma_zyx_px),
        ):
            if len(values) != 3 or not all(math.isfinite(float(item)) for item in values):
                raise ValueError(f"{name} must contain three finite values")
        if any(float(item) <= 0 for item in self.sigma_zyx_px):
            raise ValueError("sigma_zyx_px values must be positive")
        if not math.isfinite(float(self.amplitude)) or self.amplitude <= 0:
            raise ValueError("amplitude must be positive and finite")


@dataclass(frozen=True, slots=True)
class SyntheticMovie:
    """Rendered movie plus exact object and lineage ground truth."""

    name: str
    frames_tzyx: np.ndarray
    objects: tuple[SyntheticObject, ...]
    calibration: Calibration
    expected_radius_um: float
    description: str = ""

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames_tzyx)
        if not self.name:
            raise ValueError("Synthetic movie name cannot be empty")
        if frames.ndim != 4 or any(size < 1 for size in frames.shape):
            raise ValueError("frames_tzyx must be a non-empty TZYX array")
        if not np.issubdtype(frames.dtype, np.number) or not np.all(np.isfinite(frames)):
            raise ValueError("frames_tzyx must contain finite numeric values")
        if self.expected_radius_um <= 0:
            raise ValueError("expected_radius_um must be positive")
        if any(item.frame > frames.shape[0] for item in self.objects):
            raise ValueError("Synthetic truth references a frame outside the movie")
        object.__setattr__(self, "frames_tzyx", np.asarray(frames, dtype=np.float32))
        object.__setattr__(self, "objects", tuple(self.objects))

    @property
    def frame_count(self) -> int:
        return int(self.frames_tzyx.shape[0])

    def truth_for_frame(self, frame: int) -> tuple[SyntheticObject, ...]:
        if not 1 <= frame <= self.frame_count:
            raise IndexError(f"frame must be in 1..{self.frame_count}")
        return tuple(item for item in self.objects if item.frame == frame)

    def truth_positions_xyz_px(self, frame: int) -> np.ndarray:
        """Return ground-truth positions in zero-based ``(x, y, z)`` pixels."""

        objects = self.truth_for_frame(frame)
        if not objects:
            return np.empty((0, 3), dtype=np.float64)
        positions_zyx = np.asarray(
            [item.position_zyx_px for item in objects], dtype=np.float64
        )
        return positions_zyx[:, ::-1]


def render_movie(
    name: str,
    shape_zyx: tuple[int, int, int],
    objects: Iterable[SyntheticObject],
    *,
    calibration: Calibration = Calibration(0.5, 1.0),
    expected_radius_um: float = 2.0,
    background: float = 100.0,
    gradient_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gaussian_noise_std: float = 0.0,
    poisson_scale: float | None = None,
    seed: int = 0,
    description: str = "",
) -> SyntheticMovie:
    """Render Gaussian nuclei into a reproducible floating-point TZYX movie."""

    if len(shape_zyx) != 3 or any(int(size) < 5 for size in shape_zyx):
        raise ValueError("shape_zyx must contain three sizes of at least five pixels")
    truth = tuple(objects)
    if not truth:
        frame_count = 1
    else:
        frame_count = max(item.frame for item in truth)
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(tuple(map(int, shape_zyx)), dtype=np.float64)
    gradient = (
        float(gradient_xyz[0]) * x
        + float(gradient_xyz[1]) * y
        + float(gradient_xyz[2]) * z
    )
    frames = np.empty((frame_count, *shape_zyx), dtype=np.float32)
    for frame in range(1, frame_count + 1):
        image = np.full(shape_zyx, float(background), dtype=np.float64) + gradient
        for item in truth:
            if item.frame != frame:
                continue
            center_z, center_y, center_x = item.position_zyx_px
            sigma_z, sigma_y, sigma_x = item.sigma_zyx_px
            exponent = (
                ((z - center_z) / sigma_z) ** 2
                + ((y - center_y) / sigma_y) ** 2
                + ((x - center_x) / sigma_x) ** 2
            )
            image += float(item.amplitude) * np.exp(-0.5 * exponent)
        if poisson_scale is not None:
            if not math.isfinite(poisson_scale) or poisson_scale <= 0:
                raise ValueError("poisson_scale must be positive and finite")
            image = rng.poisson(np.maximum(image, 0.0) * poisson_scale) / poisson_scale
        if gaussian_noise_std:
            if not math.isfinite(gaussian_noise_std) or gaussian_noise_std < 0:
                raise ValueError("gaussian_noise_std must be finite and non-negative")
            image += rng.normal(0.0, gaussian_noise_std, size=shape_zyx)
        frames[frame - 1] = np.asarray(image, dtype=np.float32)
    return SyntheticMovie(
        name=name,
        frames_tzyx=frames,
        objects=truth,
        calibration=calibration,
        expected_radius_um=float(expected_radius_um),
        description=description,
    )


def default_synthetic_suite(seed: int = 1731) -> tuple[SyntheticMovie, ...]:
    """Return the standard parity suite, from isolated detection to division."""

    sigma = (1.25, 2.5, 2.5)
    calibration = Calibration(0.5, 1.0)
    static_shape = (17, 49, 49)
    isolated = render_movie(
        "isolated",
        static_shape,
        (SyntheticObject("a", 1, (8.0, 24.0, 24.0), sigma, 350.0),),
        calibration=calibration,
        gaussian_noise_std=1.5,
        seed=seed,
        description="One centered nucleus; establishes numerical/filter parity.",
    )
    close_pair = render_movie(
        "close_pair",
        static_shape,
        (
            SyntheticObject("a", 1, (8.0, 24.0, 20.5), sigma, 350.0),
            SyntheticObject("b", 1, (8.0, 24.0, 27.5), sigma, 350.0),
        ),
        calibration=calibration,
        gaussian_noise_std=1.5,
        seed=seed + 1,
        description="Two near-resolution nuclei; probes maxima suppression and merging.",
    )
    unequal_pair = render_movie(
        "unequal_pair",
        static_shape,
        (
            SyntheticObject("bright", 1, (8.0, 20.0, 19.0), sigma, 400.0),
            SyntheticObject("dim", 1, (8.0, 29.0, 30.0), sigma, 105.0),
        ),
        calibration=calibration,
        gradient_xyz=(0.04, -0.02, 0.1),
        gaussian_noise_std=2.5,
        seed=seed + 2,
        description="Bright and dim nuclei on a gradient; probes threshold transitions.",
    )
    boundary = render_movie(
        "boundary",
        static_shape,
        (
            SyntheticObject("edge", 1, (2.0, 5.0, 6.0), sigma, 350.0),
            SyntheticObject("center", 1, (9.0, 30.0, 31.0), sigma, 300.0),
        ),
        calibration=calibration,
        gaussian_noise_std=1.5,
        seed=seed + 3,
        description="One edge nucleus and one interior control; probes replicate padding.",
    )

    division_objects: list[SyntheticObject] = []
    for frame, x_position in enumerate((20.0, 21.0, 22.0), start=1):
        division_objects.append(
            SyntheticObject("parent", frame, (8.0, 24.0, x_position), sigma, 340.0)
        )
    for frame, separation in enumerate((2.8, 4.0, 5.2), start=4):
        division_objects.extend(
            (
                SyntheticObject(
                    "daughter_l",
                    frame,
                    (8.0, 24.0 - 0.3 * (frame - 4), 23.0 - separation),
                    (1.1, 2.1, 2.1),
                    240.0,
                    parent_id="parent" if frame == 4 else None,
                ),
                SyntheticObject(
                    "daughter_r",
                    frame,
                    (8.0, 24.0 + 0.3 * (frame - 4), 23.0 + separation),
                    (1.1, 2.1, 2.1),
                    235.0,
                    parent_id="parent" if frame == 4 else None,
                ),
            )
        )
    division = render_movie(
        "division",
        static_shape,
        division_objects,
        calibration=calibration,
        gaussian_noise_std=2.0,
        seed=seed + 4,
        description="A moving parent separates into two daughters over six frames.",
    )
    return (isolated, close_pair, unequal_pair, boundary, division)


def resolution_noise_suite(
    *,
    seed: int = 1731,
    separations_xy_px: tuple[float, ...] = (
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        7.0,
        8.0,
    ),
    noise_std: tuple[float, ...] = (0.0, 5.0, 20.0),
    seed_count: int = 3,
) -> tuple[SyntheticMovie, ...]:
    """Build a multiseed daughter-separation and noise stress matrix."""

    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    if not separations_xy_px or any(
        not math.isfinite(value) or value <= 0 for value in separations_xy_px
    ):
        raise ValueError("separations_xy_px must contain positive finite values")
    if not noise_std or any(
        not math.isfinite(value) or value < 0 for value in noise_std
    ):
        raise ValueError("noise_std must contain finite non-negative values")

    movies: list[SyntheticMovie] = []
    calibration = Calibration(0.5, 1.0)
    shape = (17, 49, 49)
    sigma = (1.1, 2.1, 2.1)
    for seed_index in range(seed_count):
        for noise_index, noise in enumerate(noise_std):
            case_seed = seed + seed_index * 10_000 + noise_index * 1_000
            noise_label = f"{noise:g}".replace(".", "p")
            movies.append(
                render_movie(
                    f"noise-s{seed_index:02d}-n{noise_label}",
                    shape,
                    (),
                    calibration=calibration,
                    gaussian_noise_std=float(noise),
                    seed=case_seed,
                    description=(
                        "Noise-only false-positive control for the resolution matrix."
                    ),
                )
            )
            for separation_index, separation in enumerate(separations_xy_px):
                separation_label = f"{separation:g}".replace(".", "p")
                center_x = 23.0
                movies.append(
                    render_movie(
                        (
                            f"pair-s{seed_index:02d}-n{noise_label}-"
                            f"d{separation_label}"
                        ),
                        shape,
                        (
                            SyntheticObject(
                                "daughter_l",
                                1,
                                (8.0, 24.0, center_x - separation / 2.0),
                                sigma,
                                240.0,
                            ),
                            SyntheticObject(
                                "daughter_r",
                                1,
                                (8.0, 24.0, center_x + separation / 2.0),
                                sigma,
                                235.0,
                            ),
                        ),
                        calibration=calibration,
                        gaussian_noise_std=float(noise),
                        seed=case_seed + separation_index + 1,
                        description=(
                            "Two daughter-like nuclei at a controlled XY separation "
                            "and noise level."
                        ),
                    )
                )
    return tuple(movies)


def lineage_synthetic_suite(seed: int = 2718) -> tuple[SyntheticMovie, ...]:
    """Return multiframe cases isolating links, gaps, divisions, and cleanup."""

    calibration = Calibration(1.0, 3.0)
    shape = (13, 43, 55)
    sigma = (2.2 / 3.0, 2.2, 2.2)
    controls = ((14.0, 13.0, 5.0), (39.0, 13.0, 6.0), (14.0, 31.0, 7.0))

    translation_objects: list[SyntheticObject] = []
    for frame in range(1, 6):
        for index, (x, y, z) in enumerate((*controls, (39.0, 31.0, 8.0))):
            translation_objects.append(
                SyntheticObject(
                    f"cell-{index}",
                    frame,
                    (z - 1.0, y - 1.0, x - 1.0 + frame - 1),
                    sigma,
                    100.0,
                )
            )
    translation = render_movie(
        "lineage-translation",
        shape,
        translation_objects,
        calibration=calibration,
        expected_radius_um=4.0,
        background=0.0,
        seed=seed,
        description="Four separated nuclei translating for five frames.",
    )

    gap_objects: list[SyntheticObject] = []
    for frame in range(1, 9):
        for index, (x, y, z) in enumerate(controls):
            if index == 1 and frame == 4:
                continue
            gap_objects.append(
                SyntheticObject(
                    f"cell-{index}",
                    frame,
                    (z - 1.0, y - 1.0, x - 1.0 + 0.6 * (frame - 1)),
                    sigma,
                    100.0,
                )
            )
    gap = render_movie(
        "lineage-one-frame-gap",
        shape,
        gap_objects,
        calibration=calibration,
        expected_radius_um=4.0,
        background=0.0,
        seed=seed + 1,
        description="One of three tracks has exactly one missed observation.",
    )

    division_objects: list[SyntheticObject] = []
    for frame in range(1, 12):
        for index, (x, y, z) in enumerate(controls):
            division_objects.append(
                SyntheticObject(
                    f"control-{index}",
                    frame,
                    (z - 1.0, y - 1.0, x - 1.0 + 0.35 * (frame - 1)),
                    sigma,
                    100.0,
                )
            )
        if frame <= 5:
            division_objects.append(
                SyntheticObject(
                    "parent",
                    frame,
                    (7.0, 30.0, 27.0 + 0.35 * (frame - 1)),
                    sigma,
                    110.0,
                )
            )
        else:
            separation = 4.2 + 0.35 * (frame - 6)
            center_x = 27.0 + 0.35 * (frame - 1)
            division_objects.extend(
                (
                    SyntheticObject(
                        "daughter-left",
                        frame,
                        (7.0, 30.0 - 0.2 * (frame - 6), center_x - separation),
                        sigma,
                        75.0,
                        parent_id="parent" if frame == 6 else None,
                    ),
                    SyntheticObject(
                        "daughter-right",
                        frame,
                        (7.0, 30.0 + 0.2 * (frame - 6), center_x + separation),
                        sigma,
                        75.0,
                        parent_id="parent" if frame == 6 else None,
                    ),
                )
            )
    division = render_movie(
        "lineage-long-division",
        shape,
        division_objects,
        calibration=calibration,
        expected_radius_um=4.0,
        background=0.0,
        seed=seed + 2,
        description="A parent becomes two daughters with six-frame branches.",
    )

    model_division_objects: list[SyntheticObject] = []
    model_controls = ((10.0, 10.0, 5.0), (27.0, 10.0, 6.0), (44.0, 10.0, 7.0))
    daughter_sigma = tuple(value * 0.95 for value in sigma)
    for frame in range(1, 8):
        for index, (x, y, z) in enumerate(model_controls):
            model_division_objects.append(
                SyntheticObject(
                    f"control-{index}",
                    frame,
                    (z - 1.0, y - 1.0, x - 1.0 + 0.35 * (frame - 1)),
                    sigma,
                    110.0,
                )
            )
        center_x = 27.0 + 0.35 * (frame - 1)
        if frame <= 2:
            model_division_objects.append(
                SyntheticObject(
                    "parent",
                    frame,
                    (7.0, 30.0, center_x),
                    sigma,
                    110.0,
                )
            )
        else:
            extra = 0.2 * (frame - 3)
            model_division_objects.extend(
                (
                    SyntheticObject(
                        "daughter-left",
                        frame,
                        (7.0, 30.0 - 0.1 * extra, center_x - 8.0 - extra),
                        daughter_sigma,
                        99.0,
                        parent_id="parent" if frame == 3 else None,
                    ),
                    SyntheticObject(
                        "daughter-right",
                        frame,
                        (7.0, 30.0 + 0.1 * extra, center_x + 8.0 + extra),
                        daughter_sigma,
                        99.0,
                        parent_id="parent" if frame == 3 else None,
                    ),
                )
            )
    model_division = render_movie(
        "lineage-model-division",
        (13, 45, 55),
        model_division_objects,
        calibration=calibration,
        expected_radius_um=4.0,
        background=0.0,
        seed=seed + 3,
        description=(
            "A seven-frame robust positive class-1 division for the 2019 model."
        ),
    )

    artifact_objects: list[SyntheticObject] = []
    for frame in range(1, 9):
        for index, (x, y, z) in enumerate(controls):
            artifact_objects.append(
                SyntheticObject(
                    f"cell-{index}",
                    frame,
                    (z - 1.0, y - 1.0, x - 1.0 + 0.5 * (frame - 1)),
                    sigma,
                    100.0,
                )
            )
        if frame == 4:
            artifact_objects.append(
                SyntheticObject("transient", frame, (9.0, 36.0, 26.0), sigma, 90.0)
            )
    artifact = render_movie(
        "lineage-transient-artifact",
        shape,
        artifact_objects,
        calibration=calibration,
        expected_radius_um=4.0,
        background=0.0,
        seed=seed + 4,
        description="A one-frame detector artifact probes isolated-fragment deletion.",
    )
    return translation, gap, division, model_division, artifact


__all__ = [
    "SyntheticMovie",
    "SyntheticObject",
    "default_synthetic_suite",
    "lineage_synthetic_suite",
    "resolution_noise_suite",
    "render_movie",
]
