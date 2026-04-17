"""Pixel-level measurement of nucleus fluorescence intensity.

Direct port of the Java AceTree measure routine (``ExtractRed`` +
``RedBkgComp2``) from ``org.rhwlab.analyze``.  Each nucleus is
modelled as a 3D sphere of diameter ``nuc.size`` centred at
``(nuc.x, nuc.y, nuc.z)``; at each Z-plane that the sphere
intersects we rasterise a 2D disk whose diameter follows the
spherical projection::

    R = nuc.size / 2
    y = (nuc.z - z_plane) * z_pix_res / R
    r(z_plane) = sqrt(1 - y**2) * R    # 0 when the plane exits the sphere

We sum pixel intensities (and pixel counts) over every disk.  A
concentric "annulus" around each disk is measured the same way
and used to estimate a local background (AceTree's ``rwcorr1``
global-background variant).

The raw aggregation is kept un-normalised so callers can combine
nuclei, write CSVs, or apply downstream corrections without
double-dividing.  The AceTree scaling convention (``* 1000``) is
applied at the consumer (e.g. ``run_measure`` in
``nuclei_manager.py``), matching ``NucleiMgr.computeRWeight``::

    rwraw   = round(sum_in  * 1000 / count_in)
    rwcorr1 = round(sum_ann * 1000 / count_ann)

Ported from: ``org.rhwlab.analyze.ExtractRed.processImageUsingPolygon``
            + ``nucDiameter`` + ``org.rhwlab.analyze.RedBkgComp2``
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..core.nucleus import Nucleus

logger = logging.getLogger(__name__)


# Default annulus outer radius as a multiple of the inner disk radius.
# AceTree's RedBkgComp2 used an annulus roughly 1.5x the nucleus radius;
# callers can override per-call via `annulus_scale`.
DEFAULT_ANNULUS_SCALE: float = 1.5


def project_radius(
    nuc_z: float,
    z_plane: int,
    radius: float,
    z_pix_res: float,
) -> float:
    """Projected in-plane radius of a sphere at a given Z-plane.

    Returns the radius (in XY pixels) of the 2D cross-section of a
    sphere centred at ``(_, _, nuc_z)`` with radius ``radius`` (XY
    pixels) at integer plane ``z_plane``.

    Args:
        nuc_z: Nucleus Z position (may be sub-plane, in pixel units).
        z_plane: 0-based or 1-based integer plane index (units match
            ``nuc_z``).
        radius: Nucleus XY radius in pixels (= ``nuc.size / 2``).
        z_pix_res: Z resolution / XY resolution (scale factor applied
            to Z offset before normalising by radius).

    Returns:
        Non-negative projected radius in XY pixels, or 0 if the plane
        is outside the sphere.
    """
    if radius <= 0:
        return 0.0
    y = (nuc_z - z_plane) * z_pix_res / radius
    r2 = 1.0 - y * y
    if r2 <= 0:
        return 0.0
    return math.sqrt(r2) * radius


def _disk_masks(
    shape: tuple[int, int],
    cx: int,
    cy: int,
    inner_r: float,
    outer_r: float,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    """Return (inner_mask, annulus_mask, bbox_slices) for a disk + annulus.

    The masks are returned cropped to the minimum bounding box around
    the outer disk (clipped to the image bounds).  This avoids
    rasterising a full-image mask for every nucleus.

    Args:
        shape: (H, W) of the image plane.
        cx, cy: disk centre pixel coords.
        inner_r: inner disk radius in pixels.
        outer_r: annulus outer radius in pixels (>= inner_r).

    Returns:
        (inner_mask, annulus_mask, (slice_y, slice_x)) where masks
        are 2D boolean arrays matching the bbox shape, and slices
        can be used to index into the full plane.
    """
    h, w = shape
    # Bounding box around the outer disk, clipped to image
    x0 = max(0, int(math.floor(cx - outer_r)))
    x1 = min(w, int(math.ceil(cx + outer_r)) + 1)
    y0 = max(0, int(math.floor(cy - outer_r)))
    y1 = min(h, int(math.ceil(cy + outer_r)) + 1)

    if x1 <= x0 or y1 <= y0:
        # Disk entirely outside the image — return empty masks
        empty = np.zeros((0, 0), dtype=bool)
        return empty, empty, (slice(0, 0), slice(0, 0))

    ys = np.arange(y0, y1).reshape(-1, 1)
    xs = np.arange(x0, x1).reshape(1, -1)
    dx = xs - cx
    dy = ys - cy
    d2 = dx * dx + dy * dy

    inner = d2 <= inner_r * inner_r
    outer = d2 <= outer_r * outer_r
    annulus = outer & ~inner

    return inner, annulus, (slice(y0, y1), slice(x0, x1))


def measure_nucleus(
    stack: np.ndarray,
    nuc: Nucleus,
    z_pix_res: float,
    annulus_scale: float = DEFAULT_ANNULUS_SCALE,
) -> tuple[int, int, int, int]:
    """Measure fluorescence inside and around a single nucleus.

    Iterates every Z-plane the nucleus bounding sphere intersects,
    rasterises the projected disk (inner) and an annulus (outer minus
    inner), and accumulates pixel sums + counts.

    Args:
        stack: 3D image stack, shape ``(Z, Y, X)``, any integer or
            floating-point dtype.  Pixels are interpreted as intensity.
        nuc: The nucleus to measure.  ``nuc.status < 1`` is treated
            as a dead nucleus and returns all zeros.
        z_pix_res: Z / XY pixel resolution ratio.
        annulus_scale: Annulus outer-radius multiplier relative to the
            inner disk radius at each plane.  Default 1.5.

    Returns:
        ``(sum_in, count_in, sum_ann, count_ann)`` as ints.  ``count_``
        fields are pixel counts (never negative).  The sums are cast
        to Python int to avoid numpy-overflow surprises.
    """
    if nuc.status < 1:
        return 0, 0, 0, 0

    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (Z,Y,X), got shape {stack.shape}")

    nz, ny, nx = stack.shape
    radius = nuc.size / 2.0
    if radius <= 0:
        return 0, 0, 0, 0

    # Z extent of the sphere, in plane units.  Java code walked all
    # planes; here we clamp to planes that could touch the sphere.
    half_z = radius / z_pix_res if z_pix_res > 0 else radius
    z_lo = max(0, int(math.floor(nuc.z - half_z)))
    z_hi = min(nz - 1, int(math.ceil(nuc.z + half_z)))

    sum_in = 0
    count_in = 0
    sum_ann = 0
    count_ann = 0

    for z in range(z_lo, z_hi + 1):
        r_in = project_radius(nuc.z, z, radius, z_pix_res)
        if r_in <= 0:
            continue
        r_out = r_in * annulus_scale

        inner, annulus, (sy, sx) = _disk_masks(
            (ny, nx), int(round(nuc.x)), int(round(nuc.y)), r_in, r_out,
        )
        if inner.size == 0:
            continue

        plane = stack[z, sy, sx]
        if inner.any():
            sum_in += int(plane[inner].sum())
            count_in += int(inner.sum())
        if annulus.any():
            sum_ann += int(plane[annulus].sum())
            count_ann += int(annulus.sum())

    return sum_in, count_in, sum_ann, count_ann


def measure_timepoint(
    stack: np.ndarray,
    nuclei: list[Nucleus],
    z_pix_res: float,
    annulus_scale: float = DEFAULT_ANNULUS_SCALE,
) -> list[tuple[int, int, int, int]]:
    """Measure every nucleus at one timepoint.

    Args:
        stack: 3D image stack for the timepoint, shape ``(Z, Y, X)``.
        nuclei: List of Nucleus objects at this timepoint.  Dead
            nuclei produce ``(0, 0, 0, 0)`` so the output length
            matches the input length (index-aligned).
        z_pix_res: Z / XY pixel resolution ratio.
        annulus_scale: Annulus outer-radius multiplier.

    Returns:
        List of ``(sum_in, count_in, sum_ann, count_ann)`` tuples,
        one per input nucleus, index-aligned with ``nuclei``.
    """
    return [
        measure_nucleus(stack, n, z_pix_res, annulus_scale)
        for n in nuclei
    ]


def measure_timepoint_with_blot(
    stack: np.ndarray,
    nuclei: list[Nucleus],
    z_pix_res: float,
    annulus_scale: float = DEFAULT_ANNULUS_SCALE,
) -> list[tuple[int, int, int, int, int, int]]:
    """Like :func:`measure_timepoint` but also computes **blot-corrected**
    annulus sums that exclude contaminating neighbor nuclei.

    The global-annulus background (``rwcorr1``) over-estimates the true
    background whenever another nucleus overlaps the annulus — the
    neighbor's bright pixels get averaged in as if they were empty
    space.  The "blot" variant, named after the historical Java
    ``rwcorr3`` slot, builds the same annulus and then **masks out the
    inner disks of every nucleus at that Z-plane** before summing.
    The resulting background estimate reflects only pixels that belong
    to no nucleus — a cleaner local-background measure in crowded
    regions.  With no neighbors nearby, blot equals the global annulus.

    Args:
        stack: 3D image stack, shape ``(Z, Y, X)``.
        nuclei: Nucleus list for this timepoint (index-aligned output).
        z_pix_res: Z / XY pixel resolution ratio.
        annulus_scale: Annulus outer-radius multiplier.

    Returns:
        Index-aligned list of
        ``(sum_in, count_in, sum_ann, count_ann, sum_blot, count_blot)``
        tuples.  ``sum_blot`` / ``count_blot`` are the neighbor-masked
        annulus aggregations.  Dead nuclei yield six zeros.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (Z,Y,X), got shape {stack.shape}")
    nz, ny, nx = stack.shape

    # Gather per-nucleus metadata once.  Dead / zero-size nuclei become
    # None so their slot in the output is all zeros.
    info: list[dict | None] = []
    for nuc in nuclei:
        if nuc.status < 1 or nuc.size <= 0:
            info.append(None)
            continue
        r = nuc.size / 2.0
        half_z = r / z_pix_res if z_pix_res > 0 else r
        z_lo = max(0, int(math.floor(nuc.z - half_z)))
        z_hi = min(nz - 1, int(math.ceil(nuc.z + half_z)))
        info.append({
            "nuc": nuc,
            "radius": r,
            "cx": int(round(nuc.x)),
            "cy": int(round(nuc.y)),
            "z_lo": z_lo,
            "z_hi": z_hi,
        })

    # Union-of-inner-disks mask per Z plane, built lazily.  Any annulus
    # pixel falling inside *any* nucleus's inner disk is excluded from
    # the blot aggregation.  (A nucleus's own inner disk is already
    # outside its annulus by construction, so the "exclude all"
    # formulation naturally excludes only neighbors.)
    plane_union_cache: dict[int, np.ndarray | None] = {}

    def _union_mask(z: int) -> np.ndarray | None:
        if z in plane_union_cache:
            return plane_union_cache[z]
        disks: list[tuple[int, int, float]] = []
        for entry in info:
            if entry is None or not (entry["z_lo"] <= z <= entry["z_hi"]):
                continue
            r_in = project_radius(
                entry["nuc"].z, z, entry["radius"], z_pix_res,
            )
            if r_in > 0:
                disks.append((entry["cx"], entry["cy"], r_in))
        if not disks:
            plane_union_cache[z] = None
            return None
        mask = np.zeros((ny, nx), dtype=bool)
        for cx, cy, r in disks:
            x0 = max(0, int(math.floor(cx - r)))
            x1 = min(nx, int(math.ceil(cx + r)) + 1)
            y0 = max(0, int(math.floor(cy - r)))
            y1 = min(ny, int(math.ceil(cy + r)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            ys = np.arange(y0, y1).reshape(-1, 1)
            xs = np.arange(x0, x1).reshape(1, -1)
            dy = ys - cy
            dx = xs - cx
            mask[y0:y1, x0:x1] |= (dx * dx + dy * dy) <= r * r
        plane_union_cache[z] = mask
        return mask

    results: list[tuple[int, int, int, int, int, int]] = []
    for entry in info:
        if entry is None:
            results.append((0, 0, 0, 0, 0, 0))
            continue

        nuc = entry["nuc"]
        cx, cy = entry["cx"], entry["cy"]
        sum_in = count_in = sum_ann = count_ann = sum_blot = count_blot = 0

        for z in range(entry["z_lo"], entry["z_hi"] + 1):
            r_in = project_radius(nuc.z, z, entry["radius"], z_pix_res)
            if r_in <= 0:
                continue
            r_out = r_in * annulus_scale

            inner, annulus, (sy, sx) = _disk_masks(
                (ny, nx), cx, cy, r_in, r_out,
            )
            if inner.size == 0:
                continue

            plane = stack[z, sy, sx]
            if inner.any():
                sum_in += int(plane[inner].sum())
                count_in += int(inner.sum())
            if annulus.any():
                sum_ann += int(plane[annulus].sum())
                count_ann += int(annulus.sum())

                union = _union_mask(z)
                if union is not None:
                    blot_mask = annulus & ~union[sy, sx]
                else:
                    blot_mask = annulus
                if blot_mask.any():
                    sum_blot += int(plane[blot_mask].sum())
                    count_blot += int(blot_mask.sum())

        results.append(
            (sum_in, count_in, sum_ann, count_ann, sum_blot, count_blot)
        )

    return results
