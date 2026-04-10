"""Per-timepoint body axis estimation from lineage-based cell centroids.

Computes AP, LR, and DV axes at each timepoint using the spatial
distribution of cells grouped by their lineage membership (ABa-lineage,
ABp-lineage, P1-lineage).  Because the axes are re-derived at every
timepoint from the *current* cell positions, this approach is inherently
robust to global embryo rotations around the AP axis that can occur
during imaging of compressed embryos.

Algorithm:
    1. build_lineage_map() — forward-propagate founder identity through
       successor chains so every nucleus is labelled ABa/ABp/EMS/P2.
    2. compute_local_axes() — at a given timepoint, collect centroids of
       ABa-lineage and ABp-lineage cells to derive LR, and centroids of
       AB-lineage vs P1-lineage cells to derive AP.
    3. The DivisionCaller uses compute_local_axes() at each division
       event instead of a single fixed rotation.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.nucleus import NILLI, Nucleus

logger = logging.getLogger(__name__)

# Lineage labels
LINEAGE_ABa = "ABa"
LINEAGE_ABp = "ABp"
LINEAGE_EMS = "EMS"
LINEAGE_P2 = "P2"


def build_lineage_map(
    nuclei_record: list[list[Nucleus]],
    four_cell_time: int,
    aba_idx: int,
    abp_idx: int,
    ems_idx: int,
    p2_idx: int,
) -> list[list[str]]:
    """Build a per-nucleus lineage label array.

    Forward-propagates lineage membership from the 4 founder cells
    through successor chains.  Each entry is one of 'ABa', 'ABp',
    'EMS', 'P2', or '' (unlabelled).

    Args:
        nuclei_record: Full nuclei record.
        four_cell_time: 0-based timepoint at which the 4 founders
            are identified (midpoint of the 4-cell stage).
        aba_idx, abp_idx, ems_idx, p2_idx: 0-based indices of the
            founder nuclei at *four_cell_time*.

    Returns:
        A list-of-lists parallel to nuclei_record, where
        lineage_map[t][j] is the lineage label for nuclei_record[t][j].
    """
    n_timepoints = len(nuclei_record)
    lineage_map: list[list[str]] = [
        [""] * len(nuclei_record[t]) for t in range(n_timepoints)
    ]

    # Seed the founders
    if four_cell_time < n_timepoints:
        nucs = nuclei_record[four_cell_time]
        if aba_idx < len(nucs):
            lineage_map[four_cell_time][aba_idx] = LINEAGE_ABa
        if abp_idx < len(nucs):
            lineage_map[four_cell_time][abp_idx] = LINEAGE_ABp
        if ems_idx < len(nucs):
            lineage_map[four_cell_time][ems_idx] = LINEAGE_EMS
        if p2_idx < len(nucs):
            lineage_map[four_cell_time][p2_idx] = LINEAGE_P2

    # Back-propagate: from four_cell_time backwards to t=0
    for t in range(four_cell_time, 0, -1):
        for j, nuc in enumerate(nuclei_record[t]):
            label = lineage_map[t][j]
            if not label:
                continue
            pred = nuc.predecessor
            if pred == NILLI:
                continue
            pred_idx = pred - 1  # 1-based to 0-based
            if 0 <= pred_idx < len(nuclei_record[t - 1]):
                prev_label = lineage_map[t - 1][pred_idx]
                if not prev_label:
                    lineage_map[t - 1][pred_idx] = label

    # Forward-propagate: from four_cell_time to end via successor chains
    for t in range(four_cell_time, n_timepoints - 1):
        for j, nuc in enumerate(nuclei_record[t]):
            label = lineage_map[t][j]
            if not label:
                continue
            # Propagate to successor1
            if nuc.successor1 > 0:
                s_idx = nuc.successor1 - 1
                if 0 <= s_idx < len(nuclei_record[t + 1]):
                    lineage_map[t + 1][s_idx] = label
            # Propagate to successor2 (division — both daughters inherit)
            if nuc.successor2 > 0:
                s_idx = nuc.successor2 - 1
                if 0 <= s_idx < len(nuclei_record[t + 1]):
                    lineage_map[t + 1][s_idx] = label

    return lineage_map


def compute_local_axes(
    nuclei_record: list[list[Nucleus]],
    lineage_map: list[list[str]],
    t: int,
    z_pix_res: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
    """Compute body axes at timepoint *t* from lineage centroids.

    Args:
        nuclei_record: Full nuclei record.
        lineage_map: Output of build_lineage_map().
        t: 0-based timepoint.
        z_pix_res: Z pixel resolution (z_res / xy_res).

    Returns:
        (ap_vec, lr_vec, dv_vec, lr_quality) as unit vectors in the lab
        frame plus a quality metric for the LR axis (0-1).  Returns
        (None, None, None, 0.0) if there aren't enough labelled cells.

        *lr_quality* is the fraction of the ABa-ABp separation that is
        perpendicular to AP.  When ABa and ABp centroids are nearly
        collinear with AP this ratio approaches 0 and the LR axis is
        unreliable.
    """
    if t >= len(nuclei_record) or t >= len(lineage_map):
        return None, None, None, 0.0

    nucs = nuclei_record[t]
    labels = lineage_map[t]

    # Collect positions grouped by lineage
    ab_positions: list[np.ndarray] = []   # ABa + ABp descendants
    p1_positions: list[np.ndarray] = []   # EMS + P2 descendants
    aba_positions: list[np.ndarray] = []
    abp_positions: list[np.ndarray] = []

    for j, nuc in enumerate(nucs):
        if nuc.status < 1:
            continue
        if j >= len(labels) or not labels[j]:
            continue
        pos = np.array([float(nuc.x), float(nuc.y), float(nuc.z) * z_pix_res])
        label = labels[j]
        if label == LINEAGE_ABa:
            aba_positions.append(pos)
            ab_positions.append(pos)
        elif label == LINEAGE_ABp:
            abp_positions.append(pos)
            ab_positions.append(pos)
        elif label in (LINEAGE_EMS, LINEAGE_P2):
            p1_positions.append(pos)

    # Need at least 1 cell in each group.
    if not ab_positions or not p1_positions:
        return None, None, None, 0.0
    if not aba_positions or not abp_positions:
        return None, None, None, 0.0

    ab_centroid = np.mean(ab_positions, axis=0)
    p1_centroid = np.mean(p1_positions, axis=0)
    aba_centroid = np.mean(aba_positions, axis=0)
    abp_centroid = np.mean(abp_positions, axis=0)

    # AP: posterior (P1) -> anterior (AB)
    ap_raw = ab_centroid - p1_centroid
    ap_norm = np.linalg.norm(ap_raw)
    if ap_norm < 1e-6:
        return None, None, None, 0.0
    ap_vec = ap_raw / ap_norm

    # LR: ABp-centroid -> ABa-centroid, projected perpendicular to AP
    lr_raw = aba_centroid - abp_centroid
    lr_total = np.linalg.norm(lr_raw)
    lr_perp = lr_raw - np.dot(lr_raw, ap_vec) * ap_vec
    lr_norm = np.linalg.norm(lr_perp)

    # LR quality: fraction of ABa-ABp separation that is perpendicular to AP
    lr_quality = lr_norm / lr_total if lr_total > 1e-6 else 0.0

    if lr_norm < 1e-6:
        return ap_vec, None, None, 0.0
    lr_vec = lr_perp / lr_norm

    # DV: completes right-handed frame
    dv_vec = np.cross(ap_vec, lr_vec)
    dv_norm = np.linalg.norm(dv_vec)
    if dv_norm < 1e-6:
        return ap_vec, None, None, 0.0
    dv_vec = dv_vec / dv_norm

    return ap_vec, lr_vec, dv_vec, lr_quality


def check_axis_continuity(
    current: tuple[np.ndarray, np.ndarray, np.ndarray],
    previous: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure axis continuity between consecutive timepoints.

    If the AP or LR axis flips by more than 90 degrees compared to the
    previous timepoint, negate the axis to maintain consistent orientation.
    This handles cases where the centroid-based computation produces
    an arbitrary sign flip.

    Args:
        current: (ap, lr, dv) axes at the current timepoint.
        previous: (ap, lr, dv) axes at the previous timepoint.

    Returns:
        Corrected (ap, lr, dv) with consistent orientation.
    """
    ap, lr, dv = current
    prev_ap, prev_lr, prev_dv = previous

    # Check AP axis continuity
    if np.dot(ap, prev_ap) < 0:
        ap = -ap
        # Flipping AP requires flipping one other axis to maintain handedness
        dv = -dv
        logger.debug("AP axis flip corrected at current timepoint")

    # Check LR axis continuity
    if np.dot(lr, prev_lr) < 0:
        lr = -lr
        dv = -dv  # maintain right-handedness
        logger.debug("LR axis flip corrected at current timepoint")

    return ap, lr, dv


def axes_to_canonical(
    da: np.ndarray,
    ap_vec: np.ndarray,
    lr_vec: np.ndarray,
    dv_vec: np.ndarray,
) -> np.ndarray:
    """Transform a lab-frame vector into the canonical frame.

    Canonical frame convention:
        AP -> (-1, 0, 0)
        DV -> ( 0, 1, 0)
        LR -> ( 0, 0, 1)

    Args:
        da: Vector in lab frame (already z-scaled).
        ap_vec, lr_vec, dv_vec: Unit basis vectors in the lab frame.

    Returns:
        Vector in the canonical frame.
    """
    ap_component = np.dot(da, ap_vec)
    lr_component = np.dot(da, lr_vec)
    dv_component = np.dot(da, dv_vec)

    # Map to canonical: AP -> -x, DV -> y, LR -> z
    return np.array([-ap_component, dv_component, lr_component])
