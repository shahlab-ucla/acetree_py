"""Per-timepoint body axis estimation from lineage-based cell centroids.

Computes AP, DV, and LR axes at each timepoint using the spatial
distribution of cells grouped by their four-cell lineage membership.
Because the axes are re-derived at every
timepoint from the *current* cell positions, this approach is inherently
robust to global embryo rotations around the AP axis that can occur
during imaging of compressed embryos.

Algorithm:
    1. build_lineage_map() — forward-propagate founder identity through
       successor chains so every nucleus is labelled ABa/ABp/EMS/P2.
    2. compute_local_axes() — at a given timepoint, use P2→ABa for AP and
       EMS→ABp for DV, then complete a right-handed frame for LR.
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


def _validated_outgoing_children(
    current: list[Nucleus],
    following: list[Nucleus],
) -> list[tuple[int, ...] | None]:
    """Return exact reciprocal child indices, or ``None`` for malformed links.

    Lineage centroids are anatomical evidence, so they must not be populated
    from a partially valid edge set.  Duplicate successor slots and undeclared
    live reverse claimers invalidate the whole parent's outgoing relationship.
    """
    declared_claimers: dict[int, set[int]] = {}
    for parent_index, parent in enumerate(current):
        if not parent.is_alive:
            continue
        for successor in (parent.successor1, parent.successor2):
            child_index = successor - 1
            if successor > 0 and 0 <= child_index < len(following):
                declared_claimers.setdefault(child_index, set()).add(parent_index)

    result: list[tuple[int, ...] | None] = []
    for parent_index, parent in enumerate(current):
        if not parent.is_alive:
            result.append(None)
            continue

        raw_successors = (parent.successor1, parent.successor2)
        positive = tuple(successor for successor in raw_successors if successor > 0)
        child_indices = tuple(successor - 1 for successor in positive)
        reverse_live = {
            child_index
            for child_index, child in enumerate(following)
            if child.is_alive and child.predecessor == parent_index + 1
        }

        malformed = (
            (parent.successor1 <= 0 < parent.successor2)
            or len(set(positive)) != len(positive)
            or any(not (0 <= child_index < len(following)) for child_index in child_indices)
            or set(child_indices) != reverse_live
        )
        if not malformed:
            malformed = any(
                not following[child_index].is_alive
                or following[child_index].predecessor != parent_index + 1
                or declared_claimers.get(child_index) != {parent_index}
                for child_index in child_indices
            )
        result.append(None if malformed else child_indices)
    return result


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
        if 0 <= aba_idx < len(nucs) and nucs[aba_idx].is_alive:
            lineage_map[four_cell_time][aba_idx] = LINEAGE_ABa
        if 0 <= abp_idx < len(nucs) and nucs[abp_idx].is_alive:
            lineage_map[four_cell_time][abp_idx] = LINEAGE_ABp
        if 0 <= ems_idx < len(nucs) and nucs[ems_idx].is_alive:
            lineage_map[four_cell_time][ems_idx] = LINEAGE_EMS
        if 0 <= p2_idx < len(nucs) and nucs[p2_idx].is_alive:
            lineage_map[four_cell_time][p2_idx] = LINEAGE_P2

    # Back-propagate: from four_cell_time backwards to t=0
    for t in range(four_cell_time, 0, -1):
        validated_children = _validated_outgoing_children(
            nuclei_record[t - 1], nuclei_record[t],
        )
        for j, nuc in enumerate(nuclei_record[t]):
            label = lineage_map[t][j]
            if not label or not nuc.is_alive:
                continue
            pred = nuc.predecessor
            if pred == NILLI:
                continue
            pred_idx = pred - 1  # 1-based to 0-based
            if 0 <= pred_idx < len(nuclei_record[t - 1]):
                predecessor = nuclei_record[t - 1][pred_idx]
                if (
                    not predecessor.is_alive
                    or validated_children[pred_idx] is None
                    or j not in validated_children[pred_idx]
                ):
                    continue
                prev_label = lineage_map[t - 1][pred_idx]
                if not prev_label:
                    lineage_map[t - 1][pred_idx] = label

    # Forward-propagate: from four_cell_time to end via successor chains
    for t in range(four_cell_time, n_timepoints - 1):
        current = nuclei_record[t]
        following = nuclei_record[t + 1]
        validated_children = _validated_outgoing_children(current, following)
        for j, nuc in enumerate(nuclei_record[t]):
            label = lineage_map[t][j]
            if not label or not nuc.is_alive:
                continue
            child_indices = validated_children[j]
            if child_indices is None:
                continue
            for child_index in child_indices:
                existing = lineage_map[t + 1][child_index]
                if not existing or existing == label:
                    lineage_map[t + 1][child_index] = label
                else:
                    # A conflicting lineage claim is malformed input; keep the
                    # row out of anatomical centroid inference.
                    lineage_map[t + 1][child_index] = ""

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
        (ap_vec, lr_vec, dv_vec, secondary_quality) as unit vectors in the lab
        frame plus a quality metric for the DV landmark geometry (0-1).  Returns
        (None, None, None, 0.0) if there aren't enough labelled cells.

        *secondary_quality* is the fraction of the EMS-ABp separation that is
        perpendicular to AP.  When those centroids are nearly collinear with
        AP, both the DV estimate and the derived LR axis are unreliable.
    """
    if t >= len(nuclei_record) or t >= len(lineage_map):
        return None, None, None, 0.0

    nucs = nuclei_record[t]
    labels = lineage_map[t]

    # Collect positions grouped by four-cell lineage.  Keeping the four
    # landmarks separate matters: ABa/ABp do not form the LR axis.
    aba_positions: list[np.ndarray] = []
    abp_positions: list[np.ndarray] = []
    ems_positions: list[np.ndarray] = []
    p2_positions: list[np.ndarray] = []

    for j, nuc in enumerate(nucs):
        if nuc.status < 1:
            continue
        if j >= len(labels) or not labels[j]:
            continue
        pos = np.array([float(nuc.x), float(nuc.y), float(nuc.z) * z_pix_res])
        label = labels[j]
        if label == LINEAGE_ABa:
            aba_positions.append(pos)
        elif label == LINEAGE_ABp:
            abp_positions.append(pos)
        elif label == LINEAGE_EMS:
            ems_positions.append(pos)
        elif label == LINEAGE_P2:
            p2_positions.append(pos)

    # Need at least one descendant in every landmark lineage.
    if not aba_positions or not abp_positions or not ems_positions or not p2_positions:
        return None, None, None, 0.0

    aba_centroid = np.mean(aba_positions, axis=0)
    abp_centroid = np.mean(abp_positions, axis=0)
    ems_centroid = np.mean(ems_positions, axis=0)
    p2_centroid = np.mean(p2_positions, axis=0)

    # AP: posterior (P2 lineage) -> anterior (ABa lineage)
    ap_raw = aba_centroid - p2_centroid
    ap_norm = np.linalg.norm(ap_raw)
    if ap_norm < 1e-6:
        return None, None, None, 0.0
    ap_vec = ap_raw / ap_norm

    # DV: ventral (EMS lineage) -> dorsal (ABp lineage), projected
    # perpendicular to AP.  The quality is the usable perpendicular fraction.
    dv_raw = abp_centroid - ems_centroid
    dv_total = np.linalg.norm(dv_raw)
    dv_perp = dv_raw - np.dot(dv_raw, ap_vec) * ap_vec
    dv_norm = np.linalg.norm(dv_perp)
    secondary_quality = dv_norm / dv_total if dv_total > 1e-6 else 0.0

    if dv_norm < 1e-6:
        return ap_vec, None, None, 0.0
    dv_vec = dv_perp / dv_norm

    # LR completes the right-handed anatomical frame.  With the canonical
    # convention AP=-X and DV=+Y, cross(DV, AP)=+Z (left).
    lr_vec = np.cross(dv_vec, ap_vec)
    lr_norm = np.linalg.norm(lr_vec)
    if lr_norm < 1e-6:
        return ap_vec, None, None, 0.0
    lr_vec = lr_vec / lr_norm

    return ap_vec, lr_vec, dv_vec, secondary_quality


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
