"""Topology-based founder cell identification — rotation-invariant.

Replaces the cardinal-direction diamond pattern approach in initial_id.py
with an algorithm that uses only division topology and relative timing
to identify the 4-cell stage founders (ABa, ABp, EMS, P2) and trace
back to P0, AB, P1.

Key insight: We do NOT need to know embryo orientation to determine
WHICH cells are which. The topological structure is sufficient:
  - P0 divides into AB + P1
  - AB divides BEFORE P1 (by 1-3 timepoints typically)
  - P1 divides into EMS + P2
  - EMS is typically larger than P2
  - AB daughters (ABa, ABp) are roughly equal size

This approach works regardless of embryo mounting orientation.

Algorithm:
  1. Find timepoint windows where exactly 4 alive cells exist
  2. For each candidate window, trace back division events
  3. Identify which pair divided first (AB daughters) vs second (P1 daughters)
  4. Within each pair, use size to distinguish EMS/P2 and position to distinguish ABa/ABp
  5. Back-trace to assign AB, P1, P0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..core.nucleus import NILLI, Nucleus

logger = logging.getLogger(__name__)

# Minimum number of frames the 4-cell stage must persist
MIN_FOUR_CELL_FRAMES = 2

# Maximum number of timepoints to search back for sister relationship.
# In real data, the 4-cell stage can persist for 30-50+ frames.
# The midpoint can be 20+ frames from the actual division events.
# Use a generous limit to ensure we always reach the division point.
MAX_SISTER_SEARCH_DEPTH = 100


@dataclass
class DivisionEvent:
    """A detected cell division event.

    Attributes:
        parent_time: 0-based timepoint of the parent (last frame before division).
        parent_idx: 0-based index of the parent nucleus at parent_time.
        daughter_time: 0-based timepoint of the daughters (first frame after division).
        daughter1_idx: 0-based index of first daughter at daughter_time.
        daughter2_idx: 0-based index of second daughter at daughter_time.
        confidence: 0-1 confidence score for this division event.
    """

    parent_time: int
    parent_idx: int
    daughter_time: int
    daughter1_idx: int
    daughter2_idx: int
    confidence: float = 1.0

    @property
    def size_ratio(self) -> float:
        """Placeholder — set externally after construction."""
        return getattr(self, "_size_ratio", 1.0)


@dataclass
class FounderAssignment:
    """Result of founder cell identification.

    Contains indices (0-based) into the nuclei_record for each identified cell,
    plus confidence scores and the determined embryo axes.
    """

    success: bool = False
    confidence: float = 0.0

    # 4-cell stage identification
    four_cell_time: int = -1  # 0-based timepoint of 4-cell stage midpoint

    # Nucleus references at four_cell_time (0-based indices)
    aba_idx: int = -1
    abp_idx: int = -1
    ems_idx: int = -1
    p2_idx: int = -1

    # Division events that created the 4-cell stage
    ab_division: DivisionEvent | None = None   # AB -> ABa + ABp
    p1_division: DivisionEvent | None = None   # P1 -> EMS + P2

    # Back-traced cells
    ab_first_time: int = -1   # First timepoint where AB exists (0-based)
    p1_first_time: int = -1   # First timepoint where P1 exists (0-based)
    p0_first_time: int = -1   # First timepoint where P0 exists (0-based)

    # Determined embryo axes (set by axes_from_founders)
    ap_vector: np.ndarray | None = None
    lr_vector: np.ndarray | None = None
    dv_vector: np.ndarray | None = None

    # Start index for canonical naming (0-based)
    start_index: int = 0

    # Warnings accumulated during identification
    warnings: list[str] = field(default_factory=list)


def identify_founders(
    nuclei_record: list[list[Nucleus]],
    starting_index: int = 0,
    ending_index: int = -1,
    z_pix_res: float = 11.1,
) -> FounderAssignment:
    """Identify founder cells using topology and division timing.

    This is the main entry point for the rotation-invariant founder
    identification algorithm.

    Args:
        nuclei_record: The full nuclei record (list of timepoints).
        starting_index: 0-based starting timepoint.
        ending_index: Ending timepoint (-1 for all).
        z_pix_res: Z pixel resolution for physical distance calculations.

    Returns:
        FounderAssignment with identification results.
    """
    result = FounderAssignment()

    if ending_index < 0:
        ending_index = len(nuclei_record)

    if not nuclei_record:
        return result

    # Step 1: Find 4-cell stage windows
    windows = _find_four_cell_windows(nuclei_record, starting_index, ending_index)

    if not windows:
        logger.warning("Could not find any 4-cell stage window")
        result.start_index = starting_index
        return result

    # Step 2: For each candidate window, try to identify founders
    best_result: FounderAssignment | None = None
    best_score = -1.0

    for first_four, last_four in windows:
        candidate = _try_identify_from_window(
            nuclei_record, first_four, last_four,
            starting_index, ending_index, z_pix_res,
        )
        if candidate.success and candidate.confidence > best_score:
            best_result = candidate
            best_score = candidate.confidence

    if best_result is not None:
        return best_result

    # Fallback: return empty result with start_index set
    logger.warning("Founder identification failed for all candidate windows")
    result.start_index = starting_index
    return result


def _is_polar_body(n: Nucleus) -> bool:
    """Check if a nucleus is a polar body.

    Checks both identity and assigned_id, since identity may have been
    cleared by _clear_all_names() before founder identification runs.
    """
    name = n.assigned_id or n.identity
    return "polar" in name.lower()


def _count_alive(nuclei: list[Nucleus]) -> int:
    """Count alive nuclei (status >= 1), excluding polar bodies."""
    count = 0
    for n in nuclei:
        if n.status >= 1 and not _is_polar_body(n):
            count += 1
    return count


def _get_alive(nuclei: list[Nucleus]) -> list[tuple[int, Nucleus]]:
    """Get alive nuclei with their 0-based indices, excluding polar bodies."""
    result = []
    for i, n in enumerate(nuclei):
        if n.status >= 1 and not _is_polar_body(n):
            result.append((i, n))
    return result


def _find_four_cell_windows(
    nuclei_record: list[list[Nucleus]],
    start_index: int,
    ending_index: int,
) -> list[tuple[int, int]]:
    """Find contiguous windows where exactly 4 alive cells exist.

    Returns a list of (first_four, last_four) 0-based timepoint pairs.
    Multiple windows can exist if the 4-cell count is interrupted.
    """
    windows: list[tuple[int, int]] = []
    first_four = -1

    for i in range(start_index, min(ending_index, len(nuclei_record))):
        cell_ct = _count_alive(nuclei_record[i])

        if cell_ct == 4:
            if first_four < 0:
                first_four = i
        else:
            if first_four >= 0:
                windows.append((first_four, i - 1))
                first_four = -1
            if cell_ct > 4:
                break  # Past the 4-cell stage

    # Close any open window
    if first_four >= 0:
        last_i = min(ending_index, len(nuclei_record)) - 1
        windows.append((first_four, last_i))

    return windows


def _try_identify_from_window(
    nuclei_record: list[list[Nucleus]],
    first_four: int,
    last_four: int,
    starting_index: int,
    ending_index: int,
    z_pix_res: float,
) -> FounderAssignment:
    """Try to identify founders from a specific 4-cell stage window.

    Args:
        nuclei_record: Full nuclei record.
        first_four: First timepoint with 4 cells (0-based).
        last_four: Last timepoint with 4 cells (0-based).
        starting_index: Dataset start.
        ending_index: Dataset end.
        z_pix_res: Z pixel resolution.

    Returns:
        FounderAssignment (success=True if identification worked).
    """
    result = FounderAssignment()
    result.four_cell_time = (first_four + last_four) // 2

    # Validate window duration
    window_len = last_four - first_four + 1
    if window_len < MIN_FOUR_CELL_FRAMES:
        result.warnings.append(
            f"4-cell window too short: {window_len} frames (need {MIN_FOUR_CELL_FRAMES})"
        )
        # Still try — just lower confidence
        confidence_penalty = 0.3
    else:
        confidence_penalty = 0.0

    # Get the 4 cells at the midpoint
    mid_time = result.four_cell_time
    alive = _get_alive(nuclei_record[mid_time])

    if len(alive) != 4:
        result.warnings.append(f"Expected 4 alive cells at t={mid_time}, found {len(alive)}")
        return result

    logger.info(
        "4-cell stage: first=%d, last=%d, mid=%d (window=%d frames)",
        first_four, last_four, mid_time, window_len,
    )

    # Step 1: Find sister pairs by tracing back predecessors
    pairs = _find_sister_pairs(nuclei_record, alive, mid_time)

    if pairs is None:
        result.warnings.append("Could not determine sister pairs at 4-cell stage")
        return result

    (pair_a, pair_a_birth), (pair_b, pair_b_birth) = pairs

    # Step 2: The pair that appeared FIRST is from AB division (AB divides before P1)
    if pair_a_birth <= pair_b_birth:
        ab_pair = pair_a
        p1_pair = pair_b
        ab_birth = pair_a_birth
        p1_birth = pair_b_birth
    else:
        ab_pair = pair_b
        p1_pair = pair_a
        ab_birth = pair_b_birth
        p1_birth = pair_a_birth

    # Confidence based on timing separation
    timing_gap = abs(p1_birth - ab_birth)
    if timing_gap == 0:
        # Can't distinguish pairs by backward tracing.  Try forward
        # division timing: cells that divide at the same time in the
        # future are likely sisters.  This is critical for datasets
        # that start at the 4-cell stage (no predecessor data).
        fwd = _forward_division_pairing(
            nuclei_record, alive, mid_time, last_four, ending_index,
        )
        if fwd is not None:
            (fwd_pair_a, fwd_div_a), (fwd_pair_b, fwd_div_b) = fwd
            fwd_gap = abs(fwd_div_b - fwd_div_a)

            # Override the pair assignment using forward division timing.
            # In C. elegans, AB daughters (ABa, ABp) divide BEFORE P1
            # daughters (EMS, P2) at the 4→8 cell transition.  So the
            # pair whose members divide FIRST is the AB pair.
            if fwd_div_a <= fwd_div_b:
                ab_pair = fwd_pair_a
                p1_pair = fwd_pair_b
            else:
                ab_pair = fwd_pair_b
                p1_pair = fwd_pair_a

            if fwd_gap >= 1:
                timing_confidence = min(1.0, 0.5 + fwd_gap * 0.1)
            else:
                timing_confidence = 0.5

            logger.info(
                "Forward division timing: AB pair div=%d, P1 pair div=%d, gap=%d",
                min(fwd_div_a, fwd_div_b), max(fwd_div_a, fwd_div_b), fwd_gap,
            )
        else:
            timing_confidence = 0.3
            result.warnings.append("AB and P1 divisions appear simultaneous — assignment uncertain")
    elif timing_gap == 1:
        timing_confidence = 0.6
    else:
        timing_confidence = min(1.0, 0.6 + timing_gap * 0.1)

    # Step 3: Within P1 pair, EMS is typically larger than P2
    (p1_d1_idx, p1_d1), (p1_d2_idx, p1_d2) = p1_pair
    if p1_d1.size >= p1_d2.size:
        ems_idx, ems_nuc = p1_d1_idx, p1_d1
        p2_idx, p2_nuc = p1_d2_idx, p1_d2
    else:
        ems_idx, ems_nuc = p1_d2_idx, p1_d2
        p2_idx, p2_nuc = p1_d1_idx, p1_d1

    # Size confidence for EMS/P2 distinction
    size_sum = max(1, ems_nuc.size + p2_nuc.size)
    size_diff = abs(ems_nuc.size - p2_nuc.size)
    size_confidence = min(1.0, 0.5 + size_diff / size_sum)

    if size_diff < 2:
        result.warnings.append(
            f"EMS/P2 size difference small ({ems_nuc.size} vs {p2_nuc.size})"
        )

    # Step 4: Within AB pair, distinguish ABa from ABp
    # ABa is more anterior — but we don't know orientation yet.
    # For now, assign arbitrarily; axis determination will fix the labels later.
    # Use x-coordinate as a provisional heuristic (will be refined).
    (ab_d1_idx, ab_d1), (ab_d2_idx, ab_d2) = ab_pair
    # Assign by x-position (smaller x = more anterior in most setups)
    if ab_d1.x <= ab_d2.x:
        aba_idx, aba_nuc = ab_d1_idx, ab_d1
        abp_idx, abp_nuc = ab_d2_idx, ab_d2
    else:
        aba_idx, aba_nuc = ab_d2_idx, ab_d2
        abp_idx, abp_nuc = ab_d1_idx, ab_d1

    # Step 5: Assign names to nuclei
    aba_nuc.identity = "ABa"
    abp_nuc.identity = "ABp"
    ems_nuc.identity = "EMS"
    p2_nuc.identity = "P2"

    result.aba_idx = aba_idx
    result.abp_idx = abp_idx
    result.ems_idx = ems_idx
    result.p2_idx = p2_idx

    logger.info(
        "4-cell ID: ABa=idx%d(x=%d), ABp=idx%d(x=%d), "
        "EMS=idx%d(size=%d), P2=idx%d(size=%d)",
        aba_idx, aba_nuc.x, abp_idx, abp_nuc.x,
        ems_idx, ems_nuc.size, p2_idx, p2_nuc.size,
    )

    # Step 6: Back-trace to assign AB, P1, P0 and name all continuation cells
    start_idx = _back_trace_founders(
        nuclei_record, mid_time, aba_nuc, abp_nuc, ems_nuc, p2_nuc,
    )
    result.start_index = start_idx

    # Step 7: Determine axes from founder positions
    result.ap_vector, result.lr_vector, result.dv_vector = _axes_from_founders(
        aba_nuc, abp_nuc, ems_nuc, p2_nuc, z_pix_res,
    )

    # Compute overall confidence
    axis_confidence = 1.0
    if result.ap_vector is None:
        axis_confidence = 0.0
        result.warnings.append("Could not determine embryo axes from founder positions")

    result.confidence = max(
        0.0,
        timing_confidence * size_confidence * axis_confidence - confidence_penalty,
    )
    result.success = True

    logger.info(
        "Founder identification succeeded (confidence=%.2f, timing_gap=%d, "
        "size_diff=%d, warnings=%d)",
        result.confidence, timing_gap, size_diff, len(result.warnings),
    )

    return result


def _find_sister_pairs(
    nuclei_record: list[list[Nucleus]],
    alive: list[tuple[int, Nucleus]],
    current_time: int,
) -> tuple[
    tuple[tuple[int, Nucleus], tuple[int, Nucleus]],
    tuple[tuple[int, Nucleus], tuple[int, Nucleus]],
] | None:
    """Find the two sister pairs among 4 cells at a timepoint.

    Two cells are sisters if they share the same predecessor (parent)
    that has two successors (i.e., the parent divided into these two cells).

    We trace backwards to find which cells share a common dividing parent.

    Returns:
        ((pair1_cell_a, pair1_cell_b), birth_time1),
        ((pair2_cell_a, pair2_cell_b), birth_time2)
        or None if pairs cannot be determined.
    """
    if len(alive) != 4:
        return None

    # Trace each cell back to find its birth time (when it was created by division)
    cell_births: list[int] = []
    cell_parents: list[tuple[int, int]] = []  # (parent_time, parent_0based_idx)

    for idx, nuc in alive:
        birth_time, parent_info = _trace_back_to_birth(
            nuclei_record, nuc, current_time,
        )
        cell_births.append(birth_time)
        cell_parents.append(parent_info)

    # Find pairs: two cells are sisters if they share the same parent
    # Try all 3 possible pairings of 4 cells: (01,23), (02,13), (03,12)
    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    best_pairing = None
    best_score = -1.0

    for (a1, a2), (b1, b2) in pairings:
        score = 0.0

        # Check if a1 and a2 share a parent
        if cell_parents[a1][0] >= 0 and cell_parents[a1] == cell_parents[a2]:
            score += 1.0
        elif cell_births[a1] == cell_births[a2]:
            # Same birth time is suggestive
            score += 0.3

        # Check if b1 and b2 share a parent
        if cell_parents[b1][0] >= 0 and cell_parents[b1] == cell_parents[b2]:
            score += 1.0
        elif cell_births[b1] == cell_births[b2]:
            score += 0.3

        if score > best_score:
            best_score = score
            best_pairing = ((a1, a2), (b1, b2))

    if best_pairing is None or best_score < 0.3:
        # Try fallback: use birth times to group cells
        return _fallback_pairing_by_birth(alive, cell_births)

    (a1, a2), (b1, b2) = best_pairing
    pair_a_birth = min(cell_births[a1], cell_births[a2])
    pair_b_birth = min(cell_births[b1], cell_births[b2])

    return (
        ((alive[a1], alive[a2]), pair_a_birth),
        ((alive[b1], alive[b2]), pair_b_birth),
    )


def _fallback_pairing_by_birth(
    alive: list[tuple[int, Nucleus]],
    cell_births: list[int],
) -> tuple | None:
    """Fallback: pair cells by matching birth times.

    If two cells were born at the same time, they're likely sisters.
    """
    # Sort by birth time
    indexed = sorted(range(4), key=lambda i: cell_births[i])

    # The two earliest-born should be one pair, the two latest another
    a1, a2 = indexed[0], indexed[1]
    b1, b2 = indexed[2], indexed[3]

    pair_a_birth = min(cell_births[a1], cell_births[a2])
    pair_b_birth = min(cell_births[b1], cell_births[b2])

    return (
        ((alive[a1], alive[a2]), pair_a_birth),
        ((alive[b1], alive[b2]), pair_b_birth),
    )


def _forward_division_pairing(
    nuclei_record: list[list[Nucleus]],
    alive: list[tuple[int, Nucleus]],
    mid_time: int,
    last_four: int,
    ending_index: int,
) -> tuple | None:
    """Determine sister pairs by looking FORWARD at when cells divide.

    When backward tracing fails (dataset starts at 4-cell stage), we look
    at which cells divide first after the 4-cell window ends.  Cells that
    divide at the same time are likely sisters (from the same parent).

    Returns the same format as _find_sister_pairs, or None if not enough
    division events are found.
    """
    if len(alive) != 4:
        return None

    # For each cell, find when it next divides (successor2 != NILLI)
    div_times: list[int] = []
    for idx, nuc in alive:
        div_t = _trace_forward_to_division(nuclei_record, nuc, mid_time, ending_index)
        div_times.append(div_t)

    logger.debug("Forward division times: %s", div_times)

    # We need at least 2 cells to have divided
    dividing = [(i, dt) for i, dt in enumerate(div_times) if dt < ending_index]
    if len(dividing) < 2:
        return None

    # Sort by division time
    dividing.sort(key=lambda x: x[1])

    # Group cells that divide at the same time (within 1 frame)
    groups: list[list[int]] = []
    current_group: list[int] = [dividing[0][0]]
    current_dt = dividing[0][1]
    for i, dt in dividing[1:]:
        if abs(dt - current_dt) <= 1:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
            current_dt = dt
    groups.append(current_group)

    # Best case: two groups of 2 cells each
    if len(groups) >= 2 and len(groups[0]) >= 2 and len(groups[1]) >= 2:
        g1 = groups[0][:2]
        g2 = groups[1][:2]
        pair_a_div = div_times[g1[0]]
        pair_b_div = div_times[g2[0]]
        return (
            ((alive[g1[0]], alive[g1[1]]), pair_a_div),
            ((alive[g2[0]], alive[g2[1]]), pair_b_div),
        )

    # Fallback: first 2 dividing cells are one pair, remaining 2 are the other
    if len(dividing) >= 2:
        first_pair = [dividing[0][0], dividing[1][0]]
        remaining = [i for i in range(4) if i not in first_pair]
        if len(remaining) == 2:
            pair_a_div = div_times[first_pair[0]]
            pair_b_div = max(div_times[remaining[0]], div_times[remaining[1]])
            return (
                ((alive[first_pair[0]], alive[first_pair[1]]), pair_a_div),
                ((alive[remaining[0]], alive[remaining[1]]), pair_b_div),
            )

    return None


def _trace_forward_to_division(
    nuclei_record: list[list[Nucleus]],
    nuc: Nucleus,
    current_time: int,
    ending_index: int,
) -> int:
    """Trace a nucleus forward to find when it divides (successor2 != NILLI).

    Returns the 0-based timepoint of the division, or ending_index if
    the cell never divides within the search range.
    """
    t = current_time
    current = nuc

    for _ in range(MAX_SISTER_SEARCH_DEPTH):
        if current.successor2 != NILLI:
            return t  # This cell divides at timepoint t

        if current.successor1 == NILLI or t + 1 >= ending_index:
            return ending_index  # Doesn't divide

        t += 1
        next_nuclei = nuclei_record[t]
        succ_idx = current.successor1 - 1  # 1-based to 0-based

        if not (0 <= succ_idx < len(next_nuclei)):
            return ending_index

        current = next_nuclei[succ_idx]

    return ending_index


def _trace_back_to_birth(
    nuclei_record: list[list[Nucleus]],
    nuc: Nucleus,
    current_time: int,
) -> tuple[int, tuple[int, int]]:
    """Trace a nucleus backwards to find when it was born (by division).

    Returns:
        (birth_time, (parent_time, parent_0based_idx))
        birth_time: 0-based timepoint when this cell first appeared
        parent_info: (-1, -1) if no dividing parent found
    """
    t = current_time
    current = nuc

    for _ in range(MAX_SISTER_SEARCH_DEPTH):
        if t <= 0 or current.predecessor == NILLI:
            return t, (-1, -1)

        prev_nuclei = nuclei_record[t - 1]
        pred_idx = current.predecessor - 1  # 1-based to 0-based

        if not (0 <= pred_idx < len(prev_nuclei)):
            return t, (-1, -1)

        pred = prev_nuclei[pred_idx]

        # If predecessor has two successors, this cell was born by division at time t
        if pred.successor2 != NILLI:
            return t, (t - 1, pred_idx)

        # Otherwise, continue tracing back
        t -= 1
        current = pred

    return t, (-1, -1)


def _back_trace_founders(
    nuclei_record: list[list[Nucleus]],
    four_cells_time: int,
    aba_nuc: Nucleus,
    abp_nuc: Nucleus,
    ems_nuc: Nucleus,
    p2_nuc: Nucleus,
) -> int:
    """Trace backwards from 4-cell stage to assign AB, P1, P0.

    Traces ALL four founder cells backward to their birth, naming their
    continuation cells and parent cells (AB, P1, P0).

    Returns:
        The 0-based index of the first timepoint where we have named cells.
    """
    start_index = four_cells_time

    # --- Trace ABa back to find AB ---
    ab_division_time = -1
    ab_nuc = None
    t = four_cells_time
    current = aba_nuc
    while t > 0 and current.predecessor != NILLI:
        t -= 1
        prev_nuclei = nuclei_record[t]
        pred_idx = current.predecessor - 1
        if not (0 <= pred_idx < len(prev_nuclei)):
            break

        pred = prev_nuclei[pred_idx]
        if pred.successor2 != NILLI:
            # This is AB (parent of ABa and ABp)
            pred.identity = "AB"
            start_index = min(start_index, t)
            ab_division_time = t
            ab_nuc = pred
            break
        else:
            pred.identity = current.identity  # inherit name backward
            current = pred
            start_index = min(start_index, t)

    # --- Trace ABp back (should reach same AB division) ---
    t2 = four_cells_time
    current2 = abp_nuc
    while t2 > 0 and current2.predecessor != NILLI:
        t2 -= 1
        prev_nuclei = nuclei_record[t2]
        pred_idx = current2.predecessor - 1
        if not (0 <= pred_idx < len(prev_nuclei)):
            break

        pred = prev_nuclei[pred_idx]
        if pred.successor2 != NILLI:
            if ab_nuc is not None and pred.identity == "AB":
                # ABa trace already confirmed this as AB — stop here
                break
            # If ABa trace did NOT find AB, this "division" is a data artifact
            # (e.g., multiple nuclei pointing to the same predecessor).
            # Continue tracing backward and name as ABp continuation.
            if not pred.identity:
                pred.identity = current2.identity
            current2 = pred
            start_index = min(start_index, t2)
        else:
            if not pred.identity:
                pred.identity = current2.identity
            current2 = pred
            start_index = min(start_index, t2)

    # --- Trace AB back to find P0 ---
    if ab_nuc is not None:
        ab = ab_nuc
        t = ab_division_time
        while t > 0 and ab.predecessor != NILLI:
            t -= 1
            prev_nuclei = nuclei_record[t]
            pred_idx = ab.predecessor - 1
            if not (0 <= pred_idx < len(prev_nuclei)):
                break

            pred = prev_nuclei[pred_idx]
            if pred.successor2 != NILLI:
                pred.identity = "P0"
                start_index = min(start_index, t)

                # Name the sister cell (P1)
                s1_idx = pred.successor1 - 1
                s2_idx = pred.successor2 - 1
                next_nuclei = nuclei_record[t + 1]
                ab_idx_in_next = next_nuclei.index(ab) if ab in next_nuclei else -1

                if ab_idx_in_next == s1_idx and 0 <= s2_idx < len(next_nuclei):
                    next_nuclei[s2_idx].identity = "P1"
                elif ab_idx_in_next == s2_idx and 0 <= s1_idx < len(next_nuclei):
                    next_nuclei[s1_idx].identity = "P1"

                # Continue tracing P0 backward through its continuation cells
                p0 = pred
                tp = t
                while tp > 0 and p0.predecessor != NILLI:
                    tp -= 1
                    prev2 = nuclei_record[tp]
                    pi = p0.predecessor - 1
                    if not (0 <= pi < len(prev2)):
                        break
                    prev2[pi].identity = "P0"
                    start_index = min(start_index, tp)
                    p0 = prev2[pi]

                break
            else:
                pred.identity = ab.identity
                ab = pred
                start_index = min(start_index, t)

    # --- Trace EMS back to name P1 ---
    p1_nuc = None
    t3 = four_cells_time
    current3 = ems_nuc
    while t3 > 0 and current3.predecessor != NILLI:
        t3 -= 1
        prev_nuclei = nuclei_record[t3]
        pred_idx = current3.predecessor - 1
        if not (0 <= pred_idx < len(prev_nuclei)):
            break

        pred = prev_nuclei[pred_idx]
        if pred.successor2 != NILLI:
            if not pred.identity:
                pred.identity = "P1"
            p1_nuc = pred
            start_index = min(start_index, t3)
            break
        else:
            if not pred.identity:
                pred.identity = current3.identity
            current3 = pred
            start_index = min(start_index, t3)

    # --- Trace P2 back (should reach same P1 division) ---
    t4 = four_cells_time
    current4 = p2_nuc
    while t4 > 0 and current4.predecessor != NILLI:
        t4 -= 1
        prev_nuclei = nuclei_record[t4]
        pred_idx = current4.predecessor - 1
        if not (0 <= pred_idx < len(prev_nuclei)):
            break

        pred = prev_nuclei[pred_idx]
        if pred.successor2 != NILLI:
            if p1_nuc is not None and pred.identity == "P1":
                # EMS trace already confirmed this as P1 — stop here
                break
            # If EMS trace did NOT find P1, this "division" is a data artifact.
            # Continue tracing backward and name as P2 continuation.
            if not pred.identity:
                pred.identity = current4.identity
            current4 = pred
            start_index = min(start_index, t4)
        else:
            if not pred.identity:
                pred.identity = current4.identity
            current4 = pred
            start_index = min(start_index, t4)

    return start_index


def _axes_from_founders(
    aba: Nucleus,
    abp: Nucleus,
    ems: Nucleus,
    p2: Nucleus,
    z_pix_res: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Determine embryo axes from the 4 founder cell positions.

    Derives AP, LR, DV axes directly from cell geometry without
    requiring AuxInfo or external orientation measurements.

    The biological basis:
    - AP axis: P2 (posterior) -> AB centroid (anterior)
    - LR axis: perpendicular to AP, in the ABa-ABp separation plane
    - DV axis: cross(AP, LR)

    Args:
        aba, abp, ems, p2: The four identified founder cells.
        z_pix_res: Z pixel resolution for converting to physical coords.

    Returns:
        (ap_vector, lr_vector, dv_vector) as unit vectors, or (None, None, None)
        if the geometry is degenerate.
    """
    # Get positions in physical coordinates (z scaled)
    pos_aba = np.array([float(aba.x), float(aba.y), float(aba.z) * z_pix_res])
    pos_abp = np.array([float(abp.x), float(abp.y), float(abp.z) * z_pix_res])
    pos_ems = np.array([float(ems.x), float(ems.y), float(ems.z) * z_pix_res])
    pos_p2 = np.array([float(p2.x), float(p2.y), float(p2.z) * z_pix_res])

    # AP axis: posterior (P2) -> anterior (AB midpoint)
    ab_center = (pos_aba + pos_abp) / 2.0
    ap_raw = ab_center - pos_p2
    ap_norm = np.linalg.norm(ap_raw)

    if ap_norm < 1e-6:
        logger.warning("AP axis degenerate (P2 and AB centroid coincide)")
        return None, None, None

    ap_vector = ap_raw / ap_norm

    # ABa-ABp separation vector
    ab_sep = pos_aba - pos_abp
    # Project out the AP component to get the component in the LR+DV plane
    ab_sep_perp = ab_sep - np.dot(ab_sep, ap_vector) * ap_vector
    ab_sep_norm = np.linalg.norm(ab_sep_perp)

    if ab_sep_norm < 1e-6:
        # ABa and ABp have the same projection perpendicular to AP
        # Fall back to EMS-P2 separation for LR determination
        ep_sep = pos_ems - pos_p2
        ep_sep_perp = ep_sep - np.dot(ep_sep, ap_vector) * ap_vector
        ab_sep_perp = ep_sep_perp
        ab_sep_norm = np.linalg.norm(ab_sep_perp)

        if ab_sep_norm < 1e-6:
            logger.warning("Cannot determine LR axis — cells are collinear")
            return ap_vector, None, None

    # LR axis: we define it as perpendicular to AP in the ABa-ABp plane
    # Convention: ABa is on the left. The LR vector points from right to left.
    # cross(AP, ab_sep_perp) gives DV, then cross(AP, DV) gives LR
    # Or equivalently: normalize ab_sep_perp → that's a proxy for LR
    # But we need to ensure right-handedness.
    dv_vector = np.cross(ap_vector, ab_sep_perp)
    dv_norm = np.linalg.norm(dv_vector)
    if dv_norm < 1e-6:
        logger.warning("DV axis degenerate")
        return ap_vector, None, None

    dv_vector = dv_vector / dv_norm

    # LR is the remaining axis
    lr_vector = np.cross(dv_vector, ap_vector)
    lr_norm = np.linalg.norm(lr_vector)
    if lr_norm < 1e-6:
        logger.warning("LR axis degenerate")
        return ap_vector, None, None

    lr_vector = lr_vector / lr_norm

    # Ensure ABa is on the "left" side (positive LR projection)
    aba_lr = np.dot(pos_aba - ab_center, lr_vector)
    if aba_lr < 0:
        lr_vector = -lr_vector
        dv_vector = -dv_vector  # Maintain right-handedness

    logger.info(
        "Axes from founders: AP=%s, LR=%s, DV=%s",
        np.round(ap_vector, 3),
        np.round(lr_vector, 3),
        np.round(dv_vector, 3),
    )

    return ap_vector, lr_vector, dv_vector
