"""Early cell identification — finds P0, AB, P1, EMS, P2 in the lineage.

This module identifies the first few cells in the C. elegans embryo by
finding the 4-cell stage and using geometric relationships (the "diamond"
pattern of ABa, ABp, EMS, P2) along with division timing to assign
canonical names.

Ported from: org.rhwlab.snight.InitialID

Algorithm overview:
  1. Find the 4-cell stage timepoints (first_four..last_four)
  2. At the mid-point of the 4-cell stage, identify the 4 cells
  3. alignDiamond(): rotate the 4 cells into canonical orientation,
     assign N/S/E/W positions
  4. fourCellIDAssignment(): use division timing to determine which
     pair is ABa/ABp and which is EMS/P2
  5. backAssignment(): trace predecessors back to assign AB, P1, P0
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..core.nucleus import NILLI, Nucleus
from .canonical_transform import CanonicalTransform

logger = logging.getLogger(__name__)

# Naming constants
NUC = "Nuc"
POLAR = "polar"


class InitialIDResult:
    """Result of initial cell identification.

    Attributes:
        success: True if 4-cell stage was found and identified.
        axis_found: True if an orientation axis was determined.
        start_index: 0-based index of the timepoint to start canonical naming from.
        nuc_count: Running counter for unnamed nuclei.
        orientation: The determined AP/DV/LR orientation parameters.
    """

    def __init__(self) -> None:
        self.success: bool = False
        self.axis_found: bool = False
        self.start_index: int = 0
        self.nuc_count: int = 1
        # Orientation parameters (set by alignDiamond)
        self.ap: int = 0  # +1 or -1
        self.dv: int = 0
        self.lr: int = 0


def _count_alive(nuclei: list[Nucleus]) -> int:
    """Count alive nuclei (status >= 1), excluding polar bodies."""
    count = 0
    for n in nuclei:
        if n.status >= 1 and POLAR not in n.identity.lower():
            count += 1
    return count


def _find_four_cell_stage(
    nuclei_record: list[list[Nucleus]],
    start_index: int,
    ending_index: int,
) -> tuple[int, int]:
    """Find the first and last timepoints with exactly 4 alive cells.

    Returns:
        (first_four, last_four) as 0-based indices, or (-1, -1) if not found.
    """
    first_four = -1
    last_four = -1

    for i in range(start_index, min(ending_index, len(nuclei_record))):
        nuclei = nuclei_record[i]
        cell_ct = _count_alive(nuclei)

        if cell_ct > 4:
            break
        if cell_ct == 4:
            if first_four < 0:
                first_four = i
            last_four = i

    return first_four, last_four


def identify_initial_cells(
    nuclei_record: list[list[Nucleus]],
    starting_index: int = 0,
    ending_index: int = -1,
    canonical_transform: CanonicalTransform | None = None,
    angle: float = 0.0,
    z_pix_res: float = 11.1,
) -> InitialIDResult:
    """Identify the early cells (P0, AB, P1, EMS, P2) in the lineage.

    This is the main entry point, corresponding to InitialID.initialID() in Java.

    Args:
        nuclei_record: The full nuclei record (list of timepoints).
        starting_index: 0-based starting timepoint index.
        ending_index: Ending timepoint index (-1 for all).
        canonical_transform: For v2 embryos, the CanonicalTransform.
        angle: For v1 embryos, the rotation angle in radians.
        z_pix_res: Z pixel resolution.

    Returns:
        InitialIDResult with identification results.
    """
    result = InitialIDResult()

    if ending_index < 0:
        ending_index = len(nuclei_record)

    if not nuclei_record:
        return result

    # Check initial cell count
    nuclei = nuclei_record[starting_index]
    cell_ct = _count_alive(nuclei)

    if cell_ct > 4:
        # Too many cells at start — can't do initial ID
        logger.info("Starting with >4 cells (%d). Assigning generic names.", cell_ct)
        nuc_count = 1
        for n in nuclei:
            if n.status < 1:
                continue
            if POLAR in n.identity.lower():
                continue
            n.identity = f"{NUC}{nuc_count}"
            nuc_count += 1
        result.nuc_count = nuc_count
        result.start_index = 0
        return result

    # Find 4-cell stage
    first_four, last_four = _find_four_cell_stage(
        nuclei_record, starting_index, ending_index
    )

    if first_four < 0:
        logger.warning("Could not find 4-cell stage")
        result.start_index = starting_index
        return result

    # Use midpoint of 4-cell stage
    four_cells = (first_four + last_four) // 2
    logger.info(
        "4-cell stage: first=%d, last=%d, mid=%d",
        first_four, last_four, four_cells,
    )

    # Get the 4 alive cells at this timepoint
    nuclei = nuclei_record[four_cells]
    alive_cells = [n for n in nuclei if n.status >= 1 and POLAR not in n.identity.lower()]

    if len(alive_cells) != 4:
        logger.warning("Expected 4 alive cells at t=%d, found %d", four_cells, len(alive_cells))
        result.start_index = starting_index
        return result

    # Align the diamond pattern to identify N/S/E/W
    positions = _align_diamond(
        alive_cells, canonical_transform, angle, z_pix_res
    )

    if positions is None:
        logger.warning("Failed to align diamond pattern")
        result.start_index = starting_index
        return result

    # Assign ABa, ABp, EMS, P2 based on division timing
    _four_cell_id_assignment(nuclei_record, positions, four_cells, first_four)

    # Back-trace to assign AB, P1, P0
    start_idx = _back_assignment(nuclei_record, four_cells)

    result.success = True
    result.axis_found = True
    result.start_index = start_idx
    result.ap = positions["ap"]
    result.dv = positions["dv"]
    result.lr = positions["lr"]
    result.nuc_count = 1

    return result


def _align_diamond(
    cells: list[Nucleus],
    canonical_transform: CanonicalTransform | None,
    angle: float,
    z_pix_res: float,
) -> dict | None:
    """Align 4 cells in diamond pattern, determine N/S/E/W.

    The 4 cells at the 4-cell stage form a diamond shape. We identify
    the N (North/anterior), S (South/posterior), E (East), W (West)
    positions by computing centroid and distances.

    Returns:
        Dict with keys: 'north', 'south', 'east', 'west', 'ap', 'dv', 'lr',
        or None on failure.
    """
    # Get 3D positions, z-scaled
    positions = []
    for c in cells:
        pos = np.array([float(c.x), float(c.y), float(c.z) * z_pix_res])
        positions.append(pos)

    # Apply canonical transform if available (v2)
    if canonical_transform is not None and canonical_transform.active:
        positions = [canonical_transform.apply(p) for p in positions]
    elif angle != 0.0:
        # V1: rotate in XY plane
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        rotated = []
        for p in positions:
            x_new = p[0] * cos_a - p[1] * sin_a
            y_new = p[0] * sin_a + p[1] * cos_a
            rotated.append(np.array([x_new, y_new, p[2]]))
        positions = rotated

    # Compute centroid
    centroid = np.mean(positions, axis=0)

    # Find the cell furthest from centroid in the x-direction (AP axis)
    # In canonical frame: AP is along x, smaller x = more anterior
    x_vals = [(positions[i][0], i) for i in range(4)]
    x_vals.sort()

    # Most negative x = most anterior = North
    # Most positive x = most posterior = South
    north_idx = x_vals[0][1]
    south_idx = x_vals[3][1]

    # The other two are East/West, determined by z-coordinate (LR axis)
    remaining = [x_vals[1][1], x_vals[2][1]]
    if positions[remaining[0]][2] > positions[remaining[1]][2]:
        east_idx = remaining[0]  # more positive z = left in canonical
        west_idx = remaining[1]
    else:
        east_idx = remaining[1]
        west_idx = remaining[0]

    # Determine orientation signs
    ap = 1 if positions[north_idx][0] < positions[south_idx][0] else -1
    dv = 1 if positions[east_idx][1] > positions[west_idx][1] else -1
    lr = 1 if positions[east_idx][2] > positions[west_idx][2] else -1

    return {
        "north": cells[north_idx],
        "south": cells[south_idx],
        "east": cells[east_idx],
        "west": cells[west_idx],
        "ap": ap,
        "dv": dv,
        "lr": lr,
    }


def _four_cell_id_assignment(
    nuclei_record: list[list[Nucleus]],
    positions: dict,
    four_cells: int,
    first_four: int,
) -> None:
    """Assign ABa, ABp, EMS, P2 using division timing.

    In C. elegans, AB divides before P1. So the pair that appeared first
    (tracing back predecessors) is ABa/ABp, and the other pair is EMS/P2.

    The North cell of each pair gets the anterior name:
        AB pair: North=ABa, South=ABp
        P1 pair: North=EMS, South=P2
    """
    north: Nucleus = positions["north"]
    south: Nucleus = positions["south"]
    east: Nucleus = positions["east"]
    west: Nucleus = positions["west"]

    # Find which pair divided earlier by tracing back predecessors
    # The pair that shares an earlier common ancestor = AB daughters
    # Try to trace north/south vs east/west back
    north_birth = _find_birth_time(nuclei_record, north, four_cells)
    south_birth = _find_birth_time(nuclei_record, south, four_cells)
    east_birth = _find_birth_time(nuclei_record, east, four_cells)
    west_birth = _find_birth_time(nuclei_record, west, four_cells)

    # Group into pairs: the pair that appeared earlier is from AB division
    # Check if N/S are sisters or if N/E are sisters, etc.
    ns_pair = _are_sisters(nuclei_record, north, south, four_cells)
    ne_pair = _are_sisters(nuclei_record, north, east, four_cells)
    nw_pair = _are_sisters(nuclei_record, north, west, four_cells)

    if ns_pair:
        # N/S are sisters (from same division), E/W are sisters
        ab_pair = (north, south)
        p1_pair = (east, west)
        # Determine which pair is AB (divided first)
        ns_birth = min(north_birth, south_birth)
        ew_birth = min(east_birth, west_birth)
        if ns_birth > ew_birth:
            # E/W appeared first -> they're from AB
            ab_pair, p1_pair = p1_pair, ab_pair
    elif ne_pair:
        ab_pair = (north, east)
        p1_pair = (south, west)
        ne_birth = min(north_birth, east_birth)
        sw_birth = min(south_birth, west_birth)
        if ne_birth > sw_birth:
            ab_pair, p1_pair = p1_pair, ab_pair
    elif nw_pair:
        ab_pair = (north, west)
        p1_pair = (south, east)
        nw_birth = min(north_birth, west_birth)
        se_birth = min(south_birth, east_birth)
        if nw_birth > se_birth:
            ab_pair, p1_pair = p1_pair, ab_pair
    else:
        # Fallback: N/S = AB, E/W = P1 (use spatial heuristic)
        logger.warning("Could not determine sister pairs; using spatial heuristic")
        ab_pair = (north, south)
        p1_pair = (east, west)

    # Within AB pair: more anterior (smaller x in canonical) = ABa
    # Within P1 pair: more anterior = EMS
    # For simplicity, use the North cell as ABa and EMS
    a_cell, b_cell = ab_pair
    c_cell, d_cell = p1_pair

    # Assign: the more anterior cell in each pair gets ABa/EMS
    if a_cell.x <= b_cell.x:
        a_cell.identity = "ABa"
        b_cell.identity = "ABp"
    else:
        b_cell.identity = "ABa"
        a_cell.identity = "ABp"

    if c_cell.x <= d_cell.x:
        c_cell.identity = "EMS"
        d_cell.identity = "P2"
    else:
        d_cell.identity = "EMS"
        c_cell.identity = "P2"

    logger.info(
        "4-cell ID: ABa=%s, ABp=%s, EMS=%s, P2=%s",
        [n for n in [a_cell, b_cell, c_cell, d_cell] if n.identity == "ABa"],
        [n for n in [a_cell, b_cell, c_cell, d_cell] if n.identity == "ABp"],
        [n for n in [a_cell, b_cell, c_cell, d_cell] if n.identity == "EMS"],
        [n for n in [a_cell, b_cell, c_cell, d_cell] if n.identity == "P2"],
    )


def _find_birth_time(
    nuclei_record: list[list[Nucleus]],
    nuc: Nucleus,
    current_time: int,
) -> int:
    """Find the earliest timepoint this cell lineage exists at.

    Traces predecessors backward in time.
    """
    t = current_time
    current = nuc
    while t > 0 and current.predecessor != NILLI:
        t -= 1
        prev_nuclei = nuclei_record[t]
        pred_idx = current.predecessor - 1  # 1-based to 0-based
        if 0 <= pred_idx < len(prev_nuclei):
            pred = prev_nuclei[pred_idx]
            # If the predecessor has two successors, this is where the cell was born
            if pred.successor2 != NILLI:
                return t + 1
            current = pred
        else:
            break
    return t


def _are_sisters(
    nuclei_record: list[list[Nucleus]],
    nuc1: Nucleus,
    nuc2: Nucleus,
    current_time: int,
) -> bool:
    """Check if two nuclei at the same timepoint are sister cells.

    Two cells are sisters if they share the same predecessor (parent).
    """
    if current_time <= 0:
        return False

    prev_nuclei = nuclei_record[current_time - 1]

    # Check if both predecessors point to the same parent that has two successors
    if nuc1.predecessor == NILLI or nuc2.predecessor == NILLI:
        return False

    # They share a parent if one cell's predecessor's successor1 or successor2
    # is the other cell
    pred1_idx = nuc1.predecessor - 1
    if 0 <= pred1_idx < len(prev_nuclei):
        pred = prev_nuclei[pred1_idx]
        if pred.successor2 != NILLI:
            # Check if both nuclei are successors of this parent
            s1_idx = pred.successor1 - 1
            s2_idx = pred.successor2 - 1
            nuclei_at_time = nuclei_record[current_time]
            idx1 = nuclei_at_time.index(nuc1) if nuc1 in nuclei_at_time else -1
            idx2 = nuclei_at_time.index(nuc2) if nuc2 in nuclei_at_time else -1
            if (idx1 == s1_idx and idx2 == s2_idx) or (idx1 == s2_idx and idx2 == s1_idx):
                return True

    return False


def _back_assignment(
    nuclei_record: list[list[Nucleus]],
    four_cells_time: int,
) -> int:
    """Trace backward from 4-cell stage to assign AB, P1, P0.

    Starting from ABa (or ABp) at the 4-cell stage, traces predecessors
    backward to find:
        - AB (parent of ABa and ABp)
        - P1 (parent of EMS and P2)
        - P0 (parent of AB and P1)

    Returns:
        The 0-based index of the first timepoint where we have named cells.
    """
    start_index = four_cells_time

    # Find the 4-cell nuclei
    nuclei = nuclei_record[four_cells_time]
    aba = None
    ems = None
    for n in nuclei:
        if n.identity == "ABa":
            aba = n
        elif n.identity == "EMS":
            ems = n

    if aba is None or ems is None:
        logger.warning("Cannot find ABa or EMS for back-assignment")
        return start_index

    # Trace ABa back to find AB
    t = four_cells_time
    current = aba
    while t > 0 and current.predecessor != NILLI:
        t -= 1
        prev_nuclei = nuclei_record[t]
        pred_idx = current.predecessor - 1
        if 0 <= pred_idx < len(prev_nuclei):
            pred = prev_nuclei[pred_idx]
            if pred.successor2 != NILLI:
                # This is AB (parent of ABa and ABp)
                pred.identity = "AB"
                start_index = t
                current = pred
                # Continue tracing to find P0
                break
            else:
                pred.identity = current.identity  # inherit name backward
                current = pred
                start_index = t
        else:
            break

    # Now trace AB back to find P0
    ab = current
    while t > 0 and ab.predecessor != NILLI:
        t -= 1
        prev_nuclei = nuclei_record[t]
        pred_idx = ab.predecessor - 1
        if 0 <= pred_idx < len(prev_nuclei):
            pred = prev_nuclei[pred_idx]
            if pred.successor2 != NILLI:
                # This is P0 (parent of AB and P1)
                pred.identity = "P0"
                start_index = t
                # Also name the sister (P1)
                s2_idx = pred.successor2 - 1
                s1_idx = pred.successor1 - 1
                next_nuclei = nuclei_record[t + 1]
                # Figure out which successor is AB and which is P1
                ab_idx = next_nuclei.index(ab) if ab in next_nuclei else -1
                if ab_idx == s1_idx and 0 <= s2_idx < len(next_nuclei):
                    next_nuclei[s2_idx].identity = "P1"
                elif ab_idx == s2_idx and 0 <= s1_idx < len(next_nuclei):
                    next_nuclei[s1_idx].identity = "P1"
                break
            else:
                pred.identity = ab.identity  # inherit
                ab = pred
                start_index = t
        else:
            break

    # Also trace EMS back to name P1 (parent of EMS and P2)
    t = four_cells_time
    current = ems
    while t > 0 and current.predecessor != NILLI:
        t -= 1
        prev_nuclei = nuclei_record[t]
        pred_idx = current.predecessor - 1
        if 0 <= pred_idx < len(prev_nuclei):
            pred = prev_nuclei[pred_idx]
            if pred.successor2 != NILLI:
                # This is P1 (parent of EMS and P2)
                if not pred.identity:
                    pred.identity = "P1"
                break
            else:
                if not pred.identity:
                    pred.identity = current.identity
                current = pred
        else:
            break

    return start_index
