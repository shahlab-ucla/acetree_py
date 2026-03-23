"""Pre-edit validation checks.

Validators verify that an edit operation is safe before executing.
They return a list of error messages (empty = valid).

This provides explicit validation that Java mostly lacked —
the Java code did ad-hoc null checks scattered across dialog code.
"""

from __future__ import annotations

from ..core.nucleus import NILLI, Nucleus

NucleiRecord = list[list[Nucleus]]


def validate_add_nucleus(
    nuclei_record: NucleiRecord,
    time: int,
    predecessor: int = NILLI,
) -> list[str]:
    """Validate adding a new nucleus.

    Args:
        nuclei_record: The nuclei data.
        time: 1-based timepoint.
        predecessor: 1-based predecessor index (or NILLI).

    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    if time < 1:
        errors.append(f"Timepoint must be >= 1, got {time}")

    if predecessor != NILLI and time >= 2:
        t_idx = time - 2  # Previous timepoint
        if t_idx < 0 or t_idx >= len(nuclei_record):
            errors.append(f"Predecessor timepoint {time - 1} out of range")
        else:
            nuclei = nuclei_record[t_idx]
            p_idx = predecessor - 1
            if p_idx < 0 or p_idx >= len(nuclei):
                errors.append(f"Predecessor index {predecessor} out of range at t={time - 1}")
            else:
                parent = nuclei[p_idx]
                if parent.successor1 != NILLI and parent.successor2 != NILLI:
                    errors.append(
                        f"Predecessor at t={time - 1} idx={predecessor} "
                        f"already has 2 successors"
                    )
    return errors


def validate_remove_nucleus(
    nuclei_record: NucleiRecord,
    time: int,
    index: int,
) -> list[str]:
    """Validate removing (killing) a nucleus.

    Args:
        nuclei_record: The nuclei data.
        time: 1-based timepoint.
        index: 1-based nucleus index.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []
    t_idx = time - 1
    if t_idx < 0 or t_idx >= len(nuclei_record):
        errors.append(f"Timepoint {time} out of range")
        return errors

    nuclei = nuclei_record[t_idx]
    n_idx = index - 1
    if n_idx < 0 or n_idx >= len(nuclei):
        errors.append(f"Nucleus index {index} out of range at t={time}")
        return errors

    nuc = nuclei[n_idx]
    if not nuc.is_alive:
        errors.append(f"Nucleus at t={time} idx={index} is already dead")

    return errors


def validate_relink(
    nuclei_record: NucleiRecord,
    time: int,
    index: int,
    new_predecessor: int,
) -> list[str]:
    """Validate relinking a nucleus to a new predecessor.

    Args:
        nuclei_record: The nuclei data.
        time: 1-based timepoint of the nucleus.
        index: 1-based index of the nucleus.
        new_predecessor: 1-based index of the new predecessor (or NILLI).

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    # Check nucleus exists
    t_idx = time - 1
    if t_idx < 0 or t_idx >= len(nuclei_record):
        errors.append(f"Timepoint {time} out of range")
        return errors
    nuclei = nuclei_record[t_idx]
    n_idx = index - 1
    if n_idx < 0 or n_idx >= len(nuclei):
        errors.append(f"Nucleus index {index} out of range at t={time}")
        return errors

    # Check new predecessor exists
    if new_predecessor != NILLI:
        if time < 2:
            errors.append("Cannot set predecessor for timepoint 1")
        else:
            prev_t_idx = time - 2
            if prev_t_idx >= len(nuclei_record):
                errors.append(f"Previous timepoint {time - 1} out of range")
            else:
                prev_nuclei = nuclei_record[prev_t_idx]
                p_idx = new_predecessor - 1
                if p_idx < 0 or p_idx >= len(prev_nuclei):
                    errors.append(f"New predecessor index {new_predecessor} out of range at t={time - 1}")
                else:
                    parent = prev_nuclei[p_idx]
                    if parent.successor1 != NILLI and parent.successor2 != NILLI:
                        # Check if one of the successors is the current nucleus
                        if parent.successor1 != index and parent.successor2 != index:
                            errors.append(
                                f"New predecessor at t={time - 1} idx={new_predecessor} "
                                f"already has 2 successors"
                            )

    return errors


def validate_kill_cell(
    nuclei_record: NucleiRecord,
    cell_name: str,
    start_time: int,
) -> list[str]:
    """Validate killing a named cell.

    Args:
        nuclei_record: The nuclei data.
        cell_name: Name of the cell to kill.
        start_time: 1-based starting timepoint.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    if not cell_name:
        errors.append("Cell name cannot be empty")
        return errors

    if start_time < 1:
        errors.append(f"Start time must be >= 1, got {start_time}")
        return errors

    if start_time > len(nuclei_record):
        errors.append(f"Start time {start_time} exceeds data range ({len(nuclei_record)} timepoints)")
        return errors

    # Check that the cell exists at the start time
    t_idx = start_time - 1
    found = any(n.identity == cell_name and n.is_alive for n in nuclei_record[t_idx])
    if not found:
        errors.append(f"Cell '{cell_name}' not found alive at t={start_time}")

    return errors


def validate_relink_interpolation(
    nuclei_record: NucleiRecord,
    start_time: int,
    start_index: int,
    end_time: int,
    end_index: int,
) -> list[str]:
    """Validate a relink-with-interpolation operation.

    Args:
        nuclei_record: The nuclei data.
        start_time: 1-based timepoint of start nucleus.
        start_index: 1-based index of start nucleus.
        end_time: 1-based timepoint of end nucleus.
        end_index: 1-based index of end nucleus.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    if end_time <= start_time:
        errors.append(f"End time ({end_time}) must be after start time ({start_time})")
        return errors

    # Check start nucleus exists
    st_idx = start_time - 1
    if st_idx < 0 or st_idx >= len(nuclei_record):
        errors.append(f"Start timepoint {start_time} out of range")
    else:
        nuclei = nuclei_record[st_idx]
        si = start_index - 1
        if si < 0 or si >= len(nuclei):
            errors.append(f"Start index {start_index} out of range at t={start_time}")
        else:
            parent = nuclei[si]
            if parent.successor1 != NILLI and parent.successor2 != NILLI:
                errors.append(f"Start nucleus at t={start_time} idx={start_index} already has 2 successors")

    # Check end nucleus exists
    et_idx = end_time - 1
    if et_idx < 0 or et_idx >= len(nuclei_record):
        errors.append(f"End timepoint {end_time} out of range")
    else:
        nuclei = nuclei_record[et_idx]
        ei = end_index - 1
        if ei < 0 or ei >= len(nuclei):
            errors.append(f"End index {end_index} out of range at t={end_time}")

    return errors
