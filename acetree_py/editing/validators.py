"""Pre-edit validation checks.

Validators verify that an edit operation is safe before executing.
They return a list of error messages (empty = valid).

This provides explicit validation that Java mostly lacked —
the Java code did ad-hoc null checks scattered across dialog code.
"""

from __future__ import annotations

from ..core.nucleus import NILLI, Nucleus, validate_storable_name

NucleiRecord = list[list[Nucleus]]


def _validate_name_text(name: str) -> list[str]:
    """Validate a cell name against the delimiter-based persistence format."""
    error = validate_storable_name(name, allow_empty=False)
    return [error] if error else []


def _forced_ids_on_chain_side(
    nuclei_record: NucleiRecord,
    t0: int,
    j0: int,
    *,
    backward: bool,
) -> set[str]:
    """Collect forced IDs up to, or forward from, a continuation anchor."""
    from .commands import _walk_continuation_chain

    chain = _walk_continuation_chain(nuclei_record, t0, j0)
    if backward:
        chain = [(t, j) for t, j in chain if t <= t0]
    else:
        chain = [(t, j) for t, j in chain if t >= t0]
    return {
        nuclei_record[t][j].assigned_id
        for t, j in chain
        if nuclei_record[t][j].assigned_id
    }


def _validate_forced_id_merge(
    nuclei_record: NucleiRecord,
    parent_t0: int,
    parent_j0: int,
    child_t0: int,
    child_j0: int,
) -> list[str]:
    """Reject a continuation that would silently reconcile two forced names."""
    parent_ids = _forced_ids_on_chain_side(
        nuclei_record, parent_t0, parent_j0, backward=True
    )
    child_ids = _forced_ids_on_chain_side(
        nuclei_record, child_t0, child_j0, backward=False
    )
    if parent_ids and child_ids and len(parent_ids | child_ids) > 1:
        names = ", ".join(sorted(parent_ids | child_ids))
        return [
            "Cannot merge continuation components with different forced "
            f"names ({names}); clear or reconcile an override first"
        ]
    return []


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

                    # If no other daughter remains, the prospective edge is
                    # a continuation and its manual naming state must be
                    # unambiguous.  A second, different successor is a
                    # division boundary, so daughter overrides may differ.
                    other_successors = {
                        successor
                        for successor in (parent.successor1, parent.successor2)
                        if successor not in (NILLI, index)
                    }
                    if not other_successors:
                        errors.extend(_validate_forced_id_merge(
                            nuclei_record,
                            prev_t_idx,
                            p_idx,
                            t_idx,
                            n_idx,
                        ))

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
    found = any(
        n.effective_name == cell_name and n.is_alive
        for n in nuclei_record[t_idx]
    )
    if not found:
        errors.append(f"Cell '{cell_name}' not found alive at t={start_time}")

    return errors


def validate_rename_cell(
    nuclei_record: NucleiRecord,
    time: int,
    index: int,
    new_name: str,
) -> tuple[list[str], tuple[int, int] | None]:
    """Validate renaming a cell to *new_name*.

    Checks for collisions with other cells in the nuclei record.  A
    *collision* means another cell — not in the continuation chain of
    (time, index) — already has *new_name* as its effective name
    (assigned_id or identity).

    Args:
        nuclei_record: The nuclei data.
        time: 1-based timepoint of the anchor nucleus.
        index: 1-based nucleus index.
        new_name: Proposed new name.

    Returns:
        (errors, collision_anchor) where
          - errors is a list of error messages (empty if valid).
          - collision_anchor is a (time, index) 1-based tuple pointing to
            an anchor nucleus of the conflicting cell if a collision was
            detected, else None.  The UI can use this to offer a Swap.
    """
    from .commands import _walk_continuation_chain

    errors: list[str] = []

    # Basic validation.  Validate before trimming so hidden control
    # characters cannot be normalized away and written to the CSV record.
    errors.extend(_validate_name_text(new_name))
    if errors:
        return errors, None

    new_name = new_name.strip()

    t_idx = time - 1
    if t_idx < 0 or t_idx >= len(nuclei_record):
        errors.append(f"Timepoint {time} out of range")
        return errors, None
    n_idx = index - 1
    if n_idx < 0 or n_idx >= len(nuclei_record[t_idx]):
        errors.append(f"Nucleus index {index} out of range at t={time}")
        return errors, None

    # No-op: renaming to the same name is allowed
    target_nuc = nuclei_record[t_idx][n_idx]
    if target_nuc.effective_name == new_name:
        return errors, None

    # Collect the set of (t0, j0) in the target cell's continuation chain
    target_chain = set(_walk_continuation_chain(nuclei_record, t_idx, n_idx))

    # Scan all nuclei: if any nucleus outside the target chain has
    # effective_name == new_name, it's a collision.  Report the first one.
    for t0, nucs in enumerate(nuclei_record):
        for j0, nuc in enumerate(nucs):
            if not nuc.is_alive:
                continue
            if (t0, j0) in target_chain:
                continue
            if nuc.effective_name == new_name:
                errors.append(
                    f"Name '{new_name}' is already used by another cell "
                    f"(at t={t0 + 1} idx={j0 + 1})."
                )
                return errors, (t0 + 1, j0 + 1)

    return errors, None


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
    start_nuc: Nucleus | None = None
    end_nuc: Nucleus | None = None
    si = start_index - 1
    ei = end_index - 1

    if end_time <= start_time:
        errors.append(f"End time ({end_time}) must be after start time ({start_time})")
        return errors

    # Check start nucleus exists
    st_idx = start_time - 1
    if st_idx < 0 or st_idx >= len(nuclei_record):
        errors.append(f"Start timepoint {start_time} out of range")
    else:
        nuclei = nuclei_record[st_idx]
        if si < 0 or si >= len(nuclei):
            errors.append(f"Start index {start_index} out of range at t={start_time}")
        else:
            start_nuc = nuclei[si]
            if start_nuc.successor1 != NILLI and start_nuc.successor2 != NILLI:
                errors.append(f"Start nucleus at t={start_time} idx={start_index} already has 2 successors")

    # Check end nucleus exists
    et_idx = end_time - 1
    if et_idx < 0 or et_idx >= len(nuclei_record):
        errors.append(f"End timepoint {end_time} out of range")
    else:
        nuclei = nuclei_record[et_idx]
        if ei < 0 or ei >= len(nuclei):
            errors.append(f"End index {end_index} out of range at t={end_time}")
        else:
            end_nuc = nuclei[ei]

    if start_nuc is not None and end_nuc is not None:
        successors = {
            successor
            for successor in (start_nuc.successor1, start_nuc.successor2)
            if successor != NILLI
        }
        if end_time == start_time + 1:
            successors.discard(end_index)
        if not successors:
            errors.extend(_validate_forced_id_merge(
                nuclei_record,
                st_idx,
                si,
                et_idx,
                ei,
            ))

    return errors
