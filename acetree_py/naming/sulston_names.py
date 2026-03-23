"""Sulston naming conventions and helper functions.

The Sulston naming system names cells in C. elegans using hierarchical
suffixes that indicate division planes:
    a/p = anterior/posterior (AP axis)
    l/r = left/right (LR axis)
    d/v = dorsal/ventral (DV axis)

Ported from: DivisionCaller.complement() and naming constants in the Java codebase.
"""

from __future__ import annotations

# ── Sulston letter complements ──
# Each division produces two daughters named by opposite letters.
_COMPLEMENT_MAP: dict[str, str] = {
    "a": "p",
    "p": "a",
    "d": "v",
    "v": "d",
    "l": "r",
    "r": "l",
}

# Letter -> axis mapping for generating rules for unknown parents.
# "a" or "p" -> AP axis [1, 0, 0]
# "l" or "r" -> LR axis [0, 0, 1]
# "d" or "v" -> DV axis [0, 1, 0]
LETTER_TO_AXIS = {
    "a": (1.0, 0.0, 0.0),
    "p": (1.0, 0.0, 0.0),
    "l": (0.0, 0.0, 1.0),
    "r": (0.0, 0.0, 1.0),
    "d": (0.0, 1.0, 0.0),
    "v": (0.0, 1.0, 0.0),
}

# Well-known early cells in the lineage
FOUNDER_CELLS = ("P0", "AB", "P1", "EMS", "P2", "E", "MS", "C", "P3", "D", "P4", "Z2", "Z3")


def complement(letter: str) -> str:
    """Return the Sulston complement of a division letter.

    Args:
        letter: One of 'a', 'p', 'l', 'r', 'd', 'v'.

    Returns:
        The complement letter, or 'g' for unknown input (matching Java behavior).
    """
    return _COMPLEMENT_MAP.get(letter, "g")


def daughter_names(parent: str, sulston_letter: str) -> tuple[str, str]:
    """Generate the two daughter names for a division.

    Args:
        parent: The parent cell name (e.g. "ABa").
        sulston_letter: The Sulston letter for daughter1 (e.g. "l").

    Returns:
        (daughter1_name, daughter2_name) — e.g. ("ABal", "ABar").
    """
    comp = complement(sulston_letter)
    return parent + sulston_letter, parent + comp


def is_anterior_daughter(name: str) -> bool:
    """Check if a cell name ends with an anterior-type suffix."""
    if not name:
        return False
    return name[-1] in ("a", "l", "d")


def is_posterior_daughter(name: str) -> bool:
    """Check if a cell name ends with a posterior-type suffix."""
    if not name:
        return False
    return name[-1] in ("p", "r", "v")
