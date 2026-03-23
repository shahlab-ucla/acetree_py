"""Nucleus data model — a detected nucleus at a single timepoint.

This is the central data record in AceTree. Each line in a nuclei file
within the ZIP archive corresponds to one Nucleus instance.

File format (new format, comma-separated):
    Column indices defined in _NEW_FORMAT_COLS.
    Fields: index, status, predecessor, successor1, successor2, x, y, z,
            size, identity, weight, rweight, rsum, rcount, assignedID,
            rwraw, rwcorr1, rwcorr2, rwcorr3, rwcorr4

Ported from: org.rhwlab.snight.Nucleus (Nucleus.java)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Sentinel for "no link" (predecessor/successor not set)
NILLI = -1
NILL_STR = "nill"

# ── Column indices for the NEW nuclei file format ──
# Matches Nucleus.java: INDEX=0, STATUS=1, PRED=2, SUCC1=3, SUCC2=4,
# X=5, Y=6, Z=7, SIZE=8, IDENTITY=9, WT=10, RWT=11, RSUM=12, RCOUNT=13,
# ASSIGNEDID=14, RWRAW=15, RWCORR1=16, RWCORR2=17, RWCORR3=18, RWCORR4=19
_COL_INDEX = 0
_COL_STATUS = 1
_COL_PRED = 2
_COL_SUCC1 = 3
_COL_SUCC2 = 4
_COL_X = 5
_COL_Y = 6
_COL_Z = 7
_COL_SIZE = 8
_COL_IDENTITY = 9
_COL_WT = 10
_COL_RWT = 11
_COL_RSUM = 12
_COL_RCOUNT = 13
_COL_ASSIGNEDID = 14
_COL_RWRAW = 15
_COL_RWCORR1 = 16
_COL_RWCORR2 = 17
_COL_RWCORR3 = 18
_COL_RWCORR4 = 19

# ── Column indices for the OLD nuclei file format ──
_OLD_INDEX = 0
_OLD_X = 1
_OLD_Y = 2
_OLD_Z = 3
_OLD_IDENTITY = 4
_OLD_SIZE = 5
_OLD_WT = 6
_OLD_STATUS = 7
_OLD_PRED = 12
_OLD_SUCC1 = 13
_OLD_SUCC2 = 14

# Red correction method names (matches Config.REDCHOICE)
RED_CORRECTIONS = ("none", "global", "local", "blot", "cross")


@dataclass
class Nucleus:
    """A detected nucleus at a single timepoint.

    Attributes match the fields in the nuclei text files. All linking indices
    (predecessor, successor1, successor2) are 1-based into their respective
    timepoint's nucleus list, or NILLI (-1) if absent.
    """

    index: int = 0           # 1-based index within timepoint
    x: int = 0               # pixel x
    y: int = 0               # pixel y
    z: float = 0.0           # z-plane (float for sub-plane precision)
    size: int = 0            # diameter in pixels
    identity: str = ""       # Sulston name (e.g. "ABala")
    assigned_id: str = ""    # Manually forced name (survives re-naming)
    status: int = -1         # >= 1 alive, -1 dead/invalid
    predecessor: int = NILLI  # index into previous timepoint
    successor1: int = NILLI   # index into next timepoint
    successor2: int = NILLI   # second successor if dividing
    weight: int = 0          # GFP intensity
    rweight: int = 0         # computed red weight
    rsum: int = 0            # raw red sum
    rcount: int = 0          # raw red count
    rwraw: int = 0           # raw red weight
    rwcorr1: int = 0         # global background correction
    rwcorr2: int = 0         # local background correction
    rwcorr3: int = 0         # blot correction
    rwcorr4: int = 0         # crosstalk correction
    hash_key: str | None = None  # key for tree lookup (set by AncesTree)

    @property
    def is_alive(self) -> bool:
        """True if this nucleus is alive (status >= 1)."""
        return self.status >= 1

    @property
    def is_dividing(self) -> bool:
        """True if this nucleus has two successors (is dividing)."""
        return self.successor2 != NILLI

    @property
    def effective_name(self) -> str:
        """Return assigned_id if set, otherwise identity."""
        return self.assigned_id if self.assigned_id else self.identity

    def corrected_red(self, method: str = "none") -> int:
        """Compute red weight with the specified correction applied.

        Args:
            method: One of "none", "global", "local", "blot", "cross".

        Returns:
            The corrected red weight value.
        """
        if method == "none" or method not in RED_CORRECTIONS:
            return self.rwraw
        idx = RED_CORRECTIONS.index(method)
        corrections = [0, self.rwcorr1, self.rwcorr2, self.rwcorr3, self.rwcorr4]
        return self.rwraw - corrections[idx]

    def copy(self) -> Nucleus:
        """Create a deep copy of this Nucleus."""
        return Nucleus(
            index=self.index,
            x=self.x,
            y=self.y,
            z=self.z,
            size=self.size,
            identity=self.identity,
            assigned_id=self.assigned_id,
            status=self.status,
            predecessor=self.predecessor,
            successor1=self.successor1,
            successor2=self.successor2,
            weight=self.weight,
            rweight=self.rweight,
            rsum=self.rsum,
            rcount=self.rcount,
            rwraw=self.rwraw,
            rwcorr1=self.rwcorr1,
            rwcorr2=self.rwcorr2,
            rwcorr3=self.rwcorr3,
            rwcorr4=self.rwcorr4,
            hash_key=self.hash_key,
        )

    # ── Serialization ────────────────────────────────────────────────

    @classmethod
    def from_text_line(cls, line: str, *, old_format: bool = False) -> Nucleus:
        """Parse a Nucleus from a comma-separated text line.

        Args:
            line: A single line from a nuclei file (comma-separated fields).
            old_format: If True, parse using the old column layout.

        Returns:
            A Nucleus instance with all fields populated.
        """
        sa = [s.strip() for s in line.split(",")]
        nuc = cls()

        if old_format:
            nuc._parse_old_format(sa)
        else:
            nuc._parse_new_format(sa)

        return nuc

    def _parse_new_format(self, sa: list[str]) -> None:
        """Parse fields from the new file format column layout."""
        self.index = int(sa[_COL_INDEX])
        self.x = int(sa[_COL_X])
        self.y = int(sa[_COL_Y])
        self.z = float(sa[_COL_Z])
        self.identity = sa[_COL_IDENTITY] if len(sa) > _COL_IDENTITY else ""
        self.size = int(sa[_COL_SIZE])
        self.weight = int(sa[_COL_WT])

        # Status: only positive values count as alive
        xstat = int(sa[_COL_STATUS])
        self.status = xstat if xstat > 0 else -1

        # Predecessor/successor links: "nill" or -1 means no link
        self.predecessor = _parse_link(sa[_COL_PRED])
        self.successor1 = _parse_link(sa[_COL_SUCC1])
        self.successor2 = _parse_link(sa[_COL_SUCC2]) if len(sa) > _COL_SUCC2 and sa[_COL_SUCC2] else NILLI

        # Optional red-channel fields — may be absent in older datasets
        self.rweight = _safe_int(sa, _COL_RWT)
        self.rsum = _safe_int(sa, _COL_RSUM)
        self.rcount = _safe_int(sa, _COL_RCOUNT)
        self.assigned_id = _safe_str(sa, _COL_ASSIGNEDID)
        self.rwraw = _safe_int(sa, _COL_RWRAW)
        self.rwcorr1 = _safe_int(sa, _COL_RWCORR1)
        self.rwcorr2 = _safe_int(sa, _COL_RWCORR2)
        self.rwcorr3 = _safe_int(sa, _COL_RWCORR3)
        self.rwcorr4 = _safe_int(sa, _COL_RWCORR4)

    def _parse_old_format(self, sa: list[str]) -> None:
        """Parse fields from the old file format column layout."""
        self.index = int(sa[_OLD_INDEX])
        self.x = int(sa[_OLD_X])
        self.y = int(sa[_OLD_Y])
        self.z = float(sa[_OLD_Z])
        self.identity = sa[_OLD_IDENTITY]
        self.size = int(sa[_OLD_SIZE])
        self.weight = int(sa[_OLD_WT])

        xstat = int(sa[_OLD_STATUS])
        self.status = xstat if xstat >= 0 else -1

        self.predecessor = _parse_link(sa[_OLD_PRED])
        self.successor1 = _parse_link(sa[_OLD_SUCC1])
        self.successor2 = _parse_link(sa[_OLD_SUCC2]) if len(sa) > _OLD_SUCC2 and sa[_OLD_SUCC2] else NILLI

    def to_text_line(self) -> str:
        """Serialize this Nucleus to a comma-separated text line (new format).

        The output format matches NucZipper.formatNucleus() in Java:
            index, status, predecessor, successor1, successor2, x, y, z,
            size, identity, weight, rweight, rsum, rcount, assignedID,
            rwraw, rwcorr1, rwcorr2, rwcorr3, rwcorr4

        This matches the column indices used by the constructor for parsing,
        ensuring round-trip compatibility.
        """
        # Status: write 1 if alive, 0 if not (matches NucZipper behavior)
        status_out = 1 if self.status > 0 else 0
        parts = [
            str(self.index),
            str(status_out),
            str(self.predecessor),
            str(self.successor1),
            str(self.successor2),
            str(self.x),
            str(self.y),
            str(self.z),
            str(self.size),
            self.identity,
            str(self.weight),
            str(self.rweight),
            str(self.rsum),
            str(self.rcount),
            self.assigned_id,
            str(self.rwraw),
            str(self.rwcorr1),
            str(self.rwcorr2),
            str(self.rwcorr3),
            str(self.rwcorr4),
        ]
        return ", ".join(parts)

    def __repr__(self) -> str:
        name = self.effective_name or f"Nuc{self.index}"
        return f"Nucleus({name}, t_idx={self.index}, pos=({self.x},{self.y},{self.z:.1f}), status={self.status})"


# ── Helper functions ──────────────────────────────────────────────


def _parse_link(s: str) -> int:
    """Parse a predecessor/successor link field.

    Returns NILLI (-1) for "nill" or -1, otherwise the integer value.
    """
    s = s.strip()
    if not s or s.lower() == NILL_STR:
        return NILLI
    val = int(s)
    return NILLI if val == -1 else val


def _safe_int(sa: list[str], idx: int) -> int:
    """Safely extract an integer from a string array, returning 0 on failure."""
    if idx >= len(sa):
        return 0
    s = sa[idx].strip()
    if not s:
        return 0
    try:
        return int(s)
    except (ValueError, IndexError):
        return 0


def _safe_str(sa: list[str], idx: int) -> str:
    """Safely extract a string from a string array, returning '' on failure."""
    if idx >= len(sa):
        return ""
    return sa[idx].strip()
