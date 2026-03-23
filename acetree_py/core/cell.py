"""Cell lineage tree node — a cell across its lifetime in the embryo.

A Cell represents a single cell from birth to division/death. It holds
references to its parent and children (forming the lineage tree), plus
a list of Nucleus snapshots at each timepoint during which the cell exists.

Contains NO drawing logic — rendering is handled entirely in gui/.

Ported from: org.rhwlab.tree.Cell (Cell.java) and CellData.java
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from acetree_py.core.nucleus import Nucleus

logger = logging.getLogger(__name__)


class CellFate(Enum):
    """How a cell's lifetime ends."""

    ALIVE = 0       # Still alive at the last observed timepoint
    DIVIDED = 1     # Divided into two daughter cells
    DIED = 2        # Died (programmed cell death)


@dataclass
class Cell:
    """A cell in the lineage tree.

    Attributes:
        name: Sulston name (e.g. "ABala") or auto-generated name.
        start_time: First timepoint where this cell exists (1-based).
        end_time: Last timepoint where this cell exists (1-based).
        end_fate: How this cell's lifetime ended.
        parent: Parent cell (None for the root/P0).
        children: Daughter cells (0 for leaf, 2 for division).
        nuclei: List of (timepoint, Nucleus) tuples across the cell's lifetime.
        hash_key: Key used for tree lookups (matches Nucleus.hash_key).
    """

    name: str = ""
    start_time: int = 0
    end_time: int = 0
    end_fate: CellFate = CellFate.ALIVE
    parent: Cell | None = field(default=None, repr=False)
    children: list[Cell] = field(default_factory=list, repr=False)
    nuclei: list[tuple[int, Nucleus]] = field(default_factory=list, repr=False)
    hash_key: str | None = None

    @property
    def lifetime(self) -> int:
        """Number of timepoints this cell exists."""
        return self.end_time - self.start_time + 1

    @property
    def is_leaf(self) -> bool:
        """True if this cell has no children."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """True if this cell has no parent."""
        return self.parent is None

    def add_child(self, child: Cell) -> None:
        """Add a daughter cell."""
        child.parent = self
        self.children.append(child)

    def get_nucleus_at(self, time: int) -> Nucleus | None:
        """Get the Nucleus snapshot at a specific timepoint.

        Args:
            time: The 1-based timepoint.

        Returns:
            The Nucleus at that time, or None if not found.
        """
        for t, nuc in self.nuclei:
            if t == time:
                return nuc
        return None

    def add_nucleus(self, time: int, nuc: Nucleus) -> None:
        """Add a Nucleus snapshot for a given timepoint."""
        self.nuclei.append((time, nuc))

    def iter_ancestors(self) -> Iterator[Cell]:
        """Iterate from parent up to root (not including self)."""
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def iter_descendants(self) -> Iterator[Cell]:
        """Iterate over all descendants in pre-order (not including self)."""
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def iter_subtree_preorder(self) -> Iterator[Cell]:
        """Iterate over self and all descendants in pre-order."""
        yield self
        for child in self.children:
            yield from child.iter_subtree_preorder()

    def iter_leaves(self) -> Iterator[Cell]:
        """Iterate over all leaf cells in this subtree."""
        if self.is_leaf:
            yield self
        else:
            for child in self.children:
                yield from child.iter_leaves()

    def depth(self) -> int:
        """Number of divisions from root to this cell."""
        d = 0
        current = self.parent
        while current is not None:
            d += 1
            current = current.parent
        return d

    def __repr__(self) -> str:
        fate_str = self.end_fate.name[0]  # A, D, or D
        return f"Cell({self.name}, t={self.start_time}-{self.end_time}, fate={fate_str}, children={len(self.children)})"
