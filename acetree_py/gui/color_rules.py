"""Color rule engine for visualization-mode nucleus coloring.

Provides a flexible, rule-based system for assigning colors to nuclei
based on cell properties (lineage depth, fate, expression level, name
pattern, etc.).  Rules are evaluated in priority order; the first
matching rule wins.

The engine is only active when the UI is in *Visualization* mode.
In *Editing* mode the hardcoded status-based palette
(white / purple / orange / gray) is used instead.

Designed to be extended with a GUI rule editor in a later phase.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..core.cell import Cell
    from ..core.nuclei_manager import NucleiManager
    from ..core.nucleus import Nucleus

logger = logging.getLogger(__name__)


# ── Rule criteria ────────────────────────────────────────────────


class RuleCriterion(Enum):
    """What property a color rule matches against."""

    ALL = "all"                     # Matches every nucleus
    NAME_EXACT = "name_exact"       # Exact cell name
    NAME_PATTERN = "name_pattern"   # Glob/wildcard (e.g. "AB*", "MS*")
    NAME_REGEX = "name_regex"       # Full regex pattern
    LINEAGE_DEPTH = "lineage_depth" # Depth range (divisions from P0)
    FATE = "fate"                   # End fate (alive / divided / died)
    EXPRESSION = "expression"       # Expression value range (rweight)


# ── Color mapping modes ──────────────────────────────────────────


class ColorMode(Enum):
    """How a rule determines its output color."""

    SOLID = "solid"         # Single fixed RGBA color
    COLORMAP = "colormap"   # Map a numeric value through a colormap


# ── ColorRule ────────────────────────────────────────────────────


@dataclass
class ColorRule:
    """A single color rule that maps matching nuclei to a color.

    Rules are evaluated by :class:`ColorRuleEngine` in descending
    *priority* order.  The first matching rule determines the nucleus
    color; unmatched nuclei fall through to a configurable default.

    Attributes:
        name: Human-readable label (e.g. "AB lineage", "Dying cells").
        criterion: Which property to test.
        pattern: Criterion-specific match value:
            - ``ALL``: ignored
            - ``NAME_EXACT``: exact cell name string
            - ``NAME_PATTERN``: glob pattern (``fnmatch``)
            - ``NAME_REGEX``: regex pattern string
            - ``LINEAGE_DEPTH``: ``"min-max"`` inclusive range (e.g. ``"3-5"``)
            - ``FATE``: fate name (``"alive"``, ``"divided"``, ``"died"``)
            - ``EXPRESSION``: ``"min-max"`` value range
        color_mode: How to compute the output color.
        color: RGBA tuple for ``SOLID`` mode.
        colormap: Matplotlib colormap name for ``COLORMAP`` mode.
        vmin: Minimum value for colormap normalization.
        vmax: Maximum value for colormap normalization.
        priority: Higher values are evaluated first (default 0).
        enabled: If False, the rule is skipped.
    """

    name: str = ""
    criterion: RuleCriterion = RuleCriterion.ALL
    pattern: str = ""

    color_mode: ColorMode = ColorMode.SOLID
    color: tuple[float, float, float, float] = (0.55, 0.27, 1.0, 0.8)

    colormap: str = "viridis"
    vmin: float = 0.0
    vmax: float = 5000.0

    priority: int = 0
    enabled: bool = True

    # Cached compiled regex (populated on first match)
    _compiled_re: re.Pattern | None = field(
        default=None, repr=False, compare=False
    )

    def matches(
        self,
        nuc: Nucleus,
        cell: Cell | None,
    ) -> bool:
        """Test whether this rule matches a nucleus/cell pair."""
        if not self.enabled:
            return False

        ename = nuc.effective_name or ""

        if self.criterion == RuleCriterion.ALL:
            return True

        if self.criterion == RuleCriterion.NAME_EXACT:
            return ename == self.pattern

        if self.criterion == RuleCriterion.NAME_PATTERN:
            return fnmatch.fnmatchcase(ename, self.pattern)

        if self.criterion == RuleCriterion.NAME_REGEX:
            if self._compiled_re is None:
                try:
                    self._compiled_re = re.compile(self.pattern)
                except re.error:
                    return False
            return bool(self._compiled_re.search(ename))

        if self.criterion == RuleCriterion.LINEAGE_DEPTH:
            if cell is None:
                return False
            lo, hi = _parse_range(self.pattern)
            return lo <= cell.depth() <= hi

        if self.criterion == RuleCriterion.FATE:
            if cell is None:
                return False
            return cell.end_fate.name.lower() == self.pattern.lower()

        if self.criterion == RuleCriterion.EXPRESSION:
            lo, hi = _parse_range(self.pattern)
            return lo <= float(nuc.rweight) <= hi

        return False

    def resolve_color(
        self,
        nuc: Nucleus,
        cell: Cell | None,
    ) -> tuple[float, float, float, float]:
        """Compute the RGBA color for a matching nucleus.

        For ``SOLID`` mode, returns the fixed ``self.color``.
        For ``COLORMAP`` mode, maps the expression value through
        the named matplotlib colormap.
        """
        if self.color_mode == ColorMode.SOLID:
            return self.color

        # COLORMAP mode — map expression value
        value = float(nuc.rweight)
        return _colormap_rgba(value, self.vmin, self.vmax, self.colormap)


# ── Built-in rule presets ────────────────────────────────────────


def lineage_depth_rules(max_depth: int = 10) -> list[ColorRule]:
    """Generate rainbow-colored rules for lineage depth 0..max_depth."""
    rules = []
    for d in range(max_depth + 1):
        hue = d / (max_depth + 1)
        r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
        rules.append(ColorRule(
            name=f"Depth {d}",
            criterion=RuleCriterion.LINEAGE_DEPTH,
            pattern=f"{d}-{d}",
            color_mode=ColorMode.SOLID,
            color=(r, g, b, 0.85),
            priority=10,
        ))
    return rules


def expression_colormap_rule(
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 5000.0,
) -> ColorRule:
    """Create a rule that colors all nuclei by expression level."""
    return ColorRule(
        name=f"Expression ({cmap})",
        criterion=RuleCriterion.ALL,
        pattern="",
        color_mode=ColorMode.COLORMAP,
        colormap=cmap,
        vmin=vmin,
        vmax=vmax,
        priority=0,
    )


PRESET_EDITING = "editing"
PRESET_LINEAGE_DEPTH = "lineage_depth"
PRESET_EXPRESSION = "expression"

PRESET_NAMES = {
    PRESET_EDITING: "Editing (status colors)",
    PRESET_LINEAGE_DEPTH: "Lineage depth (rainbow)",
    PRESET_EXPRESSION: "Expression (viridis)",
}


# ── Color Rule Engine ────────────────────────────────────────────


class ColorRuleEngine:
    """Evaluates color rules against nuclei to produce RGBA colors.

    The engine maintains an ordered list of rules and provides methods
    to query colors for individual nuclei or batches.

    Attributes:
        rules: The active rule list, evaluated in priority order.
        default_color: RGBA fallback for nuclei matching no rule.
        selected_color: Override color for the currently selected cell.
    """

    def __init__(self) -> None:
        self.rules: list[ColorRule] = []
        self.default_color: tuple[float, float, float, float] = (
            1.0, 1.0, 1.0, 0.3
        )
        self.selected_color: tuple[float, float, float, float] = (
            1.0, 1.0, 1.0, 1.0
        )
        # Cache: cell name → Cell object (refreshed per frame)
        self._cell_cache: dict[str, Cell | None] = {}
        self._cell_cache_time: int = -1

    def set_rules(self, rules: list[ColorRule]) -> None:
        """Replace the active rule list."""
        self.rules = sorted(rules, key=lambda r: -r.priority)

    def load_preset(self, preset: str) -> None:
        """Load a named preset rule set.

        Args:
            preset: One of ``PRESET_LINEAGE_DEPTH``, ``PRESET_EXPRESSION``.
        """
        if preset == PRESET_LINEAGE_DEPTH:
            self.set_rules(lineage_depth_rules())
        elif preset == PRESET_EXPRESSION:
            self.set_rules([expression_colormap_rule()])
        else:
            self.set_rules([])

    def color_for_nucleus(
        self,
        nuc: Nucleus,
        manager: NucleiManager,
        time: int,
        is_selected: bool = False,
    ) -> tuple[float, float, float, float]:
        """Compute the display color for a single nucleus.

        If *is_selected* is True, returns ``self.selected_color``
        regardless of rules (the selection highlight always wins).
        """
        if is_selected:
            return self.selected_color

        cell = self._lookup_cell(nuc, manager, time)

        for rule in self.rules:
            if rule.matches(nuc, cell):
                return rule.resolve_color(nuc, cell)

        return self.default_color

    def colors_for_frame(
        self,
        nuclei: list[Nucleus],
        manager: NucleiManager,
        time: int,
        selected_name: str = "",
    ) -> list[tuple[float, float, float, float]]:
        """Compute colors for all nuclei in a frame.

        Args:
            nuclei: Alive nuclei at *time*.
            manager: The NucleiManager (for cell lookups).
            time: Current timepoint (1-based).
            selected_name: Name of the selected cell (highlighted white).

        Returns:
            List of RGBA tuples, one per nucleus in *nuclei*.
        """
        # Refresh cell cache if timepoint changed
        if time != self._cell_cache_time:
            self._cell_cache.clear()
            self._cell_cache_time = time

        colors = []
        for nuc in nuclei:
            ename = nuc.effective_name or ""
            is_sel = bool(ename and ename == selected_name)
            colors.append(
                self.color_for_nucleus(nuc, manager, time, is_selected=is_sel)
            )
        return colors

    def _lookup_cell(
        self,
        nuc: Nucleus,
        manager: NucleiManager,
        time: int,
    ) -> Cell | None:
        """Look up the Cell for a nucleus, with per-frame caching."""
        ename = nuc.effective_name or ""
        if not ename:
            return None
        if ename in self._cell_cache:
            return self._cell_cache[ename]
        cell = manager.get_cell(ename)
        self._cell_cache[ename] = cell
        return cell


# ── Helpers ──────────────────────────────────────────────────────


def _parse_range(s: str) -> tuple[float, float]:
    """Parse a ``"min-max"`` range string, returning ``(lo, hi)``."""
    parts = s.split("-", 1)
    try:
        lo = float(parts[0])
        hi = float(parts[1]) if len(parts) > 1 else lo
    except (ValueError, IndexError):
        return (0.0, 0.0)
    return (lo, hi)


def _colormap_rgba(
    value: float,
    vmin: float,
    vmax: float,
    cmap_name: str,
) -> tuple[float, float, float, float]:
    """Map a value to RGBA via a matplotlib colormap."""
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    try:
        from .lineage_layout import _matplotlib_color

        r, g, b = _matplotlib_color(t, cmap_name)
        return (r, g, b, 0.85)
    except Exception:
        return (0.5, 0.5, 0.5, 0.5)


def _hsv_to_rgb(
    h: float, s: float, v: float
) -> tuple[float, float, float]:
    """Convert HSV (all 0-1) to RGB (all 0-1)."""
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)
