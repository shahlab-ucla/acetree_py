"""Tests for the color rule engine (acetree_py.gui.color_rules)."""

from __future__ import annotations

import pytest

from acetree_py.gui.color_rules import (
    ColorMode,
    ColorRule,
    ColorRuleEngine,
    RuleCriterion,
    PRESET_LINEAGE_DEPTH,
    PRESET_EXPRESSION,
    lineage_depth_rules,
    expression_colormap_rule,
    _parse_range,
    _hsv_to_rgb,
)


# ── Lightweight stubs ────────────────────────────────────────────


class _StubNucleus:
    """Minimal Nucleus-like object for rule matching tests."""

    def __init__(
        self,
        effective_name: str = "ABa",
        identity: str = "",
        rweight: int = 1000,
        index: int = 1,
    ):
        self.effective_name = effective_name
        self.identity = identity or effective_name
        self.rweight = rweight
        self.index = index


class _StubCell:
    """Minimal Cell-like object for rule matching tests."""

    def __init__(self, name: str = "ABa", depth_val: int = 2, fate_name: str = "ALIVE"):
        self.name = name
        self._depth = depth_val

        class _Fate:
            pass
        f = _Fate()
        f.name = fate_name
        self.end_fate = f

    def depth(self) -> int:
        return self._depth


class _StubManager:
    """Minimal NucleiManager-like object for engine tests."""

    def __init__(self, cells: dict[str, _StubCell] | None = None):
        self._cells = cells or {}

    def get_cell(self, name: str):
        return self._cells.get(name)


# ── Helpers ──────────────────────────────────────────────────────


class TestParseRange:
    def test_single_value(self):
        assert _parse_range("5") == (5.0, 5.0)

    def test_range(self):
        assert _parse_range("2-8") == (2.0, 8.0)

    def test_bad_input(self):
        assert _parse_range("abc") == (0.0, 0.0)


class TestHsvToRgb:
    def test_red(self):
        r, g, b = _hsv_to_rgb(0.0, 1.0, 1.0)
        assert pytest.approx(r, abs=0.01) == 1.0
        assert pytest.approx(g, abs=0.01) == 0.0
        assert pytest.approx(b, abs=0.01) == 0.0

    def test_green(self):
        r, g, b = _hsv_to_rgb(1 / 3, 1.0, 1.0)
        assert pytest.approx(g, abs=0.01) == 1.0

    def test_gray(self):
        r, g, b = _hsv_to_rgb(0.5, 0.0, 0.7)
        assert r == g == b == pytest.approx(0.7)


# ── Rule matching ────────────────────────────────────────────────


class TestColorRuleMatching:
    def test_all_matches_everything(self):
        rule = ColorRule(criterion=RuleCriterion.ALL)
        nuc = _StubNucleus()
        assert rule.matches(nuc, None)

    def test_disabled_rule_never_matches(self):
        rule = ColorRule(criterion=RuleCriterion.ALL, enabled=False)
        nuc = _StubNucleus()
        assert not rule.matches(nuc, None)

    def test_name_exact(self):
        rule = ColorRule(criterion=RuleCriterion.NAME_EXACT, pattern="ABa")
        assert rule.matches(_StubNucleus("ABa"), None)
        assert not rule.matches(_StubNucleus("ABp"), None)

    def test_name_pattern_wildcard(self):
        rule = ColorRule(criterion=RuleCriterion.NAME_PATTERN, pattern="AB*")
        assert rule.matches(_StubNucleus("ABala"), None)
        assert not rule.matches(_StubNucleus("MSa"), None)

    def test_name_regex(self):
        rule = ColorRule(criterion=RuleCriterion.NAME_REGEX, pattern=r"^AB[ap]$")
        assert rule.matches(_StubNucleus("ABa"), None)
        assert rule.matches(_StubNucleus("ABp"), None)
        assert not rule.matches(_StubNucleus("ABala"), None)

    def test_name_regex_bad_pattern(self):
        rule = ColorRule(criterion=RuleCriterion.NAME_REGEX, pattern="[invalid")
        assert not rule.matches(_StubNucleus("ABa"), None)

    def test_lineage_depth(self):
        rule = ColorRule(criterion=RuleCriterion.LINEAGE_DEPTH, pattern="2-4")
        cell_2 = _StubCell(depth_val=2)
        cell_5 = _StubCell(depth_val=5)
        nuc = _StubNucleus()
        assert rule.matches(nuc, cell_2)
        assert not rule.matches(nuc, cell_5)

    def test_lineage_depth_no_cell(self):
        rule = ColorRule(criterion=RuleCriterion.LINEAGE_DEPTH, pattern="2-4")
        assert not rule.matches(_StubNucleus(), None)

    def test_fate(self):
        rule = ColorRule(criterion=RuleCriterion.FATE, pattern="divided")
        cell_div = _StubCell(fate_name="DIVIDED")
        cell_alive = _StubCell(fate_name="ALIVE")
        nuc = _StubNucleus()
        assert rule.matches(nuc, cell_div)
        assert not rule.matches(nuc, cell_alive)

    def test_expression_range(self):
        rule = ColorRule(criterion=RuleCriterion.EXPRESSION, pattern="500-2000")
        assert rule.matches(_StubNucleus(rweight=1000), None)
        assert not rule.matches(_StubNucleus(rweight=100), None)
        assert not rule.matches(_StubNucleus(rweight=3000), None)


class TestColorRuleResolveColor:
    def test_solid_mode(self):
        rule = ColorRule(
            color_mode=ColorMode.SOLID,
            color=(1.0, 0.0, 0.0, 0.8),
        )
        assert rule.resolve_color(_StubNucleus(), None) == (1.0, 0.0, 0.0, 0.8)

    def test_colormap_mode(self):
        rule = ColorRule(
            color_mode=ColorMode.COLORMAP,
            colormap="viridis",
            vmin=0.0,
            vmax=2000.0,
        )
        color = rule.resolve_color(_StubNucleus(rweight=1000), None)
        assert len(color) == 4
        assert all(0.0 <= c <= 1.0 for c in color)


# ── Engine ───────────────────────────────────────────────────────


class TestColorRuleEngine:
    def test_default_color_when_no_rules(self):
        engine = ColorRuleEngine()
        nuc = _StubNucleus()
        mgr = _StubManager()
        color = engine.color_for_nucleus(nuc, mgr, time=1)
        assert color == engine.default_color

    def test_selected_override(self):
        engine = ColorRuleEngine()
        engine.set_rules([
            ColorRule(
                criterion=RuleCriterion.ALL,
                color=(1.0, 0.0, 0.0, 1.0),
            ),
        ])
        nuc = _StubNucleus()
        mgr = _StubManager()
        # Selected cell always gets white regardless of rules
        color = engine.color_for_nucleus(nuc, mgr, time=1, is_selected=True)
        assert color == engine.selected_color

    def test_first_matching_rule_wins(self):
        engine = ColorRuleEngine()
        engine.set_rules([
            ColorRule(
                name="specific",
                criterion=RuleCriterion.NAME_EXACT,
                pattern="ABa",
                color=(1.0, 0.0, 0.0, 1.0),
                priority=10,
            ),
            ColorRule(
                name="catch-all",
                criterion=RuleCriterion.ALL,
                color=(0.0, 0.0, 1.0, 1.0),
                priority=0,
            ),
        ])
        mgr = _StubManager()
        # ABa matches the specific rule (higher priority)
        assert engine.color_for_nucleus(
            _StubNucleus("ABa"), mgr, time=1
        ) == (1.0, 0.0, 0.0, 1.0)
        # MSa doesn't match specific, falls through to catch-all
        assert engine.color_for_nucleus(
            _StubNucleus("MSa"), mgr, time=1
        ) == (0.0, 0.0, 1.0, 1.0)

    def test_priority_ordering(self):
        engine = ColorRuleEngine()
        # Rules added in wrong order — engine should sort by priority
        engine.set_rules([
            ColorRule(
                criterion=RuleCriterion.ALL,
                color=(0.0, 0.0, 1.0, 1.0),
                priority=0,
            ),
            ColorRule(
                criterion=RuleCriterion.ALL,
                color=(1.0, 0.0, 0.0, 1.0),
                priority=10,
            ),
        ])
        mgr = _StubManager()
        # Higher priority rule wins
        color = engine.color_for_nucleus(_StubNucleus(), mgr, time=1)
        assert color == (1.0, 0.0, 0.0, 1.0)

    def test_colors_for_frame(self):
        engine = ColorRuleEngine()
        engine.set_rules([
            ColorRule(
                criterion=RuleCriterion.NAME_PATTERN,
                pattern="AB*",
                color=(1.0, 0.0, 0.0, 0.8),
                priority=10,
            ),
            ColorRule(
                criterion=RuleCriterion.ALL,
                color=(0.0, 0.0, 1.0, 0.8),
                priority=0,
            ),
        ])
        nuclei = [
            _StubNucleus("ABa"),
            _StubNucleus("MSa"),
            _StubNucleus("ABp"),
        ]
        mgr = _StubManager()
        colors = engine.colors_for_frame(nuclei, mgr, time=1, selected_name="MSa")
        assert colors[0] == (1.0, 0.0, 0.0, 0.8)  # ABa → AB* rule
        assert colors[1] == engine.selected_color   # MSa → selected override
        assert colors[2] == (1.0, 0.0, 0.0, 0.8)  # ABp → AB* rule

    def test_cell_lookup_caching(self):
        cell = _StubCell("ABa", depth_val=3)
        mgr = _StubManager(cells={"ABa": cell})
        engine = ColorRuleEngine()
        engine.set_rules([
            ColorRule(
                criterion=RuleCriterion.LINEAGE_DEPTH,
                pattern="3-3",
                color=(0.0, 1.0, 0.0, 1.0),
            ),
        ])
        nuc = _StubNucleus("ABa")
        color = engine.color_for_nucleus(nuc, mgr, time=1)
        assert color == (0.0, 1.0, 0.0, 1.0)
        # Second call should use cache
        color2 = engine.color_for_nucleus(nuc, mgr, time=1)
        assert color2 == color

    def test_cache_clears_on_time_change(self):
        engine = ColorRuleEngine()
        engine.set_rules([
            ColorRule(
                criterion=RuleCriterion.LINEAGE_DEPTH,
                pattern="0-10",
                color=(0.0, 1.0, 0.0, 1.0),
            ),
        ])
        cell = _StubCell("ABa", depth_val=3)
        mgr = _StubManager(cells={"ABa": cell})
        nuclei = [_StubNucleus("ABa")]
        engine.colors_for_frame(nuclei, mgr, time=1)
        assert engine._cell_cache_time == 1
        engine.colors_for_frame(nuclei, mgr, time=2)
        assert engine._cell_cache_time == 2


# ── Presets ──────────────────────────────────────────────────────


class TestPresets:
    def test_lineage_depth_preset(self):
        engine = ColorRuleEngine()
        engine.load_preset(PRESET_LINEAGE_DEPTH)
        assert len(engine.rules) == 11  # depth 0-10

    def test_expression_preset(self):
        engine = ColorRuleEngine()
        engine.load_preset(PRESET_EXPRESSION)
        assert len(engine.rules) == 1
        assert engine.rules[0].color_mode == ColorMode.COLORMAP

    def test_lineage_depth_rules_colors_unique(self):
        rules = lineage_depth_rules(5)
        colors = [r.color for r in rules]
        # All depth colors should be distinct
        assert len(set(colors)) == len(colors)

    def test_expression_colormap_rule(self):
        rule = expression_colormap_rule("inferno", 100, 3000)
        assert rule.criterion == RuleCriterion.ALL
        assert rule.colormap == "inferno"
        assert rule.vmin == 100
        assert rule.vmax == 3000
