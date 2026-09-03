"""Parity tests for legacy StarryNite repair-candidate extraction."""

from __future__ import annotations

from dataclasses import replace

import pytest

import acetree_py.tracking.starrynite.repair_candidates as repair_module
from acetree_py.tracking.api import TrackEdge
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
)
from acetree_py.tracking.starrynite.repair_candidates import (
    BackwardRepairCandidate,
    GapScoreResult,
    build_false_negative_rewire_plan,
    enumerate_backward_candidates,
    enumerate_class_zero_candidates,
    enumerate_forward_candidates,
    extract_backward_repair_candidates,
    find_fn_players,
    score_gap_candidate,
)


def _nucleus(
    identifier: str,
    frame: int,
    row: int,
    x: float,
    y: float = 0.0,
    z: float = 0.0,
) -> LegacyNucleus:
    return LegacyNucleus(
        identifier,
        frame,
        row,
        (x, y, z),
        4.0,
        100.0,
        10.0,
        1.0,
        1.0,
        2,
        1.0,
    )


def _parameters(
    end_frame: int,
    *,
    candidate_cutoff: float = 1.0,
    temporal_cutoff: int = 4,
    temporal_cutoff_start: int = 2,
    interval: float = 1.0,
) -> LegacyFeatureParameters:
    return LegacyFeatureParameters(
        interval=interval,
        candidate_cutoff=candidate_cutoff,
        temporal_cutoff=temporal_cutoff,
        temporal_cutoff_start=temporal_cutoff_start,
        small_cutoff=4.0,
        anisotropy_xyz=(1.0, 1.0, 1.0),
        end_frame=end_frame,
    )


def _edge(source: str, target: str) -> TrackEdge:
    return TrackEdge(source, target, 0.0)


def _context(
    nuclei: tuple[LegacyNucleus, ...],
    edges: tuple[TrackEdge, ...] = (),
    *,
    parameters: LegacyFeatureParameters | None = None,
    deleted_ids: tuple[str, ...] = (),
) -> LegacyTrackingContext:
    return LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        edges,
        parameters or _parameters(max(item.frame for item in nuclei)),
        deleted_ids=deleted_ids,
    )


class _InjectedContext:
    """Small topology-injection harness for clean conflict fixtures."""

    def __init__(
        self,
        nuclei: tuple[LegacyNucleus, ...],
        slots: dict[str, tuple[str | None, str | None]],
        *,
        f_nn: dict[str, str | None] | None = None,
        b_nn: dict[str, str | None] | None = None,
        predecessor_suitors: dict[str, tuple[str, ...]] | None = None,
        successor_suitors: dict[str, tuple[str, ...]] | None = None,
        deleted_ids: tuple[str, ...] = (),
    ) -> None:
        self._nuclei = {item.nucleus_id: item for item in nuclei}
        self._frames: dict[int, tuple[str, ...]] = {}
        for frame in sorted({item.frame for item in nuclei}):
            self._frames[frame] = tuple(
                item.nucleus_id
                for item in sorted(
                    (item for item in nuclei if item.frame == frame),
                    key=lambda item: item.matlab_row,
                )
            )
        self._slots = {item.nucleus_id: (None, None) for item in nuclei}
        self._slots.update(slots)
        self._predecessors = {item.nucleus_id: None for item in nuclei}
        for source, targets in self._slots.items():
            for target in targets:
                if target is not None:
                    self._predecessors[target] = source
        self._f_nn = f_nn or {}
        self._b_nn = b_nn or {}
        self._predecessor_suitors = predecessor_suitors or {}
        self._successor_suitors = successor_suitors or {}
        self.deleted_ids = frozenset(deleted_ids)

    def nucleus(self, identifier: str) -> LegacyNucleus:
        return self._nuclei[identifier]

    def frame_ids(self, frame: int, *, include_deleted: bool = False) -> tuple[str, ...]:
        values = self._frames.get(frame, ())
        if include_deleted:
            return values
        return tuple(item for item in values if item not in self.deleted_ids)

    def successor_slots(self, identifier: str) -> tuple[str | None, str | None]:
        return self._slots[identifier]

    def predecessor(self, identifier: str) -> str | None:
        return self._predecessors[identifier]

    def f_nn(self, identifier: str) -> str | None:
        return self._f_nn.get(identifier)

    def b_nn(self, identifier: str) -> str | None:
        return self._b_nn.get(identifier)

    def predecessor_suitors(self, identifier: str) -> tuple[str, ...]:
        return self._predecessor_suitors.get(identifier, ())

    def successor_suitors(self, identifier: str) -> tuple[str, ...]:
        return self._successor_suitors.get(identifier, ())

    def mean_self_distance(self, _frame: int) -> float:
        return 1.0


def test_backward_candidates_use_strict_cutoff_offset_then_row_order():
    nuclei = (
        _nucleus("f1-near", 1, 0, 0.0),
        _nucleus("f1-boundary", 1, 1, 10.0),
        _nucleus("f2-near", 2, 0, 0.0),
        _nucleus("f2-boundary", 2, 1, 10.0),
        _nucleus("target", 4, 0, 0.0),
        _nucleus("target-peer", 4, 1, 20.0),
    )
    context = _context(nuclei, parameters=_parameters(4, temporal_cutoff=3))

    candidates = enumerate_backward_candidates(
        context,
        "target",
        context.parameters,
        daughter_index=1,
    )

    assert [(item.source_id, item.offset) for item in candidates] == [
        ("f2-near", 2),
        ("f1-near", 3),
    ]
    assert [item.local_index for item in candidates] == [1, 2]
    assert all(item.endpoint_distance == 0.0 for item in candidates)


def test_backward_candidates_exclude_deleted_and_any_source_with_suc1():
    nuclei = (
        _nucleus("loose", 1, 0, 0.0),
        _nucleus("occupied", 1, 1, 1.0),
        _nucleus("deleted", 1, 2, 2.0),
        _nucleus("occupied-child", 2, 0, 1.0),
        _nucleus("target", 3, 0, 0.0),
        _nucleus("target-peer", 3, 1, 10.0),
    )
    context = _context(
        nuclei,
        (_edge("occupied", "occupied-child"),),
        parameters=_parameters(3, candidate_cutoff=10.0, temporal_cutoff=2),
        deleted_ids=("deleted",),
    )

    candidates = enumerate_backward_candidates(
        context, "target", context.parameters, daughter_index=1
    )

    assert tuple(item.source_id for item in candidates) == ("loose",)


def test_backward_selection_prefers_daughter_one_on_exact_score_tie():
    nuclei = (
        _nucleus("source-a", 1, 0, 0.0),
        _nucleus("source-b", 1, 1, 10.0),
        _nucleus("daughter1", 3, 0, 0.0),
        _nucleus("daughter2", 3, 1, 0.0),
        _nucleus("peer", 3, 2, 20.0),
    )
    context = _context(
        nuclei,
        parameters=_parameters(3, candidate_cutoff=2.0, temporal_cutoff=2),
    )

    result = extract_backward_repair_candidates(
        context, "daughter1", "daughter2", context.parameters
    )

    assert result.selected is not None
    assert result.selected.daughter_index == 1
    assert result.selected.source_id == "source-a"
    assert result.selected.enumeration_index == 1
    assert result.max_daughter1_branch_length == 1.0
    assert result.max_daughter2_branch_length == 1.0


def _clean2_context() -> _InjectedContext:
    nuclei = (
        _nucleus("s1", 1, 0, 0.0),
        _nucleus("s2", 1, 1, 0.0),
        _nucleus("middle", 2, 0, 0.0),
        _nucleus("parent", 3, 0, 0.0),
        _nucleus("e1", 4, 0, 0.0),
        _nucleus("e2", 4, 1, 0.0),
    )
    return _InjectedContext(
        nuclei,
        {
            "s2": ("middle", None),
            "middle": ("parent", None),
            "parent": ("e1", "e2"),
        },
        f_nn={"s1": "middle"},
        b_nn={"e1": "parent"},
        predecessor_suitors={"middle": ("s1", "s2")},
        successor_suitors={"parent": ("e1", "e2")},
    )


def test_clean_two_player_gap_tie_uses_swapped_assignment():
    context = _clean2_context()
    parameters = _parameters(4)

    players = find_fn_players(context, "s1", "e1")  # type: ignore[arg-type]
    score = score_gap_candidate(
        context, "s1", "e1", parameters  # type: ignore[arg-type]
    )

    assert players.is_clean
    assert players.topology == "clean2"
    assert players.start_backtrace is None
    assert players.end_forward_trace is None
    assert score.matching == (2, 1, 0)
    assert score.score == 0.0


def test_clean_two_player_score_retains_matlab_single_accumulation(
    monkeypatch: pytest.MonkeyPatch,
):
    context = _clean2_context()
    distances = {
        ("s1", "e1"): 16_777_216.0,
        ("s2", "e2"): 1.0,
        ("s1", "e2"): 16_777_216.0,
        ("s2", "e1"): 2.0,
    }

    def injected_distance(first, second, _anisotropy):
        return distances[(first.nucleus_id, second.nucleus_id)]

    monkeypatch.setattr(repair_module, "anisotropic_distance", injected_distance)

    score = score_gap_candidate(
        context, "s1", "e1", _parameters(4)  # type: ignore[arg-type]
    )

    # At this magnitude binary32 cannot retain the +1 before division. A
    # widened double-precision implementation would return 8_388_608.5.
    assert score.unnormalized_score == 8_388_608.0
    assert score.score == 8_388_608.0
    assert score.matching == (1, 2, 0)


def test_frozen_deleted_nearest_neighbor_invalidates_trace_but_not_suitor_lookup():
    nuclei = (
        _nucleus("source", 1, 0, 0.0),
        _nucleus("deleted-middle", 2, 0, 0.0),
        _nucleus("target", 3, 0, 0.0),
    )
    context = _InjectedContext(
        nuclei,
        {},
        f_nn={"source": "deleted-middle"},
        b_nn={"target": "deleted-middle"},
        predecessor_suitors={"deleted-middle": ("source",)},
        successor_suitors={"deleted-middle": ("target",)},
        deleted_ids=("deleted-middle",),
    )

    players = find_fn_players(context, "source", "target")  # type: ignore[arg-type]
    score = score_gap_candidate(
        context, "source", "target", _parameters(3)  # type: ignore[arg-type]
    )

    assert players.start_players == ("source",)
    assert players.end_players == ("target",)
    assert not players.start_trace_valid
    assert not players.end_trace_valid
    assert score.topology == "dirty"


def test_clean_two_player_rewire_tie_assigns_second_claimant_to_middle():
    context = _clean2_context()
    score = score_gap_candidate(
        context, "s1", "e1", _parameters(4)  # type: ignore[arg-type]
    )
    candidate = BackwardRepairCandidate(
        1,
        1,
        1,
        "s1",
        "e1",
        1,
        4,
        3,
        1.0,
        0.0,
        score,
        (1.0, 1.0, 1.0),
    )

    plan = build_false_negative_rewire_plan(
        context, "parent", "e1", "e2", candidate  # type: ignore[arg-type]
    )

    assert plan.remove_edges == (("parent", "e2"),)
    assert {(edge.source_id, edge.target_id, edge.kind) for edge in plan.add_edges} == {
        ("s1", "e2", "gap")
    }
    assert tuple(
        edge.features["LEGACY_SUCCESSOR_SLOT"] for edge in plan.add_edges
    ) == (0,)
    assert plan.label == "legacy-clean2-fn-rewire"


def test_clean_score_is_demoted_to_simple_when_no_start_link_is_present():
    context = _clean2_context()
    # Retain the scored clean player sets but remove the only middle claimant.
    context._slots["s2"] = (None, None)
    score = GapScoreResult(
        0.5,
        0.5,
        1.0,
        "clean2",
        (2, 1, 0),
        ("s1", "s2", None),
        ("e1", "e2", None),
    )
    candidate = BackwardRepairCandidate(
        1,
        1,
        1,
        "s1",
        "e1",
        1,
        4,
        3,
        1.0,
        0.0,
        score,
        (1.0, 1.0, 1.0),
    )

    plan = build_false_negative_rewire_plan(
        context, "parent", "e1", "e2", candidate  # type: ignore[arg-type]
    )

    assert plan.remove_edges == (("parent", "e1"),)
    assert [(edge.source_id, edge.target_id) for edge in plan.add_edges] == [
        ("s1", "e1")
    ]
    assert plan.label == "legacy-clean2-demoted-to-simple-fn-rewire"


def _clean3_context() -> _InjectedContext:
    nuclei = (
        _nucleus("s1", 1, 0, 0.0),
        _nucleus("s2", 1, 1, 0.0),
        _nucleus("s3", 1, 2, 0.0),
        _nucleus("m1", 2, 0, 0.0),
        _nucleus("n1", 2, 1, 0.0),
        _nucleus("m2", 3, 0, 0.0),
        _nucleus("parent", 3, 1, 0.0),
        _nucleus("e1", 4, 0, 0.0),
        _nucleus("e2", 4, 1, 0.0),
        _nucleus("e3", 4, 2, 0.0),
    )
    return _InjectedContext(
        nuclei,
        {
            "s2": ("m1", None),
            "s3": ("n1", None),
            "m1": ("m2", None),
            "n1": ("parent", None),
            "m2": ("e3", None),
            "parent": ("e1", "e2"),
        },
        f_nn={"s1": "m1"},
        b_nn={"e1": "parent"},
        predecessor_suitors={"m1": ("s1", "s2")},
        successor_suitors={"parent": ("e1", "e2")},
    )


def test_clean_three_player_tie_uses_first_permutation():
    context = _clean3_context()

    score = score_gap_candidate(
        context, "s1", "e1", _parameters(4)  # type: ignore[arg-type]
    )

    assert score.topology == "clean3"
    assert score.start_players == ("s1", "s2", "s3")
    assert score.end_players == ("e1", "e2", "e3")
    assert score.matching == (1, 2, 3)


def test_clean_three_player_rewire_transfers_middle_link_when_loose_maps_to_third():
    context = _clean3_context()
    original = score_gap_candidate(
        context, "s1", "e1", _parameters(4)  # type: ignore[arg-type]
    )
    score = replace(original, matching=(3, 2, 1))
    candidate = BackwardRepairCandidate(
        1,
        1,
        1,
        "s1",
        "e1",
        1,
        4,
        3,
        1.0,
        0.0,
        score,
        (1.0, 1.0, 1.0),
    )

    plan = build_false_negative_rewire_plan(
        context, "parent", "e1", "e2", candidate  # type: ignore[arg-type]
    )

    assert set(plan.remove_edges) == {("s2", "m1"), ("parent", "e2")}
    assert {(edge.source_id, edge.target_id, edge.kind) for edge in plan.add_edges} == {
        ("s1", "m1", "link"),
        ("s2", "e2", "gap"),
    }


def test_dirty_simple_rewire_handles_each_daughter_slot_exactly():
    context = _clean2_context()
    dirty = GapScoreResult(
        0.25,
        0.25,
        1.0,
        "dirty",
        (1, 0, 0),
        ("s1", None, None),
        ("e1", None, None),
    )
    first = BackwardRepairCandidate(
        1, 1, 1, "s1", "e1", 1, 4, 3, 1.0, 0.0, dirty, (1.0, 1.0, 1.0)
    )
    second = replace(
        first,
        daughter_index=2,
        target_id="e2",
        gap=replace(dirty, end_players=("e2", None, None)),
    )

    first_plan = build_false_negative_rewire_plan(
        context, "parent", "e1", "e2", first  # type: ignore[arg-type]
    )
    second_plan = build_false_negative_rewire_plan(
        context, "parent", "e1", "e2", second  # type: ignore[arg-type]
    )

    assert first_plan.remove_edges == (("parent", "e1"),)
    assert [(edge.source_id, edge.target_id) for edge in first_plan.add_edges] == [
        ("s1", "e1")
    ]
    assert second_plan.remove_edges == (("parent", "e2"),)
    assert [(edge.source_id, edge.target_id) for edge in second_plan.add_edges] == [
        ("s1", "e2")
    ]


def test_class_zero_uses_four_raw_nearest_and_skips_consume_attempts():
    nuclei = (
        _nucleus("parent", 2, 0, 0.0),
        _nucleus("full", 2, 1, 1.0),
        _nucleus("loose", 2, 2, 2.0),
        _nucleus("one-child", 2, 3, 3.0),
        _nucleus("fifth", 2, 4, 4.0),
        _nucleus("detached", 3, 0, 0.0),
        _nucleus("kept", 3, 1, 20.0),
        _nucleus("full-1", 3, 2, 21.0),
        _nucleus("full-2", 3, 3, 22.0),
        _nucleus("one-target", 3, 4, 23.0),
    )
    context = _context(
        nuclei,
        (
            _edge("parent", "detached"),
            _edge("parent", "kept"),
            _edge("full", "full-1"),
            _edge("full", "full-2"),
            _edge("one-child", "one-target"),
        ),
        parameters=_parameters(3),
    )

    candidates = enumerate_class_zero_candidates(
        context, "parent", "detached", context.parameters
    )

    assert [(item.source_id, item.eligibility) for item in candidates] == [
        ("parent", "original_parent"),
        ("full", "already_has_two_successors"),
        ("loose", "direct_attach"),
        ("one-child", "tentative_bifurcation"),
    ]
    assert all(item.source_id != "fifth" for item in candidates)


def test_class_zero_keeps_deleted_rows_and_matlab_row_ties():
    nuclei = (
        _nucleus("parent", 1, 0, 10.0),
        _nucleus("deleted", 1, 1, -1.0),
        _nucleus("live", 1, 2, 1.0),
        _nucleus("spare", 1, 3, 20.0),
        _nucleus("detached", 2, 0, 0.0),
        _nucleus("kept", 2, 1, 30.0),
    )
    context = _context(
        nuclei,
        (_edge("parent", "detached"), _edge("parent", "kept")),
        parameters=_parameters(2),
        deleted_ids=("deleted",),
    )

    candidates = enumerate_class_zero_candidates(
        context, "parent", "detached", context.parameters
    )

    assert candidates[0].source_id == "deleted"
    assert candidates[0].deleted
    assert candidates[0].eligibility == "direct_attach"
    assert candidates[1].source_id == "live"


def test_forward_candidates_apply_root_and_immediate_division_gates_and_raw_length_override():
    nuclei = (
        _nucleus("source", 1, 0, 0.0),
        _nucleus("frame1-peer", 1, 1, 20.0),
        _nucleus("ordinary-parent", 2, 0, 10.0),
        _nucleus("division-parent", 2, 1, 0.0),
        _nucleus("root", 3, 0, 5.0),
        _nucleus("ordinary-child", 3, 1, 4.0),
        _nucleus("division-d1", 3, 2, 0.1),
        _nucleus("division-d2", 3, 3, 2.0),
        _nucleus("gap-child", 3, 4, 3.0),
        _nucleus("d1-mid", 4, 0, 0.1),
        _nucleus("d2-end", 4, 1, 2.0),
        _nucleus("d1-end", 5, 0, 0.1),
    )
    context = _context(
        nuclei,
        (
            _edge("ordinary-parent", "ordinary-child"),
            _edge("division-parent", "division-d1"),
            _edge("division-parent", "division-d2"),
            _edge("frame1-peer", "gap-child"),
            _edge("division-d1", "d1-mid"),
            _edge("d1-mid", "d1-end"),
            _edge("division-d2", "d2-end"),
        ),
        parameters=_parameters(
            5,
            candidate_cutoff=100.0,
            temporal_cutoff=4,
            interval=2.0,
        ),
    )

    result = enumerate_forward_candidates(context, "source", context.parameters)

    assert [item.target_id for item in result.candidates[:3]] == [
        "root",
        "division-d1",
        "division-d2",
    ]
    assert "ordinary-child" not in {item.target_id for item in result.candidates}
    assert "gap-child" not in {item.target_id for item in result.candidates}
    assert result.selected is not None
    assert result.selected.target_id == "division-d1"
    assert result.selected.target_branch_length == 1.5
    # Legacy overwrites a division-origin candidate with min branch length in
    # raw frames rather than interval-normalized units.
    assert result.selected_feature_length == 2.0


def test_daughter_two_scoring_uses_its_row_in_daughter_one_frame():
    nuclei = (
        _nucleus("source", 2, 0, 0.0),
        _nucleus("source-peer", 2, 1, -10.0),
        _nucleus("daughter1", 3, 0, -10.0),
        _nucleus("d2-row-proxy", 3, 1, 100.0),
        _nucleus("frame4-peer", 4, 0, 20.0),
        _nucleus("daughter2", 4, 1, 0.0),
    )
    context = _context(
        nuclei,
        parameters=_parameters(
            4,
            candidate_cutoff=1.0,
            temporal_cutoff=2,
        ),
    )

    result = extract_backward_repair_candidates(
        context, "daughter1", "daughter2", context.parameters
    )

    assert len(result.daughter2) == 1
    candidate = result.daughter2[0]
    assert candidate.source_id == "source"
    assert candidate.target_id == "daughter2"
    assert candidate.endpoint_distance == 0.0
    # MATLAB's computeBestFNBackOption pairs d2's row with tcur1, so gapScore
    # sees the row-1 proxy at x=100 rather than actual d2 at x=0.
    assert candidate.gap.end_players[0] == "d2-row-proxy"
    assert candidate.gap.unnormalized_score == 100.0
    assert candidate.score == 10.0
