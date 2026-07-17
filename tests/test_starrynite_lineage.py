"""Unit tests for immutable StarryNite bifurcation resolution."""

from __future__ import annotations

import pytest

from acetree_py.tracking.api import Detection, TrackEdge
from acetree_py.tracking.starrynite.lineage import (
    BifurcationDecision,
    FalseNegativeRewirePlan,
    LineageGraphState,
    LineageReattachmentCandidate,
    LineageResolutionError,
    resolve_bifurcation,
)


def _detection(identifier: str, frame: int) -> Detection:
    return Detection(identifier, frame, float(frame), 0.0, 0.0, 1.0, 1.0)


def _edge(source: str, target: str, *, kind: str = "link", cost: float = 1.0):
    return TrackEdge(source, target, cost, kind)


def _pairs(state: LineageGraphState) -> set[tuple[str, str, str]]:
    return {(edge.source_id, edge.target_id, edge.kind) for edge in state.edges}


def _simple_division_state(*extra: Detection) -> LineageGraphState:
    detections = (
        _detection("parent", 2),
        _detection("daughter1", 3),
        _detection("daughter2", 3),
        *extra,
    )
    return LineageGraphState.from_detections(
        detections,
        (
            _edge("parent", "daughter1", kind="split"),
            _edge("parent", "daughter2", kind="split"),
        ),
    )


def test_graph_state_rejects_merges_backward_edges_and_three_successors():
    detections = tuple(_detection(identifier, frame) for identifier, frame in (
        ("a", 1),
        ("b", 1),
        ("c", 2),
        ("d", 2),
        ("e", 2),
    ))

    with pytest.raises(LineageResolutionError, match="merge"):
        LineageGraphState.from_detections(
            detections,
            (_edge("a", "c"), _edge("b", "c")),
        )
    with pytest.raises(LineageResolutionError, match="forward"):
        LineageGraphState.from_detections(detections, (_edge("c", "a"),))
    with pytest.raises(LineageResolutionError, match="more than two"):
        LineageGraphState.from_detections(
            detections,
            (_edge("a", "c"), _edge("a", "d"), _edge("a", "e")),
        )


def test_class_one_preserves_both_daughters_and_input_state():
    state = _simple_division_state()
    result = resolve_bifurcation(
        state,
        BifurcationDecision(1, "parent", "daughter1", "daughter2"),
    )

    assert result.state is state
    assert result.actions == ()
    assert result.diagnostics.classification == 1
    assert _pairs(result.state) == {
        ("parent", "daughter1", "split"),
        ("parent", "daughter2", "split"),
    }


def test_class_three_deletes_shorter_branch_and_retains_deleted_ids():
    state = _simple_division_state(
        _detection("d1_end", 4),
        _detection("d2_mid", 4),
        _detection("d2_end", 5),
    )
    state = LineageGraphState(
        state.frames,
        (
            *state.edges,
            _edge("daughter1", "d1_end"),
            _edge("daughter2", "d2_mid"),
            _edge("d2_mid", "d2_end"),
        ),
    )

    result = resolve_bifurcation(
        state,
        BifurcationDecision(3, "parent", "daughter1", "daughter2"),
    )

    assert result.state.deleted_ids == frozenset({"daughter1", "d1_end"})
    assert result.diagnostics.selected_daughter_id == "daughter1"
    assert _pairs(result.state) == {
        ("parent", "daughter2", "link"),
        ("daughter2", "d2_mid", "link"),
        ("d2_mid", "d2_end", "link"),
    }
    # The original immutable state remains untouched.
    assert state.deleted_ids == frozenset()
    assert state.edge("parent", "daughter1") is not None


def test_class_three_tie_deletes_daughter_one_like_matlab():
    state = _simple_division_state()
    result = resolve_bifurcation(
        state,
        BifurcationDecision(3, "parent", "daughter1", "daughter2"),
    )

    assert result.state.deleted_ids == frozenset({"daughter1"})
    assert result.state.successors("parent") == ("daughter2",)


def test_class_three_follows_legacy_slot_zero_through_nested_split():
    state = _simple_division_state(
        _detection("d1_slot0", 4),
        _detection("d1_slot1", 4),
        _detection("d2_mid", 4),
        _detection("d2_end", 5),
    )
    state = LineageGraphState(
        state.frames,
        (
            *state.edges,
            TrackEdge(
                "daughter1",
                "d1_slot1",
                1.0,
                "split",
                {"LEGACY_SUCCESSOR_SLOT": 1},
            ),
            TrackEdge(
                "daughter1",
                "d1_slot0",
                1.0,
                "split",
                {"LEGACY_SUCCESSOR_SLOT": 0},
            ),
            _edge("daughter2", "d2_mid"),
            _edge("d2_mid", "d2_end"),
        ),
    )

    result = resolve_bifurcation(
        state,
        BifurcationDecision(3, "parent", "daughter1", "daughter2"),
    )

    assert result.state.deleted_ids == frozenset({"daughter1", "d1_slot0"})
    assert "d1_slot1" in result.state.active_ids
    assert result.state.predecessor("d1_slot1") is None
    assert result.state.successors("parent") == ("daughter2",)


def test_class_zero_detaches_worse_branch_and_reattaches_deterministically():
    state = _simple_division_state(
        _detection("full", 2),
        _detection("full_target1", 3),
        _detection("full_target2", 3),
        _detection("alternative", 2),
    )
    state = LineageGraphState(
        state.frames,
        (
            *state.edges,
            _edge("full", "full_target1", kind="split"),
            _edge("full", "full_target2", kind="split"),
        ),
    )
    decision = BifurcationDecision(
        0,
        "parent",
        "daughter1",
        "daughter2",
        daughter1_nondivision_score=10.0,
        daughter2_nondivision_score=2.0,
        reattachment_candidates=(
            LineageReattachmentCandidate("full", 0.25),
            LineageReattachmentCandidate("alternative", 0.5),
        ),
    )

    result = resolve_bifurcation(state, decision)

    assert result.diagnostics.detached_daughter_id == "daughter1"
    assert result.diagnostics.reattached_to_id == "alternative"
    assert result.diagnostics.skipped_reattachment_candidates == (
        ("full", "source already has two successors"),
    )
    assert _pairs(result.state) == {
        ("parent", "daughter2", "link"),
        ("alternative", "daughter1", "link"),
        ("full", "full_target1", "split"),
        ("full", "full_target2", "split"),
    }


def test_class_zero_score_tie_detaches_daughter_two():
    state = _simple_division_state()
    result = resolve_bifurcation(
        state,
        BifurcationDecision(
            0,
            "parent",
            "daughter1",
            "daughter2",
            daughter1_nondivision_score=4.0,
            daughter2_nondivision_score=4.0,
        ),
    )

    assert result.diagnostics.detached_daughter_id == "daughter2"
    assert result.state.successors("parent") == ("daughter1",)
    assert result.state.predecessor("daughter2") is None


def test_class_two_applies_only_an_explicit_validated_gap_plan():
    state = _simple_division_state(_detection("gap_source", 1))
    gap = _edge("gap_source", "daughter1", kind="gap", cost=3.0)
    plan = FalseNegativeRewirePlan(
        remove_edges=(("parent", "daughter1"),),
        add_edges=(gap,),
        label="simple-one-player-gap",
    )

    result = resolve_bifurcation(
        state,
        BifurcationDecision(
            2,
            "parent",
            "daughter1",
            "daughter2",
            false_negative_plan=plan,
        ),
    )

    assert _pairs(result.state) == {
        ("parent", "daughter2", "link"),
        ("gap_source", "daughter1", "gap"),
    }
    assert result.diagnostics.applied_gap_edges == (("gap_source", "daughter1"),)
    assert result.diagnostics.notes == ("simple-one-player-gap",)

    with pytest.raises(LineageResolutionError, match="explicit"):
        resolve_bifurcation(
            state,
            BifurcationDecision(2, "parent", "daughter1", "daughter2"),
        )


def test_class_two_rejects_a_plan_that_would_create_a_merge():
    state = _simple_division_state(
        _detection("gap_source", 1),
        _detection("second_gap_source", 1),
    )
    plan = FalseNegativeRewirePlan(
        remove_edges=(("parent", "daughter1"),),
        add_edges=(
            _edge("gap_source", "daughter1", kind="gap"),
            _edge("second_gap_source", "daughter1", kind="gap"),
        ),
    )

    with pytest.raises(LineageResolutionError, match="merge"):
        resolve_bifurcation(
            state,
            BifurcationDecision(
                2,
                "parent",
                "daughter1",
                "daughter2",
                false_negative_plan=plan,
            ),
        )
