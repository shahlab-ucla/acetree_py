"""Parity tests for the isolated-fragment loop before classifier scanning."""

from __future__ import annotations

import pytest

from acetree_py.tracking.api import TrackEdge
from acetree_py.tracking.starrynite.legacy_isolated import (
    LegacyIsolatedFragmentParameters,
    apply_legacy_isolated_fragment_prepass,
)
from acetree_py.tracking.starrynite.legacy_state import (
    LegacyFeatureParameters,
    LegacyNucleus,
    LegacyTrackingContext,
)
from acetree_py.tracking.starrynite.oracle.event_trace import (
    checkpoint_from_legacy_context,
    legacy_context_from_checkpoint,
)


def _nucleus(identifier: str, frame: int, row: int, x: float) -> LegacyNucleus:
    return LegacyNucleus(
        identifier,
        frame,
        row,
        (x, 0.0, 0.0),
        4.0,
        10.0,
        2.0,
        1.0,
        1.0,
        2,
        2.0,
        1.0,
    )


def _edge(source: str, target: str, slot: int = 0) -> TrackEdge:
    return TrackEdge(
        source,
        target,
        0.0,
        "split" if slot == 1 else "link",
        {"LEGACY_SUCCESSOR_SLOT": slot},
    )


def _context(
    nuclei: list[LegacyNucleus],
    edges: list[TrackEdge],
    end_frame: int,
) -> LegacyTrackingContext:
    return LegacyTrackingContext.from_nuclei_and_edges(
        nuclei,
        edges,
        LegacyFeatureParameters(
            interval=1.0,
            candidate_cutoff=2.0,
            temporal_cutoff=max(2, end_frame),
            temporal_cutoff_start=2,
            small_cutoff=4.0,
            anisotropy_xyz=(1.0, 1.0, 1.0),
            end_frame=end_frame,
        ),
    )


def _parameters(**overrides) -> LegacyIsolatedFragmentParameters:
    values = {
        "enabled": True,
        "start_frame": 1,
        "end_frame": 3,
        "fp_size_threshold": 3,
        "early_cell_threshold": 250,
        "fp_size_threshold_small": 0,
    }
    values.update(overrides)
    return LegacyIsolatedFragmentParameters(**values)


def test_short_single_successor_fragment_clears_parent_slot_and_keeps_stale_pred() -> None:
    context = _context(
        [
            _nucleus("p", 1, 0, 0.0),
            _nucleus("d", 2, 0, 1.0),
            _nucleus("tail", 3, 0, 2.0),
        ],
        [_edge("p", "d"), _edge("d", "tail")],
        3,
    )

    result = apply_legacy_isolated_fragment_prepass(context, _parameters())

    assert result.context.successor_slots("p") == (None, None)
    assert result.context.predecessor("d") == "p"
    assert result.context.stale_predecessor_by_id == {"d": "p"}
    assert result.context.successor_slots("d") == ("tail", None)
    assert result.context.deleted_ids == {"p", "d", "tail"}
    assert result.context.to_lineage_graph_state().edges == ()


def test_short_first_branch_shifts_slot_two_without_clearing_deleted_root_pred() -> None:
    context = _context(
        [
            _nucleus("p", 1, 0, 0.0),
            _nucleus("short", 2, 0, -1.0),
            _nucleus("keep", 2, 1, 1.0),
            _nucleus("keep_tail", 3, 0, 2.0),
        ],
        [
            _edge("p", "short", 0),
            _edge("p", "keep", 1),
            _edge("keep", "keep_tail"),
        ],
        3,
    )

    result = apply_legacy_isolated_fragment_prepass(
        context,
        _parameters(fp_size_threshold=0, fp_size_threshold_small=1),
    )

    assert result.context.successor_slots("p") == ("keep", None)
    assert result.context.predecessor("keep") == "p"
    assert result.context.predecessor("short") == "p"
    assert result.context.stale_predecessor_by_id == {"short": "p"}
    assert "short" in result.context.deleted_ids
    assert "p" not in result.context.deleted_ids
    assert result.deletions[0].shifted_successor_id == "keep"


def test_totally_isolated_row_is_deleted_without_inventing_a_pointer() -> None:
    context = _context(
        [
            _nucleus("isolated", 1, 0, 0.0),
            _nucleus("last", 3, 0, 0.0),
        ],
        [],
        3,
    )

    result = apply_legacy_isolated_fragment_prepass(context, _parameters())

    assert "isolated" in result.context.deleted_ids
    # The legacy loop stops at endtime-1, so the final-frame row is untouched.
    assert "last" not in result.context.deleted_ids
    assert result.context.stale_predecessor_by_id == {}
    assert result.deletions[0].totally_isolated


@pytest.mark.parametrize(
    ("fp_threshold", "early_threshold"),
    ((2, 250), (3, 0)),
)
def test_strict_size_and_early_cell_thresholds_do_not_delete(
    fp_threshold: int,
    early_threshold: int,
) -> None:
    context = _context(
        [
            _nucleus("p", 1, 0, 0.0),
            _nucleus("peer", 1, 1, 20.0),
            _nucleus("d", 2, 0, 1.0),
            _nucleus("tail", 3, 0, 2.0),
        ],
        [_edge("p", "d"), _edge("d", "tail")],
        3,
    )

    result = apply_legacy_isolated_fragment_prepass(
        context,
        _parameters(
            fp_size_threshold=fp_threshold,
            early_cell_threshold=early_threshold,
            fp_size_threshold_small=1,
        ),
    )

    assert "p" not in result.context.deleted_ids
    assert result.context.successor_slots("p") == ("d", None)


def test_deleted_row_stale_predecessor_roundtrips_through_raw_checkpoint() -> None:
    template = _context(
        [
            _nucleus("p", 1, 0, 0.0),
            _nucleus("short", 2, 0, -1.0),
            _nucleus("keep", 2, 1, 1.0),
        ],
        [_edge("p", "short", 0), _edge("p", "keep", 1)],
        2,
    )
    result = apply_legacy_isolated_fragment_prepass(
        template,
        LegacyIsolatedFragmentParameters(
            enabled=True,
            start_frame=1,
            end_frame=2,
            fp_size_threshold=0,
            early_cell_threshold=0,
            fp_size_threshold_small=1,
        ),
    )
    checkpoint = checkpoint_from_legacy_context(
        result.context,
        0,
        "pre_classification",
    )

    reconstructed = legacy_context_from_checkpoint(template, checkpoint)

    assert checkpoint_from_legacy_context(
        reconstructed,
        0,
        "pre_classification",
    ) == checkpoint
    assert reconstructed.stale_predecessor_by_id == {"short": "p"}
