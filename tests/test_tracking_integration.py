"""Tests for atomic acceptance of tracking proposals into nuclei records."""

from __future__ import annotations

from copy import deepcopy

import pytest

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.editing.history import EditHistory
from acetree_py.tracking.api import (
    Calibration,
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
)
from acetree_py.tracking.integration import (
    ApplyTrackingProposal,
    TrackingProposalConflict,
    proposal_to_nucleus_mapping,
)


CALIBRATION = Calibration(xy_um=0.5, z_um=2.0, plane_start=1)


def _request(start: int = 1, end: int = 5) -> TrackingRequest:
    return TrackingRequest(
        detector=ComponentSpec("org.acetree.detector.dog", {"RADIUS": 2.0}),
        tracker=ComponentSpec("org.acetree.tracker.lap", {"MAX_FRAME_GAP": 2}),
        scope=TrackingScope("global", start, end),
    )


def _seed(
    *,
    x: int = 10,
    y: int = 20,
    z: float = 1.0,
    successor1: int = NILLI,
    successor2: int = NILLI,
) -> Nucleus:
    return Nucleus(
        index=1,
        x=x,
        y=y,
        z=z,
        size=8,
        identity="EMS",
        assigned_id="EMS",
        status=1,
        predecessor=NILLI,
        successor1=successor1,
        successor2=successor2,
        weight=17,
    )


def _detection(
    detection_id: str,
    frame: int,
    x_um: float,
    y_um: float = 10.0,
    z_um: float = 0.0,
    radius_um: float = 2.0,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        frame=frame,
        x_um=x_um,
        y_um=y_um,
        z_um=z_um,
        radius_um=radius_um,
        quality=10.0,
    )


def _result(
    detections: tuple[Detection, ...],
    edges: tuple[TrackEdge, ...],
    anchors: dict[str, tuple[int, int]] | None = None,
) -> TrackingResult:
    return TrackingResult(
        request=_request(1, max(d.frame for d in detections)),
        detections=detections,
        edges=edges,
        existing_anchors=anchors or {},
        provenance={"run_id": "integration-test"},
    )


def test_apply_is_one_history_edit_preserves_seed_and_undoes_exactly():
    seed = _seed()
    first_frame = [seed]
    record = [first_frame]
    before = deepcopy(record)
    detections = (
        _detection("seed", 1, 5.0),
        _detection("t2", 2, 6.0, y_um=11.0, z_um=2.0, radius_um=2.5),
        _detection("t3", 3, 7.0, y_um=12.0, z_um=4.0, radius_um=2.0),
    )
    result = _result(
        detections,
        (
            TrackEdge("seed", "t2", 1.0),
            TrackEdge("t2", "t3", 1.0),
        ),
        {"seed": (1, 1)},
    )
    command = ApplyTrackingProposal(result, CALIBRATION)
    history = EditHistory(record)

    history.do(command)

    assert history.num_undoable == 1
    assert record[0] is first_frame
    assert record[0][0] is seed
    assert seed.assigned_id == "EMS"
    assert seed.identity == "EMS"
    assert seed.weight == 17
    assert seed.successor1 == 1
    assert [len(frame) for frame in record] == [1, 1, 1]
    t2 = record[1][0]
    t3 = record[2][0]
    assert (t2.x, t2.y, t2.z, t2.size) == (12, 22, 2.0, 10)
    assert (t3.x, t3.y, t3.z, t3.size) == (14, 24, 3.0, 8)
    assert t2.predecessor == 1 and t2.successor1 == 1
    assert t3.predecessor == 1
    # Automatic naming is run after the structural edit; proposal acceptance
    # must never turn inherited names into manual overrides.
    assert t2.identity == "" and t2.assigned_id == ""
    assert t3.identity == "" and t3.assigned_id == ""
    assert proposal_to_nucleus_mapping(command) == {
        "seed": (1, 1),
        "t2": (2, 1),
        "t3": (3, 1),
    }

    history.undo()

    assert record == before
    assert record[0] is first_frame
    assert record[0][0] is seed
    assert len(record) == 1
    with pytest.raises(RuntimeError, match="not currently applied"):
        proposal_to_nucleus_mapping(command)

    history.redo()
    assert history.num_undoable == 1
    assert proposal_to_nucleus_mapping(command)["t3"] == (3, 1)


def test_gap_edge_is_materialized_with_interpolated_adjacent_nuclei():
    seed = _seed(x=0, y=0, z=1.0)
    record = [[seed]]
    result = _result(
        (
            _detection("seed", 1, 0.0, y_um=0.0, z_um=0.0, radius_um=1.0),
            _detection("target", 4, 3.0, y_um=6.0, z_um=6.0, radius_um=2.5),
        ),
        (TrackEdge("seed", "target", 4.0, kind="gap"),),
        {"seed": (1, 1)},
    )
    command = ApplyTrackingProposal(result, Calibration(1.0, 2.0, plane_start=1))

    command.execute(record)

    assert [len(frame) for frame in record] == [1, 1, 1, 1]
    # Physical x/y and radius are linearly interpolated; z is converted back
    # to a 1-based AceTree plane.
    assert (record[1][0].x, record[1][0].y, record[1][0].z) == (1, 2, 2.0)
    assert (record[2][0].x, record[2][0].y, record[2][0].z) == (2, 4, 3.0)
    assert record[1][0].size == 3
    assert record[2][0].size == 4
    for time in range(1, 4):
        assert record[time - 1][0].successor1 == 1
        assert record[time][0].predecessor == 1
    assert proposal_to_nucleus_mapping(command) == {
        "seed": (1, 1),
        "target": (4, 1),
    }

    command.undo(record)
    assert record == [[seed]]
    assert seed.successor1 == NILLI


@pytest.mark.parametrize(
    ("detections", "edges", "message"),
    [
        (
            (
                _detection("a", 1, 0.0),
                _detection("b", 1, 1.0),
                _detection("child", 2, 0.5),
            ),
            (
                TrackEdge("a", "child", 1.0),
                TrackEdge("b", "child", 1.0),
            ),
            "two parents",
        ),
        (
            (
                _detection("parent", 1, 0.0),
                _detection("a", 2, 1.0),
                _detection("b", 2, 2.0),
                _detection("c", 2, 3.0),
            ),
            (
                TrackEdge("parent", "a", 1.0, kind="split"),
                TrackEdge("parent", "b", 1.0, kind="split"),
                TrackEdge("parent", "c", 1.0, kind="split"),
            ),
            "more than two children",
        ),
        (
            (_detection("later", 2, 1.0), _detection("earlier", 1, 0.0)),
            (TrackEdge("later", "earlier", 1.0),),
            "strictly forward",
        ),
        (
            (_detection("start", 1, 0.0), _detection("end", 3, 2.0)),
            (TrackEdge("start", "end", 1.0),),
            "must be marked as a gap",
        ),
        (
            (_detection("parent", 1, 0.0), _detection("child", 2, 1.0)),
            (TrackEdge("parent", "child", 1.0, kind="split"),),
            "exactly two split edges",
        ),
        (
            (
                _detection("parent", 1, 0.0),
                _detection("a", 2, 1.0),
                _detection("b", 2, 2.0),
            ),
            (
                TrackEdge("parent", "a", 1.0),
                TrackEdge("parent", "b", 1.0),
            ),
            "without an explicit split event",
        ),
    ],
)
def test_invalid_graphs_are_rejected_without_mutation(detections, edges, message):
    seed = _seed()
    record = [[seed]]
    before = deepcopy(record)
    result = _result(detections, edges)

    with pytest.raises(TrackingProposalConflict, match=message):
        ApplyTrackingProposal(result, CALIBRATION).execute(record)

    assert record == before
    assert record[0][0] is seed


def test_duplicate_existing_anchor_location_is_rejected():
    seed = _seed()
    record = [[seed]]
    detections = (
        _detection("first", 1, 5.0),
        _detection("second", 1, 5.0),
    )
    result = _result(
        detections,
        (),
        {"first": (1, 1), "second": (1, 1)},
    )

    with pytest.raises(TrackingProposalConflict, match="More than one detection"):
        ApplyTrackingProposal(result, CALIBRATION).execute(record)

    assert record == [[seed]]


def test_existing_records_are_not_replaced_or_renamed_when_new_detection_is_added():
    seed = _seed()
    unrelated = Nucleus(
        index=1,
        x=40,
        y=50,
        z=2.0,
        size=7,
        status=1,
        identity="AB",
        assigned_id="trusted-AB",
        predecessor=NILLI,
        weight=91,
    )
    second_frame = [unrelated]
    record = [[seed], second_frame]
    unrelated_before = unrelated.copy()
    result = _result(
        (_detection("seed", 1, 5.0), _detection("new", 2, 6.0)),
        (TrackEdge("seed", "new", 1.0),),
        {"seed": (1, 1)},
    )

    ApplyTrackingProposal(result, CALIBRATION).execute(record)

    assert record[1] is second_frame
    assert record[1][0] is unrelated
    assert unrelated == unrelated_before
    assert record[1][1].index == 2
    assert seed.successor1 == 2


def test_existing_seed_with_two_children_rejects_new_child_without_mutation():
    seed = _seed(successor1=1, successor2=2)
    record = [
        [seed],
        [
            Nucleus(index=1, status=1, predecessor=1),
            Nucleus(index=2, status=1, predecessor=1),
        ],
    ]
    before = deepcopy(record)
    result = _result(
        (_detection("seed", 1, 5.0), _detection("new", 2, 6.0)),
        (TrackEdge("seed", "new", 1.0),),
        {"seed": (1, 1)},
    )

    with pytest.raises(TrackingProposalConflict, match="already has two successors"):
        ApplyTrackingProposal(result, CALIBRATION).execute(record)

    assert record == before
    assert record[0][0] is seed


def test_proposal_cannot_reparent_an_existing_anchored_nucleus():
    old_parent = _seed()
    existing_child = Nucleus(
        index=1,
        x=11,
        y=20,
        z=1.0,
        size=8,
        identity="EMS",
        assigned_id="EMS",
        status=1,
        predecessor=1,
    )
    old_parent.successor1 = 1
    record = [[old_parent], [existing_child]]
    result = _result(
        (
            _detection("other", 1, 7.0),
            _detection("existing", 2, 5.5),
        ),
        (TrackEdge("other", "existing", 1.0),),
        {"existing": (2, 1)},
    )

    with pytest.raises(TrackingProposalConflict, match="curated predecessor"):
        ApplyTrackingProposal(result, CALIBRATION).execute(record)

    assert existing_child.predecessor == 1
    assert old_parent.successor1 == 1
