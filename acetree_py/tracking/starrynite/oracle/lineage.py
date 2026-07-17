"""Normalized snapshots and rich MATLAB/Python lineage parity comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from ...api import Calibration, Detection, TrackEdge
from .metrics import LineageGraphSimilarity, LineageNodeState, compare_lineage_graphs

if TYPE_CHECKING:
    from .matlab_backend import MatlabOracleRun


@dataclass(frozen=True, slots=True)
class LineageSnapshot:
    """Portable, ID-stable lineage graph suitable for frozen oracle replay."""

    nodes: tuple[LineageNodeState, ...]
    edges: tuple[tuple[str, str, str], ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        if any(not isinstance(item.node_id, str) or not item.node_id for item in nodes):
            raise ValueError("Lineage snapshot node IDs must be non-empty strings")
        identifiers = {item.node_id for item in nodes}
        if len(identifiers) != len(nodes):
            raise ValueError("Lineage snapshot node IDs must be unique")
        edges = tuple((str(source), str(target), str(kind)) for source, target, kind in self.edges)
        if any(source not in identifiers or target not in identifiers for source, target, _ in edges):
            raise ValueError("Lineage snapshot edges must reference known nodes")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acetree.starrynite.lineage-snapshot/v1",
            "nodes": [
                {
                    "id": item.node_id,
                    "frame": item.frame,
                    "position": list(item.position),
                    "retained": item.retained,
                }
                for item in self.nodes
            ],
            "edges": [
                {"source": source, "target": target, "kind": kind}
                for source, target, kind in self.edges
            ],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LineageSnapshot:
        if value.get("schema") != "acetree.starrynite.lineage-snapshot/v1":
            raise ValueError("Unsupported StarryNite lineage snapshot schema")
        return cls(
            nodes=tuple(
                LineageNodeState(
                    str(item["id"]),
                    int(item["frame"]),
                    tuple(float(coordinate) for coordinate in item["position"]),
                    bool(item.get("retained", True)),
                )
                for item in value.get("nodes", ())
            ),
            edges=tuple(
                (str(item["source"]), str(item["target"]), str(item["kind"]))
                for item in value.get("edges", ())
            ),
            provenance=value.get("provenance", {}),
        )


def matlab_lineage_snapshot(
    run: MatlabOracleRun,
    calibration: Calibration,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> LineageSnapshot:
    """Convert one normalized MATLAB tracking run into physical coordinates."""

    table = run.node_table()
    nodes: list[LineageNodeState] = []
    identifiers: dict[tuple[int, int], str] = {}
    for row in table:
        frame = int(row[0])
        node = int(row[1])
        identifier = f"matlab:{frame}:{node}"
        identifiers[(frame, node)] = identifier
        x_um, y_um, z_um = calibration.pixel_to_physical(
            float(row[2]),
            float(row[3]),
            float(row[4]) + calibration.plane_start,
        )
        nodes.append(
            LineageNodeState(
                identifier,
                frame,
                (x_um, y_um, z_um),
                retained=not bool(row[7]),
            )
        )
    edges = tuple(
        (
            identifiers[source],
            identifiers[target],
            kind,
        )
        for source, target, kind in run.normalized_edges(include_deleted_sources=True)
    )
    metadata = {
        "oracle_operation": run.operation,
        "matlab_version": run.matlab_version,
        **dict(provenance or {}),
    }
    return LineageSnapshot(tuple(nodes), edges, metadata)


def python_lineage_snapshot(
    raw_detections: Sequence[Detection],
    retained_detection_ids: Sequence[str] | set[str],
    edges: Sequence[TrackEdge],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> LineageSnapshot:
    """Normalize Python raw detections plus a refined retained graph."""

    raw = tuple(raw_detections)
    identifiers = {item.detection_id for item in raw}
    if len(identifiers) != len(raw):
        raise ValueError("Raw Python detection IDs must be unique")
    retained = {str(item) for item in retained_detection_ids}
    if not retained <= identifiers:
        raise ValueError("Retained IDs must reference raw Python detections")
    nodes = tuple(
        LineageNodeState(
            item.detection_id,
            item.frame - 1,
            item.position_um,
            retained=item.detection_id in retained,
        )
        for item in raw
    )
    normalized_edges = tuple(
        (item.source_id, item.target_id, item.kind) for item in edges
    )
    return LineageSnapshot(nodes, normalized_edges, dict(provenance or {}))


def compare_lineage_snapshots(
    reference: LineageSnapshot,
    candidate: LineageSnapshot,
    *,
    tolerance_um: float,
) -> LineageGraphSimilarity:
    """Spatially align and compare two normalized physical lineage snapshots."""

    return compare_lineage_graphs(
        reference.nodes,
        reference.edges,
        candidate.nodes,
        candidate.edges,
        tolerance=tolerance_um,
    )


def write_lineage_snapshot(path: str | Path, snapshot: LineageSnapshot) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_lineage_snapshot(path: str | Path) -> LineageSnapshot:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Lineage snapshot root must be an object")
    return LineageSnapshot.from_dict(value)


__all__ = [
    "LineageSnapshot",
    "compare_lineage_snapshots",
    "matlab_lineage_snapshot",
    "python_lineage_snapshot",
    "read_lineage_snapshot",
    "write_lineage_snapshot",
]
