"""Pure presentation helpers for non-destructive tracking proposal previews.

Tracking results contain detector endpoints.  Applying a gap edge also creates
linearly interpolated nuclei in the intervening frames, so the review overlay
must expand those points before the user accepts the draft.  Keeping this logic
free of Qt and napari makes the visible proposal easy to test and prevents the
preview from mutating the lineage document.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tracking.api import TrackingResult


@dataclass(frozen=True, slots=True)
class PreviewSpot:
    """One position displayed while reviewing a tracking proposal."""

    preview_id: str
    detection_id: str | None
    frame: int
    x_um: float
    y_um: float
    z_um: float
    radius_um: float
    quality: float
    kind: str  # ``seed``, ``detection``, or ``interpolated``


@dataclass(frozen=True, slots=True)
class PreviewLink:
    """One adjacent-frame segment displayed in the proposal overlay."""

    source_id: str
    target_id: str
    kind: str  # ``link`` or ``gap``
    cost: float


@dataclass(frozen=True, slots=True)
class ExpandedTrackingPreview:
    """Materialized, display-only view of a :class:`TrackingResult`."""

    spots: tuple[PreviewSpot, ...]
    links: tuple[PreviewLink, ...]

    @property
    def by_id(self) -> dict[str, PreviewSpot]:
        return {spot.preview_id: spot for spot in self.spots}

    @property
    def proposed_count(self) -> int:
        return sum(spot.kind != "seed" for spot in self.spots)

    @property
    def interpolated_count(self) -> int:
        return sum(spot.kind == "interpolated" for spot in self.spots)


def expand_tracking_preview(result: TrackingResult) -> ExpandedTrackingPreview:
    """Expand a proposal into exactly the positions the commit will create.

    Existing anchors are retained as ``seed`` spots so outgoing links have a
    visible source, but callers can omit their circles because AceTree already
    draws curated nuclei.  Gap interpolation intentionally mirrors
    ``tracking.integration``: every intermediate frame uses a linear blend of
    position, radius, and quality.
    """

    spots: list[PreviewSpot] = []
    by_detection_id = {detection.detection_id: detection for detection in result.detections}
    for detection in result.detections:
        spots.append(
            PreviewSpot(
                preview_id=detection.detection_id,
                detection_id=detection.detection_id,
                frame=detection.frame,
                x_um=detection.x_um,
                y_um=detection.y_um,
                z_um=detection.z_um,
                radius_um=detection.radius_um,
                quality=detection.quality,
                kind=("seed" if detection.detection_id in result.existing_anchors else "detection"),
            )
        )

    links: list[PreviewLink] = []
    for edge_number, edge in enumerate(result.edges):
        source = by_detection_id[edge.source_id]
        target = by_detection_id[edge.target_id]
        delta = target.frame - source.frame
        previous_id = source.detection_id
        for step in range(1, delta):
            fraction = step / delta
            preview_id = f"__preview_gap__:{edge_number}:{step}"
            spots.append(
                PreviewSpot(
                    preview_id=preview_id,
                    detection_id=None,
                    frame=source.frame + step,
                    x_um=_lerp(source.x_um, target.x_um, fraction),
                    y_um=_lerp(source.y_um, target.y_um, fraction),
                    z_um=_lerp(source.z_um, target.z_um, fraction),
                    radius_um=_lerp(source.radius_um, target.radius_um, fraction),
                    quality=_lerp(source.quality, target.quality, fraction),
                    kind="interpolated",
                )
            )
            links.append(
                PreviewLink(
                    source_id=previous_id,
                    target_id=preview_id,
                    kind="gap",
                    cost=edge.cost,
                )
            )
            previous_id = preview_id
        links.append(
            PreviewLink(
                source_id=previous_id,
                target_id=target.detection_id,
                kind="gap" if edge.kind == "gap" else "link",
                cost=edge.cost,
            )
        )

    spots.sort(key=lambda spot: (spot.frame, spot.preview_id))
    frame_by_id = {spot.preview_id: spot.frame for spot in spots}
    links.sort(
        key=lambda link: (
            frame_by_id[link.target_id],
            link.target_id,
        )
    )
    return ExpandedTrackingPreview(tuple(spots), tuple(links))


def _lerp(first: float, second: float, fraction: float) -> float:
    return first + (second - first) * fraction
