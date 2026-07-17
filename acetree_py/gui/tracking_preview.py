"""Pure presentation helpers for non-destructive tracking proposal previews.

Tracking results contain detector endpoints.  Applying a gap edge also creates
linearly interpolated nuclei in the intervening frames, so the review overlay
must expand those points before the user accepts the draft.  Keeping this logic
free of Qt and napari makes the visible proposal easy to test and prevents the
preview from mutating the lineage document.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tracking.api import Detection, TrackingResult


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
    kind: str  # ``seed``, ``detection``, ``interpolated``, or ``candidate``


@dataclass(frozen=True, slots=True)
class PreviewLink:
    """One adjacent-frame segment displayed in the proposal overlay."""

    source_id: str
    target_id: str
    kind: str  # ``link``, ``gap``, or ``split``
    cost: float


@dataclass(frozen=True, slots=True)
class PreviewSearchRegion:
    """Stopped-frame physical search region displayed for diagnosis."""

    frame: int
    x_um: float
    y_um: float
    z_um: float
    radius_um: float
    outcome_code: str


@dataclass(frozen=True, slots=True)
class ExpandedTrackingPreview:
    """Materialized, display-only view of a :class:`TrackingResult`."""

    spots: tuple[PreviewSpot, ...]
    links: tuple[PreviewLink, ...]
    candidates: tuple[PreviewSpot, ...] = ()
    outcome_code: str | None = None
    search_region: PreviewSearchRegion | None = None
    split_event_count: int = 0

    @property
    def by_id(self) -> dict[str, PreviewSpot]:
        return {spot.preview_id: spot for spot in self.review_spots}

    @property
    def review_spots(self) -> tuple[PreviewSpot, ...]:
        """Proposal spots followed by non-committable diagnostic candidates."""
        return (*self.spots, *self.candidates)

    @property
    def proposed_count(self) -> int:
        return sum(spot.kind in {"detection", "interpolated"} for spot in self.spots)

    @property
    def interpolated_count(self) -> int:
        return sum(spot.kind == "interpolated" for spot in self.spots)

    @property
    def candidate_count(self) -> int:
        """Number of explanatory observations that will not be committed."""
        return len(self.candidates)

    @property
    def split_count(self) -> int:
        """Number of proposed division events represented by split links."""
        return self.split_event_count


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
                kind=(
                    "seed"
                    if detection.detection_id in result.existing_anchors
                    else "detection"
                ),
            )
        )

    outcome = result.outcome
    candidates: list[PreviewSpot] = []
    search_region = None
    if outcome is not None:
        for candidate in outcome.review_candidates:
            candidates.append(
                PreviewSpot(
                    preview_id=f"__review_candidate__:{candidate.detection_id}",
                    detection_id=candidate.detection_id,
                    frame=candidate.frame,
                    x_um=candidate.x_um,
                    y_um=candidate.y_um,
                    z_um=candidate.z_um,
                    radius_um=candidate.radius_um,
                    quality=candidate.quality,
                    kind="candidate",
                )
            )
        if (
            outcome.stop_frame is not None
            and outcome.predicted_position_um is not None
        ):
            x_um, y_um, z_um = outcome.predicted_position_um
            search_region = PreviewSearchRegion(
                frame=outcome.stop_frame,
                x_um=x_um,
                y_um=y_um,
                z_um=z_um,
                radius_um=outcome.search_radius_um,
                outcome_code=outcome.code,
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
                kind=edge.kind,
                cost=edge.cost,
            )
        )

    spots.sort(key=lambda spot: (spot.frame, spot.preview_id))
    candidates.sort(key=lambda spot: (spot.frame, spot.preview_id))
    frame_by_id = {spot.preview_id: spot.frame for spot in spots}
    links.sort(
        key=lambda link: (
            frame_by_id[link.target_id],
            link.target_id,
        )
    )
    return ExpandedTrackingPreview(
        spots=tuple(spots),
        links=tuple(links),
        candidates=tuple(candidates),
        outcome_code=None if outcome is None else outcome.code,
        search_region=search_region,
        split_event_count=len(
            {edge.source_id for edge in result.edges if edge.kind == "split"}
        ),
    )


def expand_detector_preview(
    detections: tuple[Detection, ...],
) -> ExpandedTrackingPreview:
    """Build a transient, non-committable current-frame detector overlay.

    Detector tests deliberately do not masquerade as ``TrackingResult``
    objects.  Prefixing their presentation IDs keeps them distinct from a
    stale whole-dataset draft that may remain visible for comparison.
    """

    spots = tuple(
        PreviewSpot(
            preview_id=f"__detector_test__:{detection.detection_id}",
            detection_id=detection.detection_id,
            frame=detection.frame,
            x_um=detection.x_um,
            y_um=detection.y_um,
            z_um=detection.z_um,
            radius_um=detection.radius_um,
            quality=detection.quality,
            kind="detector_test",
        )
        for detection in detections
    )
    return ExpandedTrackingPreview(spots=spots, links=())


def _lerp(first: float, second: float, fraction: float) -> float:
    return first + (second - first) * fraction
