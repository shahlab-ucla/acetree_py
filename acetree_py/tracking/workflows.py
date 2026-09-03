"""User-facing tracking workflows built from detector/tracker components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackingWorkflow:
    """A stable, understandable detector/tracker pairing for the UI."""

    workflow_id: str
    display_name: str
    description: str
    detector_id: str
    tracker_id: str
    supports_global: bool = True
    supports_forward: bool = True
    uses_bundled_starrynite: bool = False


MODERN_STARRYNITE = TrackingWorkflow(
    "modern_starrynite",
    "Modern StarryNite (recommended)",
    "Stage-aware StarryNite detection with fast division-aware native tracking.",
    "acetree.starrynite_detector",
    "acetree.starrynite_division",
    uses_bundled_starrynite=True,
)
LOG_LAP = TrackingWorkflow(
    "log_lap",
    "LoG detection + LAP tracking",
    "General-purpose 3D Laplacian-of-Gaussian detection with one-to-one LAP linking.",
    "acetree.log3d",
    "acetree.simple_lap",
)
DOG_LAP = TrackingWorkflow(
    "dog_lap",
    "DoG detection + LAP tracking",
    "Fast 3D Difference-of-Gaussians detection with one-to-one LAP linking.",
    "acetree.dog3d",
    "acetree.simple_lap",
)
LEGACY_STARRYNITE_EXACT = TrackingWorkflow(
    "legacy_starrynite_exact",
    "Legacy StarryNite exact replay (advanced)",
    "Whole-movie source-bound replay for compatible legacy parameter/model bundles; "
    "requires matching calibration and complete movie scope.",
    "acetree.starrynite_detector",
    "acetree.starrynite_legacy_exact",
    supports_forward=False,
    uses_bundled_starrynite=True,
)
CUSTOM_COMPONENTS = TrackingWorkflow(
    "custom",
    "Custom detector and tracker",
    "Choose individual installed components in Advanced settings.",
    "",
    "",
)

GLOBAL_TRACKING_WORKFLOWS = (
    MODERN_STARRYNITE,
    LOG_LAP,
    DOG_LAP,
    LEGACY_STARRYNITE_EXACT,
    CUSTOM_COMPONENTS,
)
FORWARD_TRACKING_WORKFLOWS = (
    MODERN_STARRYNITE,
    LOG_LAP,
    DOG_LAP,
    CUSTOM_COMPONENTS,
)
INITIAL_TRACKING_WORKFLOWS = (
    MODERN_STARRYNITE,
    LOG_LAP,
    DOG_LAP,
)


def tracking_workflow(workflow_id: str, *, forward: bool = False) -> TrackingWorkflow:
    """Resolve a workflow by its stable ID."""

    choices = FORWARD_TRACKING_WORKFLOWS if forward else GLOBAL_TRACKING_WORKFLOWS
    requested = str(workflow_id)
    for workflow in choices:
        if workflow.workflow_id == requested:
            return workflow
    raise KeyError(f"Unknown tracking workflow {requested!r}")


def workflow_for_components(
    detector_id: str | None,
    tracker_id: str | None,
    *,
    forward: bool = False,
) -> TrackingWorkflow:
    """Map an existing request back to a visible workflow without changing it."""

    choices = FORWARD_TRACKING_WORKFLOWS if forward else GLOBAL_TRACKING_WORKFLOWS
    for workflow in choices:
        if workflow is CUSTOM_COMPONENTS:
            continue
        if workflow.detector_id == detector_id and workflow.tracker_id == tracker_id:
            return workflow
    return CUSTOM_COMPONENTS


__all__ = [
    "CUSTOM_COMPONENTS",
    "DOG_LAP",
    "FORWARD_TRACKING_WORKFLOWS",
    "GLOBAL_TRACKING_WORKFLOWS",
    "INITIAL_TRACKING_WORKFLOWS",
    "LEGACY_STARRYNITE_EXACT",
    "LOG_LAP",
    "MODERN_STARRYNITE",
    "TrackingWorkflow",
    "tracking_workflow",
    "workflow_for_components",
]
