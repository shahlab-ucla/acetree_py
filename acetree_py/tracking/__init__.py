"""Modular detection and temporal-linking primitives for AceTree."""

from .api import (
    TRACKING_API_MAJOR,
    TRACKING_API_VERSION,
    Calibration,
    ComponentSpec,
    Detection,
    TrackEdge,
    TrackerGraphResult,
    TrackingOutcome,
    TrackingRequest,
    TrackingResult,
    TrackingScope,
    WholeMoviePreflightContext,
)
from .detectors import DoGDetector, DogDetector, LoGDetector, LogDetector
from .lap import LAPTracker, SimpleLAPTracker
from .starrynite import (
    StarryNiteDetector,
    StarryNiteDivisionTracker,
    StarryNiteLegacyExactTracker,
    StarryNiteTracker,
    StarryNiteTuningProfile,
    load_tuning_profile,
)
from .integration import ApplyTrackingProposal, TrackingProposalConflict
from .persistence import (
    TrackingProposalFormatError,
    read_tracking_proposal,
    tracking_sidecar_path,
    write_tracking_proposal,
)
from .pipeline import TrackingCancelled, TrackingPipeline
from .registry import (
    ComponentDescriptor,
    PluginContribution,
    TrackingRegistry,
    build_default_registry,
    get_default_registry,
)

__all__ = [
    "TRACKING_API_MAJOR",
    "TRACKING_API_VERSION",
    "Calibration",
    "ApplyTrackingProposal",
    "ComponentDescriptor",
    "ComponentSpec",
    "Detection",
    "DoGDetector",
    "DogDetector",
    "LAPTracker",
    "LoGDetector",
    "LogDetector",
    "PluginContribution",
    "SimpleLAPTracker",
    "StarryNiteDetector",
    "StarryNiteDivisionTracker",
    "StarryNiteLegacyExactTracker",
    "StarryNiteTracker",
    "StarryNiteTuningProfile",
    "TrackEdge",
    "TrackerGraphResult",
    "TrackingOutcome",
    "TrackingRegistry",
    "TrackingCancelled",
    "TrackingPipeline",
    "TrackingProposalConflict",
    "TrackingProposalFormatError",
    "TrackingRequest",
    "TrackingResult",
    "TrackingScope",
    "WholeMoviePreflightContext",
    "build_default_registry",
    "get_default_registry",
    "load_tuning_profile",
    "read_tracking_proposal",
    "tracking_sidecar_path",
    "write_tracking_proposal",
]
