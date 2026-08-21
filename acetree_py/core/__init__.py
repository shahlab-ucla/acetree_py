"""Core, GUI-independent AceTree domain models."""

from .subcellular_roi import (
    AssociationResolution,
    AssociationStatus,
    CellRef,
    CellResolution,
    ContourSlice,
    ContourStack3D,
    CoordinateSpaceSnapshot,
    Geometry,
    GeometryKind,
    NucleusAnchor,
    ObjectClass,
    Polygon2D,
    Presence,
    QuarantinedRoiRecord,
    ReviewState,
    RoiFrameRecord,
    RoiObjectTrack,
    RoiValidationError,
    SamplingMode,
    SubcellularRoiDocument,
    ThickPolyline2D,
    Thickness,
    ThicknessUnit,
)

__all__ = [
    "AssociationResolution",
    "AssociationStatus",
    "CellRef",
    "CellResolution",
    "ContourSlice",
    "ContourStack3D",
    "CoordinateSpaceSnapshot",
    "Geometry",
    "GeometryKind",
    "NucleusAnchor",
    "ObjectClass",
    "Polygon2D",
    "Presence",
    "QuarantinedRoiRecord",
    "ReviewState",
    "RoiFrameRecord",
    "RoiManager",
    "RoiManagerError",
    "RoiManagerSaveStage",
    "RoiNotFoundError",
    "RoiObjectTrack",
    "RoiValidationError",
    "RoiWriteProtectedError",
    "SamplingMode",
    "SubcellularRoiDocument",
    "ThickPolyline2D",
    "Thickness",
    "ThicknessUnit",
    "coordinate_space_from_config",
]


_MANAGER_EXPORTS = {
    "RoiManager",
    "RoiManagerError",
    "RoiManagerSaveStage",
    "RoiNotFoundError",
    "RoiWriteProtectedError",
    "coordinate_space_from_config",
}


def __getattr__(name: str):
    """Load manager exports lazily to keep ``core``/``io`` imports acyclic."""
    if name in _MANAGER_EXPORTS:
        from . import roi_manager

        return getattr(roi_manager, name)
    raise AttributeError(name)
