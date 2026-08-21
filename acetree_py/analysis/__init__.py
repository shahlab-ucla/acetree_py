"""Analysis package — expression analysis and data export.

Modules:
    expression: Per-cell expression time series, subtree stats, sister comparisons
    export: CSV and Newick tree format export
    measure: Pixel-level fluorescence measurement (port of AceBatch2 measure)
    measure_csv: CSV writer for measure output
    expression_plot: Renderer-neutral multi-cell expression plot snapshots
    expression_smoothing: Gap-preserving Gaussian smoothing primitives
    expression_comparison: Cross-dataset alignment, summaries, and tidy export
    expression_dataset_repository: Detached XML datasets and session cache
    expression_measurements: Revision-bound storage for every measured channel
    measure_runner: Orchestrator — iterates channels/timepoints and writes CSVs
    roi_rasterization: Cropped 2D/3D masks for subcellular ROI geometry
    roi_measure: Finite-only ROI reducers and thick-line spatial profiles
    roi_measurements: Immutable ROI snapshots, caches, and measurement engine
"""

from .roi_measure import (
    ROI_MEASUREMENT_ALGORITHM_VERSION,
    RoiDistribution,
    RoiIntensityReducer,
    RoiIntensityResult,
    RoiMetricValue,
    RoiProfileSampler,
    RoiSpatialProfile,
    reduce_roi_intensity,
    sample_roi_profile,
)
from .roi_measurements import (
    ROI_ANALYSIS_ALGORITHM_VERSION,
    RoiCacheStats,
    RoiMeasurementCache,
    RoiMeasurementCancelled,
    RoiMeasurementEngine,
    RoiMeasurementRequest,
    RoiMeasurementSample,
    RoiMeasurementSnapshot,
    RoiMeasurementStore,
    RoiScalarSeriesChannel,
    geometry_fingerprint,
    roi_metric_key,
    roi_temporal_subject,
    roi_temporal_subjects,
)
from .roi_rasterization import (
    ROI_RASTERIZATION_VERSION,
    CroppedRoiMask,
    RasterizedRoi,
    RoiCalibration,
    RoiMaskRasterizer,
    RoiRasterizationError,
    estimate_surface_area,
    rasterize_contour_stack,
    rasterize_polygon,
    rasterize_roi,
    rasterize_thick_polyline,
)

__all__ = [
    "ROI_ANALYSIS_ALGORITHM_VERSION",
    "ROI_MEASUREMENT_ALGORITHM_VERSION",
    "ROI_RASTERIZATION_VERSION",
    "CroppedRoiMask",
    "RasterizedRoi",
    "RoiCacheStats",
    "RoiCalibration",
    "RoiDistribution",
    "RoiIntensityReducer",
    "RoiIntensityResult",
    "RoiMaskRasterizer",
    "RoiMeasurementCache",
    "RoiMeasurementCancelled",
    "RoiMeasurementEngine",
    "RoiMeasurementRequest",
    "RoiMeasurementSample",
    "RoiMeasurementSnapshot",
    "RoiMeasurementStore",
    "RoiMetricValue",
    "RoiProfileSampler",
    "RoiRasterizationError",
    "RoiScalarSeriesChannel",
    "RoiSpatialProfile",
    "estimate_surface_area",
    "geometry_fingerprint",
    "rasterize_contour_stack",
    "rasterize_polygon",
    "rasterize_roi",
    "rasterize_thick_polyline",
    "reduce_roi_intensity",
    "roi_metric_key",
    "roi_temporal_subject",
    "roi_temporal_subjects",
    "sample_roi_profile",
]
