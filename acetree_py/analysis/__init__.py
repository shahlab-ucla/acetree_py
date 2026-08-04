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
"""
