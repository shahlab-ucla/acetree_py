# StarryNite MATLAB oracle adapter

This directory contains a test-only MATLAB adapter for differential testing of
the Python rewrite against a user-supplied checkout of
[`zhirongbaolab/StarryNite`](https://github.com/zhirongbaolab/StarryNite).
It does not vendor or translate StarryNite implementation code.

## Requirements

- MATLAB with Image Processing Toolbox for detection. The original tracking
  model additionally requires Statistics and Machine Learning Toolbox.
- An external StarryNite checkout containing `distribution_code` and, for
  tracking, `distribution_lineaging` plus `2019TrackingModelv2.mat`.
- Request/result files in MATLAB v7 format, which `scipy.io.loadmat` and
  `scipy.io.savemat` can read and write.
- At least four Z planes for genuine 3-D calls to StarryNite's
  `imgaussianAnisotropy`; the upstream function treats smaller third dimensions
  as color/2-D data.

## Array and coordinate conventions

- Request volumes are `[Y, X, Z]`, matching MATLAB image indexing. Convert a
  Python `[Z, Y, X]` array with `numpy.transpose(volume, (1, 2, 0))`.
- StarryNite's coordinate tables are one-based `[X, Y, Z]`.
- The result retains those original tables and adds zero-based `[Z, Y, X]`
  tables for Python-side matching.

## Request schema

Every request file contains a scalar struct called `request` with
`schema_version = 1` and one of these operations:

- `gaussian_filter`: `volume_yxz`, `sigma_yxz`, `kernel_size_yxz`
- `separable_dog`: `volume_yxz`, `inner_sigma_yxz`,
  `inner_kernel_size_yxz`, `outer_sigma_yxz`, `outer_kernel_size_yxz`
- `tiled_dog`: `volume_yxz`, `sigma_xy`, `anisotropy`
- `resolve_parameter`: `parameter_name`, `staging`, `parameter_values`,
  `num_cells`, and `location_xyz`; optionally the complete trio
  `regional_stage_index`, `regional_area`, and `regional_value`
- `slice_candidates`: `filtered_volume_yxz`, `maxima_threshold`,
  `cell_diameter_xy`, `anisotropy`, `num_cells`, and `legacy_parameters`;
  optional `z_level`, `roi_points_xy`, `roi_is_local`, `roi_x_min`, and
  `roi_y_min`
- `full_detection`: `volume_yxz`, `cell_diameter_xy`, `anisotropy`,
  `num_cells`, and the complete detector `legacy_parameters`; optional
  `distribution_file`
- `full_tracking`: `movie_yxzt`, the full-detection fields, and optionally
  `model_file` plus an allowlisted scalar `tracking_overrides` struct

Historical classifier conversion uses the separate
`export_starrynite_classifier_numeric.m` helper. It intentionally does not use
the main request schema or `-batch`, because releases that reconstruct the old
`NaiveBayes` class can predate those APIs. Its v7 request has operation
`export_classifier_numeric_compatible`, `model_file`, and `expected_kind`
(`auto`, `single`, or the serialized spelling `ambigious`).

The caller supplies the separable DoG kernel parameters. Parameter derivation
stays in the test harness, while both Gaussian calculations are performed by
the upstream StarryNite function.

`resolve_parameter` constructs only the requested inert numeric fields and
calls the checkout's own `getParameter.m`. It is the golden gate for strict
stage boundaries and lower-exclusive/upper-inclusive regional boxes; it never
runs `readParameters.m` or a user-supplied statement.

`legacy_parameters` is the scalar parameter struct normally created in the
MATLAB workspace by a StarryNite parameter file. Candidate extraction requires
at least `staging`, `boundary_percent`, `large_ray_threshold`, and
`small_ray_threshold`. The adapter assigns it to StarryNite's expected global
only for the duration of the call. It intentionally does not execute a legacy
parameter file because upstream `readParameters.m` evaluates each line as
MATLAB code; the Python harness should parse the file safely and serialize the
resulting values into the request.

`full_detection` and `full_tracking` also require `sigma`,
`intensitythreshold`, `rangethreshold`, `nndist_merge`, `mergelower`,
`armerge`, `mergesplit`, and `split`. The Python bridge fills conservative
early-stage defaults, then applies safely parsed legacy values and explicit
sweep overrides. A high-threshold trial receives a deterministic `zlevel`
fallback because the upstream script otherwise leaves that variable undefined
when no plane clears its preliminary scan; the current candidate path does not
consume this value.

Tracking results are normalized to numeric node and edge tables. Frames and
node indices are zero-based, coordinates are zero-based XYZ pixels, and edge
kind codes distinguish continuation, gap, and division. MATLAB object IDs are
never compared with Python IDs.

Full-tracking results also include inert legacy node measurements, MATLAB's
chosen same-frame nearest-neighbor row plus confidence-geometry intermediates,
and raw `pred`/ordered `suc`/`delete` checkpoints before each classifier call.
These tables let Python distinguish a feature-arithmetic drift from a classifier
or mutation-order drift and reconstruct the exact representable state at the
post-greedy decision boundary. A transient asymmetric pointer snapshot is
reported as unsupported rather than coerced into a `TrackEdge`.

## Invocation

Add this directory to the MATLAB path, then invoke:

```matlab
run_starrynite_matlab_oracle(request_path, result_path, starrynite_root)
```

Parameter matrices should use the batch entry point so MATLAB and its parallel
pool start once:

```matlab
run_starrynite_matlab_oracle_batch(manifest_path, starrynite_root)
```

The manifest contains equally sized `request_paths` and `result_paths` cell
arrays. Each trial writes an independent structured result; an error in one
trial does not prevent later trials from running.

For unattended runs, the Python harness can use MATLAB's `-batch` option. Paths
passed into a MATLAB expression must have single quotes escaped by doubling
them. The adapter writes a `result` struct even when an operation fails, then
rethrows the MATLAB exception so the process also returns a failure status.

The adapter does not run `readParameters.m`, vendor StarryNite code, or modify
the supplied model. Full operations explicitly stage the minimal legacy script
workspace and use static diameter/cell-count inputs so one sweep point cannot
silently alter the next. Adaptive-diameter behavior should be tested as a
separate stateful experiment.

The historical classifier helper is narrower still: it only loads
`trackingparameters.bifurcationclassifier`, converts reconstructable numeric
state, and evaluates validation rows through the same loaded object. It accepts
only complete single/four-model layouts, finite numeric classes and priors,
normal or mvmn distributions, and unbounded normal-kernel distributions with
uncensored input data. Any property layout it cannot prove is rejected. Python
then validates source identity and replays the returned MATLAB class/posterior
probes before writing a neutral model and provenance manifest.
