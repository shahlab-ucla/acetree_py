# StarryNite MATLAB/Python parity report

Status: **completed**

| Measure | Result |
|---|---:|
| Detection trials | 98 |
| Maximum filter relative L2 error | 1.04501e-05 |
| Maximum filter absolute error | 9.15527e-05 |
| Minimum filter correlation | 1 |
| Mean engine detection F1 | 1 |
| Minimum engine detection F1 | 1 |
| Exact-count trial fraction | 1 |
| Matched detection coverage | 1 |
| Centroid p95 (um) | 0 |
| Diameter MAE (px) | 0 |
| Diameter median relative error | 0 |
| Final quality p95 relative error | 7.14864e-07 |
| Mean MATLAB-vs-truth F1 | 0.955782 |
| Mean Python-vs-truth F1 | 0.955782 |
| Mean count-curve slope agreement | 1 |

## Compatibility gates

| Gate | Status |
|---|---|
| separable_dog | pass |
| final_detections | pass |
| parameter_sensitivity | pass |
| candidate_geometry | pass |
| final_quality | pass |
| tracking_classifier | diagnostic |

The JSON report contains every matched point, per-engine truth score, parameter response curve, provenance hash, and known compatibility gap.
