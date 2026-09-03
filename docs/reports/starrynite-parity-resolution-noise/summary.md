# StarryNite MATLAB/Python parity report

Status: **completed**

| Measure | Result |
|---|---:|
| Detection trials | 81 |
| Maximum filter relative L2 error | 1 |
| Maximum filter absolute error | 6.10352e-05 |
| Minimum filter correlation | 1 |
| Mean engine detection F1 | 1 |
| Minimum engine detection F1 | 1 |
| Exact-count trial fraction | 1 |
| Matched detection coverage | 1 |
| Centroid p95 (um) | 0 |
| Diameter MAE (px) | 0 |
| Diameter median relative error | 0 |
| Final quality p95 relative error | 6.60261e-07 |
| Mean MATLAB-vs-truth F1 | 0.860082 |
| Mean Python-vs-truth F1 | 0.860082 |
| Mean count-curve slope agreement | n/a |

## Compatibility gates

| Gate | Status |
|---|---|
| separable_dog | pass |
| final_detections | pass |
| parameter_sensitivity | diagnostic |
| candidate_geometry | pass |
| final_quality | pass |
| tracking_classifier | diagnostic |

Relative L2 is ill-conditioned for near-zero DoG controls. The filter gate therefore also requires absolute error at most 1e-4, correlation at least 0.999999999 when defined, and zero peak displacement on non-negligible responses.

The JSON report contains every matched point, per-engine truth score, parameter response curve, provenance hash, and known compatibility gap.
