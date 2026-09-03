# StarryNite detector regression analysis

## Outcome

The apparent MATLAB/Python performance gap was not a broad filtering failure.
It was a small set of post-filter semantic differences concentrated at the
daughter-resolution boundary. After correcting those stages, MATLAB and Python
produce identical counts, centers, XY diameters, and sensitivity curves on the
checked smoke and full matrices, and identical decisions on the multiseed
resolution/noise matrix.

| Matrix | Cases | Engine F1 | Exact-count fraction | Centroid p95 | Diameter MAE |
|---|---:|---:|---:|---:|---:|
| Original coarse report | 30 rows / 21 unique settings | `0.922222` | `0.766667` | localized | broad mismatch |
| Maxima correction only | 30 rows / 21 unique settings | `0.988889` | `0.966667` | localized | broad mismatch |
| Corrected smoke | 21 unique settings | `1.0` | `1.0` | `0 µm` | `0 px` |
| Corrected full | 98 | `1.0` | `1.0` | `0 µm` | `0 px` |
| Resolution/noise | 81 | `1.0` | `1.0` | `0 µm` | `0 px` |

## Causal localization

| Stage | Evidence | Regression | Correction |
|---|---|---|---|
| DoG filtering | Peak displacement `0`; full-matrix correlation above `0.9999999999`; absolute error below `9.16e-5` | None affecting decisions | Retained finite separable filter and added scale-aware numerical gates |
| 3-D maxima | Frame-4 peaks at X `21` and `25` are exactly `2.0 µm` apart | Python used the expected radius as an implicit suppression distance and erased the weaker daughter | Reproduced per-slice regional maxima plus MATLAB's adjacent-Z 18-neighbor test; physical suppression is now explicit-only |
| Candidate geometry | Matched diameter error reached `38.6%`; support centroids pulled close daughters together | Python used a 3-D threshold support for position and radius | Ported the 16-ray crossing/valley estimator, adjacent-ray repair, polygon recentering, coverage, and odd-pixel 80th-percentile diameter |
| Boundary response | At `boundary_percent=.2`, MATLAB returned one point while corrected maxima produced two | Boundary geometry changes disk claims before conflict resolution; support-mask overlap has the wrong monotonic trend | Added contiguous plane claims, overlap graph construction, and overlap-gated legacy distance/aspect-ratio conflict predicates |
| Production defaults | The parity runner disabled subpixel refinement but the UI enabled it | Normal use could still collapse close daughter coordinates | StarryNite registry, legacy profiles, dataset wizard, global tracking, and sparse forward tracking now default to ray-recentered positions; native refinement remains opt-in |

The fine boundary transition is frozen in unit tests. On the early-division
fixture, MATLAB and Python both merge through `.27` and retain two candidates at
`.28`; the merged X coordinate at `.18` is `22.1666667`, demonstrating why
identical-coordinate deduplication would not have been sufficient.

## Stress-matrix behavior

The resolution/noise suite renders eight daughter separations (`3.5` through
`8` XY pixels), three Gaussian-noise levels (`0`, `5`, `20`), and three seeds,
plus one noise-only control for every seed/noise pair.

- MATLAB/Python count, center, diameter, and quality parity is exact in all 81
  scenes under a half-pixel engine match gate.
- All nine noise-only controls yield zero detections in both engines.
- The one-to-two transition occurs between `5` and `5.5` pixels. At separation
  `5`, noise can move an individual seed across the transition, but both engines
  always move together.
- Below the transition, both engines intentionally miss one of two truth
  objects. Equal truth scores distinguish shared resolution limits from rewrite
  drift.

## Remaining compatibility boundary

The native detector now covers ray geometry, contiguous claims, and the two
geometric conflict predicates. StarryNite's distribution-backed z-disk log-odds
range, merge, and split predicates are not yet claimed equivalent. The unchanged
2019 MATLAB tracking classifier has its own live smoke gate; native
classifier-score and end-to-end lineage parity remain diagnostic tiers.

Detailed outputs:

- [corrected smoke](starrynite-parity-final-smoke/summary.md)
- [corrected full matrix](starrynite-parity-full/summary.md)
- [resolution/noise matrix](starrynite-parity-resolution-noise/summary.md)
