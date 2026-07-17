# StarryNite MATLAB/Python differential testing

## Purpose

The MATLAB implementation is treated as a local behavioral oracle, not as a
source translation target. A deterministic simulator renders each image once;
the exact same floating-point array is passed to MATLAB and Python. Reports
compare intermediate numerical artifacts, final detections, lineage topology,
and response curves under legacy parameter changes.

This answers two different questions independently:

1. Does the Python rewrite react like MATLAB?
2. Is either engine correct relative to the known synthetic truth?

A high engine-to-engine score cannot hide a shared error because both engines
are also scored against truth.

## Comparison tiers

| Tier | MATLAB artifact | Python artifact | Current status |
|---|---|---|---|
| Effective parameters | staged `parameters` / `trackingparameters` | safe parsed tuning profile | covered |
| DoG kernel and volume | `imgaussianAnisotropy` / `processVolume` default path | `legacy_dog_response` | covered and gated |
| Slice/3-D candidates | `createDiskSet` diagnostic tables | per-slice maxima, 16-ray geometry, plane claims | covered and gated |
| Final detections | `processVolume.e.final*` | conflict-resolved AT `Detection` records | covered and gated on the synthetic corpus |
| Sequential detector state | prior `e.diams` / `e.finalpoints`, effective diameter, cell count, and stage | one detector instance with source-bound per-frame audit features | live-gated with 12 nuclei and an observable `10 -> 7 px` median-diameter update |
| Early geometry stages | initialization, easy links, polar-filter boundary, candidates, every nondivision/division threshold, final geometry | `run_legacy_early_tracking` snapshots with ordered forward/back candidates | exact at every recorded boundary on the live translation and class 0/1/2/3 movies |
| Tracking graph | `esequence.suc` / `suc_time` | AT `TrackEdge` records | normalized and compared live across five cases |
| 2019 classifier | trusted numeric export of unchanged `2019TrackingModelv2.mat` plus direct `predict` | typed neutral Naive Bayes runtime bound to the source MAT SHA-256 | covered in feature space and in a live post-greedy positive-division replay |
| Feature/class scores | 22/11/13 blocks, topology branch, masks, posterior, and final/forced class | exact legacy feature/candidate extraction, topology/mask assembly, log scores, posterior, cost decision, and forced class | covered independently and exercised through the live post-greedy movie driver; historical four-model live certification awaits its MAT corpus |
| Lineage cleanup | raw `pred`/ordered `suc`/`delete` checkpoints around every original classifier call | atomic post-greedy movie scan, immutable class 0/1/2/3 mutations, and raw ordered event trace | synthetic dynamic coverage for all classes; live raw-checkpoint-to-final parity for class 1, two recursive class-0 movies, a class-2 backward repair, and a class-3 branch deletion |
| Registered whole movie | sequential `processVolume` plus the original classifier tracking driver | `TrackingPipeline` + StarryNite detector + `acetree.starrynite_legacy_exact` | live positive-division movie matches detector rows, retained nodes, every edge/stage summary, classifier event/counts, full lineage state, and ancestry |

The raw class-0 extractor intentionally records delete-marked MATLAB rows,
because such a row can consume one of the four nearest attempts or receive the
detached daughter. Matching MATLAB, attachment changes its raw successor and
the daughter's predecessor slots without clearing the source delete flag. The
incident edge is therefore retained for parity in `LegacyTrackingContext` but
omitted from the executable active lineage. A deleted one-child source is
temporarily activated only for nested class resolution, then atomically restored
to deleted state while preserving the resolved raw slot order.

The Python movie driver starts with the raw state immediately before the
classifier/repair scan inside `greedydeleteFPbranches`. It follows MATLAB's
frame-then-original-row order, records nested class-0 calls inline as round 2,
and returns the original input atomically if a mutation is unrepresentable. A
checkpoint adapter reconstructs that input from the live oracle's initial raw
snapshot and fails closed on asymmetric predecessor/successor arrays. The live
positive-division gate now matches MATLAB's classifier checkpoint sequence and
every net pointer delta in each checkpoint interval through the final state.
Checkpoint snapshots cannot reveal the order of individual pointer writes
inside one interval, so the comparison does not claim that stronger causal
property. The isolated-fragment prepass and the earlier initialization,
easy-link, candidate, nondivision, and division-score passes are now executable
and stage-gated. Polar-body and hysteresis modes remain fail-closed because the
AT detection boundary does not yet carry their additional raw measurements.

The default and `conservememory=true` MATLAB DoG paths are separate legacy
modes. The ordinary `processVolume` path uses the separable Gaussian operation;
the tiled path constructs a materially different kernel and must not be mixed
into the same numerical gate.

## Coordinate contract

- Simulations and Python volumes: zero-based `T, Z, Y, X`.
- MATLAB volumes: `Y, X, Z, T`.
- Raw StarryNite points: one-based `X, Y, Z`.
- Normalized points: zero-based `X, Y, Z` pixels and physical XYZ microns.
- Legacy distances restore one-based coordinates and binary32 arithmetic before
  evaluating the original Gram expression. The squared norm uses native
  float32 dot/matrix multiplication, matching MATLAB's single-precision
  reduction rather than a Python-level sequential sum; the difference can be
  one ULP and can change a nearest-neighbor cutoff.
- When MATLAB concatenates a single-valued geometry vector with double-valued
  ratios, the complete feature row becomes single. Python mirrors that
  whole-row cast for nondivision, triple, daughter, backward, and forward
  classifier blocks before applying downstream decisions.
- MATLAB's platform single-precision `log` can choose the adjacent float for
  the two nearest-neighbor confidence columns even when the captured input is
  bit-identical. Oracle reconstruction permits exactly one binary32 ULP only
  for those two columns; all other confidence values and graph/event decisions
  retain their strict comparisons.
- MATLAB lineage nodes: `(frame - 1, local_index - 1)`.
- IDs are never compared across engines.
- A division is one atomic parent plus an unordered two-daughter set. Matching
  only one daughter does not count as a matched division.

Dedicated tests cover orientation, one-based offsets, radius versus diameter,
anisotropic distance, empty outputs, and deleted lineage nodes.

## Synthetic corpus

The always-available suite currently includes:

- one interior anisotropic Gaussian nucleus;
- two nuclei near the resolution/merge boundary;
- unequal bright/dim nuclei on a background gradient;
- an edge nucleus plus an interior control;
- a six-frame parent-to-two-daughter movie;
- an 81-scene matrix spanning three noise seeds, three noise levels, eight
  daughter separations, and nine noise-only false-positive controls;
- a five-frame, four-cell translating movie used with the original 2019 model;
- a five-case lineage matrix covering translation, a missing observation, a
  classifier-rejected daughter-branch event, a robust positive class-1
  division, and a transient detector artifact;
- two nine-frame live cleanup fixtures: an attached one-frame branch for class
  3, and a near crossing with one missed observation and a real backward
  candidate for class 2.

The simulator uses fixed NumPy random seeds. Noise is generated once and its
array hash is recorded; MATLAB never regenerates random values. Targeted
candidate-stage tests now cover a true flat maximum, a long equal-valued shelf
connected to a higher value, diagonal 8-connectivity, strict thresholding, and
MATLAB column-major `find` order; the three plateau masks were also checked
directly against R2025a `imregionalmax`. The next broad corpus expansion should
add more face/edge/corner plateaus, Poisson noise, clipping and integer dtypes,
ROI offsets, crossings, gaps, false positives, simultaneous divisions, and
exact threshold/cost boundary cases.

Overlooked-nucleus recovery is now part of the strict detector tail rather than
a geometry-only approximation. The runtime loads and hashes the referenced
numeric distribution MAT, reconstructs the per-disk `calculateLogodds` values,
applies `vcalculateMaximalRange`, iterates overlooked-center recovery, and then
runs the legacy merge/split conflict predicates. Polygon filtering,
source-defined rectangular ROI crop/coordinate restoration, and previous-frame
diameter/final-count adaptation are also executable and live-gated. If a
required distribution is absent, changed, unsupported, or not explicitly
selected, exact mode stops; it never substitutes another distribution.
Expanded pathological integer-image and crowded-embryo coverage remains corpus
expansion, not a missing runtime stage.

AceTree's legacy nuclei-file `weight` is kept distinct from the native support
sum. It follows `initializeTrackingStructures.m` and `saveGreedyNucleiFiles.m`:
the polygon/disk `integrateGFP` total is divided by 256, rounded with MATLAB's
positive `uint16` conversion behavior, and saturated to the 16-bit range.
`TOTAL_INTENSITY` remains available as a separate native measurement and is not
used for that legacy field.

## Metrics

### Numerical filter

- relative L2 error;
- dynamic-range-normalized RMSE;
- maximum and 99th-percentile absolute error;
- Pearson and cosine similarity;
- peak displacement.

### Detections

Points are matched per frame with maximum-cardinality/minimum-distance bipartite
assignment under a hard anisotropic physical-distance gate. Engine parity uses
a half-XY-pixel gate; truth scoring keeps the wider radius-aware gate. Reports
include:

- count difference, precision, recall, and F1;
- centroid RMSE, p95, and per-axis bias;
- raw paired diameter and quality values, signed bias, MAE/RMSE, p95, relative
  errors, and matched numerator/denominator coverage;
- independent MATLAB-to-truth and Python-to-truth scores.

### Lineages

- frame-local physical-coordinate node matching with retained/deleted-state
  accuracy and retained-node precision/recall/F1;
- edge precision/recall/F1 overall and independently for continuation, gap,
  and split edges;
- unordered atomic division-event F1;
- root, termination, and connected-component counts;
- ancestor-pair agreement, so an endpoint match cannot hide an earlier
  identity switch;
- portable MATLAB/Python graph snapshots containing raw nodes, cleanup state,
  normalized edges, and provenance.

The suite runner batches all movies into one MATLAB startup. This matters for
routine regression use because MATLAB and toolbox initialization dominate the
small deterministic movies.

### Parameter sensitivity

One-factor sweeps start from an effective staged legacy parameter set. Each
curve records:

- Pearson/Spearman correlation when defined;
- signed finite-difference agreement with a dead band;
- normalized area between curves;
- transition-point distance;
- each engine's truth score at every parameter value.

Direct detector sweeps currently cover `sigma`, `intensitythreshold`, and
`boundary_percent`. The MATLAB bridge also accepts allowlisted tracking
overrides for `candidateCutoff`, `safefactor`, `nnnumber`, `forwardnnnumber`,
`temporalcutoff`, score ranges, `smallcutoff`, and `wideWindow`. Tracking sweeps
must compare normalized graphs. Native-tracker sweeps remain diagnostic because
that backend uses its geometry scorer; the exact backend executes only the
source-bound inert classifier export, never the serialized MATLAB object.

The pinned upstream parameter corpus contains 11 published files. All 11 now
parse and build profiles, and all 22 lineaging scalar controls are present in
each file. Every detector field that is present resolves across 13 exact stage
boundary counts: five files cover 156 lookups, while the six iSIM/SD files cover
143 because those upstream files omit `parameters.selection_dist`. Safe
source-ordered aliases to earlier inert values are resolved; any later dynamic
reassignment invalidates the alias and fails closed. The remaining 24 opaque
statements are 22 behavior-bearing cost-function handles and two ROI polygon
matrices. Only the dispim-model example colocates its active MAT file with the
parameter file; the other legacy or distribution-relative references are
reported as unresolved rather than found by an implicit basename search.

Classifier sensitivity is tested at the feature boundary. MATLAB exports only
inert numeric state, including class names, priors, costs, feature masks, and
per-class categorical/normal/kernel distribution parameters. Python binds the
runtime model to the original MAT SHA-256 and rejects shape drift, unsupported
distributions, censored/truncated kernels, nonidentity standardization or score
transforms, or a hash mismatch. Raw infinite predictors are retained because
MATLAB passes them to `predict`; all-zero likelihoods use the same prior
fallback as unknown categories and underflowed kernel tails. For each applicable raw block
feature, the live
matrix perturbs the value below and above a deterministic baseline and compares
the assembled vector, topology class, posterior, and final class. No learned
parameters are committed as test fixtures.

### Exporting a source-bound classifier

The current MATLAB R2025a compatibility boundary can export modern
`ClassificationNaiveBayes` models, including the upstream 21-predictor normal
model and the alternate 37-predictor kernel model. The older bundled
`NaiveBayes` object reconstructs as an empty value in R2025a. Supporting it
requires a one-time inert numeric export under a compatible older MATLAB
release. The Python bridge accepts that legacy-class export, but still fails
closed when only the unreconstructed serialized object is available.

The one-time conversion is now a single version-aware command:

```powershell
acetree-starrynite-export-model `
  --starrynite-root C:\path\to\StarryNite `
  --matlab "C:\Program Files\MATLAB\R20xx\bin\matlab.exe" `
  --model C:\path\to\tracking-model.mat `
  --output C:\path\to\tracking-model.atpy-model `
  --kind auto
```

Use a MATLAB release that actually reconstructs the source object's saved
class. The compatibility helper deliberately uses the older
`-nodesktop -nosplash -r` launch interface and v7 MAT exchange rather than
`-batch`, so it can run under releases old enough to load `NaiveBayes`.
`ClassificationNaiveBayes` is handed to the current exporter; if the selected
release is too old for that wrapper, the command explicitly requests MATLAB
R2019a or newer for the modern model. It never substitutes a retrained or
approximate classifier.

For old models the helper reads `ClassLevels`, `Prior`, `Dist`, `Params`, and
the kernel/category metadata from the reconstructed object. It supports normal,
multivariate-multinomial, and unbounded Gaussian-kernel predictors. Before any
runtime model is saved, it creates finite parameter-center/mode and scale/category-
perturbed rows per class, calls
the same loaded object's `predict(..., 'HandleMissing', 'on')` and `posterior`,
and returns those probes with the numeric state. Python reconstructs the inert
model and must reproduce every probe class and posterior within a tight numeric
tolerance. This catches transposed `Params`, wrong category ordering, lost
kernel samples or weights, and class-order drift.

`--kind single` and `--kind ambigious` pin the expected layout. Auto mode
accepts either one `classifiermodel` or all four historical fields
`ambigious`, `fp_div`, `dirtyfp_fn`, and `divfp`; partial families, files that
contain both layouts, wrong branch class orders, and mask/predictor-count drift
all fail closed. The output receives a sibling `.provenance.json` manifest with
the source MAT SHA-256 and size, neutral JSON SHA-256, MATLAB executable,
version/release, compatibility-helper SHA-256, and StarryNite revision. The
source is hashed again after MATLAB exits, and the saved pair can be rechecked
with `validate_classifier_export_artifact` before use.

The native compatibility runtime also represents the older four-model family.
It preserves the serialized spelling `ambigious`, routes the six topology
cases to `ambigious`, `fp_div`, `dirtyfp_fn`, or `divfp`, and assembles only the
daughter/backward/forward blocks consumed by that branch. Its force mode uses
the original hard-coded positional class arrays, including the final class-1
fallback and class-2 demotion when no backward repair exists. A live parity
gate still requires an actual multi-model MAT file and a MATLAB release that
can reconstruct its four `NaiveBayes` objects; neither is present in the
upstream checkout used by the R2025a corpus.

The live R2025a gate for `2019TrackingModelv2.mat` currently covers all five
topology classes, the second topology-5 branch, and forced class selection.
MATLAB and Python assemble identical 21-value vectors (including NaN masks) and
choose the same class in all seven cases. The feature-sensitivity matrix adds a
baseline plus `-0.05`/`+0.05` perturbations for each of the 20 selected raw
features: all 41 classes agree and the maximum posterior absolute difference is
`4.86e-08`.

The alternate 37-predictor kernel model is also converted and scored, not only
inspected. Sample-center, weighted-mixture, periodic-NaN, and far-tail cases
match exactly in class with a maximum posterior absolute difference of
`3.05e-15`; both engines fall back to the prior after far-tail PDF underflow.
The Python high-level classifier evaluates distributions once per event and
caches read-only vectorized KDE arrays. On a local model-sized synthetic
122,544-component workload this reduced scoring from roughly `19 ms/event` in
the initial scalar implementation to `3.2 ms/event`.

Stage behavior is tested separately at every staging boundary using
`boundary - 1`, `boundary`, and `boundary + 1` cell counts. Ordinary sweeps pin
cell count and static diameter so feedback from one trial cannot silently alter
the next trial's effective kernel.

## Running locally

Run these commands from the AceTree-Py repository root with the development
dependencies installed. The live gate also needs (1) a local checkout of the
pinned StarryNite revision, (2) a MATLAB release able to run that checkout and
load the selected tracking model, and (3) the original parameter,
distribution, and model files at their source-resolved paths. MATLAB is an
oracle/test dependency only; it is not required to run an already exported,
supported classifier through AceTree-Py.

The smoke suite uses three scenarios and three points per detector parameter:

```powershell
python -m acetree_py.tracking.starrynite.oracle `
  --matlab "C:\Program Files\MATLAB\R2025a\bin\matlab.exe" `
  --starrynite-root "C:\path\to\StarryNite" `
  --parameter-file "C:\path\to\standard_parameters.txt" `
  --cell-count 4 `
  --suite smoke `
  --output C:\temp\starrynite-parity-report
```

Use `--suite full` for the 98-case scenario/sweep matrix. Use `--suite
resolution --seed-count 3` for the 81-scene multiseed daughter-separation and
noise matrix. The runner batches all requests into one MATLAB process because
upstream diameter estimation starts a parallel pool. Outputs are:

- `parity_report.json`: full metrics, matches, settings, hashes, versions, and
  known gaps;
- `detection_trials.csv`: flat table for plotting/statistics;
- `summary.md`: concise human-readable result.

Those outputs are generated diagnostics. For an ad hoc embryo, write them to a
temporary or study-specific directory; commit only deliberately curated parity
baselines and their provenance.

Live pytest checks are opt-in:

```powershell
$env:MATLAB_EXECUTABLE = "C:\Program Files\MATLAB\R2025a\bin\matlab.exe"
$env:STARRYNITE_ROOT = "C:\path\to\StarryNite"
python -m pytest -q -m matlab_oracle tests/test_starrynite_matlab_oracle.py
```

The live marker suite includes an exact post-greedy replay that batches the
original full tracking run and classifier export, rebuilds Python state from
the MATLAB initial checkpoint, and compares the complete ordered event trace.
It also exercises the registered public pipeline from synthetic images through
sequential detector state, every staged geometry summary, classifier cleanup,
division edges, and ancestry. Expect roughly 12 minutes for the current full
R2025a suite. A single skip is expected until an older MATLAB can reconstruct
the historical four-model `NaiveBayes` objects and a real four-model corpus is
available.

Under R2025a, the upstream classifier's `parfor` workers do not inherit the
driver's global learned-score state and return zero scores. The oracle harness
therefore runs a temporary serial-instrumented copy of the unchanged classifier
operations to observe the intended MATLAB result; it does not modify the local
StarryNite checkout. This exception is recorded as oracle provenance rather
than hidden as a Python tolerance.

Ordinary CI runs simulator, schema, orientation, matching, metric, curve, and
graph-normalization tests without MATLAB. Live MATLAB failures are reported as
oracle errors, never converted into similarity score zero.

## Regression diagnosis and corrected R2025a result

The first report used upstream `SD_red_40x.txt` staged values. Its 30 rows
contained only 21 unique effective settings because each sweep repeated the
baseline. The apparent broad loss was actually confined to the first
daughter-separation frame:

| Measure | Result |
|---|---:|
| Detector trials | 30 |
| Maximum filter relative L2 error | `4.82608e-06` |
| Mean MATLAB/Python detection F1 | `0.922222` |
| Minimum MATLAB/Python detection F1 | `0.666667` |
| Mean MATLAB-to-truth F1 | `0.977778` |
| Mean Python-to-truth F1 | `0.9` |
| Mean count-curve slope-sign agreement | `0.833333` |

Stage diagnostics identified three independent implementation regressions:

1. Python implicitly used the expected nucleus radius as a maxima-suppression
   distance. MATLAB uses 2-D `imregionalmax` and only the 18 voxels on adjacent
   Z planes, so a valid second daughter exactly 2 µm away was erased.
2. Python's support-weighted centroid pulled close daughters toward the same
   midpoint. MATLAB uses an integer 16-ray, valley-aware polygon recentering
   step and an odd-pixel 80th-percentile diameter.
3. `boundary_percent` must affect ray geometry, contiguous per-plane disk
   claims, the overlap graph, and conflict predicates. At low boundary values
   MATLAB merges two overlapping hypotheses; a simple distance or support-mask
   merge produces the wrong trend.

The rewrite now follows that staged sequence. Legacy profiles and every UI entry
point default to the ray-recentered position; support-weighted refinement is an
explicit native opt-in. The final reports are:

| Matrix | Trials | Engine F1 | Exact counts | Centroid p95 | Diameter MAE | Sensitivity |
|---|---:|---:|---:|---:|---:|---:|
| Smoke, unique settings | 21 | `1.0` | `1.0` | `0 µm` | `0 px` | `1.0` |
| Full scenario/parameter | 98 | `1.0` | `1.0` | `0 µm` | `0 px` | `1.0` |
| Resolution/noise, 3 seeds | 81 | `1.0` | `1.0` | `0 µm` | `0 px` | n/a |

Checked outputs: [smoke](reports/starrynite-parity-final-smoke/summary.md),
[full](reports/starrynite-parity-full-20260716/summary.md), and
[resolution/noise](reports/starrynite-parity-resolution-noise/summary.md).
The stage-by-stage diagnosis is recorded in the
[regression analysis](reports/STARRYNITE_REGRESSION_ANALYSIS.md).

In the resolution matrix both engines transition from one to two detections at
the same separation for every seed/noise combination. All nine noise-only
controls produce zero detections in both. Separations below the shared optical
transition intentionally score below perfect truth recall; equal MATLAB/Python
truth scores confirm this is shared resolution behavior, not rewrite drift.

Near-zero constant DoG controls make relative L2 ill-conditioned (MATLAB retains
single-precision residuals while Python can subtract to exact zero). The filter
gate therefore combines relative L2 with an absolute `1e-4` bound, correlation
of at least `0.999999999` when defined, and zero peak displacement for
non-negligible responses. The full matrix maximum absolute error is
`9.15527e-05`; no detector output changes.

The original `2019TrackingModelv2.mat` was also loaded unchanged in MATLAB
R2025a. On a deterministic five-frame movie with four translating nuclei it
produced four detections per frame and the expected 16 continuation edges with
no divisions. This is a model-compatibility smoke gate, not evidence that the
native geometry scorer reproduces the classifier.

## First full-lineage baseline and localized differences

The richer graph comparator was run against five deterministic movies using
the unchanged 2019 model. This table preserves the earlier native-geometry
baseline recorded before the exact 22/11/13 extractor and post-greedy decision
driver landed:

| Scenario | Node F1 | Edge F1 | Split-event F1 | Ancestry agreement | Main difference |
|---|---:|---:|---:|---:|---|
| Translation | `1.0` | `1.0` | `1.0` | `1.0` | Exact: 20 nodes and 16 links. |
| One missing observation | `1.0` | `0.974359` | `1.0` | `0.976285` | Python closes one gap; MATLAB leaves that branch fragmented (four components versus three). |
| Long daughter branches | `1.0` | `0.967033` | `0.0` | `0.987755` | Python's geometry scorer creates one split; four MATLAB classifier calls return class 0, so MATLAB retains no split and one additional root/component. |
| Positive class-1 division | `1.0` | `1.0` | `1.0` | `1.0` | Exact 33-node/29-edge graph with one atomic division after setting the native daughter-separation gate to `20 µm` for the fixture's `16 µm` pair. |
| Transient artifact | `0.96` | `0.904762` | `1.0` | `1.0` | One position misses the half-pixel match and two continuation identities differ; neither engine deletes a node in this fixture. |

This narrows the tracking regression substantially. Translation/linking is not
generally broken. The dominant differences are specific policy choices:
native eager gap closure, native geometry-only division acceptance, and one
crowded-frame detection/identity case. The model-backed classifier,
feature/candidate extractor, lineage resolver, early geometry driver,
isolated-fragment pass, and post-greedy dynamic event driver now form the
registered global-only exact runtime. The native tracker continues to identify
its split scorer as geometry-based; selecting a parameter file does not
silently switch behaviors. The exact positive-division result now runs from
images through the public pipeline and confirms detector row identity, split
representation, classifier replay, staged mutations, and AT graph plumbing in
one source-bound request. Historical native differences therefore remain useful
policy baselines, not limitations of the exact graph boundary.

The oracle now records complete `pred`/`suc`/`delete` pointer snapshots before
every classifier call in both legacy classifier families, including recursive
round-two calls. Python decodes these into an ordered stream of classifier
events and deterministic mutation batches, supports ID-aligned first-divergence
comparison, and freezes/replays traces as JSON. The live positive-division gate
records one round-one class-1 event and three checkpoints (initial,
pre-classifier, final). Each mutation batch is the exact normalized net delta
between snapshots, not an inferred order for writes within that interval.
Two additional live movies each exercise four genuine 2019-model class-0 calls
in rounds `[1, 2, 2, 2]`; Python matches all nine interleaved classification
and mutation-batch events in both. A separate batched gate now covers a genuine
class-3 deletion of an attached sub-four-frame branch and a genuine class-2
false-negative rewire across a missed-observation crossing. Each has one
round-one call and three observable events; Python matches the complete trace,
including the deleted branch or applied gap edge. Broader noisy and multi-event
class-2/3 movies remain corpus-expansion work, not an observability or
trace-format boundary.

The runtime provenance flag `event_order_validated` has a deliberately narrow
meaning: sequence indices, trace spans, record coverage, and unique MATLAB
frame/row scan order are structurally consistent within the events that the
runtime received. It does not prove that the source driver emitted every event.
Only comparison with an external MATLAB checkpoint/event oracle can support an
event-omission completeness claim for a fixture.

## Compatibility gates

The active gates are:

- Parameter selection, orientation, and kernel supports: exact.
- Default separable DoG: relative L2 at most `1.1e-5`, or correlation at least
  `0.999999999`, always with maximum absolute error `1e-4` and no meaningful
  peak displacement.
- Final detections: mean F1 at least `0.95`, no supported scenario below `0.8`,
  and explicit exact-count/matched-coverage reporting.
- Candidate geometry: at least `0.99` matched coverage, centroid p95 at most
  `0.25 µm`, diameter median relative error at most `5%`, p95 at most `10%`,
  and absolute signed bias at most `0.25 px`.
- Final DoG quality: p95 relative error at most `1e-5` on the matched cohort.
- Sensitivity characterization: mean slope-sign agreement at least `0.9` and
  normalized area between count curves at most `0.1`.
- Classifier: exact assembled vector/topology/class, posterior maximum absolute
  error within the frozen live tolerance, and `1.0` class agreement across the
  deterministic per-feature perturbation matrix.
- Tracking: exact graph for unambiguous translation and the registered
  image-to-lineage positive class-1 division fixture. Every early geometry
  boundary matches on the class 0/1/2/3 live corpus. Ordered
  classifier/checkpoint-delta tracing passes the positive division, two
  recursive class-0 movies, one class-2 false-negative repair, and one class-3
  cleanup. Broader noisy multi-event and representative whole-embryo cases are
  corpus-expansion work. Division events are always scored atomically rather
  than as two independent edges.

Frozen MATLAB outputs may be added to ordinary CI only after fixture and model
redistribution licensing is reviewed. Until then, local/nightly MATLAB runs and
fully reproducible manifests are the authoritative oracle.

Latest local verification on 2026-07-16 used the pinned upstream revision and
MATLAB R2025a. The non-live repository run completed with `1308 passed` and `71
skipped`. The complete opt-in MATLAB-oracle gate then completed with `19 passed`
and `1 skipped` in 11 minutes 37 seconds; the skipped case is the expected
historical-object export boundary that requires an older MATLAB release.
