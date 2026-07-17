# StarryNite Python Rebuild and Compatibility Plan

- **Status:** registered native and strict whole-movie compatibility backends; compatibility corpus expansion remains in progress
- **Plan version:** 1.0
- **Last updated:** 2026-07-16
- **Upstream reference:** [zhirongbaolab/StarryNite](https://github.com/zhirongbaolab/StarryNite), audited at commit [`e3d5ddc`](https://github.com/zhirongbaolab/StarryNite/tree/e3d5ddc381223ae8ce031f64947ffbd944a11593)
- **Host contract:** [AceTree tracking pipeline specification](TRACKING_PIPELINE_SPEC.md)
- **Latest verification:** `1308 passed, 71 skipped` without MATLAB; `19 passed, 1 skipped` in the complete opt-in R2025a oracle gate

## 1. Purpose and scope

This plan defines a systematic, clean-room rebuild of the StarryNite (SN)
detection and lineage-tracking workflow in Python. The end state is a native,
modular pipeline that:

1. accepts existing StarryNite parameter files and tracking models without
   silently changing their meaning;
2. can reproduce legacy detector, linker, division, false-positive, gap, and
   export behavior within declared tolerances;
3. plugs into the same AceTree-Py detector/tracker registry, proposal preview,
   validation, undo/redo, persistence, and provenance path as the existing LoG,
   DoG, and Simple LAP components;
4. provides an efficient native backend with deterministic results and an
   explicitly selectable legacy-exact mode;
5. supports a sparse, semi-automated selected-cell workflow that can follow
   both daughters after a division without attempting an embryo-wide solve;
6. exposes useful, comprehensible tuning controls while retaining a lossless
   link to the standard parameter file from which a run started.

The implementation now includes a registered strict compatibility profile and
a separate native-fast profile. The strict profile has a live image-to-lineage
gate for the supported 2019 classifier path, but it is still a scoped promise:
unsupported parameter statements, missing raw measurements, unmatched
calibration, non-unit downsampling, unvalidated classifier families, or changed
source identities fail closed. Release language must name the tested profile
rather than imply that every historical StarryNite installation is certified.

## 2. Upstream behavioral inventory

The rebuild is based on observable behavior and file contracts from the pinned
upstream revision, especially:

- the current MATLAB entry point,
  [`detect_track_driver_allmatlab_v2.m`](https://github.com/zhirongbaolab/StarryNite/blob/e3d5ddc381223ae8ce031f64947ffbd944a11593/launcher_interface/detect_track_driver_allmatlab_v2.m);
- line-oriented parameter loading in
  [`readParameters.m`](https://github.com/zhirongbaolab/StarryNite/blob/e3d5ddc381223ae8ce031f64947ffbd944a11593/distribution_code/readParameters.m);
- stage- and region-dependent parameter lookup in
  [`getParameter.m`](https://github.com/zhirongbaolab/StarryNite/blob/e3d5ddc381223ae8ce031f64947ffbd944a11593/distribution_code/getParameter.m);
- the current classifier-based lineage driver,
  [`tracking_driver_new_classifier_based_version.m`](https://github.com/zhirongbaolab/StarryNite/blob/e3d5ddc381223ae8ce031f64947ffbd944a11593/distribution_lineaging/tracking_driver_new_classifier_based_version.m);
- the upstream distribution notice,
  [`distribution_code/license.txt`](https://github.com/zhirongbaolab/StarryNite/blob/e3d5ddc381223ae8ce031f64947ffbd944a11593/distribution_code/license.txt).

The observable pipeline is treated as the following set of replaceable stages,
not as one monolithic tracker:

```text
legacy parameters + image metadata + model
  -> processSequence / processVolume
  -> createDiskSet
  -> findOverlookedNuclei
  -> resolveConflicts
  -> initializeTrackingStructures
  -> linkEasyCases
  -> gatherEndCandidates
  -> greedyEndScore
  -> greedydeleteFPbranches
  -> saveGreedyNucleiFiles / AceTree conversion
```

Important behaviors that must be captured in fixtures before declaring parity:

- Parameters are MATLAB statements evaluated line by line upstream. Real files
  can contain comments, quoted strings, booleans, vectors, arithmetic, repeated
  assignments, missing semicolons, `load` statements, function handles, and
  arbitrary unsupported statements.
- A scalar parameter is global. A vector can be stage-indexed. Upstream stage
  selection uses a strict less-than boundary, and regional overrides use boxes
  with lower-exclusive and upper-inclusive coordinate tests.
- The tracked sequence contains per-frame structures, marks false positives for
  deletion, represents divisions explicitly, and materializes missing frames
  during legacy export.
- Coordinates pass through MATLAB 1-based, ROI-local, downsampled image space
  before AceTree export. XY downsampling, ROI offsets, the legacy export offset,
  and Z anisotropy (`zres / xyres * downsampling`) must be tested independently.
- Bundled classic MAT models contain numeric tracking parameters plus serialized
  MATLAB classifier objects. Both the older `NaiveBayes` and newer
  `ClassificationNaiveBayes` families occur. Those opaque objects cannot be
  assumed to produce correct predictions merely because `scipy.io.loadmat` can
  inspect their container.

## 3. What “compatible” means

Compatibility is versioned and reported by tier. A release must not use the
unqualified phrase “StarryNite compatible” unless all required tiers for that
release profile pass.

| Tier | Promise | Required evidence |
|---|---|---|
| C0: preservation | Open a legacy parameter/model reference and preserve its source, unknown statements, path, and hash without executing code. | Lossless round-trip and malicious-input tests. |
| C1: parameter semantics | Produce the same effective scalar, staged, and regional value at a specified cell count and coordinate. | Golden queries against MATLAB `getParameter`. |
| C2: model semantics | Produce the same feature transformation, class score/probability, and decision for old and new model families. | Neutral model export plus golden feature vectors and predictions. |
| C3: stage parity | Match each detection/tracking stage within a declared numerical and tie-breaking tolerance. | Intermediate-stage fixture comparison. |
| C4: lineage parity | Match links, divisions, gaps, deletions, and retained nuclei on the compatibility corpus. | Canonical graph/event comparison. |
| C5: export/workflow | Produce semantically equivalent AceTree nuclei records and measurements through the standard proposal/commit path. | Reload, round-trip, undo/redo, and Java/Python AceTree checks. |

The native fast profile may intentionally differ from C3 numerical details, but
must still satisfy the API, graph, export, determinism, and provenance contracts.
Its output must identify the backend as `native-fast`, never as `legacy-exact`.

## 4. Current implementation status

The following table describes the experimental foundation in this branch. It
is deliberately conservative.

| Area | Available now | Still required for parity |
|---|---|---|
| Parameter files | Non-executable parser for a safe MATLAB/classic data subset; typed values; dotted keys; last-assignment-wins; opaque-record preservation; source-lossless rendering; model-reference discovery; exact strict-boundary stage selection; lower-exclusive/upper-inclusive regional resolution for inert numeric region records; actionable compatibility reports; conservative mapping into native settings; and live `getParameter` golden tests at stage and box boundaries. | Handling/conversion policy for dynamic/control-flow constructs, a broader real-file corpus, and golden queries for every parameter used by an enabled exact profile. |
| Models | Classic MAT inspection; SHA-256 identity; versioned, immutable neutral Naive Bayes schemas for the 2019 single model and historical four-model `ambigious` family; categorical, normal, and unbounded Gaussian-kernel predictors; strict source-hash binding; exact topology/mask assembly and multi-model routing; native posterior, cost-sensitive prediction, and branch-specific force-mode evaluation; and a one-command old-release exporter that replays per-class MATLAB prediction/posterior probes before saving a provenance-bound artifact. Serialized MATLAB objects are never executed in Python. | Run the exporter under a MATLAB release that reconstructs the bundled old `NaiveBayes` object and acquire an actual four-model MAT corpus. R2025a correctly fails closed because the old object reconstructs as empty; Python accepts only inert numeric state whose probes match the same loaded MATLAB object. |
| Detection | Native anisotropic DoG plus an exact distribution-backed tail: 8-connected `imregionalmax` plateau semantics and MATLAB `find` order, adjacent-plane maxima, 16-ray diameter/recentering, disk log odds, maximal ranges, overlooked-nucleus rounds, merge/split conflict predicates, staged/regional lookup, sequential previous-frame diameter/cell-count state (including zero-output frames), polygon filtering, and source-defined rectangular ROI cropping with coordinate restoration. Exact requests bind a strictly parsed distribution MAT by SHA-256, rehash sources during execution, reject non-finite voxels, and preserve legacy row identity; source caches reset on discontinuity. | Broader integer-image, crowded-embryo, regional-ROI, and pathological tie corpus; legacy image-loader/orientation choices remain an input-adapter concern rather than hidden detector behavior. |
| Tracking | Exact initialization, easy links, polar-filter boundary, asymmetric candidate gathering, nondivision/division threshold sweeps, isolated-fragment prepass, immutable row/slot state, one-based binary32 distances, exact 22/11/13 features, single- and four-model classifiers, four-attempt class-0 repair, deleted-row attachment behavior, class 0/1/2/3 mutations, and complete MATLAB frame/row event order. Every early geometry stage and the live class 0/1/2/3 post-greedy cases match the pinned oracle; a registered image-to-lineage division case matches nodes, rows, edges, stages, classifier event, and ancestry. Production retains bounded stage summaries while the full snapshot path remains available for differential testing. | Polar-body and hysteresis modes remain fail-closed until their raw detector measurements cross the AT boundary; expand noisy multi-event class-2/3 and whole-embryo performance corpora. Historical four-model execution is implemented but still needs a real MAT corpus for live certification. |
| AT integration | Native and `acetree.starrynite_legacy_exact` trackers coexist with LoG, DoG, Simple LAP, and the shared StarryNite detector. An immutable `preflight_movie` boundary validates complete global scope, channel, calibration, detector binding, and every legacy source before frame 1; `TrackingPipeline.refine_movie` revalidates those inputs and supplies progress/cancellation without a stale-cache bypass. Returned omissions and invented positions are validated. Split edges use the ordinary preview and atomic lineage-commit path. Compatibility provenance records parameter, distribution, MAT model, classifier hashes, bounded early-stage summaries, class counts, a classifier-trace hash, and structural event-order validation. That validation proves internal sequence/span/coverage/order consistency for received events, not that the source driver omitted none. | Complete import/export corpus validation, large-movie profiling, packaging/release hardening, and downstream validation against representative curated embryos. |
| Sparse forward mode | Selected-forward scope has stop/follow-best/follow-both policy hooks; splitting capability is gated; preview and commit retain two daughter branches; standard parameter files populate editable basic controls and can be saved losslessly as tuned copies; compatibility/model diagnostics are visible; and the most recent usable file is shared across the whole-movie and sparse workbenches. | Broader usability tests and branch-budget/ambiguity acceptance studies on partial lineages. |

This status is a feature inventory, not a release guarantee. The verification
gates in section 11 determine when a row may be promoted from experimental.

## 5. Target architecture

The native implementation remains headless and is divided into compatibility,
algorithm, orchestration, and presentation layers:

```text
acetree_py/tracking/starrynite/
  parameters.py       safe parse, lossless render, model references
  parameter_view.py   staged/regional legacy lookup and diagnostics
  models.py           MAT inspection, hashes, normalized model metadata
  model_schema.py     versioned neutral classifier representation
  presets.py          legacy -> detector/tracker settings mapping
  detector.py         public Detector implementation
  legacy_detector_tail.py exact learned/recovery/conflict detector stages
  detection/
    preprocessing.py  anisotropy, filtering, ROI/downsampling transforms
    candidates.py     maxima and disk/support construction
    recovery.py       overlooked-nucleus pass
    conflicts.py      duplicate and overlap resolution
    measurements.py   volume/intensity/weight feature extraction
  tracker.py          public Tracker implementation
  classifier.py       neutral model runtime and exact topology masks
  legacy_state.py     immutable row/slot/nearest-neighbor feature state
  legacy_features.py  exact 22/11/13 event and retained-snapshot extraction
  repair_candidates.py exact backward/forward/class-0 candidate enumeration
  legacy_class_zero.py recursive provisional class-0 repair orchestration
  legacy_mutations.py raw deleted-row-aware class-3 mutations
  legacy_driver.py    atomic post-greedy movie scan and decisions
  legacy_early.py     initialization through staged geometry sweeps
  legacy_isolated.py  greedydeleteFPbranches isolated-fragment prepass
  legacy_runtime.py   source-order MAT/parameter runtime preparation
  legacy_exact_tracker.py registered fail-closed whole-movie composition
  lineage.py          immutable class 0/1/2/3 graph mutations
  bifurcation.py      typed classifier-to-lineage boundary
  tracking/
    features.py       link/end/division feature vectors
    candidates.py     bounded nearest-neighbor graph
    easy_links.py     conservative assignments
    endpoints.py      continuation, gap, division hypotheses
    scoring.py        geometry or normalized legacy model scorer
    cleanup.py        false-positive branch decisions
  coordinates.py      explicit MATLAB/image/physical/AceTree transforms
  compatibility.py    reports, backend selection, unsupported-feature policy
  provenance.py       source/model hashes and resolved-setting snapshot
```

The public detector and tracker only consume and return the immutable AT
tracking API. They do not access Qt, napari, `NucleiManager`, naming, or ZIP
persistence. UI code edits a validated settings map and submits a normal
`TrackingRequest`; lineage mutation still happens only through
`ApplyTrackingProposal`.

The compatibility adapters must not become a second pipeline. Legacy and native
components can be combined through the registry when their units and feature
requirements are satisfied:

| Detector | Tracker | Supported intent |
|---|---|---|
| LoG or DoG | Simple LAP | Existing baseline. |
| LoG or DoG | SN native division tracker | Division-aware linking of generic spots, with feature fallbacks recorded. |
| SN native detector | Simple LAP | SN-style candidates with one-to-one tracking. |
| SN native detector | SN native division tracker | Full native SN workflow. |
| SN source-bound exact detector | SN legacy-exact tracker | Registered global-only compatibility profile when its readiness report passes. |

Component schemas must declare required detection features and capabilities.
The registry or pipeline must reject an invalid pairing before starting a long
run rather than silently dropping a classifier feature.

## 6. Legacy parameter-file strategy

### 6.1 Safety and preservation

Arbitrary parameter files must never be evaluated by Python. The parser accepts
only a bounded data grammar. It retains comments, whitespace, unknown statements,
line endings, and statement order so that an unchanged file can be rendered
losslessly. Repeated assignments retain their history while the effective value
follows the legacy last-assignment-wins rule.

Every load produces a compatibility report containing:

- detected syntax and encoding;
- parsed settings and their source locations;
- opaque statements and why they were not interpreted;
- discovered model references and resolved paths;
- required but missing parameters;
- effective values at the selected cell count, stage, and coordinate;
- all native overrides and their provenance.

Unknown statements are never treated as harmless by default. A strict
compatibility run fails closed if an opaque statement can affect a required
setting. The tuning UI may proceed in native mode after showing the limitation
and recording it in the proposal.

### 6.2 Exact semantic resolver

Implement a typed resolver separate from parsing. It must reproduce upstream
`getParameter` rules, including:

1. scalar/global values;
2. vector values indexed by developmental cell-count thresholds;
3. strict threshold-boundary behavior;
4. regional override precedence;
5. lower-exclusive and upper-inclusive region bounds;
6. MATLAB row/column and one-based indexing where it affects selection;
7. explicit errors for ambiguous dimensions instead of shape guessing.

Each resolved setting carries `(value, source record, stage, region, unit)` so
the wizard and provenance ledger can explain why a value was selected.

### 6.3 Editing and presets

Loading a standard parameter file creates an immutable base preset. User tuning
is stored as a small override layer, not by rewriting the original assignments.
When the user asks to save an updated legacy file, append supported overrides so
legacy last-assignment behavior is preserved and include an adjacent manifest
with source hash, output hash, AT version, and unsupported constructs.

## 7. Tracking-model strategy

### 7.1 Normalized model format

Define a versioned, implementation-neutral format such as
`acetree.starrynite-model/v1`, stored as JSON metadata plus NPZ arrays. It must
contain:

- feature names, order, units, masks, and missing-value policy;
- normalization constants;
- class priors and class-conditional parameters;
- old/new classifier family and upstream version evidence;
- decision threshold and tie policy;
- original MAT SHA-256, size, and conversion-tool version;
- a set of non-sensitive golden input vectors and expected scores/classes.

Python must validate shapes, finiteness, feature ordering, and golden predictions
before enabling the model. A model is unusable for exact mode if any required
classifier state remains opaque.

### 7.2 One-time MATLAB exporter

Provide a small, separately distributed converter that runs in a licensed
MATLAB environment. It loads the original model using MATLAB's own class
implementation and exports only the neutral numeric representation and golden
predictions. The converter is a compatibility boundary, not a runtime dependency
of AceTree-Py.

If a classifier cannot be represented faithfully, offer one of two explicit
choices:

- use a controlled, out-of-process MATLAB oracle for validation only; or
- use the native geometry/model scorer and label the run `native`, not exact.

Never deserialize an opaque MATLAB object into executable Python and never infer
a classifier layout from array shape alone.

### 7.3 Model conformance

For every supported old/new model family, compare Python with MATLAB on raw
feature construction, normalized features, per-class scores/posteriors, selected
class at ordinary and boundary cases, and the downstream link/division choice.
The model hash, neutral-schema hash, backend, and converter version become part
of proposal provenance.

## 8. Algorithm rebuild

### 8.1 Detection

Rebuild and validate detection one stage at a time:

1. **Input normalization:** channel choice, bit depth, ROI, XY downsampling,
   plane range, dark/bright convention, and physical calibration.
2. **Filtering:** anisotropic smoothing/DoG with documented boundary behavior
   and explicit float precision.
3. **Candidate generation:** deterministic local maxima and plateau tie rules.
4. **Disk/support estimation:** bounded 3-D support, radius/volume, centroid,
   and intensity measurements, including the legacy Z-disk log-odds extension
   where `rangethreshold` is a tolerance rather than a local contrast cutoff.
5. **Recovery:** overlooked-nucleus candidates using the same exclusion and
   threshold semantics as upstream.
6. **Conflict resolution:** duplicates, overlapping supports, boundary cases,
   and stable selection ordering.
7. **Contract conversion:** emit physical-coordinate `Detection` values plus
   named legacy features; do not leak internal mutable structures.

Intermediate artifacts must be serializable in a test-only diagnostic format.
That makes a mismatch attributable to a stage instead of only to final counts.

### 8.2 Tracking

Rebuild tracking as a sparse hypothesis graph with deterministic passes:

1. initialize one node per accepted detection and import fixed anchors;
2. build bounded forward/backward nearest-neighbor candidate lists;
3. apply conservative easy-link rules;
4. gather open endpoints and continuation, gap, deletion, and division
   hypotheses;
5. compute the exact legacy feature vector or native interpretable score;
6. select hypotheses in a stable greedy order matching the compatibility
   backend's tie policy;
7. prune false-positive branches without deleting protected/manual nuclei;
8. validate graph invariants and convert to AT `TrackEdge` values.

Required graph invariants are: at most one predecessor, at most two successors,
exactly two outgoing `split` edges for a division, no merges, forward-only time,
and explicit `gap` edges for non-adjacent frames. The host remains responsible
for materializing intermediate legacy nuclei at commit time.

### 8.3 Exact and fast profiles

- **`legacy-exact`:** fixed float precision, legacy ordering, legacy boundaries,
  normalized legacy classifier, and all conformance gates enabled. Optimizations
  are allowed only when proven output-equivalent.
- **`native-deterministic`:** interpretable scoring and modern sparse algorithms
  with stable ordering. This is the default while exact parity is incomplete.
- **`native-fast`:** optional parallel/GPU acceleration. It may use documented
  numerical tolerances but must remain graph-deterministic for a fixed backend,
  hardware class, version, and seed.

## 9. Performance and engineering practices

Modernization must improve implementation quality without obscuring legacy
behavior:

- vectorize filtering and measurement operations with NumPy/SciPy;
- use `scipy.spatial.cKDTree` or an equivalent spatial index for bounded
  candidates instead of all-pairs distance matrices;
- solve only sparse local assignment/hypothesis problems;
- stream volumes and retain only the frame window required for links/gaps;
- parallelize independent per-frame detection with ordered result reduction;
- keep arrays contiguous, typed, and bounded; avoid per-voxel Python objects;
- cache parameter resolution and model normalization by content hash;
- make cancellation checks and progress stages part of all long passes;
- enforce deterministic IDs and explicit secondary sort keys;
- keep optional Numba/GPU backends behind the same conformance suite;
- profile on representative small, medium, and full embryo datasets before
  selecting optimizations.

Benchmark reports record wall time, peak resident memory, candidate-edge count,
backend, CPU/GPU, thread count, Python/SciPy versions, and result hash. Performance
budgets are established from the compatibility corpus before release; speedups
never compensate for a failed correctness gate.

## 10. AceTree integration and sparse forward workflow

### 10.1 Shared AT integration

All SN modes use the existing AT pipeline:

```text
ComponentRegistry -> PipelineRunner -> immutable TrackingProposal
  -> preview/validation -> explicit Accept
  -> one ApplyTrackingProposal command -> undo/redo + standard save
```

Integration requirements:

- use calibrated physical coordinates at the component boundary;
- centralize and test conversion to AceTree pixel/plane coordinates;
- preserve manual anchors and forced names;
- map supported legacy measurement fields (`weight`, `rweight`, raw/corrected
  intensity) explicitly rather than through positional columns;
- materialize gaps and divisions through the normal lineage data model;
- never commit analysis results automatically;
- record source parameter hash, effective settings, overrides, model hash,
  detector/tracker IDs and versions, backend, calibration, ROI, image identity,
  warnings, and proposal hash;
- keep accepted nuclei ZIP files readable when SN components are absent.

Whole-embryo tracking and selected-forward tracking share the detector and
tracker implementations. The scope changes orchestration and review behavior,
not file format or lineage mutation.

### 10.2 Sparse semi-automated forward mode

Sparse mode starts from one selected live nucleus and only explores local search
regions around active branches. It is intended for a few cells, not complete
embryo tracking.

At each frame it must:

1. predict a local search region for every active branch;
2. detect and deduplicate candidates within those regions;
3. score continuation and division hypotheses;
4. accept a continuation, atomically accept a two-daughter split, or pause;
5. stop on ambiguity, collision with protected lineage, branch-budget overflow,
   missing image, or an unsupported model/parameter construct;
6. preview all proposed positions, gaps, and daughter labels before commit.

Division behavior is an explicit policy:

- **Stop at division:** end the proposal at the parent for manual handling.
- **Follow best:** continue one branch only; never pretend that the untracked
  daughter does not exist in provenance.
- **Follow both:** require a tracker advertising `splitting`, emit exactly two
  split edges, and add both daughters to the active branch set.

The default is conservative: stop when confidence is insufficient. Multiple
terminal branches must not cause navigation to select an arbitrary endpoint.

### 10.3 Parameter-file setup and tuning workbenches

The implemented UI has two wizard-like entry points. The dataset-creation
wizard can open the whole-dataset workbench for an empty movie, while **Auto
Forward** opens the selected-lineage workbench from one live nucleus. Both offer
**Start from StarryNite parameters…**, a shared **Use recent: _filename_**
shortcut, **Save tuned parameter copy…**, and **Compatibility details…**.

Loading a standard file safely parses its inert assignments, resolves model and
distribution references relative to the source, chooses effective staged values
for the starting cell count, and fills the basic editable controls. The recent
path is persisted only while it remains a usable file. A validated classifier
export is remembered separately by the SHA-256 of its source MAT model, so one
model's numeric state cannot be silently reused for another.

The sparse control surface keeps expected radius, intensity threshold, local
search area, maximum movement, gap allowance, ambiguity caution, end time, and
division policy visible. Native-only advanced detector controls remain
available without being written into a legacy file. **Follow both daughters**
is enabled only for a splitting tracker and grows only the selected sparse
branches.

Saving never overwrites the loaded source implicitly. It preserves original
text, comments, and unsupported statements and appends compatible edits for
stage-aware radius, intensity threshold, and missing-frame allowance. The saved
copy is reloaded immediately and becomes the recent source. Search radius,
movement caution, and other Python-only review settings remain in the workbench,
not the legacy copy. Relative model paths therefore require the copy to remain
beside the original source or to be updated explicitly.

The global workbench alone can select **StarryNite legacy exact (whole movie)**
and execute the attached source-bound numeric classifier. Exact mode requires
the full movie from time 1, an empty nuclei record, matching calibration, unit
downsampling, and every referenced source. Any visible tuning change must first
be saved and reloaded as a parameter copy. Auto Forward deliberately uses the
native division tracker; its classifier action validates compatibility for the
report only.

## 11. Verification gates

### Gate A — governance and corpus

- Pin every audited upstream revision and record source file hashes.
- Obtain legal/licensing review for redistribution, fixtures, and converter
  boundaries.
- Build a corpus spanning parameter syntax, old/new models, stages, regional
  overrides, ROI/downsampling combinations, divisions, gaps, false positives,
  and boundary nuclei.
- Store only redistributable fixtures in this repository. Keep restricted user
  data/models outside the tree and version their manifests/hashes instead.
- Freeze expected tolerances and tie policies before tuning Python against the
  corpus.

### Gate B — parameter and coordinate compatibility

- Byte-for-text round-trip for unchanged files, including CRLF and comments.
- No execution for malicious/function-handle/system-call inputs.
- Golden parity for every scalar/stage/region query.
- Coordinate round trips for ROI-local, downsampled, anisotropic, MATLAB, AT,
  negative-offset, and half-pixel boundary cases.
- Actionable diagnostics for every unsupported statement.

### Gate C — model compatibility

- Neutral export validates original and converted hashes.
- Python matches MATLAB golden vectors for both classifier families within the
  frozen numerical tolerance.
- Boundary, missing-feature, NaN, and exact-tie behavior matches.
- Corrupt, wrong-family, v7.3, incomplete, and oversized models fail safely.

### Gate D — detection stages

- Compare filtered volumes, candidate maxima, support masks/statistics,
  recovered nuclei, and conflict decisions separately.
- Compare final detection count, coordinates, radius/volume, intensity, and
  stable ordering on every corpus frame.
- Verify results across supported Python/SciPy platforms.

### Gate E — tracking and lineage

- Compare easy links, endpoint candidates, scores, greedy choice order,
  deletions, gaps, and divisions separately.
- Canonical graph comparison matches retained nodes and event topology.
- Determinism holds across repeated runs and allowed thread counts.
- No graph violates AT predecessor/successor, time, merge, split, or gap rules.

### Gate F — workflow and export

- Preview is non-mutating; cancel is a no-op; accept is one atomic command.
- Undo and redo restore exact lineage and measurement state.
- Legacy nuclei output reloads in AceTree-Py and the supported Java AceTree
  reader with equivalent topology, coordinates, status, and measurements.
- Sparse forward mode covers continuation, split, ambiguity, collision, gap,
  branch-limit, cancellation, stale seed, and multi-terminal navigation cases.
- Provenance is sufficient to reproduce settings or explain why exact
  reproduction is unavailable.

### Gate G — performance and release

- Meet frozen runtime and memory budgets without correctness regressions.
- Complete full-suite, fuzz/property, platform, and long-run cancellation tests.
- Publish a compatibility matrix by parameter/model family and backend.
- Mark `legacy-exact` available only for combinations that pass C0–C5; other
  combinations remain explicitly experimental/native.

## 12. Delivery sequence

Work should land in reviewable increments, each usable by the next phase:

| Phase | Deliverable | Exit condition |
|---|---|---|
| 0 | Governance, pinned audit, fixture policy, MATLAB oracle harness. | Gate A complete. |
| 1 | Safe parser/model inspector and compatibility-report schema. | C0 and parser security/round-trip tests pass. |
| 2 | Exact staged/regional parameter resolver and coordinate module. | Gate B complete. |
| 3 | Modular detector stages with diagnostic artifacts. | Gate D passes for an initial representative corpus. |
| 4 | Sparse candidate graph, easy links, endpoints, gaps, divisions, cleanup. | Native tracker graph invariants and deterministic tests pass. |
| 5 | Neutral model exporter and old/new model scorers. | Gate C complete for each enabled model family. |
| 6 | Legacy-exact tracking parity and AT export conformance. | Gates E and F complete for the compatibility profile. |
| 7 | Sparse forward preset mapper, wizard, branch review, usability pass. | Sparse workflow cases in Gate F pass. |
| 8 | Profiling, optional acceleration, platform matrix, release documentation. | Gate G complete. |

Phases may overlap where interfaces are already frozen, but compatibility claims
advance only by gate. The current branch contains substantial Phase 1/2
foundations, a native Phase 3/4 implementation with detector parity gates passing
on the current synthetic corpus, Phase 5 conformance for the modern 2019 model,
an old-release exporter ready to certify historical models, the Phase 7
workflow, and a registered Phase 6 exact runtime for the supported 2019 profile.
Phase 5 remains open for the historical four-model family. Phase 6 remains open
for broader noisy and representative whole-embryo corpora and for legacy modes
whose raw detector measurements do not yet cross the AT boundary; the
`greedydeleteFPbranches` decision scan and registered exact-runtime/export
boundaries are implemented for the supported profile.

The executable MATLAB-oracle protocol, metric definitions, simulator corpus,
local command, diagnosed before/after R2025a measurements, and promotion thresholds are
specified in [STARRYNITE_DIFFERENTIAL_TESTING.md](STARRYNITE_DIFFERENTIAL_TESTING.md).

## 13. Licensing and clean-room boundary

AceTree-Py is MIT licensed. The audited StarryNite tree does not present a single
uniform root license: the core distribution includes a GPL notice and bundled
components carry additional notices. This plan is not legal advice. Before
redistributing upstream-derived assets or claiming a source-compatible port,
the maintainers must obtain an appropriate licensing review.

Implementation rules until that review is complete:

- do not copy, mechanically translate, or lightly rewrite upstream MATLAB code;
- derive requirements from public interfaces, observable behavior, independent
  tests, and separately recorded audit notes;
- keep the Python algorithm implementation independent and reviewable;
- keep MATLAB oracle/export tools and any restricted models or fixtures outside
  the MIT runtime distribution unless their licenses are resolved;
- document the origin and license of every fixture and model;
- do not bundle upstream JARs, NIFTI/LSM utilities, models, or sample data by
  assumption;
- make the optional external compatibility boundary explicit to users.

This boundary permits a useful native tracker now while preserving a credible,
testable route to full legacy compatibility later.
