# Alpha v2 validation log

## Baseline (2026-09-05, e31b61f)

Existing non-MATLAB suite: 1823 passed, 56 skipped, 19 deselected, 66 warnings, 63.66 seconds.
Command: python -m pytest tests -q -m "not matlab_oracle" -p no:cacheprovider --basetemp <new workspace temp path> --durations=10.
Runtime: bundled Python 3.12; existing workspace .test_deps; QT_QPA_PLATFORM=offscreen; MPLBACKEND=Agg; bytecode writes disabled. Full napari unavailable at initial audit.
Static Python parse: no syntax errors in package/tests.

## Audit probes

- Absolute plane 1 with intensity 100 yielded nuclear raw/blot mean 0.
- Missing configured channel 1 silently remapped channel 2 to logical 0.
- Same-parent relink/undo broke reciprocal successor; canonical duplicate roots merged.
- Tracking follow_both persisted/reopened as stop; nested request settings retained caller mutations.
- Qt: Class B filter left hidden Class A target deletable and measured A under Current class.
- Qt minimum widths: ROI Measure 1414 px; Objects panel 692 px.
- Object row calculation: 200 tracks x 400 frames averaged 28.2 ms before Qt work.
- Real naming/history: 42 edit/undo/redo states had matching nucleus/tree names.

## Implementation validation

Append focused commands/results with each outcome. Reuse existing behavioral tests; add regressions for demonstrated gaps only. Do not add coverage quotas or implementation-shaped microtests.

## Final acceptance requirements

- Full existing non-MATLAB gate, plus new focused behavioral regressions.
- Live napari workflow smoke at 1280x720: editing, ROI creation/filter/review, measurement/cancel, plots/export, save/reopen, tracking launchers, independent 3D/expression windows.
- Memory evidence for ROI task groups and tracking affected-only snapshots; navigation evidence on 200 tracks x 400 frames.
- Inspect final worktree status and commits; preserve source/untracked reports.

### R1 â€” clean ROI Save As

Save As now copies clean ROI state when the destination changes and retargets only after the coordinated commit. Added one clean load/copy/reopen/edit/save scenario that checks the original sidecar remains unchanged.

`pytest tests/test_app_save.py tests/test_roi_save_transaction.py tests/test_roi_end_to_end.py -q`: **23 passed** in 1.41s.

### R2 â€” alpha installation

Branch guards and CLI identify alpha v2. Historical plugin proposal is explicitly deferred. Replaced brittle prose/source-string checks with existing executable installer cases and one CLI version check. Remote install instructions explicitly require branch publication.

`pytest tests/test_tracking_branch_installation.py tests/test_cli.py -q`: **18 passed** in 6.52s. Workflow documentation follows UI integration.

### B1 â€” editing and lineage integrity

Distinct canonical roots remain explicit conflicts; same-parent relinks are no-ops; Add/Relink/Interpolation validate before mutation. Interpolation now composes existing commands instead of duplicating mutation/undo logic.

Agent gate: editing, lineage, nuclei_manager, edit_panel, post_commit_ui, identity, tracking_integration: **346 passed** in 5.33s. Reviewed diff before integration.

### C1 â€” ROI selection, scope, visibility

Filtered-out objects no longer remain mutation targets; class measurement uses visible class; missing-cell filter has an empty state; image overlays share filter IDs and retain explicit visibility across 3D. Invalid measurement requests stay in a wrapped, scrollable dialog.

Agent gate: ROI UI models, viewer integration, and Objects panel: **24 passed**. ROI Measure now opens at **560 x 650**, down from minimum width 1414. Reviewed diff before integration.

### GUI environment

Installed tested napari 0.6.6 in isolated workspace .alpha-v2-venv. Hidden Windows-backend real-OpenGL canvas probe succeeded (nonuniform synthetic image, RGB standard deviation 82.3). Offscreen plugin alone cannot render OpenGL.

### A2 â€” stable image channel identity

Explicit channel sets reject missing paths, gaps, unreadable samples, or unparseable time patterns instead of renumbering survivors. The app can still open nuclei and reports the source error. One missing-first-channel/recovery scenario checks both scientific identities.

`pytest tests/test_image_split.py tests/test_image_provider.py tests/test_interleaved_tiff.py tests/test_gui_app.py tests/test_cli.py -q`: **214 passed**, 14 existing TIFF warnings in 6.13s.

### A1 â€” nuclear coordinates and historical provenance

Absolute Z origin now reaches raw/blot reducers. Algorithm 2 includes origin freshness; algorithm1 captures open as historical without relabeling, and incompatible numerical comparisons request recompute/exclusion. Recomputing older measurements intentionally changes values affected by the original shifted sampling.

Agent measurement/repository/comparison gate: **219 passed** in 43.74s. Reviewed implementation and compatibility changes before integration.

### B2 â€” reproducible tracking requests

Scope serialization includes branch_policy (missing old fields default stop). Nested JSON inputs are frozen; mutable public exports detach nested values. Immutable tuples retain sharing for existing exact-tracker provenance deduplication.

Agent tracking/global/StarryNite gate: **226 passed, 2 skipped**; graph provenance export follow-up: **53 passed, 1 skipped**. Reviewed diff before integration.

### C3 — object management and navigation

Class name/color management and expected-span editing use undoable commands. Completeness requires reviewed decisions throughout the expected interval, including reviewed absence. History rows cache immutable document revisions; Z-only navigation does not rebuild the object list.

Agent gate: **33 passed**; touched-file lint clean. Same-process 200 objects x 400 frames benchmark: full history rebuild **66.8 ms**, cached time change **0.973 ms**, full-panel Z change **0.012 ms** (medians). Reviewed new dialog and panel code before integration.

### B3 — bounded tracking acceptance memory

Acceptance stores only changed existing successor fields and appended-frame lengths; global tracking submission no longer copies unused nuclei. Failure after partial installation, tombstones, empty records, undo and redo retain exact results and nucleus identity.

Agent gate: **125 passed**. Sparse continuation across 100,000 existing nuclei/200 frames: command peak allocation **30,497.43 KiB -> 6.89 KiB**; retained **30,480.60 KiB -> 4.13 KiB**. Measured ApplyTrackingProposal itself, excluding EditHistory naming snapshots. Identical accepted geometry, indices, links and undo were verified.

### Integrated checkpoint

At ca96488, venv/offscreen run: **1889 passed, 8 skipped, 19 deselected, 8 failed**. Seven failures were OpenGL context creation under Qt offscreen; real Windows-backend rendering works. One old build-label assertion was corrected to alpha v2. Final gate uses the functioning renderer.

### A3 — bounded ROI image memory and operation freshness

Decoded planes/stacks are released when sorted time/channel task groups advance and reused by objects within each group. Prepared read contexts validate image manifests once per plot/export and index the immutable document for sample checks. Direct readers continue to validate their own dependencies.

Agent gate: **37 passed**. Mixed 2D/3D movie regression verified one live decoded group, six plane and six stack reads across three times/two channels, unchanged values, and all decoded arrays released after completion. Plot/export scenario verifies one manifest read per operation, metadata stability, geometry invalidation/recompute and external-file rejection.

Windows GUI checkpoint: **73 passed, 1 failed**. All OpenGL context failures resolved; remaining contrast test called a removed single-channel method. Updated the existing scenario to click Auto All and compare real layer limits with image percentiles, then exercise manual limits.

### A5 preparation boundary (coordinator)

Nuclear measurement now exposes private prepare/discard and atomic commit operations. The existing synchronous run_measure wraps them. Computation, CSV staging, and immutable store construction finish before the live manager is touched; commit rechecks the live source and retains the existing files/fields/store rollback. One workflow regression covers no publication during prepare, discard cleanup, a later edit preserved on rejection, and successful current-store publication.

Numerical, transaction, expression-window and real-layer contrast gate: **74 passed** in 14.39s. GUI background integration follows A4.

### C2 — live ROI plot provenance

App-derived profile windows now bind selected objects and measured profiles to their source snapshot. Scalar and profile windows refresh after edits; CSV/SVG/toolbar exports reject stale measurements and recover after remeasurement. Metadata-only edits retain valid image measurements. Existing standalone profile APIs remain usable for caller-owned data.

Agent gate: **26 passed**, including edit -> visible stale state in both windows -> all exports blocked -> remeasure -> exports restored. One prepared context per profile refresh/CSV/SVG and scalar SVG; external image/calibration changes rejected. Reviewed before integration; provider/dataset replacement follow-up assigned during C4.

### B4 — shared tracking settings

One component-settings builder applies registry defaults, source/preset settings, then advertised form controls across selected-cell, whole-movie and creation workflows. Shared helpers retain preset identity rebasing and calibration warnings. Native/exact source bindings and wizard localization defaults retain their existing semantics.

Agent gate: **118 passed** in 11.65s. Extended one existing workflow to transfer a tuned wizard request through both workbenches and compare full component specifications. Reviewed helper and caller changes before integration.

### Integrated ROI worker and settings checkpoint

ROI jobs compute with private providers and private result stores, then validate source identity/revision/calibration/manifest on the GUI thread before publishing. One job slot rejects duplicate starts and supports cancellation. Focused combined ROI lifecycle, profiles, scalar plots, cache, tracking GUI and discoverability gate: **54 passed** in 10.32s. Independent review found deleted-progress-dialog teardown after WA_DeleteOnClose; follow-up is required before A4 is complete.

C2 source-identity follow-up 104a559 rejects image-provider removal/replacement and replacement ROI managers even with identical fingerprints. Agent reused the same 26-test gate. Nuclear prepare/commit split received an independent atomicity/cleanup review with no concrete regression found.

### ROI memory comparison

Fresh-process tracemalloc with two synthetic objects (2D polygon and 3D contours), two channels, four 256x256 float32 planes; warmup excluded. All scalar means match their time/channel input values. Source e31b61f: **15.079 MiB** peak at six times, **60.287 MiB** at 24. Alpha: **1.325 MiB** and **1.517 MiB** respectively. Retained result/cache data remains proportional to sample count; decoded movie image retention is bounded.

### Differential functional lint

At 101589a, production F/E9 diagnostics: **32 baseline, 32 alpha, zero introduced**. Existing style-only and unused-code cleanup outside the accepted tasks was not expanded into a formatter sweep.

### A4/A5 — background measurement completed

Nuclear snapshots copy nuclei and movie/config settings plus flat CSV cell rows; naming/parent/daughter graphs remain on the GUI side. ROI/nuclear jobs share a single active slot. GUI completion validates manager/provider identities, source manifests and edit freshness before publication; expression channel changes only after success. Cancelled, edited, replaced-source and closed-window runs preserve prior results and discard staged files.

Agent gate: **65 passed**, including 11 worker lifecycle cases and existing expression transactions. Independent reproductions after fixes: deleted-progress window close had no exceptions and discarded once; application quit during a three-second read ran provider cleanup/discard, stopped the thread and exited with code 0. Ordinary cancellation/window close remains nonblocking; final process exit cooperatively drains its owned worker.

### Real napari workflow checkpoint

At 586a8b0 with Windows Qt/OpenGL and synthetic two-channel images, absolute planeStart=7: edit/undo/redo, real napari polygon drawing, review, filtering/overlay clearing, ROI async measure -> scalar CSV/SVG -> vertex edit/export rejection -> remeasure/export recovery, nuclear async measure/channel selection, clean ZIP Save As/ROI reopen/source preservation, 3D visibility restoration, detached 3D and expression windows all passed. The old layout after saving a long path still grew to 1280x1125; C4 addresses that before final fit acceptance. Harness and screenshots are workspace scratch artifacts.

### R3 — measurement save state

Successful nuclear publication marks the archive fields unsaved independently of undo history. A complete Save/Save As clears that state; cancellation, failed saves, failed retarget and intermediate copies preserve it. The workspace consumes the flag. Extended existing async/save workflows: **28 passed**; functional lint passed. Reviewed before integration.

### C4/R2 — compact workspace and matching documentation

Workflow now holds shared save/undo/target/status controls around Nuclei, Objects and Tracking tabs. Browse & Channels combines the lineage list and channel controls, Player Controls occupies two compact top rows, and independent lineage panels remain below the canvas. Menu launchers reveal the relevant tab. Objects retains persistent edit/measure/plot actions with scrollable review and metadata controls; long paths wrap.

Agent Windows gate: **127 passed**, final context-ordering/Qt-guard gate **10 passed**. Real napari stays **1280x720**, minimum hint **811x425**; a sidecar path over 300 characters cannot enlarge it. Reviewed actual screenshots and code before integration. User guide, README and architecture now describe the final workspace and background workflows. Final combined gate follows.
