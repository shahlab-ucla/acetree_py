# Alpha v2 implementation audit

Base: subcellular-measurements, e31b61f5e439db1005444852cd8f64d84f948c72.
Date: 2026-09-05. Integration branch: alpha-v2.

The accepted scope is balanced correctness/performance refinement and a Nuclei / Objects / Tracking tabbed napari workspace. Preserve file compatibility, forced naming, ROI identity, measurement atomicity, and exact-tracking semantics.

## Confirmed findings

| ID | Priority | Evidence and outcome |
|---|---|---|
| A1 | P1 | analysis/measure.py compares absolute nucleus Z with stack indices. Raw/blot first-plane probe returns 0 instead of 100. Correct coordinates and version provenance. |
| A2 | P1 | io/image_provider.py compacts configured channels after skipping unavailable sources. Reject incomplete channel sets without renumbering. |
| R1 | P1 | gui/app.py saves clean ROIs only when dirty, losing/incorrectly targeting them after ZIP Save As. Persist changed destinations. |
| B1 | P1 | core/lineage.py reuses already populated canonical dummy cells; same-parent RelinkNucleus undo breaks successor links. Keep distinct cells and reciprocal topology. |
| B2 | P1 | tracking/persistence.py omits branch_policy; api mappings freeze only outer dictionaries. Preserve exact requests and nested values. |
| C1 | P1 | Hidden object selections remain deletable and Current class can measure the prior class. Unify selection/filter state. |
| C2 | P1 | ROI profile snapshots/export omit freshness checks; scalar windows do not update visible stale status after edits. Guard and refresh both. |
| C1/C4 | P2 | Offscreen minimum widths: ROI Measure 1414px, Objects 692px. Fit laptop screens and simplify workflows. |
| A3 | P2 | ROI runs retain decoded images for all times; per-sample plot freshness scans full image manifests and objects. Bound memory/work. |
| B3 | P2 | Tracking acceptance snapshots every nucleus; global analysis copies nuclei it does not use. Snapshot only affected state. |
| C3 | P2 | Object rows rescan all histories on Z changes; missing expected frames can still be Complete. Existing class/span commands lack UI. |
| A4/A5 | P2 | Measurement runs synchronously with processEvents. Prepare privately and validate/publish on GUI thread. |
| R3 | P2 | Successful nuclear measurement with unchanged correction has no unsaved indicator; the new workspace must keep this distinct from undo history until a successful save. |
| B4 | P3 | Tracking dialogs duplicate preset/settings and compatibility presentation; twelve methods/148 lines are AST-identical. Extract focused composed helpers. |

## Deferred findings and boundaries

- Naming/history callback ordering is a risk, but 42 actual integrated edit/undo/redo states had matching nucleus/tree names. No broad naming/history rewrite.
- MoveNucleus geometry influences biological naming; retain its current naming semantics.
- Preserve legacy MATLAB tests and exact implementation; no plugin migration, new scientific capability, broad formatter churn, or coverage quota.
- Existing source checkout has untracked reports/graph artifacts; these are not alpha changes.
