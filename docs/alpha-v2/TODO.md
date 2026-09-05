# Alpha v2 task list

Each checkbox is an outcome, not a coverage target. Specialists work in isolated worktrees; coordinator integrates app glue and commits completed tasks to alpha-v2. At most three specialists run concurrently. Commit records are recorded here without self-referential hashes; the next update records prior commits.

| Task | Owner | Size | Depends on | Acceptance | Status / commit |
|---|---|---|---|---|---|
| A1 Correct nuclear Z and provenance | measurement | M | baseline | origins 1/7 raw/blot; historical cache compatibility | complete; 70661fe (agent), 219 focused tests |
| B1 Editing/lineage integrity | core | M | baseline | same-parent no-op; duplicate canonical roots; do/undo/redo links | complete; bf6d30a (agent), 346 focused tests |
| C1 ROI selection/scope/visibility | UI | M | baseline | hidden target cleared; class scope correct; dialog fits | complete; fb9c869 (agent), 24 focused tests |
| R1 Clean ROI Save As | coordinator | S | baseline | new ZIP reopens clean ROI; next save preserves source | complete; 23 save tests passed |
| A2 Channel identity | coordinator | S | baseline (independent of A1) | missing-first channel rejected without renumbering | complete; 214 focused tests |
| B2 Tracking request preservation | core | M | B1 | branch policies round-trip; nested inputs detached | complete; 3ee0990 (agent), 226 passed/2 skipped |
| A3 ROI bounded memory/freshness | measurement | M | A2 | images bounded per group; manifest once per plot/export | complete; c0700aa (agent), 37 focused tests |
| C2 ROI plot freshness | UI | M | C1,A3 contract | edit blocks all export; remeasure restores | complete; 21b01f1 (agent), 26 focused tests |
| B3 Bounded tracking undo | core | M | B2 | affected-only snapshots; rollback and nucleus identity | complete; ec8bdb5 (agent), 125 focused tests |
| C3 Object management/navigation | UI | M | C1 | class/span actions; true completeness; no Z list rebuild | complete; 2b9574b (agent), 33 focused tests |
| A4 Async ROI measurement | measurement | M | A1-A3 | responsive; cancellation/stale result safe; current snapshot | pending |
| A5 Async nuclear measurement | measurement | M | A4 | atomic files/fields/store; stale/cancel preserves previous | preparation complete 4b18613; worker integration pending |
| B4 Shared tracking settings | core | M | B3 | focused helpers, selected/global workflows unchanged | pending |
| C4 Workflow workspace | UI | M | C1,C3 | Nuclei/Objects/Tracking; 1280x720; all actions reachable | pending |
| R2 Alpha installation/docs | coordinator | S | integration | installers identify alpha-v2; docs match UI | installation complete; workflow docs pending |
| Final acceptance | coordinator | M | all above | full non-MATLAB suite; live napari; memory/navigation evidence | pending |

## Checklist

- [x] Audit and baseline recorded; isolated alpha-v2 worktree created.
- [x] A1
- [x] B1
- [x] C1
- [x] R1
- [x] A2
- [x] B2
- [x] A3
- [x] C2
- [x] B3
- [x] C3
- [ ] A4
- [ ] A5
- [ ] B4
- [ ] C4
- [ ] R2
- [ ] Final acceptance

## Interface decisions

- Nuclear reducers gain keyword-only plane_start=1; callers with zero-origin data can explicitly use 0. Dataset calls always pass config origin.
- Expression algorithm version becomes 2; existing .aceexpr schema remains. Older captures remain historical, never relabeled; mixed corrected/historical numerical comparisons require recompute/exclusion.
- ROI freshness validates external image dependencies once per plot/export operation and cheap geometry checks per sample.
- Measurement workers own private providers/results. One measurement job per app; prepare/stage off-thread and validate/commit on GUI thread. Existing synchronous run_measure remains supported.
- GUI filters clear hidden targets, apply to overlays, and preserve explicit visibility through 3D.
