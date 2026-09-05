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

### R1 — clean ROI Save As

Save As now copies clean ROI state when the destination changes and retargets only after the coordinated commit. Added one clean load/copy/reopen/edit/save scenario that checks the original sidecar remains unchanged.

`pytest tests/test_app_save.py tests/test_roi_save_transaction.py tests/test_roi_end_to_end.py -q`: **23 passed** in 1.41s.

### R2 — alpha installation

Branch guards and CLI identify alpha v2. Historical plugin proposal is explicitly deferred. Replaced brittle prose/source-string checks with existing executable installer cases and one CLI version check. Remote install instructions explicitly require branch publication.

`pytest tests/test_tracking_branch_installation.py tests/test_cli.py -q`: **18 passed** in 6.52s. Workflow documentation follows UI integration.

### B1 — editing and lineage integrity

Distinct canonical roots remain explicit conflicts; same-parent relinks are no-ops; Add/Relink/Interpolation validate before mutation. Interpolation now composes existing commands instead of duplicating mutation/undo logic.

Agent gate: editing, lineage, nuclei_manager, edit_panel, post_commit_ui, identity, tracking_integration: **346 passed** in 5.33s. Reviewed diff before integration.

### C1 — ROI selection, scope, visibility

Filtered-out objects no longer remain mutation targets; class measurement uses visible class; missing-cell filter has an empty state; image overlays share filter IDs and retain explicit visibility across 3D. Invalid measurement requests stay in a wrapped, scrollable dialog.

Agent gate: ROI UI models, viewer integration, and Objects panel: **24 passed**. ROI Measure now opens at **560 x 650**, down from minimum width 1414. Reviewed diff before integration.

### GUI environment

Installed tested napari 0.6.6 in isolated workspace .alpha-v2-venv. Hidden Windows-backend real-OpenGL canvas probe succeeded (nonuniform synthetic image, RGB standard deviation 82.3). Offscreen plugin alone cannot render OpenGL.

### A2 — stable image channel identity

Explicit channel sets reject missing paths, gaps, unreadable samples, or unparseable time patterns instead of renumbering survivors. The app can still open nuclei and reports the source error. One missing-first-channel/recovery scenario checks both scientific identities.

`pytest tests/test_image_split.py tests/test_image_provider.py tests/test_interleaved_tiff.py tests/test_gui_app.py tests/test_cli.py -q`: **214 passed**, 14 existing TIFF warnings in 6.13s.
