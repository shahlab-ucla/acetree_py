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
