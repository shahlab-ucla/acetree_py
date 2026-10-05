# Python compatibility work

Branch: alpha-v2. Starting commit: b29829e.

## Support policy

- Retain Python 3.10 and 3.11 support; validate standard-GIL CPython through 3.14.
- Python 3.14 GUI: Windows, Linux, and Apple Silicon macOS.
- Intel macOS GUI: Python 3.10-3.13; Python 3.14 GUI is intentionally unsupported because of the optional Numba dependency chain.
- Preserve scientific results, file formats, package version, and existing worker architecture unless a reproduced compatibility failure requires a change.
- Free-threaded Python and MATLAB oracle execution are outside this pass.

## Assignments and acceptance

| Work | Owner | Acceptance | Status |
|---|---|---|---|
| Language/stdlib review | GPT-6 Luna panel member 1 | Evidence-backed Python changes review | Complete; no confirmed application blocker |
| Dependency/install review | GPT-6 Luna panel member 2 | Python/platform resolution audit | Complete; napari <0.7 excludes documented 3.14 support |
| GUI/scientific/worker review | GPT-6 Luna panel member 3 | Runtime risks and existing regression inventory | Complete; no speculative worker changes |
| Dependency markers, installer preflight, text serialization | GPT-6.1 Sol implementer | Older Python preserved; modern GUI path; clear unsupported-platform error | Complete; 587ecb2 and ab5cabd |
| Cross-platform CI and reusable validation | GPT-6.1 Sol CI implementer | Fresh wheel/core/GUI environments; required GUI tests execute | In progress |
| Independent implementation review | GPT-6.1 Sol reviewer | Diff and validation review; concrete concerns resolved | Initial fixes and Windows graphics follow-up reviewed; final CI pending |
| Local runtime validation and documentation | Coordinator | Fresh 3.10/3.11/3.14 evidence and support instructions | In progress |
| Commit, push, remote verification | Coordinator | Focused commits on alpha-v2; required CI results inspected | Pending |

## Evidence

Planning review was static, not proof of runtime compatibility. Validation results will be recorded here as checks finish. The initial GitHub tip matches the local starting commit, and GitHub Actions is enabled.

- Baseline reproduction: a fresh Python 3.14.8 environment using the old napari <0.7 GUI range selected napari 0.6.6 and attempted to build triangle 20200424; installation failed because no compatible binary was selected and Microsoft C++ build tools were required.
- Implementation gate: 33 focused installer/overlay tests passed on the existing Python 3.12 environment. The source-text dependency assertion was subsequently removed in favor of actual fresh wheel resolution in the compatibility matrix.
- Independent GPT-6.1 Sol review found no actionable issues in the initial changes or validator; fresh runtime and CI evidence remains required.

## First runtime results

- Fresh Windows Python 3.14.8: wheel/core/GUI installation, pip checks, CLI, native OpenGL, and all 13 required GUI workflows passed. Full suite: **1936 passed, 8 skipped, 19 deselected** in 216.34 seconds. napari 0.7.1 / PyQt6 6.11.0 (Qt runtime 6.11.2), NumPy 2.5.3, SciPy 1.18.1. No inherited workspace dependencies.
- Fresh Windows Python 3.11.17 and 3.10.22 also passed wheel/core/GUI installation, pip checks, CLI, native OpenGL, and all 13 required GUI workflows; full results follow.
- First remote matrix run: https://github.com/shahlab-ucla/acetree_py/actions/runs/37351723882 . Apple Silicon Python 3.11 passed. Failures exposed missing XCB libraries on Linux, unavailable hosted Windows OpenGL, Intel Mac optional dependencies requiring unsupported source builds, and a macOS scroll-visibility check requiring geometry diagnostics. These failures are not treated as successful validation.
- Intel Mac Python 3.10-3.13 now keeps napari below 0.7 but selects its optional extras without triangle and restricts Numba to 0.62.x, the last series with Intel Mac wheels. Other platforms retain their original extras. The macOS GUI test retains strict full-widget visibility and adds bounded layout settling and screenshot/geometry diagnostics.

- Fresh Windows Python 3.11.17 full suite: **1936 passed, 8 skipped, 19 deselected** in 205.21 seconds; Python 3.10.22: **1936 passed, 8 skipped, 19 deselected** in 154.11 seconds. Both driver processes exited successfully, as did Python 3.14.8.
- Independent Sol verification of hosted-Windows graphics strategy: nonblank software-rendered screenshots on Python 3.10/3.11/3.14, plus all 13 required Python 3.14 GUI workflows passing with no skips. Qt, VisPy, and PyOpenGL share Qt's bundled Mesa DLL within the temporary validation environment; this does not alter application graphics defaults.
- Linux CI now installs the missing XCB runtime libraries and checks Qt plugin native dependencies before GUI execution. The second matrix run is https://github.com/shahlab-ucla/acetree_py/actions/runs/37353284256 . Documentation-only commits do not rerun the matrix.
- The second matrix run passed both alternate PySide6 rows and Intel Mac Python 3.10/3.11. Other Linux/macOS rows exposed a real Objects filter-row minimum-width overflow under native font metrics. The filter controls now use two rows, with a compact-dock regression that retains strict full-action visibility.
- All Windows default rows passed required GUI workflows. Full-suite failures were test assumptions: ANSI styling split CLI option text and fast recomputations could share a wall-clock tick. Help assertions now inspect unstyled text; force-recompute checks fresh provider reads and atomic replacement instead of timestamp uniqueness.
