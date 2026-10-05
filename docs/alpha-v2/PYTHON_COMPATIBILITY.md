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
| Dependency markers, installer preflight, text serialization | GPT-6.1 Sol implementer | Older Python preserved; modern GUI path; clear unsupported-platform error | In progress |
| Cross-platform CI and reusable validation | GPT-6.1 Sol CI implementer | Fresh wheel/core/GUI environments; required GUI tests execute | In progress |
| Independent implementation review | GPT-6.1 Sol reviewer | Diff and validation review; concrete concerns resolved | Pending |
| Local runtime validation and documentation | Coordinator | Fresh 3.10/3.11/3.14 evidence and support instructions | In progress |
| Commit, push, remote verification | Coordinator | Focused commits on alpha-v2; required CI results inspected | Pending |

## Evidence

Planning review was static, not proof of runtime compatibility. Validation results will be recorded here as checks finish. The initial GitHub tip matches the local starting commit, and GitHub Actions is enabled.

- Baseline reproduction: a fresh Python 3.14.8 environment using the old napari <0.7 GUI range selected napari 0.6.6 and attempted to build triangle 20200424; installation failed because no compatible binary was selected and Microsoft C++ build tools were required.
- Implementation gate: 33 focused installer/overlay tests passed on the existing Python 3.12 environment. The source-text dependency assertion was subsequently removed in favor of actual fresh wheel resolution in the compatibility matrix.
- Independent GPT-6.1 Sol review found no actionable issues in the initial changes or validator; fresh runtime and CI evidence remains required.
