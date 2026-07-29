# AceTree-Py

Python reimplementation of [AceTree](https://github.com/zhirongbaolab/AceTree) for *C. elegans* embryogenesis visualization and lineage annotation.

Built on [napari](https://napari.org) with full undo/redo, topology-based cell naming, interactive relink, 3D volume view, multi-channel display, rule-based visualization, manual tracking, and multi-panel lineage tree display.

## Installation

Requires **Python 3.10+** and Git. The tracking-enabled build currently lives
on the `tracking-integration` branch; the repository's default branch does not
yet contain these tools.

### Source install (recommended for testing)

```bash
# Select the tracking branch explicitly (a plain clone currently selects main).
git clone --branch tracking-integration --single-branch https://github.com/shahlab-ucla/acetree_py.git
cd acetree_py
```

The branch-aware installers verify the checkout before installing the
recommended GUI dependencies:

```powershell
# Windows PowerShell
.\scripts\install_tracking_integration.ps1
```

```bash
# macOS or Linux
sh scripts/install_tracking_integration.sh
```

Pass `-Variant core` / `--variant core` for the CLI-only install, or
`-Variant all` / `--variant all` to include development tools. The equivalent
manual commands are below. If the Python executable is not named `python`, pass
it explicitly (for example, `-Python py` or `--python /path/to/python3`).

```bash
# Core (CLI only, no GUI)
python -m pip install -e .

# With napari GUI (recommended)
python -m pip install -e ".[gui]"

# Everything (GUI + dev tools)
python -m pip install -e ".[all]"

# Confirm that this is the tracking-enabled build
python -m acetree_py --version
# AceTree-Py 0.2.0 (tracking integration)
```

To install without keeping a source checkout, use a branch-pinned VCS
requirement (not an unqualified PyPI install):

```bash
python -m pip install "acetree-py[gui] @ git+https://github.com/shahlab-ucla/acetree_py.git@tracking-integration"
```

### Tested versions

| Package    | Tested | Required            |
|------------|--------|---------------------|
| Python     | 3.12   | >= 3.10             |
| napari     | 0.6.6  | >= 0.5, < 0.7      |
| numpy      | 2.3    | >= 1.24             |
| scipy      | 1.16   | >= 1.10             |
| matplotlib | 3.10   | >= 3.7              |
| qtpy       | 2.4    | >= 2.3              |
| tifffile   | 2026.3 | >= 2023.1           |

**napari version note:** The GUI uses some napari-internal APIs for dock widget management. These are tested against napari 0.5.x–0.6.x. The upper bound (`<0.7`) guards against potential breaking changes.

## Usage

### GUI — open an existing dataset

```bash
acetree-py gui path/to/config.xml
```

### GUI — create a new dataset from raw images

```bash
# Interactive wizard:
acetree-py create

# From the command line with explicit parameters:
acetree-py create path/to/images/ --output path/to/output/ --xy-res 0.1625 --z-res 0.65 --split

# Prepare a reviewed Modern StarryNite draft from a bundled preset:
acetree-py create path/to/images/ --output path/to/output/ --tracking starrynite --starrynite-preset dispim_singleview

# Confirm that the installed build contains the tracking integration:
acetree-py --version
```

See [Tracking & Dataset Creation](docs/user_guide.md#14-tracking--dataset-creation)
in the User Guide for a full walkthrough.

### Real-world StarryNite tracking

AceTree-Py exposes four named workflows in the tracking dialogs:

- **Modern StarryNite (recommended)** is the practical whole-movie and sparse
  workflow, combining stage-aware detection with fast division-aware tracking.
- **LoG detection + LAP tracking** and **DoG detection + LAP tracking** are
  continuation-only alternatives that need no StarryNite assets.
- **Legacy StarryNite exact replay (advanced)** is the global MATLAB-compatibility
  backend, with sequential detection, staged geometry, source-bound
  classification, cleanup, and divisions in one reviewable draft.

All six parameter files from StarryNite's `example_parameter_files/newmatlab`
directory are bundled as **Imaging preset** choices. Their distributions, the
2019 model, and the additional Gaussian light-sheet model are source-bound and
ready to use; users do not generate JSON files. The pre-2019 red-channel model
source and a MATLAB export helper are included for advanced compatibility work.

Use **Track Selected Cell...** for sparse forward tracking in any new or loaded
XML dataset. Use **Track Whole Movie...** for an empty record, including the
initial draft offered by the dataset wizard. Both entry points, plus Manual
Track, are available from the top-level **Tracking** menu and the scrollable
**Edit & Tracking Tools** dock. **Advanced and custom settings**
can load, edit, and save an external legacy parameter file, and AceTree remembers
the most recently selected external file. Exact mode never silently falls back
to native scoring, and no nuclei change until **Accept Draft**.

See the [real-world StarryNite checklist](docs/user_guide.md#real-world-starrynite-test-checklist)
for the complete UI sequence, prerequisites, failure guidance, current
boundaries, and dated validation status. The opt-in MATLAB commands and parity
method are in [StarryNite Differential Testing](docs/STARRYNITE_DIFFERENTIAL_TESTING.md#running-locally).
The 2026-07-29 non-live release gate completed with `1347 passed, 71 skipped`.
The latest complete MATLAB-oracle gate, run with R2025a on 2026-07-16, completed
with `19 passed, 1 skipped`; the one expected skip is the historical four-model
export boundary.

### CLI

```bash
# Print dataset summary:
acetree-py load config.xml

# Export cell data:
acetree-py export config.xml --format cell_csv --output cells.csv

# Run naming pipeline and save:
acetree-py rename config.xml --output renamed.zip

# Query a specific cell:
acetree-py info config.xml --cell ABala
```

### Config file format

```xml
<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="path/to/nuclei.zip"/>
    <image file="path/to/image.tif"/>
    <end index="350"/>
    <naming method="NEWCANONICAL"/>
    <resolution xyRes="0.09" zRes="1.0" planeEnd="30"/>
</embryo>
```

## Features

- **Napari-based viewer** with nucleus overlay, z-plane navigation, cell tracking, and hover tooltips
- **Multi-channel display** — per-channel contrast sliders, visibility toggles, green/magenta colormaps for dual-channel data
- **Interleaved multichannel TIFFs** — single TIFF per timepoint with pages laid out as `Z1C1, Z1C2, Z2C1, Z2C2, …` (or planar `Z1C1..ZnC1, Z1C2..ZnC2`); supported by the XML config (`<image numChannels="N" channelOrder="CZ|ZC"/>`), the dataset-creation wizard, and the `acetree-py create --interleaved` CLI flag
- **Rule-based visualization** — color nuclei by lineage depth, expression level, cell fate, name pattern, or custom rules with a full GUI rule editor
- **3D volume view** — toggle 2D/3D in the main viewer, or open a detached 3D window with independent visualization controls
- **Ghost trails** — visualize selected cell's movement history as a semi-transparent trail
- **Manual tracking** — click-to-add nuclei, D-pad nudge controls, create datasets from raw images
- **StarryNite tracking** — use a fast native detector/division tracker or an explicitly selected, fail-closed whole-movie compatibility backend. The exact backend reads source-bound legacy parameter, distribution, tracking-model, and neutral-classifier files; reproduces sequential detection, staged geometry, class 0/1/2/3 cleanup, and divisions through the ordinary tracking proposal workflow; and records hashes plus stage/classifier validation provenance. Unsupported legacy options are reported before a run rather than approximated silently.
- **Topology-based naming** — automatic Sulston name assignment from lineage structure, with rotation-invariant axis estimation robust to embryo rotations during imaging
- **Interactive relink** — click-based predecessor editing with automatic interpolation
- **Cell-scoped rename and atomic swap** — the Rename command writes a forced name across the cell's entire continuation chain in one undoable step; name collisions can be resolved with an atomic swap between two cells
- **Full undo/redo** — up to 1000 edit commands with `Ctrl+Z` / `Ctrl+Y`
- **Multi-panel lineage trees** — open multiple Sulston tree views with independent root cells, time ranges, expression ranges, and colormaps
- **Pixel measurement (File → Measure…)** — port of the Java `AceBatch2` measure tool: samples fluorescence per nucleus in every image channel, writes one CSV per channel (cell × absolute time), updates `rwraw` / `rwcorr1` / `rwcorr3` / `rweight` for the chosen AT channel so the lineage tree re-colors from live measurement. Background-correction selector in the dialog: *None*, *Global* (annulus mean), or *Blot* (annulus with every nucleus's projected disk masked out — cleaner estimate in crowded regions).
- **Screenshot and recording** — capture single frames or export image sequences across timepoints
- **Save/Save As** — persist edits to ZIP files compatible with Java AceTree
- **Export** — cell tables, nucleus tables, expression time series, Newick trees

## Documentation

- [User Guide](docs/user_guide.md) — navigation, editing, saving, manual tracking, 3D view
- [Architecture Reference](docs/architecture.md) — package structure, data model, GUI system
- [Algorithm Reference](docs/algorithms.md) — naming pipeline, coordinate transforms, edit commands
- [StarryNite Rebuild Plan](docs/STARRYNITE_REBUILD_PLAN.md) — clean-room compatibility roadmap, native tracker architecture, and verification gates
- [StarryNite Differential Testing](docs/STARRYNITE_DIFFERENTIAL_TESTING.md) — deterministic simulations, local MATLAB oracle, parity metrics, parameter sweeps, and current results

## Development

```bash
# Install with dev tools:
pip install -e ".[all]"

# Run tests:
pytest tests/

# Lint:
ruff check acetree_py/
```

## License

MIT
