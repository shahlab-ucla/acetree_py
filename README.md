# AceTree-Py

Python reimplementation of [AceTree](https://github.com/zhirongbaolab/AceTree) for *C. elegans* embryogenesis visualization and lineage annotation.

Built on [napari](https://napari.org) with full undo/redo, topology-based cell naming, interactive relink, 3D volume view, multi-channel display, rule-based visualization, manual tracking, and multi-panel lineage tree display.

## Installation

Requires **Python 3.10+**.

```bash
# Clone the repository
git clone https://github.com/shahlab-ucla/acetree_py.git
cd acetree_py

# Core (CLI only, no GUI):
pip install -e .

# With napari GUI (recommended):
pip install -e ".[gui]"

# Everything (GUI + dev tools):
pip install -e ".[all]"
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
```

See [Manual Tracking & Dataset Creation](docs/user_guide.md#14-manual-tracking--dataset-creation) in the User Guide for a full walkthrough.

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
