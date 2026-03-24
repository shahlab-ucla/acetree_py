# AceTree-Py

Python reimplementation of [AceTree](https://github.com/zhirongbaolab/AceTree) for *C. elegans* embryogenesis visualization and lineage annotation.

Built on [napari](https://napari.org) with full undo/redo, topology-based cell naming, interactive relink, and multi-panel lineage tree display.

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

### GUI

```bash
acetree-py gui path/to/config.xml
```

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

- **Napari-based viewer** with nucleus overlay, z-plane navigation, and cell tracking
- **Topology-based naming** — automatic Sulston name assignment from lineage structure
- **Interactive relink** — click-based predecessor editing with automatic interpolation
- **Full undo/redo** — up to 1000 edit commands with `Ctrl+Z` / `Ctrl+Y`
- **Multi-panel lineage trees** — open multiple Sulston tree views with independent root cells, time ranges, expression ranges, and colormaps
- **Save/Save As** — persist edits to ZIP files compatible with Java AceTree
- **Export** — cell tables, nucleus tables, expression time series, Newick trees

## Documentation

- [User Guide](docs/user_guide.md) — navigation, editing, saving, lineage tree panels
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
