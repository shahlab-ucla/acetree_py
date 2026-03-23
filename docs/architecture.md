# AceTree-Py Architecture Reference

**Version 0.1.0** | Python reimplementation of AceTree for *C. elegans* embryogenesis

---

## 1. Project Overview

AceTree-Py is a from-scratch Python rewrite of the Java AceTree application, which visualizes and annotates *C. elegans* embryonic cell lineage data. The rewrite targets modern tooling (napari, NumPy, SciPy) and adds full undo/redo support, topology-based cell naming, and a clean modular architecture.

### Package Structure

```
acetree_py/                    # Root package (__version__ = "0.1.0")
  __main__.py                  # CLI entry point (typer)
  core/                        # Data model — no GUI dependencies
    nucleus.py                 # Nucleus dataclass (central record)
    cell.py                    # Cell dataclass, CellFate enum
    lineage.py                 # LineageTree, build_lineage_tree()
    movie.py                   # Movie dataclass (dimensions/timing)
    nuclei_manager.py          # NucleiManager (central orchestrator)
  naming/                      # Cell identity assignment — no GUI deps
    identity.py                # IdentityAssigner (pipeline orchestrator)
    founder_id.py              # Topology-based founder identification
    initial_id.py              # Legacy diamond-pattern identification
    division_caller.py         # Division vector analysis + daughter naming
    canonical_transform.py     # Rotation to canonical frame (Wahba solver)
    rules.py                   # Rule, RuleManager (naming rules)
    sulston_names.py           # Sulston conventions + letter maps
    validation.py              # Post-naming validation
  editing/                     # Command-pattern edit system — no GUI deps
    commands.py                # EditCommand ABC + 8 concrete commands
    history.py                 # EditHistory (undo/redo stacks)
    validators.py              # Pre-edit validation functions
  io/                          # File I/O — no GUI dependencies
    config.py                  # AceTreeConfig, load_config()
    nuclei_reader.py           # read_nuclei_zip()
    nuclei_writer.py           # write_nuclei_zip()
    image_provider.py          # ImageProvider protocol + 7 providers
    auxinfo.py                 # AuxInfo (embryo orientation data)
  gui/                         # napari GUI — all Qt/napari deps isolated here
    app.py                     # AceTreeApp (main application)
    viewer_integration.py      # ViewerIntegration (nucleus overlay)
    lineage_widget.py          # LineageWidget (Sulston tree)
    lineage_layout.py          # Layout engine (pure computation)
    lineage_list.py            # LineageListWidget (hierarchical list)
    player_controls.py         # PlayerControls (time/plane navigation)
    cell_info_panel.py         # CellInfoPanel (info display)
    contrast_tools.py          # ContrastTools (brightness/contrast)
    edit_panel.py              # EditPanel + 7 dialog classes
  analysis/                    # Post-hoc analysis — no GUI dependencies
    expression.py              # Expression time series analysis
    export.py                  # CSV, Newick export functions
  utils/
    geometry.py                # 3D vector math helpers
  resources/
    new_rules.tsv              # ~620 pre-computed division rules
    names_hash.csv             # ~60 Sulston letter mappings
```

### Dependency Architecture

```
gui/  ──depends-on──►  core/  ◄──depends-on──  naming/
  │                      │                        │
  │                      ▼                        │
  ├──depends-on──►  editing/                      │
  │                      │                        │
  └──depends-on──►    io/   ◄─────────────────────┘
```

- **`core/`**, **`naming/`**, **`editing/`**, **`io/`** have zero GUI imports.
- **`gui/`** depends on all other packages, plus napari and qtpy.
- This isolation means headless (CLI) operation works without Qt/napari.

---

## 2. Core Data Model

### 2.1 Nucleus (`core/nucleus.py`)

The fundamental record. Represents one detected nucleus at one timepoint.

| Field          | Type         | Description                                       |
|----------------|-------------|---------------------------------------------------|
| `index`        | `int`        | 1-based index within its timepoint                |
| `x`, `y`       | `int`        | Pixel coordinates                                 |
| `z`            | `float`      | Z-plane (float for sub-plane precision)           |
| `size`         | `int`        | Nucleus diameter in pixels                        |
| `identity`     | `str`        | Auto-assigned Sulston name (e.g., `"ABala"`)      |
| `assigned_id`  | `str`        | Manually forced name (survives re-naming)         |
| `status`       | `int`        | ≥1 = alive, -1 = dead/invalid                    |
| `predecessor`  | `int`        | 1-based index into previous timepoint (NILLI = -1)|
| `successor1`   | `int`        | 1-based index into next timepoint                 |
| `successor2`   | `int`        | Second successor (if dividing)                    |
| `weight`       | `int`        | GFP expression intensity                          |
| `rweight`      | `int`        | Computed red channel weight                       |
| `rsum`–`rwcorr4` | `int`     | Raw and corrected red channel values              |
| `hash_key`     | `str\|None`  | Tree lookup key: `str(time * 100000 + index)`     |

**Key properties:**
- `is_alive` → `status >= 1`
- `is_dividing` → `successor2 != NILLI`
- `effective_name` → `assigned_id if assigned_id else identity`

**Serialization:** CSV lines in ZIP entries, with both old-format (Java legacy) and new-format parsers.

### 2.2 Cell (`core/cell.py`)

A lineage tree node spanning a cell's lifetime (birth to division/death).

| Field        | Type             | Description                                |
|-------------|------------------|--------------------------------------------|
| `name`       | `str`            | Sulston name or auto-generated             |
| `start_time` | `int`            | First timepoint (1-based)                  |
| `end_time`   | `int`            | Last timepoint (1-based)                   |
| `end_fate`   | `CellFate`       | `ALIVE`, `DIVIDED`, or `DIED`              |
| `parent`     | `Cell\|None`     | Parent cell                                |
| `children`   | `list[Cell]`     | 0 (leaf) or 2 (divided) daughters          |
| `nuclei`     | `list[tuple]`    | `(timepoint, Nucleus)` pairs               |
| `hash_key`   | `str\|None`      | Matches Nucleus hash for lookup            |

**Traversal methods:** `iter_ancestors()`, `iter_descendants()`, `iter_subtree_preorder()`, `iter_leaves()`, `depth()`

### 2.3 LineageTree (`core/lineage.py`)

| Field            | Type                   | Description                    |
|-----------------|------------------------|--------------------------------|
| `root`           | `Cell\|None`           | Root cell (P0 or first)        |
| `cells_by_name`  | `dict[str, Cell]`      | Name → Cell lookup             |
| `cells_by_hash`  | `dict[str, Cell]`      | Hash key → Cell lookup         |
| `cell_counts`    | `list[int]`            | Alive cells per timepoint      |

**`build_lineage_tree(nuclei_record, starting_index, ending_index, create_dummy_ancestors)`:**

1. Optionally creates dummy ancestors for standard Sulston names (P0, AB, P1, ABa, ABp, EMS, P2, MS, E, C, P3, D, P4).
2. Iterates timepoints. For each nucleus:
   - If it has a predecessor, links to the parent cell or merges into a dummy ancestor.
   - If no predecessor, creates a new root cell.
3. Marks cells as `DIVIDED` when they produce two successors.
4. Builds name and hash lookups.
5. Adjusts dummy ancestor timing to match real data.

**Hash key formula:** `hash_key = str(time_1based * 100000 + nuc_index_1based)`

### 2.4 Movie (`core/movie.py`)

Temporal and spatial bounds. Key property: `z_pix_res = z_res / xy_res` (anisotropy ratio).

### 2.5 NucleiManager (`core/nuclei_manager.py`)

Central orchestrator that owns `nuclei_record: list[list[Nucleus]]` (indexed `[timepoint_0based][nucleus_index]`).

**Processing pipeline (`process()`):**
1. `set_all_successors()` — compute forward links from predecessor fields (only alive nuclei — dead nuclei with stale predecessor links are excluded to prevent false division signals)
2. `compute_red_weights()` — apply expression corrections
3. `_run_naming()` → `IdentityAssigner.assign_identities()`
4. `_build_tree()` → `build_lineage_tree()`

**Nucleus search:**
- `find_closest_nucleus(x, y, z, time)` — 3D Euclidean nearest, z scaled by `z_pix_res`
- `find_closest_nucleus_2d(x, y, time)` — ignores z
- `nucleus_diameter(nuc, image_plane)` — projected circle diameter at a given z-plane

---

## 3. I/O System

### 3.1 Config (`io/config.py`)

`AceTreeConfig` holds ~20 fields parsed from XML:

```xml
<embryo>
    <nuclei file="path/to/nuclei.zip"/>
    <image file="path/to/image.tif"/>
    <end index="350"/>
    <naming method="NEWCANONICAL"/>
    <axis axis="adl"/>
    <resolution xyRes="0.09" zRes="1.0" planeEnd="30"/>
    <exprCorr type="blot"/>
</embryo>
```

`NamingMethod` enum: `STANDARD=2`, `MANUAL=2` (skip naming), `NEWCANONICAL=3`.

### 3.2 Nuclei Reader/Writer

**ZIP structure:**
```
nuclei/
    t001-nuclei    # CSV: one Nucleus per line
    t002-nuclei
    ...
```

- `read_nuclei_zip(path)` → `list[list[Nucleus]]`
- `write_nuclei_zip(nuclei_record, path, start_time=1)` — writes new-format CSV

### 3.3 Image Providers (`io/image_provider.py`)

`ImageProvider` protocol:
- `get_plane(time, plane, channel=0) -> np.ndarray`
- `get_stack(time, channel=0) -> np.ndarray`
- Properties: `num_timepoints`, `num_planes`, `num_channels`, `image_shape`

Seven concrete implementations:

| Provider                    | Source                              |
|----------------------------|-------------------------------------|
| `ZipTiffProvider`           | TIFF images in ZIP files (Java fmt) |
| `TiffDirectoryProvider`     | Loose TIFFs with pattern naming     |
| `StackTiffProvider`         | Multi-page TIFF stacks              |
| `OmeTiffProvider`           | OME-TIFF with metadata              |
| `SplitChannelProvider`      | 16-bit TIFFs split into 2 channels  |
| `MultiChannelFolderProvider`| Separate folders per channel        |
| `NumpyProvider`             | In-memory NumPy arrays (testing)    |

### 3.4 AuxInfo (`io/auxinfo.py`)

Embryo orientation metadata.

- **v1** (`_AuxInfo.csv`): 3-char axis string (e.g., `"ADL"`) + rotation angle
- **v2** (`_AuxInfo_v2.csv`): AP and LR orientation vectors (3D)

---

## 4. Naming System

### 4.1 Pipeline (`naming/identity.py`)

`IdentityAssigner.assign_identities()`:

1. Clear non-forced names.
2. Build `CanonicalTransform` (if v2 AuxInfo available).
3. **Try topology-based identification** (`identify_founders()`).
4. If topology fails, **fall back to legacy** diamond-pattern identification.
5. Set up `DivisionCaller` using derived or provided axes.
6. **Forward pass**: apply canonical rules from 4-cell stage onward.
7. Assign generic `Nuc_t_z_x_y` names to remaining unnamed cells.

### 4.2 Founder ID (`naming/founder_id.py`)

Topology-based identification of ABa, ABp, EMS, P2 at the 4-cell stage:

1. Find 4-cell windows (exactly 4 alive nuclei).
2. **Sister pair identification** — three strategies:
   - Backward trace: cells sharing a parent are sisters.
   - Birth time grouping: cells born at the same time are sisters.
   - **Forward division pairing**: cells that next divide at similar times are sisters (for datasets starting at the 4-cell stage with no predecessor data).
3. **AB vs P1 pair**: the pair that divides first are AB daughters; the pair that divides second are P1 daughters. This is a biological invariant of *C. elegans*.
4. **Within-pair assignment**: EMS is larger than P2 (size ratio); ABa/ABp determined by position relative to LR axis.
5. **Back-trace**: trace predecessors to name AB, P1, P0 and their continuation cells.
6. **Axis derivation**: compute AP, DV, LR vectors from the 4 cell positions.

### 4.3 Division Caller (`naming/division_caller.py`)

Classifies each cell division to determine daughter names:

1. Look up the division `Rule` for the parent name.
2. Compute the division vector (daughter2 − daughter1), z-scaled by `z_pix_res`.
3. Rotate the vector into the canonical frame.
4. Dot product with the rule's axis vector determines which daughter gets which name.
5. Angle between division vector and rule axis maps to a confidence score.

Three coordinate transform modes:
- **v2**: Full `CanonicalTransform` rotation (Wahba's problem solver).
- **v1**: Sign-flip matrix + 2D rotation by angle.
- **Founder**: Project onto founder-derived AP/DV/LR axes.

### 4.4 Rules (`naming/rules.py`)

`RuleManager` lookup priority:
1. Pre-computed rules from `resources/new_rules.tsv` (~620 empirical rules).
2. Generated rules from `resources/names_hash.csv` Sulston letter mappings.
3. Default: use `"a"` (AP axis) as the division axis.

Each `Rule` contains: `parent`, `sulston_letter`, `daughter1`, `daughter2`, `axis_vector` (unit 3-vector).

### 4.5 Sulston Names (`naming/sulston_names.py`)

- Letter complements: a↔p, d↔v, l↔r
- Letter-to-axis mapping: a/p→AP, d/v→DV, l/r→LR
- Founder cells: P0, AB, P1, EMS, P2, E, MS, C, P3, D, P4, Z2, Z3
- `daughter_names("ABa", "l")` → `("ABal", "ABar")`

---

## 5. Editing System

### 5.1 Command Pattern (`editing/commands.py`)

Abstract base: `EditCommand` with `execute()`, `undo()`, `description`.

| Command                    | Operation                                 | State Captured                        |
|---------------------------|-------------------------------------------|---------------------------------------|
| `AddNucleus`               | Create nucleus at position               | Added index                           |
| `RemoveNucleus`            | Kill nucleus (status=-1)                  | Old status, identity, assigned_id     |
| `MoveNucleus`              | Change position/size                      | Old x, y, z, size                     |
| `RenameCell`               | Set identity + assigned_id                | Old identity, assigned_id             |
| `RelinkNucleus`            | Change predecessor link                   | Old/new pred, both parents' successors|
| `KillCell`                 | Kill all nuclei of a named cell           | List of (time, idx, old state) tuples |
| `ResurrectCell`            | Restore dead nucleus                      | Old status, identity, assigned_id     |
| `RelinkWithInterpolation`  | Link with interpolated intermediates      | Added nuclei list, old/new links      |

### 5.2 Undo/Redo (`editing/history.py`)

`EditHistory` maintains two stacks:
- `_undo_stack`: commands that have been executed
- `_redo_stack`: commands that have been undone

Flow: `do(cmd)` → execute + push undo + clear redo. `undo()` → pop undo + reverse + push redo. `redo()` → pop redo + re-execute + push undo. New edits always clear the redo stack. Max 1000 commands (configurable).

### 5.3 Validators (`editing/validators.py`)

Pre-edit validation returns `list[str]` error messages (empty = valid). Checks index ranges, alive status, successor capacity (max 2 children), and time ordering.

---

## 6. GUI System

### 6.1 AceTreeApp (`gui/app.py`)

Main coordinator. Owns the napari `Viewer`, `NucleiManager`, `EditHistory`, and all dock widgets.

**Widget layout:**
```
┌──────────────────────────────────────────────┐
│  napari Viewer (image + nucleus overlay)      │
├──────────┬───────────────────┬───────────────┤
│ Cell Info│                   │ Contrast      │
│ Panel    │                   │ Tools         │
│          │                   ├───────────────┤
│ Lineage  │                   │ Edit          │
│ List     │                   │ Panel         │
├──────────┴───────────────────┴───────────────┤
│ Player Controls (time/plane navigation)       │
│ Lineage Tree (Sulston tree visualization)     │
└──────────────────────────────────────────────┘
```

**Keyboard shortcuts:**
| Key            | Action                   |
|----------------|--------------------------|
| `Right`/`Left` | Next/previous timepoint  |
| `Up`/`Down`    | Next/previous z-plane    |
| `Ctrl+S`       | Save                     |
| `Ctrl+Shift+S` | Save As                  |
| `Ctrl+Z`       | Undo                     |
| `Ctrl+Y`       | Redo                     |

### 6.2 ViewerIntegration (`gui/viewer_integration.py`)

Draws nucleus circles as a napari Shapes layer (polygon approximation with 32 vertices for aspect-ratio-independent circles).

**Mouse interaction:**
- **Left-click** on nucleus: toggle label visibility on/off
- **Right-click** on nucleus: select cell (make active)
- In **relink pick mode**: right-click selects the relink target

**Division line overlay:** When the selected cell has just divided (current_time == cell.end_time + 1), a yellow line connects the two daughter cell positions. Disappears on any navigation or selection change.

### 6.3 LineageWidget (`gui/lineage_widget.py`)

Sulston tree rendered in a `QGraphicsView` with expression-colored branch segments.

Uses `_ClickableGraphicsView` subclass to handle single-click despite `ScrollHandDrag` mode (detects clicks on mouseRelease when mouse movement < 5px).

**Mouse interaction:**
- **Left-click** on cell: select cell, jump to clicked y-position timepoint
- **Right-click** on cell: select cell, jump to cell's end time
- **Mouse wheel**: zoom in/out

### 6.4 LineageLayout (`gui/lineage_layout.py`)

Pure computational layout engine (no Qt dependency):
- `compute_layout(root_cell, params)` → `dict[str, LayoutNode]`
- Recursive leaf-counting for x-position assignment
- Y-axis = time (start_time → end_time scaled by `y_scale`)
- Expression coloring via `expression_to_color()` (heatmap)
- Daughter ordering follows Java AncesTree convention

### 6.5 Other Widgets

| Widget             | Purpose                                      |
|--------------------|----------------------------------------------|
| `LineageListWidget` | Hierarchical QTreeWidget with search/filter  |
| `PlayerControls`    | Time/plane navigation, play/pause animation  |
| `CellInfoPanel`     | Read-only cell details display               |
| `ContrastTools`     | Min/max contrast sliders with auto-contrast  |
| `EditPanel`         | Edit buttons, interactive relink, undo/redo  |

### 6.6 Interactive Relink

Replaces index-based dialogs with a unified pick-mode workflow (single "Relink" button):
1. Select either cell in the pair you want to link (order doesn't matter).
2. Click **Relink** → enters pick mode (status shows instructions).
3. Navigate in the viewer to the other cell.
4. **Right-click** the target → system sorts the two cells by time (earlier = predecessor, later = child).
5. Gap = 1 frame: simple relink. Gap > 1: automatic interpolation with linearly interpolated intermediates.
6. Confirmation dialog → execute command.

---

## 7. Analysis Module

### 7.1 Expression Analysis (`analysis/expression.py`)

- `ExpressionTimeSeries`: per-cell expression values with `mean`, `max_value`, `onset_time`
- `SubtreeStats`: expression aggregated across subtrees
- `SisterComparison`: sister cell expression ratios

### 7.2 Export (`analysis/export.py`)

| Function                     | Output                          |
|------------------------------|--------------------------------|
| `export_cell_table_csv()`     | Cell-level CSV (name, time, fate, parent, children) |
| `export_nucleus_table_csv()`  | Nucleus-level CSV (all fields)  |
| `export_expression_csv()`     | Expression time series CSV      |
| `export_newick()`             | Newick tree format              |

---

## 8. CLI

Entry point: `acetree_py/__main__.py` (typer app)

```
acetree-py load <config.xml>                    # Print dataset summary
acetree-py gui <config.xml>                     # Launch napari GUI
acetree-py export <config.xml> -f cell_csv      # Export data
acetree-py rename <config.xml> -o renamed.zip   # Run naming + save
acetree-py info <config.xml> -c ABala           # Query cell details
```

---

## 9. Testing

25 test files in `tests/` covering all modules: data model, I/O, naming, editing, GUI, CLI, analysis, and integration tests. Run with `pytest tests/`.

---

## 10. Build & Installation

```toml
# pyproject.toml
[project]
name = "acetree-py"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "scipy>=1.10", "tifffile>=2023.1", "typer>=0.9"]

[project.optional-dependencies]
gui = ["napari[all]>=0.4.18", "qtpy>=2.0"]
dev = ["pytest>=7.0", "pytest-qt>=4.2", "ruff>=0.1"]

[project.scripts]
acetree-py = "acetree_py.__main__:app"
```

Install: `pip install -e .` (core) or `pip install -e ".[gui]"` (with GUI) or `pip install -e ".[all]"` (everything).
