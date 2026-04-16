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
    lineage_axes.py            # Per-timepoint body axis estimation from lineage centroids + LR quality metric
    validation.py              # Post-naming validation
  editing/                     # Command-pattern edit system — no GUI deps
    commands.py                # EditCommand ABC + 10 concrete commands
    history.py                 # EditHistory (undo/redo stacks)
    validators.py              # Pre-edit validation functions
  io/                          # File I/O — no GUI dependencies
    config.py                  # AceTreeConfig, load_config()
    config_writer.py           # write_config_xml() (round-trip XML serialization)
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
    player_controls.py         # PlayerControls (time/plane/labels/3D)
    cell_info_panel.py         # CellInfoPanel (hover tooltip builder)
    contrast_tools.py          # ContrastTools (per-channel contrast)
    color_rules.py             # ColorRuleEngine, ColorRule, presets
    edit_panel.py              # EditPanel + dialog classes
    viewer_3d_window.py        # Viewer3DWindow (detached 3D viewer)
    dataset_dialog.py          # DatasetCreationDialog (4-page wizard)
    measure_dialog.py          # MeasureDialog (channel + output picker)
  analysis/                    # Post-hoc analysis — no GUI dependencies
    expression.py              # Expression time series analysis
    export.py                  # CSV, Newick export functions
    measure.py                 # Per-nucleus pixel sampling (port of ExtractRed)
    measure_csv.py             # Measure CSV writer (per-channel, absolute time)
    measure_runner.py          # Measure orchestrator (iterates channels/timepoints)
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

**Construction:**
- `NucleiManager.from_config(config)` — load nuclei from ZIP file
- `NucleiManager.new_empty(config, num_timepoints)` — create empty manager for manual annotation (no nuclei loaded, all timepoints initialized to empty lists)

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

`AceTreeConfig` holds ~20 fields parsed from XML. Round-trip serialization is supported via `write_config_xml()` in `io/config_writer.py`, which produces XML using the exact same element and attribute names as the parser (case-sensitive: `SplitMode`, `FlipMode`, `xyRes`, `zRes`, `planeEnd`, `numChannels`, `channelOrder`).

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

The `<image>` element supports three shapes:

- `<image file="..."/>` — single-channel multi-page TIFF (or per-plane TIFFs if the filename contains `-p`).
- `<image numChannels="N" channel1="..." channel2="..."/>` — one directory per channel; routed to `MultiChannelFolderProvider`.
- `<image file="..." numChannels="N" channelOrder="CZ|ZC"/>` — single TIFF per timepoint whose pages are interleaved multichannel. Parsed into `config.stack_interleaved=True` and `config.num_channels=N`; routed to `StackTiffProvider` with native de-interleaving (see §3.3). `channelOrder` accepts aliases (`interleaved` → `CZ`, `planar` → `ZC`); unknown values log a warning and fall back to `CZ`.

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

| Provider                    | Source                                                        |
|-----------------------------|---------------------------------------------------------------|
| `ZipTiffProvider`           | TIFF images in ZIP files (Java fmt)                           |
| `TiffDirectoryProvider`     | Loose TIFFs with pattern naming                               |
| `StackTiffProvider`         | Multi-page TIFF stacks, incl. interleaved multichannel (CZ/ZC)|
| `OmeTiffProvider`           | OME-TIFF with metadata                                        |
| `SplitChannelProvider`      | 16-bit TIFFs split into 2 channels                            |
| `MultiChannelFolderProvider`| Separate folders per channel                                  |
| `NumpyProvider`             | In-memory NumPy arrays (testing)                              |

**Interleaved multichannel stacks:** `StackTiffProvider(num_channels=N, channel_order=...)` de-interleaves page sequences natively — no wrapper. Page index for a given `(plane, channel)` is computed by `_page_index()`:

- `channel_order="CZ"` (channel-fastest; pages `Z1C1, Z1C2, Z2C1, …`): `page = (plane-1) * num_channels + channel`
- `channel_order="ZC"` (planar; pages `Z1C1..ZnC1, Z1C2..ZnC2`): `page = channel * num_planes + (plane-1)`

`num_planes` is derived as `n_pages // num_channels` when in multichannel mode. `get_stack(time, channel)` strides the pages for the requested channel only and returns shape `(Z, Y, X)` — downstream consumers (napari layers, `measure_runner`) don't need to know about the interleaving.

The `create_image_provider_from_config()` factory routes `<image file="..." numChannels="N" channelOrder="CZ|ZC"/>` configs to this mode and **skips the `SplitChannelProvider` wrap** (interleaved channels are resolved at the page level; horizontal split would halve a valid image).

### 3.4 AuxInfo (`io/auxinfo.py`)

Embryo orientation metadata.

- **v1** (`_AuxInfo.csv`): 3-char axis string (e.g., `"ADL"`) + rotation angle
- **v2** (`_AuxInfo_v2.csv`): AP and LR orientation vectors (3D)

---

## 4. Naming System

### 4.1 Pipeline (`naming/identity.py`)

`IdentityAssigner.assign_identities()`:

1. Clear non-forced names (cells with `assigned_id` are preserved).
2. **Propagate forced names** (`_propagate_assigned_ids()`): extend each `assigned_id` forward through `successor1` chains and backward through `predecessor` chains, covering the cell's entire lifetime. Stops at division boundaries.
3. Build `CanonicalTransform` (if v2 AuxInfo available, used for cross-validation only).
4. **Topology-based identification** (`identify_founders()`).
5. If topology fails (confidence < 0.3): warn and assign generic names. Legacy diamond-pattern identification available via `legacy_mode=True`.
6. Set up `DivisionCaller` with per-timepoint lineage centroid axes and seed axes from 4-cell midpoint.
7. **Forward pass**: apply canonical rules from 4-cell stage onward (single-frame classification with quality-aware axis smoothing; multi-frame averaging disabled in lineage mode).
8. Assign generic `Nuc_t_z_x_y` names to remaining unnamed cells.

### 4.2 Founder ID (`naming/founder_id.py`)

Topology-based identification of ABa, ABp, EMS, P2 at the 4-cell stage:

1. Find 4-cell windows (exactly 4 alive nuclei).
2. **Sister pair identification** — three strategies:
   - Backward trace: cells sharing a parent are sisters.
   - Birth time grouping: cells born at the same time are sisters.
   - **Forward division pairing**: cells that next divide at similar times are sisters (for datasets starting at the 4-cell stage with no predecessor data).
3. **AB vs P1 pair**: the pair that divides first are AB daughters; the pair that divides second are P1 daughters. This is a biological invariant of *C. elegans*.
4. **Within-pair assignment**:
   - **EMS vs P2**: Primary signal is forward division timing (EMS divides before P2); secondary signal is nucleus size (EMS is typically larger).
   - **ABa vs ABp**: Projection onto the AP axis vector, averaged over the 4-cell window for robustness (more anterior = ABa). Falls back to PC1 of 4-cell point cloud when no 2-cell stage is available.
5. **Back-trace**: trace predecessors to name AB, P1, P0 and their continuation cells.
6. **Axis derivation**: compute AP, DV, LR vectors from the 4 cell positions.
7. **Confidence**: composite of timing, size, and axis confidence with per-component breakdown.

### 4.3 Division Caller (`naming/division_caller.py`)

Classifies each cell division to determine daughter names:

1. Look up the division `Rule` for the parent name.
2. Compute the division vector (daughter2 − daughter1), z-scaled by `z_pix_res`.
3. Rotate the vector into the canonical frame.
4. Dot product with the rule's axis vector determines which daughter gets which name.
5. Angle between division vector and rule axis maps to a confidence score.
6. If confidence < 0.3, **deferred majority-vote evaluation**: follow daughters forward up to 8 frames, re-classify at each, and use majority vote.

Four coordinate transform modes (selected automatically based on available data):
- **v2**: Full `CanonicalTransform` rotation (Wahba's problem solver). Used when AuxInfo v2 is available.
- **v1**: Sign-flip matrix + 2D rotation by angle. Used when AuxInfo v1 is available.
- **Lineage centroid** (primary no-AuxInfo mode): Per-timepoint axes derived from ABa/ABp/EMS/P2 lineage centroids via `lineage_axes.py`. Rotation-invariant — automatically handles embryo rotations during imaging. Includes LR quality metric, quality-aware sign correction with gap limits, and temporal LR smoothing for degenerate frames.
- **Static founder** (legacy fallback): Project onto axes derived once from the 4-cell stage positions. Used only when lineage centroid axes are unavailable at a given timepoint.

Multi-frame averaging is disabled in lineage centroid mode (per-timepoint axes make cross-frame averaging unreliable). Seed axes from the 4-cell midpoint provide initial sign anchoring.

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

Abstract base: `EditCommand` with `execute()`, `undo()`, `description`, `structural`.

The `structural` property (default `True`) indicates whether the edit changes lineage structure (links, identity, etc.). Non-structural edits like `MoveNucleus` (`structural = False`) skip the expensive naming + tree rebuild in the edit callback and only refresh the display. This prevents cell deselection when nudging positions.

| Command                    | Operation                                 | State Captured                        |
|---------------------------|-------------------------------------------|---------------------------------------|
| `AddNucleus`               | Create nucleus at position               | Added index                           |
| `RemoveNucleus`            | Kill nucleus (status=-1)                  | Old status, identity, assigned_id     |
| `MoveNucleus`              | Change position/size                      | Old x, y, z, size                     |
| `RenameCell`               | Set identity + assigned_id across the cell's entire continuation chain (atomic, cell-scoped) | List of (time, idx, old identity, old assigned_id) tuples for every nucleus in the chain |
| `SwapCellNames`            | Atomically swap the forced names of two cells (writes B's name onto all of A's chain and vice versa) | Two lists of (time, idx, old identity, old assigned_id) tuples, one per chain |
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

**Multi-channel display:** The app creates one napari Image layer per channel. Single-channel data uses a gray colormap; multi-channel (e.g. split-channel dual-color) uses green/magenta with additive blending. Channel visibility and per-channel contrast are controlled by `ContrastTools`.

**Visualization mode:** The app supports two color modes for nucleus display, toggled via the Edit Panel:
- **Editing mode** (default): hardcoded status palette — white=selected, purple=named, orange=unnamed, gray=none.
- **Visualization mode**: rule-based coloring via `ColorRuleEngine`. Presets include lineage-depth (rainbow) and expression (viridis colormap). Users can define custom rules.

**Z-plane deselect:** Manually changing z-plane deselects the active cell (the user is exploring, not tracking). Time navigation continues to follow the tracked cell's centroid.

**Widget layout:**
```
┌──────────────────────────────────────────────┐
│  napari Viewer (image + nucleus overlay)      │
├──────────┬───────────────────┬───────────────┤
│ Contrast │                   │ Edit          │
│ (per-ch) │                   │ Tools         │
│          │                   │               │
│ Lineage  │                   │               │
│ List     │                   │               │
├──────────┴───────────────────┴───────────────┤
│ Player Controls (time/plane/labels/deselect/3D)│
│ Lineage Tree (Sulston tree visualization)     │
└──────────────────────────────────────────────┘
```

Napari's default layer list and layer controls panels are hidden on startup to save screen space. They remain accessible via napari's Window menu.

**Keyboard shortcuts:**
| Key            | Action                   |
|----------------|--------------------------|
| `Right`/`Left` | Next/previous timepoint  |
| `Up`/`Down`    | Next/previous z-plane (deselects active cell) |
| `Ctrl+S`       | Save                     |
| `Ctrl+Shift+S` | Save As                  |
| `Ctrl+Z`       | Undo                     |
| `Ctrl+Y`       | Redo                     |
| `Delete`       | Remove nucleus at current timepoint |
| `Escape`       | Exit active mode (Add, Track, Relink pick) |

### 6.2 ViewerIntegration (`gui/viewer_integration.py`)

Draws nucleus circles as a napari Shapes layer (polygon approximation with 32 vertices for aspect-ratio-independent circles).

**Mouse interaction (priority order):**
- In **add mode**: left-click places a nucleus at the click position
- In **relink pick mode**: right-click selects the relink target (works in both 2D and 3D views)
- In **track/placement mode**: right-click places a tracking nucleus
- **Right-click** on nucleus: select cell (make active) — requires click within the drawn circle
- **Left-click** on nucleus: toggle label visibility on/off — requires click within the drawn circle

**Division line overlay:** When the selected cell has just divided (current_time == cell.end_time + 1), a yellow line connects the two daughter cell positions. Disappears on any navigation or selection change.

**Ghost trail layer:** When enabled (via the Trails button in Edit Tools), a semi-transparent trail of the selected cell's past positions is drawn as shapes connected by lines. Trail length is configurable (default 10 timepoints). Works in both 2D (shapes) and 3D (points).

**Hover tooltip:** A floating tooltip appears when hovering over a nucleus, showing the cell name and basic info. Uses a delay (`_hover_delay_ms = 300ms`) to avoid flicker.

### 6.3 LineageWidget (`gui/lineage_widget.py`)

Sulston tree rendered in a `QGraphicsView` with expression-colored branch segments. Multiple panels can be open simultaneously, each with independent configuration.

Uses `_ClickableGraphicsView` subclass to handle single-click despite `ScrollHandDrag` mode (detects clicks on mouseRelease when mouse movement < 5px).

**Per-panel configuration** (via Settings button or `LineagePanelConfigDialog`):
- **Root cell**: display any subtree (e.g. "ABa" for only ABa descendants), or auto-detect
- **Time range**: restrict display to a timepoint window
- **Expression range**: min/max values for color mapping
- **Colormap**: matplotlib colormap (viridis, plasma, inferno, etc.) or legacy green-to-red

**Mouse interaction:**
- **Left-click** on cell: select cell, jump to clicked y-position timepoint
- **Right-click** on cell: select cell, jump to cell's end time
- **Mouse wheel**: zoom in/out

**Multi-panel management** (`app.py`):
- `_lineage_widgets: list[LineageWidget]` tracks all open panels
- `add_lineage_panel()` creates a new panel with configurable parameters
- All panels rebuild synchronously after edit operations
- **Window > New Lineage Panel...** menu action opens a config dialog

**Menu bar additions** (injected into napari's menu bar at launch):
- **File → Measure…** — opens `MeasureDialog`, runs per-channel pixel measurement via `analysis.measure_runner.run_measure`, writes CSVs, refreshes tree colors (see §7.3).
- **Window → \<panel toggles\>** — auto-generated `toggleViewAction()` entries for every dock widget.
- **Window → New Lineage Panel…** — opens `LineagePanelConfigDialog`.

### 6.4 LineageLayout (`gui/lineage_layout.py`)

Pure computational layout engine (no Qt dependency):
- `compute_layout(root_cell, params)` → `dict[str, LayoutNode]`
- Recursive leaf-counting for x-position assignment
- Y-axis = time (start_time → end_time scaled by `y_scale`)
- Expression coloring via `expression_to_color()` — supports matplotlib colormaps (passed as `cmap_name`) or legacy green-to-red gradient
- Daughter ordering follows Java AncesTree convention

### 6.5 Other Widgets

| Widget             | Purpose                                      |
|--------------------|----------------------------------------------|
| `LineageListWidget` | Hierarchical QTreeWidget with search/filter  |
| `PlayerControls`    | Time/plane navigation, play/pause, labels toggle, deselect, 3D mode, 3D window |
| `CellInfoPanel`     | Cell info builder (used by hover tooltip)    |
| `ContrastTools`     | Per-channel contrast sliders with visibility toggles, auto-contrast |
| `EditPanel`         | Color mode toggle, edit buttons, D-pad move (popup), relink, add/track, trails, screenshot/record, edit history (popup) |
| `ColorRulesDialog`  | Rule list editor popup: add/edit/delete/reorder rules, "All other cells" default color, apply to engine |
| `_RuleEditorDialog` | Single rule editor: criterion, pattern, color mode, color picker, colormap settings, match mode help |

### 6.6 Interactive Modes

**Relink pick mode** — Replaces index-based dialogs with a unified pick-mode workflow:
1. Select either cell in the pair you want to link (order doesn't matter).
2. Click **Relink** → enters pick mode.
3. Navigate to the other cell, **right-click** to select target.
4. Gap = 1: simple relink. Gap > 1: automatic interpolation.

**Add mode** — Click-to-place nucleus with automatic predecessor linking:
1. (Optional) Select an existing cell. Click **Add** (toggle).
2. **Left-click** in viewer to place. Inherits identity, diameter, and predecessor from selected cell.
3. Gap > 1 triggers automatic interpolation.

**Track mode** — Continuous click-to-place across timepoints:
1. Select parent cell. Click **Track** (toggle).
2. Navigate to later timepoints, **right-click** to place.
3. Mode stays active until Esc or re-click Track.

All modes are mutually exclusive and can be cancelled with **Escape**.

### 6.7 Color Rule Engine (`gui/color_rules.py`)

Provides a flexible, rule-based system for assigning colors to nuclei in visualization mode. Rules are evaluated in priority order; the first matching rule wins. Unmatched nuclei fall through to a configurable default color (white semi-transparent by default).

**`RuleCriterion` enum:** `ALL`, `NAME_EXACT`, `NAME_PATTERN` (glob), `NAME_REGEX`, `LINEAGE_DEPTH` (range), `FATE`, `EXPRESSION` (rweight range).

**`ColorMode` enum:** `SOLID` (fixed RGBA), `COLORMAP` (map a numeric value through a matplotlib colormap).

**`ColorRule` dataclass:** name, criterion, pattern, color_mode, color, colormap, vmin, vmax, priority, enabled.

**`ColorRuleEngine`:**
- `set_rules(rules)` — sort by descending priority
- `load_preset(preset)` — load built-in rule sets
- `color_for_nucleus(nuc, manager, time)` — evaluate rules, return RGBA
- `colors_for_frame(nuclei, manager, time)` — batch evaluation with per-frame cell cache

**Built-in presets:** `PRESET_LINEAGE_DEPTH` (rainbow by depth 0–10), `PRESET_EXPRESSION` (viridis colormap by rweight).

### 6.8 3D Volume View

Toggled via the **3D** button in player controls. Switches napari to `ndisplay=3` and creates a `Points` layer with:
- Sphere size proportional to nucleus diameter.
- Anisotropic z-scaling via `scale=(z_pix_res, 1.0, 1.0)`.
- In editing mode: white=selected, purple=named, orange=unnamed (Nuc\*), gray=no name.
- In visualization mode: colors from the active `ColorRuleEngine` rules.

All channels are loaded as 3D stacks when entering 3D mode. Click-to-select and relink pick mode work in 3D.

### 6.9 Detached 3D Viewer (`gui/viewer_3d_window.py`)

A standalone `QWidget` window containing an embedded napari viewer, always in 3D mode with visualization-mode coloring. Launched via the **3D Window** button in player controls.

**Features:**
- **Time sync:** Time slider/spinner with a Sync toggle. When Sync is on, the window follows the main viewer's timepoint. When off, it navigates independently.
- **Color preset selector:** Dropdown for switching visualization presets (lineage depth, expression).
- **Per-channel contrast:** Same controls as main viewer — visibility checkboxes, min/max sliders, auto/reset per channel.
- **Label controls:** Left-click on a 3D sphere toggles its label. "Labels: ON/OFF" button for global toggle. "Clear Labels" to remove all.
- **Ghost trails:** Mirrors the main viewer's trail visibility settings.
- **Multi-channel:** Loads all image channels with green/magenta colormaps.

Multiple 3D windows can be open simultaneously. Each is tracked in `app._3d_windows` and refreshed by `update_display()`.

### 6.10 Dataset Creation

`AceTreeApp.from_new_dataset(config, num_timepoints, output_dir)` creates an app with an empty `NucleiManager` for manual annotation. `AceTreeApp.from_dialog()` shows the `DatasetCreationDialog` wizard first.

The `create` CLI command (`__main__.py`) provides both interactive (wizard dialog) and non-interactive (CLI flags) paths to dataset creation.

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

### 7.3 Pixel Measurement (`analysis/measure.py`, `measure_csv.py`, `measure_runner.py`)

Port of the Java `AceBatch2` measure routine (`org.rhwlab.analyze.ExtractRed` + `RedBkgComp2` + `NucleiMgr.computeRWeight`). Samples pixel intensity at every nucleus in every image channel, writes per-channel CSVs, and feeds the user-chosen channel's measurements back into `nuc.rwraw`/`rwcorr1` so the lineage tree re-colors from live measurement.

**`measure.py` — pixel sampling:**

| Function             | Purpose                                                             |
|----------------------|---------------------------------------------------------------------|
| `project_radius()`   | Projected XY radius of a spherical nucleus at a given Z-plane       |
| `measure_nucleus()`  | Sum + count of pixels in the inner disk and outer annulus at every plane the nucleus touches |
| `measure_timepoint()`| Map `measure_nucleus` across all nuclei at one timepoint            |

Each nucleus is modelled as a sphere of diameter `nuc.size` centred at `(x, y, z)`; at each plane that the sphere intersects, a 2D disk (inner) and concentric annulus (`DEFAULT_ANNULUS_SCALE = 1.5`) are rasterised and pixel sums/counts accumulated. Dead nuclei (`status < 1`) return `(0, 0, 0, 0)`.

**`measure_csv.py` — CSV writer:**

`write_measure_csv(path, rows, n_timepoints)` produces one CSV per channel with absolute-time columns: `cell_name, start_time, end_time, t1, t2, …, tN`. Cells absent at a given timepoint get an empty cell in that column. The AT channel's file is named `measure_channel{n}_AT.csv`; other channels get `measure_channel{n}.csv`.

**`measure_runner.py` — orchestrator:**

`run_measure(manager, image_provider, output_dir, at_channel, progress_cb=None)`:

1. Iterates every channel, every timepoint. For each `(t, channel)` it calls `image_provider.get_stack(t, channel)` and `measure_timepoint`, collecting `(sum_in, count_in, sum_ann, count_ann)` for every nucleus.
2. For the chosen `at_channel` only: writes `rwraw = round(sum_in * 1000 / count_in)` and `rwcorr1 = round(sum_ann * 1000 / count_ann)` back onto each `Nucleus` (the `* 1000` SCALE matches Java `NucleiMgr.computeRWeight`). Leaves untouched nuclei (dead or unmeasured) unchanged so prior values aren't blown away.
3. Recomputes `rweight` via `manager.compute_red_weights()` (under `correction = "none"`, falls back to copying `rwraw` directly).
4. Writes one CSV per channel. The per-timepoint CSV value follows the session's current correction method — plain `rwraw` for `"none"`, `rwraw - rwcorr1` otherwise.

**Cancellation:** `progress_cb(channel_idx, n_channels, t_1based, n_timepoints) -> bool | None` is fired after every timepoint. Returning `False` raises `RuntimeError("Measure cancelled by user")`.

**Scope note:** Only `rwcorr1` (global background) is computed. `rwcorr2` / `rwcorr3` came from external MATLAB in the Java pipeline; `rwcorr4` (crosstalk) was deferred. `"local"` / `"blot"` / `"cross"` correction modes fall back to `rwraw - rwcorr1` for the CSV output.

**GUI wiring:** `File → Measure…` (added in `gui/app.py::_add_file_menu_actions`) opens `MeasureDialog` (channel combo + output-dir picker), runs the orchestrator under a `QProgressDialog`, and rebuilds every lineage widget on completion so the fresh `rweight` values show up.

---

## 8. CLI

Entry point: `acetree_py/__main__.py` (typer app)

```
acetree-py load <config.xml>                    # Print dataset summary
acetree-py gui <config.xml>                     # Launch napari GUI
acetree-py create [<image_dir>] [OPTIONS]       # Create new dataset from raw images
acetree-py export <config.xml> -f cell_csv      # Export data
acetree-py rename <config.xml> -o renamed.zip   # Run naming + save
acetree-py info <config.xml> -c ABala           # Query cell details
```

---

## 9. Testing

31 test files in `tests/` covering all modules: data model, I/O (including interleaved multichannel TIFFs), naming, editing, GUI widgets, color rules, CLI, analysis (including pixel measurement), and integration tests. Run with `pytest tests/` (≥646 tests, napari-integration tests can be deselected with `--deselect tests/test_gui_widgets.py::TestNapariIntegration`).

---

## 10. Build & Installation

```toml
# pyproject.toml
[project]
name = "acetree-py"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24", "scipy>=1.10", "tifffile>=2023.1",
    "typer>=0.9", "matplotlib>=3.7",
]

[project.optional-dependencies]
gui = ["napari[all]>=0.5,<0.7", "qtpy>=2.3"]
dev = ["pytest>=7.0", "pytest-qt>=4.2", "ruff>=0.1"]

[project.scripts]
acetree-py = "acetree_py.__main__:app"
```

Install: `pip install -e .` (core) or `pip install -e ".[gui]"` (with GUI) or `pip install -e ".[all]"` (everything).

**napari version note:** The GUI uses napari's `Window.add_dock_widget()` and `Window._dock_widgets` APIs for panel management. These were tested against napari 0.5.x–0.6.x. The upper bound (`<0.7`) guards against breaking changes to these internal APIs.
