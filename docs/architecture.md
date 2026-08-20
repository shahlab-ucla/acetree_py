# AceTree-Py Architecture Reference

The normative cross-module naming and edit invariants are collected in [Naming and Manual-Curation Workflows](naming_workflows.md). Detector/tracker contracts, plugin rules, workflows, and the StarryNite migration plan are defined in the [Tracking Pipeline Specification](TRACKING_PIPELINE_SPEC.md).

**Version 0.2.0** | Python reimplementation of AceTree for *C. elegans* embryogenesis

---

## 1. Project Overview

AceTree-Py is a from-scratch Python rewrite of the Java AceTree application, which visualizes and annotates *C. elegans* embryonic cell lineage data. The rewrite targets modern tooling (napari, NumPy, SciPy) and adds full undo/redo support, topology-based cell naming, and a clean modular architecture.

### Package Structure

```
acetree_py/                    # Root package (__version__ = "0.2.0")
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
    body_axes.py               # Anatomical landmarks and validated body-axis frames
    lineage_axes.py            # Per-timepoint body axes + secondary-axis quality
    validation.py              # Post-naming validation
  editing/                     # Command-pattern edit system — no GUI deps
    commands.py                # Reversible edit commands and composites
    history.py                 # EditHistory (undo/redo stacks)
    validators.py              # Pre-edit validation functions
  io/                          # File I/O — no GUI dependencies
    config.py                  # AceTreeConfig, load_config()
    config_writer.py           # write_config_xml() (round-trip XML serialization)
    nuclei_reader.py           # read_nuclei_zip()
    nuclei_writer.py           # write_nuclei_zip()
    image_provider.py          # ImageProvider protocol + 7 providers
    auxinfo.py                 # AuxInfo (embryo orientation data)
  tracking/                    # Headless image-analysis and proposal layer
    api.py                     # Versioned requests, scopes, detections, links, results
    registry.py                # Built-ins + installed detector/tracker entry points
    detectors.py               # Anisotropic 3D DoG and LoG detectors
    lap.py                     # One-to-one Simple LAP tracker with gap closing
    pipeline.py                # Global and selected-forward orchestration
    integration.py             # ApplyTrackingProposal undoable adapter
    persistence.py             # Versioned .tracking.json sidecar
  gui/                         # napari GUI — all Qt/napari deps isolated here
    app.py                     # AceTreeApp (main application)
    viewer_integration.py      # ViewerIntegration (nucleus overlay)
    auto_tracking_dialog.py    # Modeless selected-cell configure/review workbench
    global_tracking_dialog.py  # Modeless whole-dataset draft workbench
    tracking_worker.py         # Cancellable Qt-thread analysis adapter
    tracking_preview.py        # Proposal/gap/diagnostic expansion for review
    lineage_widget.py          # LineageWidget (Sulston tree)
    lineage_layout.py          # Layout engine (pure computation)
    lineage_list.py            # LineageListWidget (hierarchical list)
    player_controls.py         # PlayerControls (time/plane/labels/3D)
    cell_info_panel.py         # CellInfoPanel (hover tooltip builder)
    contrast_tools.py          # ContrastTools (per-channel contrast)
    color_rules.py             # ColorRuleEngine, ColorRule, presets
    edit_panel.py              # EditPanel + dialog classes
    viewer_3d_window.py        # Viewer3DWindow (detached 3D viewer)
    dataset_dialog.py          # DatasetCreationDialog (5-page wizard)
    measure_dialog.py          # MeasureDialog (channel + output picker)
    expression_plot_window.py  # Multi-instance modeless expression plotting UI
    expression_comparison_window.py # Multi-dataset replicate comparison UI
  analysis/                    # Post-hoc analysis — no GUI dependencies
    expression.py              # Expression time series analysis
    expression_plot.py         # Plot snapshots, time transforms, tidy CSV
    expression_smoothing.py    # Gap-preserving Gaussian smoothing primitives
    expression_comparison.py   # Cross-dataset grids, summaries, snapshots/export
    expression_comparison_result.py # Versioned portable .aceexpr captures
    expression_dataset_repository.py # Detached XMLs + session measurement cache
    expression_measurements.py # Revision-bound all-channel Measure store
    export.py                  # CSV, Newick export functions
    measure.py                 # Per-nucleus pixel sampling (port of ExtractRed)
    measure_csv.py             # Measure CSV writer (per-channel, absolute time)
    measure_runner.py          # Measure orchestrator + correction-neutral families
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

- **`core/`**, **`naming/`**, **`editing/`**, **`io/`**, and **`tracking/`** have zero GUI imports.
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
| `identity`     | `str`        | Current automatic/computed Sulston name (e.g., `"ABala"`) |
| `assigned_id`  | `str`        | Explicit user override (survives re-naming)       |
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

`effective_name` is the read boundary for UI labels, cell lookup, edit targeting, validation, and parent-rule selection. `identity` may be recalculated; `assigned_id` changes only through an explicit editing command. Automatic suggestions are never promoted to forced state.

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
- `NucleiManager.new_empty(config, num_timepoints)` — create an empty manager for manual annotation or as the target of an accepted tracking proposal (all timepoints initialized to empty lists)

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

`NamingMethod` enum: `STANDARD=2`, `MANUAL=2`, `NEWCANONICAL=3`. Manual mode normalizes/propagates explicit overrides and skips automatic founder/division assignment; it does not bypass forced-name consistency checks.

### 3.2 Nuclei Reader/Writer

**ZIP structure:**
```
nuclei/
    t001-nuclei    # CSV: one Nucleus per line
    t002-nuclei
    ...
```

- `read_nuclei_zip(path)` → `list[list[Nucleus]]`
- `write_nuclei_zip(nuclei_record, path, start_time=1)` — writes new-format CSV to a temporary archive in the destination directory, then atomically replaces the destination. Atomic replacements preserve an existing destination's file mode; a new file uses the normal process umask rather than inheriting the private `0600` mode of its staging file.

When a manager contains manual body axes, Save also writes the matching AuxInfo v2 sidecar. The archive and sidecar are fully staged before either visible file changes. The sidecar is committed first with a same-directory rollback copy, and the archive is normally the final commit; if either commit raises, the prior archive/sidecar set is restored. When Measure has changed XML-backed correction state, ordinary Save also pre-stages the dirty XML and retains the old archive until the additional final XML replacement succeeds; a staging, archive, or XML-commit failure restores the prior archive/sidecar/config set. Undoing a manual frame removes only an AceTree-created sidecar under the same transaction—acquisition-provided sidecars are retained. Save As updates the config's nuclei path only after the data save succeeds, then atomically rewrites the source XML so reopening that config follows the new ZIP. If XML persistence fails, the in-memory target and savepoint remain unchanged (the newly written ZIP is retained as a standalone safety copy).

When at least one tracking proposal has been accepted, the application also writes the latest `TrackingResult` beside the nuclei ZIP as `<stem>.tracking.json`. This versioned JSON records the request, detections, links, warnings, plugin provenance, and optional selected-forward `TrackingOutcome` (stop reason, prediction, search radius, and review-only candidates); the nuclei ZIP remains the authoritative curated dataset. Opening a dataset restores the optional sidecar for provenance, and a missing or malformed sidecar does not prevent the backward-compatible ZIP from opening.

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
- **v2** (`_AuxInfo_v2.csv`): AP and LR orientation vectors (3D), plus orientation source, quality, and reference time

`BodyAxisFrame` (`naming/body_axes.py`) is the validated in-memory representation. AP points posterior→anterior, DV ventral→dorsal, and LR right→left; `DV = AP × LR`. Frames can be built from AuxInfo vectors or from manual anatomical endpoints. All z coordinates are multiplied by `z_pix_res` before vector construction. Manual frames are persisted as v2 sidecars.

AuxInfo selection is based on usability, not merely file presence. A v2 record must contain finite, non-zero, non-parallel AP/LR vectors; otherwise a valid supported v1 orientation is selected. If neither file supplies an orientation, readable shape and resolution measurements remain available, but the record is not allowed to claim a body frame.

---

## 4. Naming System

### 4.1 Pipeline (`naming/identity.py`)

`IdentityAssigner.assign_identities()`:

1. Clear non-forced names (cells with `assigned_id` are preserved).
2. **Propagate forced names** (`_propagate_assigned_ids()`): extend each `assigned_id` only through live reciprocal one-successor continuations. Stop at divisions, dead/missing links, non-reciprocal links, or a different forced identity.
3. Select orientation in precedence order: valid v2 (manual or imported), supported v1, per-timepoint lineage geometry, then static founder geometry.
4. **Topology-based identification** (`identify_founders()`).
5. If topology fails (confidence < 0.3): preserve compatible loaded names for partial movies, but invalidate any automatic founder hypothesis contradicted by current topology. A curated false four-object stage (two blastomeres plus two substantially smaller deleted polar detections) is reconciled to AB/P1 from timing, explicit AP, or blastomere-size evidence. Ambiguous ordering receives neutral names rather than stale four-cell labels. Legacy diamond-pattern identification remains available via `legacy_mode=True`.
6. Set up `DivisionCaller` with the selected orientation source and deterministic chronological axis caching.
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
6. **Axis derivation**: AP is P2→ABa; the DV seed is EMS→ABp projected perpendicular to AP; LR completes the right-handed frame.
7. **Confidence**: composite of timing, size, and axis confidence with per-component breakdown.

When manual initialization initially mistakes two polar bodies for blastomeres,
the retained rows provide a narrow correction footprint: four rows, two live
unforced founder-labelled survivors, and two substantially smaller dead rows.
After the second removal, naming invalidates the rejected four-cell hypothesis
and reconstructs `AB`/`P1`. Evidence precedence is observed division timing,
trusted posterior→anterior metadata, then two-cell blastomere-size asymmetry.
One missing division is treated as right-censored unless its intact sister is
observed beyond the other division. Reciprocal lineage ancestry is traced back
through continuation frames; a genuine pair-of-sister-pairs four-cell lineage,
or malformed claimed topology, vetoes recovery even when dead rows are small.
Automatic name changes made during the structural rebuild are recorded in the
same history boundary, so Undo/Redo restores exact name ownership as well as
live/dead state. A failed post-commit naming pass is rolled back to its clean
callback boundary before one safe retry, and the retry result refreshes that
same history entry.

### 4.3 Division Caller (`naming/division_caller.py`)

Classifies each cell division to determine daughter names:

1. Look up the division `Rule` for the parent name.
2. Compute the division vector (daughter2 − daughter1), z-scaled by `z_pix_res`.
3. Rotate the vector into the canonical frame.
4. Dot product with the rule's axis vector determines which daughter gets which name.
5. Angle between division vector and rule axis maps to a confidence score.
6. If confidence < 0.3, **deferred majority-vote evaluation**: follow daughters forward up to 8 frames, re-classify at each, and use majority vote.

Four coordinate transform modes are selected by explicit precedence:

- **v2**: Full `CanonicalTransform` rotation, including manual landmark frames. A valid explicit v2 frame wins over inferred geometry.
- **v1**: Sign-flip matrix plus 2D rotation for supported anatomical strings (`ADL`, `AVR`, `PDR`, `PVL`). Placeholders such as `XXX` are not orientation.
- **Lineage centroid**: Per-timepoint AP and DV estimates from ABa/P2 and ABp/EMS lineage centroids via `lineage_axes.py`, with quality-aware continuity.
- **Static founder**: The same construction at the four-cell midpoint, used when current lineage axes are unavailable.

Multi-frame averaging is disabled in lineage centroid mode (per-timepoint axes make cross-frame averaging unreliable). Seed axes from the 4-cell midpoint provide initial sign anchoring.

Signed LR cannot be derived from the ABa–ABp pair alone at the four-cell stage. A trusted secondary orientation, manual cue, or later handedness is required for a biologically grounded sign. Therefore every division suggestion carries confidence, axis label, and provenance; weak geometry remains correctable rather than being converted into a forced name.

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

The `structural` property (default `True`) indicates whether an edit can affect lineage or naming. `MoveNucleus` is structural because division classification is geometry-dependent. Selection stability is provided by the app's `(time,index)` anchor rather than by skipping reprocessing.

| Command                    | Operation                                 | State Captured                        |
|---------------------------|-------------------------------------------|---------------------------------------|
| `AddNucleus`               | Create nucleus at position               | Added index                           |
| `RemoveNucleus`            | Kill nucleus (status=-1)                  | Old status, identity, assigned_id     |
| `MoveNucleus`              | Change position/size                      | Old x, y, z, size                     |
| `RenameCell`               | Set identity + assigned_id across the cell's entire continuation chain (atomic, cell-scoped) | List of (time, idx, old identity, old assigned_id) tuples for every nucleus in the chain |
| `ClearNameOverride`        | Return a cell continuation to automatic naming | Old identity and assigned_id per nucleus |
| `SetCellNameState`         | Set automatic/forced state over one anchored continuation component | Old identity and assigned_id per nucleus |
| `SwapCellNames`            | Atomically swap the forced names of two cells (writes B's name onto all of A's chain and vice versa) | Two lists of (time, idx, old identity, old assigned_id) tuples, one per chain |
| `RelinkNucleus`            | Change predecessor link                   | Old/new pred, both parents' successors|
| `KillCell`                 | Kill all nuclei of a named cell           | List of (time, idx, old state) tuples |
| `ResurrectCell`            | Restore dead nucleus                      | Old status, identity, assigned_id     |
| `RelinkWithInterpolation`  | Link with interpolated intermediates      | Added nuclei list, old/new links      |
| `SetBodyAxes`              | Install manual orientation and invalidate naming | Previous AuxInfo/frame state |
| `CompositeCommand`         | Group one user gesture into one history entry | Ordered child commands |

`tracking/integration.py` adds `ApplyTrackingProposal`, an `EditCommand` that validates an accepted proposal against curated nuclei, expands gap links into adjacent AceTree records, and commits all new records and links as one exact undo/redo unit. `AceTreeApp` also compares the edit-history revision captured before analysis with the current revision, so a stale proposal cannot overwrite edits made while analysis was running.

### 5.2 Undo/Redo (`editing/history.py`)

`EditHistory` maintains two stacks:
- `_undo_stack`: commands that have been executed
- `_redo_stack`: commands that have been undone

Flow: `do(cmd)` → execute + push undo + clear redo. `undo()` → pop undo + reverse + push redo. `redo()` → pop redo + re-execute + push undo. New edits always clear the redo stack. Max 1000 commands (configurable).

One completed GUI gesture produces one command. A composite executes children in order and undoes them in reverse order, so add/track/relink interpolation cannot be left half-committed by a single Undo.

Dirty state is savepoint-based, not stack-length-based. Save and Save As mark the current state only after all files are written successfully. Undo/redo can return exactly to that state; editing after Undo creates a distinct branch. Save As also updates `config.zip_file`, making the new archive the target of subsequent Save operations.

### 5.3 Validators (`editing/validators.py`)

Pre-edit validation returns `list[str]` error messages (empty = valid). Checks index ranges, alive status, reciprocal continuity, successor capacity (max 2 children), time ordering, and forced-name conflicts. Names are trimmed and reject commas, CR/LF, and control characters before serialization.

---

## 6. GUI System

### 6.1 AceTreeApp (`gui/app.py`)

Main coordinator. Owns the napari `Viewer`, `NucleiManager`, `EditHistory`, and all dock widgets.

**Multi-channel display:** The app creates one napari Image layer per channel. Single-channel data uses a gray colormap; multi-channel (e.g. split-channel dual-color) uses green/magenta with additive blending. Channel visibility and per-channel contrast are controlled by `ContrastTools`.

**Visualization mode:** The app supports two color modes for nucleus display, toggled via the Edit Panel:
- **Editing mode** (default): hardcoded status palette — white=selected, purple=named, orange=unnamed, gray=none.
- **Visualization mode**: rule-based coloring via `ColorRuleEngine`. Presets include lineage-depth (rainbow) and expression (viridis colormap). Users can define custom rules.

**Stable selection:** The app stores `_selection_anchor = (time,index)` and re-resolves the selected nucleus after naming/tree rebuilds. Changing z-plane does not clear selection; explicit Deselect does. Any index fallback is time-qualified so duplicate per-frame indices cannot select the wrong nucleus.

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
| `Up`/`Down`    | Next/previous z-plane      |
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

**Ghost trail layer:** When enabled (via the Trails button in **Edit & Tracking Tools**), a semi-transparent trail of the selected cell's past positions is drawn as shapes connected by lines. Trail length is configurable (default 10 timepoints). Works in both 2D (shapes) and 3D (points).

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
| `EditPanel`         | Color mode toggle, edit buttons, body-axis landmarks, D-pad move (popup), relink, manual/selected-forward/empty-dataset tracking, trails, screenshot/record, edit history (popup) |
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
2. **Left-click** in viewer to place. Inherits diameter and predecessor. Automatic identity stays automatic; only an existing parent `assigned_id` propagates as forced state.
3. Gap > 1 triggers automatic interpolation. Placement plus interpolation is one `CompositeCommand`.

**Manual Track mode** — Continuous click-to-place across timepoints:
1. Select parent cell. Click **Manual Track** (toggle).
2. Navigate to later timepoints, **right-click** to place.
3. Mode stays active until Esc or re-click Manual Track.

**Track Selected Cell Forward** — A semiautomated, selected-cell proposal:
1. Select a live nucleus and choose **Tracking > Track Selected Cell Forward…**
   or the same action in the scrollable **Edit & Tracking Tools** dock.
2. The modeless workbench builds a `selected_forward` request using DoG or LoG, Simple LAP, a moving local ROI, and an ambiguity threshold. Common settings, advanced filtering, and session-preserved refinements remain editable.
3. Analysis reports per-frame progress and supports cancellation. The pipeline follows only the seeded continuation and stops rather than guessing at ambiguity or a likely division.
4. `tracking_preview.py` expands gap links into the same interpolated positions acceptance will create and retains diagnostic candidates/search regions separately. `ViewerIntegration` renders them in dedicated read-only napari layers with redundant color and circle/diamond/cross/path/ring symbols that never share callbacks or selection with curated `Nuclei`.
5. The review table and image overlay remain available while users navigate time/Z, inspect the stopping frame, change parameters, and rerun. Changed settings or document edits disable acceptance until a fresh preview completes.
6. Discard restores the original view and changes nothing. Accepting uses `ApplyTrackingProposal`, creates one undo entry, clears the temporary layers, and selects the new terminal nucleus.

All modes are mutually exclusive and can be cancelled with **Escape**.

**Division preview** — A second-daughter placement calls `NucleiManager.suggest_division_names()` with the actual parent and raw daughter coordinates. The manager applies physical z scaling and the parent's rule, then returns `first_name`, `second_name`, `confidence`, `axis_label`, `source`, and ambiguity. Suggestions populate `identity` only. This prevents manual placement from hard-coding `a/p` or freezing a prediction in `assigned_id`.

**Body orientation** — The panel accumulates anatomical endpoint labels at one reference frame: posterior+anterior and either ventral+dorsal or right+left. `BodyAxisFrame.from_landmarks()` validates and constructs the third axis; `SetBodyAxes` applies it as one undoable command. Re-labeling/swapping an endpoint pair is the correction path. Manual frame source and quality are shown and persisted in AuxInfo v2.

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

All channels are loaded as 3D stacks when entering 3D mode. Click-to-select and relink pick mode work in 3D. Tracking proposals use separate read-only napari Points and Shapes/path layers, with ring/diamond/cross symbols for detections, interpolation, and diagnostic candidates. A stopped selected-forward search is a calibrated three-ring wireframe sphere. Proposal rendering normalizes Z to stack-local coordinates; the current image-provider/navigation contract still assumes datasets begin at plane 1, so non-default `plane_start` is not advertised as an end-to-end loading feature.

### 6.9 Detached 3D Viewer (`gui/viewer_3d_window.py`)

A standalone `QWidget` window containing an embedded napari viewer, always in 3D mode with visualization-mode coloring. Launched via the **3D Window** button in player controls.

**Features:**
- **Time sync:** Time slider/spinner with a Sync toggle. When Sync is on, the window follows the main viewer's timepoint. When off, its image, curated nuclei, trails, and proposal layers all navigate independently.
- **Color preset selector:** Dropdown for switching visualization presets (lineage depth, expression).
- **Per-channel contrast:** Same controls as main viewer — visibility checkboxes, min/max sliders, auto/reset per channel.
- **Label controls:** Left-click on a 3D sphere toggles its label. "Labels: ON/OFF" button for global toggle. "Clear Labels" to remove all.
- **Ghost trails:** Mirrors the main viewer's trail visibility settings.
- **Multi-channel:** Loads all image channels with green/magenta colormaps.
- **Tracking visualization mirroring:** Receives proposal visibility, stale styling, selected review point, diagnostic candidates, paths, search-region state, and the separate current-frame detector-test rings from `ViewerIntegration` without changing its local time.

Multiple 3D windows can be open simultaneously. Each is tracked in `app._3d_windows` and refreshed by `update_display()`.

### 6.10 Dataset Creation

`AceTreeApp.from_new_dataset(config, num_timepoints, output_dir, tracking_request=None)` creates and writes an empty `NucleiManager`. With no request it preserves the manual workflow. With a request it retains the settings until `launch()` has initialized the image viewer and preview layers, then opens `GlobalTrackingDialog` and schedules a detector-only test on the first requested frame. The user explicitly starts the full tracking draft after tuning. Both modes run through `TrackingAnalysisWorker`; only a reviewed full draft can invoke `ApplyTrackingProposal` as one undoable edit.

`AceTreeApp.from_dialog()` shows the five-page `DatasetCreationDialog`. Step 4 defaults to **Manual annotation** or can request a global draft from any installed detector/tracker pair for pre-commit review after the viewer opens. The division option is capability-driven: it stays disabled for Simple LAP and follows the advertised default for StarryNite's division-aware tracker. Image-layout changes update the valid channel range immediately and block invalid automated configurations. The non-interactive `create` path also defaults to manual and accepts `--tracking dog-lap` or `--tracking log-lap` plus detector/linker settings.

Both the global workbench and selected-cell forward workbench share one recent-StarryNite-parameter setting. They map compatible legacy values into component settings, preserve source text and unsupported statements when writing tuned copies, and keep parameter/model paths plus hashes in request provenance. The global workbench can explicitly select either native geometry scoring or the registered legacy-exact whole-movie tracker. Exact selection additionally requires a source-bound neutral classifier export and distribution MAT file; it never deserializes or executes a MATLAB classifier object and never falls back to native behavior.

### 6.11 Tracking Pipeline Integration

`TrackingRequest` combines versioned detector/tracker `ComponentSpec` values with a `TrackingScope` (`global` or `selected_forward`). `TrackingPipeline` resolves components through `TrackingRegistry`, runs image analysis against a copied nucleus snapshot without mutating `NucleiManager`, and returns an immutable `TrackingResult` proposal. Trackers may implement the ordinary edge-only `track()` boundary, `refine_graph()` for an atomic detection-graph cleanup, or `refine_movie()` for a global-only backend. A tracker advertising `whole_movie_preflight` also receives an immutable detector/source/scope context before frame 1 is read; the registered exact backend uses it to fail closed on incomplete scope, detector identity, channel bounds, calibration, and every source-bound legacy file, then revalidates those inputs during refinement. Its separate `detect_frame()` operation accepts only a detector spec and one timepoint, loads one complete ZYX stack, and returns immutable detections without constructing a tracker. `DetectorPreviewSnapshot` intentionally omits the nuclei copy, keeping this tuning path lightweight.

GUI workbenches run both modes through a cancellable `TrackingAnalysisWorker` and a GUI-thread callback relay; built-in image providers are reconstructed with independent TIFF/ZIP handle caches so playback and analysis do not share a file handle. A monotonic edit/change token prevents an edit→Undo cycle from reviving a stale result. Built-ins include anisotropy-aware 3D DoG/LoG and StarryNite detectors, deterministic Simple LAP with optional gap closing, StarryNite's candidate-limited native division tracker, and the global-only StarryNite exact tracker. The exact path composes sequential distribution-backed detection, early staged geometry, isolated-fragment cleanup, neutral classifier decisions, and lineage mutations inside one fail-closed `refine_movie()` call. Its `event_order_validated` provenance is structural validation of the received sequence, spans, coverage, and unique frame/row order; it is not proof that an external source omitted no event. Simple LAP is deliberately one-to-one; both StarryNite trackers can emit two-daughter split edges while still disallowing merges.

Selected-forward results also carry a structured `TrackingOutcome`: completed, lost, ambiguity, likely division, or curated-data conflict, plus the stop frame, last accepted frame, predicted physical position, search radius, and non-committable review candidates. `ExpandedTrackingPreview` materializes accepted interpolation separately from those diagnostics. Current-frame detector results are GUI-local and transient: `expand_detector_preview()` marks them as non-committable and `ViewerIntegration` renders them in separate purple, read-only Shapes/Points layers. Proposal and detector-test state both mirror into the main 2D/3D view and detached 3D windows while preserving the active editing layer, but detector tests never enter persistence or edit history.

`TrackingRegistry` also discovers installed plugins from the `acetree_py.tracking.detectors` and `acetree_py.tracking.trackers` entry-point groups; one broken plugin is reported without preventing other components from loading. The stable contracts, TrackMate-compatible settings vocabulary, plugin packaging rules, UI states, and implemented native/exact StarryNite adapters are specified in [Tracking Pipeline Specification](TRACKING_PIPELINE_SPEC.md).

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

1. Traverses timepoints once and collects every image channel. Providers with
   `get_all_channel_stacks(t)` decode/load a combined timepoint once and
   distribute it to channels; providers backed by physically separate channel
   files load each required stack once inside the same pass. Sampling collects
   `(sum_in, count_in, sum_ann, count_ann)` and, for blot runs, blot sums/counts
   for every nucleus.
2. Builds one all-channel immutable measurement snapshot and stages one CSV per channel. The per-timepoint value follows the requested correction method — plain `rwraw` for `"none"`, `rwraw - rwcorr1` for `"global"`, and `rwraw - rwcorr3` for `"blot"`.
3. Revalidates the starting document fingerprint, then installs the staged CSV set while retaining the prior files as rollback copies.
4. For the chosen `at_channel` only, writes scaled `rwraw`, `rwcorr1`, optional
   `rwcorr3`, `rsum`, `rcount`, and the matching `rweight` onto measurable
   nuclei. A sample with no valid inner pixels clears every persisted legacy
   red field (`rweight`, `rsum`, `rcount`, `rwraw`, and `rwcorr1`–`rwcorr4`) so
   a partial run cannot retain a stale value that looks valid after reload. It
   then publishes the all-channel snapshot and releases the rollback copies.

Publication requires at least one valid sample in the selected AT channel. An
all-failed selected channel raises before staging or publication and preserves
the prior CSVs, legacy fields, measurement snapshot, correction/config value,
and config-dirty flag. With one or more valid samples, the run is deliberately
partial: valid samples publish and invalid selected-channel samples are cleared
as described above.

The runner writes every channel to private sibling files, validates the source
again, then installs the complete CSV set while retaining rollback copies. It
publishes an immutable `ExpressionMeasurementSet` and the legacy AT fields as
part of the same transaction; rollback copies are deleted only after all
in-memory publication succeeds. A cancellation, source mutation, CSV error, or
late application error restores the prior files, all legacy fields (including
`rwcorr2`/`rwcorr4`), correction mode/config-dirty state, freshness flag, and
measurement snapshot. A successful correction change marks the XML config
dirty so ordinary Save persists the correction identity alongside the nuclei
ZIP. The public
`run_measure() -> list[Path]` return contract remains unchanged.

The measurement set is keyed by `(timepoint, nucleus.index)` and stores raw,
annulus, blot, pixel-count, and selected expression values for every channel.
Each sample has a nucleus geometry signature. Calibration is checked for every
mode, and blot results also retain a movie-wide geometry dependency fingerprint
because an unselected neighbour can change the projected exclusion mask.

`measure_expression_set(manager, image_provider, ...)` exposes the same
all-channel measurement core as a nonmutating entry point. It returns an
immutable `ExpressionMeasurementSet` without publishing it on the manager,
rewriting legacy expression fields, changing correction state, or writing CSV
files. It remains the single-correction nonmutating boundary; the built-in
cross-dataset repository uses the family boundary below instead. `run_measure()`
remains the transactional, user-visible persistence path described above.

`measure_expression_family(manager, image_provider, ...)` is the optimized
cross-dataset boundary. One blot-capable movie pass builds immutable
`MeasuredExpressionAggregate` records for every nucleus and image channel:
raw intensity, global-annulus background, blot-annulus background, and pixel
counts. `ExpressionMeasurementFamily` derives `none`, `global`, and `blot`
values from those aggregates; compatible `local`/`cross` requests deliberately
use the global fallback. The family is revision-, calibration-,
dependency-fingerprint-, and geometry-bound, so it cannot silently serve data
after a relevant edit.

**Cancellation:** `progress_cb(channel_idx, n_channels, t_1based, n_timepoints) -> bool | None` is fired after every timepoint. Returning `False` raises `RuntimeError("Measure cancelled by user")`.

**Scope note:** The port computes `rwcorr1` (global annulus background) and `rwcorr3` (blot — annulus with every nucleus's projected inner disk masked out; see `measure_timepoint_with_blot` in `analysis/measure.py`). `rwcorr2` / `rwcorr4` are not computed (Java's pipeline filled them via external MATLAB and a crosstalk solver). The correction method is selected in the Measure dialog (`gui/measure_dialog.py::MeasureDialog`) and threaded through `run_measure(…, correction_method=…)`: `"none"` writes no subtraction, `"global"` writes `rwraw - rwcorr1`, and `"blot"` writes `rwraw - rwcorr3`. Compatible legacy modes `"local"` and `"cross"` use the documented fresh global fallback; unknown modes are rejected.

**GUI wiring:** `File → Measure…` (added in `gui/app.py::_add_file_menu_actions`) opens `MeasureDialog` (channel combo + output-dir picker), runs the orchestrator under a `QProgressDialog`, and rebuilds every lineage widget on completion so the fresh `rweight` values show up.

---

### 7.4 Expression Plot (`analysis/expression_plot.py`, `gui/expression_plot_window.py`)

`ExpressionPlotService` produces immutable `ExpressionPlotData` snapshots for
absolute, birth-relative, and normalized lifetime axes. Missing measurements
remain `None` in the snapshot and become NaN gaps only at the Matplotlib
boundary. Optional Gaussian smoothing is applied independently to each
continuous run, so values cannot leak across a missing sample. Each series
retains both its raw and displayed values, and the exact same immutable
snapshot feeds the renderer and tidy CSV exporter.

**Window lifecycle:** `Window → New Expression Plot…` creates an independent
`Qt.Window`; repeated actions create distinct instances with monotonic titles.
Each removes itself from `AceTreeApp._expression_plot_windows` on close.

**Concurrency:** `NucleiManager.data_revision` advances after every committed
GUI edit, including Undo and Redo. Each measurement set records its source
revision, calibration, and per-nucleus geometry; blot sets additionally record
movie-wide geometry dependencies. A mismatch fails closed: the window retains
an existing plot only as a watermarked stale visual reference, displays a
**Run Measure…** prompt, and disables both dedicated and Matplotlib-toolbar
CSV/SVG export until a new Measure run publishes data for the current revision.
Reloaded legacy data has no provenance, so it receives a nonblocking
freshness-unverified advisory rather than a false claim that the values are
current.

---

### 7.5 Cross-Dataset Expression Comparison

(`analysis/expression_dataset_repository.py`,
`analysis/expression_comparison.py`,
`gui/expression_comparison_window.py`)

**Application-scoped repository:** `ExpressionDatasetRepository` owns detached
`NucleiManager` instances for XML configurations selected in any comparison
window. Canonical resolved paths are deduplicated, image providers are created
lazily, and provider access is serialized per dataset. `AceTreeApp` owns one
repository shared by every modeless comparison window and closes it at
application shutdown, releasing image and ZIP handles. Loaded managers and
repository measurement families are session-only; Measure CSVs are neither
read as cache entries nor written by this workflow. A user can explicitly
detach a completed family into a schema-v2 `.aceexpr` measurement set. That
portable cache is a separate immutable authority and does not keep the source
manager, provider, XML, ZIP, or movie open.

Every load receives a monotonically increasing session generation. A snapshot
token combines that generation with the fingerprint of the XML, nuclei, and
representative image sources. Once a built-in image provider is opened, a
stat-only full-movie manifest additionally covers every file Measure can read
at every nonempty absolute nuclei timepoint, including non-representative
siblings and all per-plane paths, and becomes part of the token. Sources and
tokens are checked before cache reuse and export. An explicit reload closes the
provider, clears that dataset's shared measurement family and compatibility
caches, and creates a new generation even when file metadata is otherwise
identical, so snapshots held by other windows cannot silently become current
again.

A source mismatch, appearance, or disappearance raises an explicit
`DatasetSourceChangedError`. Recovery-only `session_status()` /
`session_statuses()` access keeps a stale row visible without authorizing cache
reuse or export, allowing the user to select **Reload selected**. The old figure
may remain as a visual reference, but CSV and SVG export fail closed until the
row is reloaded and prepared against the new generation. If the same XML is active
in the main viewer with unsaved edit-history or config changes, comparison also
blocks preparation/export and directs the user to Save, Reload, and prepare.

**Trace acquisition:** A comparison window requests one exact, case-sensitive
canonical cell name from each selected dataset. Missing and ambiguous cells,
missing channels, and incomplete acquired traces become explicit status
records rather than approximate matches. These selected replicates remain in
the provenance and denominators even though they contribute no invented
numeric values. There are two source boundaries:

- Saved built-in legacy expression values are accepted only when the requested
  cell has a complete trace. Because legacy nuclei archives do not establish
  source provenance, physical channel identity, or correction identity, their
  trace provenance marks freshness, channel, and correction as unverified; the
  GUI requires explicit acknowledgement before exporting an available numeric
  legacy trace. Status-only CSV contains no legacy numeric value and therefore
  does not require that acknowledgement.
- Recomputed values use the nonmutating measurement APIs. The built-in
  repository path calls `measure_expression_family()` and caches one
  correction-neutral family per dataset, not one movie result per correction.
  Its first request measures every image channel and raw/global/blot aggregate
  in one timepoint pass. Later cell, channel, correction, and window requests
  only derive/extract from that family and perform no movie reread. `none`,
  `global`, and `blot` are exact family derivations; `local`/`cross` retain the
  documented global fallback. Injected legacy test/plugin measurement
  functions continue through the compatibility per-mode cache rather than
  being misrepresented as a family.

The family is retained only after a complete movie pass. Cancellation or a
failure before completion publishes no replacement. A successfully completed
family remains authoritative even when individual nuclei could not be sampled;
those samples are explicit gaps, and ordinary new/stale preparation reuses the
completed family. **Recompute all…** is the explicit force-retry path when the
user wants to remeasure such gaps. A manager revision/calibration/geometry or
dependency mismatch drops it lazily; source fingerprint/manifest failure,
explicit reload, dataset removal, and application shutdown clear the live
repository copy and close the corresponding provider. `cached_corrections`
reports every supported derivation once the family is valid.

**Renderer-neutral comparison model:** `ExpressionComparisonService` consumes
immutable native traces; the dataset is the replicate unit. It transforms
absolute, birth-relative, or normalized-lifetime coordinates and aligns each
cell to a common union or intersection grid. Absolute and relative grids accept
an explicit step, while normalized grids use a specified point count. Alignment
never extrapolates or interpolates across an explicit gap.

`gaussian_smooth_missing()` is applied to each aligned replicate before any
pointwise summary is calculated. Sigma is expressed in displayed-axis units
and converted to grid bins, preserving missing-data segments. The center can be
none, mean, or median. A mean permits no band, sample standard deviation,
standard error, or Student-t 95% confidence interval; a median permits no band,
interquartile range, or scaled median absolute deviation. `SummarySpec` rejects
cross-family pairings, and the GUI exposes only compatible choices. Dataset
`group_id` is the condition boundary: each group is summarized independently,
and each point records selected, trace-available, and numerically valid
replicate counts. One global switch and opacity value control display of all
included individual traces; a dataset's **Use** state instead controls its
membership in both traces and summaries. Dataset colors and all other
appearance settings remain window-local in a live comparison, or become
explicit presentation metadata when the user saves a portable result or
measurement set.

**Live snapshot and export:** Each render creates an immutable numeric
`ExpressionComparisonData` containing the comparison specification,
provenance, trace availability, native values, common-grid values, displayed
smoothed values, and group-specific summary rows. Legacy acknowledgement is
recorded in provenance metadata rather than inferred later. This object drives
the numeric Matplotlib series and tidy CSV; dataset labels and trace colors
remain in its trace records. Figure-only appearance—title and axis text, fonts,
line and marker styles and widths, opacity, legend, limits, grid, and
figure/axes/text colors—is window-local live state; SVG and the toolbar Save
action write the currently rendered figure rather than claim
that appearance belongs to the numeric snapshot. All export paths first
revalidate repository generation/source tokens. A prepared comparison with only
unavailable-status records may export CSV, but SVG/toolbar Save remain disabled
because there is no numeric plot.

**Portable result and measurement-set boundary:**
`analysis/expression_comparison_result.py` reads both `.aceexpr` schema versions.
A schema-v1 legacy capture contains validated materialized `ExpressionDataset`
inputs and one fixed cell/channel/correction request. It can rebuild supported
time, grid, smoothing, summary, inclusion, and appearance views without a
repository, but it cannot answer a different measurement request. Those files
remain read compatible and deliberately open with the cell and acquisition
selectors disabled.

A schema-v2 measurement set additionally embeds one
`FrozenDatasetMeasurementCache` per fully measured dataset. Each cache is a
correction-neutral snapshot of every named observed cell, every measured image
channel, nucleus geometry/provenance, raw/global-annulus/blot-annulus
aggregates, and explicit missing-sample reasons. The five UI correction choices
are derived offline: `none`, `global`, and `blot` are direct derivations, while
`local` and `cross` deliberately use the documented global fallback. The
materialized default `ExpressionDataset` and `ComparisonSpec` remain in the
file to define its initial plot; they are a view, not the measurement-cache
authority. Changing the selected exact cell, physical channel, or correction
materializes new datasets from the caches and performs no source or image I/O.
Missing cells/channels/samples and duplicate exact cell names remain explicit
acquisition statuses or gaps instead of being filled or chosen heuristically.

The v2 cache workflow is hybrid. An opened set is immediately usable offline,
but **Add XMLs…** or XML drag/drop may attach the original source to a cached row
or add another dataset. **Recompute new or stale** visits attached rows
regardless of their plot **Use** state and measures only rows without a current
full cache; **Recompute all…** explicitly rereads every attached movie and
attempts to replace all of those caches. Unattached rows retain their portable
cache. Each dataset replacement is atomic: cancellation or failure keeps that
row's previous good cache, and other completed rows remain published. Reloading
an attached source creates a fresh repository generation; changed fingerprints
or manifests make the row eligible for the new/stale pass, while recompute-all
is the explicit force-refresh path.

The **Use** checkbox controls only participation in the current plot, summaries,
and plot/CSV export. It does not delete a cache, exclude it from measurement-set
save, or suppress either recomputation policy. Consequently an unchecked
dataset can be re-enabled or retargeted later without image I/O. A newly added
row with no completed cache is omitted from a measurement-set save until it is
recomputed; already cached rows can still be saved. Pure schema-v1 rows and
fixed rows retained in a mixed v2 file answer only their captured default
request. If the user changes cell/channel/correction, such a row must be
unchecked, removed, or attached and recomputed before saving the retargeted
measurement set.

`capture_expression_comparison_result()` establishes the immutable native-data
boundary for fixed captures;
`capture_expression_comparison_measurement_caches()` and
`revise_expression_comparison_measurement_caches()` capture and merge the v2
cache authority. A presentation revision can change dataset labels/groups,
trace labels/colors, inclusion, numeric-view settings, and appearance, while
validation protects captured numbers and source provenance. Resaving creates a
child UUID and retains the parent/capture lineage. CSV and SVG are generated
from the currently materialized offline view; **Save measurement set…** is
enabled whenever at least one full cache exists and does not require a valid
numeric plot or trigger recomputation.

Serialization uses strict finite RFC-compatible UTF-8 JSON, version-specific
field validation, duplicate-key rejection, enum/UUID/timestamp validation, and
a SHA-256 checksum over the canonical result payload. Writes stage a
same-directory temporary file, flush/fsync it, preserve the destination mode,
and commit with `os.replace`; failure leaves the previous file intact. The
checksum detects corruption or modification but is not keyed, signed, or
evidence of authenticity. The current format is one monolithic JSON document
with a 256 MiB encoded-file cap. Large cohorts must be split across multiple
measurement sets rather than relying on sharding or external payloads.

**Offline UI modes:** **Window → Open Expression Measurement Set / Result…**,
the comparison window's open button, or `.aceexpr` drag/drop validates the
complete file before atomically registering an independent window. Schema-v2
sets show a **MEASUREMENT SET** notice, keep cell/channel/correction selectors
active, expose Add/Reload and both recomputation policies, and permit offline
CSV, SVG, and measurement-set saves. Fixed/cacheless files, including schema-v1
legacy captures, show **FROZEN RESULT**; their Add/Reload/Prepare and acquisition
controls are disabled, while
Use/label/group/color, time/grid/smoothing/statistics, appearance, CSV, SVG, and
presentation resave remain available within the captured request. Status-only
views may save/export CSV but have no SVG/toolbar image render. Numeric fixed
saved/mixed rows without recorded legacy-provenance acknowledgement remain
viewable but fail closed for CSV, SVG, and portable resave.

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

The 81 test modules in `tests/` cover the data model, I/O (including
interleaved multichannel TIFFs), naming, editing, GUI widgets, color rules,
CLI, analysis (including pixel measurement), tracking, and integration. The
2026-07-29 non-live verification completed with `1351 passed` and `71 skipped`;
run it with `pytest tests/`. Live MATLAB-oracle status is reported separately in
[StarryNite Differential Testing](STARRYNITE_DIFFERENTIAL_TESTING.md).

---

## 10. Build & Installation

```toml
# pyproject.toml
[project]
name = "acetree-py"
version = "0.2.0"
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

These commands install the current checkout. For the tracking-enabled build,
first use the branch-pinned clone and guarded installer in the
[Installation guide](../README.md#installation); a plain clone currently
selects the non-tracking default branch. Manual installs are
`python -m pip install -e .` (core), `python -m pip install -e ".[gui]"` (with
GUI), or `python -m pip install -e ".[all]"` (everything).

**napari version note:** The GUI uses napari's `Window.add_dock_widget()` and `Window._dock_widgets` APIs for panel management. These were tested against napari 0.5.x–0.6.x. The upper bound (`<0.7`) guards against breaking changes to these internal APIs.
