# AceTree-Py User Guide

A practical guide to opening datasets, navigating the viewer, editing cells, and saving your work.

---

## 1. Installation

```bash
# Core (CLI only, no GUI):
pip install -e .

# With napari GUI:
pip install -e ".[gui]"

# Everything (GUI + dev tools):
pip install -e ".[all]"
```

**Requirements:** Python 3.10+, numpy, scipy, tifffile, typer, matplotlib. GUI additionally requires napari (0.5–0.6.x) and qtpy.

**Tested versions:** napari 0.6.6, numpy 2.3, scipy 1.16, matplotlib 3.10, qtpy 2.4, Python 3.12.

---

## 2. Opening a Dataset

### 2.1 GUI Launch

```bash
# From the command line:
acetree-py gui path/to/config.xml
```

Or from Python:
```python
from acetree_py.gui.app import AceTreeApp

app = AceTreeApp.from_config("path/to/config.xml")
app.run()
```

### 2.2 Config File Format

AceTree-Py reads XML config files that point to your data:

```xml
<?xml version='1.0' encoding='utf-8'?>
<embryo>
    <nuclei file="path/to/nuclei.zip"/>
    <image file="path/to/typical_image.tif"/>
    <end index="350"/>
    <naming method="NEWCANONICAL"/>
    <resolution xyRes="0.09" zRes="1.0" planeEnd="30"/>
</embryo>
```

Key fields:
- `<nuclei file="..."/>` — Path to the nuclei ZIP file (required)
- `<image file="..."/>` — Path to a representative image (used to find the image directory)
- `<end index="N"/>` — Last timepoint to load
- `<naming method="NEWCANONICAL"/>` — Naming algorithm (`NEWCANONICAL` recommended)
- `<resolution xyRes="..." zRes="..." planeEnd="..."/>` — Physical resolution and z-planes
- `<axis axis="adl"/>` — Embryo orientation hint (optional, auto-detected if AuxInfo exists)

### 2.3 CLI Quick Look

```bash
# Print dataset summary without launching GUI:
acetree-py load config.xml

# Query a specific cell:
acetree-py info config.xml --cell ABala
```

---

## 3. The GUI Window

When you launch the GUI, you'll see:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│             Main Image Viewer                       │
│  (napari canvas with nucleus overlay + hover tips)  │
│                                                     │
├──────────┬──────────────────────┬───────────────────┤
│ Contrast │                      │ Edit Tools        │
│ (per-ch) │                      │ (color mode,      │
│          │                      │  buttons, viz)    │
│ Lineage  │                      │                   │
│ List     │                      │                   │
├──────────┴──────────────────────┴───────────────────┤
│  Player Controls (time/plane/labels/deselect/3D)    │
│  Lineage Tree (Sulston tree visualization)          │
└─────────────────────────────────────────────────────┘
```

### Panels

| Panel               | Location    | Purpose                                        |
|---------------------|------------|------------------------------------------------|
| **Image Viewer**     | Center     | Shows the current z-plane with nucleus circles; hover over a cell to see a tooltip |
| **Contrast**         | Left       | Per-channel brightness/contrast with visibility toggles |
| **Lineage List**     | Left       | Searchable hierarchical cell tree               |
| **Player Controls**  | Bottom     | Time/plane navigation, labels, deselect, 3D mode, 3D window |
| **Lineage Tree**     | Bottom     | Visual Sulston tree (multiple panels supported) |
| **Edit Tools**       | Right      | Color mode toggle, editing operations, visualization tools |

Napari's built-in layer list and layer controls are hidden by default to save screen space. They remain accessible via the napari Window menu.

---

## 4. Navigation

### 4.1 Keyboard Shortcuts

| Key              | Action                    |
|------------------|---------------------------|
| `Right Arrow`    | Next timepoint (follows tracked cell) |
| `Left Arrow`     | Previous timepoint        |
| `Up Arrow`       | Next z-plane (deselects active cell) |
| `Down Arrow`     | Previous z-plane (deselects active cell) |
| `Ctrl+S`         | Save                      |
| `Ctrl+Shift+S`   | Save As                   |
| `Ctrl+Z`         | Undo                      |
| `Ctrl+Y`         | Redo                      |
| `Delete`         | Remove active cell's nucleus at the current timepoint |
| `Escape`         | Exit active mode (Add, Track, Relink pick) |

**Note:** Changing z-plane deselects the active cell because the user is manually exploring rather than following a tracked cell. Time navigation continues to follow the tracked cell's centroid.

### 4.2 Player Controls

The bottom panel provides full playback controls:

```
[⏮] [◀] [◀◀] [⏸] [▶▶] [▶] [⏭]  t= [___] / 350
[═══════════════ time slider ═══════════════════]
[▲] [▼]  z= [___] / 30  [Labels: ON] [Clear Labels] [Deselect] [3D] [3D Window]
```

- **⏮ / ⏭**: Jump to first / last timepoint
- **◀ / ▶**: Step one timepoint back / forward
- **◀◀ / ▶▶**: Play backward / forward (animated)
- **⏸**: Pause playback
- **Labels: ON/OFF**: Toggle label display on all toggled cells
- **Clear Labels**: Remove all shown cell name labels
- **Deselect**: Clear the current cell selection and disable tracking
- **3D**: Toggle between 2D slice view and 3D volume rendering (see Section 6.6)
- **3D Window**: Open a detached 3D viewer window (see Section 6.7)
- Type directly into the spinboxes for precise navigation

### 4.3 Cell Tracking

When a cell is selected and **tracking** is enabled (the default), the viewer automatically:
- Follows the cell's z-position as you step through time
- Follows the first daughter cell when a division occurs
- Follows the parent cell when stepping backward past birth

---

## 5. Selecting Cells

### 5.1 In the Image Viewer

- **Right-click** on a nucleus circle → **select that cell** (makes it the active cell; viewer centers on it, cell info updates). The click must land within the drawn circle — clicking on empty space does nothing.
- **Left-click** on a nucleus circle → **toggle the label on/off** (useful for decluttering the display). Also requires clicking within the drawn circle.
- **Left-click** (in **Add mode**) → **place a new nucleus** at the click position (see Section 6.2)
- **Right-click** (in **Track mode**) → **place a tracking nucleus** at the click position (see Section 6.5)
- **Right-click** (in **Relink pick mode**) → **select relink target** (see Section 6.4)

### 5.2 In the Lineage Tree

- **Left-click** on a branch → select that cell and jump to the timepoint at the click's y-position
- **Right-click** on a branch → select that cell and jump to the **end** of its lifetime (useful for seeing divisions)
- **Mouse wheel** → zoom in/out
- **Click and drag** → pan the tree view
- **Fit button** → fit the entire tree into the view

### 5.3 In the Lineage List

- **Left-click** on a cell name → select it and jump to its start time
- **Right-click** on a cell name → select it and jump to its end time
- Use the **search box** at the top to filter by name

### 5.4 Division Line

When you step to the frame immediately after the selected cell divides, a **yellow line** briefly connects the two daughter cells in the image viewer. This makes it easy to see where daughters ended up. The line disappears automatically when you change time, z-plane, or selection.

---

## 6. Editing

All edits are **undoable** (`Ctrl+Z`) and **redoable** (`Ctrl+Y`). Up to 1000 edits are saved in the history.

### 6.1 Edit Panel Layout

```
┌─ Edit Tools ─────────────────┐
│ Color Mode                   │
│ (o) Editing  (o) Visualization│
│ Preset: [Lineage depth ▾]   │
│ [Edit Rules...]              │
│                              │
│ File: [Save] [Save As...]    │
│       [Undo] [Redo]          │
│                              │
│ Nucleus Operations           │
│ [Add] [Remove] [Move/Resize↗]│
│                              │
│ Cell Operations              │
│ [Rename] [Kill] [Resurrect]  │
│                              │
│ Link Operations              │
│ [Relink] [Track]             │
│                              │
│ Status: Ready                │
│                              │
│ Visualization                │
│ [Trails] Length: [10]        │
│ [Screenshot] [Record...]     │
│                              │
│ [Edit History...]            │
└──────────────────────────────┘
```

The Move/Resize D-pad controls are now in a popup dialog (click **Move / Resize** to open). Edit History is also a popup window. This keeps the edit panel compact.

### 6.2 Nucleus Operations

#### Add Nucleus (Interactive Click-to-Add)

The **Add** button is a toggle that activates click-to-add mode:

1. (Optional) Select an existing cell to use as the predecessor.
2. Click **Add** — the button stays pressed and the status bar shows instructions.
3. **Left-click** anywhere in the image viewer to place a nucleus at that position, at the current z-plane and timepoint.
4. Press **Esc** or click **Add** again to exit add mode.

**Predecessor linking:**
- If a cell is selected when you click, the new nucleus inherits the selected cell's identity and is linked as its successor.
- If the selected cell's last timepoint is adjacent (gap = 1), a direct predecessor link is made.
- If there is a gap > 1 timepoint, the system automatically interpolates intermediate nuclei to fill the gap.
- If no cell is selected, a new independent root nucleus is created.

**Inherited properties:** When adding from an existing cell, the new nucleus inherits the parent cell's diameter (size). Root nuclei use the default diameter (20 pixels).

#### Remove Nucleus
Select a cell, then click **Remove** (or press **Delete**). The selected nucleus at the current timepoint is killed (marked dead). It remains in the data but is no longer displayed or tracked. The **Delete** key acts immediately without a confirmation dialog; the **Remove** button shows a confirmation prompt first.

### 6.3 Move / Resize (D-Pad Controls)

The Move / Resize group provides instant nudge buttons for adjusting the selected nucleus's position and size without opening a dialog:

- **XY arrows** (`← → ↑ ↓`): Move the nucleus by 1 or 5 pixels in each direction.
- **Z** (`-5`, `-1`, `+1`, `+5`): Shift the nucleus up or down in z-planes.
- **Size** (`-5`, `-1`, `+1`, `+5`): Increase or decrease the nucleus diameter.

Each button press executes immediately and is individually undoable with `Ctrl+Z`. The cell remains selected between presses, so you can rapidly adjust position by clicking multiple times. The status bar shows the delta applied (e.g. "Moved: x+5, y-1").

### 6.4 Cell Operations

#### Rename
Select a cell, then click **Rename**. Enter a new name. This sets the `assigned_id` field, which is a manual override that persists through automatic re-naming.

#### Kill
Select a cell, then click **Kill**. Choose a time range. All nuclei of that cell within the range are marked dead.

#### Resurrect
Select a dead nucleus, then click **Resurrect**. The nucleus is restored to alive status.

### 6.5 Link Operations

#### Interactive Relink

The relink operation lets you change which cell a nucleus is linked to as its predecessor. This is the primary tool for correcting tracking errors.

**How to relink:**

1. Select **either** cell in the pair you want to link (the order doesn't matter — the system automatically determines which is earlier/later).
2. Click **Relink**.
3. The status bar shows **"PICK MODE"** — the system is waiting for you to choose the other cell.
4. Navigate through time and z-planes to find the cell you want to link to.
5. **Right-click** on the target cell.
6. A confirmation dialog appears showing source and target details.

**Bidirectional:** You can pick the earlier cell first and then the later cell, or vice versa. The system sorts the two cells by timepoint — the earlier one becomes the predecessor, the later one becomes the child.

**Automatic interpolation:** If the two cells are more than 1 timepoint apart, the system automatically creates interpolated nuclei to fill the gap. This is required by the data format — every cell must have a continuous chain of nuclei across consecutive timepoints. The interpolated nuclei are placed at linearly interpolated positions and sizes between the two endpoints.

**Adjacent links (gap = 1 frame):** A simple predecessor change is made, no interpolation needed.

#### Track Mode

The **Track** button enables continuous click-to-place tracking across timepoints:

1. Select a cell to track from (the parent).
2. Click **Track** — the button stays pressed.
3. Navigate to a later timepoint.
4. **Right-click** in the viewer to place a nucleus. It is automatically linked to the parent cell with the correct identity and predecessor.
5. If there is a time gap > 1, intermediate nuclei are interpolated.
6. The mode stays active so you can advance to the next timepoint and place again.
7. Press **Esc** or click **Track** again to exit.

If no cell is selected, Track enters root mode: a single right-click places one independent nucleus and exits.

### 6.6 3D Volume View

Toggle the **3D** button in the player controls to switch between 2D slice view and 3D volume rendering.

In 3D mode, all nuclei at the current timepoint are displayed as colored spheres with correct anisotropic scaling (z-spacing accounts for the physical z-resolution). The color scheme depends on the active color mode:

**Editing mode (default):**
| Color    | Meaning                                      |
|----------|----------------------------------------------|
| White    | Currently selected cell                      |
| Purple   | Named cell (Sulston name assigned)           |
| Orange   | Unnamed cell (auto-generated `Nuc*` name)    |
| Gray     | No name / placeholder                        |

**Visualization mode:** Colors are determined by the active color rules (see Section 6.8).

All image channels are loaded as 3D stacks when entering 3D mode. Clicking on a sphere selects the corresponding cell. Relink pick mode and track mode also work in 3D.

### 6.7 Detached 3D Viewer Window

Click **3D Window** in the player controls to open a separate 3D viewer window. This window is designed for visualization and always uses rule-engine coloring (visualization mode), regardless of the main viewer's color mode. This allows you to edit in 2D in the main viewer while simultaneously viewing the embryo in 3D.

**Controls:**
- **Time slider + Sync button**: When Sync is on (default), the 3D window follows the main viewer's timepoint. Toggle off to navigate independently.
- **Color Preset dropdown**: Switch between visualization presets (lineage depth, expression).
- **Per-channel contrast**: Each image channel has visibility checkbox, min/max sliders, and auto/reset buttons.
- **Labels: ON/OFF**: Toggle label visibility globally.
- **Clear Labels**: Remove all shown labels.
- **Left-click** on a 3D sphere: Toggle that cell's label on/off.

Multiple 3D windows can be open simultaneously.

### 6.8 Color Mode and Visualization Rules

The Edit Panel provides a **Color Mode** toggle at the top:

- **Editing** (default): Uses the hardcoded status-based palette (white/purple/orange/gray).
- **Visualization**: Uses a rule-based color engine. Select a preset from the dropdown, or click **Edit Rules...** to open the full rule editor.

**Color Rules dialog** (Edit Rules...):
- Lists all active rules with enable/disable checkboxes.
- **Add / Edit / Delete**: Manage individual rules. Double-click a rule to edit.
- **Up / Down arrows**: Reorder rules (first matching rule wins).
- **All other cells**: Configure the default color for cells that don't match any rule (white semi-transparent by default).
- **Apply**: Push rules to the engine and re-render.

**Rule editor** (per rule):
- **Name**: Human-readable label.
- **Match** (criterion): What property to test. Click the **?** button for help on each mode:
  - `all` — matches every cell
  - `name_exact` — exact cell name (e.g. `ABala`)
  - `name_pattern` — wildcard glob (e.g. `AB*`, `MS?`)
  - `name_regex` — regular expression (e.g. `^AB[ap]$`)
  - `lineage_depth` — depth range from P0 (e.g. `2-4`)
  - `fate` — end fate (`divided`, `alive`, `died`)
  - `expression` — rweight value range (e.g. `500-2000`)
- **Pattern**: The match value (depends on criterion).
- **Color mode**: Solid (fixed color with alpha) or Colormap (map expression through a matplotlib colormap).

**Built-in presets:**
- *Lineage depth (rainbow)*: Rainbow colors by division depth (0-10).
- *Expression (viridis)*: Map rweight through the viridis colormap.

---

## 7. Contrast Adjustment

The **Contrast** panel on the left side provides per-channel controls. For multi-channel data (e.g. split-channel dual-color images), each channel gets its own control group:

- **Visible checkbox**: Toggle channel visibility (multi-channel only)
- **Min/Max sliders**: Drag to adjust the display range
- **Auto**: Automatically compute optimal contrast from the current image data (1st/99th percentile)
- **Reset**: Reset to full dynamic range (0–65535)
- **Auto All / Reset All**: Apply to all channels at once

For single-channel data, a simplified layout without the visibility checkbox is shown.

Multi-channel images are displayed as separate napari layers with green/magenta colormaps (standard fluorescence convention) and additive blending.

---

## 8. Saving

### 8.1 Save / Save As

- **Ctrl+S** or the **Save** button: Overwrites the original nuclei ZIP file.
- **Ctrl+Shift+S** or **Save As**: Opens a file dialog to choose a new location.

The saved file is a ZIP containing CSV-formatted nucleus data, one entry per timepoint. This is the standard AceTree nuclei format and can be opened by both AceTree-Py and the original Java AceTree.

### 8.2 What Gets Saved

- All nucleus positions, sizes, and names
- All predecessor/successor links
- Manual name overrides (`assigned_id`)
- Expression values

Edits that haven't been saved are tracked — the Edit Tools panel will show unsaved state.

---

## 9. Exporting Data

Use the CLI to export data in various formats:

```bash
# Cell-level CSV (name, lifetime, fate, parent, children):
acetree-py export config.xml --format cell_csv --output cells.csv

# Nucleus-level CSV (all fields for every nucleus at every timepoint):
acetree-py export config.xml --format nucleus_csv --output nuclei.csv

# Expression time series CSV:
acetree-py export config.xml --format expression_csv --output expression.csv

# Newick tree format (for phylogenetic tools):
acetree-py export config.xml --format newick --output tree.nwk
```

If `--output` is omitted, the output filename is automatically derived from the config filename.

---

## 10. Understanding Cell Names

### 10.1 Automatic Naming

When a dataset is loaded, the naming pipeline automatically identifies cells:

1. **Founder identification**: Finds the 4-cell stage and identifies ABa, ABp, EMS, P2 using topology and timing. ABa/ABp are distinguished by projection onto the anterior-posterior axis (not image coordinates), making this step robust to arbitrary embryo orientations.
2. **Back-tracing**: Names earlier cells (AB, P1, P0) by tracing predecessor links backward.
3. **Forward naming**: Names all subsequent cells by classifying each division using 3D geometry. The body axes (AP, LR, DV) are re-derived at every timepoint from lineage centroid positions, making naming robust to embryo rotations during imaging.

If AuxInfo orientation data is available (v1 or v2), it is used for higher-precision axis estimation. Without AuxInfo, the pipeline uses the rotation-invariant lineage centroid approach described above.

### 10.2 Unnamed Cells

Cells that can't be automatically named receive placeholder names like `Nuc042_15_200_300` (3-digit zero-padded timepoint, then z, x, y). These are typically polar bodies or cells at the edges of the tracked lineage.

### 10.3 Manual Overrides

When you **Rename** a cell (Section 6.3), the name is stored as a permanent override (`assigned_id`). This name survives automatic re-naming — even if you save and reload the dataset, the override persists.

---

## 11. Lineage Tree View

The Sulston lineage tree at the bottom of the window shows the full cell lineage:

- **Vertical lines** = cell lifetimes (branches)
- **Horizontal connectors** = cell divisions (mother splits into two daughters)
- **Yellow dashed line** = current timepoint indicator
- **Yellow highlighting** = currently selected cell
- **Expression coloring** = branches colored by GFP expression intensity (configurable colormap)

### Controls

| Action              | Effect                                   |
|---------------------|------------------------------------------|
| Click branch        | Select cell                              |
| Mouse wheel         | Zoom in/out                              |
| Click + drag        | Pan                                      |
| **+** / **−**       | Zoom in / out (toolbar)                  |
| **Settings**        | Configure panel display settings         |
| **Fit**             | Fit entire tree to view                  |
| **Export**           | Save tree as PNG or SVG image            |

### Multiple Lineage Panels

You can open multiple lineage tree panels, each showing a different subtree or using different display settings:

1. **Window > New Lineage Panel...** — opens a configuration dialog to create a new panel.
2. Each panel's **Settings** button lets you reconfigure it at any time.

**Panel settings:**
- **Root cell** — choose which cell to use as the tree root (e.g. "ABa" to see only ABa's descendants). Set to "(auto-detect)" for the full tree.
- **Time range** — restrict the display to a window of timepoints.
- **Expression range** — set min/max values for expression color mapping.
- **Colormap** — choose from matplotlib colormaps (viridis, plasma, inferno, hot, coolwarm, etc.) or the legacy green-to-red gradient.

All open panels update synchronously when edits are committed (relink, kill, rename, etc.).

---

## 12. Tips and Workflow

### Correcting a tracking error
1. Step through time until you see a cell jump or swap.
2. Select either the incorrectly tracked cell or the cell it should be linked to.
3. Click **Relink**.
4. Navigate to the other cell (can be earlier or later in time).
5. Right-click the other cell → confirm. The system automatically sorts by time and determines the predecessor/child relationship.

### Identifying unnamed cells
1. Look for **orange** circles in the image (unnamed `Nuc*` cells are orange, named ones are purple, gray indicates no name at all). Or switch to visualization mode with the lineage depth preset for a rainbow view.
2. Right-click to select, then hover over the cell to see the tooltip.
3. Use **Rename** to assign a name if you know the identity.

### Decluttering labels
- Left-click on any nucleus to toggle its label off. Left-click again to toggle it back on.
- This is useful when many cells overlap and labels are hard to read.

### Viewing a division
1. Select the parent cell.
2. Step forward in time until the division occurs.
3. On the frame after division, a yellow line connects the two daughters.
4. The viewer automatically follows the first daughter.

### Using the 3D viewer alongside editing
1. Open the main viewer in 2D editing mode as usual.
2. Click **3D Window** in the player controls to open a synced 3D view.
3. Edit in the main viewer — the 3D window updates in real time.
4. Use the 3D window's color preset dropdown to switch between lineage depth and expression views independently of the main viewer.

### Screenshots and recording
- Click **Screenshot** in the Visualization section of Edit Tools to capture the current view as a PNG.
- Click **Record...** to export a sequence of PNGs across a timepoint range (useful for making movies).

### Exporting for analysis
```bash
# Get a table of all cells with their lineage info:
acetree-py export config.xml -f cell_csv -o my_cells.csv

# Get per-nucleus data for custom analysis:
acetree-py export config.xml -f nucleus_csv -o my_nuclei.csv
```

---

## 13. Manual Tracking & Dataset Creation

AceTree-Py can create new datasets from raw TIFF images and provides interactive tools for manually placing, tracking, and linking nuclei. This is useful when:

- You have image data that hasn't been processed by StarryNite or another detection pipeline.
- You want to manually annotate nuclei positions in a single frame (detection-only, no tracking).
- You want to manually track a subset of cells across time.

### 13.1 Creating a New Dataset

#### Interactive Wizard (GUI)

```bash
acetree-py create
```

This opens a 4-page wizard dialog:

1. **Image directory** — select the folder containing your TIFF files. The wizard auto-detects the naming pattern and image dimensions.
2. **Channel layout** — choose how channels are arranged: single channel, side-by-side (split), separate directories, or multichannel stack. Set flip if needed.
3. **Voxel parameters** — set XY resolution (µm/pixel), Z resolution (µm/plane), number of timepoints and planes (auto-filled from detection).
4. **Output** — choose where to save the dataset config XML and nuclei ZIP.

#### CLI (Non-Interactive)

```bash
acetree-py create <image_directory> [OPTIONS]
```

Options:

| Option         | Default | Description                              |
|----------------|---------|------------------------------------------|
| `--output`     | auto    | Output directory for config + nuclei ZIP |
| `--xy-res`     | 0.09    | XY pixel resolution in µm               |
| `--z-res`      | 1.0     | Z plane spacing in µm                   |
| `--split`      | off     | Split side-by-side dual-channel images   |
| `--flip`       | off     | Flip images left/right                   |

**Examples:**

```bash
# Create dataset from a folder of TIFFs with default resolution:
acetree-py create /data/embryo/images/

# With specific resolution and split channels:
acetree-py create /data/embryo/SPIMA/ --output /data/embryo/manual_output/ --xy-res 0.1625 --z-res 0.65 --split

# Single-frame annotation (one TIFF file in the directory):
acetree-py create /data/single_frame/
```

The `create` command:
1. Scans the image directory for TIFF files and probes the first image for z-plane count.
2. Generates an `AceTreeConfig` with the correct image paths and resolution.
3. Creates an empty nuclei ZIP (no detections).
4. Writes the XML config file to the output directory.
5. Launches the GUI for interactive annotation.

### 13.2 Placing Nuclei (Add Mode)

Once the GUI is open on a new (empty) dataset:

1. Navigate to the desired timepoint and z-plane.
2. Click **Add** in the Edit Tools panel to enter add mode.
3. **Left-click** anywhere in the image to place a nucleus at that position.
4. The nucleus is created at the current z-plane with a default diameter of 20 pixels.
5. Press **Esc** to exit add mode.

**Adding onto an existing cell:**
1. Right-click an existing nucleus to select its cell.
2. Navigate to a later timepoint.
3. Click **Add**, then left-click to place. The new nucleus inherits the selected cell's identity, predecessor link, and diameter.
4. If there is a gap > 1 timepoint, intermediate nuclei are automatically interpolated.

### 13.3 Adjusting Nuclei (D-Pad Controls)

After placing a nucleus, use the **Move / Resize** D-pad buttons to fine-tune:

- **XY arrows**: nudge position by 1 or 5 pixels.
- **Z buttons**: shift the z-plane by 1 or 5.
- **Size buttons**: grow or shrink the diameter by 1 or 5 pixels.

Each press is individually undoable. The cell stays selected between presses for rapid adjustment.

### 13.4 Tracking Across Time

For continuous tracking across many timepoints, use the **Track** button:

1. Select the cell you want to extend.
2. Click **Track** to enter tracking mode.
3. Advance to the next timepoint (right arrow).
4. **Right-click** to place the next position. The nucleus is automatically linked.
5. Repeat steps 3–4 for as many timepoints as needed.
6. Press **Esc** to exit tracking mode.

Track mode automatically handles:
- **Identity inheritance**: each placed nucleus gets the parent cell's name.
- **Predecessor linking**: direct link if adjacent, interpolation if there's a gap.
- **Size inheritance**: the placed nucleus inherits the parent's diameter.

### 13.5 Workflow for Single-Frame Annotation

For annotating nuclei in a single image (no tracking):

```bash
# Create a dataset with the single image:
acetree-py create /data/single_frame/
```

1. Use **Add** mode to left-click on each nucleus.
2. Use the D-pad controls to adjust positions and sizes.
3. Use **Rename** to assign cell identities.
4. **Save** (`Ctrl+S`) to persist annotations.

Each placed nucleus becomes an independent root cell. The Track button is not useful in single-frame mode (there are no future timepoints to track to).

### 13.6 Saving and Reloading

After annotation:

- **Save** (`Ctrl+S`) writes the nuclei to the ZIP file and the config XML.
- To reopen later: `acetree-py gui path/to/output/config.xml`
- The automatic naming pipeline runs on load. If enough cells have been placed for the 4→8 cell transition to be detected, Sulston names will be assigned automatically.

### 13.7 Tips for Manual Tracking

- **Use the 3D view** (Section 6.6) to verify nucleus positions in three dimensions.
- **Place nuclei on the z-plane where the nucleus is brightest** for the most accurate position.
- **Use Track mode** for long cell tracks — it's much faster than individual Add operations.
- **Use Relink** to correct mistakes after the fact rather than undoing many steps.
- **Save frequently** (`Ctrl+S`) — there is no autosave.
