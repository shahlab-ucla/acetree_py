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

**Requirements:** Python 3.10+, numpy, scipy, tifffile, typer. GUI additionally requires napari and qtpy.

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
│     (napari canvas with nucleus overlay)            │
│                                                     │
├──────────┬──────────────────────┬───────────────────┤
│ Cell     │                      │ Contrast          │
│ Info     │                      │ (min/max sliders) │
│          │                      ├───────────────────┤
│ Lineage  │                      │ Edit Tools        │
│ List     │                      │ (buttons + history)│
├──────────┴──────────────────────┴───────────────────┤
│  Player Controls    │  Lineage Tree (Sulston view)  │
└─────────────────────┴───────────────────────────────┘
```

### Panels

| Panel               | Location    | Purpose                                        |
|---------------------|------------|------------------------------------------------|
| **Image Viewer**     | Center     | Shows the current z-plane with nucleus circles  |
| **Cell Info**        | Left       | Displays details of the selected cell           |
| **Lineage List**     | Left       | Searchable hierarchical cell tree               |
| **Player Controls**  | Bottom     | Time/plane navigation with play/pause           |
| **Lineage Tree**     | Bottom     | Visual Sulston tree with expression coloring    |
| **Contrast**         | Right      | Image brightness/contrast adjustment            |
| **Edit Tools**       | Right      | Editing operations, undo/redo, save             |

---

## 4. Navigation

### 4.1 Keyboard Shortcuts

| Key              | Action                    |
|------------------|---------------------------|
| `Right Arrow`    | Next timepoint            |
| `Left Arrow`     | Previous timepoint        |
| `Up Arrow`       | Next z-plane (up)         |
| `Down Arrow`     | Previous z-plane (down)   |
| `Ctrl+S`         | Save                      |
| `Ctrl+Shift+S`   | Save As                   |
| `Ctrl+Z`         | Undo                      |
| `Ctrl+Y`         | Redo                      |

### 4.2 Player Controls

The bottom panel provides full playback controls:

```
[⏮] [◀] [◀◀] [⏸] [▶▶] [▶] [⏭]  t= [___] / 350
[═══════════════ time slider ═══════════════════]
[▲] [▼]  z= [___] / 30
```

- **⏮ / ⏭**: Jump to first / last timepoint
- **◀ / ▶**: Step one timepoint back / forward
- **◀◀ / ▶▶**: Play backward / forward (animated)
- **⏸**: Pause playback
- Type directly into the spinboxes for precise navigation

### 4.3 Cell Tracking

When a cell is selected and **tracking** is enabled (the default), the viewer automatically:
- Follows the cell's z-position as you step through time
- Follows the first daughter cell when a division occurs
- Follows the parent cell when stepping backward past birth

---

## 5. Selecting Cells

### 5.1 In the Image Viewer

- **Right-click** on a nucleus circle or its label → **select that cell** (makes it the active cell; viewer centers on it, cell info updates)
- **Left-click** on a nucleus circle or its label → **toggle the label on/off** (useful for decluttering the display)

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
┌─ Edit Tools ─────────────┐
│ [Save] [Save As...]      │
│ [Undo] [Redo]            │
│                          │
│ Nucleus Operations       │
│ [Add] [Remove] [Move]    │
│                          │
│ Cell Operations          │
│ [Rename] [Kill] [Resurrect]│
│                          │
│ Link Operations          │
│ [Relink]                 │
│                          │
│ Status: Ready            │
│                          │
│ Edit History             │
│ ┌──────────────────────┐ │
│ │ 1. Renamed ABa → X  │ │
│ │ 2. Moved idx=3 ...   │ │
│ └──────────────────────┘ │
└──────────────────────────┘
```

### 6.2 Nucleus Operations

#### Add Nucleus
Click **Add** → fill in position (x, y, z), size, optional name, and optional predecessor → **OK**. A new nucleus is created at the specified timepoint.

#### Remove Nucleus
Select a cell, then click **Remove**. The selected nucleus is killed (marked dead). It remains in the data but is no longer displayed or tracked.

#### Move Nucleus
Select a cell, then click **Move**. A dialog shows the current position and lets you enter new coordinates and/or size. Only changed fields are updated.

### 6.3 Cell Operations

#### Rename
Select a cell, then click **Rename**. Enter a new name. This sets the `assigned_id` field, which is a manual override that persists through automatic re-naming.

#### Kill
Select a cell, then click **Kill**. Choose a time range. All nuclei of that cell within the range are marked dead.

#### Resurrect
Select a dead nucleus, then click **Resurrect**. The nucleus is restored to alive status.

### 6.4 Link Operations (Interactive Relink)

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

---

## 7. Contrast Adjustment

The **Contrast** panel on the right side provides:

- **Min/Max sliders**: Drag to adjust the display range
- **Auto**: Automatically compute optimal contrast from the current image
- **Reset**: Reset to full dynamic range

These affect only the display, not the underlying data.

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

1. **Founder identification**: Finds the 4-cell stage and identifies ABa, ABp, EMS, P2 using topology and timing.
2. **Back-tracing**: Names earlier cells (AB, P1, P0) by tracing predecessor links backward.
3. **Forward naming**: Names all subsequent cells by classifying each division using 3D geometry.

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
- **Expression coloring** = branches colored by GFP expression intensity (blue = low, warm colors = high)

### Controls

| Action              | Effect                                   |
|---------------------|------------------------------------------|
| Click branch        | Select cell                              |
| Mouse wheel         | Zoom in/out                              |
| Click + drag        | Pan                                      |
| **+** / **−**       | Zoom in / out (toolbar)                  |
| **Fit**             | Fit entire tree to view                  |
| **Export**           | Save tree as PNG or SVG image            |

---

## 12. Tips and Workflow

### Correcting a tracking error
1. Step through time until you see a cell jump or swap.
2. Select either the incorrectly tracked cell or the cell it should be linked to.
3. Click **Relink**.
4. Navigate to the other cell (can be earlier or later in time).
5. Right-click the other cell → confirm. The system automatically sorts by time and determines the predecessor/child relationship.

### Identifying unnamed cells
1. Look for gray circles in the image (unnamed cells are gray, named ones are purple).
2. Right-click to select, then check the Cell Info panel.
3. Use **Rename** to assign a name if you know the identity.

### Decluttering labels
- Left-click on any nucleus to toggle its label off. Left-click again to toggle it back on.
- This is useful when many cells overlap and labels are hard to read.

### Viewing a division
1. Select the parent cell.
2. Step forward in time until the division occurs.
3. On the frame after division, a yellow line connects the two daughters.
4. The viewer automatically follows the first daughter.

### Exporting for analysis
```bash
# Get a table of all cells with their lineage info:
acetree-py export config.xml -f cell_csv -o my_cells.csv

# Get per-nucleus data for custom analysis:
acetree-py export config.xml -f nucleus_csv -o my_nuclei.csv
```
