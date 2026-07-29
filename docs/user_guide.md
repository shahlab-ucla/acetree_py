# AceTree-Py User Guide

A practical guide to opening datasets, navigating the viewer, editing cells, and saving your work.

For automatic-versus-forced names and the complete correction contract, see [Naming and Manual-Curation Workflows](naming_workflows.md). For detector/tracker extension points and the longer-term StarryNite plan, see the [Tracking Pipeline Specification](TRACKING_PIPELINE_SPEC.md).

---

## 1. Installation

The tracking-enabled build is currently on `tracking-integration`, while the
repository default is still `main`. Select the branch explicitly before using
an editable install:

```bash
git clone --branch tracking-integration --single-branch https://github.com/shahlab-ucla/acetree_py.git
cd acetree_py
```

For the recommended GUI install, run the branch guard for your platform:

```powershell
# Windows PowerShell
.\scripts\install_tracking_integration.ps1
```

```bash
# macOS or Linux
sh scripts/install_tracking_integration.sh
```

The scripts accept `core`, `gui` (the default), or `all` through
`-Variant` on PowerShell and `--variant` on macOS/Linux. To install manually
from the checked-out branch instead, use the commands below. If needed, select
the interpreter with `-Python py` or `--python /path/to/python3`.

```bash
# Core (CLI only, no GUI)
python -m pip install -e .

# With napari GUI
python -m pip install -e ".[gui]"

# Everything (GUI + dev tools)
python -m pip install -e ".[all]"

# Must include "(tracking integration)"
python -m acetree_py --version
```

**Requirements:** Python 3.10+, Git, numpy, scipy, tifffile, typer, matplotlib. GUI additionally requires napari (0.5–0.6.x) and qtpy.

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
│ Contrast │                      │ Edit & Tracking   │
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
| **Edit & Tracking Tools** | Right | Manual/automated tracking, color mode, editing, and visualization tools |

Napari's built-in layer list and layer controls are hidden by default to save screen space. They remain accessible via the napari Window menu.

---

## 4. Navigation

### 4.1 Keyboard Shortcuts

| Key              | Action                    |
|------------------|---------------------------|
| `Right Arrow`    | Next timepoint (follows tracked cell) |
| `Left Arrow`     | Previous timepoint        |
| `Up Arrow`       | Next z-plane              |
| `Down Arrow`     | Previous z-plane          |
| `Ctrl+S`         | Save                      |
| `Ctrl+Shift+S`   | Save As                   |
| `Ctrl+Z`         | Undo                      |
| `Ctrl+Y`         | Redo                      |
| `Delete`         | Remove active cell's nucleus at the current timepoint |
| `Escape`         | Exit active mode (Add, Track, Relink pick) |

Changing z-plane does **not** clear the active cell. Selection is anchored by timepoint and nucleus index, so it remains stable across display refreshes and automatic renaming. Use **Deselect** when you intentionally want to clear it.

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
- **Right-click** (in **Manual Track mode**) → **place a tracking nucleus** at the click position (see Section 6.5)
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
┌─ Edit & Tracking Tools ──────┐
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
│ [Relink] [Manual Track]      │
│ [Track Selected Cell...]     │
│                              │
│ Body Orientation             │
│ Endpoint: [Anterior ▾]       │
│ [Label selected] [Apply axes]│
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
- If a cell is selected when you click, the new nucleus is linked as its successor. A continuation may carry the current automatic identity, but an automatic name is never converted into a forced override. `assigned_id` is inherited only when the parent was already manually forced.
- If the selected cell's last timepoint is adjacent (gap = 1), a direct predecessor link is made.
- If there is a gap > 1 timepoint, the system automatically interpolates intermediate nuclei to fill the gap. Placement, interpolation, and linking are one action, so one Undo removes the entire gesture.
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
Select a cell, then click **Rename**. Enter a new name.

The rename is **cell-scoped**: the forced name (`assigned_id`) is written atomically onto *every nucleus in the cell's continuation chain* — from the cell's birth (previous division or first appearance) through its next division or disappearance. You don't need to rename at a specific timepoint; clicking at any point in the cell's lifetime produces the same result.

The forced name persists through automatic re-naming on reload. When the cell divides, the forced name is used as the parent name for Sulston daughter naming rules. Undo restores the previous `identity` and `assigned_id` on every nucleus the rename touched.

If the displayed name is already correct, closing or accepting Rename without a change does nothing and adds no undo entry. Choose **Use Automatic** to remove the forced override from the cell continuation. AceTree then recomputes its automatic identity; Undo restores the override.

**Name collisions:** If the target name is already in use by a different cell, AceTree offers a **Swap** — pressing "Swap" runs the `SwapCellNames` command, which atomically exchanges the forced names of the two cells (writes cell B's effective name onto every nucleus in cell A's chain, and vice versa). This is the same one-step undo.

#### Kill
Select a cell, then click **Kill**. Choose a time range. All nuclei of that cell within the range are marked dead.

#### Resurrect
Click **Resurrect** to choose from dead nuclei at the current timepoint (dead records are normally hidden from the overlay). If an explicitly anchored dead nucleus is available, it is used directly. A live selection is never silently toggled; the command rejects it or offers the dead-record chooser. Resurrection restores the requested automatic/forced name state and is undoable.

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

The link must be alive, reciprocal, and unambiguous. AceTree rejects a relink that would merge two different forced identities instead of silently choosing whichever name it encounters first. Relinking and all interpolation are a single undoable action. **Escape** exits pick mode and restores the normal selection controls.

**Adjacent links (gap = 1 frame):** A simple predecessor change is made, no interpolation needed.

#### Manual Track Mode

The **Manual Track** button enables continuous click-to-place tracking across timepoints:

1. Select a cell to track from (the parent).
2. Click **Manual Track** — the button stays pressed.
3. Navigate to a later timepoint.
4. **Right-click** in the viewer to place a nucleus. It is automatically linked to the parent cell. Existing manual override state is preserved; a merely automatic parent name is not locked.
5. If there is a time gap > 1, intermediate nuclei are interpolated.
6. The mode stays active so you can advance to the next timepoint and place again.
7. Press **Esc** or click **Manual Track** again to exit.

If no cell is selected, Manual Track enters root mode: a single right-click places one independent nucleus and exits.

When a placement creates a second daughter, AceTree evaluates the division rule for the **actual parent**. The preview can therefore propose `ABal/ABar`, `E/MS`, `C/P3`, or other parent-specific pairs rather than always proposing `a/p`. The preview reports its axis, confidence, and orientation source. Suggested names remain automatic (`identity`); use Rename only when you intend to force a correction. Low-confidence or ambiguous geometry is an invitation to inspect the body axes and daughter positions.

#### Track Selected Cell Workbench

**Tracking > Track Selected Cell Forward…** (also in **Edit & Tracking Tools**) follows one selected cell without detecting or replacing the rest of the embryo. It is available after manual initialization and after opening an existing XML dataset:

1. Select a live cell and click **Track Selected Cell**. If the selection is earlier in an existing one-child continuation, AceTree safely starts from its terminal nucleus. It never chooses a daughter at a division.
2. In **1. Configure**, choose **Modern StarryNite (recommended)**, **LoG detection + LAP tracking**, or **DoG detection + LAP tracking**. Modern StarryNite also offers six bundled imaging presets. The guided surface starts with a short range of up to ten future frames and keeps the channel, radius, threshold, end time, and division behavior visible. Component selection, local-search tuning, gap/ambiguity controls, and custom parameter/model tools are hidden under **Show advanced and custom settings** by default.
3. For Modern StarryNite, choose the embryo's actual **Developmental stage**. **Automatic from annotated cells** remains available, but a sparse lineage contains far fewer annotations than the embryo contains cells; an explicit stage prevents that sparse count from selecting an inappropriately early entry in a staged legacy parameter array.
4. Use **Test Next Frame** for a fast one-frame check of the local detection and link. Its overlay and review table are diagnostic only: it cannot be accepted. Tune the visible controls if needed, then choose **Build Preview** to analyze the requested range. Analysis runs in the background, progress is reported by frame, and **Cancel analysis** stops cooperatively without retaining a partial result or changing the dataset. Closing the window during analysis requests cancellation before its worker is released.
5. In **2. Review**, inspect every proposed and interpolated position in the table. The overlay uses both shape and color: circles are proposed detections, diamonds are interpolated gaps, square/cross marks are diagnostic candidates that will not be accepted, and paths show movement. When a run stops, a ring/crosshair in 2D or wireframe search sphere in 3D shows the predicted search region. Click or keyboard-select a row, use **Previous**, **Next**, **Play Draft**, or **Go to stop**, and optionally center the camera on the selected position.
6. If the draft needs work, change any parameter. AceTree marks the old overlay as out of date and disables acceptance until **Update Preview** finishes. Native selected-cell settings, including the relative tracking horizon, workflow/preset, explicit stage, visible tuning values, division policy, review options, and Advanced-panel state, are remembered across application sessions. Because the horizon is stored as a length rather than an absolute time, it is reapplied safely when the next selected cell starts at another frame.
7. Read the human-language stopping explanation. Selected-cell tracking stops rather than guessing at similarly likely candidates, likely divisions, conflicts with existing annotations, or a lost continuation. If a stop identifies a likely division while the conservative stop policy is active, **Rerun Following Both Daughters** switches to the two-daughter policy and rebuilds the draft. The stopped frame, last accepted frame, predicted location, search radius, and review-only candidates are structured proposal data and remain available after save/reload of the tracking sidecar.
8. Choose **Accept Draft** to add the complete visible proposal. To keep only an earlier reliable prefix, select an ordinary proposed-detection row and choose **Accept through selected frame**; diagnostic candidates and interpolated rows are not valid cutoff points. Either action applies exactly the displayed full draft or trimmed prefix as one undoable edit.
9. Acceptance keeps the workbench open, clears the draft overlay, and selects the new terminal nucleus when it is unique. Choose **Undo Accepted Draft** to reverse that just-applied history command and return to configuration, **Save Dataset** to persist the accepted nuclei and tracking provenance, or **Close** to leave the accepted but possibly unsaved document open in AceTree.

Use **Solo detection channel while reviewing** when other fluorescence channels obscure the detector input; it applies to the main and all open detached viewers, then restores each layer's prior visibility on close. Draft layers remain read-only and visible in editing colors, visualization-rule colors, the main 3D volume, and every detached 3D window. Detached windows honor their own timepoint when Sync is off, and changing a review highlight updates only draft layers rather than rereading the image stacks.

Simple LAP supports one-to-one continuations and short gaps. It does **not** create divisions or merges. The StarryNite division tracker can stop at a likely division, follow the best daughter, or include both daughters in the review draft; merges remain disabled. When a conservative run stops at a division, the dedicated **Rerun Following Both Daughters** action avoids a trip back through Advanced settings. A manual edit, Undo, or Redo while an uncommitted draft is open makes that draft stale and requires **Update Preview** before it can be accepted.

The default imaging preset is bundled and needs no conversion. Under **Show advanced and custom settings**, **Use another parameter file...** safely reads an external legacy file and fills the editable controls. Confirm or override its staged values with **Developmental stage** instead of assuming that the sparse annotated-cell count represents the whole embryo. **Save tuned parameter copy...** preserves the original text and comments while appending compatible edits. The copy becomes active immediately, and the most recently selected external file is offered the next time either tracking workbench opens.

Selected-cell tracking always uses the native StarryNite division tracker; the global
legacy-exact backend is intentionally unavailable for a selected-cell scope.
Its moving **Search ROI** is distinct from the fixed, one-based camera ROI in a
legacy parameter file. The fixed ROI and distribution-backed exact detector are
reserved for whole-movie replay; sparse tracking keeps the preset's staged native
detector values without activating those global exact inputs.

**Validate classifier export (report only)…** can confirm that a numeric export
belongs to the referenced MAT model, but does not execute that classifier or
change the sparse draft. Choose **Stop and review likely divisions**, **Follow
the best daughter only**, or **Follow both daughters** according to the curation
goal. Following both daughters adds exactly two reviewed branches when the
tracker supports splitting; it does not start tracking the rest of the embryo.

#### Track Whole Movie Workbench

Choosing automated tracking in the dataset wizard creates the empty ZIP/XML first, launches the viewer, and opens **Review Initial Tracking Draft**. The wizard defaults to **Modern StarryNite (recommended)** and exposes the same LoG+LAP and DoG+LAP alternatives. Modern StarryNite offers every upstream `newmatlab` imaging preset and defaults to reviewed two-daughter proposals; LAP remains continuation-only.

The whole-movie workbench uses the bundled presets directly. External parameter loading, the remembered recent file, tuned-copy saving, and alternate legacy-model selection remain available under **Advanced and custom settings**.

After loading a StarryNite file, keep tuning the visible controls normally and
open **Compatibility details…** to see the selected stage, effective values,
model identity, and any fail-closed limitations. In this whole-dataset
workbench, the bundled tracking model is attached automatically and checked
against the exact MAT source hash. **Use another legacy model...** accepts both
the `.atpy-model` format and older JSON exports for backward compatibility.
Editing or replacing a parameter, distribution, model, or runtime-model source
invalidates the corresponding identity check and requires fresh validation.

For MATLAB-equivalent whole-movie processing, select **Legacy StarryNite exact
replay (advanced)** and choose the matching bundled imaging preset. Exact mode
automatically pairs the StarryNite detector and ready source-bound model, requires time 1
as the start, uses the parameter-file calibration, enables divisions, and runs
sequential detection followed by every staged geometry and classifier cleanup
pass. It never falls back to the native scorer. Missing source fields, changed
hashes, unsupported statements, non-unit downsampling, calibration mismatch,
native detector overrides, or legacy options whose raw measurements are not
available block the draft with an actionable message. Save visible tuning
changes to a copy and reload that copy before starting an exact run.

Use **Test Detector at t=N** while viewing a representative frame. Purple read-only rings show the detector candidates in 2D, main 3D, and synced detached 3D windows. Adjust channel, radius, threshold, subpixel localization, or median filtering and retest; changing only the tracker, gap, displacement, or requested time range does not invalidate a detector test. Moving to another timepoint clears the transient rings so a result cannot be mistaken for the new current frame. A successful zero-candidate test is still useful feedback. In exact mode, a standalone test is available only at time 1: later detection depends on the preceding frame's final count and candidate-diameter distribution, so **Build Full Draft** is required to warm that history correctly.

When the detector looks sane on representative frames, choose **Build Full Draft**. That clears the transient detector-test layer, runs detection across the requested time range, and invokes the selected tracker to build links and any enabled divisions. Only this full result can enable **Accept Draft**. Cancellation, failure, closing the workbench, or a document/time change during a detector test retains no partial overlay and never changes nuclei, undo history, AuxInfo, or the tracking sidecar.

Before creation, the wizard verifies every requested channel source, multichannel stack divisibility, output folder, and dataset name. If the target ZIP or XML already exists, AceTree asks before replacing it and defaults to keeping the existing files.

The per-frame table appears only for a full draft and includes every requested frame, including frames with no detections, interpolated gap positions, new track starts, mean quality, and warnings. Select rows by mouse or keyboard while inspecting the same read-only overlay in 2D, main 3D, or detached 3D. Change settings and choose **Update Full Draft** as often as needed; the previous overlay is visibly stale and cannot be accepted after a setting or document change. **Accept Draft** is the only action that adds positions, as one undoable edit. **Discard Draft**, cancellation, failure, or closing the window leaves the dataset empty.

For discoverability, **Tracking > Track Whole Movie...** and **Edit & Tracking Tools > Tracking & Links > Track Whole Movie...** reopen this workbench while the dataset is still empty. Once any nucleus record exists, use **Track Selected Cell...** for a selected lineage or Undo the accepted initial draft before rerunning whole-movie tracking. This restriction prevents an embryo-wide run from duplicating curated nuclei.

#### Real-world StarryNite test checklist

For a first embryo, keep the original MATLAB run and its outputs unchanged so
the accepted AceTree draft can be compared independently. Exact global tracking
currently needs all of the following before it reads frame 1:

- an empty AceTree nuclei record covering the complete requested movie;
- image timepoints beginning at time 1, with the intended detection channel
  available at every requested frame;
- dataset XY and Z calibration equal to the values resolved from the parameter
  file, and XY downsampling equal to 1;
- one of the six bundled imaging presets with its ready detector distribution,
  MAT source model, and source-bound `.atpy-model` runtime model; or an external
  source set whose parameter, distribution, and model paths resolve exactly.

No export is needed for the bundled 2019 or Gaussian models. For the bundled
pre-2019 red-channel source or another retired external model, run
`acetree-starrynite-export-model` under a MATLAB release that reconstructs the
object and write an `.atpy-model` file. The exporter verifies MATLAB predictions
and writes a provenance manifest. See [Exporting a source-bound classifier](STARRYNITE_DIFFERENTIAL_TESTING.md#exporting-a-source-bound-classifier)
for the command and old/new MATLAB model boundary.

Recommended whole-movie sequence:

1. Create or open an empty dataset, then open **Review Initial Tracking Draft**
   or choose **Tracking > Track Whole Movie...** (also available in the
   **Edit & Tracking Tools** dock).
2. Choose **Modern StarryNite (recommended)** or **Legacy StarryNite exact replay
   (advanced)**, then select the bundled imaging preset matching the microscope.
3. If radius, intensity threshold, or missing-frame allowance needs adjustment,
   edit the visible control and choose **Save tuned parameter copy…**. AceTree
   preserves the original statements and comments, appends supported edits,
   reloads the copy, and makes it the new recent file. It never edits the source
   model. Put the copy beside the source file when its model paths are relative,
   or update those paths explicitly.
4. For exact replay, the bundled runtime model is selected automatically. Use
   **Use another legacy model...** only for an externally exported model.
5. Open **Compatibility details…**. Continue only when the exact backend is
   reported runnable. A detector test at time 1 is optional; later exact frames
   require prior-frame state and must be assessed with **Build Full Draft**.
6. Build the full draft, inspect counts, warnings, positions, links, and daughter
   branches across representative early, crowded, division, and late frames,
   then either **Accept Draft** once or **Discard Draft**. Failure, cancellation,
   discard, or closing the workbench leaves the nuclei record unchanged.

Common fail-closed messages identify a corrective next step:

| Message category | What to do |
|---|---|
| Parameter, model, distribution, or classifier source changed | Reload the parameter file and attach a newly validated export. Exact mode rehashes sources and will not use a stale association. |
| Required legacy value or referenced file is missing | Add or correct it in a parameter-file copy. AceTree does not invent a default distribution or search for a same-named file elsewhere. |
| Calibration mismatch or downsampling is not 1 | Correct the dataset calibration/source selection, or use the native tracker. Do not rescale exact inputs implicitly. |
| Exact mode must start at time 1 or the dataset is not empty | Create/restore an empty record and run the complete range from time 1; use Track Selected Cell for an existing curated lineage. |
| Unsupported statement, polar-body mode, or hysteresis mode | Use the native workflow or retain MATLAB for that run. The exact backend stops because the required raw measurement or safe parameter meaning is unavailable. |
| Model cannot be reconstructed or is not source-bound | Export inert numeric state with a MATLAB release that can load the original object, then select the resulting `.atpy-model`. Never substitute a retrained model. |

For sparse real-world testing, select a trustworthy live nucleus and use **Track
Selected Cell** with the native StarryNite tracker. Choose the same bundled
imaging preset, tune the local search and caution controls, choose a
division policy, build and inspect a short draft, and accept only the visible
branches. This is the supported way to track a few cells in an already curated
embryo; it is not a partial invocation of the exact whole-movie classifier.

The current exact compatibility claim is scoped to the source-bound 2019
single-model profile, unit downsampling, matching calibration, and modes whose
raw detector measurements cross the AT boundary. Polar-body and hysteresis
modes remain blocked. The historical four-model `ambigious` runtime is
implemented but still needs live certification with a real four-model MAT
corpus and an older MATLAB release that reconstructs those objects. Noisy,
representative whole-embryo performance and import/export conformance remain
real-world release-validation work.

As of 2026-07-29, the complete non-live repository suite passed with `1351
passed, 71 skipped`. The latest opt-in MATLAB-oracle suite, run on 2026-07-16,
passed with `19 passed, 1 skipped` in 11 minutes 37 seconds. The expected skip is
the historical-object export boundary above. See [StarryNite Differential Testing](STARRYNITE_DIFFERENTIAL_TESTING.md#running-locally)
to repeat the live comparison with a local MATLAB and StarryNite checkout.

#### Curate a Known Sublineage: EMS to E to Ea/Ep

This workflow is supported even when the rest of the embryo is incomplete:

1. At a clear reference frame, define **Body Orientation** using Posterior + Anterior and either Ventral + Dorsal or Right + Left, then click **Apply Axes**.
2. Select any trusted nucleus in the EMS continuation, choose **Rename**, and enter `EMS`. The forced EMS identity propagates only along that one-successor continuation.
3. Enter **Manual Track** and place the EMS continuation through successive frames.
4. At the EMS division frame, place one daughter, then place the second daughter at the same frame. The second placement establishes the division. AceTree keeps `EMS` on the parent, removes the inherited EMS lock from the first daughter, and proposes `E` and `MS` from their geometry in the manual body frame.
5. Exit Manual Track, select `E`, and enter Manual Track again to follow that branch. At the E division, place both daughters at the same frame; AceTree applies the E rule and proposes `Ea` and `Ep`.
6. Repeat by selecting whichever automatically named daughter you want to follow. Re-selecting at a division is intentional: Manual Track stays anchored to the branch that was selected when the mode began.

`E`, `MS`, `Ea`, and `Ep` are automatic identities, not new manual locks. Correcting the body frame and clicking **Apply Axes** can therefore reorder them while preserving the forced `EMS` anchor. If one proposed daughter is independently known, Rename only that cell; **Use Automatic** later returns it to geometry-based naming. The status preview reports the rule axis, source, confidence, and ambiguity, so inspect uncertain calls before continuing deeply down the branch.

At a selected cell's terminal frame, a click near the existing nucleus is interpreted as continuation, while a click farther away than its displayed diameter is interpreted as the second daughter. If newborn daughters are too close to separate reliably, track to a later frame where they have separated or place them and use **Relink** explicitly. Body axes determine daughter ordering; two reciprocal successor links establish that a division occurred.

### 6.5.1 Body Orientation Correction

Use **Body Orientation** when automatic daughter ordering is systematically mirrored or rotated:

1. Go to one clear reference timepoint. Select a nucleus at the posterior endpoint, choose **Posterior**, and click **Label selected**. Repeat for **Anterior**.
2. Add either **Ventral** and **Dorsal**, or **Right** and **Left**. All endpoints for a frame must be labeled at the same timepoint.
3. Click **Apply Axes**. AceTree validates that the two directions are non-zero and not parallel, then displays the committed source and geometry quality.
4. Inspect several known divisions. If an axis is reversed, relabel or swap that endpoint pair and apply again. **Undo** restores the prior orientation in one step.
5. Save the dataset to persist the manual orientation beside the nuclei ZIP as AuxInfo v2.

The conventions are AP **posterior → anterior**, DV **ventral → dorsal**, and LR **right → left**, with `DV = AP × LR`. AP plus either DV or LR is sufficient; AceTree constructs the third axis and orthogonalizes the frame. Z separation is scaled by the dataset's physical z resolution, so a one-plane z shift is not assumed to equal one x/y pixel.

Do not infer left/right from the ABa–ABp separation alone. At the four-cell stage that pair does not by itself establish signed LR; use trusted metadata, a manual anatomical cue, or later handedness. Manual orientation takes precedence over automatic geometry, while manual **cell-name** overrides remain intact when naming is rerun.

If AceTree can identify founders from topology but cannot construct a complete AP/DV/LR frame, it keeps those founder names but does not pretend that microscope x/y/z are anatomical coordinates. Existing loaded descendant names are preserved; new uncertain branches use neutral `Nuc...` names until a valid body frame is supplied and naming is rerun.

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

All image channels are loaded as 3D stacks when entering 3D mode. Clicking on a sphere selects the corresponding cell. Relink pick mode and track mode also work in 3D. An open tracking proposal changes to read-only napari Points and 3D path layers automatically; diagnostic candidates use a cross symbol and a stopped search region is shown as three calibrated wireframe rings. Switching back to 2D restores the slice overlay without changing the proposal.

### 6.7 Detached 3D Viewer Window

Click **3D Window** in the player controls to open a separate 3D viewer window. This window is designed for visualization and always uses rule-engine coloring (visualization mode), regardless of the main viewer's color mode. This allows you to edit in 2D in the main viewer while simultaneously viewing the embryo in 3D.

**Controls:**
- **Time slider + Sync button**: When Sync is on (default), the 3D window follows the main viewer's timepoint. Toggle off to navigate independently.
- **Color Preset dropdown**: Switch between visualization presets (lineage depth, expression).
- **Per-channel contrast**: Each image channel has visibility checkbox, min/max sliders, and auto/reset buttons.
- **Labels: ON/OFF**: Toggle label visibility globally.
- **Clear Labels**: Remove all shown labels.
- **Left-click** on a 3D sphere: Toggle that cell's label on/off.
- **Tracking drafts**: The current proposal, selected review point, diagnostic candidates, paths, and stopped search region mirror the main viewer. With Sync off they render at this window's independent timepoint.

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
- **Ctrl+Shift+S** or **Save As**: Opens a file dialog to choose a new location, makes that location the target of subsequent Save operations, and updates the source XML config so reopening it follows the new ZIP. The retarget happens only after the data save and config rewrite both succeed.

The saved file is a ZIP containing CSV-formatted nucleus data, one entry per timepoint. This is the standard AceTree nuclei format and can be opened by both AceTree-Py and the original Java AceTree. AceTree fully prepares the nuclei ZIP and any manual-orientation sidecar before committing them. A failure leaves the previous ZIP and sidecar together, rather than mixing one new file with one old file. Existing file permissions are retained across replacement. If an automated tracking run has been accepted, Save also writes the latest run beside the nuclei ZIP as `<stem>.tracking.json`.

### 8.2 What Gets Saved

- All nucleus positions, sizes, and names
- All predecessor/successor links
- Manual name overrides (`assigned_id`)
- Manual body orientation in an AuxInfo v2 sidecar, including source, quality, and reference time
- The latest accepted automated tracking run and provenance in `<stem>.tracking.json`, when present
- Expression values

Edits that haven't been saved are tracked against an explicit savepoint. Undoing a saved edit makes the dataset dirty; redoing exactly back to the saved state makes it clean again. A new edit after Undo creates a new branch and remains dirty even if the history happens to have the same number of entries. The savepoint advances only after a successful Save or Save As.

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

## 10. Measuring Expression from Image Pixels

The **Measure** tool re-derives per-nucleus expression values directly from the image data. It's a Python port of the Java `AceBatch2` measure routine — at every timepoint and every image channel it sums pixel intensities inside each nucleus (modelled as a sphere of diameter `size`) plus a surrounding annulus used as a local background estimate.

Use Measure when:

- Your `.nuclei` zip was loaded from StarryNite without pre-computed `rweight` values.
- You want to re-measure the dataset against a *different* expression channel (the `.nuclei` format only stores one channel's expression).
- You want per-cell time-series CSVs for downstream analysis.

### 10.1 Running Measure

1. Load a dataset with image data (File → config.xml / `acetree-py gui …`).
2. Click **File → Measure…** in the menu bar.
3. In the dialog:
   - **AT channel** — the channel whose measurements will be written back onto each `Nucleus` as `rwraw` / `rwcorr1` and drive the lineage-tree coloring. All channels are still measured; this just picks which one becomes the "AT" channel.
   - **Output folder** — where the per-channel CSVs go. Defaults to `<zip_dir>/measurements` when a nuclei zip is loaded.
4. Click **OK**. A progress dialog shows `channel c/N, timepoint t/T…` and can be cancelled.
5. When the run completes, every open lineage tree panel rebuilds to reflect the fresh `rweight` values. A status message reports how many CSVs were written.

### 10.2 Output CSVs

One CSV per image channel, with filenames like:

- `measure_channel1_AT.csv` — the channel you selected as the AT channel.
- `measure_channel2.csv`, `measure_channel3.csv`, … — every other channel.

Each row is one cell, each column is one absolute timepoint:

```
cell_name, start_time, end_time, t1, t2, t3, …, tN
ABa,       4,          12,       ,   ,   , 1234.5, 1256.2, …
ABp,       4,          13,       ,   ,   , 987.3,  1002.1, …
```

Cells absent at a timepoint get an empty column value. The per-timepoint number is the same formula the lineage tree uses for color — plain `rwraw` when the session's correction is `"none"`, `rwraw - rwcorr1` otherwise.

Cells are sorted by `start_time` then name. Start and end times are 1-based and inclusive.

### 10.3 Effects on the Dataset

- For the chosen AT channel, `nuc.rwraw`, `nuc.rwcorr1`, `nuc.rsum`, and `nuc.rcount` are **updated** on every live, measurable nucleus. Dead nuclei and nuclei that fall outside the image bounds are left untouched.
- `compute_red_weights()` runs afterwards so `nuc.rweight` reflects the session's current correction mode.
- **Save (`Ctrl+S`) to persist the new values** — Measure only changes the in-memory manager; the zip on disk is unchanged until you save.

### 10.4 Correction Modes and Limitations

The Python port computes:

- **`rwraw`** — mean intensity inside the nucleus.
- **`rwcorr1`** — mean intensity in the surrounding annulus (classic "global background").
- **`rwcorr3`** — "blot" correction: mean intensity of the annulus with *every nucleus's projected disk masked out* at that Z plane. When neighbouring nuclei intrude on the annulus, their bright pixels are excluded from the background estimate, giving a cleaner local background in crowded regions. With no neighbours nearby, blot equals the global annulus.

`rwcorr2` / `rwcorr4` are not computed by this port (they came from external MATLAB and a crosstalk solver in the Java pipeline). If the session uses `"local"` or `"cross"` correction, the CSV value falls back to `rwraw - rwcorr1` as a best-effort approximation.

### Choosing a correction

The Measure dialog exposes three background-correction modes:

| Mode | CSV value per timepoint | When to pick |
|---|---|---|
| **None** | `rwraw` | Raw intensity only. |
| **Global — annulus mean** | `rwraw − rwcorr1` | Sparse embryos, isolated nuclei. Fastest. |
| **Blot — annulus with neighbors masked (rwcorr3)** | `rwraw − rwcorr3` | Crowded embryos where neighbouring nuclei poke into the annulus and inflate the global background. |

The dialog also writes the chosen mode back onto `manager._expr_corr`, so the lineage tree immediately re-colours using the corresponding correction field.

---

## 11. Understanding Cell Names

### 11.1 Automatic Naming

When a dataset is loaded, the naming pipeline automatically identifies cells:

1. **Founder identification**: Finds the 4-cell stage and identifies ABa, ABp, EMS, P2 using topology and timing. ABa/ABp are distinguished by projection onto a posterior→anterior axis rather than assuming image x. Explicit orientation is preferred; weak geometric fallbacks are reported with lower confidence.
2. **Back-tracing**: Names earlier cells (AB, P1, P0) by tracing predecessor links backward.
3. **Forward naming**: Names all subsequent cells by applying the selected parent's Sulston division rule to physically scaled 3D geometry. Each result carries an axis, confidence, and orientation source.

Orientation precedence is: valid explicit AuxInfo v2 (including a manual landmark frame), a supported AuxInfo v1 orientation, per-timepoint lineage geometry, then static founder geometry. A present but unusable v2 file (missing, malformed, non-finite, zero, or parallel AP/LR vectors) does not mask valid v1 metadata. Invalid placeholder metadata is ignored. Automatic geometry uses AP from P2 toward ABa and a DV seed from EMS toward ABp, projected perpendicular to AP; it does not treat ABa–ABp as LR. Manual correction is recommended when compression, sparse tracking, or uncertain handedness makes that estimate weak.

### 11.2 Unnamed Cells

Cells that can't be automatically named receive placeholder names like `Nuc042_15_200_300` (3-digit zero-padded timepoint, then z, x, y). These are typically polar bodies or cells at the edges of the tracked lineage.

### 11.3 Manual Overrides

When you **Rename** a cell (Section 6.4), the name is stored as a permanent override (`assigned_id`). This name survives automatic re-naming — even if you save and reload the dataset, the override persists.

**Automatic propagation:** When the naming pipeline runs (on load or after edits), the forced name is automatically propagated to every timepoint the cell exists — both forward through continuation links and backward to the cell's birth. This means you only need to rename the cell at one timepoint; the override covers its entire lifetime.

**Division naming:** When a renamed cell divides, the forced name is used as the parent name for determining daughter names via the standard Sulston rules. For example, forcing `EMS` lets its daughters be assigned automatically as `E` and `MS`; tracking `E` to its next division then produces `Ea` and `Ep`. Only the known anchor remains forced, so a later body-axis correction can safely reorder its automatic descendants.

The three name concepts are deliberately separate:

| Displayed concept | Stored value | Meaning |
|---|---|---|
| Automatic/current identity | `identity` | May change after a move, relink, or orientation correction |
| Forced identity | `assigned_id` | Explicit user decision; survives automatic processing |
| Name used by the UI | `effective_name` | Forced identity when present, otherwise automatic identity |

**Use Automatic** clears `assigned_id` for that cell continuation without erasing the user's ability to Undo. Add and Track do not turn an automatic suggestion into a forced identity. Forced-name propagation follows only live reciprocal one-successor continuations, stops at divisions, and stops on a conflicting override. A conflict is shown for correction; AceTree does not append an arbitrary suffix or let traversal order settle it.

---

## 12. Lineage Tree View

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

## 13. Tips and Workflow

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

### Correcting mirrored daughter names

1. Check that the two daughter tracks and their predecessor links are correct.
2. Look at the division preview's axis, confidence, and orientation source.
3. If several divisions are mirrored in the same direction, correct **Body Orientation** rather than renaming every daughter individually.
4. If one biological identity is known with certainty, Rename that cell to make a forced override. Use **Use Automatic** later if you want geometry to control it again.

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
- Click **Screenshot** in the Visualization section of **Edit & Tracking Tools** to capture the current view as a PNG.
- Click **Record...** to export a sequence of PNGs across a timepoint range (useful for making movies).

### Exporting for analysis
```bash
# Get a table of all cells with their lineage info:
acetree-py export config.xml -f cell_csv -o my_cells.csv

# Get per-nucleus data for custom analysis:
acetree-py export config.xml -f nucleus_csv -o my_nuclei.csv
```

---

## 14. Tracking & Dataset Creation

AceTree-Py can create new datasets from raw TIFF images and start either with
empty manual annotation (the default) or an editable automated draft. The GUI
can use installed DoG, LoG, and StarryNite detector/tracker combinations; the
non-interactive CLI currently exposes the DoG/LoG plus Simple LAP presets.
Interactive tools remain available for placing, tracking, and linking selected
cells. This is useful when:

- You have image data that hasn't been processed by StarryNite or another detection pipeline.
- You want to manually annotate nuclei positions in a single frame (detection-only, no tracking).
- You want to manually track a subset of cells across time.
- You want an initial automated draft to curate rather than treating tracker output as ground truth.

### 14.1 Creating a New Dataset

#### Interactive Wizard (GUI)

```bash
acetree-py create
```

This opens a 5-page wizard dialog:

1. **Image directory** — select the folder containing your TIFF files. The wizard auto-detects the naming pattern and image dimensions.
2. **Channel layout** — choose how channels are arranged:
   - *Single channel* — one channel per TIFF page.
   - *Side-by-side (split)* — one image per timepoint with two channels in left/right halves.
   - *Separate directory per channel* — one directory per channel, each with its own per-timepoint TIFFs.
   - *Multichannel TIFF stack* — one TIFF per timepoint with pages interleaved across channels. Pick the number of channels (2–8) and the page order:
     - **Interleaved** (`Z1C1, Z1C2, Z2C1, Z2C2, …`) — channel-fastest; the common ImageJ / MicroManager default.
     - **Planar** (all Z for channel 1, then all Z for channel 2) — plane-fastest.
   Set the flip checkbox if your images are mirrored horizontally.
3. **Voxel parameters** — set XY resolution (µm/pixel), Z resolution (µm/plane), number of timepoints and planes (auto-filled from detection; for interleaved stacks the Z count is automatically `pages / num_channels`).
4. **Initial tracking** — keep the recommended **Manual annotation** default, or choose an automated draft from the installed detectors and trackers. Configure the channel, expected nucleus radius, quality threshold, maximum displacement, allowed missing frames, and whether a division-aware tracker should propose two-daughter branches. The channel range follows the selected image layout, and unusable plugin/channel/stack combinations block creation with an explanation. Simple LAP keeps divisions off; the StarryNite tracker enables reviewed divisions by default. Merges are always disabled.
5. **Output** — choose where to save the dataset config XML and nuclei ZIP.

The wizard creates the empty dataset before opening its review workbench. To
replace the initial native preset with a standard StarryNite parameter file—or
to select the exact global backend—follow the [real-world StarryNite test
checklist](#real-world-starrynite-test-checklist) in that workbench. Exact mode
is a reviewed GUI workflow and is not a `create --tracking` CLI preset.

#### CLI (Non-Interactive)

```bash
acetree-py create <image_directory> [OPTIONS]
```

Options:

| Option             | Default | Description                                                         |
|--------------------|---------|---------------------------------------------------------------------|
| `--output`         | auto    | Output directory for config + nuclei ZIP                            |
| `--xy-res`         | 0.09    | XY pixel resolution in µm                                           |
| `--z-res`          | 1.0     | Z plane spacing in µm                                               |
| `--split`          | off     | Split side-by-side dual-channel images                              |
| `--flip`           | off     | Flip images left/right                                              |
| `--interleaved`    | off     | Single TIFF per timepoint contains interleaved multichannel pages   |
| `--num-channels`   | 1       | Number of channels (required with `--interleaved`, must be ≥ 2)    |
| `--channel-order`  | `CZ`    | Page order for interleaved stacks: `CZ` (channel-fastest) or `ZC` |
| `--tracking`       | `manual`| Initial workflow: `manual`, `dog-lap`, or `log-lap`                |
| `--detection-channel` | 1    | One-based channel for automated detection                           |
| `--nucleus-radius` | 4.0     | Expected physical nucleus radius in microns                         |
| `--detection-threshold` | 5.0 | Minimum LoG/DoG response                                           |
| `--linking-distance` | 8.0   | Maximum Simple LAP displacement in microns                          |
| `--missing-frames` | 1       | Maximum missed frames to bridge                                     |

**Examples:**

```bash
# Create dataset from a folder of TIFFs with default resolution:
acetree-py create /data/embryo/images/

# With specific resolution and split channels:
acetree-py create /data/embryo/SPIMA/ --output /data/embryo/manual_output/ --xy-res 0.1625 --z-res 0.65 --split

# Interleaved 2-channel TIFF stacks (Z1C1, Z1C2, Z2C1, Z2C2, …):
acetree-py create /data/embryo/multichannel/ --output /data/embryo/out/ --interleaved --num-channels 2 --channel-order CZ

# Single-frame annotation (one TIFF file in the directory):
acetree-py create /data/single_frame/

# Build an initial DoG + Simple LAP draft for review:
acetree-py create /data/embryo/images/ --tracking dog-lap --nucleus-radius 3.5 --linking-distance 7
```

> `--interleaved` bypasses `--split`/`--flip` — channels are already resolved at the page level, so the horizontal split wrapper would halve a valid image.

The `create` workflow:
1. Scans the image directory for TIFF files and probes the first image for z-plane count.
2. Generates an `AceTreeConfig` with the correct image paths and resolution.
3. Creates an empty nuclei ZIP (no detections).
4. Writes the XML config file to the output directory.
5. Launches the GUI with an empty nuclei record. In automated mode it opens the modeless whole-dataset workbench, runs analysis in the background, and shows the result in the 2D/3D image viewers before any commit.
6. Adds the global result only after **Accept Draft**; discard, cancellation, failure, or window close leaves the valid dataset empty for manual annotation.

The non-interactive CLI also defaults to manual mode; select `dog-lap` or
`log-lap` explicitly to open an automated draft in the same pre-commit review workbench.

### 14.2 Placing Nuclei (Add Mode)

Once the GUI is open on a new (empty) dataset:

1. Navigate to the desired timepoint and z-plane.
2. Click **Add** in the **Edit & Tracking Tools** panel to enter add mode.
3. **Left-click** anywhere in the image to place a nucleus at that position.
4. The nucleus is created at the current z-plane with a default diameter of 20 pixels.
5. Press **Esc** to exit add mode.

**Adding onto an existing cell:**
1. Right-click an existing nucleus to select its cell.
2. Navigate to a later timepoint.
3. Click **Add**, then left-click to place. The new nucleus inherits the predecessor link and diameter. It preserves an existing forced override, but a merely automatic name remains automatic.
4. If there is a gap > 1 timepoint, intermediate nuclei are automatically interpolated.

### 14.3 Adjusting Nuclei (D-Pad Controls)

After placing a nucleus, use the **Move / Resize** D-pad buttons to fine-tune:

- **XY arrows**: nudge position by 1 or 5 pixels.
- **Z buttons**: shift the z-plane by 1 or 5.
- **Size buttons**: grow or shrink the diameter by 1 or 5 pixels.

Each press is individually undoable. The cell stays selected between presses for rapid adjustment.

### 14.4 Tracking Across Time

For continuous manual tracking across many timepoints, use the **Manual Track** button:

1. Select the cell you want to extend.
2. Click **Manual Track** to enter tracking mode.
3. Advance to the next timepoint (right arrow).
4. **Right-click** to place the next position. The nucleus is automatically linked.
5. Repeat steps 3–4 for as many timepoints as needed.
6. Press **Esc** to exit tracking mode.

Manual Track mode automatically handles:
- **Name continuity**: the current automatic identity can continue, while only a genuinely forced parent override is inherited as `assigned_id`.
- **Predecessor linking**: direct link if adjacent, interpolation if there's a gap.
- **Size inheritance**: the placed nucleus inherits the parent's diameter.

To follow only one existing cell automatically, use **Tracking > Track Selected Cell Forward…** as described in Section 6.5. It defaults to a ten-frame horizon, can run a review-only **Test Next Frame**, searches a moving local ROI, and stops on ambiguity. Accept the full draft or a selected reliable prefix as one undoable edit, then save or undo directly from the still-open workbench.

### 14.5 Workflow for Single-Frame Annotation

For annotating nuclei in a single image (no tracking):

```bash
# Create a dataset with the single image:
acetree-py create /data/single_frame/
```

1. Use **Add** mode to left-click on each nucleus.
2. Use the D-pad controls to adjust positions and sizes.
3. Use **Rename** to assign cell identities.
4. **Save** (`Ctrl+S`) to persist annotations.

Each placed nucleus becomes an independent root cell. Manual Track and Track Selected Cell are not useful in single-frame mode (there are no future timepoints to track to).

### 14.6 Saving and Reloading

After annotation:

- **Save** (`Ctrl+S`) writes the nuclei to the ZIP file and the config XML.
- A manually applied body orientation is written as an AuxInfo v2 sidecar and loaded before automatic geometry next time.
- If an automated run was accepted, the latest run is written beside the nuclei ZIP as `<stem>.tracking.json`.
- To reopen later: `acetree-py gui path/to/output/config.xml`
- The automatic naming pipeline runs on load. If enough cells have been placed for the 4→8 cell transition to be detected, Sulston names will be assigned automatically.

### 14.7 Tips for Manual Tracking

- **Use the 3D view** (Section 6.6) to verify nucleus positions in three dimensions.
- **Place nuclei on the z-plane where the nucleus is brightest** for the most accurate position.
- **Use Manual Track** for curated long tracks, or **Track Selected Cell** for a reviewable draft of one continuation.
- **Use Relink** to correct mistakes after the fact rather than undoing many steps.
- **Save frequently** (`Ctrl+S`) — there is no autosave.
