# Subcellular Measurements Design

Status: V1 implementation complete; full regression suite passing
Branch: `subcellular-measurements`
Base: `tracking-integration` at `44df6446f79d07c53e47b1401a8bb5708a37c4c3`

## Outcome

Add a second, manually curated annotation stream alongside nuclei. It will support:

- 2D closed polygons on a single Z plane.
- 2D polylines with adjustable physical thickness.
- 3D structures authored as closed polygon contours on consecutive Z planes and sampled either as a filled volume or an inner shell.
- A stable object track across time, identified by an immutable UUID and a user-facing object class plus positive integer, such as `Golgi #2`.
- Optional association of every timepoint observation with a nucleus/cell at that same timepoint.
- Raw multi-channel scalar intensity measures, spatial line profiles, and scalar time-series plotting through a generalized form of the existing Expression Plot interface.
- A versioned, checksummed JSON sidecar beside the dataset XML, loaded automatically and saved as authoritative annotation data in the same coordinated save boundary as nuclei.

This is a parallel stream, not an extension of `Nucleus`. Nuclear ZIP compatibility remains unchanged.

## Review team

| Review area | Main repository seams reviewed | Result |
|---|---|---|
| ROI data model and persistence | `core/nuclei_manager.py`, `core/lineage.py`, `io/config.py`, `io/nuclei_reader.py`, `tracking/persistence.py`, save rollback paths | Separate ROI document/manager, strict XML-derived sidecar, physical cell anchors, coordinated save |
| Measurement backend and expression integration | `analysis/measure.py`, `analysis/measure_runner.py`, `analysis/expression_measurements.py`, `analysis/expression_plot.py`, `io/image_provider.py` | Pure raster/reducer services, immutable provenance-bound cache, generic scalar series adapter, separate vector profile model |
| UI and usability | `gui/app.py`, `gui/viewer_integration.py`, `gui/player_controls.py`, `gui/edit_panel.py`, `gui/viewer_3d_window.py`, `gui/expression_plot_window.py` | Dedicated Objects dock, read-only projection plus transient editor, explicit temporal/review states, mode-safe 2D authoring and read-only 3D preview |

## V1 decisions

The following previously ambiguous points are fixed for the first release:

1. A “3D polygon” is a stack of manually drawn planar contours, not an arbitrary triangle mesh.
2. A shell is the physical-width band inside the boundary of the filled volume. Centered and exterior shells are future explicit modes.
3. Object instance indices are unique dataset-wide within a class. A UUID remains the persistence identity.
4. A cell division never silently chooses a daughter. Continuing, ending, or forking an object is explicit.
5. Missing Z contours and missing timepoints are not silently interpolated. Copying an earlier contour/frame creates a draft that must be reviewed.
6. Scalar missing data is `None` plus a reason, never zero or NaN. Zero intensity is a valid result.
7. V1 measurement values are raw intensities. Nucleus-specific global/blot correction is not applied to ROIs.
8. Time axes remain timepoints, time since first segmentation, or normalized observed track. The config has no time-interval calibration for minutes.

True meshes, multiple disconnected components in one object, polygon holes, automatic time tracking, local-background correction, and time-by-distance heatmaps are later extensions.

## Architecture

New headless services remain independent of Qt and napari:

```text
AceTreeApp
  ├── NucleiManager ── existing nuclei/tree stream
  ├── RoiManager ───── authoritative ROI stream and associations
  ├── EditHistory ──── one chronological undo/redo history with edit domains
  ├── ImageProvider ── existing image access
  ├── RoiMeasurementEngine
  └── ViewerIntegration
        └── RoiViewerIntegration

RoiManager
  └── SubcellularRoiDocument
        ├── ObjectClass definitions
        └── RoiObjectTrack[]
              └── frame records keyed by 1-based timepoint
```

Recommended modules:

- `acetree_py/core/subcellular_roi.py`: immutable identifiers, geometry unions, tracks, frame/review states, validation.
- `acetree_py/core/roi_manager.py`: indexes, allocation, ROI revisions, cell reconciliation, queries.
- `acetree_py/io/roi_sidecar.py`: naming, strict JSON, checksum, staging, migrations, external-change protection.
- `acetree_py/io/dataset_transaction.py`: reusable staged multi-artifact save/rollback coordinator.
- `acetree_py/editing/roi_commands.py`: create/edit/delete/reclass/reindex/associate/review commands.
- `acetree_py/analysis/roi_rasterization.py`: pure geometry-to-cropped-mask conversion.
- `acetree_py/analysis/roi_measure.py`: scalar reducers and spatial profiles.
- `acetree_py/analysis/roi_measurements.py`: immutable snapshots, cache, freshness, expression adapters.
- `acetree_py/gui/subcellular_objects_panel.py`: object browser, inspector, drawing/review workflow.
- `acetree_py/gui/roi_viewer_integration.py`: record-derived overlay and transient editor.
- `acetree_py/gui/roi_measure_dialog.py`: scoped measurement configuration and progress.
- `acetree_py/gui/roi_scalar_plot_window.py`: object-centric scalar time-series plots and guarded CSV/SVG export.
- `acetree_py/gui/roi_profile_window.py`: vector profile plots and CSV export.

## Domain model

```text
SubcellularRoiDocument
  document_id: UUID
  file_revision: int
  roi_revision: monotonic in-memory token
  coordinate_space: CoordinateSpaceSnapshot
  object_classes: ObjectClass[]
  objects: RoiObjectTrack[]

ObjectClass
  class_id: UUID
  name: nonblank descriptive name
  color: accessible RGBA
  next_instance_index: positive monotonic allocator
  default_geometry_kind: optional

RoiObjectTrack
  object_id: UUID
  class_id: UUID
  instance_index: positive int
  expected_start_time: optional int
  expected_end_time: optional int
  frames: mapping[timepoint, RoiFrameRecord]

RoiFrameRecord
  frame_id: UUID
  revision: int
  timepoint: positive int
  presence: segmented | absent
  review_state: draft | reviewed | needs_review
  cell_ref: CellRef | null
  geometry: Polygon2D | ThickPolyline2D | ContourStack3D | null
```

Invariants:

- `object_id` and `class_id` never change. Class names and display indices are editable metadata.
- `(class_id, instance_index)` is unique across the dataset. Deleted indices are not automatically reused; the persisted allocator prevents identity drift after reopen.
- A track has no more than one frame record per timepoint.
- `presence=absent` has no geometry and represents an explicit biological/review decision. A missing map entry means “not segmented/undecided.”
- A geometry or association edit moves `reviewed` to `needs_review`. Saving never changes scientific review state.
- Unassociated or orphaned geometry remains object-visible and measurable; only cell-based grouping/alignment is unavailable.

### Geometry unions

All XY points are finite floating-point provider/display pixel-center coordinates in `(x, y)` order. Z is an absolute AceTree plane number. Time is absolute and 1-based.

```json
{
  "kind": "polygon_2d",
  "z_plane": 12,
  "exterior_xy_px": [[10.5, 20.0], [18.0, 21.0], [14.0, 30.0]]
}
```

```json
{
  "kind": "thick_polyline_2d",
  "z_plane": 12,
  "points_xy_px": [[10.0, 20.0], [14.0, 25.0], [22.0, 28.0]],
  "thickness": {"value": 0.8, "unit": "um"},
  "cap_style": "round",
  "join_style": "round"
}
```

Pixel thickness is allowed only when physical calibration is unavailable. The UI defaults to micrometres and shows the derived pixel width.

```json
{
  "kind": "contour_stack_3d",
  "sampling_mode": "inner_shell",
  "shell_thickness_um": 0.6,
  "slices": [
    {"z_plane": 11, "exterior_xy_px": [[10, 20], [18, 21], [14, 30]]},
    {"z_plane": 12, "exterior_xy_px": [[11, 20], [19, 22], [15, 31]]}
  ]
}
```

V1 permits one connected exterior polygon per 3D slice. Volume/shell measurement requires consecutive annotated planes. A later schema can add holes, multiple components, explicit interpolation provenance, or mesh geometry without changing the existing discriminator.

### Coordinate and calibration contract

This must be centralized because current code has both absolute 1-based AceTree planes and local 0-based NumPy Z indices.

- Provider arrays: `(Z, Y, X)` or `(Y, X)`.
- Provider calls: 1-based time and plane, 0-based channel.
- Model XY: `(x, y)` in the displayed provider space after configured split/flip handling.
- Napari conversion: model `(x, y)` to layer `(y, x)` only at the GUI boundary.
- Model Z: absolute AceTree plane.
- Stack Z: `z_index = z_plane - plane_start`.
- Physical conversion: `x_um=x*xy_res`, `y_um=y*xy_res`, `z_um=(z_plane-plane_start)*z_res`.

The sidecar records `plane_start`, `xy_res`, `z_res`, image width/height/plane count, split/flip settings, and coordinate-space version. A calibration/view mismatch preserves annotations but marks affected records as needing review and blocks physical normalization until acknowledged. ROI implementation must also make `plane_start` round-trip through XML; it is present in `AceTreeConfig` but is not currently written in the resolution element.

### Geometry validation

New or edited records require:

- finite coordinates and parameters;
- time and Z inside dataset bounds;
- at least three distinct polygon vertices, nonzero area, and no self-intersection;
- at least two distinct polyline points and positive thickness;
- unique, sorted, consecutive 3D Z planes for measurable volume/shell records;
- positive shell thickness with valid physical calibration;
- a nonempty rasterized mask.

The writer emits canonical ring winding and no duplicate closing vertex. Loading quarantines an invalid object/frame independently so one corrupt record cannot discard the rest of the ROI document.

## Object identity and cell association

The user-facing label is `<class name> #<instance index>`. Stable measurement/export keys use UUIDs rather than mutable names:

```text
roi:<object_uuid>:ch<1-based-channel>:<metric>
```

Each segmented or absent frame may carry:

```json
{
  "nucleus_anchor": {"timepoint": 42, "index": 7},
  "cell_birth_anchor": {"timepoint": 30, "index": 4},
  "name_snapshot": "ABalap",
  "centroid_snapshot_xyz_px": [101, 76, 12.0]
}
```

The same-frame `(timepoint, nucleus index)` anchor is authoritative. The birth anchor, name, and centroid are reconciliation hints. This follows the stable physical-selection boundary already used by `AceTreeApp.selection_anchor` (`gui/app.py`) and avoids treating mutable/colliding cell names as identifiers.

Resolution is fail-closed:

1. Resolve the exact alive same-frame nucleus and current cell.
2. Compare the saved birth/name hints for a visible association-change warning.
3. If the anchor is unavailable, retain the ROI and mark it orphaned.
4. Offer ranked relink suggestions by birth anchor, unique name, and spatial proximity, but never apply one without confirmation.

Renames do not orphan an exact physical anchor. Relinks can mark the frame `needs_review`. Killing a nucleus retains geometry as orphaned. At division, the next frame must be explicitly associated with a daughter, left unassociated, ended, or forked.

## Sidecar and dataset opening

The conventional authoritative path is derived from the opened XML:

```text
embryo.xml
embryo.subcellular-rois.json
```

Use `config.config_file.with_suffix(".subcellular-rois.json")`, not the nuclei ZIP stem. A headless manager without an XML may fall back to `<zip-stem>.subcellular-rois.json`; if both candidates exist, the XML-derived file wins and the conflict is reported.

Top-level envelope:

```json
{
  "schema": "acetree.subcellular-rois",
  "schema_version": 1,
  "checksum": {"algorithm": "sha256", "sha256": "..."},
  "document": {
    "document_id": "uuid",
    "file_revision": 7,
    "created_at": "...Z",
    "saved_at": "...Z",
    "producer_version": "0.2.0",
    "dataset_fingerprint": "...",
    "coordinate_space": {},
    "object_classes": [],
    "objects": [],
    "extensions": {}
  }
}
```

Persistence requirements:

- strict RFC JSON (`allow_nan=False`), UTF-8, duplicate-key rejection, finite values, collection/depth/file-size limits, stable ordering, and canonical SHA-256 checksum;
- same-directory private staging, flush and `fsync`, existing-mode preservation, then `os.replace`;
- pure version-to-version migrations, never rewriting during load;
- unknown newer versions load ROI features read-only;
- loaded checksum plus `document_id`/`file_revision` checked immediately before save to prevent silent last-writer-wins;
- an explicit empty document is saved after the final object is deleted, preventing old annotations from reappearing.

`AceTreeApp.from_config()` is the current autoload seam (`gui/app.py`). Domain parsing should be owned by `RoiManager.from_config()`, followed by cell reconciliation after `NucleiManager.process()` builds the lineage.

An absent file is normal. A malformed, unsupported, or checksum-invalid file must not block the nuclear dataset, but it produces a persistent error and protects the existing ROI path from overwrite. The user may retry, import a repaired copy, save a recovery copy, or explicitly discard/replace the invalid sidecar.

### Coordinated save

ROI geometry is authoritative annotation data, unlike optional tracking provenance. It must be staged and committed before `EditHistory.mark_saved()`.

Generalize the rollback approach in `NucleiManager.save()` and `AceTreeApp._do_save()`:

1. Stage nuclei ZIP, AuxInfo changes, dirty XML, and ROI JSON completely.
2. Capture prior destinations as sibling rollback files.
3. Install staged artifacts; commit XML last.
4. On any failure, restore every changed prior artifact and keep the document dirty.
5. Mark the shared edit-history savepoint only after all authoritative artifacts succeed.

True crash-atomic replacement across different filesystems is impossible. The payload fingerprint detects mixed generations after a crash; a future short save journal can automate recovery. The existing best-effort tracking sidecar may remain outside this authoritative transaction.

Current Save As retargets the nuclei ZIP while retaining and updating the source XML. Therefore the ROI sidecar stays beside that XML. A future “Save Dataset As” should copy XML, ZIP, AuxInfo, and ROI sidecar as a unit.

## Editing and revisions

Use one chronological Ctrl+Z/Ctrl+Y stream for nucleus and ROI edits. Extend `EditCommand` with effect domains while preserving existing defaults:

```text
nuclei_topology
nucleus_geometry
roi_geometry
roi_association
roi_metadata
config
```

ROI commands can use the current command protocol as a transition, but post-edit handling in `AceTreeApp._on_edit()` must route by effects:

- nuclear effects advance `NucleiManager.data_revision` and rebuild lineage only when required;
- ROI geometry effects advance `RoiManager.roi_revision` and invalidate only affected ROI mask/measurement cache entries;
- association-only effects rebuild lookup/plot indexes but do not reread image pixels;
- mixed commands can declare both domains.

This prevents an ROI vertex drag from invalidating every nuclear expression measurement or rerunning naming. Dirty/savepoint state remains unified.

## Measurement model

Do not add ROI branches to `analysis/measure.py`; it encodes nucleus-specific spherical/annular behavior. Reuse the orchestration patterns in `analysis/measure_runner.py`: time-major scheduling, optional all-channel bulk reads, cancellation, image-manifest validation before and after work, and atomic snapshot publication.

### Services

```text
RoiMaskRasterizer
  geometry + image shape + calibration -> cropped mask, bounding slices, coverage

RoiIntensityReducer
  cropped image + mask -> scalar aggregates

RoiProfileSampler
  image plane + thick polyline -> arclength profile

RoiMeasurementEngine
  request + store snapshot + provider -> immutable RoiMeasurementSnapshot
```

Use floating-point source values and `np.sum(..., dtype=np.float64)`. Do not copy the legacy nuclei-file scaling/truncation convention.

### Stable scalar metrics

| Metric key | Applies to | Unit/meaning |
|---|---|---|
| `intensity.sum` | every mask | integrated arbitrary units |
| `intensity.mean` | every mask | arbitrary units |
| `intensity.median` | every mask | arbitrary units |
| `intensity.sum_per_length_um` | thick line | a.u./µm centerline |
| `intensity.sum_per_area_um2` | 2D polygon and thick-line footprint | a.u./µm² sampled area |
| `intensity.sum_per_volume_um3` | volume and shell band | a.u./µm³ |
| `intensity.sum_per_surface_area_um2` | shell | a.u./µm² estimated parent surface |
| `geometry.length_um` | line | calibrated centerline length |
| `geometry.area_um2` | 2D masks | finite sampled area |
| `geometry.volume_um3` | 3D masks | finite sampled volume |
| `geometry.surface_area_um2` | 3D masks | parent surface area |

Mean and median are already per sampled pixel/voxel and are not divided again by length/area. Always retain nominal geometry, in-bounds raster geometry, finite sample count, and coverage fraction. Sum/mean/median use finite values only; normalized sums divide by finite sampled physical support. No finite samples is missing, not zero.

### Rasterization

- Polygon: include a pixel whose center is inside or on the polygon; rasterize only a clipped bounding box.
- Thick line: include pixel centers within `thickness/2` of any segment; union segment masks to avoid double-counting bends; use round caps and joins.
- Contour-stack volume: rasterize every explicitly annotated consecutive plane into one `(Z,Y,X)` voxel mask.
- Inner shell: `volume_mask & (distance_transform_edt(volume_mask, sampling=(z_res, xy_res, xy_res)) <= thickness_um)`.
- Surface area: marching cubes with physical spacing. Boundary-touching objects remain measurable with a clipping warning rather than being reported as closed.

Add `scikit-image` as a declared direct dependency for deterministic polygon rasterization and marching-cubes surface area. Continue using existing SciPy for distance transforms and interpolated profile sampling. Do not depend on napari internals or transitive packages in headless analysis.

### Line profiles and distributions

Aggregate line masks and spatial profiles are separate algorithms. Resample the centerline at a declared physical step, default one XY pixel; sample across its width no more coarsely than one XY pixel using `scipy.ndimage.map_coordinates(order=1)`. Return distance from the first vertex plus cross-width mean, median, optional sum, and sample count. Reversing vertices reverses the profile but does not change aggregate measurements.

Profiles are vector values and cannot enter the existing scalar `ExpressionChannel` contract. Store them in an immutable `RoiSpatialProfile` and plot/export them in a dedicated profile view. Scalar profile features such as AUC, peak intensity, or peak position may be registered as normal time-series metrics later.

For polygons/volumes, V1 adds histograms and standard quantiles as distribution outputs; per-Z 3D summaries are useful follow-up. Distance-to-boundary/radial profiles are deferred until their scientific semantics are agreed.

### Snapshot, cache, and missing data

An immutable measurement snapshot binds to:

- ROI document ID/revision and per-frame geometry fingerprint;
- image-source manifest token;
- calibration including `plane_start`;
- channel, raster/profile parameters, and algorithm version;
- optional frame-wide dependency token for future background exclusion.

Cache cropped masks by geometry fingerprint and cache raw pixel aggregates separately from cell association. A rename/reassociation updates indexes without pixel recomputation. Group 2D work by `(time, plane, channel)` and call `get_plane()`; load a full stack once per frame/channel only when 3D work requires it. Bound mask caches by bytes rather than retaining full-movie boolean arrays.

Every sample carries a status such as:

```text
valid
valid_clipped
roi_absent
image_unavailable
channel_unavailable
invalid_geometry
empty_mask
no_finite_pixels
calibration_unavailable
association_orphaned
stale
```

Association orphaning does not invalidate an object-centric pixel measurement. At the expression reader boundary, invalid/absent/stale results become `None` and retain a missing reason for UI and CSV export.

## Expression and time plotting

There is no expression-language parser today. The existing interface is a registry of scalar `ExpressionChannel` readers, and `ExpressionPlotService.build()` is hard-coded to `Cell + time + Nucleus` (`analysis/expression_plot.py`).

Preserve current cell behavior while factoring a generic temporal-series core:

```text
TemporalSeriesSubject
  key
  label
  start_time
  end_time
  sample_times

ScalarSeriesChannel
  key
  label
  unit
  reader(subject, time)
  source_token()
  validate_coverage(subjects)
```

`ExpressionPlotService.build(cells, ...)` becomes a compatibility adapter. A new ROI adapter builds subjects from `RoiObjectTrack`. The cell Expression Plot remains unchanged for backward compatibility; the Objects dock opens a dedicated modeless ROI scalar window that reuses the generic temporal-series core, smoothing-with-gaps, stale guards, CSV/SVG export, and absolute/relative/normalized axes.

The dedicated ROI scalar window exposes:

```text
Objects        [Golgi #1 — ABpl, Golgi #2 — ABpr]
Image channel  [Channel 2]
Metric         [Median intensity]
Time axis      [Absolute | Since first segmentation | Normalized track]
```

Channel freshness and coverage become channel-owned validators rather than the current nuclear-only `_measure_issue()` logic. Export snapshots include the channel source token, object/class IDs and names, instance index, source image channel, metric, algorithm version, raw/plotted value, and missing reason.

A cell-centric mode may also expose ROI metrics for frames whose exact cell anchors match the selected cell. Unassociated/orphaned tracks remain available in object-centric mode. Cross-dataset ROI comparisons are deferred, but source fingerprints should include the ROI sidecar when that repository is later extended.

## UI design

Add a dedicated right-side `Subcellular Objects` dock and an `Objects` menu. The current Edit & Tracking dock is already dense, and napari's native layer list is hidden.

```text
SUBCELLULAR OBJECTS

t=42  z=15  Selected cell: ABpl
[Use selected cell] [Pick cell] [Clear association]

Class [Membrane ▼] [+ Manage classes]
[2D Polygon] [Thick Line] [3D Contour Stack]
MODE: INSPECT

☑ Show ROIs  Cell [Current ▼]  Class [All ▼]  State [All ▼]
Search [________________]

Tracks
▾ ✓ Membrane #1  ABpl  t37–52  14/16 reviewed
    t40 ✓  t41 ✓  t42 ● Draft  t43 —  t44 Ø
  ! Junction #3  ABpl  t42–48  Needs review

Selected object
Identity       Membrane #1
Geometry       Thick line, z=15
Thickness      0.80 µm
Association    ABpl
Expected span  t37–52
Frame state    Draft                  [Mark reviewed]
Temporal       [Previous] [Next] [Copy previous] [Mark absent]
Actions        [Edit] [Measure] [Plot track] [Delete frame…]

ROI file: Unsaved — embryo.subcellular-rois.json        [Save]
```

### Layer ownership and drawing

Use two layers:

1. `Subcellular ROI Overlay`: permanent, read-only, rebuilt from the model for current time/Z.
2. `Subcellular ROI Editor`: temporary and editable, containing only the active frame or contour.

This follows `ViewerIntegration`'s existing rule that curated layers are projections, not mutable authority. Napari mutation events are staged and converted to one validated undo command on finish/drag release. A failed redraw restores the prior complete layer state.

Drawing behaviors:

- 2D polygon: click vertices; Enter or double-click closes; Backspace removes the last in-progress vertex; Escape cancels.
- Thick line: click centerline vertices; physical thickness is editable numerically and with `[`/`]` while drawing.
- 3D stack: author closed contours slice-by-slice in the main 2D view. Main/detached 3D views are read-only previews; camera-ray editing is not offered.
- A persistent canvas banner names the active class/object/time/Z and available controls.
- Entering ROI drawing exits Add, Manual Track, Relink, and 3D display. Entering 3D exits drawing with an explanation.
- Space temporarily pans in drawing mode and does not trigger cell deselection. Global letter shortcuts are suppressed in editable controls.

Selecting an ROI does not clear the selected cell; maintain separate `current_roi_object_id` and nucleus selection. New objects default to the selected live cell but may be created visibly unassociated.

### Temporal and review workflow

Keep the object track selected across time navigation:

- existing record: show/edit it;
- missing record: show a dim non-measured ghost and offer Draw, Copy previous, Copy plus associated-cell displacement, or Mark absent;
- copied geometry is always Draft;
- previous/next segmentation skips missing frames;
- expected start/end distinguishes incomplete segmentation from an intentionally short track;
- a division presents explicit daughter/end/fork choices.

Review state is independent of save state:

```text
Missing -> Draw/Copy -> Draft -> Mark reviewed -> Reviewed
Reviewed -> geometry/association/semantics edit -> Needs review
Needs review -> Mark reviewed -> Reviewed
Missing -> Mark absent -> Absent
Absent -> Draw -> Draft
Invalid -> Repair/replace -> Draft
```

Track status is derived as Complete, In progress, or Needs attention. Use text/glyphs as well as color. Deleting one frame and deleting an entire multi-frame track are separate actions; track deletion shows the affected frame count and remains undoable.

### Measurement UX

Create `Measure Subcellular Objects…` under Objects rather than changing nucleus File → Measure. A Basic preset exposes scope, channels, geometry support, sum/mean/median, and the appropriate normalized sums. Advanced reveals line profile step/reducer and distributions. Bulk work is cancellable and publishes no partial snapshot.

`Plot track` opens the dedicated scalar ROI window preselected. `Plot profiles` opens the separate thick-line profile window and supports a single timepoint or multiple-time overlay; a later view can add a time-by-distance heatmap.

### Accessibility and protection

- Accessible text/name/tooltip and keyboard equivalent for every icon action.
- No class/review/stale state communicated by color alone.
- High-contrast class palette plus selected white outline.
- Predictable tab order and focus-aware shortcut suppression.
- Persistent text for mode, validation, save, and measurement status.
- Title-bar dirty marker and close prompt summarizing both streams: Save All, Discard, Cancel.
- V1 keeps explicit manual Save. A later non-authoritative recovery journal may protect long drawing sessions without pretending the dataset is saved.

## Implementation phases

### Phase 1 — contracts, sidecar, and read-only substrate

- Implement model/validation, coordinate conversion, object/class allocation, associations, strict sidecar, migrations, and ROI manager revision.
- Autoload beside XML, quarantine invalid records, render a read-only current-time/current-Z overlay.
- Add command effect domains and coordinated save/rollback.
- Add the ROI dock in browse-only mode and protected malformed-file UX.

Exit criteria: existing datasets open unchanged; absent ROI file is quiet; valid sidecar round-trips losslessly; invalid sidecar cannot be overwritten silently; failure at any save commit step preserves every prior authoritative file.

### Phase 2 — complete 2D vertical slice

- Author/edit polygon and thick-line frames through transient editor layers.
- Implement class/index, same-frame cell association, unified undo/redo, filters, review states, and explicit absence.
- Implement deterministic masks, raw scalar metrics, line profiles, measurement runner, and immutable cache.

Exit criteria: draw → associate → measure all channels → save → reopen reproduces geometry and exact measurements on synthetic images; one drag is one undo action; ROI edits do not stale nuclear measurements.

### Phase 3 — temporal and expression integration

- Track selection across time, expected span, copy/translated-copy drafts, division choices, completeness states.
- Factor generic temporal-series plotting, ROI object source selector, channel validators/source tokens, exact CSV/SVG export, and profile window.

Exit criteria: missing/absent frames render as plot gaps with reasons; zero intensity plots as zero; stale ROI geometry disables export until remeasured; object UUID remains stable through class rename/reindex.

### Phase 4 — 3D volume and inner shell

- Contour-stack editor, consecutive-slice validation, read-only 3D preview.
- Anisotropic volume/shell masks, voxel volume, surface area, scalar metrics, and per-Z summaries.

Exit criteria: calibrated synthetic solids produce volume/shell measurements within declared raster/surface tolerances; missing contours cannot be measured as an implicitly interpolated solid; boundary clipping is visible in results.

### Phase 5 — advanced analysis and resilience

- Explicit local-border backgrounds, optional interpolation drafts with provenance, fork/split tools, bulk QA, cross-dataset ROI comparison, heatmaps, and recovery journal.

## Test plan

New suites:

- `test_roi_model.py`: IDs, class allocation, index collisions, review/presence state machine, cell anchors, rename/relink/kill/division/orphan behavior.
- `test_roi_sidecar.py`: every geometry round-trip, checksum, strict JSON, duplicate keys, nonfinite numbers, limits, migrations, unknown versions, external revision conflict, empty managed store.
- `test_roi_save_transaction.py`: injected failure at each ZIP/AuxInfo/XML/ROI commit and rollback step, Save As behavior, dirty-state preservation.
- `test_roi_rasterization.py`: convex/concave/reversed polygons, boundaries, self-intersection, clipping; horizontal/diagonal/bent lines, repeated vertices, caps/joins.
- `test_roi_measurements.py`: integer/float/large images, constant/XY/Z gradients, valid zero, nonfinite pixels, exact aggregate metrics, physical normalization, missing images/channels, cancellation and no partial publish.
- `test_roi_profiles.py`: constant/linear profiles, width reducers, bends, reversal, endpoint/out-of-bounds gaps.
- `test_roi_3d_measurements.py`: anisotropic solids, contour gaps, inner shells, thickness, surface tolerance, clipping.
- `test_roi_cache.py`: single-frame invalidation, image manifest/calibration/algorithm changes, association-only changes, bounded provider reads.
- `test_roi_expression.py`: object keys, scalar gaps/reasons, stale tokens, smoothing gaps, export metadata, cell-associated and object-centric series.
- `test_subcellular_objects_panel.py`: modes, button state, auto-indexing, filters, association scope, focus/shortcuts, accessible names.
- `test_roi_viewer_integration.py`: time/Z filtering, projection/editor separation, atomic redraw, drag command boundary, mode exclusivity, 3D read-only behavior.
- `test_roi_end_to_end.py`: XML autoload → draw across time/Z → associate → measure → plot → save → reopen equivalence, plus corrupt-sidecar protected recovery.

Reuse the failure-injection and savepoint style in `tests/test_app_save.py`, strict JSON/atomic replace patterns in `tests/test_tracking_persistence.py`, numeric synthetic images in `tests/test_measure.py`, valid-zero/freshness checks in `tests/test_expression_measurements.py`, plot-gap/export tests in `tests/test_expression_plot_model.py`, and napari atomic-layer tests in `tests/test_marker_layers.py`.

## Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| Plane/provider/napari coordinate mismatch | One declared coordinate frame and conversion service; dedicated `plane_start`, split, flip, and anisotropy tests |
| Mutable or duplicate cell names | Exact physical same-frame anchors; names only as hints; explicit orphan/relink UI |
| Native napari edits bypass the model | Read-only authoritative projection plus one transient editor committed through commands |
| ROI edits invalidate unrelated nuclear results | Independent ROI revision and command effect domains |
| Partial save loses high-effort annotations | Stage all authoritative artifacts, rollback on failure, protect malformed/external changes |
| Ambiguous 3D/shell semantics | V1 contour stacks, consecutive planes, inner physical shell, parameters in provenance |
| Vector profiles forced into scalar expression channels | Separate profile type/window; only scalar reducers use time-series channels |
| Missing treated as biological zero | `None` plus reason end-to-end; zero remains valid |
| Rendering/performance degrades with long movies | Current-time/current-Z projection, cropped masks, time-major image reads, byte-bounded cache |

## Acceptance definition

The feature is complete when a user can create several same-class indexed objects, manually segment them over time in 2D or 3D, associate each observation to a same-frame cell, review them, measure every image channel, inspect line profiles, plot scalar metrics over time, save all authoritative annotations atomically beside the XML, and reopen the dataset with identical identities, geometry, associations, review states, and reproducible measurement inputs. Existing nuclei, tracking, measurement, expression, and save workflows must remain backward compatible.
