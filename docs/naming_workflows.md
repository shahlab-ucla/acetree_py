# Naming and Manual-Curation Workflows

This document is the cross-cutting contract for automatic lineage naming, manual overrides, track edits, body-axis correction, undo/redo, and persistence. It is intentionally user-task oriented: the same invariants must hold whether a change begins in the image viewer, lineage tree, edit panel, or a command-level test.

## 1. The Three Name Concepts

A nucleus never has a single mutable “name” with mixed ownership.

| Concept | Field/expression | Owner | Persistence |
|---|---|---|---|
| Automatic/current identity | `identity` | Naming pipeline | May be recomputed after geometry or topology changes |
| Forced identity | `assigned_id` | User | Survives automatic renaming and ZIP round trips |
| Effective name | `assigned_id if assigned_id else identity` | Derived | Used everywhere a current name is needed |

Consequences:

- Automatic division suggestions write `identity`, never `assigned_id`.
- Add and Track may continue the current automatic identity, but they do not lock it. They inherit a forced override only when the parent already has `assigned_id`.
- Rename is an explicit ownership change from automatic to user-forced.
- **Lock Current Name** is the explicit way to keep an already-correct automatic name without changing its text.
- **Use Automatic** is an explicit ownership change back to the pipeline: it clears `assigned_id`, reruns naming, and remains undoable.
- A rename equal to the existing effective name is a no-op. Merely accepting a dialog must not freeze an automatic name.
- UI labels, tree construction, cell lookup, kill selection, validation, exports, and division-parent rule lookup use `effective_name`.

## 2. Cell Continuations and Forced-Name Propagation

A cell is a chain of nuclei at consecutive timepoints, bounded by birth and division/disappearance. A continuation edge is valid only if all of the following are true:

1. Both nuclei are alive.
2. The timepoints are consecutive.
3. The earlier nucleus has exactly one relevant live successor.
4. That successor is the later nucleus.
5. The later nucleus's predecessor points back to the earlier nucleus.

Forced-name propagation follows only these reciprocal continuation edges. It stops at:

- a division (two live successors);
- a dead nucleus;
- a missing or non-consecutive endpoint;
- a non-reciprocal link;
- an ambiguous successor set; or
- a different non-empty `assigned_id`.

A conflict is data requiring correction, not a tiebreaking opportunity. Iteration order must never overwrite one user's forced identity with another. In particular, automatic naming does not append an `X` or mutate a visible cell name to hide a duplicate.

### Canonical daughter-family invariant

For every named parent with exactly two alive, distinct, reciprocal successors,
the automatic effective daughter set is the exact unordered pair returned by
`RuleManager(parent.effective_name)`. This holds across founder recovery,
partial movies, tracking commits, reloads, and local body-axis dropout. Timing
and geometry order the pair over the two successors; they do not select the
family. When ordering evidence is absent, AceTree preserves a compatible exact
loaded order or uses stable successor-slot order and emits a low-confidence
warning. Consequently, a valid division cannot turn `ABa`, `ABp`, `EMS`, or
`P2` descendants into unrelated `Nuc...` roots.

“Build on the predecessor” means follow its RuleManager lineage rule, not always
append a literal suffix. Canonical exceptions include `P0 → AB/P1`,
`P1 → EMS/P2`, `EMS → E/MS`, `P2 → C/P3`, `P3 → D/P4`, and
`P4 → Z2/Z3`. A malformed/non-reciprocal link is not evidence of a division
and fails closed. A conflicting `assigned_id` remains an explicit curator-owned
exception and is reported rather than overwritten.

## 3. Edit-Commit Contract

### Stable selection

The selected object is anchored by `(timepoint, nucleus index)`, not by its mutable name. Reprocessing can change `identity`, rebuild cells, or rearrange lookup maps without changing which record the user selected. Changing z-plane preserves selection. Explicit **Deselect**, deleting the selected record, or leaving the dataset clears it.

### One gesture, one history entry

One intentional user gesture is atomic:

| Gesture | Mutations grouped into the same undo step |
|---|---|
| Add across a gap | Add endpoint, add all interpolated nuclei, set every reciprocal link, apply automatic daughter state |
| Track placement | Add/interpolate/link, advance tracking anchor, apply automatic daughter state |
| Relink across a gap | Detach old link, add intermediates, attach new link, restore reciprocal successors |
| Rename / Lock Current Name / Use Automatic | Change name ownership over the entire valid cell continuation |
| Apply body axes | Replace orientation metadata, invalidate/rebuild naming, refresh views |
| Remove a false polar detection | Kill the row, rebuild the early-stage hypothesis, and retain every resulting automatic-name change in the same history boundary |

If validation fails, none of the gesture commits. Undo reverses a composite in reverse mutation order; Redo replays it in forward order.

### Dirty state and saving

Dirty state is a comparison with an explicit savepoint, not `len(undo_stack) > 0`.

- Successful Save and Save As mark the current state saved.
- Undoing away from the saved state is dirty.
- Redoing exactly back to it is clean.
- Editing after Undo discards the redo branch and creates a distinct dirty state even when stack lengths match.
- A failed save does not move the savepoint.
- Save stages the complete ZIP and manual AuxInfo sidecar before committing either one. If a commit fails, the previous pair is restored; replacement also retains existing file permissions.
- Save As atomically rewrites the source XML only after the new data set is durable, so reopening the same config follows the new ZIP. A config-write failure leaves the current target and history savepoint unchanged.
- Save As updates the active nuclei path, so the next Save targets the chosen archive.
- A manual body-axis frame is saved in the adjacent AuxInfo v2 sidecar with provenance, quality, and reference time. An unusable v2 frame cannot mask a valid supported v1 orientation on reload.

## 4. Human Workflows

### Rename a known cell

1. Select the cell at any time during its lifetime.
2. Choose **Rename**, enter the known biological identity, and confirm.
3. AceTree trims and validates the name. Commas, CR/LF, and control characters are rejected because nucleus data is CSV-compatible.
4. The forced identity is applied to the valid continuation from birth to the next division/disappearance.
5. Automatic naming reruns downstream, using the effective forced identity as the division parent.

If another disconnected cell already has the target identity, resolve the explicit collision (for example, by an intentional Swap) rather than creating an alias.

### Keep an already-correct automatic name

1. Select the automatically named cell at any time during its lifetime.
2. Choose **Lock Current Name**. This is intentionally separate from accepting the unchanged, pre-filled Rename dialog, which remains a no-op.
3. AceTree checks that the current name is non-empty, safe to save, and not duplicated on a disconnected live cell.
4. The current name becomes a manual override over the valid continuation, and automatic naming uses that forced parent name for downstream divisions.
5. Undo restores the exact prior automatic state; **Use Automatic** later releases the lock.

### Return a cell to automatic naming

1. Select a manually forced cell.
2. Choose **Use Automatic**.
3. AceTree clears the continuation's `assigned_id` and recalculates `identity` from current topology, body axes, and division rules.
4. Undo restores the exact prior automatic and forced fields.

### Extend a track

1. Select the last trusted nucleus.
2. Enter Track mode and navigate to a later frame.
3. Place the next nucleus. The selected parent and target time determine whether this is a continuation or division.
4. If the gap is longer than one frame, AceTree inserts physically interpolated nuclei and reciprocal links as one edit.
5. Escape or toggling Track exits cleanly. Root Track places one independent nucleus and exits after that placement.

If a second daughter is created, AceTree requests a division suggestion for the actual parent. A prediction is not a forced name.

### Curate a forced anchor into an automatic sublineage

The canonical example is a focused EMS trace:

1. Apply a manual body frame at a clear reference time using AP plus either DV or LR endpoints.
2. Rename one trusted continuation to `EMS`. This writes the only required forced identity.
3. Track EMS forward. The first placed successor temporarily remains part of the forced EMS continuation.
4. Place the sister at the same division frame. The atomic division gesture moves the inherited forced state back behind the new division boundary, calls the EMS rule in the current body frame, and writes `E`/`MS` as automatic identities.
5. Select E and begin a new Track gesture. When both E daughters have been placed, the E rule writes `Ea`/`Ep` automatically. Repeat from any daughter to curate a deeper branch.

This late-start workflow does not require successful four-cell founder discovery. A valid manual AuxInfo v2 frame plus the forced lineage anchor is sufficient to run canonical division rules forward. Reapplying corrected axes recomputes automatic descendants while preserving the EMS override. Track remains anchored to the cell selected when the mode began, so the curator explicitly selects a daughter after each division; this prevents the UI from silently choosing which biological branch to follow.

At a terminal frame, the placement gesture treats a click within the existing nucleus diameter as continuation and a farther click as the second daughter. For tightly apposed newborn daughters, use a later frame with clearer separation or place and Relink explicitly. Orientation selects which daughter receives each biological name; reciprocal topology is what establishes the division.

### Correct a tracking link

1. Select either endpoint and enter Relink pick mode.
2. Navigate freely in time and z, then select the other endpoint.
3. AceTree orders endpoints by time, checks successor capacity and forced-name compatibility, and previews the operation.
4. Confirm to commit the detach/interpolation/attach gesture atomically.
5. Escape cancels pick mode and re-enables ordinary selection.

A relink cannot use a dead endpoint, create a non-forward edge, overfill a parent beyond two children, or merge incompatible forced identities.

### Kill or resurrect

Kill begins from the selected `(time,index)` anchor and walks that component. It uses `effective_name` for user-facing identity and does not kill a disconnected same-named cell. Resurrect operates on an explicitly chosen dead nucleus and rejects live targets. Its command records `status`, `identity`, and `assigned_id`, so Undo/Redo cannot lose a forced name.

### Correct a false four-object two-cell stage

1. At the biological two-cell stage, select one small polar-body false
   detection and choose **Remove Nucleus**.
2. Remove the second small false detection. Dead rows are retained for stable
   legacy indices, but are excluded from founder counting.
3. AceTree invalidates the old unforced four-cell hypothesis only when the two
   deleted rows have the strong small-object footprint expected for polar
   bodies. Retained lineage ancestry is traced through continuation frames;
   a real four-cell sister-pair topology, including an ablation of small
   founders, retains its founder names. Malformed claimed topology fails closed.
4. `AB` versus `P1` is resolved from observed division timing, then a trusted
   AP direction, then blastomere-size asymmetry. Rejected four-cell labels are
   never used as independent evidence.
5. If AB/P1 root ordering is still ambiguous, the survivors and their
   unforced descendants receive neutral names. If the roots resolve, every
   subsequent valid reciprocal division remains in its predecessor's exact
   RuleManager family, beginning with `ABa`/`ABp` and `EMS`/`P2`. Missing body
   axes can lower confidence in which sister receives which name, but cannot
   turn a named lineage into unrelated `Nuc...` roots.

Use **Remove Nucleus** for isolated false detections. Use **Kill Cell** when an
entire tracked continuation is spurious. Automatic renaming and the live/dead
mutation share one history boundary, so Undo/Redo restores the exact prior or
corrected state. Explicit `assigned_id` locks are never replaced.

## 5. Body-Axis Model

### Direction conventions

- AP: posterior → anterior.
- DV: ventral → dorsal.
- LR: right → left.
- Handedness: `DV = AP × LR`.

Coordinates are physical: before vector arithmetic, stored z planes are multiplied by `z_pix_res`. This is essential because typical microscopy stacks have different in-plane and inter-plane spacing.

### Automatic four-cell geometry

At a timepoint with ABa, ABp, EMS, and P2 lineage groups, compute physical centroids and use:

```
AP seed = centroid(ABa) - centroid(P2)
DV seed = centroid(ABp) - centroid(EMS)
DV      = normalize(DV seed projected perpendicular to AP)
LR      = DV × AP
```

The same construction is used per timepoint when lineage groups are available.
Across the valid four-cell window, AceTree also selects the complete frame with
the highest perpendicular secondary-axis quality (earliest wins an exact tie)
and retains it as a static fallback. Dynamic axes are preferred because they
follow embryo motion; the retained frame handles a missing lineage group,
degenerate local geometry, or a temporary quality dropout. Chronological
caching makes results independent of the order in which frames happen to be
requested.

ABa–ABp is **not** the LR axis. Moreover, the signed biological LR direction cannot be established from that pair alone at the four-cell stage. Trusted orientation metadata, manually labeled anatomy, or later embryonic handedness is needed to ground the sign. Compression can distort relative positions and should lower confidence rather than produce silently authoritative names.

### Orientation precedence

1. Valid explicit AuxInfo v2, including a user-applied manual frame.
2. Supported legacy AuxInfo v1 orientation (`ADL`, `AVR`, `PDR`, `PVL`).
3. Per-timepoint lineage-centroid frame.
4. Static founder-derived frame.
5. If no defensible frame exists, report ambiguity and use conservative fallback naming rather than pretending image +x is anatomically anterior.

Placeholder metadata such as `XXX`, zero-length vectors, or parallel AP/secondary vectors is invalid.

AP alone is not a complete anatomical frame. If topology identifies the
founders but DV/LR is unavailable, AceTree retains the trusted founder
identities and any compatible loaded downstream order. Once AB/P1 are resolved,
their rules produce `ABa`/`ABp` and `EMS`/`P2`; once the quartet exists, the
best complete four-cell frame is retained and dynamic lineage axes take
precedence when usable. At any later valid reciprocal division, an unavailable
axis defers only the sister ordering: AceTree preserves an exact loaded order
or uses deterministic successor order with a low-confidence warning. It never
substitutes unrelated `Nuc...` names for a named parent's RuleManager family.
`Nuc...` is reserved for genuinely unknown/disconnected roots and topology that
is too malformed to establish a reciprocal division. Applying a valid manual
frame reruns naming and can improve or correct automatic sister ordering.

## 6. Manual Body-Axis Labeling and Correction

The manual workflow uses endpoint labels instead of asking a curator for Euler angles or matrix components.

1. Choose a clear reference timepoint.
2. Label selected nuclei/positions **Posterior** and **Anterior**.
3. Label either **Ventral** and **Dorsal**, or **Right** and **Left**.
4. Apply the frame.

Definitions:

```
AP = anterior - posterior
DV = dorsal - ventral
LR = left - right
```

AP is required. One complete secondary pair is required. The builder applies physical z scaling, projects the secondary vector perpendicular to AP, normalizes the vectors, constructs the third axis, and verifies handedness. All endpoints must come from the same reference timepoint.

The panel shows the current source, reference time, and quality. To correct a reversed axis, relabel or swap that endpoint pair and Apply again. The operation is one undo step. Applying body axes reruns automatic naming but preserves explicit cell-name overrides.

## 7. Parent-Specific Division Suggestions

Manual placement must use the same biological rules as batch naming. The suggestion service receives:

- the actual parent nucleus/effective parent name;
- both raw daughter positions;
- the 1-based division time;
- current orientation and `z_pix_res`.

It returns both daughter names, confidence, governing axis label, source/provenance, and an ambiguity flag. This supports founder-specific pairs such as P0 → AB/P1, EMS → E/MS, and P2 → C/P3 as well as later `a/p`, `d/v`, or `l/r` divisions. The unordered pair always comes from RuleManager; geometry only orders it. If ordering evidence is unavailable, stable successor order is returned with low confidence rather than a foreign family. Suggested names are written only as automatic identity. A forced daughter is respected; a compatible automatic sister assignment can be swapped. Two incompatible forced daughters remain a validation error.

## 8. Edge-Case Acceptance Matrix

| Scenario | Required result |
|---|---|
| Accept Rename without changing an automatic name | No command; `assigned_id` remains empty |
| Lock Current Name on an automatic cell | Persist the visible name over its continuation; one Undo restores the automatic state |
| Lock Current Name when a disconnected cell has the same name | Reject without creating a forced-name conflict |
| Rename contains surrounding spaces | Trim once, then validate and apply |
| Rename contains comma/newline/control byte | Reject before mutation |
| Use Automatic on forced cell | Clear override over valid continuation; recompute; one Undo restores |
| Forced seed meets a different forced name | Stop and report conflict; neither wins by scan order |
| Continuation passes through a dead nucleus | Stop at dead boundary |
| Child points to parent but parent does not point back | Stop propagation; validation reports non-reciprocal edge |
| Parent has two live successors | Stop parent-name propagation at division |
| Automatic sister result conflicts with one forced daughter | Preserve forced daughter and use compatible complement when possible |
| Both daughters forced incompatibly | Preserve both assertions and report conflict |
| Add/Track from automatically named parent | Do not populate `assigned_id` |
| Add/Track from forced parent continuation | Carry the forced state only within the same continuation |
| Manual second daughter of EMS or P2 | Use parent-specific E/MS or C/P3 rule, not hard-coded a/p |
| Forced EMS in a partial lineage with manual axes | Name EMS daughters E/MS and the tracked E daughters Ea/Ep; keep only EMS forced |
| Division primarily separated in z | Classification uses `z * z_pix_res` |
| Move changes division geometry | Naming reruns; selection stays on anchored nucleus |
| Relink across several frames | One undo removes all interpolants and restores both endpoint links |
| Relink would merge different forced identities | Reject without partial mutation |
| Escape during Relink/Add/Track | Cancel mode and restore ordinary controls |
| Kill forced-name cell | Resolve via effective name and anchored component |
| Same effective name exists in disconnected components | Expose collision; do not create `_2` tree alias or kill both |
| Resurrect live nucleus | Reject |
| Resurrect forced dead nucleus | Restore status and forced state; Undo returns exact dead state |
| Change z-plane | Keep selection |
| Rebuild after automatic rename | Re-resolve selection by `(time,index)`, not old name |
| AP and secondary landmarks are parallel | Reject frame with actionable explanation |
| Landmarks span different reference times | Reject frame |
| Only ABa and ABp are used to claim LR | Treat as insufficient anatomical evidence |
| Founder topology is strong but no complete AP/DV/LR frame exists | Keep founder/loaded names; assign exact RuleManager families at valid divisions; use stable successor order with a low-confidence warning when no geometry can order the sisters |
| Two small polar detections are removed from a false four-object two-cell frame | Replace stale unforced four-cell labels with AB/P1 from timing/AP/size evidence; otherwise use neutral names; one full Undo restores names and status |
| Recovered AB or P1 reaches its first division | Use the exact `ABa`/`ABp` or `EMS`/`P2` pair; preserve a valid loaded pair or warn when stable successor order is required |
| Reopen a dataset saved with repaired AB/P1 roots but legacy `Nuc...` daughters | Upgrade those first daughters from the retained polar-body footprint without requiring another delete gesture |
| Valid four-cell window has one degenerate frame and another complete frame | Retain the best complete frame; prefer dynamic axes and use the retained frame during later dropout |
| Later dynamic lineage axes are missing or low quality | Fall back to the retained four-cell frame; retain/log its source time and report the division's ordinary angle-based confidence |
| Named parent has a valid reciprocal division but no usable axis | Assign its exact RuleManager pair in deterministic successor order and warn; never create unrelated `Nuc...` daughters |
| Automatic classifier returns a foreign or empty daughter pair | Replace it with the effective parent's RuleManager pair and warn; geometry/classifier failure may lower ordering confidence but not erase the family |
| Division links are dead, non-reciprocal, duplicated, or out of range | Fail closed; do not synthesize a biological daughter pair from malformed topology |
| Two founders are absent from a real four-cell sister-pair lineage | Preserve the surviving founder identities, even in later continuation frames or when the absent rows are small; do not invoke polar recovery |
| Save fails midway | Old ZIP remains usable; history stays dirty |
| Save As succeeds, then Save | Second save targets the new path |
| Undo back to savepoint | Clean; redo away from it becomes dirty |
| Edit after Undo at same stack depth as savepoint | Dirty because it is a new branch |

## 9. Biological Basis and Limits

The invariant embryonic lineage and canonical names originate with [Sulston et al. (1983)](https://www.wormatlas.org/papers/Sulston_embryonic_lineage_1983.pdf). Automated lineaging and geometry-based identity assignment are described by [Bao et al. (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1413828/), while the practical lineaging protocol and supported legacy orientations are documented by the [Murray/Bao protocol](https://cshprotocols.cshlp.org/content/2012/8/pdb.prot070615.full). [Pohl and Bao (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2952354/) describe early embryonic chirality, which is relevant to signed left/right interpretation. [Hench et al. (2009)](https://pubmed.ncbi.nlm.nih.gov/19527702/) show why mechanical compression should be treated as a source of geometric distortion. The [AceTree update and AuxInfo v2 description](https://pmc.ncbi.nlm.nih.gov/articles/PMC5885296/) provides the orientation-metadata context used for persistence and compatibility.

Geometry is evidence, not ground truth. A robust curator workflow therefore combines explicit provenance and confidence, automatic previews, stable manual overrides, and a simple reversible orientation correction rather than making every uncertain prediction permanent.
