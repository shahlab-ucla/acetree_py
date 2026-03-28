# AceTree-Py Algorithm Reference

Mathematically precise descriptions of the Sulston naming system, coordinate transforms, division classification, undo mechanism, and editing operations.

---

## 1. Sulston Naming System

### 1.1 Naming Convention

In *C. elegans*, every somatic cell is uniquely identified by its lineage history. Each cell division appends a letter indicating the division axis and the daughter's position along that axis:

| Letter | Axis                | Meaning              | Complement |
|--------|---------------------|----------------------|------------|
| `a`    | Anterior-Posterior  | Anterior daughter    | `p`        |
| `p`    | Anterior-Posterior  | Posterior daughter   | `a`        |
| `d`    | Dorsal-Ventral      | Dorsal daughter      | `v`        |
| `v`    | Dorsal-Ventral      | Ventral daughter     | `d`        |
| `l`    | Left-Right          | Left daughter        | `r`        |
| `r`    | Left-Right          | Right daughter       | `l`        |

**Daughter name generation:**

Given parent name $P$ and Sulston letter $s$:
$$\text{daughter}_1 = P \| s, \quad \text{daughter}_2 = P \| \overline{s}$$

where $\overline{s}$ denotes the complement of $s$.

**Example:** Parent `ABa` divides along the LR axis → daughters `ABal` (left) and `ABar` (right).

### 1.2 Founder Cell Hierarchy

```
P0 ─┬─ AB ─┬─ ABa
    │      └─ ABp
    └─ P1 ─┬─ EMS ─┬─ E
            │       └─ MS
            └─ P2 ──┬─ C
                     └─ P3 ─┬─ D
                             └─ P4 ─┬─ Z2
                                     └─ Z3
```

The first ~5 divisions have special names (P0, AB, P1, EMS, P2, etc.) rather than letter-based names.

---

## 2. Naming Pipeline

### 2.1 Overall Flow

```
Input: nuclei_record[t][i] for all timepoints t, nucleus index i
       AuxInfo (v1 or v2 orientation), naming_method

Step 1:  Clear non-forced names
         ∀ nuc: if assigned_id = "": identity ← ""

Step 2:  Build CanonicalTransform (if AuxInfo v2)

Step 3:  Topology-based founder identification
         → FounderAssignment with ABa, ABp, EMS, P2 indices + confidence

Step 4:  If Step 3 fails (confidence < 0.3): legacy InitialID fallback

Step 5:  Set up DivisionCaller with coordinate axes

Step 6:  Forward pass — apply canonical rules:
         for t = four_cell_time to ending_index:
           for each nucleus nuc at time t:
             if nuc has no name and has a predecessor:
               parent = predecessor at t-1
               if parent is NOT dividing: nuc.identity ← parent.identity
               if parent IS dividing:
                 (d1, d2) = DivisionCaller.assign_names(parent, daughter1, daughter2)
                 daughter1.identity ← d1
                 daughter2.identity ← d2

Step 7:  Assign generic names to remaining unnamed nuclei
         name = "Nuc{time:03d}_{z}_{x}_{y}" (3-digit zero-padded, matching Java format)
```

### 2.2 Pre-assigned Name Handling

If a nucleus has `assigned_id` set (manual override), its identity is forced. When both daughters of a division have pre-assigned names, the automatic classification is skipped. If only one daughter has a pre-assigned name, the other receives the complement name.

If automatic naming would produce a name collision with an existing `assigned_id`, the automatic name gets an `"X"` suffix appended.

---

## 3. Topology-Based Founder Identification

### 3.1 Four-Cell Window Detection

A **four-cell window** is a contiguous range of timepoints $[t_\text{first}, t_\text{last}]$ where exactly 4 alive, non-polar-body nuclei exist:

$$\forall t \in [t_\text{first}, t_\text{last}]: \quad |\{n \in \text{nuclei}(t) : n.\text{status} \geq 1 \wedge n.\text{size} < \text{polar\_size}\}| = 4$$

The midpoint is: $t_\text{mid} = \lfloor (t_\text{first} + t_\text{last}) / 2 \rfloor$

Minimum window duration: `MIN_FOUR_CELL_FRAMES = 2`.

### 3.2 Sister Pair Identification

Given 4 alive nuclei $\{n_0, n_1, n_2, n_3\}$, there are 3 possible sister pairings:

$$\text{pairings} = \{(01, 23),\ (02, 13),\ (03, 12)\}$$

Each pairing is scored by tracing cells backward through predecessor links:

**Primary scoring (shared parent):** For each pair $(n_i, n_j)$, trace both backward. If they share a common parent at some timepoint, score $+2$. If their birth times match, score $+1$.

**Fallback (birth time grouping):** If primary scoring is inconclusive, group cells by birth time. Cells born at the same time are sisters.

**Forward division pairing (for datasets starting at 4-cell stage):**

When backward tracing fails (no predecessor data), look *forward*:

$$\forall n_i: \quad t_\text{div}(n_i) = \text{first time } n_i \text{ has two successors}$$

Group cells that divide within 1 frame of each other:

$$\text{pair}_A = \{n_i, n_j\} \text{ where } |t_\text{div}(n_i) - t_\text{div}(n_j)| \leq 1$$

### 3.3 AB vs P1 Pair Assignment

**Biological invariant:** In *C. elegans*, AB daughters (ABa, ABp) divide **before** P1 daughters (EMS, P2) at the 4→8 cell transition.

Given two sister pairs, the pair with the earlier division time is the AB pair:

$$t_A = \min(t_\text{div}(\text{pair}_A)), \quad t_B = \min(t_\text{div}(\text{pair}_B))$$

$$\text{AB pair} = \begin{cases} \text{pair}_A & \text{if } t_A \leq t_B \\ \text{pair}_B & \text{otherwise} \end{cases}$$

### 3.4 Within-Pair Assignment

**P1 pair (EMS vs P2):** EMS is typically larger than P2.

$$\text{EMS} = \arg\max_{n \in \text{P1 pair}} n.\text{size}$$

**AB pair (ABa vs ABp):** Determined by projection onto the AP axis vector. The AP direction is estimated from the centroid of the AB pair toward the P1 pair (specifically, AB centroid − P2 position). ABa is the daughter with the larger projection onto this AP vector (more anterior):

$$\vec{u}_\text{AP} = \frac{\vec{c}_\text{AB} - \vec{r}_{P2}}{\|\vec{c}_\text{AB} - \vec{r}_{P2}\|}$$
$$\text{ABa} = \arg\max_{n \in \text{AB pair}} (\vec{r}_n \cdot \vec{u}_\text{AP})$$

This projection-based method is robust regardless of embryo orientation in the image frame, unlike the legacy approach which used raw image-X coordinates.

### 3.5 Confidence Calculation

The overall confidence combines three factors:

$$C = C_\text{timing} \times C_\text{size} \times C_\text{axis} - \text{penalty}$$

**Timing confidence** from backward trace:

$$C_\text{timing} = \begin{cases}
\text{(from forward pairing)} & \text{if timing gap} = 0 \\
0.6 & \text{if timing gap} = 1 \\
\min(1.0,\ 0.6 + \text{gap} \times 0.1) & \text{if timing gap} \geq 2
\end{cases}$$

**Forward pairing confidence** (when backward trace yields gap = 0):

$$C_\text{timing}^\text{fwd} = \begin{cases}
\min(1.0,\ 0.5 + \text{fwd\_gap} \times 0.1) & \text{if fwd\_gap} \geq 1 \\
0.5 & \text{if fwd\_gap} = 0
\end{cases}$$

**Size confidence:**

Let $s_\text{diff}$ = absolute size difference between the larger and smaller cells in the P1 pair, and $s_\text{sum}$ = sum of their sizes:

$$C_\text{size} = \min\left(1.0,\ 0.5 + \frac{s_\text{diff}}{s_\text{sum}}\right)$$

**Axis confidence:**

$$C_\text{axis} = \begin{cases} 1.0 & \text{if axes successfully determined} \\ 0.0 & \text{otherwise} \end{cases}$$

**Threshold:** Confidence must be ≥ 0.3 for the identification to be accepted.

### 3.6 Back-Tracing

Once the 4 founders are identified, trace backward through predecessor links:

1. Trace ABa backward → find AB (where ABa's predecessor has two successors).
2. At the AB division point: the predecessor is P0 dividing → name AB and P1 (the other successor).
3. Trace P0 backward through continuation cells (predecessors with single successor).
4. Trace EMS and P2 backward → confirm P1 (their shared predecessor).

**Datasets starting at the 4-cell stage:** When the dataset begins at or near the 4-cell stage, AB and P1 may not exist as distinct cells. The back-trace handles this by checking whether the ABa trace actually confirmed an AB cell before treating a division signal as the AB/P0 split. If ABa's trace did not find AB (because the data starts too late), ABp's trace continues naming predecessors as "ABp" continuations rather than falsely identifying them as "AB". The same logic applies to the P2/P1 pair relative to EMS.

---

## 4. Coordinate Transforms

### 4.1 Canonical Frame

The canonical coordinate system is defined as:

$$\vec{e}_\text{AP} = (-1, 0, 0), \quad \vec{e}_\text{LR} = (0, 0, 1), \quad \vec{e}_\text{DV} = \vec{e}_\text{AP} \times \vec{e}_\text{LR} = (0, 1, 0)$$

### 4.2 v2 Transform (Wahba's Problem)

Given measured AP vector $\vec{a}$ and LR vector $\vec{l}$ from AuxInfo v2:

1. Normalize: $\hat{a} = \vec{a}/\|\vec{a}\|$, $\hat{l} = \vec{l}/\|\vec{l}\|$
2. Compute DV: $\hat{d} = \frac{\hat{a} \times \hat{l}}{\|\hat{a} \times \hat{l}\|}$
3. Form source basis: $S = [\hat{a}; \hat{d}; \hat{l}]$ (3×3, rows = vectors)
4. Form target basis: $T = [\vec{e}_\text{AP}; \vec{e}_\text{DV}; \vec{e}_\text{LR}]$
5. Solve for rotation $R$ minimizing $\sum_i \|T_i - R(S_i)\|^2$ via SVD (Wahba's problem).
6. Validate: $\|R(\hat{a}) - \vec{e}_\text{AP}\| < \epsilon$ and $\|R(\hat{l}) - \vec{e}_\text{LR}\| < \epsilon$ with $\epsilon = 10^{-4}$.

**Application:** For any measured vector $\vec{v}$: $\vec{v}_\text{canonical} = R(\vec{v})$.

Implemented via `scipy.spatial.transform.Rotation.align_vectors()`.

### 4.3 v1 Transform (Sign-Flip + Rotation)

Given axis string (e.g., `"ADL"`) and rotation angle $\theta$:

1. Build sign matrix $M$ from axis string:
   - Position 0: `A` → $m_{11} = +1$, `P` → $m_{11} = -1$ (AP/x-axis)
   - Position 1: `D` → $m_{22} = +1$, `V` → $m_{22} = -1$ (DV/y-axis)
   - Position 2: `L` → $m_{33} = +1$, `R` → $m_{33} = -1$ (LR/z-axis)

2. Apply 2D rotation in the XY plane:
$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

3. Apply sign flips: $\vec{v}_\text{corrected} = M \cdot (x', y', z)^T$

### 4.4 Per-Timepoint Lineage Centroid Axes (Primary No-AuxInfo Mode)

When no AuxInfo is available, axes are derived **at each timepoint** from the spatial distribution of lineage-labelled cells. This is the primary coordinate transform mode when AuxInfo is absent, and is inherently robust to global embryo rotations during imaging (common in compressed embryos).

**Lineage map construction** (`naming/lineage_axes.py`):

A lineage label (ABa, ABp, EMS, or P2) is assigned to every nucleus by propagating founder identity through predecessor/successor chains:

1. Seed the 4 founders at the 4-cell midpoint with their labels.
2. Back-propagate from 4-cell time to $t=0$: each nucleus inherits the label of its successor.
3. Forward-propagate from 4-cell time to the end: each successor inherits its predecessor's label (both daughters of a dividing cell get the same lineage label).

**Axis computation at timepoint $t$:**

Let $\mathcal{A}_a(t), \mathcal{A}_p(t), \mathcal{E}(t), \mathcal{P}_2(t)$ be the sets of alive labelled cells at time $t$.

1. **Group centroids:**
$$\vec{c}_\text{AB}(t) = \text{mean}(\mathcal{A}_a(t) \cup \mathcal{A}_p(t)), \quad \vec{c}_{P1}(t) = \text{mean}(\mathcal{E}(t) \cup \mathcal{P}_2(t))$$
$$\vec{c}_\text{ABa}(t) = \text{mean}(\mathcal{A}_a(t)), \quad \vec{c}_\text{ABp}(t) = \text{mean}(\mathcal{A}_p(t))$$

2. **AP axis** (P1 → AB direction):
$$\vec{u}_\text{AP}(t) = \frac{\vec{c}_\text{AB}(t) - \vec{c}_{P1}(t)}{\|\vec{c}_\text{AB}(t) - \vec{c}_{P1}(t)\|}$$

3. **LR axis** (ABp → ABa, projected perpendicular to AP):
$$\vec{s}(t) = \vec{c}_\text{ABa}(t) - \vec{c}_\text{ABp}(t)$$
$$\vec{s}_\perp(t) = \vec{s}(t) - (\vec{s}(t) \cdot \vec{u}_\text{AP}(t))\, \vec{u}_\text{AP}(t)$$
$$\vec{u}_\text{LR}(t) = \frac{\vec{s}_\perp(t)}{\|\vec{s}_\perp(t)\|}$$

4. **DV axis** (completes right-handed frame):
$$\vec{u}_\text{DV}(t) = \vec{u}_\text{AP}(t) \times \vec{u}_\text{LR}(t)$$

**Division vector projection:**

Given a raw division vector $\vec{d}$ at timepoint $t$, project onto the local axes:
$$\vec{d}_\text{canonical}(t) = (-\vec{d} \cdot \vec{u}_\text{AP}(t),\ \vec{d} \cdot \vec{u}_\text{DV}(t),\ \vec{d} \cdot \vec{u}_\text{LR}(t))$$

**Why per-timepoint?** In compressed embryos, the embryo can rotate around its AP axis during imaging. A static axis estimate from the 4-cell stage becomes progressively incorrect. By re-deriving axes from the *current* positions of ABa-lineage vs ABp-lineage centroids at each timepoint, the system automatically tracks these rotations.

### 4.5 Static Founder-Derived Transform (Legacy Fallback)

Used only as a fallback when the lineage centroid approach fails (e.g., too few labelled cells at a given timepoint). Axes are derived once from the 4-cell positions:

Let $\vec{r}_a, \vec{r}_b, \vec{r}_e, \vec{r}_p$ be the 3D positions (with z scaled by `z_pix_res`) of ABa, ABp, EMS, P2 respectively.

1. **AB centroid:**
$$\vec{c}_\text{AB} = \frac{\vec{r}_a + \vec{r}_b}{2}$$

2. **AP axis** (anterior-posterior):
$$\vec{u}_\text{AP} = \frac{\vec{c}_\text{AB} - \vec{r}_p}{\|\vec{c}_\text{AB} - \vec{r}_p\|}$$

3. **AB separation projected perpendicular to AP:**
$$\vec{s} = \vec{r}_a - \vec{r}_b$$
$$\vec{s}_\perp = \vec{s} - (\vec{s} \cdot \vec{u}_\text{AP})\, \vec{u}_\text{AP}$$

4. **DV axis** (dorsal-ventral):
$$\vec{u}_\text{DV} = \frac{\vec{u}_\text{AP} \times \vec{s}_\perp}{\|\vec{u}_\text{AP} \times \vec{s}_\perp\|}$$

5. **LR axis** (left-right):
$$\vec{u}_\text{LR} = \frac{\vec{u}_\text{DV} \times \vec{u}_\text{AP}}{\|\vec{u}_\text{DV} \times \vec{u}_\text{AP}\|}$$

6. **Handedness check:** Ensure ABa is on the positive LR side.
$$\text{lr}_\text{ABa} = (\vec{r}_a - \vec{c}_\text{AB}) \cdot \vec{u}_\text{LR}$$
If $\text{lr}_\text{ABa} < 0$, flip both: $\vec{u}_\text{LR} \leftarrow -\vec{u}_\text{LR}$, $\vec{u}_\text{DV} \leftarrow -\vec{u}_\text{DV}$.

**Division vector projection in static founder mode:**

Given a raw division vector $\vec{d}$ (z-scaled), project onto the founder basis:
$$d_\text{AP} = \vec{d} \cdot \vec{u}_\text{AP}, \quad d_\text{DV} = \vec{d} \cdot \vec{u}_\text{DV}, \quad d_\text{LR} = \vec{d} \cdot \vec{u}_\text{LR}$$

Map to canonical frame:
$$\vec{d}_\text{canonical} = (-d_\text{AP},\ d_\text{DV},\ d_\text{LR})$$

The negation of AP maps to the canonical AP direction $(-1, 0, 0)$.

**Limitation:** This static approach assumes the embryo orientation does not change after the 4-cell stage. For compressed embryos that rotate during imaging, the per-timepoint lineage centroid approach (Section 4.4) is preferred.

---

## 5. Division Classification

### 5.1 Classification Algorithm

Given parent nucleus $P$ dividing into daughters $D_1, D_2$, and division rule $(s, \vec{a})$ where $s$ is the Sulston letter and $\vec{a}$ is the rule's axis unit vector:

1. **Raw division vector:**
$$\vec{\delta} = (D_2.x - D_1.x,\ D_2.y - D_1.y,\ (D_2.z - D_1.z) \times z_\text{pix\_res})$$

2. **Rotate to canonical frame:** $\vec{\delta}_c = T(\vec{\delta})$ where $T$ is the active transform (v2, v1, lineage centroid, or static founder).

3. **Dot product:**
$$\alpha = \vec{\delta}_c \cdot \vec{a}$$

4. **Angle from rule axis:**
$$\theta = \arccos\left(\frac{|\alpha|}{\|\vec{\delta}_c\| \cdot \|\vec{a}\|}\right) \quad \text{(in degrees)}$$

5. **Name assignment:**
$$\text{if } \alpha \geq 0: \quad D_1 \gets \text{daughter}_1,\ D_2 \gets \text{daughter}_2$$
$$\text{if } \alpha < 0: \quad D_1 \gets \text{daughter}_2,\ D_2 \gets \text{daughter}_1$$

### 5.2 Confidence from Angle

$$C(\theta) = \begin{cases}
1.0 & \text{if } \theta \leq 20° \\
1.0 - 0.5 \cdot \frac{\theta - 20}{20} & \text{if } 20° < \theta \leq 40° \\
0.5 - 0.3 \cdot \frac{\theta - 40}{15} & \text{if } 40° < \theta \leq 55° \\
\max(0.1,\ 0.2 - \frac{\theta - 55}{180}) & \text{if } \theta > 55°
\end{cases}$$

Constants: `HIGH_CONFIDENCE_ANGLE = 20°`, `LOW_CONFIDENCE_ANGLE = 40°`, `RULE_OVERRIDE_ANGLE = 55°`.

### 5.3 Multi-Frame Averaging

For improved robustness, division vectors can be averaged over $N$ frames after division (default $N = 3$):

1. For each frame $t_\text{div} + k$ ($k = 0, \ldots, N-1$):
$$\vec{\delta}_k = T\left(\frac{D_2^{(k)} - D_1^{(k)}}{\|D_2^{(k)} - D_1^{(k)}\|}\right)$$
where $D_i^{(k)}$ is daughter $i$'s position at frame $t_\text{div} + k$.

2. Average unit vectors:
$$\vec{\delta}_\text{avg} = \frac{1}{n} \sum_k \hat{\delta}_k$$

3. Consistency metric: $\|\vec{\delta}_\text{avg}\|$ (1.0 = all frames agree, 0.0 = random).

The averaged vector is then used in the standard classification algorithm (Section 5.1).

---

## 6. Division Rule System

### 6.1 Rule Lookup Priority

For parent name $P$:

1. **Pre-computed rules** (`new_rules.tsv`): ~620 empirically determined rules with axis vectors derived from actual embryo measurements. Format: `Parent\tLetter\tD1\tD2\tX\tY\tZ`.

2. **Names hash** (`names_hash.csv`): ~60 Sulston letter mappings for less-common divisions. Letter is decoded from an encoded integer value. The axis vector is the standard axis for that letter.

3. **Default**: Use letter `"a"` (AP axis), axis vector $(1, 0, 0)$.

### 6.2 Axis Vector Convention

$$\text{LETTER\_TO\_AXIS}: \quad \begin{cases}
a, p \to (1, 0, 0) & \text{AP axis} \\
d, v \to (0, 1, 0) & \text{DV axis} \\
l, r \to (0, 0, 1) & \text{LR axis}
\end{cases}$$

The sign of the dot product (Section 5.1, step 5) determines which daughter gets the "positive" letter (a, d, l) vs the "negative" letter (p, v, r).

---

## 7. Undo/Redo System

### 7.1 Data Structure

Two stacks implement a linear undo history:

$$U = [c_1, c_2, \ldots, c_n] \quad \text{(undo stack)}$$
$$R = [c_k, c_{k-1}, \ldots] \quad \text{(redo stack)}$$

### 7.2 Operations

**`do(c)`:**
$$\text{execute}(c), \quad U \leftarrow U \| [c], \quad R \leftarrow []$$

Redo stack is cleared on every new edit (branching history is discarded).

**`undo()`:**
$$c \leftarrow U.\text{pop}(), \quad \text{reverse}(c), \quad R \leftarrow R \| [c]$$

**`redo()`:**
$$c \leftarrow R.\text{pop}(), \quad \text{execute}(c), \quad U \leftarrow U \| [c]$$

**Stack size limit:** $|U| \leq 1000$. When exceeded: $U \leftarrow U[1:]$ (oldest command discarded).

### 7.3 Callback Architecture

After every `do`/`undo`/`redo`, the `on_edit` callback is invoked. In the GUI, this triggers:
1. `set_all_successors()` — recompute forward links
2. `process()` — rerun naming and tree building
3. `rebuild_tree()` — full lineage tree layout recomputation (structural edits change the tree topology, so incremental refresh is insufficient)
4. `update_display()` — refresh all visual components

---

## 8. Edit Commands — State Capture and Reversal

### 8.1 AddNucleus

**Execute:** Create `Nucleus(x, y, z, size, identity, predecessor, status=1)`. Append to `nuclei_record[time-1]`. Save the appended index.

**Undo:** Pop the last element from `nuclei_record[time-1]`.

### 8.2 RemoveNucleus

**Execute:** Save `(status, identity, assigned_id)`. Set `status ← -1`, `identity ← ""`, `assigned_id ← ""`.

**Undo:** Restore all three saved fields.

### 8.3 MoveNucleus

**Execute:** Save `(x_0, y_0, z_0, \text{size}_0)`. Apply non-None new values.

**Undo:** Restore `(x_0, y_0, z_0, \text{size}_0)`.

### 8.4 RenameCell

**Execute:** Save `(identity_0, \text{assigned\_id}_0)`. Set both to `new_name`.

**Undo:** Restore both.

### 8.5 RelinkNucleus

This is the most complex command, managing bidirectional links.

**Execute:**
1. Save old predecessor: $\text{pred}_\text{old}$.
2. If old parent exists: save `(succ1, succ2)` of old parent. Remove child from old parent's successors.
3. Set `nuc.predecessor ← new_pred`.
4. If new parent exists: save `(succ1, succ2)` of new parent. Add child to new parent's successors.

**Undo:** Restore the predecessor field and both parents' successor fields.

**Successor management:**
- `_remove_successor(parent, child_idx)`: If `succ1 = child_idx`, shift `succ2 → succ1`. If `succ2 = child_idx`, clear it.
- `_add_successor(parent, child_idx)`: Fill `succ1` first, then `succ2`.

### 8.6 KillCell

**Execute:** For each timepoint in `[start_time, end_time]`, find all alive nuclei with matching identity. Save `(time, index, status, identity, assigned_id)` for each. Kill them all.

**Undo:** Restore all saved tuples.

### 8.7 ResurrectCell

Inverse of RemoveNucleus. Sets `status ← 1` and optionally applies a new identity.

### 8.8 RelinkWithInterpolation

**Execute:**
1. Let $n = \text{end\_time} - \text{start\_time}$.
2. Get start nucleus $S$ at `(start_time, start_index)` and end nucleus $E$ at `(end_time, end_index)`.
3. For each intermediate timepoint $t = \text{start\_time} + k$ ($k = 1, \ldots, n-1$):

$$x_k = S.x + (E.x - S.x) \cdot \frac{k}{n}$$
$$y_k = S.y + (E.y - S.y) \cdot \frac{k}{n}$$
$$z_k = S.z + (E.z - S.z) \cdot \frac{k}{n}$$
$$\text{size}_k = S.\text{size} + (E.\text{size} - S.\text{size}) \cdot \frac{k}{n}$$

4. Create new nucleus at each intermediate timepoint with interpolated values.
5. Chain all nuclei via predecessor/successor links: $S \to I_1 \to I_2 \to \ldots \to E$.

**Undo:** Remove all interpolated nuclei in reverse order. Restore all predecessor/successor links.

---

## 9. Validation System

### 9.1 Pre-Edit Validators

Each validator returns a list of error message strings. An empty list means the operation is valid.

| Validator                        | Checks                                                     |
|---------------------------------|------------------------------------------------------------|
| `validate_add_nucleus`           | Time ≥ 1; predecessor exists; predecessor has < 2 successors |
| `validate_remove_nucleus`        | Valid time/index; nucleus is alive                          |
| `validate_relink`                | Valid time/index; new pred exists; new pred has < 2 successors |
| `validate_kill_cell`             | Name non-empty; start_time valid; cell exists and is alive  |
| `validate_relink_interpolation`  | end_time > start_time; both nuclei exist; start has < 2 successors |

### 9.2 Post-Naming Validation (`naming/validation.py`)

`validate_naming()` checks for:
- Naming gaps (unnamed alive cells in the middle of lineages)
- Duplicate names at a single timepoint
- Name inconsistencies (parent-child name mismatches)

Returns a list of `NamingWarning` objects.

---

## 10. Projected Nucleus Diameter

The image viewer shows nuclei as circles whose size reflects their distance from the current z-plane.

Given nucleus at z-position $z_n$, current image plane $z_p$, nucleus size $s$, and z pixel resolution $z_r = z_\text{res} / xy_\text{res}$:

$$\Delta z = |z_n - z_p| \times z_r$$

$$r = \frac{s}{2}$$

$$d_\text{projected} = \begin{cases}
2\sqrt{r^2 - \Delta z^2} & \text{if } \Delta z < r \\
0 & \text{otherwise}
\end{cases}$$

This is the chord length of a sphere at a given distance from the focal plane — standard projection geometry.
