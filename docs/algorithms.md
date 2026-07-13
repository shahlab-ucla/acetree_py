# AceTree-Py Algorithm Reference

Mathematically precise descriptions of the Sulston naming system, coordinate transforms, division classification, undo mechanism, and editing operations.

For the cross-cutting user/task contract and edge-case matrix, see [Naming and Manual-Curation Workflows](naming_workflows.md).

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

### 2.0 Name State Model

Every live nucleus carries two distinct name fields:

| Field | Meaning | May automatic naming change it? |
|---|---|---|
| `identity` | Current automatic/computed identity | Yes |
| `assigned_id` | Explicit user override | No |
| `effective_name` | `assigned_id` when non-empty, otherwise `identity` | Derived |

GUI labels, tree lookup, validation, kill/rename targeting, and division-parent lookup use `effective_name`. A suggested automatic name must never be copied into `assigned_id`; doing so would silently convert a prediction into a permanent user decision. Rename creates an override only when the requested name differs from the effective name. **Use Automatic** clears `assigned_id` across the same cell continuation and lets the next naming pass recompute `identity`.

### 2.1 Overall Flow

```
Input: nuclei_record[t][i] for all timepoints t, nucleus index i
       AuxInfo (v1 or v2 orientation), naming_method

Step 1:  Clear non-forced names
         ∀ nuc: if assigned_id = "": identity ← ""

Step 1b: Propagate forced names (assigned_id) through valid continuation chains
         For each nucleus with assigned_id set:
           Forward: follow the single alive, reciprocal successor (non-dividing only),
                    set assigned_id + identity on each continuation cell
           Backward: follow the alive, reciprocal predecessor (stop at division boundaries),
                     set assigned_id + identity back to cell's birth

Step 2:  Select the highest-precedence valid orientation source
         (v2/manual → supported v1 → lineage → founder fallback)

Step 3:  Topology-based founder identification
         → FounderAssignment with ABa, ABp, EMS, P2 indices + confidence

Step 4:  If Step 3 fails (confidence < 0.3): warn and assign generic names
         (legacy InitialID fallback available via legacy_mode=True)

Step 5:  Set up DivisionCaller with coordinate axes
         Compute seed axes (AP, DV, LR) at 4-cell midpoint for sign anchoring

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

If a live nucleus has `assigned_id` set (manual override via Rename), the forced name is **propagated through the cell's valid continuation** before canonical rules run:

1. **Forward propagation**: The forced name follows the sole live successor through consecutive timepoints, but only when the child's predecessor points back to the same parent.
2. **Backward propagation**: The forced name follows the live predecessor back to the cell's birth, but only when that parent's sole successor points back to the child.
3. **Division boundary**: Propagation does not cross division boundaries. When the forced-name cell eventually divides, its name is used as the parent name for the division caller, which applies the standard Sulston rules to name the daughters.
4. **Invalid/dead boundary**: Dead nuclei, missing timepoints, non-reciprocal links, and ambiguous successor sets stop propagation rather than being repaired implicitly.

If a different `assigned_id` already exists on a nucleus in the chain, propagation stops and reports a conflict. Traversal order never decides which user assertion wins.

When both daughters of a division have pre-assigned names, the automatic classification is skipped. If only one daughter has a pre-assigned name, the other receives the complement name.

Automatic naming never invents an `"X"` suffix to hide a collision. A forced/automatic mismatch may swap the automatically assigned sister pair when that resolves the intended complement. Two incompatible forced daughter identities remain explicit validation conflicts for the user to correct.

---

## 3. Topology-Based Founder Identification

### 3.1 Four-Cell Window Detection

A **four-cell window** is a contiguous range of timepoints $[t_\text{first}, t_\text{last}]$ where exactly 4 alive, non-polar-body nuclei exist:

$$\forall t \in [t_\text{first}, t_\text{last}]: \quad |\{n \in \text{nuclei}(t) : n.\text{status} \geq 1 \wedge n.\text{size} < \text{polar}\_\text{size}\}| = 4$$

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

**P1 pair (EMS vs P2):** Two signals are used, with forward division timing as the primary discriminator:

1. **Primary — forward division timing:** EMS divides before P2 at the 8→16 cell transition. Both cells are traced forward through their successor chains until they divide. If one divides at least 1 frame before the other, that cell is EMS.

2. **Secondary — nucleus size:** EMS is typically larger than P2. Used when forward timing is unavailable or simultaneous.

$$\text{EMS} = \begin{cases} \text{earlier divider} & \text{if forward timing gap} \geq 1 \\ \arg\max_{n \in \text{P1 pair}} n.\text{size} & \text{otherwise} \end{cases}$$

**AB pair (ABa vs ABp):** Determined by projection onto the AP axis vector, averaged over the 4-cell stage window for robustness. A valid explicit posterior→anterior orientation is used when available; otherwise AP is estimated from P2 toward the AB pair. ABa is the daughter with the larger projection (more anterior):

$$\vec{u}_\text{AP} = \frac{\vec{c}_\text{AB} - \vec{r}_{P2}}{\|\vec{c}_\text{AB} - \vec{r}_{P2}\|}$$
$$\text{ABa} = \arg\max_{n \in \text{AB pair}} \left(\frac{1}{T}\sum_{t} \vec{r}_n(t) \cdot \vec{u}_\text{AP}(t)\right)$$

where the average is taken over all timepoints in the 4-cell window. If AP geometry is degenerate, PC1 of the four-cell point cloud is used as the embryo long axis and oriented toward the AB end. A final raw image-x tiebreaker remains deterministic for diagnostics, but its confidence is capped below the automatic-commit threshold because microscope x is not anatomy.

Projection avoids assuming a particular image direction when a defensible AP cue exists. Degenerate PC1/raw-coordinate fallbacks are retained for diagnosis but are not treated as equivalent biological evidence.

### 3.5 Confidence Calculation

The overall confidence combines three factors:

$$C = C_\text{timing} \times C_\text{size} \times (0.5+0.5C_\text{axis}) - \text{penalty}$$

**Timing confidence** from backward trace:

$$C_\text{timing} = \begin{cases}
\text{(from forward pairing)} & \text{if timing gap} = 0 \\
0.6 & \text{if timing gap} = 1 \\
\min(1.0,\ 0.6 + \text{gap} \times 0.1) & \text{if timing gap} \geq 2
\end{cases}$$

**Forward pairing confidence** (when backward trace yields gap = 0):

$$C_\text{timing}^\text{fwd} = \begin{cases}
\min(1.0,\ 0.5 + \text{fwd}\_\text{gap} \times 0.1) & \text{if fwd}\_\text{gap} \geq 1 \\
0.5 & \text{if fwd}\_\text{gap} = 0
\end{cases}$$

**Size confidence:**

Let $s_\text{diff}$ = absolute size difference between the larger and smaller cells in the P1 pair, and $s_\text{sum}$ = sum of their sizes:

$$C_\text{size} = \min\left(1.0,\ 0.5 + \frac{s_\text{diff}}{s_\text{sum}}\right)$$

**Axis confidence:** combines four-cell separation with the conditioning of the AP/DV geometry. Let $C_\text{sep}$ be the normalized minimum-to-median pairwise separation and let $q_\perp$ be the EMS→ABp vector's usable component perpendicular to P2→ABa:

$$C_\text{axis}=\min(C_\text{sep},q_\perp).$$

The component is exposed separately so a geometrically flat/compressed acquisition lowers downstream naming trust without discarding otherwise strong topology. Raw image-x fallback caps both axis and overall confidence at 0.1.

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

Body-axis vectors have biological direction, not merely an unsigned image axis:

- AP points **posterior → anterior**.
- DV points **ventral → dorsal**.
- LR points **right → left**.
- The frame is right-handed: $\vec e_\text{DV}=\vec e_\text{AP}\times\vec e_\text{LR}$.

The division-rule canonical coordinate system is represented as:

$$\vec{e}_\text{AP} = (-1, 0, 0), \quad \vec{e}_\text{LR} = (0, 0, 1), \quad \vec{e}_\text{DV} = \vec{e}_\text{AP} \times \vec{e}_\text{LR} = (0, 1, 0)$$

All geometry is evaluated in physical coordinates. For stored image coordinates $(x,y,z)$, the vector used by naming is $(x,y,z\,z_\text{pix_res})$. This applies equally to landmark vectors, founder centroids, and daughter-division vectors.

### 4.2 v2 Transform (Wahba's Problem)

Given measured AP vector $\vec{a}$ and LR vector $\vec{l}$ from AuxInfo v2:

1. Reject zero-length or nearly parallel vectors.
2. Normalize AP and project LR perpendicular to AP (Gram–Schmidt); normalize the result.
3. Compute $\hat d=\hat a\times\hat l$ and verify handedness.
4. Form source basis $S=[\hat a;\hat d;\hat l]$ and target basis $T=[\vec e_\text{AP};\vec e_\text{DV};\vec e_\text{LR}]$.
5. Solve for the best rotation $R$ with `scipy.spatial.transform.Rotation.align_vectors()`.
6. Validate the mapped axes and retain the input orthogonality as a diagnostic.

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

Only the supported v1 anatomical orientations (`ADL`, `AVR`, `PDR`, and `PVL`) count as orientation metadata. A placeholder such as `XXX` is not a body-axis assertion.

### 4.4 Manual Landmark Frame

The simplified correction workflow records anatomical endpoints at one reference timepoint. It requires an AP pair plus one signed secondary pair:

$$\vec a=\vec r_\text{anterior}-\vec r_\text{posterior}$$

and either

$$\vec d_0=\vec r_\text{dorsal}-\vec r_\text{ventral}$$

or

$$\vec l_0=\vec r_\text{left}-\vec r_\text{right}.$$

The secondary vector is projected perpendicular to AP. With AP+DV, $\vec l=\vec d\times\vec a$; with AP+LR, $\vec d=\vec a\times\vec l$. Normalization and handedness checks produce an orthonormal frame. Endpoint labels must all come from the same timepoint, and all z coordinates are physically scaled before subtraction.

Each frame records `provenance`, `reference_time`, and `quality`. A manually applied frame is serialized as AuxInfo v2 beside the nuclei archive and takes precedence over automatic lineage geometry on reload. Save and undo treat the frame change as ordinary editable state.

### 4.5 Per-Timepoint Lineage Centroid Axes (Automatic Fallback)

When no valid explicit orientation is available, axes are derived at each timepoint from the spatial distribution of founder-lineage descendants. This can follow gradual embryo motion, but its confidence and provenance remain visible because compression or sparse tracking can make the geometry ambiguous.

**Lineage map construction** (`naming/lineage_axes.py`):

A lineage label (ABa, ABp, EMS, or P2) is assigned to every nucleus by propagating founder identity through predecessor/successor chains:

1. Seed the 4 founders at the 4-cell midpoint with their labels.
2. Back-propagate from 4-cell time to $t=0$: each nucleus inherits the label of its successor.
3. Forward-propagate from 4-cell time to the end: each successor inherits its predecessor's label (both daughters of a dividing cell get the same lineage label).

**Axis computation at timepoint $t$:**

Let $\mathcal{A}_a(t), \mathcal{A}_p(t), \mathcal{E}(t), \mathcal{P}_2(t)$ be the sets of alive labelled cells at time $t$.

1. **Group centroids:**
$$\vec{c}_\text{ABa}(t) = \text{mean}(\mathcal{A}_a(t)), \quad \vec{c}_\text{ABp}(t) = \text{mean}(\mathcal{A}_p(t))$$
$$\vec{c}_\text{EMS}(t) = \text{mean}(\mathcal{E}(t)), \quad \vec{c}_\text{P2}(t) = \text{mean}(\mathcal{P}_2(t))$$

2. **AP seed** (P2 → ABa):
$$\vec a(t)=\vec c_\text{ABa}(t)-\vec c_\text{P2}(t),\qquad \vec u_\text{AP}(t)=\frac{\vec a(t)}{\|\vec a(t)\|}$$

3. **DV seed** (EMS → ABp), projected perpendicular to AP:
$$\vec d_0(t)=\vec c_\text{ABp}(t)-\vec c_\text{EMS}(t)$$
$$\vec d_\perp(t)=\vec d_0(t)-(\vec d_0(t)\cdot\vec u_\text{AP}(t))\vec u_\text{AP}(t)$$
$$\vec u_\text{DV}(t)=\frac{\vec d_\perp(t)}{\|\vec d_\perp(t)\|}$$

4. **LR axis** (right → left), completing the frame:
$$\vec u_\text{LR}(t)=\vec u_\text{DV}(t)\times\vec u_\text{AP}(t)$$

This construction satisfies $\vec u_\text{DV}=\vec u_\text{AP}\times\vec u_\text{LR}$.

**Secondary-axis quality:**

The DV seed becomes unreliable when it is nearly parallel to AP. The usable perpendicular fraction is:

$$q_\perp(t)=\frac{\|\vec d_\perp(t)\|}{\|\vec d_0(t)\|}.$$

Small separation, a small perpendicular fraction, incomplete lineage groups, or discontinuous estimates lower confidence. Cached neighboring frames may preserve temporal sign continuity; continuity is not itself evidence that the biological left/right sign is correct.

`compute_local_axes()` returns `(ap_vec, lr_vec, dv_vec, secondary_quality)`.

**Division vector projection:**

Given a raw division vector $\vec{d}$ at timepoint $t$, project onto the local axes:
$$\vec{d}_\text{canonical}(t) = (-\vec{d} \cdot \vec{u}_\text{AP}(t),\ \vec{d} \cdot \vec{u}_\text{DV}(t),\ \vec{d} \cdot \vec{u}_\text{LR}(t))$$

**Why per-timepoint?** A static early frame can become stale as an embryo moves or is mechanically compressed. Re-deriving from current lineage centroids can follow that motion, while quality thresholds prevent a weak frame from being presented as certain.

### 4.6 Static Founder-Derived Transform (Last Fallback)

Used only as a fallback when the lineage centroid approach fails (e.g., too few labelled cells at a given timepoint). Axes are derived once from the 4-cell positions:

Let $\vec{r}_a, \vec{r}_b, \vec{r}_e, \vec{r}_p$ be the 3D positions (with z scaled by `z_pix_res`) of ABa, ABp, EMS, P2 respectively.

Use the same four-cell construction as Section 4.5 at the four-cell midpoint:

$$\vec a=\vec r_\text{ABa}-\vec r_\text{P2},\qquad \vec d_0=\vec r_\text{ABp}-\vec r_\text{EMS}.$$

Normalize AP, project and normalize DV perpendicular to AP, then compute $\vec u_\text{LR}=\vec u_\text{DV}\times\vec u_\text{AP}$. Do **not** treat the ABa–ABp separation as the LR axis.

**Division vector projection in static founder mode:**

Given a raw division vector $\vec{d}$ (z-scaled), project onto the founder basis:
$$d_\text{AP} = \vec{d} \cdot \vec{u}_\text{AP}, \quad d_\text{DV} = \vec{d} \cdot \vec{u}_\text{DV}, \quad d_\text{LR} = \vec{d} \cdot \vec{u}_\text{LR}$$

Map to canonical frame:
$$\vec{d}_\text{canonical} = (-d_\text{AP},\ d_\text{DV},\ d_\text{LR})$$

The negation of AP maps to the canonical AP direction $(-1, 0, 0)$.

**Biological limitation:** the signed LR axis is not identifiable solely from the ABa/ABp pair at the four-cell stage. Establishing left versus right requires a trusted oriented secondary cue (manual landmarks or metadata), or later handedness/chirality information. Automatic four-cell geometry is therefore a fallible estimate, especially under compression, and must expose confidence rather than silently locking names. The invariant lineage described by [Sulston et al. (1983)](https://www.wormatlas.org/papers/Sulston_embryonic_lineage_1983.pdf), automated geometry in [Bao et al. (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1413828/), embryonic chirality in [Pohl and Bao (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2952354/), and compression effects in [Hench et al. (2009)](https://pubmed.ncbi.nlm.nih.gov/19527702/) provide the biological and experimental context.

---

## 5. Division Classification

### 5.1 Classification Algorithm

Given parent nucleus $P$ dividing into daughters $D_1, D_2$, and division rule $(s, \vec{a})$ where $s$ is the Sulston letter and $\vec{a}$ is the rule's axis unit vector:

1. **Raw division vector:**
$$\vec{\delta} = (D_2.x - D_1.x,\ D_2.y - D_1.y,\ (D_2.z - D_1.z) \times z_{\text{pix}\_\text{res}})$$

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

For improved robustness in v1/v2 modes, division vectors can be averaged over $N$ frames after division (default $N = 3$):

1. For each frame $t_\text{div} + k$ ($k = 0, \ldots, N-1$):
$$\vec{\delta}_k = T\left(\frac{D_2^{(k)} - D_1^{(k)}}{\|D_2^{(k)} - D_1^{(k)}\|}\right)$$
where $D_i^{(k)}$ is daughter $i$'s position at frame $t_\text{div} + k$.

2. Average unit vectors:
$$\vec{\delta}_\text{avg} = \frac{1}{n} \sum_k \hat{\delta}_k$$

3. Consistency metric: $\|\vec{\delta}_\text{avg}\|$ (1.0 = all frames agree, 0.0 = random).

The averaged vector is then used in the standard classification algorithm (Section 5.1).

**Note:** Multi-frame averaging is **disabled in lineage centroid mode**. With per-timepoint axes that may differ between frames, averaging the division vector across frames mixes coordinate systems (each frame's vector is projected through different axes). Single-frame classification with quality-aware axis smoothing (Section 4.5) gives better results empirically.

### 5.4 Deferred Majority-Vote Evaluation

When the initial single-frame classification has low confidence ($C < 0.3$, corresponding to $\theta > 55°$), the result may be unreliable — particularly during LR axis degeneracy. Rather than committing to a potentially wrong assignment, the system defers and re-evaluates using a look-ahead window.

**Algorithm:**

1. Follow both daughters forward through their successor chains for up to 8 frames.
2. At each look-ahead frame $t_\text{div} + k$, re-classify the division using the daughter positions at that frame and the axes at that frame.
3. Each frame casts a vote for the positive or negative assignment. Track the best individual confidence seen.
4. After all frames, the majority vote determines the assignment:

$$\text{margin} = \frac{|V_+ - V_-|}{V_+ + V_-}$$

$$C_\text{vote} = \max(C_\text{best},\ \text{margin})$$

5. The deferred result replaces the initial classification if $C_\text{vote} \geq C_\text{initial}$.

This mechanism is especially useful when the secondary axis at the moment of division is geometrically weak but recovers within a few frames as cells separate. The result records the axis label, confidence, and orientation source so the GUI can present it as a preview rather than a fact.

---

## 6. Division Rule System

### 6.1 Rule Lookup Priority

For parent name $P$:

1. **Pre-computed rules** (`new_rules.tsv`): ~620 empirically determined rules with axis vectors derived from actual embryo measurements. Format: `Parent\tLetter\tD1\tD2\tX\tY\tZ`.

2. **Names hash** (`names_hash.csv`): ~60 Sulston letter mappings for less-common divisions. Letter is decoded from an encoded integer value. The axis vector is the standard axis for that letter.

3. **Default**: Use letter `"a"` (AP axis), axis vector $(1, 0, 0)$.

### 6.2 Axis Vector Convention

$$\text{LETTER}\_\text{TO}\_\text{AXIS}: \quad \begin{cases}
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

**Gesture atomicity:** One completed GUI gesture creates one history entry. Add-with-interpolation, track placement, relink-with-interpolation, rename state changes, and body-axis application use a `CompositeCommand` when multiple low-level mutations are required. Execute is all-or-nothing; undo reverses children in reverse order.

**Saved state:** History maintains a savepoint independently from stack depth. A successful Save or Save As marks the current state saved. Undoing away from it is dirty; redoing exactly back to it is clean. Editing after undo discards the redo branch and cannot become clean merely because the undo-stack length happens to match the old length.

### 7.3 Callback Architecture

After every `do`/`undo`/`redo`, the `on_edit` callback is invoked. In the GUI, this triggers:
1. `set_all_successors()` — recompute forward links
2. `process()` — rerun naming and tree building
3. `rebuild_tree()` — full lineage tree layout recomputation (structural edits change the tree topology, so incremental refresh is insufficient)
4. `update_display()` — refresh all visual components

---

## 8. Edit Commands — State Capture and Reversal

### 8.1 AddNucleus

**Execute:** Create `Nucleus(x, y, z, size, identity, assigned_id, predecessor, status=1)`. Append to `nuclei_record[time-1]`, save the appended index, and immediately establish the reciprocal successor link when a predecessor is supplied. Automatic identity may be inherited for continuity, but only a pre-existing `assigned_id` may be inherited as a forced override.

**Undo:** Remove the added nucleus and restore the predecessor's prior successor slots.

### 8.2 RemoveNucleus

**Execute:** Save `(status, identity, assigned_id)`. Set `status ← -1`, `identity ← ""`, `assigned_id ← ""`.

**Undo:** Restore all three saved fields.

### 8.3 MoveNucleus

**Execute:** Save `(x_0, y_0, z_0, \text{size}_0)`. Apply non-None new values.

**Undo:** Restore `(x_0, y_0, z_0, \text{size}_0)`.

Position changes are structurally significant for naming: a move can change division geometry and therefore daughter assignment. The post-edit callback reruns naming and rebuilds affected lineage state.

### 8.4 RenameCell

**Execute:**
1. Walk the cell's *continuation chain* starting at the clicked `(time, index)`: follow `predecessor` backward until it hits a division (predecessor has two successors) or disappears, and follow `successor1` forward under the same rule. This enumerates every nucleus belonging to the same cell.
2. For each nucleus `(t_k, i_k)` in the chain, save `(t_k, i_k, identity_0, assigned_id_0)`. Set `identity ← new_name` and `assigned_id ← new_name` on that nucleus.

**Undo:** For each saved tuple, restore `identity` and `assigned_id`.

**Why cell-scoped?** A cell in AceTree is the continuation chain of a tracked nucleus — birth to next division or disappearance — not a single nucleus at one timepoint. Writing `assigned_id` on all chain members atomically keeps the manual override consistent across the cell's lifetime no matter which timepoint the user clicked, and makes a single undo/redo restore the full rename.

**Interaction with the naming pipeline:** `_propagate_assigned_ids()` (Section 2.2) is now a safety net rather than the primary propagation mechanism — the command itself has already written the forced name end-to-end. Daughters beyond the next division are still named automatically by the division caller using the forced name as parent.

Whitespace is trimmed before validation. A request equal to the existing effective name is a true no-op and creates no history entry. Commas, line breaks, and control characters are rejected because nucleus records are CSV-compatible text.

### 8.5 ClearNameOverride / Use Automatic

Snapshot `identity` and `assigned_id` across the valid continuation, clear `assigned_id`, and rerun naming. Undo restores both fields exactly. This is distinct from renaming to an empty string: it means “return responsibility to the automatic pipeline.”

### 8.6 SetCellNameState

Snapshot and set the `identity`/`assigned_id` pair over one anchored continuation component. This low-level command is used inside composite placement gestures so a parent-specific automatic daughter suggestion can be stored in `identity` while inherited forced state is retained or cleared deliberately. Undo restores both fields exactly.

### 8.7 SwapCellNames

Used to resolve name collisions when the user tries to rename a cell to a name already in use by another cell.

**Execute:**
1. Resolve `nuc_a` at `(time_a, index_a)` and `nuc_b` at `(time_b, index_b)`. Read `name_a = nuc_a.effective_name` and `name_b = nuc_b.effective_name` *before mutating anything*.
2. Walk the continuation chains `chain_a` and `chain_b` (same algorithm as RenameCell).
3. For each nucleus in `chain_a`: save its `(t, i, identity_0, assigned_id_0)`, then set `identity = assigned_id = name_b`.
4. For each nucleus in `chain_b`: save its state, then set `identity = assigned_id = name_a`.

**Undo:** Restore all saved tuples from both chains.

**Empty-name handling:** If either cell has no effective name (e.g. never got a Sulston name), that side of the swap writes the empty string, effectively clearing the other chain's forced name. This is intentional — the operation is "B now has A's name and vice versa," which means blank round-trips if A was blank.

### 8.8 RelinkNucleus

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

### 8.9 KillCell

**Execute:** Resolve the selected cell from an anchored `(time,index)` nucleus, then traverse its valid continuation within `[start_time,end_time]`. Matching uses `effective_name`, not `identity`, so forced names work correctly. Save `(time,index,status,identity,assigned_id)` and kill only members of that anchored component; a same-named disconnected cell is not collateral damage.

**Undo:** Restore all saved tuples.

### 8.10 ResurrectCell

Inverse of RemoveNucleus. It rejects an already-live target, sets `status ← 1`, and restores or explicitly applies both automatic identity and `assigned_id` as appropriate.

### 8.11 RelinkWithInterpolation

**Execute:**
1. Let $n = \text{end}\_\text{time} - \text{start}\_\text{time}$.
2. Get start nucleus $S$ at `(start_time, start_index)` and end nucleus $E$ at `(end_time, end_index)`.
3. For each intermediate timepoint $t = \text{start}\_\text{time} + k$ ($k = 1, \ldots, n-1$):

$$x_k = S.x + (E.x - S.x) \cdot \frac{k}{n}$$
$$y_k = S.y + (E.y - S.y) \cdot \frac{k}{n}$$
$$z_k = S.z + (E.z - S.z) \cdot \frac{k}{n}$$
$$\text{size}_k = S.\text{size} + (E.\text{size} - S.\text{size}) \cdot \frac{k}{n}$$

4. Create new nucleus at each intermediate timepoint with interpolated values.
5. Chain all nuclei via predecessor/successor links: $S \to I_1 \to I_2 \to \ldots \to E$.

**Undo:** Remove all interpolated nuclei in reverse order. Restore all predecessor/successor links.

The full interpolation plus endpoint relink is executed as one composite history entry. Validators reject a new link that would merge incompatible forced identities; automatic naming must not resolve that conflict by traversal order.

### 8.12 SetBodyAxes

Snapshot the prior AuxInfo/orientation state, install a validated `BodyAxisFrame`, invalidate the identity assigner, and rerun naming. Undo restores the previous frame and resulting naming state. Applying orientation is one user action and one undo step.

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
| Name validation                  | Trimmed non-empty value; no comma, CR/LF, or control character |

Relink validators also require alive reciprocal endpoints and reject merges between incompatible forced identities. Resurrect validation requires an explicitly selected dead nucleus.

### 9.2 Post-Naming Validation (`naming/validation.py`)

`validate_naming()` checks for:
- Naming gaps (unnamed alive cells in the middle of lineages)
- Duplicate names at a single timepoint
- Disconnected cells with the same effective name (reported as collisions, never renamed to synthetic `_2` aliases)
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
