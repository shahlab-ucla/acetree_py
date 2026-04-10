"""Analyze whether the LR axis computation is geometrically unstable.

The LR axis = (ABa_centroid - ABp_centroid) projected perpendicular to AP.
If ABa and ABp centroids are nearly collinear with AP, the perpendicular
component vanishes -> LR becomes noise -> sign flips easily.

For each dataset, compute:
1. The raw ABa-ABp vector magnitude
2. The AP-parallel component (removed by projection)
3. The perpendicular component (becomes LR)
4. The ratio: LR_perp / total -> stability metric
5. Frame-to-frame angular change in LR

Also investigate alternative LR definitions:
A. Current: ABa_centroid - ABp_centroid (perp to AP)
B. PCA-based: PC2 of all cell positions
C. EMS-P2 cross: cross(AP, EMS_centroid - P2_centroid)
"""
import sys
from pathlib import Path
import numpy as np
import math

sys.path.insert(0, str(Path(__file__).parent))

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.naming.founder_id import _get_alive
from acetree_py.naming.lineage_axes import build_lineage_map, LINEAGE_ABa, LINEAGE_ABp, LINEAGE_EMS, LINEAGE_P2

Z = 11.1

def get_lineage_groups(nuclei_record, lineage_map, t, z):
    """Get position arrays for each lineage group at timepoint t."""
    nucs = nuclei_record[t]
    labels = lineage_map[t]
    groups = {"ABa": [], "ABp": [], "EMS": [], "P2": [], "AB": [], "P1": []}

    for j, nuc in enumerate(nucs):
        if nuc.status < 1 or j >= len(labels) or not labels[j]:
            continue
        pos = np.array([float(nuc.x), float(nuc.y), float(nuc.z) * z])
        label = labels[j]
        if label == "ABa":
            groups["ABa"].append(pos)
            groups["AB"].append(pos)
        elif label == "ABp":
            groups["ABp"].append(pos)
            groups["AB"].append(pos)
        elif label == "EMS":
            groups["EMS"].append(pos)
            groups["P1"].append(pos)
        elif label == "P2":
            groups["P2"].append(pos)
            groups["P1"].append(pos)

    return {k: np.array(v) for k, v in groups.items() if v}


ZIP_DIR = Path(r"C:\Users\pavak\Documents\AT_test\du_lab")
ZIP_FILES = sorted(ZIP_DIR.glob("*.zip"))

for zip_path in ZIP_FILES:
    short = zip_path.stem[:40]
    print(f"\n{'='*80}")
    print(f"DATASET: {short}")

    nr = read_nuclei_zip(zip_path)
    mid = -1
    from acetree_py.naming.founder_id import _count_alive
    first = -1
    for t in range(min(len(nr), 200)):
        ct = _count_alive(nr[t])
        if ct == 4:
            if first < 0: first = t
        else:
            if first >= 0:
                mid = (first + t - 1) // 2
                break
            if ct > 4:
                break

    if mid < 0:
        print("  Could not find 4-cell stage")
        continue

    alive = _get_alive(nr[mid])
    ref_f = {}
    for idx, nuc in alive:
        name = nuc.identity.strip()
        if name in ("ABa", "ABp", "EMS", "P2"):
            ref_f[name] = (idx, nuc)

    if len(ref_f) != 4:
        print("  Could not find all 4 founders")
        continue

    lm = build_lineage_map(nr, mid, ref_f["ABa"][0], ref_f["ABp"][0],
                           ref_f["EMS"][0], ref_f["P2"][0])

    print(f"  4-cell midpoint: t={mid}")
    print(f"\n  {'t':>4s} {'nABa':>5s} {'nABp':>5s} {'|ABa-ABp|':>9s} {'|LR_perp|':>9s} {'ratio':>6s} "
          f"{'LR_ang':>7s} {'AP_ang':>7s} {'EMS-P2_perp':>11s} {'cross_LR_ang':>12s}")

    prev_lr = None
    prev_ap = None

    for t in range(mid, min(len(nr), 200)):
        groups = get_lineage_groups(nr, lm, t, Z)
        if "ABa" not in groups or "ABp" not in groups or "AB" not in groups or "P1" not in groups:
            continue

        ab_cent = np.mean(groups["AB"], axis=0)
        p1_cent = np.mean(groups["P1"], axis=0)
        aba_cent = np.mean(groups["ABa"], axis=0)
        abp_cent = np.mean(groups["ABp"], axis=0)

        # AP axis
        ap_raw = ab_cent - p1_cent
        ap_norm = np.linalg.norm(ap_raw)
        if ap_norm < 1e-6:
            continue
        ap_vec = ap_raw / ap_norm

        # LR: ABa-ABp perpendicular to AP
        lr_raw = aba_cent - abp_cent
        lr_total = np.linalg.norm(lr_raw)
        lr_parallel = abs(np.dot(lr_raw, ap_vec))
        lr_perp_vec = lr_raw - np.dot(lr_raw, ap_vec) * ap_vec
        lr_perp = np.linalg.norm(lr_perp_vec)

        ratio = lr_perp / lr_total if lr_total > 1e-6 else 0

        # Current LR direction
        lr_vec = lr_perp_vec / lr_perp if lr_perp > 1e-6 else np.zeros(3)

        # Angular change from previous frame
        lr_ang_str = "  ---"
        if prev_lr is not None and lr_perp > 1e-6:
            dot_lr = np.clip(np.dot(lr_vec, prev_lr), -1, 1)
            lr_ang = math.degrees(math.acos(abs(dot_lr)))
            lr_ang_str = f"{lr_ang:6.1f}d"
            # Check if sign flipped
            if dot_lr < 0:
                lr_ang_str += "*"  # sign flip

        ap_ang_str = "  ---"
        if prev_ap is not None:
            dot_ap = np.clip(np.dot(ap_vec, prev_ap), -1, 1)
            ap_ang = math.degrees(math.acos(abs(dot_ap)))
            ap_ang_str = f"{ap_ang:6.1f}d"

        # Alternative: EMS-P2 cross product for LR
        ems_p2_lr_ang_str = "      ---"
        if "EMS" in groups and "P2" in groups:
            ems_cent = np.mean(groups["EMS"], axis=0)
            p2_cent = np.mean(groups["P2"], axis=0)
            ep_raw = ems_cent - p2_cent
            ep_perp = ep_raw - np.dot(ep_raw, ap_vec) * ap_vec
            ep_norm = np.linalg.norm(ep_perp)

            if ep_norm > 1e-6:
                # Cross AP x (EMS-P2_perp) gives an alternative LR-like axis
                cross_lr = np.cross(ap_vec, ep_perp / ep_norm)
                cross_lr_norm = np.linalg.norm(cross_lr)
                if cross_lr_norm > 1e-6 and lr_perp > 1e-6:
                    cross_lr = cross_lr / cross_lr_norm
                    dot_c = np.clip(np.dot(cross_lr, lr_vec), -1, 1)
                    c_ang = math.degrees(math.acos(abs(dot_c)))
                    ems_p2_lr_ang_str = f"{ep_norm:6.1f}/{c_ang:4.1f}d"

        n_aba = len(groups["ABa"])
        n_abp = len(groups["ABp"])

        print(f"  {t:4d} {n_aba:5d} {n_abp:5d} {lr_total:9.1f} {lr_perp:9.1f} {ratio:6.3f} "
              f"{lr_ang_str:>7s} {ap_ang_str:>7s} {ems_p2_lr_ang_str:>12s}")

        if lr_perp > 1e-6:
            prev_lr = lr_vec
        prev_ap = ap_vec
