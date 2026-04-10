"""Find the exact first mismatch in each dataset and trace the root cause.

For each dataset:
1. Load reference names
2. Run de novo naming
3. Find the first mismatch
4. Analyze the division that caused it (axes, rule, angle)
5. Check how lineage centroid axes compare to reference axes at that timepoint
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.naming.founder_id import identify_founders, _count_alive, _get_alive
from acetree_py.naming.lineage_axes import build_lineage_map, compute_local_axes
from acetree_py.naming.identity import IdentityAssigner, NEWCANONICAL
from acetree_py.naming.rules import RuleManager

Z_PIX_RES = 11.1
ZIP_DIR = Path(r"C:\Users\pavak\Documents\AT_test\du_lab")
ZIP_FILES = sorted(ZIP_DIR.glob("*.zip"))


def pos3d(nuc):
    return np.array([float(nuc.x), float(nuc.y), float(nuc.z) * Z_PIX_RES])


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 90.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return math.degrees(math.acos(abs(cos_a)))


def find_four_cell_mid(nuclei_record):
    first = -1
    for t in range(len(nuclei_record)):
        ct = _count_alive(nuclei_record[t])
        if ct == 4:
            if first < 0:
                first = t
        else:
            if first >= 0:
                return (first + t - 1) // 2
            if ct > 4:
                break
    return -1


rule_mgr = RuleManager()

for zip_path in ZIP_FILES:
    short_name = zip_path.stem[:40]
    print(f"\n{'='*80}")
    print(f"DATASET: {short_name}")
    print(f"{'='*80}")

    # Load reference
    nuclei_record = read_nuclei_zip(zip_path)

    # Save reference names
    ref_names = {}
    for t, frame in enumerate(nuclei_record):
        for j, nuc in enumerate(frame):
            if nuc.status >= 1 and nuc.identity:
                ref_names[(t, j)] = nuc.identity

    # Find 4-cell stage info for later axis computation
    mid_time = find_four_cell_mid(nuclei_record)

    # Get reference founders for lineage map
    ref_founders = {}
    if mid_time >= 0:
        alive = _get_alive(nuclei_record[mid_time])
        for idx, nuc in alive:
            name = nuc.identity.strip()
            if name in ("ABa", "ABp", "EMS", "P2"):
                ref_founders[name] = (idx, nuc)

    # Build reference lineage map
    ref_lm = None
    if len(ref_founders) == 4:
        ref_lm = build_lineage_map(
            nuclei_record,
            four_cell_time=mid_time,
            aba_idx=ref_founders["ABa"][0],
            abp_idx=ref_founders["ABp"][0],
            ems_idx=ref_founders["EMS"][0],
            p2_idx=ref_founders["P2"][0],
        )

    # Deep copy and clear names for de novo
    nr_copy = []
    for t_nuclei in nuclei_record:
        frame = []
        for n in t_nuclei:
            nc = Nucleus()
            nc.x = n.x; nc.y = n.y; nc.z = n.z
            nc.size = n.size; nc.status = n.status
            nc.predecessor = n.predecessor
            nc.successor1 = n.successor1; nc.successor2 = n.successor2
            nc.weight = n.weight
            nc.identity = ""
            nc.assigned_id = n.assigned_id
            frame.append(nc)
        nr_copy.append(frame)

    # Run de novo naming
    assigner = IdentityAssigner(
        nr_copy, naming_method=NEWCANONICAL, z_pix_res=Z_PIX_RES,
    )
    assigner.assign_identities()

    # Find first N mismatches that are NOT cascades
    mismatches = []
    mismatch_parents = set()

    for t in range(len(nuclei_record)):
        for j in range(len(nuclei_record[t])):
            ref = ref_names.get((t, j), "")
            new = nr_copy[t][j].identity if j < len(nr_copy[t]) else ""
            if not ref or not new:
                continue
            if ref != new:
                # Check if parent was also mismatched
                nuc = nuclei_record[t][j]
                is_cascade = False
                if t > 0 and nuc.predecessor >= 1:
                    pred_idx = nuc.predecessor - 1
                    if (t-1, pred_idx) in mismatch_parents:
                        is_cascade = True

                mismatch_parents.add((t, j))
                mismatches.append({
                    "t": t, "j": j, "ref": ref, "new": new,
                    "cascade": is_cascade,
                })

    # Focus on root-cause mismatches (not cascades)
    roots = [m for m in mismatches if not m["cascade"]]
    print(f"  Total mismatches: {len(mismatches)}, root-cause: {len(roots)}")
    print(f"  First mismatch: t={mismatches[0]['t'] if mismatches else 'none'}")

    # Analyze first 15 root-cause mismatches
    print(f"\n  --- First 15 root-cause mismatches ---")
    for m in roots[:15]:
        t, j = m["t"], m["j"]
        ref_name = m["ref"]
        new_name = m["new"]
        nuc = nuclei_record[t][j]

        # Find the parent and determine if this is a division result
        parent_info = ""
        division_analysis = ""
        if t > 0 and nuc.predecessor >= 1:
            pred_idx = nuc.predecessor - 1
            if 0 <= pred_idx < len(nuclei_record[t-1]):
                pred = nuclei_record[t-1][pred_idx]
                parent_ref = ref_names.get((t-1, pred_idx), "?")
                parent_new = nr_copy[t-1][pred_idx].identity if pred_idx < len(nr_copy[t-1]) else "?"
                parent_info = f"parent_ref={parent_ref}, parent_new={parent_new}"

                # If parent divided, analyze the division
                if pred.successor2 != NILLI and pred.successor2 > 0:
                    s1_idx = pred.successor1 - 1
                    s2_idx = pred.successor2 - 1
                    if 0 <= s1_idx < len(nuclei_record[t]) and 0 <= s2_idx < len(nuclei_record[t]):
                        d1 = nuclei_record[t][s1_idx]
                        d2 = nuclei_record[t][s2_idx]
                        d1_ref = ref_names.get((t, s1_idx), "?")
                        d2_ref = ref_names.get((t, s2_idx), "?")
                        d1_new = nr_copy[t][s1_idx].identity if s1_idx < len(nr_copy[t]) else "?"
                        d2_new = nr_copy[t][s2_idx].identity if s2_idx < len(nr_copy[t]) else "?"

                        # Division vector
                        div_vec = pos3d(d2) - pos3d(d1)

                        # Get rule
                        rule = rule_mgr.get_rule(parent_ref)

                        # Get axes at this timepoint (reference lineage)
                        axes_str = "N/A"
                        angle_str = "N/A"
                        if ref_lm is not None:
                            axes = compute_local_axes(nuclei_record, ref_lm, t, Z_PIX_RES)
                            if axes[0] is not None:
                                ap_v, lr_v, dv_v = axes
                                # Project to canonical
                                ap_comp = np.dot(div_vec, ap_v)
                                lr_comp = np.dot(div_vec, lr_v)
                                dv_comp = np.dot(div_vec, dv_v)
                                canonical = np.array([-ap_comp, dv_comp, lr_comp])
                                dot = np.dot(canonical, rule.axis_vector)
                                ang = angle_between(canonical, rule.axis_vector)

                                # Get axes from OUR lineage map
                                if assigner.division_caller is not None:
                                    our_axes = assigner.division_caller._get_local_axes(t)
                                    if our_axes is not None:
                                        our_ap, our_lr, our_dv = our_axes
                                        ref_our_ap_angle = angle_between(ap_v, our_ap)
                                        axes_str = f"ref_vs_our_AP={ref_our_ap_angle:.1f}deg"

                                angle_str = f"angle_from_rule={ang:.1f}deg, dot={dot:.1f}"

                                division_analysis = (
                                    f"\n        division: {parent_ref}->{d1_ref},{d2_ref} "
                                    f"(our: {parent_new}->{d1_new},{d2_new})"
                                    f"\n        rule: axis={rule.sulston_letter}, "
                                    f"d1={rule.daughter1}, d2={rule.daughter2}"
                                    f"\n        {angle_str}, {axes_str}"
                                )

        # Check if it's a sister swap
        swap = ""
        if t > 0 and nuc.predecessor >= 1:
            pred_idx = nuc.predecessor - 1
            if 0 <= pred_idx < len(nuclei_record[t-1]):
                pred = nuclei_record[t-1][pred_idx]
                if pred.successor2 != NILLI and pred.successor2 > 0:
                    s1_idx = pred.successor1 - 1
                    s2_idx = pred.successor2 - 1
                    sister_idx = s2_idx if j == s1_idx else s1_idx
                    if 0 <= sister_idx < len(nuclei_record[t]):
                        sister_ref = ref_names.get((t, sister_idx), "")
                        sister_new = nr_copy[t][sister_idx].identity if sister_idx < len(nr_copy[t]) else ""
                        if new_name == sister_ref and sister_new == ref_name:
                            swap = " [SISTER SWAP]"

        print(f"    t={t:3d} j={j:3d}: ref={ref_name:15s} new={new_name:15s}{swap}")
        if parent_info:
            print(f"        {parent_info}")
        if division_analysis:
            print(f"        {division_analysis}")

    # Summary: count how many root mismatches are sister swaps vs genuine errors
    swap_count = 0
    for m in roots[:50]:
        t, j = m["t"], m["j"]
        nuc = nuclei_record[t][j]
        if t > 0 and nuc.predecessor >= 1:
            pred_idx = nuc.predecessor - 1
            if 0 <= pred_idx < len(nuclei_record[t-1]):
                pred = nuclei_record[t-1][pred_idx]
                if pred.successor2 != NILLI and pred.successor2 > 0:
                    s1_idx = pred.successor1 - 1
                    s2_idx = pred.successor2 - 1
                    sister_idx = s2_idx if j == s1_idx else s1_idx
                    if 0 <= sister_idx < len(nuclei_record[t]):
                        sister_ref = ref_names.get((t, sister_idx), "")
                        sister_new = nr_copy[t][sister_idx].identity if sister_idx < len(nr_copy[t]) else ""
                        if m["new"] == sister_ref and sister_new == m["ref"]:
                            swap_count += 1

    print(f"\n  Of first 50 root mismatches: {swap_count} are sister swaps "
          f"({len(roots[:50]) - swap_count} are genuine errors)")
