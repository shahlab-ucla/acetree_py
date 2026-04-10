"""Analyze naming errors among Sulston-named cells only, first 200 timepoints.

For each dataset:
1. Load reference, run de novo
2. Compare ONLY cells with Sulston names (not Nuc*)
3. Find root-cause mismatches and analyze the division that caused each
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
MAX_T = 200
ZIP_DIR = Path(r"C:\Users\pavak\Documents\AT_test\du_lab")
ZIP_FILES = sorted(ZIP_DIR.glob("*.zip"))

rule_mgr = RuleManager()


def pos3d(nuc):
    return np.array([float(nuc.x), float(nuc.y), float(nuc.z) * Z_PIX_RES])


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 90.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return math.degrees(math.acos(abs(cos_a)))


def is_sulston(name):
    """Check if name is a real Sulston lineage name (not Nuc*)."""
    if not name:
        return False
    sulston_prefixes = ("P0", "AB", "P1", "EMS", "P2", "P3", "P4",
                        "MS", "E", "C", "D", "Z2", "Z3")
    for pfx in sulston_prefixes:
        if name.startswith(pfx):
            return True
    return False


def find_four_cell_mid(nuclei_record):
    first = -1
    for t in range(min(len(nuclei_record), MAX_T)):
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


all_dataset_summary = []

for zip_path in ZIP_FILES:
    short_name = zip_path.stem[:45]
    print(f"\n{'='*80}")
    print(f"DATASET: {short_name}")
    print(f"{'='*80}")

    nuclei_record = read_nuclei_zip(zip_path)

    # Save reference Sulston names only
    ref_names = {}
    for t in range(min(len(nuclei_record), MAX_T)):
        for j, nuc in enumerate(nuclei_record[t]):
            if nuc.status >= 1 and is_sulston(nuc.identity):
                ref_names[(t, j)] = nuc.identity

    # Get reference founders
    mid_time = find_four_cell_mid(nuclei_record)
    ref_founders = {}
    if mid_time >= 0:
        alive = _get_alive(nuclei_record[mid_time])
        for idx, nuc in alive:
            name = nuc.identity.strip()
            if name in ("ABa", "ABp", "EMS", "P2"):
                ref_founders[name] = (idx, nuc)

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

    # Deep copy, clear names, run de novo
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

    assigner = IdentityAssigner(
        nr_copy, naming_method=NEWCANONICAL, z_pix_res=Z_PIX_RES,
    )
    assigner.assign_identities()

    # Collect new Sulston names
    new_names = {}
    for t in range(min(len(nr_copy), MAX_T)):
        for j, nuc in enumerate(nr_copy[t]):
            if nuc.status >= 1 and is_sulston(nuc.identity):
                new_names[(t, j)] = nuc.identity

    # Compare Sulston names only
    all_keys = set(ref_names.keys()) | set(new_names.keys())
    matched = 0
    mismatched_list = []
    mismatch_parents = set()

    for t, j in sorted(all_keys):
        if t >= MAX_T:
            continue
        ref = ref_names.get((t, j), "")
        new = new_names.get((t, j), "")
        if not ref or not new:
            continue
        if ref == new:
            matched += 1
        else:
            # Check cascade
            nuc = nuclei_record[t][j]
            is_cascade = False
            if t > 0 and nuc.predecessor >= 1:
                pred_idx = nuc.predecessor - 1
                if (t - 1, pred_idx) in mismatch_parents:
                    is_cascade = True
            mismatch_parents.add((t, j))

            # Check sister swap
            is_swap = False
            if t > 0 and nuc.predecessor >= 1:
                pred_idx = nuc.predecessor - 1
                if 0 <= pred_idx < len(nuclei_record[t - 1]):
                    pred = nuclei_record[t - 1][pred_idx]
                    if pred.successor2 != NILLI and pred.successor2 > 0:
                        s1 = pred.successor1 - 1
                        s2 = pred.successor2 - 1
                        sis = s2 if j == s1 else s1
                        if 0 <= sis < len(nuclei_record[t]):
                            sis_ref = ref_names.get((t, sis), "")
                            sis_new = new_names.get((t, sis), "")
                            if new == sis_ref and sis_new == ref:
                                is_swap = True

            mismatched_list.append({
                "t": t, "j": j, "ref": ref, "new": new,
                "cascade": is_cascade, "swap": is_swap,
            })

    total_sulston = matched + len(mismatched_list)
    roots = [m for m in mismatched_list if not m["cascade"]]
    swap_roots = [m for m in roots if m["swap"]]

    print(f"  Sulston cells (t<{MAX_T}): {total_sulston}")
    print(f"  Matched: {matched} ({matched/total_sulston*100:.1f}%)")
    print(f"  Mismatched: {len(mismatched_list)} ({len(mismatched_list)/total_sulston*100:.1f}%)")
    print(f"  Root-cause errors: {len(roots)}")
    print(f"  Root-cause sister swaps: {len(swap_roots)}")
    print(f"  Cascade descendants: {len(mismatched_list) - len(roots)}")

    all_dataset_summary.append({
        "name": short_name,
        "total": total_sulston,
        "matched": matched,
        "mismatched": len(mismatched_list),
        "match_pct": matched / total_sulston * 100 if total_sulston else 0,
        "roots": len(roots),
        "swap_roots": len(swap_roots),
    })

    # Analyze each root-cause error
    print(f"\n  --- Root-cause Sulston mismatches ---")
    for m in roots[:25]:
        t, j = m["t"], m["j"]
        ref_name, new_name = m["ref"], m["new"]
        swap_tag = " [SWAP]" if m["swap"] else ""

        # Division analysis
        nuc = nuclei_record[t][j]
        div_info = ""
        if t > 0 and nuc.predecessor >= 1:
            pred_idx = nuc.predecessor - 1
            if 0 <= pred_idx < len(nuclei_record[t - 1]):
                pred = nuclei_record[t - 1][pred_idx]
                parent_ref = ref_names.get((t - 1, pred_idx), "?")
                parent_new = new_names.get((t - 1, pred_idx), "?")

                if pred.successor2 != NILLI and pred.successor2 > 0:
                    s1 = pred.successor1 - 1
                    s2 = pred.successor2 - 1
                    if 0 <= s1 < len(nuclei_record[t]) and 0 <= s2 < len(nuclei_record[t]):
                        d1 = nuclei_record[t][s1]
                        d2 = nuclei_record[t][s2]
                        div_vec = pos3d(d2) - pos3d(d1)

                        rule = rule_mgr.get_rule(parent_ref)

                        # Compute axes at division time
                        angle_info = ""
                        if ref_lm is not None:
                            axes = compute_local_axes(nuclei_record, ref_lm, t, Z_PIX_RES)
                            if axes[0] is not None:
                                ap_v, lr_v, dv_v = axes
                                ap_comp = np.dot(div_vec, ap_v)
                                lr_comp = np.dot(div_vec, lr_v)
                                dv_comp = np.dot(div_vec, dv_v)
                                canonical = np.array([-ap_comp, dv_comp, lr_comp])
                                dot = np.dot(canonical, rule.axis_vector)
                                ang = angle_between(canonical, rule.axis_vector)

                                # Also check what our de novo axes give
                                if assigner.division_caller is not None:
                                    our_axes = assigner.division_caller._get_local_axes(t)
                                    if our_axes is not None:
                                        our_ap, our_lr, our_dv = our_axes
                                        ap_comp2 = np.dot(div_vec, our_ap)
                                        lr_comp2 = np.dot(div_vec, our_lr)
                                        dv_comp2 = np.dot(div_vec, our_dv)
                                        can2 = np.array([-ap_comp2, dv_comp2, lr_comp2])
                                        dot2 = np.dot(can2, rule.axis_vector)
                                        ang2 = angle_between(can2, rule.axis_vector)
                                        ap_diff = angle_between(ap_v, our_ap)
                                        lr_diff = angle_between(lr_v, our_lr)
                                        angle_info = (
                                            f"ref_ax: ang={ang:.0f}d dot={dot:.1f} | "
                                            f"our_ax: ang={ang2:.0f}d dot={dot2:.1f} | "
                                            f"AP_diff={ap_diff:.0f}d LR_diff={lr_diff:.0f}d"
                                        )
                                    else:
                                        angle_info = f"ref_ax: ang={ang:.0f}d dot={dot:.1f} | our_axes=None"
                                else:
                                    angle_info = f"ref_ax: ang={ang:.0f}d dot={dot:.1f}"

                        div_info = (
                            f"\n        {parent_ref}->{ref_names.get((t,s1),'?')},{ref_names.get((t,s2),'?')} "
                            f"rule={rule.sulston_letter} "
                            f"({rule.daughter1}/{rule.daughter2})"
                        )
                        if angle_info:
                            div_info += f"\n        {angle_info}"

        print(f"    t={t:3d} ref={ref_name:18s} new={new_name:18s}{swap_tag}{div_info}")

print(f"\n{'='*80}")
print(f"CROSS-DATASET SUMMARY (Sulston names, t<{MAX_T})")
print(f"{'='*80}")
print(f"{'Dataset':45s} {'Total':>7s} {'Match%':>7s} {'Roots':>6s} {'Swaps':>6s}")
for s in all_dataset_summary:
    print(f"  {s['name']:43s} {s['total']:7d} {s['match_pct']:6.1f}% {s['roots']:6d} {s['swap_roots']:6d}")
