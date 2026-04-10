"""Analyze ABa/ABp identification patterns across all 5 du_lab datasets.

Examines cell positions, division timing, axis geometry, and compares
our founder_id assignments to the curated reference names.
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
from acetree_py.naming.rules import RuleManager

Z_PIX_RES = 11.1

ZIP_DIR = Path(r"C:\Users\pavak\Documents\AT_test\du_lab")
ZIP_FILES = sorted(ZIP_DIR.glob("*.zip"))


def pos3d(nuc, z_pix_res=Z_PIX_RES):
    return np.array([float(nuc.x), float(nuc.y), float(nuc.z) * z_pix_res])


def find_four_cell_window(nuclei_record):
    """Return (first_four, last_four) or None."""
    first = -1
    for t in range(len(nuclei_record)):
        ct = _count_alive(nuclei_record[t])
        if ct == 4:
            if first < 0:
                first = t
        else:
            if first >= 0:
                return (first, t - 1)
            if ct > 4:
                break
    if first >= 0:
        return (first, len(nuclei_record) - 1)
    return None


def trace_forward_to_division(nuclei_record, nuc, start_t):
    """Return timepoint when nuc divides, or -1."""
    t = start_t
    current = nuc
    for _ in range(200):
        if current.successor2 != NILLI and current.successor2 > 0:
            return t
        if current.successor1 == NILLI or current.successor1 < 1:
            return -1
        if t + 1 >= len(nuclei_record):
            return -1
        s_idx = current.successor1 - 1
        t += 1
        if s_idx < 0 or s_idx >= len(nuclei_record[t]):
            return -1
        current = nuclei_record[t][s_idx]
    return -1


def trace_back_to_birth(nuclei_record, nuc, current_time):
    """Return (birth_time, parent_time, parent_idx) or (current_time, -1, -1)."""
    t = current_time
    current = nuc
    for _ in range(200):
        if t <= 0 or current.predecessor == NILLI or current.predecessor < 1:
            return t, -1, -1
        pred_idx = current.predecessor - 1
        if pred_idx < 0 or pred_idx >= len(nuclei_record[t - 1]):
            return t, -1, -1
        pred = nuclei_record[t - 1][pred_idx]
        if pred.successor2 != NILLI and pred.successor2 > 0:
            return t, t - 1, pred_idx
        t -= 1
        current = pred
    return t, -1, -1


def get_ref_founders(nuclei_record, mid_time):
    """Get reference founder cells by their names at mid_time."""
    alive = _get_alive(nuclei_record[mid_time])
    founders = {}
    for idx, nuc in alive:
        name = nuc.identity.strip()
        if name in ("ABa", "ABp", "EMS", "P2"):
            founders[name] = (idx, nuc)
    return founders


def build_ref_lineage_map(nuclei_record, mid_time, founders):
    """Build lineage map using reference founder assignments."""
    return build_lineage_map(
        nuclei_record,
        four_cell_time=mid_time,
        aba_idx=founders["ABa"][0],
        abp_idx=founders["ABp"][0],
        ems_idx=founders["EMS"][0],
        p2_idx=founders["P2"][0],
    )


def angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 90.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return math.degrees(math.acos(abs(cos_a)))


def find_division_daughters(nuclei_record, parent_nuc, div_time):
    """Get daughter nuclei at div_time+1."""
    s1 = parent_nuc.successor1 - 1
    s2 = parent_nuc.successor2 - 1
    next_t = div_time + 1
    if next_t >= len(nuclei_record):
        return None, None
    nn = nuclei_record[next_t]
    d1 = nn[s1] if 0 <= s1 < len(nn) else None
    d2 = nn[s2] if 0 <= s2 < len(nn) else None
    return d1, d2


print("=" * 80)
print("CROSS-DATASET ABa/ABp ANALYSIS")
print("=" * 80)

rule_mgr = RuleManager()

for zip_path in ZIP_FILES:
    short_name = zip_path.stem[:30]
    print(f"\n{'='*80}")
    print(f"DATASET: {short_name}")
    print(f"{'='*80}")

    nuclei_record = read_nuclei_zip(zip_path)

    # Find 4-cell window
    window = find_four_cell_window(nuclei_record)
    if window is None:
        print("  NO 4-CELL WINDOW FOUND")
        continue
    first_four, last_four = window
    mid_time = (first_four + last_four) // 2
    print(f"  4-cell window: t={first_four}-{last_four}, mid={mid_time}")

    # Reference founders
    founders = get_ref_founders(nuclei_record, mid_time)
    if len(founders) != 4:
        print(f"  WARNING: Found {len(founders)} founders (expected 4): {list(founders.keys())}")
        alive = _get_alive(nuclei_record[mid_time])
        print(f"  Alive cells: {[(idx, n.identity) for idx, n in alive]}")
        continue

    print(f"\n  --- Reference 4-cell positions at t={mid_time} ---")
    for name in ["ABa", "ABp", "EMS", "P2"]:
        idx, nuc = founders[name]
        p = pos3d(nuc)
        print(f"    {name:4s}: idx={idx:2d}, pos=({nuc.x:6.1f}, {nuc.y:6.1f}, {nuc.z:5.1f}), "
              f"z_scaled={p[2]:7.1f}, size={nuc.size}")

    # Compute reference AP, LR, DV from 4-cell positions
    aba_pos = pos3d(founders["ABa"][1])
    abp_pos = pos3d(founders["ABp"][1])
    ems_pos = pos3d(founders["EMS"][1])
    p2_pos = pos3d(founders["P2"][1])

    ab_center = (aba_pos + abp_pos) / 2.0
    p1_center = (ems_pos + p2_pos) / 2.0
    ap_raw = ab_center - p2_pos
    ap_norm = np.linalg.norm(ap_raw)
    ap_hat = ap_raw / ap_norm if ap_norm > 1e-6 else np.array([1, 0, 0])

    # PC1 of 4-cell point cloud
    pts = np.array([aba_pos, abp_pos, ems_pos, p2_pos])
    centroid = np.mean(pts, axis=0)
    pts_c = pts - centroid
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    pc1 = Vt[0]

    # ABa-ABp separation perpendicular to AP
    ab_sep = aba_pos - abp_pos
    ab_sep_perp = ab_sep - np.dot(ab_sep, ap_hat) * ap_hat

    print(f"\n  --- Geometry ---")
    print(f"    AP vector (P2->AB): {np.round(ap_hat, 3)}, norm={ap_norm:.1f}")
    print(f"    PC1 (long axis):   {np.round(pc1, 3)}")
    print(f"    AP·PC1 angle:      {angle_between(ap_hat, pc1):.1f}°")
    print(f"    ABa-ABp sep raw:   {np.round(ab_sep, 1)}, norm={np.linalg.norm(ab_sep):.1f}")
    print(f"    ABa-ABp sep ⊥AP:   {np.round(ab_sep_perp, 1)}, norm={np.linalg.norm(ab_sep_perp):.1f}")

    # Projections onto AP
    proj_aba = np.dot(aba_pos, ap_hat)
    proj_abp = np.dot(abp_pos, ap_hat)
    print(f"    ABa proj onto AP:  {proj_aba:.1f}")
    print(f"    ABp proj onto AP:  {proj_abp:.1f}")
    print(f"    ABa more anterior: {proj_aba > proj_abp} (diff={proj_aba - proj_abp:.1f})")

    # Projections onto PC1
    proj_aba_pc1 = np.dot(aba_pos - centroid, pc1)
    proj_abp_pc1 = np.dot(abp_pos - centroid, pc1)
    ab_mid_proj = np.dot(ab_center - centroid, pc1)
    if ab_mid_proj < 0:
        proj_aba_pc1, proj_abp_pc1 = -proj_aba_pc1, -proj_abp_pc1
    print(f"    ABa proj onto PC1: {proj_aba_pc1:.1f}")
    print(f"    ABp proj onto PC1: {proj_abp_pc1:.1f}")

    # Birth times (trace back)
    print(f"\n  --- Birth times (backward trace) ---")
    for name in ["ABa", "ABp", "EMS", "P2"]:
        idx, nuc = founders[name]
        birth_t, parent_t, parent_idx = trace_back_to_birth(nuclei_record, nuc, mid_time)
        print(f"    {name:4s}: born at t={birth_t}, parent at t={parent_t}")

    # Forward division times
    print(f"\n  --- Forward division times ---")
    div_times = {}
    for name in ["ABa", "ABp", "EMS", "P2"]:
        idx, nuc = founders[name]
        div_t = trace_forward_to_division(nuclei_record, nuc, mid_time)
        div_times[name] = div_t
        print(f"    {name:4s}: divides at t={div_t}")

    # ----- Run our founder_id -----
    print(f"\n  --- Our founder_id assignment ---")
    # Deep copy to avoid mutating reference data
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
            nc.identity = ""  # clear
            nc.assigned_id = n.assigned_id
            frame.append(nc)
        nr_copy.append(frame)

    fa = identify_founders(nr_copy, z_pix_res=Z_PIX_RES)
    if fa.success:
        our_alive = _get_alive(nr_copy[fa.four_cell_time])
        our_map = {}
        for oi, on in our_alive:
            if on.identity:
                our_map[on.identity] = oi
        print(f"    Success, confidence={fa.confidence:.2f}")
        print(f"    Our mid_time={fa.four_cell_time}, ref mid_time={mid_time}")
        print(f"    Our ABa idx={fa.aba_idx}, Our ABp idx={fa.abp_idx}")
        print(f"    Our EMS idx={fa.ems_idx}, Our P2 idx={fa.p2_idx}")

        # Check if ABa/ABp swapped
        ref_aba_idx = founders["ABa"][0]
        ref_abp_idx = founders["ABp"][0]
        ref_ems_idx = founders["EMS"][0]
        ref_p2_idx = founders["P2"][0]

        aba_match = fa.aba_idx == ref_aba_idx
        abp_match = fa.abp_idx == ref_abp_idx
        ems_match = fa.ems_idx == ref_ems_idx
        p2_match = fa.p2_idx == ref_p2_idx

        ab_swapped = (fa.aba_idx == ref_abp_idx and fa.abp_idx == ref_aba_idx)
        ep_swapped = (fa.ems_idx == ref_p2_idx and fa.p2_idx == ref_ems_idx)

        print(f"    ABa correct: {aba_match}, ABp correct: {abp_match}")
        print(f"    EMS correct: {ems_match}, P2 correct: {p2_match}")
        print(f"    AB pair SWAPPED: {ab_swapped}")
        print(f"    EMS/P2 SWAPPED:  {ep_swapped}")
    else:
        print(f"    FAILED (confidence={fa.confidence:.2f})")

    # ----- Lineage centroid axes evolution -----
    print(f"\n  --- Lineage centroid axes (using REFERENCE lineage labels) ---")
    ref_lm = build_ref_lineage_map(nuclei_record, mid_time, founders)
    sample_times = [mid_time] + list(range(mid_time + 5, min(mid_time + 30, len(nuclei_record)), 5))
    prev_ap = None
    for st in sample_times:
        axes = compute_local_axes(nuclei_record, ref_lm, st, Z_PIX_RES)
        if axes[0] is not None:
            ap_v, lr_v, dv_v = axes
            # Count cells per lineage group
            labels = ref_lm[st]
            n_aba = sum(1 for l in labels if l == "ABa")
            n_abp = sum(1 for l in labels if l == "ABp")
            n_ems = sum(1 for l in labels if l == "EMS")
            n_p2 = sum(1 for l in labels if l == "P2")
            ap_angle = angle_between(ap_v, prev_ap) if prev_ap is not None else 0.0
            print(f"    t={st:3d}: AP={np.round(ap_v,2)}, "
                  f"cells=ABa:{n_aba}/ABp:{n_abp}/EMS:{n_ems}/P2:{n_p2}, "
                  f"AP_change={ap_angle:.1f}°")
            prev_ap = ap_v
        else:
            print(f"    t={st:3d}: axes=None")

    # ----- Early division analysis -----
    print(f"\n  --- Early division events (4→8 cell and beyond) ---")
    # Find cells that divide first after the 4-cell stage
    divisions_to_check = []
    for name in ["ABa", "ABp", "EMS", "P2"]:
        idx, nuc = founders[name]
        div_t = div_times[name]
        if div_t > 0:
            # Follow nuc to div_t
            current = nuc
            t = mid_time
            while t < div_t:
                if current.successor1 < 1:
                    break
                s_idx = current.successor1 - 1
                t += 1
                if s_idx < 0 or s_idx >= len(nuclei_record[t]):
                    break
                current = nuclei_record[t][s_idx]
            if t == div_t and current.successor2 != NILLI and current.successor2 > 0:
                divisions_to_check.append((name, div_t, current))

    divisions_to_check.sort(key=lambda x: x[1])

    for parent_name, div_t, parent_nuc in divisions_to_check[:6]:
        d1, d2 = find_division_daughters(nuclei_record, parent_nuc, div_t)
        if d1 is None or d2 is None:
            continue

        p1_pos = pos3d(d1)
        p2_p = pos3d(d2)
        div_vec = p2_p - p1_pos

        # Get axes at division time from reference lineage
        axes = compute_local_axes(nuclei_record, ref_lm, div_t + 1, Z_PIX_RES)

        # Get rule for this parent
        rule = rule_mgr.get_rule(parent_name)

        print(f"    {parent_name} divides at t={div_t}:")
        print(f"      d1: {d1.identity:12s} pos=({d1.x:.0f},{d1.y:.0f},{d1.z:.1f})")
        print(f"      d2: {d2.identity:12s} pos=({d2.x:.0f},{d2.y:.0f},{d2.z:.1f})")
        print(f"      div_vec raw: ({div_vec[0]:.1f}, {div_vec[1]:.1f}, {div_vec[2]:.1f})")
        print(f"      rule: axis={rule.sulston_letter}, vec={np.round(rule.axis_vector, 2)}, "
              f"d1={rule.daughter1}, d2={rule.daughter2}")

        if axes[0] is not None:
            ap_v, lr_v, dv_v = axes
            # Project div_vec onto canonical frame
            ap_comp = np.dot(div_vec, ap_v)
            lr_comp = np.dot(div_vec, lr_v)
            dv_comp = np.dot(div_vec, dv_v)
            canonical = np.array([-ap_comp, dv_comp, lr_comp])
            dot_with_rule = np.dot(canonical, rule.axis_vector)
            angle_from_rule = angle_between(canonical, rule.axis_vector)

            # What names would our code assign?
            if dot_with_rule >= 0:
                our_d1, our_d2 = rule.daughter1, rule.daughter2
            else:
                our_d1, our_d2 = rule.daughter2, rule.daughter1

            print(f"      canonical: ({canonical[0]:.1f}, {canonical[1]:.1f}, {canonical[2]:.1f})")
            print(f"      dot_with_rule={dot_with_rule:.1f}, angle={angle_from_rule:.1f}°")
            print(f"      our assign: d1={our_d1}, d2={our_d2}")
            print(f"      ref assign: d1={d1.identity}, d2={d2.identity}")
            print(f"      CORRECT: {our_d1 == d1.identity and our_d2 == d2.identity}")
        else:
            print(f"      axes=None at t={div_t+1}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
