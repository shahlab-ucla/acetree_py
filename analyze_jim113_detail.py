"""Deep dive into JIM113 dataset — debug the t=31 ABar/ABpl/ABpr swaps.

The Sulston analysis showed ref_ax and our_ax giving different dot products
despite AP_diff=0 and LR_diff=0.  This script traces the exact computation
to find the discrepancy.
"""
import copy, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.naming.founder_id import identify_founders, _get_alive
from acetree_py.naming.lineage_axes import build_lineage_map, compute_local_axes, axes_to_canonical
from acetree_py.naming.identity import IdentityAssigner, NEWCANONICAL
from acetree_py.naming.rules import RuleManager

Z = 11.1
rule_mgr = RuleManager()

nr = read_nuclei_zip(r'C:\Users\pavak\Documents\AT_test\du_lab\YML_JIM113_20240108_3_s1_emb1_edited.zip')

# Get reference founders at 4-cell midpoint
mid = 10  # from earlier analysis
alive = _get_alive(nr[mid])
ref_f = {}
for idx, nuc in alive:
    ref_f[nuc.identity.strip()] = (idx, nuc)

ref_lm = build_lineage_map(nr, mid, ref_f["ABa"][0], ref_f["ABp"][0],
                            ref_f["EMS"][0], ref_f["P2"][0])

# Deep copy, clear, run de novo
nr2 = []
for frame in nr:
    f2 = []
    for n in frame:
        nc = Nucleus()
        nc.x=n.x; nc.y=n.y; nc.z=n.z; nc.size=n.size; nc.status=n.status
        nc.predecessor=n.predecessor; nc.successor1=n.successor1; nc.successor2=n.successor2
        nc.weight=n.weight; nc.identity=""; nc.assigned_id=n.assigned_id
        f2.append(nc)
    nr2.append(f2)

assigner = IdentityAssigner(nr2, naming_method=NEWCANONICAL, z_pix_res=Z)
assigner.assign_identities()

# Our lineage map (from de novo founder assignment)
fa = assigner.founder_assignment
our_lm = build_lineage_map(nr2, fa.four_cell_time, fa.aba_idx, fa.abp_idx,
                            fa.ems_idx, fa.p2_idx)

print("=" * 80)
print("JIM113 DEEP DIVE: t=31 division analysis")
print("=" * 80)

# Check lineage maps are identical
print("\n--- Lineage map comparison at t=31 ---")
for j in range(min(len(ref_lm[31]), len(our_lm[31]))):
    r = ref_lm[31][j]
    o = our_lm[31][j]
    if r != o:
        print(f"  j={j}: ref_label={r!r} our_label={o!r} DIFFER")
if all(ref_lm[31][j] == our_lm[31][j] for j in range(min(len(ref_lm[31]), len(our_lm[31])))):
    print("  All labels identical")

# Compute axes from both lineage maps at t=31
ref_axes = compute_local_axes(nr, ref_lm, 31, Z)
our_axes_fresh = compute_local_axes(nr2, our_lm, 31, Z)
our_axes_cached = assigner.division_caller._get_local_axes(31)

print("\n--- Axes at t=31 ---")
for name, axes in [("ref", ref_axes), ("our_fresh", our_axes_fresh), ("our_cached", our_axes_cached or (None,None,None))]:
    if axes[0] is not None:
        print(f"  {name:12s}: AP={np.round(axes[0],4)}, LR={np.round(axes[1],4)}, DV={np.round(axes[2],4)}")
    else:
        print(f"  {name:12s}: None")

# Now trace the EXACT ABar division at t=30 (parent) -> t=31 (daughters)
# Find ABar in the reference data
abar_parent = None
for j, n in enumerate(nr[30]):
    if n.identity.strip() == "ABar" and n.successor2 != NILLI and n.successor2 > 0:
        abar_parent = (j, n)
        break

if abar_parent:
    j_parent, parent = abar_parent
    s1 = parent.successor1 - 1
    s2 = parent.successor2 - 1
    d1 = nr[31][s1]
    d2 = nr[31][s2]
    print(f"\n--- ABar division at t=30 -> t=31 ---")
    print(f"  Parent: j={j_parent}, name={parent.identity}")
    print(f"  s1={s1}: ref={d1.identity}, pos=({d1.x},{d1.y},{d1.z})")
    print(f"  s2={s2}: ref={d2.identity}, pos=({d2.x},{d2.y},{d2.z})")
    print(f"  s1 de novo: {nr2[31][s1].identity}")
    print(f"  s2 de novo: {nr2[31][s2].identity}")

    # Raw division vector
    div_raw = np.array([d2.x - d1.x, d2.y - d1.y, (d2.z - d1.z) * Z])
    print(f"\n  div_vec (d2-d1) raw: {np.round(div_raw, 1)}")

    rule = rule_mgr.get_rule("ABar")
    print(f"  rule: axis={rule.sulston_letter}, vec={np.round(rule.axis_vector, 3)}, "
          f"d1={rule.daughter1}, d2={rule.daughter2}")

    # Project using reference axes
    if ref_axes[0] is not None:
        ap, lr, dv = ref_axes
        canonical_ref = axes_to_canonical(div_raw, ap, lr, dv)
        dot_ref = np.dot(canonical_ref, rule.axis_vector)
        print(f"\n  Using REF axes:")
        print(f"    canonical = {np.round(canonical_ref, 2)}")
        print(f"    dot = {dot_ref:.2f}")
        print(f"    assign: {'ABara,ABarp' if dot_ref >= 0 else 'ABarp,ABara'}")

    # Project using our cached axes
    if our_axes_cached is not None:
        ap, lr, dv = our_axes_cached
        canonical_our = axes_to_canonical(div_raw, ap, lr, dv)
        dot_our = np.dot(canonical_our, rule.axis_vector)
        print(f"\n  Using OUR cached axes:")
        print(f"    canonical = {np.round(canonical_our, 2)}")
        print(f"    dot = {dot_our:.2f}")
        print(f"    assign: {'ABara,ABarp' if dot_our >= 0 else 'ABarp,ABara'}")

    # What does multi-frame averaging do?
    print(f"\n  --- Multi-frame analysis ---")
    for frame_offset in range(4):
        t = 31 + frame_offset
        if t >= len(nr):
            break
        # Follow both daughters forward
        if frame_offset == 0:
            curr_d1, curr_d2 = d1, d2
        else:
            # Follow successor1
            if curr_d1.successor1 < 1 or curr_d2.successor1 < 1:
                print(f"    t={t}: tracking lost")
                break
            if curr_d1.successor2 != NILLI or curr_d2.successor2 != NILLI:
                print(f"    t={t}: one daughter divided, stopping")
                break
            i1 = curr_d1.successor1 - 1
            i2 = curr_d2.successor1 - 1
            if i1 >= len(nr[t]) or i2 >= len(nr[t]):
                break
            curr_d1 = nr[t][i1]
            curr_d2 = nr[t][i2]

        dv_frame = np.array([curr_d2.x - curr_d1.x, curr_d2.y - curr_d1.y,
                             (curr_d2.z - curr_d1.z) * Z])
        dv_norm = np.linalg.norm(dv_frame)

        # Get axes at this frame
        axes_t = compute_local_axes(nr, ref_lm, t, Z)
        our_axes_t = assigner.division_caller._get_local_axes(t)

        if axes_t[0] is not None:
            can = axes_to_canonical(dv_frame, *axes_t)
            dot_t = np.dot(can, rule.axis_vector)
        else:
            can = dv_frame
            dot_t = float('nan')

        if our_axes_t is not None:
            can_o = axes_to_canonical(dv_frame, *our_axes_t)
            dot_o = np.dot(can_o, rule.axis_vector)
        else:
            can_o = dv_frame
            dot_o = float('nan')

        print(f"    t={t}: dv_norm={dv_norm:.1f}, ref_dot={dot_t:.1f}, our_dot={dot_o:.1f}")

# Same analysis for ABpl
print("\n" + "=" * 80)
abpl_parent = None
for j, n in enumerate(nr[30]):
    if n.identity.strip() == "ABpl" and n.successor2 != NILLI and n.successor2 > 0:
        abpl_parent = (j, n)
        break

if abpl_parent:
    j_parent, parent = abpl_parent
    s1 = parent.successor1 - 1
    s2 = parent.successor2 - 1
    d1 = nr[31][s1]
    d2 = nr[31][s2]
    print(f"--- ABpl division at t=30 -> t=31 ---")
    print(f"  s1={s1}: ref={d1.identity}, pos=({d1.x},{d1.y},{d1.z})")
    print(f"  s2={s2}: ref={d2.identity}, pos=({d2.x},{d2.y},{d2.z})")
    print(f"  s1 de novo: {nr2[31][s1].identity}")
    print(f"  s2 de novo: {nr2[31][s2].identity}")

    div_raw = np.array([d2.x - d1.x, d2.y - d1.y, (d2.z - d1.z) * Z])
    rule = rule_mgr.get_rule("ABpl")
    print(f"  div_vec: {np.round(div_raw, 1)}")
    print(f"  rule: axis={rule.sulston_letter}, d1={rule.daughter1}, d2={rule.daughter2}")

    if ref_axes[0] is not None:
        can = axes_to_canonical(div_raw, *ref_axes)
        dot = np.dot(can, rule.axis_vector)
        print(f"  ref canonical: {np.round(can, 2)}, dot={dot:.2f}")

    if our_axes_cached is not None:
        can = axes_to_canonical(div_raw, *our_axes_cached)
        dot = np.dot(can, rule.axis_vector)
        print(f"  our canonical: {np.round(can, 2)}, dot={dot:.2f}")

# Check: is the parent named correctly in de novo?
print(f"\n--- De novo parent names at t=30 ---")
for j, n in enumerate(nr2[30]):
    if n.identity and n.identity.startswith("AB"):
        ref_name = nr[30][j].identity
        print(f"  j={j}: de_novo={n.identity:15s} ref={ref_name:15s} {'OK' if n.identity == ref_name else 'DIFFER'}")

# Check: are the axes at t=30 the same? (before division, used for single-frame)
print(f"\n--- Axes comparison at multiple timepoints ---")
for t in [20, 25, 30, 31, 35, 40]:
    ref_ax = compute_local_axes(nr, ref_lm, t, Z)
    our_ax = compute_local_axes(nr2, our_lm, t, Z)
    cached = assigner.division_caller._get_local_axes(t)
    if ref_ax[0] is not None and our_ax[0] is not None:
        ap_diff = math.degrees(math.acos(min(1, abs(np.dot(ref_ax[0], our_ax[0])))))
        lr_diff = math.degrees(math.acos(min(1, abs(np.dot(ref_ax[1], our_ax[1])))))
        print(f"  t={t:3d}: AP_diff(ref-fresh)={ap_diff:.2f}d, LR_diff={lr_diff:.2f}d")
        if cached is not None:
            ap_d2 = math.degrees(math.acos(min(1, abs(np.dot(ref_ax[0], cached[0])))))
            lr_d2 = math.degrees(math.acos(min(1, abs(np.dot(ref_ax[1], cached[1])))))
            # Check sign: does the cached LR point same way as ref LR?
            lr_sign = np.dot(ref_ax[1], cached[1])
            print(f"         AP_diff(ref-cached)={ap_d2:.2f}d, LR_diff={lr_d2:.2f}d, LR_sign={lr_sign:.4f}")
    else:
        print(f"  t={t:3d}: axes=None")
