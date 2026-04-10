"""Trace the exact cache population order to find when LR flips in JIM113."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.naming.founder_id import identify_founders, _get_alive
from acetree_py.naming.lineage_axes import build_lineage_map, compute_local_axes, check_axis_continuity
from acetree_py.naming.identity import IdentityAssigner, NEWCANONICAL

Z = 11.1
nr = read_nuclei_zip(r'C:\Users\pavak\Documents\AT_test\du_lab\YML_JIM113_20240108_3_s1_emb1_edited.zip')

# Build reference axes for comparison
mid = 10
alive = _get_alive(nr[mid])
ref_f = {}
for idx, nuc in alive:
    ref_f[nuc.identity.strip()] = (idx, nuc)
ref_lm = build_lineage_map(nr, mid, ref_f["ABa"][0], ref_f["ABp"][0],
                            ref_f["EMS"][0], ref_f["P2"][0])

# Compute reference LR at t=10 (seed) for sign comparison
ref_axes_10 = compute_local_axes(nr, ref_lm, 10, Z)
ref_lr_sign = ref_axes_10[1]  # Use this as the "true" LR direction

# Now, monkey-patch _get_local_axes to trace cache population
from acetree_py.naming.division_caller import DivisionCaller

original_get_local_axes = DivisionCaller._get_local_axes

def traced_get_local_axes(self, t):
    was_cached = t in self._axes_cache

    # Get result via original method
    result = original_get_local_axes(self, t)

    if not was_cached and result is not None:
        # This was a NEW cache entry
        lr_dot_ref = np.dot(result[1], ref_lr_sign) if ref_lr_sign is not None else 0
        # Find what prev_t was used
        prev_t = None
        if self._axes_cache:
            candidates = [k for k in self._axes_cache if k < t and k != t]
            if candidates:
                prev_t = max(candidates)

        prev_lr_dot = None
        if prev_t is not None and prev_t in self._axes_cache:
            prev_lr_dot = np.dot(self._axes_cache[prev_t][1], ref_lr_sign)

        # Compute fresh (before continuity check)
        fresh = compute_local_axes(self._nuclei_record, self._lineage_map, t, self.z_pix_res)
        fresh_lr_dot = np.dot(fresh[1], ref_lr_sign) if fresh[1] is not None else 0

        if abs(lr_dot_ref) < 0.95 or (prev_t is not None and prev_lr_dot is not None and prev_lr_dot < 0):
            sign_str = "CORRECT" if lr_dot_ref > 0 else "FLIPPED"
            fresh_str = "correct" if fresh_lr_dot > 0 else "flipped"
            prev_str = f"prev_t={prev_t} prev_lr_dot={prev_lr_dot:.3f}" if prev_t is not None else "no_prev"
            print(f"  CACHE t={t:3d}: lr_dot={lr_dot_ref:+.3f} ({sign_str}), "
                  f"fresh_lr_dot={fresh_lr_dot:+.3f} ({fresh_str}), {prev_str}")

    return result

DivisionCaller._get_local_axes = traced_get_local_axes

# Run de novo naming
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

print("Running de novo naming with cache tracing...")
print("(Showing only entries where LR sign is flipped or unusual)")
assigner = IdentityAssigner(nr2, naming_method=NEWCANONICAL, z_pix_res=Z)
assigner.assign_identities()

# Now dump the full cache LR signs around t=25-35
print("\n--- Full cache LR dot products (t=20..40) ---")
dc = assigner.division_caller
for t in range(20, 41):
    if t in dc._axes_cache:
        lr_dot = np.dot(dc._axes_cache[t][1], ref_lr_sign)
        sign = "CORRECT" if lr_dot > 0 else "FLIPPED"
        print(f"  t={t:3d}: lr_dot_ref={lr_dot:+.4f} ({sign})")
    else:
        print(f"  t={t:3d}: not cached")
