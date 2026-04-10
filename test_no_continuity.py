"""Test: remove continuity correction and fix multi-frame averaging.
Compare results with and without the fix on JIM113."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.naming.lineage_axes import compute_local_axes
from acetree_py.naming.identity import IdentityAssigner, NEWCANONICAL
from acetree_py.naming.division_caller import DivisionCaller

Z = 11.1
MAX_T = 200

def is_sulston(name):
    if not name:
        return False
    for pfx in ("P0", "AB", "P1", "EMS", "P2", "P3", "P4",
                 "MS", "E", "C", "D", "Z2", "Z3"):
        if name.startswith(pfx):
            return True
    return False

def make_copy(nr):
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
    return nr2

def compare(nr_ref, nr_new, label):
    ref_names = {}
    for t in range(min(len(nr_ref), MAX_T)):
        for j, nuc in enumerate(nr_ref[t]):
            if nuc.status >= 1 and is_sulston(nuc.identity):
                ref_names[(t, j)] = nuc.identity

    matched = 0
    mismatched = 0
    mismatch_parents = set()
    roots = 0
    for t, j in sorted(ref_names):
        if t >= MAX_T or j >= len(nr_new[t]):
            continue
        new_name = nr_new[t][j].identity
        if not new_name or not is_sulston(new_name):
            continue
        if ref_names[(t, j)] == new_name:
            matched += 1
        else:
            nuc = nr_ref[t][j]
            is_cascade = False
            if t > 0 and nuc.predecessor >= 1:
                pred_idx = nuc.predecessor - 1
                if (t-1, pred_idx) in mismatch_parents:
                    is_cascade = True
            mismatch_parents.add((t, j))
            mismatched += 1
            if not is_cascade:
                roots += 1

    total = matched + mismatched
    pct = matched / total * 100 if total else 0
    print(f"  {label}: {matched}/{total} ({pct:.1f}%) matched, {roots} root errors, {mismatched - roots} cascades")
    return matched, total, roots

# Patch: remove continuity correction
def get_local_axes_no_continuity(self, t):
    if t in self._axes_cache:
        return self._axes_cache[t]
    axes = compute_local_axes(self._nuclei_record, self._lineage_map, t, self.z_pix_res)
    if axes[0] is None or axes[1] is None or axes[2] is None:
        return None
    result = (axes[0], axes[1], axes[2])
    # NO continuity correction
    self._axes_cache[t] = result
    return result

ZIP_DIR = Path(r"C:\Users\pavak\Documents\AT_test\du_lab")
ZIP_FILES = sorted(ZIP_DIR.glob("*.zip"))

for zip_path in ZIP_FILES:
    short = zip_path.stem[:40]
    print(f"\n{'='*70}")
    print(f"DATASET: {short}")
    nr = read_nuclei_zip(zip_path)

    # WITH continuity correction (current code)
    nr_with = make_copy(nr)
    assigner1 = IdentityAssigner(nr_with, naming_method=NEWCANONICAL, z_pix_res=Z)
    assigner1.assign_identities()
    compare(nr, nr_with, "WITH continuity")

    # WITHOUT continuity correction
    original = DivisionCaller._get_local_axes
    DivisionCaller._get_local_axes = get_local_axes_no_continuity
    nr_without = make_copy(nr)
    assigner2 = IdentityAssigner(nr_without, naming_method=NEWCANONICAL, z_pix_res=Z)
    assigner2.assign_identities()
    compare(nr, nr_without, "NO continuity  ")
    DivisionCaller._get_local_axes = original
