"""Quick comparison: current code (with all fixes) vs reference on 5 datasets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.naming.identity import IdentityAssigner, NEWCANONICAL

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

def compare(nr_ref, nr_new):
    ref_names = {}
    for t in range(min(len(nr_ref), MAX_T)):
        for j, nuc in enumerate(nr_ref[t]):
            if nuc.status >= 1 and is_sulston(nuc.identity):
                ref_names[(t, j)] = nuc.identity
    matched = 0
    mismatched = 0
    mismatch_parents = set()
    roots = 0
    swap_roots = 0
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
                if (t-1, nuc.predecessor - 1) in mismatch_parents:
                    is_cascade = True
            mismatch_parents.add((t, j))
            mismatched += 1
            if not is_cascade:
                roots += 1
                # Check sister swap
                if t > 0 and nuc.predecessor >= 1:
                    pred_idx = nuc.predecessor - 1
                    if 0 <= pred_idx < len(nr_ref[t-1]):
                        pred = nr_ref[t-1][pred_idx]
                        if pred.successor2 != NILLI and pred.successor2 > 0:
                            s1 = pred.successor1 - 1
                            s2 = pred.successor2 - 1
                            sis = s2 if j == s1 else s1
                            if 0 <= sis < len(nr_ref[t]):
                                sis_ref = ref_names.get((t, sis), "")
                                sis_new = nr_new[t][sis].identity if sis < len(nr_new[t]) else ""
                                if new_name == sis_ref and sis_new == ref_names[(t,j)]:
                                    swap_roots += 1
    total = matched + mismatched
    return matched, total, roots, swap_roots, mismatched - roots

ZIP_DIR = Path(r"C:\Users\pavak\Documents\AT_test\du_lab")
ZIP_FILES = sorted(ZIP_DIR.glob("*.zip"))

print(f"{'Dataset':45s} {'Match%':>7s} {'Roots':>6s} {'Swaps':>6s} {'Cascade':>8s}")
print("-" * 80)

for zip_path in ZIP_FILES:
    short = zip_path.stem[:43]
    nr = read_nuclei_zip(zip_path)
    nr2 = make_copy(nr)
    a = IdentityAssigner(nr2, naming_method=NEWCANONICAL, z_pix_res=Z)
    a.assign_identities()
    matched, total, roots, swaps, cascades = compare(nr, nr2)
    pct = matched / total * 100 if total else 0
    print(f"  {short:43s} {pct:6.1f}% {roots:6d} {swaps:6d} {cascades:8d}")
