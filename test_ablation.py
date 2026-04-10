"""Ablation test: isolate which change caused SYS3556/SYS516_EV regression."""
import sys
from pathlib import Path
import numpy as np
import math

sys.path.insert(0, str(Path(__file__).parent))

from acetree_py.core.nucleus import NILLI, Nucleus
from acetree_py.io.nuclei_reader import read_nuclei_zip
from acetree_py.naming.lineage_axes import compute_local_axes
from acetree_py.naming.identity import IdentityAssigner, NEWCANONICAL
from acetree_py.naming.division_caller import DivisionCaller, _follow_successor, _angle_to_confidence, DivisionClassification, DEFAULT_AVG_FRAMES

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
                if (t-1, nuc.predecessor - 1) in mismatch_parents:
                    is_cascade = True
            mismatch_parents.add((t, j))
            mismatched += 1
            if not is_cascade:
                roots += 1
    total = matched + mismatched
    pct = matched / total * 100 if total else 0
    print(f"  {label}: {pct:6.1f}% ({roots} roots)")

# Only test the two regressed datasets
ZIP_DIR = Path(r"C:\Users\pavak\Documents\AT_test\du_lab")
for name in ["YML_SYS3556_10mMH2O2_20251020_3_s1_emb1_edited.zip",
             "YML_SYS516_EV_20251114_1_s1_emb2_edited.zip",
             "YML_JIM113_20240108_3_s1_emb1_edited.zip"]:
    zip_path = ZIP_DIR / name
    if not zip_path.exists():
        continue
    short = zip_path.stem[:40]
    print(f"\n{'='*70}")
    print(f"DATASET: {short}")
    nr = read_nuclei_zip(zip_path)

    # Test 1: current code (with all fixes)
    nr1 = make_copy(nr)
    a1 = IdentityAssigner(nr1, naming_method=NEWCANONICAL, z_pix_res=Z)
    a1.assign_identities()
    compare(nr, nr1, "all fixes (lab_avg + quality + smooth)")

    # Test 2: no multi-frame averaging at all
    nr2 = make_copy(nr)
    a2 = IdentityAssigner(nr2, naming_method=NEWCANONICAL, z_pix_res=Z, use_multi_frame=False)
    a2.assign_identities()
    compare(nr, nr2, "single frame (quality + smooth only)   ")
