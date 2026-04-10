"""Test: multi-frame average in lab frame, project once using division-time axes.
Also test: no multi-frame averaging at all (single frame only)."""
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
    print(f"  {label}: {matched}/{total} ({pct:.1f}%), {roots} root, {mismatched-roots} cascade")
    return matched, total, roots


# Patch: lab-frame multi-frame averaging
def assign_names_labframe_avg(self, parent, daughter1, daughter2,
                               nuclei_record, division_time, n_frames=DEFAULT_AVG_FRAMES):
    parent_name = parent.effective_name
    if not parent_name:
        return "", ""
    rule = self.rule_manager.get_rule(parent_name)

    # Collect raw lab-frame division vectors
    z = self.z_pix_res
    lab_vectors = []

    d1_curr, d2_curr = daughter1, daughter2
    for dt in range(n_frames):
        t = division_time + dt
        if t >= len(nuclei_record):
            break
        if dt > 0:
            d1_next = _follow_successor(d1_curr, nuclei_record[t])
            d2_next = _follow_successor(d2_curr, nuclei_record[t])
            if d1_next is None or d2_next is None:
                break
            d1_curr, d2_curr = d1_next, d2_next

        raw = np.array([
            d2_curr.x - d1_curr.x,
            d2_curr.y - d1_curr.y,
            (d2_curr.z - d1_curr.z) * z,
        ], dtype=np.float64)
        norm = np.linalg.norm(raw)
        if norm > 1e-6:
            lab_vectors.append(raw / norm)

    if not lab_vectors:
        return self.assign_names(parent, daughter1, daughter2, timepoint=division_time)

    # Average in lab frame
    avg_lab = np.mean(lab_vectors, axis=0)
    avg_norm = np.linalg.norm(avg_lab)
    consistency = avg_norm

    if avg_norm > 1e-6:
        avg_lab = avg_lab / avg_norm

    # Scale to first-frame magnitude
    raw0 = np.array([
        daughter2.x - daughter1.x,
        daughter2.y - daughter1.y,
        (daughter2.z - daughter1.z) * z,
    ], dtype=np.float64)
    avg_lab = avg_lab * np.linalg.norm(raw0)

    # Project to canonical using DIVISION TIME axes only
    diff = self._measurement_correction(avg_lab, timepoint=division_time)

    dot = float(np.dot(diff, rule.axis_vector))
    diff_norm = np.linalg.norm(diff)
    if diff_norm > 1e-6:
        cos_angle = abs(dot) / (diff_norm * np.linalg.norm(rule.axis_vector))
        angle_deg = math.degrees(math.acos(min(1.0, cos_angle)))
    else:
        angle_deg = 90.0

    confidence = _angle_to_confidence(angle_deg) * consistency

    if dot >= 0:
        name1, name2 = rule.daughter1, rule.daughter2
    else:
        name1, name2 = rule.daughter2, rule.daughter1

    self._classifications.append(DivisionClassification(
        parent_name=parent_name, daughter1_name=name1, daughter2_name=name2,
        axis_used=rule.sulston_letter, confidence=confidence,
        angle_from_rule=angle_deg, dot_product=dot,
    ))
    return name1, name2


ZIP_DIR = Path(r"C:\Users\pavak\Documents\AT_test\du_lab")
ZIP_FILES = sorted(ZIP_DIR.glob("*.zip"))

for zip_path in ZIP_FILES:
    short = zip_path.stem[:40]
    print(f"\n{'='*70}")
    print(f"DATASET: {short}")
    nr = read_nuclei_zip(zip_path)

    # Current code (WITH continuity, canonical-frame multi-frame avg)
    nr1 = make_copy(nr)
    a1 = IdentityAssigner(nr1, naming_method=NEWCANONICAL, z_pix_res=Z)
    a1.assign_identities()
    compare(nr, nr1, "current (cont+canon_avg)")

    # Lab-frame averaging WITH continuity
    original = DivisionCaller.assign_names_multi_frame
    DivisionCaller.assign_names_multi_frame = assign_names_labframe_avg
    nr2 = make_copy(nr)
    a2 = IdentityAssigner(nr2, naming_method=NEWCANONICAL, z_pix_res=Z)
    a2.assign_identities()
    compare(nr, nr2, "labframe_avg + cont    ")
    DivisionCaller.assign_names_multi_frame = original

    # No multi-frame (single frame only) WITH continuity
    nr3 = make_copy(nr)
    a3 = IdentityAssigner(nr3, naming_method=NEWCANONICAL, z_pix_res=Z, use_multi_frame=False)
    a3.assign_identities()
    compare(nr, nr3, "single_frame + cont    ")
