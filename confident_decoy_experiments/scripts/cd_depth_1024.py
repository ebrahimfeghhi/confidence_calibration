#!/usr/bin/env python
"""Does deep sampling rescue Protenix confident decoys?

The equivalent ESMFold2 result (51/55 traps survive 1024 seeds) does NOT transfer -- it was a
different model. For Protenix, nine confident-decoy complexes already have >=1024 free-arm seeds on
disk, so this needs no GPU; the pose index (scripts/pose_index.py) is what makes them findable.

At each depth the pool is the FIRST n seeds of the free/untemplated arms, and:
   selected = DockQ of the max interface-ipTM pose   (what you would actually deploy)
   oracle   = best DockQ present                     (the coverage ceiling)
A trap resolves if selected climbs from <0.23 to >=0.23.

Templated arms are excluded: they see the answer, so including them would not measure sampling.
"""
import collections, csv, sys
import numpy as np
sys.path.insert(0, "scripts")
from pose_index import poses
from iface_chains import resolve
import json, os, re


def _iptm_from_path(cif, m_ab, m_ag):
    """Interface ipTM read from the confidence json sitting NEXT TO this pose.

    selection_protocol.pair_iptm hardcodes predictions_template, so it silently returns None for
    poses written by the deep arms -- which made 2500-seed complexes look like 151-seed ones."""
    d, base = os.path.dirname(cif), os.path.basename(cif)
    m = re.match(r"(.+)_sample_(\d+)\.cif$", base)
    if not m:
        return None
    f = f"{d}/{m.group(1)}_summary_confidence_sample_{m.group(2)}.json"
    try:
        M = json.load(open(f)).get("chain_pair_iptm")
    except Exception:
        return None
    ci = lambda c: ord(c.split("-")[0][0].upper()) - 65
    i, j = ci(m_ab), ci(m_ag)
    if M and 0 <= i < len(M) and 0 <= j < len(M[i]):
        return M[i][j]
    return None

DEPTHS = [50, 100, 250, 512, 1024, 2048]
FREE = ("predictions_template", "freedeep", "allpose", "free1k")


def pool(pdb, ab, ag):
    """-> sorted [(seed, dockq, pair_iptm)] over the untemplated arms, deduped by seed."""
    seen, out = set(), []
    for s, k, arm, f in poses(pdb):
        if not any(t in arm for t in FREE) or "nomsa" in arm or "fullmsa" in arm:
            continue
        if (s, k) in seen:
            continue
        seen.add((s, k))
        try:
            c = resolve(pdb, str(s), str(k), ab, ag)
        except Exception:
            continue
        if c["dockq"] is None:
            continue
        v = _iptm_from_path(f, c["mdl_ab"], c["mdl_ag"])
        if v is None:
            continue
        out.append((s, c["dockq"], v))
    return sorted(out)


def main():
    P = list(csv.DictReader(open("artifacts/abag_selection_protocol.csv")))
    CD = [r["interface"] for r in P
          if float(r["sel50_dockq"]) < 0.23 <= float(r["oracle_dockq"])]
    rows = []
    for iface in CD:
        pdb, ch = iface.rsplit("_", 1)
        p = pool(pdb, ch[0], ch[1])
        if len(p) < 250:
            continue
        seeds = sorted({s for s, _, _ in p})
        rec = {"interface": iface, "n_seeds": len(seeds)}
        for d in DEPTHS:
            keep = set(seeds[:d])
            sub = [(dq, ip) for s, dq, ip in p if s in keep]
            if len(sub) < 50:
                continue
            a = np.array(sub)
            rec[f"sel{d}"] = float(a[int(np.argmax(a[:, 1])), 0])
            rec[f"orc{d}"] = float(a[:, 0].max())
        rows.append(rec)
        print(f"{iface:<12} {len(seeds):>5} seeds  " +
              "  ".join(f"d{d}:{rec.get(f'sel{d}', float('nan')):.3f}" for d in DEPTHS
                        if f"sel{d}" in rec), flush=True)
    with open("artifacts/cd_depth_1024.csv", "w", newline="") as fh:
        f = sorted({k for r in rows for k in r}, key=lambda x: (x != "interface", x))
        w = csv.DictWriter(fh, fieldnames=f); w.writeheader(); w.writerows(rows)
    print(f"\nwrote artifacts/cd_depth_1024.csv  ({len(rows)} interfaces)")
    return rows


if __name__ == "__main__":
    main()
