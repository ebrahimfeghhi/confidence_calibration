#!/usr/bin/env python
"""PHASE 1 of the 7-run template sweep: per-pose, per-chain fold quality for BOTH interface sets.

For every pdb (12 decoy + 15 good), every templatable chain, and every free-run pose (seeds 101-150 x
5 samples), compute four numbers:
    global_lddt   - whole-chain lDDT vs the native chain          (oracle criterion, needs native)
    iface_lddt    - lDDT restricted to paratope/epitope residues  (oracle criterion, needs native)
    global_plddt  - whole-chain mean pLDDT                        (deployable)
    iface_plddt   - mean pLDDT over paratope/epitope residues     (ORACLE: region from native)

All four selection criteria for the sweep derive from this one table, so the poses are scored once
rather than four times.

Conventions carried over from the rest of the project:
  - native chains are addressed by label_asym_id (see decoy_set.native_chains); gemmi's chain.name is
    auth_asym_id and differs for 8bf4/8che/8fbw/8slb/8the.
  - SINGLE-CHAIN vs SINGLE-CHAIN lDDT (model chain matched to native by sequence), so the number does
    not depend on where the pose docked.
  - interface residues = native residues within 5 A of the partner chain(s) of that chain's labelled
    interface(s); a chain in two interfaces (e.g. 8slb A) takes the union.
  - pLDDT from the model CIF B-factor column (0-100, rescaled to 0-1).

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/fold_selection_table.py [--pdbs 8slb,8aok]
Writes: artifacts/fold_selection_table.json
        {pdb: {qidx: {"native_chain":c, "n_iface":n, "poses": [[seed, sample, gl, il, gp, ip], ...]}}}
"""
import argparse, json, os, subprocess, sys, statistics, difflib, collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from decoy_set import native_chains, GT, OST, ENV, PRED, EXP, SEEDS, SAMPLES
from fold_quality import TLC, nat_res, nat_pdb, model_chain

ap = argparse.ArgumentParser(); ap.add_argument("--pdbs", default=None)
A = ap.parse_args()
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
WORK = f"{EXP}/artifacts/fold_selection"; os.makedirs(WORK, exist_ok=True)
OUTJ = f"{EXP}/artifacts/fold_selection_table.json"
NWORK = 8            # measured ~26 cores with OST's internal threading; stays under the 64 budget


def interfaces_by_pdb():
    """{pdb: [(c1, c2), ...]} for the confident decoys and the 15 goods.

    Sources BOTH decoy sets: the original 15 (decoy_set_101_150.json, which §2-§4 ran on) and §5's
    canonical 26 (native_arm_scores.json). The two are not nested -- the 15 holds 7z2m's corrected
    A-K/B-K rows that the 26 cannot contain, and the 26 holds 11 interfaces the 15 never had. Taking
    the union means extending an arm to the 26 does not silently drop the older results.
    """
    out = collections.defaultdict(set)
    d = json.load(open(f"{EXP}/artifacts/decoy_set_101_150.json"))
    for label in d["decoys"]:
        v = d["per_interface"][label]
        out[v["pdb"]].add(tuple(v["chains"].split("-")))
    for e in json.load(open(f"{EXP}/artifacts/safety15.json")):
        out[e["pdb"]].add((e["c1"], e["c2"]))
    EXCLUDE = {"7z2m_AJ", "8udz_AC"}                       # mislabelled crystal contacts, report §5
    for lab, v in json.load(open(f"{EXP}/artifacts/native_arm_scores.json")).items():
        if v["free_status"] == "confident_decoy" and lab not in EXCLUDE:
            parts = v["chains"].split("-")
            c1, c2 = (parts[0], "-".join(parts[1:])) if len(parts) > 2 else (parts[0], parts[1])
            out[v["pdb"]].add((c1, c2))
    return {k: sorted(v) for k, v in out.items()}


def iface_residues(pdb, ch, partners):
    """0-based indices into nat_res(pdb,ch) contacting ANY partner chain (union over interfaces)."""
    D = native_chains(pdb)
    XA, IA = D[ch]
    order = {v: i for i, v in enumerate(sorted(set(IA), key=lambda x: int(x)))}
    hit = set()
    for p in partners:
        if p not in D: continue
        m = np.linalg.norm(XA[:, None, :] - D[p][0][None, :, :], axis=-1) < 5.0
        hit |= {order[IA[i]] for i in np.where(m.any(1))[0] if IA[i] in order}
    return hit


def measure(pdb, ch, seed, sample, iface):
    """(global_lddt, iface_lddt, global_plddt, iface_plddt) for one chain in one pose."""
    ref = nat_pdb(pdb, ch)
    seq = "".join(TLC[r.name] for r in nat_res(pdb, ch))
    try:
        mdl, plddt = model_chain(pdb, seed, sample, seq)
    except Exception:
        return None
    o = f"{WORK}/ld_{pdb}_{ch}_{seed}_{sample}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-31", OST, "compare-structures", "-m", mdl, "-r", ref,
                        "-o", o, "--fault-tolerant", "--min-pep-length", "4", "--lddt", "--local-lddt"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
    except Exception:
        return None
    g, ll = d.get("lddt"), d.get("local_lddt") or {}
    aln = (d.get("aln") or [None])[0]
    per, mp = {}, {}
    if aln:
        parts = aln.split("\n"); ri = mi = 0
        for a, b in zip(parts[1], parts[3]):
            if a != "-" and b != "-":
                v = ll.get(f"A.{mi + 1}.")
                if v is not None: per[ri] = v
                mp[ri] = mi
            if a != "-": ri += 1
            if b != "-": mi += 1
    lv = [per[i] for i in iface if i in per]
    pv = [plddt[mp[i]] for i in iface if i in mp and mp[i] < len(plddt)]
    return (round(g, 4) if g is not None else None,
            round(statistics.mean(lv), 4) if lv else None,
            round(statistics.mean(plddt), 4) if plddt else None,
            round(statistics.mean(pv), 4) if pv else None)


IF = interfaces_by_pdb()
pdbs = A.pdbs.split(",") if A.pdbs else sorted(IF)
out = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}

for pdb in pdbs:
    if pdb in out and all(len(v["poses"]) >= 200 for v in out[pdb].values()):
        print(f"{pdb}: already done ({len(out[pdb])} chains)", flush=True); continue
    pairs = IF[pdb]
    partners = collections.defaultdict(set)
    for c1, c2 in pairs:
        partners[c1].add(c2); partners[c2].add(c1)
    base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
    obj = base[0] if isinstance(base, list) else base
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    rec = {}
    for ch in sorted(partners):
        try:
            nseq = "".join(TLC[r.name] for r in nat_res(pdb, ch))
        except KeyError as e:
            print(f"{pdb} {ch}: SKIP {e}", flush=True); continue
        qi = max(range(len(qseqs)), key=lambda i: difflib.SequenceMatcher(
            None, qseqs[i], nseq, autojunk=False).ratio())
        iface = iface_residues(pdb, ch, partners[ch])
        jobs = [(s, k) for s in SEEDS for k in SAMPLES]
        with ThreadPoolExecutor(NWORK) as ex:
            res = list(ex.map(lambda a: (a[0], a[1], measure(pdb, ch, a[0], a[1], iface)), jobs))
        poses = [[s, k, *m] for s, k, m in res if m]
        rec[str(qi)] = {"native_chain": ch, "n_iface": len(iface), "poses": poses}
        print(f"{pdb} q{qi} chain {ch}: {len(poses)} poses, {len(iface)} iface residues", flush=True)
    out[pdb] = rec
    json.dump(out, open(OUTJ, "w"))
    print(f"--- {pdb} done ({len(out)}/{len(pdbs)}) ---", flush=True)

print(f"WROTE {OUTJ}")
