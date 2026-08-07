#!/usr/bin/env python
"""PHASE 4 of the 7-run template sweep: score every run on both interface sets, seeds 101-150.

One scorer for all 7 arms so the verdicts are directly comparable:
  native50    both chains native fold
  abonly      antibody chain(s) native, antigen free
  agonly      antigen chain native, antibody free
  globallddt  both chains = model fold with best whole-chain lDDT   (ORACLE)
  ifacelddt   both chains = model fold with best paratope/epitope lDDT (ORACLE)
  globalplddt both chains = model fold with best whole-chain pLDDT   (deployable)
  ifaceplddt  both chains = model fold with best paratope/epitope pLDDT (ORACLE -- the interface
              residues come from the native complex, so this is NOT deployable; only globalplddt is)

Rescue rule, identical everywhere: the pose with the highest interface pair_iptm has DockQ >= 0.23.

Interfaces come from decoy_set_101_150.json (15 confident decoys, 7z2m under its CORRECTED A-K / B-K
labels) and safety15.json (15 goods).

SCORING PROTOCOL (tts_abag/eval/eval_ost_foldbench.py -- do not improvise): ONE OST call, full native
assembly vs full model, --fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq, OST's own
global chain mapping; then extract the labelled interface BY NAME. If the pair is absent from OST's
mapping the pose is skipped -- never substituted with another interface.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/score_sweep.py
Writes: artifacts/sweep_scores.json
"""
import json, os, subprocess, sys, glob
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import OST, ENV, EXP, SEEDS, SAMPLES, THRESH, GT

PREDROOT = f"{EXP}/predictions"
SWEEP = f"{EXP}/artifacts/sweep_out"
CACHE = f"{EXP}/artifacts/sweep_ost"; os.makedirs(CACHE, exist_ok=True)
OUTJ = f"{EXP}/artifacts/sweep_scores.json"
NWORK = 16                     # CPU policy: 64 soft target, 100 HARD cap. OST is internally
                               # multi-threaded, so the affinity mask -- not the worker count --
                               # bounds CPU. Measured: 0-89/22w = 103 (over), 0-71/16w = ~80.
                               # OST is internally multi-threaded, so the affinity mask (not the
                               # worker count) is what actually bounds CPU. taskset 0-89 + 22
                               # workers measures ~85 cores, leaving headroom under 100.
                               # multi-threaded, so worker count alone does NOT bound CPU -- the
                               # affinity mask is what actually caps it (16 workers hit 97 cores).

# run -> ordered list of (root_dir, name_suffix). First location that has the pdb wins, so the
# freshly-run seeds-101-150 jobs in sweep_out take precedence over the older prediction dirs.
LOCATIONS = {
    # native50 goods live in safety_out (a separately-launched run of the same arm: both chains
    # templated on the native fold), so all three locations are needed for full 30-interface coverage.
    "native50":    [(f"{SWEEP}/native50", "_native50"), (f"{PREDROOT}/native50_out", "_nativetmpl"),
                    (f"{PREDROOT}/safety_out", "_nativetmpl")],
    # relabel_out FIRST: 8fbw/8yxi were re-run there with CORRECTED antigen labels at seeds 101-150,
    # and must win over the stale-label runs in abonly_out / agonly_out.
    "abonly":      [(f"{EXP}/artifacts/relabel_out", "_abonly"), (f"{PREDROOT}/abonly_out", "_abonly")],
    "agonly":      [(f"{EXP}/artifacts/relabel_out", "_agonly"), (f"{SWEEP}/agonly", "_agonly"),
                    (f"{PREDROOT}/agonly_out", "_nativetmpl")],
    "globallddt":  [(f"{SWEEP}/globallddt", "_globallddt")],
    "ifacelddt":   [(f"{SWEEP}/ifacelddt", "_ifacelddt")],
    "globalplddt": [(f"{SWEEP}/globalplddt", "_globalplddt")],
    "ifaceplddt":  [(f"{SWEEP}/ifaceplddt", "_ifaceplddt")],
}

DEC = json.load(open(f"{EXP}/artifacts/decoy_set_101_150.json"))
IFACES = [{"set": "decoy", "label": l, "pdb": DEC["per_interface"][l]["pdb"],
           "c1": DEC["per_interface"][l]["chains"].split("-")[0],
           "c2": DEC["per_interface"][l]["chains"].split("-")[1]} for l in DEC["decoys"]]
IFACES += [{"set": "good", "label": e["iface"], "pdb": e["pdb"], "c1": e["c1"], "c2": e["c2"]}
           for e in json.load(open(f"{EXP}/artifacts/safety15.json"))]


def locate(run, pdb):
    """(dir, name) of the prediction folder for this run+pdb, preferring the fresh sweep output."""
    for root, suf in LOCATIONS[run]:
        d = f"{root}/{pdb}{suf}"
        if os.path.isdir(d) and glob.glob(f"{d}/seed_*"):
            return d, f"{pdb}{suf}"
    return None, None


def pose(a):
    run, pdb, name, d, c1, c2, s, k = a
    cif = f"{d}/seed_{s}/predictions/{name}_sample_{k}.cif"
    if not os.path.exists(cif):
        return None
    o = f"{CACHE}/{run}_{pdb}_{c1}{c2}_{s}_{k}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-71", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o,
                        "--fault-tolerant", "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        dd = json.load(open(o))
    except Exception:
        return None
    qs, ifs = dd.get("dockq") or [], dd.get("dockq_interfaces") or []
    q = mc = None
    for i, t in enumerate(ifs):                     # match the interface BY NAME, never dockq[0]
        if {t[0], t[1]} == {c1, c2} and i < len(qs):
            q, mc = qs[i], (t[2], t[3]); break
    if q is None:                     # labelled pair not in OST's mapping -> no score for this pose
        return None
    if q is None:
        return None
    pi = -1.0
    try:
        M = json.load(open(f"{d}/seed_{s}/predictions/"
                          f"{name}_summary_confidence_sample_{k}.json"))["chain_pair_iptm"]
        x, y = ord(mc[0]) - 65, ord(mc[1]) - 65
        if 0 <= x < len(M) and 0 <= y < len(M):
            pi = M[x][y]
    except Exception:
        pass
    return pi, q


out = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for run in LOCATIONS:
    out.setdefault(run, {})
    for it in IFACES:
        prev = out[run].get(it["label"])
        if prev and prev.get("n", 0) >= len(SEEDS) * len(SAMPLES):
            continue          # only skip a FULL 250-pose pool; a partial one gets re-scored later
        d, name = locate(run, it["pdb"])
        if d is None:
            continue
        jobs = [(run, it["pdb"], name, d, it["c1"], it["c2"], s, k) for s in SEEDS for k in SAMPLES]
        with ThreadPoolExecutor(NWORK) as ex:
            ps = [r for r in ex.map(pose, jobs) if r]
        if not ps:
            continue
        sel = max(ps, key=lambda x: x[0])
        ok = [p for p in ps if p[1] >= THRESH]
        out[run][it["label"]] = {
            "set": it["set"], "pdb": it["pdb"], "n": len(ps), "n_ok": len(ok),
            "top_pair_iptm": round(sel[0], 3), "sel_dockq": round(sel[1], 3),
            "best_dockq": round(max(p[1] for p in ps), 3), "rescued": sel[1] >= THRESH,
            "max_iptm_good": round(max([p[0] for p in ok], default=-1), 3),
            "max_iptm_bad": round(max([p[0] for p in ps if p[1] < THRESH], default=-1), 3)}
        json.dump(out, open(OUTJ, "w"), indent=1)
        v = out[run][it["label"]]
        print(f"{run:12} {it['set']:5} {it['label']:10} n={v['n']:3d} top={v['top_pair_iptm']:.3f} "
              f"sel={v['sel_dockq']:.3f} {'RESCUED' if v['rescued'] else '-'}", flush=True)

print(f"\n{'run':12} {'decoys rescued':>15} {'goods preserved':>16}")
for run in LOCATIONS:
    r = out.get(run, {})
    dec = [v for v in r.values() if v["set"] == "decoy"]
    good = [v for v in r.values() if v["set"] == "good"]
    print(f"{run:12} {sum(v['rescued'] for v in dec):8d} / {len(dec):<4d} "
          f"{sum(v['rescued'] for v in good):9d} / {len(good):<4d}")
print(f"WROTE {OUTJ}")
