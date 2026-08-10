#!/usr/bin/env python
"""Score the no-MSA donor runs against dockqdonor WITH MSA, at the identical seeds.

Both sides use the SAME donor, the SAME injected template and the SAME 10 seeds (each run's own seed
plus 9 drawn from 101-150, all inside dockqdonor's range). The only difference is that the no-MSA side
replaced both MSA paths with a 1-sequence a3m. So a difference here is the MSA, not the template, the
donor, or the seeds.

The with-MSA side is NOT re-run: it is dockqdonor's existing 50-seed output restricted to the same 10
seeds, re-read per pose from its OST cache rather than from any aggregate file.

Headline statistic is SEED CONSISTENCY -- how many of the 10 seeds produce an acceptable / good pose --
because that is what moved on 8che (6/10 -> 9/10 at >0.85) while the best pose barely changed.

SCORING PROTOCOL is the project standard (tts_abag/eval/eval_ost_foldbench.py): one OST call per pose,
full native assembly vs full model, --fault-tolerant --min-pep-length 4 --min-nuc-length 4 --dockq,
interface extracted BY NAME (never dockq[0]).

Incremental: only runs whose 10 seeds are complete are scored; the rest are listed as pending, so this
can be re-run as each interface lands.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/score_nomsa_donors.py [--workers 16]
Writes: artifacts/nomsa_donors_scores.json
"""
import argparse, glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP, OST, ENV, GT, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=16)
A = ap.parse_args()

ART = f"{EXP}/artifacts"
CACHE = f"{ART}/nomsa_donors_ost"; os.makedirs(CACHE, exist_ok=True)
DQ_CACHE = f"{ART}/dockqdonor_ost"
DQ_OUT = f"{ART}/sweep_out/dockqdonor/dockqdonor"
built = json.load(open(f"{ART}/nomsa_donors_jobs.json"))
dq_jobs = {j["interface"]: j for j in json.load(open(f"{ART}/dockq_arm_jobs.json"))}


def chains(iface):
    p = iface.split("_", 1)[1]
    return (p[0], p[1:]) if len(p) > 2 else (p[0], p[1])


def extract(ost_json, conf_json, c1, c2):
    """(pair_iptm, dockq) for one pose on one interface, matched BY CHAIN NAME."""
    try:
        d = json.load(open(ost_json)); M = json.load(open(conf_json))["chain_pair_iptm"]
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == {c1, c2} and i < len(qs):
            x, y = ord(t[2]) - 65, ord(t[3]) - 65
            if 0 <= x < len(M) and 0 <= y < len(M):
                return (M[x][y], qs[i])
            return None
    return None


def ost(cif, out):
    if not (os.path.exists(out) and os.path.getsize(out)):
        pdb = os.path.basename(cif).split("_")[0]
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{pdb}-assembly1.cif", "-o", out, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    return out


def stats(per_seed):
    """per_seed: {seed: [(iptm, dockq), ...]}"""
    if not per_seed:
        return None
    allp = [p for v in per_seed.values() for p in v]
    best_of_seed = {s: max(d for _, d in v) for s, v in per_seed.items()}
    sel = max(allp, key=lambda x: x[0])
    return {"n_seeds": len(per_seed), "n_poses": len(allp),
            "best": round(max(d for _, d in allp), 3),
            "sel_dockq": round(sel[1], 3), "sel_iptm": round(sel[0], 3),
            "seeds_acceptable": sum(1 for b in best_of_seed.values() if b >= THRESH),
            "seeds_above_06": sum(1 for b in best_of_seed.values() if b > 0.6),
            "seeds_above_085": sum(1 for b in best_of_seed.values() if b > 0.85),
            "median_seed_best": round(sorted(best_of_seed.values())[len(best_of_seed) // 2], 3)}


out, pending = {}, []
for b in built:
    seeds = b["seeds"]
    got = sorted(glob.glob(f"{ART}/sweep_out/nomsa_donors/{b['name']}/*/seed_*/predictions/"
                           f"*_summary_confidence_sample_4.json"))
    if len(got) < len(seeds):
        pending.append((b["name"], f"{len(got)}/{len(seeds)} seeds")); continue

    # --- no-MSA side: OST the new poses
    tasks = []
    for s in seeds:
        for k in range(5):
            g = glob.glob(f"{ART}/sweep_out/nomsa_donors/{b['name']}/*/seed_{s}/predictions/"
                          f"*_sample_{k}.cif")
            if g:
                tasks.append((g[0], f"{CACHE}/{b['name']}_{s}_{k}.json", s, k))
    with ThreadPoolExecutor(A.workers) as ex:
        list(ex.map(lambda t: ost(t[0], t[1]), tasks))

    for iface in b["ifaces"]:
        c1, c2 = chains(iface)
        no = {}
        for cif, oj, s, k in tasks:
            conf = cif.replace(f"_sample_{k}.cif", f"_summary_confidence_sample_{k}.json")
            r = extract(oj, conf, c1, c2)
            if r:
                no.setdefault(s, []).append(r)

        # --- with-MSA side: dockqdonor's existing poses at the SAME seeds
        dj = dq_jobs[iface]; nm = dj["name"]
        wi = {}
        for s in seeds:
            for k in range(5):
                oj = f"{DQ_CACHE}/{iface}_{s}_{k}.json"
                cf = (f"{DQ_OUT}/{nm}/seed_{s}/predictions/"
                      f"{nm}_summary_confidence_sample_{k}.json")
                if os.path.exists(oj) and os.path.exists(cf):
                    r = extract(oj, cf, c1, c2)
                    if r:
                        wi.setdefault(s, []).append(r)
        out[iface] = {"run": b["name"], "donor": f"{b['donor_seed']}_{b['donor_sample']}",
                      "donor_dockq": b.get("donor_dockq"), "seeds": seeds,
                      "no_msa": stats(no), "with_msa": stats(wi)}

hdr = (f"{'interface':>10} {'donor':>8} {'donorDQ':>8} | "
       f"{'MSA acc':>8} {'noMSA acc':>10} | {'MSA >0.6':>9} {'noMSA >0.6':>11} | "
       f"{'MSA best':>9} {'noMSA best':>11} | {'MSA sel':>8} {'noMSA sel':>10}")
print(hdr); print("-" * len(hdr))
tot = {"w_acc": 0, "n_acc": 0, "w_06": 0, "n_06": 0, "w_seeds": 0, "n_seeds": 0}
for iface, v in sorted(out.items()):
    w, n = v["with_msa"], v["no_msa"]
    if not (w and n):
        print(f"{iface:>10} incomplete baseline/new"); continue
    tot["w_acc"] += w["seeds_acceptable"]; tot["n_acc"] += n["seeds_acceptable"]
    tot["w_06"] += w["seeds_above_06"];    tot["n_06"] += n["seeds_above_06"]
    tot["w_seeds"] += w["n_seeds"];        tot["n_seeds"] += n["n_seeds"]
    print(f"{iface:>10} {v['donor']:>8} {v['donor_dockq'] or 0:>8.3f} | "
          f"{str(w['seeds_acceptable']) + '/' + str(w['n_seeds']):>8} "
          f"{str(n['seeds_acceptable']) + '/' + str(n['n_seeds']):>10} | "
          f"{str(w['seeds_above_06']) + '/' + str(w['n_seeds']):>9} "
          f"{str(n['seeds_above_06']) + '/' + str(n['n_seeds']):>11} | "
          f"{w['best']:>9.3f} {n['best']:>11.3f} | {w['sel_dockq']:>8.3f} {n['sel_dockq']:>10.3f}")

if out:
    print(f"\nTOTAL seed-trials: acceptable  WITH MSA {tot['w_acc']}/{tot['w_seeds']}  "
          f"NO MSA {tot['n_acc']}/{tot['n_seeds']}")
    print(f"                   >0.6         WITH MSA {tot['w_06']}/{tot['w_seeds']}  "
          f"NO MSA {tot['n_06']}/{tot['n_seeds']}")
for name, why in pending:
    print(f"PENDING {name}: {why}")

json.dump({"scored": out, "pending": pending, "totals": tot},
          open(f"{ART}/nomsa_donors_scores.json", "w"), indent=1)
print(f"\nWROTE {ART}/nomsa_donors_scores.json")
